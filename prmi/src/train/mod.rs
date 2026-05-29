// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer: builds a prmi sidecar from a reference FASTA.

pub mod config;
pub mod keys;
pub mod mask;
pub mod prior;
pub mod prmi;
pub mod trainer;
pub mod training_set;
pub mod verify;

use crate::encoding::{tokenize_32mer, BASE_T, KMER_LEN};
use crate::error::{Error, Result};
use crate::fasta::fasta_to_2bit_with_sha256;
use crate::sa::build_suffix_array;
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use crate::sidecar::meta::{Meta, Priors, Prmi, Ref, RmiSpec, Sa};
use crate::sidecar::model_file::{ModelFileWriter, ModelLayer};
use crate::sidecar::sa_file::{SaFileWriter, BPE_MODE2, BPE_MODE3};
use crate::sidecar::skc_file::SkcFileWriter;
use crate::sidecar::SidecarPaths;
use crate::train::config::{MemoryMode, TrainerConfig};
use crate::train::mask::MaskConfig;
use crate::train::prior::Prior;
use crate::train::trainer::default_l2_leaf_count;
use crate::train::training_set::masked_training_set;
use crate::train::verify::compute_max_error_bound;
use std::path::Path;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Build a complete sidecar (`.meta`, `.sa`, `.l1`, `.l2`) from a reference FASTA.
///
/// `prefix` is the output prefix; e.g., for `/data/ref.fa.prmi` the four
/// files become `/data/ref.fa.prmi.{meta,sa,l1,l2}`.
///
/// If `l2_leaf_count` is `None`, it is auto-scaled from the SA size (targeting
/// ~12 entries per leaf, rounded down to a power of two, clamped to [16, 2^28]).
/// If `Some(n)`, the provided `n` must be a power of two ≥ 2.
///
/// The `mask` parameter controls which (key, SA-index) training pairs are
/// excluded from model fitting and error-bound verification. The SA on disk
/// always contains all genome-region entries regardless of masking; only the
/// training set is filtered. See [`MaskConfig`] for details.
///
/// `threads` is passed to [`crate::sa::build_suffix_array`]: `0` = auto
/// (OpenMP default, typically the CPU count), `1` = single-threaded, `N > 1` =
/// exactly N threads.
///
/// `trainer_config` is optional. When `None`, the default [`TrainerConfig`] is
/// used (including `Prior::Uniform`). Pass a custom config with
/// `TrainerConfig { prior: Prior::Bed { .. }, .. }` for target-aware training.
///
/// ## Sentinel padding
///
/// To ensure every SA entry covers a full `KMER_LEN`-mer, the reference is
/// extended with `KMER_LEN - 1` T-bases before suffix-array construction.
/// SA entries pointing into the padding region are then excluded from the
/// training set and the `.sa` file, matching BWA-MEME's indexer behaviour.
/// The SHA-256 hash and `size_bytes` in `.meta` still reflect the original
/// (unpadded) FASTA, not the extended sequence.
pub fn build_sidecar(
    ref_fa: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> Result<()> {
    build_sidecar_with_config(ref_fa, prefix, l2_leaf_count, mask, threads, None)
}

/// Like [`build_sidecar`] but accepts an optional [`TrainerConfig`] for
/// non-default training parameters (e.g. a BED prior for target-aware fitting).
pub fn build_sidecar_with_config(
    ref_fa: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    let config = trainer_config.unwrap_or_default();

    let (mut bases, mut n_positions, _fa_stats, sha256_hex, fa_size_bytes) =
        fasta_to_2bit_with_sha256(ref_fa)?;

    // Record the original genome length before padding.
    let genome_len = bases.len() as u64;

    // Append KMER_LEN - 1 T-sentinel bases so that every position in the
    // original genome has a full 32-mer available without wrapping.
    bases.extend(std::iter::repeat_n(BASE_T, KMER_LEN - 1));
    // Extend n_positions to match; sentinel bases are not N.
    n_positions.extend(std::iter::repeat_n(false, KMER_LEN - 1));

    // Build the SA over the padded sequence.
    let full_sa = build_suffix_array(&bases, threads)?;

    // Filter to only entries pointing into the original genome (positions
    // 0..genome_len). The filtered slice is a subsequence of the sorted SA,
    // so it is still non-decreasing in key space.
    let sa: Vec<u64> = full_sa
        .into_iter()
        .filter(|&pos| pos < genome_len)
        .collect();

    // Resolve the optional l2_leaf_count: use provided value or auto-scale.
    let l2_leaf_count = l2_leaf_count.unwrap_or_else(|| default_l2_leaf_count(sa.len()));

    let ts = masked_training_set(&sa, &bases, &n_positions, &mask, &config.prior);
    let model = crate::train::trainer::train_with_config(&ts, l2_leaf_count, &config)?;
    let max_err = compute_max_error_bound(&model, &ts);

    let paths = SidecarPaths::from_prefix(prefix);

    // .sa — write all genome-region entries in the requested memory mode.
    let memory_mode = config.memory_mode;
    match memory_mode {
        MemoryMode::Mode1 => {
            // Position-only, 5 B/entry. Original layout; unchanged.
            let mut w = SaFileWriter::create(&paths.sa, sa.len() as u64)?;
            for &pos in &sa {
                w.write_position(pos)?;
            }
            w.finish()?;
        }
        MemoryMode::Mode2 => {
            // Position + 32-mer key, 13 B/entry.
            let mut w = SaFileWriter::create_with_mode(&paths.sa, sa.len() as u64, BPE_MODE2)?;
            for &pos in &sa {
                let key = key_for_position(pos, &bases);
                w.write_entry_with_key(pos, key)?;
            }
            w.finish()?;
        }
        MemoryMode::Mode3 => {
            // Position + key + ISA, 21 B/entry.
            // Build the ISA in one pass: isa[sa[i]] = i.
            let isa = build_isa(&sa)?;
            let mut w = SaFileWriter::create_with_mode(&paths.sa, sa.len() as u64, BPE_MODE3)?;
            for &pos in &sa {
                let key = key_for_position(pos, &bases);
                let isa_val = isa[pos as usize];
                w.write_entry_with_key_isa(pos, key, isa_val)?;
            }
            w.finish()?;
        }
        MemoryMode::SuffixKeyCache { cache_size } => {
            // .sa is mode-1 (position only, 5 B/entry).
            // Keys for the top-N SA positions go into a companion `.skc` file.
            let mut w = SaFileWriter::create(&paths.sa, sa.len() as u64)?;
            for &pos in &sa {
                w.write_position(pos)?;
            }
            w.finish()?;

            // Build the suffix-key cache: cache the first `cache_size` SA
            // entries (indices 0..cache_size). This is a placeholder selection
            // policy until workload-aware cache content (e.g. driven by a
            // `--prior-fastq-histogram` analysis) is implemented in a later
            // release. In genomic 32-mer space the lex-smallest suffixes are
            // typically dominated by all-A windows from N-runs and homopolymers
            // — often masked or excluded — so this default produces a low cache
            // hit rate in practice. Cache misses fall back to on-the-fly pac
            // tokenization, so correctness is unaffected; only the speed-up
            // from `suffix_key_cache` mode is muted until the selection policy
            // is improved.
            let n = (cache_size as usize).min(sa.len());
            let mut skc_w = SkcFileWriter::create(&paths.skc, n as u64)?;
            for (sa_idx, &pos) in sa.iter().enumerate().take(n) {
                let key = key_for_position(pos, &bases);
                skc_w.write_entry(sa_idx as u64, key)?;
            }
            skc_w.finish()?;
        }
    }

    // .l1 / .l2
    ModelFileWriter::write(&paths.l1, ModelLayer::L1, &model.l1)?;
    ModelFileWriter::write(&paths.l2, ModelLayer::L2, &model.l2)?;

    // .meta
    let created_utc = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|e| Error::Internal {
            detail: format!("timestamp: {e}"),
        })?;
    let spec = format!("pwl{},linear,linear_spline", l2_leaf_count.trailing_zeros());
    let masked_bed_path: Option<String> = mask
        .mask_bed_path
        .as_deref()
        .map(|p| p.display().to_string());

    // Determine the [priors] TOML section from the trainer config.
    let priors = priors_for_meta(&config.prior);

    // Mode-dependent meta fields.
    let (mode_str, skc_cache_size_meta) = match memory_mode {
        MemoryMode::Mode1 => ("1".to_string(), None),
        MemoryMode::Mode2 => ("2".to_string(), None),
        MemoryMode::Mode3 => ("3".to_string(), None),
        MemoryMode::SuffixKeyCache { cache_size } => (
            "suffix_key_cache".to_string(),
            Some(cache_size.min(sa.len() as u64)),
        ),
    };

    let meta = Meta {
        prmi: Prmi {
            magic: META_MAGIC.into(),
            format_version: FORMAT_VERSION,
            trainer_version: format!("prmi={}", env!("CARGO_PKG_VERSION")),
            created_utc,
        },
        ref_: Ref {
            path: ref_fa.display().to_string(),
            sha256: sha256_hex,
            size_bytes: fa_size_bytes,
        },
        sa: Sa {
            num_entries: sa.len() as u64,
            bytes_per_entry: memory_mode.bytes_per_entry(),
            encoding: memory_mode.encoding_name().to_string(),
            mode: mode_str,
            skc_cache_size: skc_cache_size_meta,
            strand: "forward_only".into(),
            masked_n_runs: mask.mask_n_runs,
            masked_homopolymers: mask.mask_homopolymers,
            masked_bed: masked_bed_path,
        },
        rmi: RmiSpec {
            spec,
            l2_leaf_count,
            bit_shift: model.bit_shift,
            max_error_bound: max_err,
        },
        priors,
    };
    meta.write_file(&paths.meta)?;
    Ok(())
}

/// Construct the [`Priors`] meta struct from the trainer's [`Prior`].
fn priors_for_meta(prior: &Prior) -> Priors {
    match prior {
        Prior::Uniform => Priors {
            kind: "uniform".into(),
            bed: None,
            weight: None,
            histogram: None,
            base_weight: None,
            formula: None,
        },
        Prior::Bed { weight, path, .. } => Priors {
            kind: "bed".into(),
            bed: path.as_deref().map(|p| p.display().to_string()),
            weight: Some(*weight),
            histogram: None,
            base_weight: None,
            formula: None,
        },
        Prior::FastqHistogram {
            base_weight, path, ..
        } => Priors {
            kind: "fastq_histogram".into(),
            bed: None,
            weight: None,
            histogram: path.as_deref().map(|p| p.display().to_string()),
            base_weight: Some(*base_weight),
            formula: Some("1.0 + log2(1 + freq)".into()),
        },
    }
}

/// Resolve a `MaskConfig` from raw CLI values, parsing the BED file if
/// `bed_path` is `Some`. The source path is stored in `MaskConfig::mask_bed_path`
/// for provenance in the meta TOML.
///
/// `#[allow(dead_code)]` because the only caller is `cli.rs`, which lands in
/// PR #6 (`feat/v0.1-cli`).
#[allow(dead_code)]
pub(crate) fn mask_config_from_cli(
    no_mask_n_runs: bool,
    mask_homopolymers: Option<u32>,
    bed_path: Option<&Path>,
) -> Result<MaskConfig> {
    use crate::train::mask::parse_bed;
    let bed = bed_path.map(parse_bed).transpose()?;
    Ok(MaskConfig {
        mask_n_runs: !no_mask_n_runs,
        mask_homopolymers,
        mask_bed: bed,
        mask_bed_path: bed_path.map(std::path::PathBuf::from),
    })
}

/// Resolve a [`Prior`] from raw CLI values.
///
/// Returns `Err` if `prior_bed_weight` is not positive.
pub fn prior_from_cli(prior_bed: Option<&Path>, prior_bed_weight: f64) -> Result<Prior> {
    use crate::train::mask::parse_bed;
    match prior_bed {
        None => Ok(Prior::Uniform),
        Some(path) => {
            if prior_bed_weight <= 0.0 {
                return Err(Error::Internal {
                    detail: format!("--prior-bed-weight must be > 0, got {prior_bed_weight}"),
                });
            }
            let intervals = parse_bed(path)?;
            Ok(Prior::Bed {
                intervals,
                weight: prior_bed_weight,
                path: Some(path.to_path_buf()),
            })
        }
    }
}

/// Compute the 32-mer key for a given SA position in the padded `bases` slice.
#[inline]
fn key_for_position(pos: u64, bases: &[u8]) -> u64 {
    let start = pos as usize;
    let n = bases.len();
    let avail = n.saturating_sub(start).min(KMER_LEN);
    tokenize_32mer(&bases[start..start + avail], avail)
}

/// Build the inverse suffix array (ISA) from a suffix array.
///
/// For a suffix array `sa` of length `N`, the ISA satisfies:
/// `isa[sa[i]] = i` for all `i` in `0..N`.
///
/// The `sa` slice must contain positions in `0..N` (no position may exceed
/// `N - 1`). Returns `Err(Error::Internal)` if any position is out of range.
fn build_isa(sa: &[u64]) -> Result<Vec<u64>> {
    let n = sa.len();
    let mut isa = vec![0u64; n];
    for (i, &pos) in sa.iter().enumerate() {
        let p = pos as usize;
        if p >= n {
            return Err(Error::Internal {
                detail: format!("build_isa: SA position {pos} at index {i} exceeds sa.len()={n}"),
            });
        }
        isa[p] = i as u64;
    }
    Ok(isa)
}

/// Resolve a [`Prior::FastqHistogram`] from raw CLI values.
///
/// Returns `Err` if `base_weight` is not positive or if the histogram TSV
/// cannot be parsed.
pub fn prior_from_cli_fastq(histogram_path: &Path, base_weight: f64) -> Result<Prior> {
    if base_weight <= 0.0 {
        return Err(Error::Internal {
            detail: format!("--prior-fastq-base-weight must be > 0, got {base_weight}"),
        });
    }
    let keys_to_freq = crate::train::prior::parse_histogram_tsv(histogram_path)?;
    Ok(Prior::FastqHistogram {
        keys_to_freq,
        base_weight,
        path: Some(histogram_path.to_path_buf()),
    })
}
