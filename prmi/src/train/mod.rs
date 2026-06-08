// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer: builds a prmi sidecar from a reference FASTA or bwa `.pac`.

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
use crate::index::smem::PacEncoding;
use crate::index::LearnedIndex;
use crate::sidecar::kmt_file::KmtFileWriter;
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use crate::sidecar::meta::{Meta, Priors, Prmi, Ref, RmiSpec, Sa};
use crate::sidecar::model_file::{ModelFileWriter, ModelLayer};
use crate::sidecar::sa_file::{SaFileWriter, BPE_MODE2};
use crate::sidecar::SidecarPaths;
use crate::train::config::{MemoryMode, TrainerConfig};
use crate::train::mask::MaskConfig;
use crate::train::prior::Prior;
use crate::train::trainer::default_l2_leaf_count;
use crate::train::training_set::{masked_training_set, streamed_training_set};
use crate::train::verify::compute_error_distribution;
use std::path::Path;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Where the forward reference bases come from.
pub enum BuildSource<'a> {
    /// bwa's forward `.pac` — byte-identical to the FMI (N already substituted).
    Pac(&'a Path),
    /// A reference FASTA — NOT byte-identical to a bwa FMI (prmi maps N→A).
    Fasta(&'a Path),
}

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
/// `threads` is passed to [`crate::sa::build_gsa`]: `0` = auto (OpenMP default,
/// typically the CPU count), `1` = single-threaded, `N > 1` = exactly N threads.
///
/// ## 2× (forward + reverse-complement) suffix array
///
/// The SA is the generalized suffix array of the doubled text
/// `[Fwd || RC] + sentinel` in the `b+1` alphabet (see
/// [`crate::sa::build_doubled_2x_text`]), byte-identical in order to the FMI.
/// All `2 * l_pac + 1` entries are retained on disk — including the RC half and
/// the sentinel/empty-suffix row — so SA positions are *doubled coordinates* in
/// `[0, 2 * l_pac]`. The SHA-256 hash and `size_bytes` in `.meta` reflect the
/// original (forward-only) FASTA.
pub fn build_sidecar(
    ref_fa: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> Result<()> {
    build_sidecar_core(
        BuildSource::Fasta(ref_fa),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        None,
    )
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
    build_sidecar_core(
        BuildSource::Fasta(ref_fa),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        trainer_config,
    )
}

/// Build a sidecar from bwa's forward `.pac` (byte-identical path).
///
/// The `.pac` carries bwa's N→random substitution, so the SA built here is
/// byte-identical in rank order to the bwa FM-index. `pac_sha256` in the
/// resulting `.meta` records the SHA-256 of the `.pac` file for provenance.
pub fn build_sidecar_from_pac(
    pac: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> Result<()> {
    build_sidecar_from_pac_with_config(pac, prefix, l2_leaf_count, mask, threads, None)
}

/// Build a sidecar from bwa's forward `.pac` (byte-identical path), with an
/// optional [`TrainerConfig`] (e.g. to select `--store-keys` / Mode2).
pub fn build_sidecar_from_pac_with_config(
    pac: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    build_sidecar_core(
        BuildSource::Pac(pac),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        trainer_config,
    )
}

/// Core sidecar builder. All public entry points delegate here.
fn build_sidecar_core(
    source: BuildSource<'_>,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    let config = trainer_config.unwrap_or_default();

    let (bases, n_positions, ref_path, ref_sha256, ref_size_bytes, pac_sha256) = match source {
        BuildSource::Fasta(p) => {
            log::warn!(
                "building sidecar from FASTA ({}): result is NOT byte-identical to a bwa \
                 FM-index (prmi maps N→A; bwa substitutes N→random at pack time). \
                 For byte-identical builds use --pac / BuildSource::Pac.",
                p.display()
            );
            let (bases, n_positions, _stats, sha, size) = fasta_to_2bit_with_sha256(p)?;
            (bases, n_positions, p.display().to_string(), sha, size, None)
        }
        BuildSource::Pac(p) => {
            let (bases, _l_pac) = crate::pac::read_bwa_pac_forward(p)?;
            let psha = crate::pac::pac_sha256(p)?;
            let n_positions = vec![false; bases.len()];
            let ref_path = p.display().to_string();
            let ref_sha256 = psha.clone();
            let ref_size_bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            (
                bases,
                n_positions,
                ref_path,
                ref_sha256,
                ref_size_bytes,
                Some(psha),
            )
        }
    };

    // Forward (unpadded) genome length = l_pac.
    let genome_len = bases.len() as u64;

    // Build the 2× generalized suffix array over [Fwd||RC]+sentinel. No
    // T-padding and no filtering: every entry is retained, including the
    // sentinel/empty-suffix row, so SA order is byte-identical to the FMI.
    // `sa` and `text` are shared (Arc) so the streamed training set can borrow
    // them zero-copy; both are also needed later for the `.sa` write.
    let text = std::sync::Arc::new(crate::sa::build_doubled_2x_text(&bases));
    let sa = std::sync::Arc::new(crate::sa::build_gsa(&text, threads)?);

    // Resolve the optional l2_leaf_count: use provided value or auto-scale.
    let l2_leaf_count = l2_leaf_count.unwrap_or_else(|| default_l2_leaf_count(sa.len()));

    // Dense/streamed fast path (the byte-identical `.pac` build): a uniform
    // prior with no homopolymer/BED mask and no effective N-mask. On this path
    // the keys are streamed from `sa`+`text` and the SA-index targets are
    // virtual, so neither the ~51.5 GB key vector, the ~51.5 GB target vector,
    // nor the `text_bases` / `n_positions_2x` arrays are materialised. The
    // resulting (key, sa_index) pairs — and the model — are byte-identical to
    // the materialized path.
    let no_n_effect = !mask.mask_n_runs || n_positions.iter().all(|&b| !b);
    let virtualize = matches!(config.prior, Prior::Uniform)
        && mask.mask_homopolymers.is_none()
        && mask.mask_bed.is_none()
        && no_n_effect;
    let ts = if virtualize {
        streamed_training_set(std::sync::Arc::clone(&sa), std::sync::Arc::clone(&text))
    } else {
        // Materialized path. The SA is in doubled coordinates over `text`
        // (length 2*l_pac+1), so the training set indexes bases/n_positions in
        // those same coordinates: the 0..=3 base array from `text` (1..=4,
        // sentinel 0 → T) and an N bitmap whose forward half carries the FASTA N
        // flags and whose RC half mirrors them (the RC of an ambiguous base is
        // itself ambiguous — those RC suffixes must see the same `mask_n_runs`
        // filtering as the forward half; the sentinel row stays non-N).
        let text_bases: Vec<u8> = text
            .iter()
            .map(|&v| crate::sa::text_value_to_base(v))
            .collect();
        let l_pac = n_positions.len();
        let mut n_positions_2x = vec![false; text.len()];
        // Forward half: positions 0..l_pac.
        n_positions_2x[..l_pac].copy_from_slice(&n_positions);
        // RC half: forward position i maps to doubled coordinate l_pac+(l_pac-1-i),
        // matching `doubled_base_at`'s reverse-complement mapping.
        for (i, &is_n) in n_positions.iter().enumerate() {
            if is_n {
                n_positions_2x[l_pac + (l_pac - 1 - i)] = true;
            }
        }
        masked_training_set(&sa, &text_bases, &n_positions_2x, &mask, &config.prior)
    };
    let model = crate::train::trainer::train_with_config(&ts, l2_leaf_count, &config)?;
    // Streaming histogram: returns the percentile distribution and the max in
    // two passes, without materialising a ~51.5 GB per-key error vector. The
    // returned `max` equals the former `compute_max_error_bound`.
    let (p50, p90, p99, max_err) = compute_error_distribution(&model, &ts);
    log::info!("prmi model error bound: max={max_err} p50={p50} p90={p90} p99={p99}");

    let paths = SidecarPaths::from_prefix(prefix);

    // Remove any pre-existing `.kmt` so a rebuild (especially one WITHOUT
    // `--kmer-table-k`) can never inherit a stale table for the old reference.
    // A fresh table is written at the end if requested. Propagate anything other
    // than "already absent": a swallowed permission/I/O error would silently
    // leave the stale table in place for the next open.
    match std::fs::remove_file(&paths.kmt) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(Error::Io {
                path: paths.kmt.clone(),
                source: e,
            })
        }
    }

    // .sa — write all genome-region entries in the requested memory mode.
    let memory_mode = config.memory_mode;
    match memory_mode {
        MemoryMode::Mode1 => {
            // Position-only, 5 B/entry. Original layout; unchanged.
            let mut w = SaFileWriter::create(&paths.sa, sa.len() as u64)?;
            for &pos in sa.iter() {
                w.write_position(pos)?;
            }
            w.finish()?;
        }
        MemoryMode::Mode2 => {
            // Position + 32-mer key, 13 B/entry.
            let mut w = SaFileWriter::create_with_mode(&paths.sa, sa.len() as u64, BPE_MODE2)?;
            for &pos in sa.iter() {
                let key = key_for_position_2x(pos, &text);
                w.write_entry_with_key(pos, key)?;
            }
            w.finish()?;
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
    };

    let meta = Meta {
        prmi: Prmi {
            magic: META_MAGIC.into(),
            format_version: FORMAT_VERSION,
            trainer_version: format!("prmi={}", env!("CARGO_PKG_VERSION")),
            created_utc,
        },
        ref_: Ref {
            path: ref_path,
            sha256: ref_sha256,
            size_bytes: ref_size_bytes,
        },
        sa: Sa {
            num_entries: sa.len() as u64,
            bytes_per_entry: memory_mode.bytes_per_entry(),
            encoding: memory_mode.encoding_name().to_string(),
            mode: mode_str,
            skc_cache_size: skc_cache_size_meta,
            strand: "forward_rc_2x".into(),
            masked_n_runs: mask.mask_n_runs,
            masked_homopolymers: mask.mask_homopolymers,
            masked_bed: masked_bed_path,
            l_pac: Some(genome_len),
            stored_keys: Some(memory_mode.bytes_per_entry() >= 13),
            pac_sha256,
        },
        rmi: RmiSpec {
            spec,
            l2_leaf_count,
            bit_shift: model.bit_shift,
            max_error_bound: max_err,
            err_p50: Some(p50),
            err_p90: Some(p90),
            err_p99: Some(p99),
        },
        priors,
    };
    meta.write_file(&paths.meta)?;

    // ── optional: build and persist the forward k-mer table ───────────────
    if let Some(requested_k) = config.kmer_table_k {
        // Reject a zero order rather than coercing it to a 1-mer table — the
        // caller would silently get a different artifact than requested.
        if requested_k == 0 {
            return Err(Error::InvalidInput {
                detail: "kmer_table_k must be >= 1".into(),
            });
        }
        let sa_num = sa.len() as u64;
        // Cap k so 4^k does not dwarf the SA (avoids a huge mostly-empty table
        // on small references); the table is self-describing via its header.
        let k_max = ((sa_num as f64).log2() / 2.0).floor().clamp(1.0, 16.0) as u32;
        let k = requested_k.min(k_max);
        if k < requested_k {
            log::warn!(
                "k-mer table k capped to {k} (requested {requested_k}) for this reference size"
            );
        }

        // Reopen the just-written sidecar to drive the table search: it streams
        // the SA positions (and, in key-storing modes, the inline 32-mer keys)
        // from the `.sa` via the page cache, reusing the exact compare
        // `forward_spectrum` uses. We deliberately do NOT rebuild positions/keys
        // from the in-memory `sa` — that would either recompute every key per
        // probe (slower) or re-materialize the ~51 GB key vector the streamed
        // build avoids, defeating the memory budget.
        //
        // The reference bases, however, are taken from the in-memory `bases` we
        // already hold (decoded from the `.pac` for a Pac build, or the FASTA
        // 2-bit), instead of re-reading the entire `.pac` from disk. `bases` is
        // byte-identical to the decoded `.pac`, so the unpacked compare yields a
        // byte-identical table to a packed `.pac` compare — verified by
        // `kmer_table_packed_pac_equals_unpacked_bases`.
        let idx = LearnedIndex::open(prefix)?;
        let table = idx.build_kmer_table(k, &bases, PacEncoding::Unpacked);
        let (tk, tlo, thi) = table.parts();
        // Bind the table to its reference: prefer `pac_sha256` (the byte-identical
        // build), else the FASTA ref sha. A 0 digest (unparsable) just means the
        // open path falls back to the full search.
        let digest_hex = meta
            .sa
            .pac_sha256
            .as_deref()
            .unwrap_or(meta.ref_.sha256.as_str());
        let ref_digest = crate::sidecar::kmt_file::hex32(digest_hex).unwrap_or([0u8; 32]);
        KmtFileWriter::write(&paths.kmt, tk, sa_num, &ref_digest, tlo, thi)?;
        log::info!("prmi forward k-mer table: k={tk}, sa_num={sa_num}");
    }

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
                return Err(Error::InvalidInput {
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

/// Compute the 32-mer key for a doubled-coordinate SA position over the
/// `b+1`-alphabet 2× text. Text values are 1..=4 (+ 0 sentinel); shift back to
/// 0..=3 for tokenisation, mapping the sentinel `0` to a T (3) terminator (it
/// only appears once, at the very end, past any real 32-mer window).
#[inline]
fn key_for_position_2x(pos: u64, text: &[u8]) -> u64 {
    let start = pos as usize;
    let avail = text.len().saturating_sub(start).min(KMER_LEN);
    let mut window = [BASE_T; KMER_LEN];
    for (slot, &v) in window.iter_mut().zip(&text[start..start + avail]) {
        *slot = crate::sa::text_value_to_base(v);
    }
    tokenize_32mer(&window[..avail], avail)
}

/// Compute the 32-mer key for a forward-only SA position. Legacy helper from the v1 forward-only build path; retained pending removal in a later plan.
#[allow(dead_code)]
#[inline]
fn key_for_position(pos: u64, bases: &[u8]) -> u64 {
    let start = pos as usize;
    let n = bases.len();
    let avail = n.saturating_sub(start).min(KMER_LEN);
    tokenize_32mer(&bases[start..start + avail], avail)
}

/// Resolve a [`Prior::FastqHistogram`] from raw CLI values.
///
/// Returns `Err` if `base_weight` is not positive or if the histogram TSV
/// cannot be parsed.
pub fn prior_from_cli_fastq(histogram_path: &Path, base_weight: f64) -> Result<Prior> {
    if base_weight <= 0.0 {
        return Err(Error::InvalidInput {
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

/// Validate that the mutually-exclusive prior inputs are not both supplied.
///
/// `--prior-bed` and `--prior-fastq-histogram` select different weighting
/// schemes and cannot be combined. Returns `Err(Error::InvalidInput)` if both
/// paths are `Some`. This is the shared guard the CLI calls before resolving a
/// [`Prior`]; living here next to `prior_from_cli` / `prior_from_cli_fastq`
/// lets it be unit-tested without driving the binary.
pub fn validate_prior_paths(prior_bed: Option<&Path>, prior_fastq: Option<&Path>) -> Result<()> {
    if prior_bed.is_some() && prior_fastq.is_some() {
        return Err(Error::InvalidInput {
            detail: "--prior-bed and --prior-fastq-histogram are mutually exclusive; \
                     supply at most one"
                .to_string(),
        });
    }
    Ok(())
}
