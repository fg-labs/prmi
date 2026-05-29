// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer: builds a prmi sidecar from a reference FASTA.

pub mod config;
pub mod keys;
pub mod mask;
pub mod prmi;
pub mod trainer;
pub mod training_set;
pub mod verify;

use crate::encoding::{BASE_T, KMER_LEN};
use crate::error::{Error, Result};
use crate::fasta::fasta_to_2bit_with_sha256;
use crate::sa::build_suffix_array;
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use crate::sidecar::meta::{Meta, Priors, Prmi, Ref, RmiSpec, Sa};
use crate::sidecar::model_file::{ModelFileWriter, ModelLayer};
use crate::sidecar::sa_file::SaFileWriter;
use crate::sidecar::SidecarPaths;
use crate::train::mask::MaskConfig;
use crate::train::prmi::train_prmi;
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

    let ts = masked_training_set(&sa, &bases, &n_positions, &mask);
    let model = train_prmi(&ts, l2_leaf_count)?;
    let max_err = compute_max_error_bound(&model, &ts);

    let paths = SidecarPaths::from_prefix(prefix);

    // .sa — write only the genome-region entries (unaffected by masking).
    {
        let mut w = SaFileWriter::create(&paths.sa, sa.len() as u64)?;
        for &pos in &sa {
            w.write_position(pos)?;
        }
        w.finish()?;
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
            bytes_per_entry: 5,
            encoding: "packed_lo8_hi32".into(),
            // Memory modes 2/3 and suffix-key-cache land in PR #4b; the
            // uniform trainer always writes the mode-1 (position-only) layout.
            mode: "1".into(),
            skc_cache_size: None,
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
        // BED / FASTQ-histogram priors land in PR #4b; the uniform trainer
        // always records the no-prior case (all extension fields `None`).
        priors: Priors {
            kind: "uniform".into(),
            bed: None,
            weight: None,
            histogram: None,
            base_weight: None,
            formula: None,
        },
    };
    meta.write_file(&paths.meta)?;
    Ok(())
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
