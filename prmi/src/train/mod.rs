// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer: builds a prmi sidecar from a reference FASTA.

pub mod config;
pub mod keys;
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
use crate::train::prmi::train_prmi;
use crate::train::training_set::uniform_training_set;
use crate::train::verify::compute_max_error_bound;
use std::path::Path;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Build a complete sidecar (`.meta`, `.sa`, `.l1`, `.l2`) from a reference FASTA.
///
/// `prefix` is the output prefix; e.g., for `/data/ref.fa.prmi` the four
/// files become `/data/ref.fa.prmi.{meta,sa,l1,l2}`.
///
/// `l2_leaf_count` must be a power of two ≥ 2. For human-genome scale, 2^28
/// is typical; for tests, 2^4 = 16 is fine.
///
/// ## Sentinel padding
///
/// To ensure every SA entry covers a full `KMER_LEN`-mer, the reference is
/// extended with `KMER_LEN - 1` T-bases before suffix-array construction.
/// SA entries pointing into the padding region are then excluded from the
/// training set and the `.sa` file, matching BWA-MEME's indexer behaviour.
/// The SHA-256 hash and `size_bytes` in `.meta` still reflect the original
/// (unpadded) FASTA, not the extended sequence.
pub fn build_sidecar(ref_fa: &Path, prefix: &Path, l2_leaf_count: u64) -> Result<()> {
    let (mut bases, _fa_stats, sha256_hex, fa_size_bytes) = fasta_to_2bit_with_sha256(ref_fa)?;

    // Record the original genome length before padding.
    let genome_len = bases.len() as u64;

    // Append KMER_LEN - 1 T-sentinel bases so that every position in the
    // original genome has a full 32-mer available without wrapping.
    bases.extend(std::iter::repeat_n(BASE_T, KMER_LEN - 1));

    // Build the SA over the padded sequence.
    let full_sa = build_suffix_array(&bases)?;

    // Filter to only entries pointing into the original genome (positions
    // 0..genome_len). The filtered slice is a subsequence of the sorted SA,
    // so it is still non-decreasing in key space.
    let sa: Vec<u64> = full_sa
        .into_iter()
        .filter(|&pos| pos < genome_len)
        .collect();

    let ts = uniform_training_set(&sa, &bases);
    let model = train_prmi(&ts, l2_leaf_count)?;
    let max_err = compute_max_error_bound(&model, &ts);

    let paths = SidecarPaths::from_prefix(prefix);

    // .sa — write only the genome-region entries.
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
        },
        rmi: RmiSpec {
            spec,
            l2_leaf_count,
            bit_shift: model.bit_shift,
            max_error_bound: max_err,
        },
        priors: Priors {
            kind: "uniform".into(),
        },
    };
    meta.write_file(&paths.meta)?;
    Ok(())
}
