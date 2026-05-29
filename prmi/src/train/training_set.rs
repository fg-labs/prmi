// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Training-set construction for the P-RMI trainer.
//!
//! A [`TrainingSet`] is a sorted list of `(key, sa_index)` pairs the trainer
//! learns to predict: given a 32-mer key, return the SA index of the matching
//! suffix array entry. For v0.1, [`uniform_training_set`] (unmasked) and
//! [`masked_training_set`] (mask-filtered) are shipped. v0.2 will add
//! `bed_filtered_training_set` (filters to on-target intervals) and v0.3 will
//! add `fastq_weighted_training_set` (weights by query-side frequency).

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::train::keys::sa_to_keys;
use crate::train::mask::{covered_by_bed, homopolymer_in_window, n_in_window, MaskConfig};
use crate::train::prior::Prior;

/// A sorted set of `(key, sa_index)` training pairs for the P-RMI trainer.
///
/// `keys[i]` and `sa_indices[i]` together form the i-th training pair: the
/// trainer learns to predict `sa_indices[i]` from `keys[i]`. The two vectors
/// always have the same length, and `keys` is non-decreasing.
///
/// `sa_num` is the total size of the **full** SA (not the number of training
/// pairs). It may be larger than `keys.len()` when masking is active, and is
/// used by the trainer to clamp predictions to the correct range.
///
/// `weights` is an optional per-pair weight vector. When `Some`, it has the
/// same length as `keys` and `sa_indices`. When `None`, all pairs are
/// implicitly weighted 1.0. The BED prior path populates this field.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TrainingSet {
    /// 32-mer keys in non-decreasing order.
    pub keys: Vec<u64>,
    /// SA-index target for each key — what the model learns to predict.
    pub sa_indices: Vec<u64>,
    /// Total number of entries in the full SA (may exceed `keys.len()` when
    /// masking is active). The trainer uses this as the prediction clamp bound.
    pub sa_num: u64,
    /// Optional per-pair training weights. `None` means uniform weight (1.0).
    /// When `Some`, the length equals `keys.len()`.
    pub weights: Option<Vec<f64>>,
}

impl TrainingSet {
    /// Number of training pairs.
    pub fn len(&self) -> usize {
        assert_eq!(self.keys.len(), self.sa_indices.len());
        self.keys.len()
    }

    /// `true` if no training pairs.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Produce a uniform training set: one `(key, sa_index)` pair per SA entry.
///
/// The SA must be the lex-sorted suffix array of `bases`. The returned
/// `TrainingSet` has `keys = sa_to_keys(sa, bases)` and
/// `sa_indices = 0..sa.len()`.
pub fn uniform_training_set(sa: &[u64], bases: &[u8]) -> TrainingSet {
    let keys = sa_to_keys(sa, bases);
    let sa_num = sa.len() as u64;
    let sa_indices: Vec<u64> = (0..sa_num).collect();
    TrainingSet {
        keys,
        sa_indices,
        sa_num,
        weights: None,
    }
}

/// Produce a mask-filtered training set: one `(key, sa_index)` pair per
/// unmasked SA entry.
///
/// The SA must be the lex-sorted suffix array of `bases`. For each SA entry
/// at index `sa_idx`, the reference position `sa[sa_idx]` is checked against
/// the enabled mask predicates; if any predicate fires, that entry is excluded
/// from the returned `TrainingSet`. The SA file on disk is **not** affected —
/// only the set of (key, SA-index) pairs used to fit and evaluate the model is
/// filtered.
///
/// `n_positions` must have the same length as `bases` and marks positions that
/// were originally N in the reference FASTA. Short windows at the end of the
/// reference (where `sa_pos + 32 > bases.len()`) are also excluded, consistent
/// with the sentinel-padding strategy in the trainer.
///
/// When `prior` is [`Prior::Bed`], the returned `TrainingSet::weights` field
/// is `Some(weights)` with each entry set to the BED weight for pairs inside
/// the BED region and 1.0 for pairs outside. When `prior` is
/// [`Prior::FastqHistogram`], `weights` is `Some(weights)` with each entry set
/// to `base_weight + log2(1 + freq(key))`. When `prior` is [`Prior::Uniform`],
/// `weights` is `None` (uniform weights are implicit).
pub fn masked_training_set(
    sa: &[u64],
    bases: &[u8],
    n_positions: &[bool],
    mask: &MaskConfig,
    prior: &Prior,
) -> TrainingSet {
    let n = bases.len();
    let mut keys: Vec<u64> = Vec::with_capacity(sa.len());
    let mut sa_indices: Vec<u64> = Vec::with_capacity(sa.len());
    let use_weights = !matches!(prior, Prior::Uniform);
    let mut weights: Vec<f64> = if use_weights {
        Vec::with_capacity(sa.len())
    } else {
        Vec::new()
    };

    for (sa_idx, &sa_pos) in sa.iter().enumerate() {
        let p = sa_pos as usize;

        // Skip short windows (past end of sequence).
        if p + KMER_LEN > n {
            continue;
        }

        // N-run mask.
        if mask.mask_n_runs && n_in_window(n_positions, p) {
            continue;
        }

        // Homopolymer mask.
        if let Some(k) = mask.mask_homopolymers {
            if homopolymer_in_window(bases, p, k) {
                continue;
            }
        }

        // BED mask.
        if let Some(ref intervals) = mask.mask_bed {
            if covered_by_bed(intervals, sa_pos) {
                continue;
            }
        }

        let avail = n.saturating_sub(p).min(KMER_LEN);
        let key = tokenize_32mer(&bases[p..p + avail], avail);
        keys.push(key);
        sa_indices.push(sa_idx as u64);
        if use_weights {
            weights.push(crate::train::prior::weight_for_pair(prior, key, sa_pos));
        }
    }

    TrainingSet {
        keys,
        sa_indices,
        sa_num: sa.len() as u64,
        weights: if use_weights { Some(weights) } else { None },
    }
}
