// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Training-set construction for the P-RMI trainer.
//!
//! A [`TrainingSet`] is a sorted list of `(key, sa_index)` pairs the trainer
//! learns to predict: given a 32-mer key, return the SA index of the matching
//! suffix array entry. For v0.1, only [`uniform_training_set`] is shipped —
//! it produces one training pair per SA entry. v0.2 will add
//! `bed_filtered_training_set` (filters to on-target intervals) and v0.3 will
//! add `fastq_weighted_training_set` (weights by query-side frequency).

use crate::train::keys::sa_to_keys;

/// A sorted set of `(key, sa_index)` training pairs for the P-RMI trainer.
///
/// `keys[i]` and `sa_indices[i]` together form the i-th training pair: the
/// trainer learns to predict `sa_indices[i]` from `keys[i]`. The two vectors
/// always have the same length, and `keys` is non-decreasing.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TrainingSet {
    /// 32-mer keys in non-decreasing order.
    pub keys: Vec<u64>,
    /// SA-index target for each key — what the model learns to predict.
    pub sa_indices: Vec<u64>,
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
    let sa_indices: Vec<u64> = (0..sa.len() as u64).collect();
    TrainingSet { keys, sa_indices }
}
