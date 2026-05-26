// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! P-RMI training driver. Delegates to the Fulcrum-authored trainer at
//! `crate::train::trainer`, which composes Marcus's RMI primitives into the
//! (L1 fallback array, L2 routing layer) layout the prmi reader consumes —
//! see v0.1 brief §4 / §4.4.

use crate::error::Result;
use crate::sidecar::model_file::ModelEntry;
use crate::train::training_set::TrainingSet;

/// An in-memory P-RMI model ready to be serialized into the `.l1` / `.l2`
/// sidecar files.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PrmiModel {
    /// Flat L1 fallback array. Empty when no L2 leaf required a fallback.
    pub l1: Vec<ModelEntry>,
    /// L2 routing layer. Length always equals `l2_leaf_count`.
    pub l2: Vec<ModelEntry>,
    /// `bit_shift = 64 - log2(l2_leaf_count)`. Reader uses this to compute
    /// the L2 index from a key.
    pub bit_shift: u32,
}

/// Train a P-RMI on the supplied training set using default
/// [`TrainerConfig`](crate::train::config::TrainerConfig).
///
/// `l2_leaf_count` must be a power of two ≥ 16. The training-set keys must
/// be non-decreasing.
///
/// Delegates to [`crate::train::trainer::train_with_config`] with
/// [`crate::train::config::TrainerConfig::default()`].
pub fn train_prmi(ts: &TrainingSet, l2_leaf_count: u64) -> Result<PrmiModel> {
    crate::train::trainer::train_with_config(
        ts,
        l2_leaf_count,
        &crate::train::config::TrainerConfig::default(),
    )
}
