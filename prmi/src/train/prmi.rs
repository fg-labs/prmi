// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! P-RMI training driver. Wraps Marcus's upstream + BWA-MEME's
//! `pwl,linear,linear_spline` trainer and reshapes the resulting trained
//! parameters into the (L1 fallback array, L2 routing layer) layout the
//! prmi reader consumes — see v0.1 brief §4 / §4.4.

use crate::error::{Error, Result};
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

/// Train a P-RMI on the supplied training set. `l2_leaf_count` must be a
/// power of two ≥ 2. The training-set keys must be non-decreasing.
pub fn train_prmi(ts: &TrainingSet, l2_leaf_count: u64) -> Result<PrmiModel> {
    if !l2_leaf_count.is_power_of_two() || l2_leaf_count < 2 {
        return Err(Error::Internal {
            detail: format!("l2_leaf_count={l2_leaf_count} must be a power of two ≥ 2"),
        });
    }
    // Reject l2_leaf_count < 16 (i.e. pwl bit-width < 4). Originally added
    // as a workaround for an upstream `minus_epsilon` underflow on key=0;
    // under Phase 5-rev this becomes a documented unsupported configuration
    // (Fulcrum trainer never produces a pwl<4 layout).
    let pwl_bits = l2_leaf_count.trailing_zeros();
    if pwl_bits < 4 {
        return Err(Error::Internal {
            detail: format!(
                "l2_leaf_count={l2_leaf_count} would use pwl{pwl_bits}; \
                 minimum supported value is 16 (pwl4)"
            ),
        });
    }
    if ts.is_empty() {
        return Err(Error::Internal {
            detail: "empty training set".into(),
        });
    }
    Err(Error::Internal {
        detail: "train_prmi is being reimplemented in Phase 5-rev; see \
                 docs/superpowers/specs/2026-05-21-prmi-cleanroom-trainer.md".into(),
    })
}

