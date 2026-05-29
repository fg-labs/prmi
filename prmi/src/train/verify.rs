// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Brute-force verification pass — computes the global maximum absolute
//! prediction error across a training set. Stored in `.meta` as
//! `max_error_bound`; runtime uses it to size local-search bounds.

use crate::index::lookup::lookup_with_components;
use crate::train::prmi::PrmiModel;
use crate::train::training_set::TrainingSet;

/// Brute-force pass: predict every training key, return the max absolute
/// prediction error. Becomes `max_error_bound` in the sidecar `.meta` header.
///
/// For v0.1 uniform priors, `ts.sa_indices[i] == i`, and the error is
/// `|pred - i|`. For future v0.2/v0.3 priors with non-identity sa_indices,
/// this function correctly compares against `ts.sa_indices[i]`.
pub fn compute_max_error_bound(model: &PrmiModel, ts: &TrainingSet) -> u64 {
    // Use ts.sa_num (full SA size) as the clamp bound, not ts.len().
    // When masking is active, ts.sa_num > ts.len() and sa_indices in the
    // training set may reference positions up to ts.sa_num - 1.
    let sa_num = ts.sa_num;
    let mut max_err = 0u64;
    for (k, target) in ts.keys.iter().zip(ts.sa_indices.iter()) {
        let (pred, _err) =
            lookup_with_components(*k, &model.l1, &model.l2, model.bit_shift, sa_num);
        let d = (pred as i64 - *target as i64).unsigned_abs();
        if d > max_err {
            max_err = d;
        }
    }
    max_err
}
