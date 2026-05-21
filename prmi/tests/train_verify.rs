// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::lookup::lookup_with_components;
use prmi::train::prmi::train_prmi;
use prmi::train::training_set::TrainingSet;
use prmi::train::verify::compute_max_error_bound;

fn make_uniform_training_set(num_keys: usize) -> TrainingSet {
    let keys: Vec<u64> = (0..num_keys as u64).map(|i| i * (1u64 << 50)).collect();
    let sa_indices: Vec<u64> = (0..num_keys as u64).collect();
    let mut ts = TrainingSet::default();
    ts.keys = keys;
    ts.sa_indices = sa_indices;
    ts
}

#[test]
fn max_error_bound_dominates_predictions_but_is_tight() {
    let ts = make_uniform_training_set(4096);
    let model = train_prmi(&ts, 16).unwrap();

    let max_err = compute_max_error_bound(&model, &ts);

    // For every training pair, the prediction error must be ≤ max_err.
    let mut observed_max = 0u64;
    let n = ts.len() as u64;
    for (k, target) in ts.keys.iter().zip(ts.sa_indices.iter()) {
        let (pred, _err) = lookup_with_components(*k, &model.l1, &model.l2, model.bit_shift, n);
        let dist = (pred as i64 - *target as i64).unsigned_abs();
        assert!(dist <= max_err, "dist={dist} max_err={max_err}");
        if dist > observed_max {
            observed_max = dist;
        }
    }
    assert_eq!(
        observed_max, max_err,
        "compute_max_error_bound must return exactly the worst observed error"
    );

    // Sanity floor: max_err shouldn't be more than half the training-set size
    // (otherwise the index is essentially random).
    assert!(
        max_err < (ts.len() as u64) / 2,
        "max_err={max_err} is too large for ts.len()={}",
        ts.len()
    );
}

#[test]
fn empty_training_set_yields_zero_max_err() {
    let model = train_prmi(&make_uniform_training_set(4096), 16).unwrap();
    let empty_ts = TrainingSet::default();
    assert_eq!(compute_max_error_bound(&model, &empty_ts), 0);
}
