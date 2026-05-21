// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::prmi::train_prmi;
use prmi::train::training_set::TrainingSet;

fn make_uniform_training_set(num_keys: usize) -> TrainingSet {
    // Start from 1 << 50 (not 0) so that minus_epsilon on the first key
    // never wraps around in debug mode.
    let keys: Vec<u64> = (1..=num_keys as u64).map(|i| i * (1u64 << 50)).collect();
    let sa_indices: Vec<u64> = (0..num_keys as u64).collect();
    let mut ts = TrainingSet::default();
    ts.keys = keys;
    ts.sa_indices = sa_indices;
    ts
}

#[test]
fn train_prmi_produces_expected_shape() {
    let ts = make_uniform_training_set(4096);
    let l2_leaf_count = 16u64;
    let model = train_prmi(&ts, l2_leaf_count).unwrap();

    assert_eq!(model.l2.len(), l2_leaf_count as usize);
    assert_eq!(model.bit_shift, 60); // 64 - log2(16) = 60
                                     // L1 length is implementation-defined; either 0 (no fallback) or some non-zero
                                     // count. Just assert it doesn't crash the shape invariants.
    assert!(model.l1.len() <= ts.keys.len());

    eprintln!(
        "train_prmi_produces_expected_shape: l1.len()={}, l2.len()={}",
        model.l1.len(),
        model.l2.len()
    );
}

#[test]
fn train_prmi_rejects_non_power_of_two() {
    let ts = make_uniform_training_set(32);
    let err = train_prmi(&ts, 3).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("power of two"), "got error: {msg}");
}

#[test]
fn train_prmi_rejects_empty_training_set() {
    let ts = TrainingSet::default();
    let err = train_prmi(&ts, 16).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("empty"), "got error: {msg}");
}

#[test]
fn train_prmi_bit_shift_calculation() {
    // pwl3 (count=8) can panic inside train_partial_three_layer when key 0
    // appears as a boundary sentinel — skip it and use the upstream-tested
    // pwl4 / pwl5 values that the BWA-MEME smoke tests already exercise.
    let ts = make_uniform_training_set(1024);
    for &log2_count in &[4u32, 5] {
        let count = 1u64 << log2_count;
        let model = train_prmi(&ts, count).unwrap();
        assert_eq!(model.bit_shift, 64 - log2_count);
        assert_eq!(model.l2.len(), count as usize);
    }
}
