// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::training_set::{uniform_training_set, TrainingSet};

#[test]
fn uniform_set_pairs_keys_with_sa_indices() {
    let bases: Vec<u8> = b"\x00\x01\x02\x03\x00\x01\x02\x03\x00\x01".to_vec();
    let sa = prmi::sa::build_suffix_array(&bases, 1).unwrap();
    let ts = uniform_training_set(&sa, &bases);

    assert_eq!(ts.len(), sa.len());
    assert!(!ts.is_empty());
    assert_eq!(ts.keys.len(), ts.sa_indices.len());

    // sa_indices is identity 0..n
    for (i, &idx) in ts.sa_indices.iter().enumerate() {
        assert_eq!(idx, i as u64, "sa_indices must be 0..n for uniform set");
    }

    // keys match what sa_to_keys would produce directly
    let direct_keys = prmi::train::keys::sa_to_keys(&sa, &bases);
    assert_eq!(ts.keys, direct_keys);
}

#[test]
fn empty_input_produces_empty_training_set() {
    let bases: Vec<u8> = vec![];
    let sa = prmi::sa::build_suffix_array(&bases, 1).unwrap();
    let ts = uniform_training_set(&sa, &bases);
    assert!(ts.is_empty());
    assert_eq!(ts.len(), 0);
}

#[test]
fn default_training_set_is_empty() {
    let ts = TrainingSet::default();
    assert!(ts.is_empty());
    assert_eq!(ts.len(), 0);
}
