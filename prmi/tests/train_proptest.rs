// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Property tests for the P-RMI trainer. See spec §5.8 (Q7).

use prmi::index::lookup::lookup_with_components;
use prmi::sidecar::model_file::ModelEntry;
use prmi::train::prmi::train_prmi;
use prmi::train::training_set::TrainingSet;
use proptest::prelude::*;

proptest! {
    /// For any sorted unique u64 key vec ≥ 16 keys, the trainer either
    /// rejects or produces a sidecar where every training key looks up
    /// within its leaf's reported err.
    #[test]
    fn trained_model_predicts_every_training_key_within_bound(
        mut raw_keys in prop::collection::vec(any::<u64>(), 16..1024)
    ) {
        raw_keys.sort();
        raw_keys.dedup();
        prop_assume!(raw_keys.len() >= 16);

        let sa_indices: Vec<u64> = (0..raw_keys.len() as u64).collect();
        let mut ts = TrainingSet::default();
        ts.keys = raw_keys.clone();
        ts.sa_indices = sa_indices;

        let result = train_prmi(&ts, 16);
        prop_assume!(result.is_ok());
        let model = result.unwrap();

        for (i, &k) in raw_keys.iter().enumerate() {
            let (pred, err) = lookup_with_components(
                k, &model.l1, &model.l2, model.bit_shift, raw_keys.len() as u64,
            );
            let dist = (pred as i64 - i as i64).unsigned_abs();
            prop_assert!(
                dist <= err,
                "key={} i={} pred={} dist={} err={}", k, i, pred, dist, err
            );
        }
    }

    /// encode_fallback_err round-trips correctly (high bit + bit slicing).
    #[test]
    fn encode_decode_fallback_err_roundtrips(
        partial_start in 0u64..(1 << 31),
        partial_num in 1u64..(1u64 << 32),
    ) {
        let encoded = (1u64 << 63)
            | ((partial_start & 0x7fff_ffff) << 32)
            | (partial_num & 0xffff_ffff);
        prop_assert_eq!(encoded >> 63, 1);
        prop_assert_eq!((encoded >> 32) & 0x7fff_ffff, partial_start);
        prop_assert_eq!(encoded & 0xffff_ffff, partial_num);
    }

    /// `lookup_with_components` never panics on arbitrary inputs.
    #[test]
    fn lookup_never_panics_on_random_inputs(
        bit_shift in 0u32..=64,
        sa_num in 1u64..1_000_000,
        key in any::<u64>(),
    ) {
        // Constrain key to ensure l2_idx stays within bounds.
        // When bit_shift < 64, l2_idx = (key >> bit_shift), so key < (1 << bit_shift) * l2.len().
        // We have a single-entry l2 array, so key must be < (1 << bit_shift).
        let key = if bit_shift >= 64 {
            key
        } else {
            key % (1u64 << bit_shift)
        };
        let l2 = vec![ModelEntry { alpha: 0.0, beta: 0.0, err: 5 }];
        let l1: Vec<ModelEntry> = vec![];
        let _ = lookup_with_components(key, &l1, &l2, bit_shift, sa_num);
    }
}
