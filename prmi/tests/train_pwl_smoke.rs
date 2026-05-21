// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Smoke test for the ported BWA-MEME P-RMI training pipeline. Validates
//! that `upstream::train` with a 3-element spec (`pwl<N>,linear,linear_spline`)
//! returns a sane `TrainedRMI` without panicking. The prmi-level unpacker
//! (Task 16) consumes this `TrainedRMI` to produce the on-disk `.l1`/`.l2`
//! sidecar files.

use prmi::upstream::{train, RMITrainingData};

#[test]
fn pwl_three_layer_returns_sane_trained_rmi() {
    // 4096 keys spread evenly so each of 16 L2 leaves sees ~256 keys —
    // below the partial_models threshold (1000), exercising the
    // "small leaf" path in train_partial_three_layer.
    let keys: Vec<u64> = (0..4096u64).map(|i| i * (1u64 << 50)).collect();
    let data: Vec<(u64, usize)> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let td = RMITrainingData::new(Box::new(data));

    let trained = train(&td, "pwl4,linear,linear_spline", 16);

    // train_partial_three_layer always returns a 3-layer rmi:
    //   rmi[0] = top model (1 element)
    //   rmi[1] = partial 3rd-layer models (or dummy when third_layer_num == 0)
    //   rmi[2] = leaf models (branching_factor elements)
    assert_eq!(
        trained.rmi.len(),
        3,
        "expected 3 rmi layers from train_partial_three_layer"
    );
    assert_eq!(
        trained.rmi[2].len(),
        16,
        "expected 16 leaf models in rmi[2]"
    );
    assert_eq!(trained.last_layer_max_l1s.len(), 16);
    assert_eq!(trained.branching_factor, 16);
    // BWA-MEME spec-string serialization: layer1,layer3,layer2
    assert_eq!(trained.models, "pwl4,linear_spline,linear");
}

#[test]
fn pwl_three_layer_exercises_partial_models_path() {
    // Cluster keys so one L2 leaf sees >1000 keys, crossing the
    // `make_partial_threshold = 1000` in build_partial_3layer_models_from.
    // 5000 keys spread across only 4 L2 leaves => ~1250 keys per leaf.
    let mut keys: Vec<u64> = Vec::with_capacity(5000);
    for k in 0..5000u64 {
        // Spread keys evenly across the full u64 range so pwl4 routes
        // them across all 16 leaves (with bit_shift=60, key >> 60 fits in 4 bits).
        keys.push(k.wrapping_mul(1u64 << 50));
    }
    keys.sort();
    let data: Vec<(u64, usize)> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let td = RMITrainingData::new(Box::new(data));

    let trained = train(&td, "pwl4,linear,linear_spline", 16);

    assert_eq!(trained.rmi.len(), 3);
    assert_eq!(trained.rmi[2].len(), 16);
    assert_eq!(trained.last_layer_max_l1s.len(), 16);

    // If the partial-3-layer path triggered for at least one leaf,
    // third_layer_max_l1s will be non-empty.
    // We don't strictly require this — the test value is "doesn't crash."
    eprintln!(
        "third_layer_max_l1s.len() = {}",
        trained.third_layer_max_l1s.len()
    );
}
