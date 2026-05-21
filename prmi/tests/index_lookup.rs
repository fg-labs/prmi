// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::lookup::lookup_with_components;
use prmi::sidecar::model_file::ModelEntry;

#[test]
fn l2_direct_path_no_fallback() {
    // 2 L2 leaves, no L1 fallback (high bit of err is 0).
    let l2 = vec![
        ModelEntry {
            alpha: 0.0,
            beta: 0.0,
            err: 4,
        },
        ModelEntry {
            alpha: 10.0,
            beta: 0.0,
            err: 4,
        },
    ];
    let l1: Vec<ModelEntry> = vec![];
    let sa_num = 100u64;
    let bit_shift = 63u32; // 2 leaves
    let (pred, err) = lookup_with_components(0, &l1, &l2, bit_shift, sa_num);
    assert_eq!(pred, 0);
    assert_eq!(err, 4);
    let (pred, _err) = lookup_with_components(1u64 << 63, &l1, &l2, bit_shift, sa_num);
    assert_eq!(pred, 10);
}

#[test]
fn fallback_via_l1_when_high_bit_set() {
    // L2[0] fallback: high bit set, partial_start=0, partial_num=2.
    // high bit set, partial_start=0 (bits 62..32), partial_num=2 (bits 31..0)
    let l2_err = (1u64 << 63) | 2u64;
    let l2 = vec![ModelEntry {
        alpha: 0.0,
        beta: 0.0,
        err: l2_err,
    }];
    let l1 = vec![
        ModelEntry {
            alpha: 5.0,
            beta: 0.0,
            err: 1,
        },
        ModelEntry {
            alpha: 7.0,
            beta: 0.0,
            err: 1,
        },
    ];
    let sa_num = 100;
    let bit_shift = 64; // 1 L2 leaf
    let (pred, err) = lookup_with_components(0, &l1, &l2, bit_shift, sa_num);
    assert_eq!(pred, 5);
    assert_eq!(err, 1);
}

#[test]
fn prediction_clamped_into_sa_range() {
    let l2 = vec![ModelEntry {
        alpha: -100.0,
        beta: 0.0,
        err: 0,
    }];
    let l1: Vec<ModelEntry> = vec![];
    let sa_num = 10;
    let bit_shift = 64;
    let (pred, _err) = lookup_with_components(0, &l1, &l2, bit_shift, sa_num);
    assert_eq!(pred, 0);

    let l2 = vec![ModelEntry {
        alpha: 1000.0,
        beta: 0.0,
        err: 0,
    }];
    let (pred, _err) = lookup_with_components(0, &l1, &l2, bit_shift, sa_num);
    assert_eq!(pred, sa_num - 1);
}
