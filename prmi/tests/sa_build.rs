// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sa::build_suffix_array;

#[test]
fn matches_brute_force_on_short_input() {
    // 2-bit-coded "ACGTACGTAC" (10 bp).
    let bases: Vec<u8> = b"\x00\x01\x02\x03\x00\x01\x02\x03\x00\x01".to_vec();
    let sa = build_suffix_array(&bases).unwrap();

    // Brute-force SA for the same input.
    let n = bases.len();
    let mut expected: Vec<u64> = (0..n as u64).collect();
    expected.sort_by(|&a, &b| bases[a as usize..].cmp(&bases[b as usize..]));
    assert_eq!(sa, expected);
}

#[test]
fn handles_empty_input() {
    let sa = build_suffix_array(&[]).unwrap();
    assert!(sa.is_empty());
}
