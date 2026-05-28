// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sa::build_suffix_array;

#[test]
fn matches_brute_force_on_short_input() {
    // 2-bit-coded "ACGTACGTAC" (10 bp).
    let bases: Vec<u8> = b"\x00\x01\x02\x03\x00\x01\x02\x03\x00\x01".to_vec();
    let sa = build_suffix_array(&bases, 1).unwrap();

    // Brute-force SA for the same input.
    let n = bases.len();
    let mut expected: Vec<u64> = (0..n as u64).collect();
    expected.sort_by(|&a, &b| bases[a as usize..].cmp(&bases[b as usize..]));
    assert_eq!(sa, expected);
}

#[test]
fn handles_empty_input() {
    let sa = build_suffix_array(&[], 1).unwrap();
    assert!(sa.is_empty());
}

/// Verify that multi-threaded SA construction produces the same result as
/// single-threaded for a moderately sized input. The input is long enough
/// that libsais will actually engage multiple threads internally.
#[test]
fn single_and_multi_threaded_produce_identical_results() {
    // 4096-byte pseudo-random 2-bit sequence.
    let bases: Vec<u8> = {
        let mut x: u64 = 0xDEAD_BEEF_C0DE_CAFE;
        (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 3) as u8
            })
            .collect()
    };

    let sa_single = build_suffix_array(&bases, 1).unwrap();
    let sa_multi = build_suffix_array(&bases, 4).unwrap();

    assert_eq!(
        sa_single, sa_multi,
        "single-threaded and 4-thread SA construction must produce identical results"
    );
}

/// Verify that threads=0 (auto) produces the same result as single-threaded.
#[test]
fn auto_thread_count_produces_correct_results() {
    let bases: Vec<u8> = b"\x00\x01\x02\x03\x00\x01\x02\x03\x00\x01".to_vec();
    let sa_auto = build_suffix_array(&bases, 0).unwrap();
    let sa_single = build_suffix_array(&bases, 1).unwrap();
    assert_eq!(
        sa_auto, sa_single,
        "auto thread count must produce the same SA as single-threaded"
    );
}
