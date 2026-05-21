// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::encoding::{base_to_2bit, tokenize_32mer};

#[test]
fn ascii_to_2bit_maps_acgt() {
    assert_eq!(base_to_2bit(b'A'), Some(0));
    assert_eq!(base_to_2bit(b'C'), Some(1));
    assert_eq!(base_to_2bit(b'G'), Some(2));
    assert_eq!(base_to_2bit(b'T'), Some(3));
    assert_eq!(base_to_2bit(b'a'), Some(0));
    assert_eq!(base_to_2bit(b'N'), None);
    assert_eq!(base_to_2bit(b'X'), None);
}

#[test]
fn tokenize_full_kmer_msb_first() {
    let bases = [0u8; 32];
    assert_eq!(tokenize_32mer(&bases, 32), 0);

    let bases = [3u8; 32];
    assert_eq!(tokenize_32mer(&bases, 32), u64::MAX);

    let mut bases = [0u8; 32];
    bases[0] = 3;
    assert_eq!(tokenize_32mer(&bases, 32), 0xC000_0000_0000_0000);

    let mut bases = [0u8; 32];
    bases[31] = 1;
    assert_eq!(tokenize_32mer(&bases, 32), 0x0000_0000_0000_0001);
}

#[test]
fn tokenize_short_kmer_pads_with_t() {
    let bases = [0u8; 32];
    let expected_pad = u64::MAX >> 2;
    assert_eq!(tokenize_32mer(&bases, 1), expected_pad);

    assert_eq!(tokenize_32mer(&[], 0), u64::MAX);
}

#[test]
fn tokenize_with_short_slice() {
    // Exercise the len.min(bases.len()) clamp: pass len == bases.len() == 3.
    let bases = [0u8, 1u8, 2u8];
    let key = tokenize_32mer(&bases[..3], 3);
    // First 3 positions: A(00), C(01), G(10); remaining 29 slots padded with T(11).
    let expected = tokenize_32mer(&[0u8, 1u8, 2u8], 3);
    assert_eq!(key, expected);
    // Verify it doesn't panic (the old code would index out of bounds if len >
    // bases.len() were not clamped).
    let _ = tokenize_32mer(&bases[..3], 32);
}
