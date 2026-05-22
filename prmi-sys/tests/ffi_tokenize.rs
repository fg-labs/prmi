// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::encoding::tokenize_32mer;
use prmi_sys::prmi_tokenize_32mer;
use std::ptr;

/// Encode a 32-mer via the FFI and compare with the pure-Rust result.
#[test]
fn ffi_matches_rust_full_32mer() {
    // 32 bases, one of each 2-bit value cycling through 0..=3.
    let bases: Vec<u8> = (0u8..32).map(|i| i % 4).collect();
    let expected = tokenize_32mer(&bases, 32);

    let mut out: u64 = 0;
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), 32, &mut out) };
    assert_eq!(rc, 0, "expected rc=0 for a valid call");
    assert_eq!(out, expected, "FFI key should match pure-Rust key");
}

/// Short query (len < 32) should be T-padded, matching Rust.
#[test]
fn ffi_matches_rust_short_query() {
    let bases: Vec<u8> = vec![0, 1, 2, 3]; // ACGT
    let expected = tokenize_32mer(&bases, 4);

    let mut out: u64 = 0;
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), 4, &mut out) };
    assert_eq!(rc, 0);
    assert_eq!(out, expected);
}

/// len=0 should produce the all-T-pad key.
#[test]
fn ffi_zero_len_gives_all_t_pad() {
    let bases: Vec<u8> = vec![0u8; 1]; // dummy, not read
    let expected = tokenize_32mer(&[], 0);

    let mut out: u64 = 0;
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), 0, &mut out) };
    assert_eq!(rc, 0);
    assert_eq!(out, expected);
}

/// len > 32 should be clamped to 32.
#[test]
fn ffi_len_over_32_is_clamped() {
    let bases: Vec<u8> = (0u8..64).map(|i| i % 4).collect();
    // Pure-Rust uses the first 32 bases too.
    let expected = tokenize_32mer(&bases[..32], 32);

    let mut out: u64 = 0;
    // Pass len=64 — should clamp to 32 internally.
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), 64, &mut out) };
    assert_eq!(rc, 0);
    assert_eq!(out, expected);
}

/// Null `bases` pointer should return -1.
#[test]
fn ffi_null_bases_returns_error() {
    let mut out: u64 = 0;
    let rc = unsafe { prmi_tokenize_32mer(ptr::null(), 4, &mut out) };
    assert_eq!(rc, -1);
}

/// Null `out_key` pointer should return -1.
#[test]
fn ffi_null_out_key_returns_error() {
    let bases: Vec<u8> = vec![0u8; 4];
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), 4, ptr::null_mut()) };
    assert_eq!(rc, -1);
}

/// Negative `len` should return -1.
#[test]
fn ffi_negative_len_returns_error() {
    let bases: Vec<u8> = vec![0u8; 4];
    let mut out: u64 = 0;
    let rc = unsafe { prmi_tokenize_32mer(bases.as_ptr(), -1, &mut out) };
    assert_eq!(rc, -1);
}
