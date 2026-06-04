// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::encoding::tokenize_32mer;
use prmi_sys::{prmi_reverse_complement_2bit, prmi_reverse_complement_key};
use std::ptr;

/// ACGT is a palindrome under reverse-complement.
#[test]
fn ffi_key_acgt_is_palindrome() {
    let bases: Vec<u8> = vec![0, 1, 2, 3]; // A C G T
    let key = tokenize_32mer(&bases, 4);
    let mut out: u64 = 0;
    let rc = unsafe { prmi_reverse_complement_key(key, 4, &mut out) };
    assert_eq!(rc, 0, "expected rc=0 for valid call");
    assert_eq!(out, key, "ACGT rev-comp should equal original (palindrome)");
}

/// Per the documented contract, `len` outside `1..=32` clamps to 32 — including
/// `len == 0`, which must behave the same as `len == 32` (not pass 0 through).
#[test]
fn ffi_key_len_zero_clamps_to_32() {
    let key = tokenize_32mer(&[0, 1, 2, 3], 4);
    let mut out_zero: u64 = 0;
    let mut out_32: u64 = 0;
    let rc0 = unsafe { prmi_reverse_complement_key(key, 0, &mut out_zero) };
    let rc32 = unsafe { prmi_reverse_complement_key(key, 32, &mut out_32) };
    assert_eq!(rc0, 0);
    assert_eq!(rc32, 0);
    assert_eq!(
        out_zero, out_32,
        "len == 0 must clamp to 32, matching len == 32"
    );
}

/// AAAA rev-comp is TTTT.
#[test]
fn ffi_key_aaaa_is_tttt() {
    let aaaa = tokenize_32mer(&[0, 0, 0, 0], 4);
    let tttt = tokenize_32mer(&[3, 3, 3, 3], 4);
    let mut out: u64 = 0;
    let rc = unsafe { prmi_reverse_complement_key(aaaa, 4, &mut out) };
    assert_eq!(rc, 0);
    assert_eq!(out, tttt);
}

/// Rev-comp of rev-comp is identity for a variety of lengths.
#[test]
fn ffi_key_double_revcomp_is_identity() {
    for len in 1u8..=32 {
        let bases: Vec<u8> = (0..len).map(|i| (i * 7 + 3) & 0x3).collect();
        let k = tokenize_32mer(&bases, len as usize);

        let mut rc: u64 = 0;
        let r1 = unsafe { prmi_reverse_complement_key(k, len as i32, &mut rc) };
        assert_eq!(r1, 0, "len={len}: first revcomp returned error");

        let mut rcrc: u64 = 0;
        let r2 = unsafe { prmi_reverse_complement_key(rc, len as i32, &mut rcrc) };
        assert_eq!(r2, 0, "len={len}: second revcomp returned error");

        assert_eq!(rcrc, k, "len={len}: double rev-comp should equal original");
    }
}

/// Null out_key pointer returns -1.
#[test]
fn ffi_key_null_out_key_returns_error() {
    let rc = unsafe { prmi_reverse_complement_key(0, 4, ptr::null_mut()) };
    assert_eq!(rc, -1);
}

/// Negative len is treated as len=32 (clamped, not an error).
#[test]
fn ffi_key_negative_len_is_clamped_to_32() {
    let bases: Vec<u8> = (0u8..32).map(|i| i % 4).collect();
    let k = tokenize_32mer(&bases, 32);
    let mut out_neg: u64 = 0;
    let mut out_32: u64 = 0;
    let r1 = unsafe { prmi_reverse_complement_key(k, -1, &mut out_neg) };
    let r2 = unsafe { prmi_reverse_complement_key(k, 32, &mut out_32) };
    assert_eq!(r1, 0);
    assert_eq!(r2, 0);
    assert_eq!(
        out_neg, out_32,
        "negative len should give same result as len=32"
    );
}

/// 2-bit array: ACGTA → TACGT.
#[test]
fn ffi_2bit_basic() {
    let bases = [0u8, 1, 2, 3, 0]; // ACGTA
    let mut out = [0u8; 5];
    let rc = unsafe { prmi_reverse_complement_2bit(bases.as_ptr(), 5, out.as_mut_ptr()) };
    assert_eq!(rc, 0);
    assert_eq!(out, [3u8, 0, 1, 2, 3]); // TACGT
}

/// Empty slice: len=0 succeeds and writes nothing.
#[test]
fn ffi_2bit_zero_len() {
    let bases = [0u8; 1]; // dummy, not accessed
    let mut out = [0u8; 1];
    let sentinel = out[0];
    let rc = unsafe { prmi_reverse_complement_2bit(bases.as_ptr(), 0, out.as_mut_ptr()) };
    assert_eq!(rc, 0);
    assert_eq!(out[0], sentinel, "zero-len call must not write to out");
}

/// Null in_ returns -1.
#[test]
fn ffi_2bit_null_in_returns_error() {
    let mut out = [0u8; 4];
    let rc = unsafe { prmi_reverse_complement_2bit(ptr::null(), 4, out.as_mut_ptr()) };
    assert_eq!(rc, -1);
}

/// Null out returns -1.
#[test]
fn ffi_2bit_null_out_returns_error() {
    let bases = [0u8; 4];
    let rc = unsafe { prmi_reverse_complement_2bit(bases.as_ptr(), 4, ptr::null_mut()) };
    assert_eq!(rc, -1);
}

/// Negative len returns -1.
#[test]
fn ffi_2bit_negative_len_returns_error() {
    let bases = [0u8; 4];
    let mut out = [0u8; 4];
    let rc = unsafe { prmi_reverse_complement_2bit(bases.as_ptr(), -1, out.as_mut_ptr()) };
    assert_eq!(rc, -1);
}
