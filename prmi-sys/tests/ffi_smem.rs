// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_open, prmi_smem_range};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

#[test]
fn smem_range_ffi_smoke() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("s.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("s.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let query: Vec<u8> = pac[10..42].to_vec();
    let mut k = 0u64;
    let mut l = 0u64;
    let mut s = 0u64;
    let rc = unsafe {
        prmi_smem_range(
            handle,
            query.as_ptr(),
            query.len() as i32,
            pac.as_ptr(),
            pac.len(),
            &mut k,
            &mut l,
            &mut s,
        )
    };
    assert_eq!(rc, 0);
    // Compare the full SA interval against the in-tree Rust reference so an
    // off-by-one in k/l or a swapped bound is caught (not just `l > 0`).
    let (ek, el, es) = prmi::index::LearnedIndex::open(&prefix)
        .unwrap()
        .smem_range(&query, &pac)
        .unwrap();
    assert_eq!(
        (k, l, s),
        (ek, el, es),
        "FFI interval must match Rust reference"
    );

    unsafe {
        prmi_close(handle);
    }
}

#[test]
fn smem_range_long_read_null_read_bases_zero_len_is_safe() {
    // The documented `read_bases == NULL` + `read_len == 0` contract must not
    // construct a slice from a null pointer (UB). With one pivot and an empty
    // read every pivot is out of range, so each result is (0, 0, 0).
    let dir = tempdir().unwrap();
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    let fa = dir.path().join("lr.fa");
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("lr.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let pivots = [0u64];
    let mut k = [9u64];
    let mut l = [9u64];
    let mut s = [9u64];
    let rc = unsafe {
        prmi_sys::prmi_smem_range_long_read(
            handle,
            ptr::null(), // read_bases == NULL
            0,           // read_len == 0
            pivots.as_ptr(),
            pivots.len() as u64,
            pac.as_ptr(),
            pac.len(),
            k.as_mut_ptr(),
            l.as_mut_ptr(),
            s.as_mut_ptr(),
        )
    };
    assert_eq!(
        rc, 0,
        "null read_bases with read_len 0 must succeed, got {rc}"
    );
    assert_eq!(
        (k[0], l[0], s[0]),
        (0, 0, 0),
        "out-of-range pivot yields empty match"
    );

    unsafe { prmi_close(handle) };
}
