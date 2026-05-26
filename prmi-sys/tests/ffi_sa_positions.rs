// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI test: `prmi_sa_positions` must resolve SA index ranges to genome
//! positions and match the Rust singleton accessor for each slot.

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_open, prmi_sa_positions};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

fn build_test_sidecar() -> (tempfile::TempDir, String) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("ffi.fa");
    // ACGT × 64 = 256-base deterministic reference.
    let mut fa_bytes = b">ffi\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("ffi.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let prefix_str = prefix.to_str().unwrap().to_owned();
    (dir, prefix_str)
}

#[test]
fn ffi_sa_positions_matches_rust_singleton() {
    let (dir, prefix_str) = build_test_sidecar();
    let cprefix = CString::new(prefix_str.clone()).unwrap();

    // Open via FFI.
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Also open via Rust to get the reference values.
    let idx = LearnedIndex::open(std::path::Path::new(&prefix_str)).unwrap();
    let sa_num = idx.sa_num();

    // Read all positions via FFI.
    let mut ffi_out = vec![0u64; sa_num as usize];
    let rc = unsafe { prmi_sa_positions(handle, 0, sa_num, ffi_out.as_mut_ptr()) };
    assert_eq!(rc, 0, "prmi_sa_positions failed with rc={rc}");

    // Compare against singleton accessor.
    for i in 0..sa_num {
        assert_eq!(
            ffi_out[i as usize],
            idx.sa_position_for(i),
            "mismatch at i={i}"
        );
    }

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn ffi_sa_positions_count_zero_ok() {
    let (dir, prefix_str) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // count == 0 → returns 0 with no writes regardless of k.
    let rc = unsafe { prmi_sa_positions(handle, 0, 0, ptr::null_mut()) };
    assert_eq!(rc, 0, "count=0 should return 0, got {rc}");

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn ffi_sa_positions_out_of_range_returns_negative() {
    let (dir, prefix_str) = build_test_sidecar();
    let cprefix = CString::new(prefix_str.clone()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let idx = LearnedIndex::open(std::path::Path::new(&prefix_str)).unwrap();
    let sa_num = idx.sa_num();

    // k + count > sa_num → should return negative.
    let mut buf = vec![0u64; 10];
    let rc = unsafe { prmi_sa_positions(handle, sa_num - 2, 10, buf.as_mut_ptr()) };
    assert!(rc < 0, "expected negative rc for out-of-range, got {rc}");

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn ffi_sa_positions_null_handle_returns_minus_one() {
    let mut buf = [0u64; 4];
    let rc = unsafe { prmi_sa_positions(ptr::null(), 0, 4, buf.as_mut_ptr()) };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

#[test]
fn ffi_sa_positions_null_out_with_count_returns_minus_two() {
    let (dir, prefix_str) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let rc = unsafe { prmi_sa_positions(handle, 0, 4, ptr::null_mut()) };
    assert_eq!(
        rc, -2,
        "null out_positions with count>0 should return -2, got {rc}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}
