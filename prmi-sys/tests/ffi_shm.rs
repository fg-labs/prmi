// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI integration tests for `prmi_open_shm`.

use prmi::index::shm::write_shm_blob;
use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_last_error_message, prmi_lookup, prmi_open, prmi_open_shm};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

/// Build a small FASTA and sidecar; return the tmpdir and sidecar prefix.
fn build_small_sidecar() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("f.fa");
    let mut content = String::from(">c\n");
    for _ in 0..64 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("f.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    (dir, prefix)
}

// ── prmi_open_shm: basic open/close ──────────────────────────────────────────

#[test]
fn ffi_open_shm_basic() {
    let (_dir, prefix) = build_small_sidecar();
    let shm_path = _dir.path().join("basic.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();

    let c_shm_path = CString::new(shm_path.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open_shm(c_shm_path.as_ptr(), &mut handle) };

    assert_eq!(rc, 0, "prmi_open_shm should return 0 on success");
    assert!(!handle.is_null(), "handle should be non-null on success");

    unsafe { prmi_close(handle) };
}

// ── prmi_open_shm: lookup matches prmi_open ───────────────────────────────────

#[test]
fn ffi_open_shm_lookup_matches_file_open() {
    let (_dir, prefix) = build_small_sidecar();
    let shm_path = _dir.path().join("lookup.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();

    let c_prefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let c_shm_path = CString::new(shm_path.to_str().unwrap()).unwrap();

    let mut file_handle = ptr::null_mut();
    let rc = unsafe { prmi_open(c_prefix.as_ptr(), &mut file_handle) };
    assert_eq!(rc, 0, "prmi_open should succeed");

    let mut shm_handle = ptr::null_mut();
    let rc = unsafe { prmi_open_shm(c_shm_path.as_ptr(), &mut shm_handle) };
    assert_eq!(rc, 0, "prmi_open_shm should succeed");

    // Compare lookup results for several keys.
    for key in [0u64, 1, 0xdeadbeefcafe, u64::MAX] {
        let mut file_pos = 0u64;
        let mut file_err = 0u64;
        let mut shm_pos = 0u64;
        let mut shm_err = 0u64;

        let file_rc = unsafe { prmi_lookup(file_handle, key, &mut file_pos, &mut file_err) };
        let shm_rc = unsafe { prmi_lookup(shm_handle, key, &mut shm_pos, &mut shm_err) };

        assert_eq!(file_rc, 0, "file lookup({key}) failed");
        assert_eq!(shm_rc, 0, "shm lookup({key}) failed");
        assert_eq!(file_pos, shm_pos, "predicted pos mismatch for key={key}");
        assert_eq!(file_err, shm_err, "err bound mismatch for key={key}");
    }

    unsafe {
        prmi_close(file_handle);
        prmi_close(shm_handle);
    }
}

// ── prmi_open_shm: error paths ────────────────────────────────────────────────

#[test]
fn ffi_open_shm_missing_file_returns_negative() {
    let c_path = CString::new("/nonexistent/prmi.shm").unwrap();
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open_shm(c_path.as_ptr(), &mut handle) };
    assert_eq!(
        rc, -3,
        "prmi_open_shm on a missing file should return -3, got {rc}"
    );
    assert!(handle.is_null(), "handle should be null on error");

    let err_ptr = prmi_last_error_message();
    assert!(!err_ptr.is_null());
    let err = unsafe { std::ffi::CStr::from_ptr(err_ptr) };
    assert!(
        !err.to_bytes().is_empty(),
        "error message should be non-empty"
    );
}

#[test]
fn ffi_open_shm_null_path_returns_negative() {
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open_shm(ptr::null(), &mut handle) };
    assert_eq!(rc, -1, "null path should return -1");
    assert!(handle.is_null());
}

#[test]
fn ffi_open_shm_null_out_handle_returns_negative() {
    let c_path = CString::new("/tmp/prmi_test_dummy.shm").unwrap();
    let rc = unsafe { prmi_open_shm(c_path.as_ptr(), ptr::null_mut()) };
    assert_eq!(rc, -1, "null out_handle should return -1");
}
