// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_last_error_message, prmi_open};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

#[test]
fn open_close_smoke() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("f.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("f.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open(cprefix.as_ptr(), &mut handle) };
    assert_eq!(rc, 0, "prmi_open failed");
    assert!(!handle.is_null());
    unsafe {
        prmi_close(handle);
    }
}

#[test]
fn open_missing_file_returns_negative_and_error() {
    let cprefix = CString::new("/nonexistent/path/to/sidecar").unwrap();
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open(cprefix.as_ptr(), &mut handle) };
    assert!(rc < 0, "expected negative rc, got {rc}");
    assert!(handle.is_null());
    let err_ptr = prmi_last_error_message();
    assert!(!err_ptr.is_null());
    let err = unsafe { std::ffi::CStr::from_ptr(err_ptr) };
    assert!(
        !err.to_bytes().is_empty(),
        "expected non-empty error message"
    );
}

#[test]
fn open_null_args_safely_returns_error() {
    let mut handle = ptr::null_mut();
    let rc = unsafe { prmi_open(ptr::null(), &mut handle) };
    assert_eq!(rc, -1);
    assert!(handle.is_null());
}
