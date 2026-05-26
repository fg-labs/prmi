// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_format_version, prmi_max_error_bound, prmi_open, prmi_sa_num};
use std::ffi::{CStr, CString};
use std::ptr;
use tempfile::tempdir;

#[test]
fn introspection_returns_sane_values() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("i.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("i.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    assert!(unsafe { prmi_sa_num(handle) } > 0);
    let _ = unsafe { prmi_max_error_bound(handle) }; // any value OK
    let ver_ptr = unsafe { prmi_format_version(handle) };
    assert!(!ver_ptr.is_null());
    let ver = unsafe { CStr::from_ptr(ver_ptr) }.to_str().unwrap();
    assert_eq!(ver, "PRMIv1");

    unsafe {
        prmi_close(handle);
    }
}

#[test]
fn introspection_with_null_returns_defaults() {
    assert_eq!(unsafe { prmi_sa_num(ptr::null()) }, 0);
    assert_eq!(unsafe { prmi_max_error_bound(ptr::null()) }, 0);
    let ver_ptr = unsafe { prmi_format_version(ptr::null()) };
    assert!(!ver_ptr.is_null());
}
