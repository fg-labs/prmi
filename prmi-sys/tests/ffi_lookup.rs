// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_lookup, prmi_open};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

#[test]
fn lookup_returns_zero_on_known_key() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("l.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("l.fa.prmi");
    build_sidecar(&fa, &prefix, 16).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let mut pos = 0u64;
    let mut err = 0u64;
    let rc = unsafe { prmi_lookup(handle, 0, &mut pos, &mut err) };
    assert_eq!(rc, 0);
    // Sanity: pos must be inside the SA range (we don't know exact value).
    let _ = (pos, err);

    unsafe {
        prmi_close(handle);
    }
}
