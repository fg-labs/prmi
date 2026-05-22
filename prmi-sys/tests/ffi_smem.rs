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
    build_sidecar(&fa, &prefix, 16).unwrap();

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
    assert!(l > 0);
    assert_eq!(s, 32);

    unsafe {
        prmi_close(handle);
    }
}
