// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI test: `prmi_smem_range_packed` must return the same (k, l, s) as
//! `prmi_smem_range` for the same query and pac, only differing in whether
//! the pac is passed as 1 byte-per-base or 2-bit packed.

use prmi::train::build_sidecar;
use prmi_sys::{prmi_close, prmi_open, prmi_smem_range, prmi_smem_range_packed};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

/// Pack a slice of unpacked bases (0..=3, one per byte) into the BWA-MEME
/// bntpac 2-bit convention: base 0 in bits 6-7, base 1 in bits 4-5,
/// base 2 in bits 2-3, base 3 in bits 0-1.
fn pack_bases(bases: &[u8]) -> Vec<u8> {
    let n = bases.len();
    let mut out = vec![0u8; n.div_ceil(4)];
    for (i, &b) in bases.iter().enumerate() {
        let shift = 6 - 2 * ((i % 4) as u32);
        out[i / 4] |= (b & 0x3) << shift;
    }
    out
}

#[test]
fn smem_range_packed_matches_unpacked() {
    // Build a small sidecar from ACGT×64 = 256-base reference.
    let dir = tempdir().unwrap();
    let fa = dir.path().join("p.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("p.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Build the unpacked pac: ACGT×64 = 256 bytes (1 base/byte).
    let pac_unpacked: Vec<u8> = (0u64..256).map(|i| (i % 4) as u8).collect();
    // Build the packed pac: same 256 bases at 4 bases/byte = 64 bytes.
    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = 256;

    // Query: 32 bases starting at offset 10 of the pac.
    let query: Vec<u8> = pac_unpacked[10..42].to_vec();

    let mut k_u = 0u64;
    let mut l_u = 0u64;
    let mut s_u = 0u64;
    let rc_unpacked = unsafe {
        prmi_smem_range(
            handle,
            query.as_ptr(),
            query.len() as i32,
            pac_unpacked.as_ptr(),
            pac_unpacked.len(),
            &mut k_u,
            &mut l_u,
            &mut s_u,
        )
    };

    let mut k_p = 0u64;
    let mut l_p = 0u64;
    let mut s_p = 0u64;
    let rc_packed = unsafe {
        prmi_smem_range_packed(
            handle,
            query.as_ptr(),
            query.len() as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            &mut k_p,
            &mut l_p,
            &mut s_p,
        )
    };

    // Both calls must succeed with a match.
    assert_eq!(rc_unpacked, 0, "unpacked call returned error");
    assert_eq!(rc_packed, 0, "packed call returned error");
    assert!(l_u > 0, "unpacked returned no match");
    assert!(l_p > 0, "packed returned no match");

    // Results must be identical.
    assert_eq!(
        (k_u, l_u, s_u),
        (k_p, l_p, s_p),
        "packed and unpacked smem_range returned different results: \
        unpacked=({k_u},{l_u},{s_u}) packed=({k_p},{l_p},{s_p})"
    );

    unsafe {
        prmi_close(handle);
    }
}

#[test]
fn smem_range_packed_no_match_returns_one() {
    // Build sidecar from AAAA×64.
    let dir = tempdir().unwrap();
    let fa = dir.path().join("a.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"AAAA");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("a.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Pac: all A (0×256 bytes).
    let pac_unpacked: Vec<u8> = vec![0u8; 256];
    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = 256;

    // Query full of T (3) — should not appear in an all-A reference.
    let query: Vec<u8> = vec![3u8; 32];

    let mut k = 0u64;
    let mut l = 0u64;
    let mut s = 0u64;
    let rc = unsafe {
        prmi_smem_range_packed(
            handle,
            query.as_ptr(),
            query.len() as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            &mut k,
            &mut l,
            &mut s,
        )
    };
    // rc == 1 means no match; l must be 0.
    assert_eq!(rc, 1, "expected no-match return code 1, got {rc}");
    assert_eq!(l, 0);

    unsafe {
        prmi_close(handle);
    }
}

#[test]
fn smem_range_packed_null_handle_returns_minus_one() {
    let query = [0u8; 32];
    let pac_packed = [0u8; 8]; // dummy
    let mut k = 0u64;
    let mut l = 0u64;
    let mut s = 0u64;
    let rc = unsafe {
        prmi_smem_range_packed(
            ptr::null(),
            query.as_ptr(),
            query.len() as i32,
            pac_packed.as_ptr(),
            32,
            &mut k,
            &mut l,
            &mut s,
        )
    };
    assert_eq!(rc, -1);
}
