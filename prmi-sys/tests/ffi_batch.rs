// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI tests for `prmi_smem_range_batch` and `prmi_smem_range_batch_packed`.

use prmi::train::build_sidecar;
use prmi_sys::{
    prmi_close, prmi_open, prmi_smem_range, prmi_smem_range_batch, prmi_smem_range_batch_packed,
};
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

// ---------------------------------------------------------------------------
// Test 1: batch results match single-key prmi_smem_range for 100 queries
// ---------------------------------------------------------------------------
#[test]
fn batch_matches_single_key_100_queries() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("b.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("b.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // 100 queries: step-2 offsets, 32 bases each. pac is ACGT×64 = 256 bytes.
    // Last offset: 99*2 = 198; 198+32 = 230 ≤ 256.
    let pac: Vec<u8> = (0u64..256).map(|i| (i % 4) as u8).collect();
    const N: usize = 100;
    let mut flat_queries = vec![0u8; N * 32];
    let mut expected_k = vec![0u64; N];
    let mut expected_l = vec![0u64; N];
    let mut expected_s = vec![0u64; N];

    for i in 0..N {
        let offset = i * 2;
        let q = &pac[offset..offset + 32];
        flat_queries[i * 32..(i + 1) * 32].copy_from_slice(q);

        let mut k = 0u64;
        let mut l = 0u64;
        let mut s = 0u64;
        let rc = unsafe {
            prmi_smem_range(
                handle,
                q.as_ptr(),
                32,
                pac.as_ptr(),
                pac.len(),
                &mut k,
                &mut l,
                &mut s,
            )
        };
        assert!(rc >= 0, "prmi_smem_range failed at query {i}: rc={rc}");
        expected_k[i] = k;
        expected_l[i] = l;
        expected_s[i] = s;
    }

    // Batch call.
    let mut out_k = vec![0u64; N];
    let mut out_l = vec![0u64; N];
    let mut out_s = vec![0u64; N];
    let rc = unsafe {
        prmi_smem_range_batch(
            handle,
            flat_queries.as_ptr(),
            N as u64,
            pac.as_ptr(),
            pac.len(),
            out_k.as_mut_ptr(),
            out_l.as_mut_ptr(),
            out_s.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "prmi_smem_range_batch returned error: rc={rc}");

    for i in 0..N {
        assert!(out_l[i] > 0, "batch query {i} returned no match");
        assert_eq!(
            (out_k[i], out_l[i], out_s[i]),
            (expected_k[i], expected_l[i], expected_s[i]),
            "batch result mismatch at query {i}"
        );
    }

    unsafe { prmi_close(handle) };
}

// ---------------------------------------------------------------------------
// Test 2: empty batch (count = 0) returns 0 with no writes
// ---------------------------------------------------------------------------
#[test]
fn batch_empty_count_returns_zero() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("e.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("e.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac: Vec<u8> = (0u64..256).map(|i| (i % 4) as u8).collect();
    let pac_packed = pack_bases(&pac);

    // Sentinel values: writes would change them.
    let mut out_k = [0xDEAD_BEEF_u64; 1];
    let mut out_l = [0xDEAD_BEEF_u64; 1];
    let mut out_s = [0xDEAD_BEEF_u64; 1];

    let rc = unsafe {
        prmi_smem_range_batch(
            handle,
            ptr::null(), // ignored when count == 0
            0,
            pac.as_ptr(),
            pac.len(),
            out_k.as_mut_ptr(),
            out_l.as_mut_ptr(),
            out_s.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "empty unpacked batch returned {rc}");
    assert_eq!(out_k[0], 0xDEAD_BEEF_u64, "sentinel was overwritten");

    let rc2 = unsafe {
        prmi_smem_range_batch_packed(
            handle,
            ptr::null(),
            0,
            pac_packed.as_ptr(),
            256,
            out_k.as_mut_ptr(),
            out_l.as_mut_ptr(),
            out_s.as_mut_ptr(),
        )
    };
    assert_eq!(rc2, 0, "empty packed batch returned {rc2}");
    assert_eq!(out_k[0], 0xDEAD_BEEF_u64, "sentinel was overwritten");

    unsafe { prmi_close(handle) };
}

// ---------------------------------------------------------------------------
// Test 3: NULL pointer paths return negative and set last-error
// ---------------------------------------------------------------------------
#[test]
fn batch_null_handle_returns_negative() {
    let query = [0u8; 32];
    let pac = [0u8; 64];
    let mut k = 0u64;
    let mut l = 0u64;
    let mut s = 0u64;

    let rc = unsafe {
        prmi_smem_range_batch(
            ptr::null(),
            query.as_ptr(),
            1,
            pac.as_ptr(),
            pac.len(),
            &mut k,
            &mut l,
            &mut s,
        )
    };
    assert_eq!(rc, -1, "null-pointer argument must return -1, got {rc}");

    let msg = unsafe { std::ffi::CStr::from_ptr(prmi_sys::prmi_last_error_message()) };
    assert!(
        !msg.to_bytes().is_empty(),
        "expected non-empty error message"
    );
}

#[test]
fn batch_packed_null_handle_returns_negative() {
    let query = [0u8; 32];
    let pac = [0u8; 64];
    let mut k = 0u64;
    let mut l = 0u64;
    let mut s = 0u64;

    let rc = unsafe {
        prmi_smem_range_batch_packed(
            ptr::null(),
            query.as_ptr(),
            1,
            pac.as_ptr(),
            256,
            &mut k,
            &mut l,
            &mut s,
        )
    };
    assert_eq!(rc, -1, "null-pointer argument must return -1, got {rc}");

    let msg = unsafe { std::ffi::CStr::from_ptr(prmi_sys::prmi_last_error_message()) };
    assert!(
        !msg.to_bytes().is_empty(),
        "expected non-empty error message"
    );
}

// ---------------------------------------------------------------------------
// Test 4: packed-pac batch parity with unpacked-pac batch
// ---------------------------------------------------------------------------
#[test]
fn batch_packed_parity_with_unpacked() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("p.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..64 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("p.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_unpacked: Vec<u8> = (0u64..256).map(|i| (i % 4) as u8).collect();
    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = 256;

    // 16 queries at step-12 offsets; last: 15*12=180, 180+32=212 ≤ 256.
    const N: usize = 16;
    let mut flat_queries = vec![0u8; N * 32];
    for i in 0..N {
        let offset = i * 12;
        flat_queries[i * 32..(i + 1) * 32].copy_from_slice(&pac_unpacked[offset..offset + 32]);
    }

    let mut out_k_u = vec![0u64; N];
    let mut out_l_u = vec![0u64; N];
    let mut out_s_u = vec![0u64; N];
    let rc_u = unsafe {
        prmi_smem_range_batch(
            handle,
            flat_queries.as_ptr(),
            N as u64,
            pac_unpacked.as_ptr(),
            pac_unpacked.len(),
            out_k_u.as_mut_ptr(),
            out_l_u.as_mut_ptr(),
            out_s_u.as_mut_ptr(),
        )
    };
    assert_eq!(rc_u, 0, "unpacked batch failed: rc={rc_u}");

    let mut out_k_p = vec![0u64; N];
    let mut out_l_p = vec![0u64; N];
    let mut out_s_p = vec![0u64; N];
    let rc_p = unsafe {
        prmi_smem_range_batch_packed(
            handle,
            flat_queries.as_ptr(),
            N as u64,
            pac_packed.as_ptr(),
            pac_num_bases,
            out_k_p.as_mut_ptr(),
            out_l_p.as_mut_ptr(),
            out_s_p.as_mut_ptr(),
        )
    };
    assert_eq!(rc_p, 0, "packed batch failed: rc={rc_p}");

    for i in 0..N {
        assert_eq!(
            (out_k_u[i], out_l_u[i], out_s_u[i]),
            (out_k_p[i], out_l_p[i], out_s_p[i]),
            "packed/unpacked batch mismatch at query {i}"
        );
    }

    unsafe { prmi_close(handle) };
}
