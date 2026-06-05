// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI tests for the inverse-SA launch hint: `prmi_has_isa` + `prmi_isa_at`.

use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi_sys::{
    prmi_close, prmi_has_isa, prmi_isa_at, prmi_mem_search, prmi_mem_search_backward, prmi_open,
    prmi_sa_num, PRMI_MEM_WANT_INTERVAL,
};
use std::ffi::CString;
use std::ptr;

/// Build a small mode-2 sidecar (optionally with the `.isa`) and return its
/// tempdir (kept alive for the mmap lifetime) plus the prefix `CString` so the
/// caller can `prmi_open` (the opaque handle type is not nameable in tests).
fn build_sidecar(with_isa: bool) -> (tempfile::TempDir, CString) {
    let dir = tempfile::tempdir().unwrap();
    let fa = dir.path().join("ref.fa");
    let mut fa_bytes = b">ref\n".to_vec();
    for i in 0u64..512 {
        fa_bytes.push(b"ACGT"[(i % 4) as usize]);
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("ref.fa.prmi");
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_isa(with_isa);
    build_sidecar_with_config(&fa, &prefix, Some(16), Default::default(), 1, Some(cfg)).unwrap();
    (dir, CString::new(prefix.to_str().unwrap()).unwrap())
}

#[test]
fn isa_present_lookups_succeed_and_invert_sa() {
    let (dir, cprefix) = build_sidecar(true);
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);
    assert_eq!(
        unsafe { prmi_has_isa(handle) },
        1,
        "build --with-isa → has_isa"
    );
    let sa_num = unsafe { prmi_sa_num(handle) } as u64;

    // isa_at over a sample of reference positions returns an SA index < sa_num,
    // and the values are distinct (it is a permutation inverse).
    let mut seen = std::collections::HashSet::new();
    for refpos in (0..sa_num).step_by(11) {
        let mut sa_index: u64 = u64::MAX;
        let rc = unsafe { prmi_isa_at(handle, refpos, &mut sa_index) };
        assert_eq!(rc, 0, "prmi_isa_at rc={rc} at refpos={refpos}");
        assert!(sa_index < sa_num, "sa_index {sa_index} out of range");
        assert!(
            seen.insert(sa_index),
            "isa_at must be injective (refpos={refpos})"
        );
    }

    // Out-of-range refpos → -2; the out-ptr is not required to be touched.
    let mut sink: u64 = 123;
    assert_eq!(unsafe { prmi_isa_at(handle, sa_num, &mut sink) }, -2);

    // Null handle / null out-ptr → -1.
    assert_eq!(unsafe { prmi_isa_at(ptr::null(), 0, &mut sink) }, -1);
    assert_eq!(unsafe { prmi_isa_at(handle, 0, ptr::null_mut()) }, -1);

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn no_isa_reports_absent_and_returns_minus_five() {
    let (dir, cprefix) = build_sidecar(false);
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);
    assert_eq!(unsafe { prmi_has_isa(handle) }, 0, "default build → no isa");
    let mut sa_index: u64 = 0;
    assert_eq!(
        unsafe { prmi_isa_at(handle, 0, &mut sa_index) },
        -5,
        "isa_at on a sidecar without .isa must return -5"
    );
    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn has_isa_null_handle_is_zero() {
    assert_eq!(unsafe { prmi_has_isa(ptr::null()) }, 0);
}

/// Backward `est_hint>0` (ISA no_search left extension) via the FFI is byte-
/// identical to `est_hint=0`, and rejects an out-of-range hint with -2. The read
/// is the reference's first 512 bases (ACGT×128), so the natural locus for the
/// anchor at `read[pivot..]` is `pivot` itself.
#[test]
fn backward_est_hint_ffi_equals_unhinted() {
    let (dir, cprefix) = build_sidecar(true);
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // read == reference bases (ACGT repeating, 512 bases).
    let read: Vec<u8> = (0u64..512).map(|i| (i % 4) as u8).collect();
    let pivot: i32 = 64;

    // Right anchor for read[pivot..] via the forward one-shot (est_hint=0).
    let (mut a_ml, mut a_ss, mut a_occ) = (0u32, 0u64, 0u64);
    let rc = unsafe {
        prmi_mem_search(
            handle,
            read[pivot as usize..].as_ptr(),
            (read.len() - pivot as usize) as i32,
            // The FFI expects packed pac (2-bit, 4 bases/byte); pack the reference.
            pack(&read).as_ptr(),
            read.len() as u64,
            0,
            PRMI_MEM_WANT_INTERVAL,
            &mut a_ml,
            &mut a_ss,
            &mut a_occ,
        )
    };
    assert_eq!(rc, 0);
    assert!(a_ml > 0);
    let anchor_len = a_ml as u64;

    // Hint = inverse SA at the anchor's natural reference position (= pivot).
    let mut hint: u64 = 0;
    assert_eq!(unsafe { prmi_isa_at(handle, pivot as u64, &mut hint) }, 0);

    let pac = pack(&read);
    let call = |est_hint: u64, sa_start: u64, occ: u64| -> (i32, u32, u64, u64) {
        let (mut ml, mut ss, mut oc) = (0u32, 0u64, 0u64);
        let rc = unsafe {
            prmi_mem_search_backward(
                handle,
                sa_start,
                occ,
                anchor_len,
                read.as_ptr(),
                read.len() as i32,
                pivot,
                pac.as_ptr(),
                read.len() as u64,
                est_hint,
                PRMI_MEM_WANT_INTERVAL,
                &mut ml,
                &mut ss,
                &mut oc,
            )
        };
        (rc, ml, ss, oc)
    };

    let unhinted = call(0, a_ss, a_occ);
    assert_eq!(unhinted.0, 0);
    let hinted = call(hint, 0, 0); // est_hint replaces the anchor interval
    assert_eq!(hinted.0, 0);
    assert_eq!(
        (hinted.1, hinted.2, hinted.3),
        (unhinted.1, unhinted.2, unhinted.3),
        "backward est_hint must be byte-identical to est_hint=0"
    );

    // Out-of-range hint → -2.
    let sa_num = unsafe { prmi_sa_num(handle) } as u64;
    assert_eq!(call(sa_num, 0, 0).0, -2);

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Pack unpacked bases (0..=3) into BWA-MEME bntpac 2-bit (MSB-first, 4/byte).
fn pack(bases: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bases.len().div_ceil(4)];
    for (i, &b) in bases.iter().enumerate() {
        out[i >> 2] |= (b & 3) << (6 - 2 * (i & 3));
    }
    out
}
