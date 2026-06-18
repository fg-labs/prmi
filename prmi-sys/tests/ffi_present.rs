// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI test for the tiered-dispatch presence pre-reject: `prmi_present`.

use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi_sys::{prmi_close, prmi_open, prmi_present};
use std::ffi::CString;
use std::ptr;

/// Pack unpacked bases (0..=3) into BWA-MEME bntpac 2-bit (MSB-first, 4/byte).
fn pack(bases: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bases.len().div_ceil(4)];
    for (i, &b) in bases.iter().enumerate() {
        out[i >> 2] |= (b & 3) << (6 - 2 * (i & 3));
    }
    out
}

/// Build a small mode-2 sidecar over `ACGT × 128` (512 bases) and return the
/// tempdir (kept alive for the mmap lifetime), the open-prefix `CString`, and the
/// packed reference pac + base count to pass to `prmi_present`.
fn fixture() -> (tempfile::TempDir, CString, Vec<u8>, u64) {
    let dir = tempfile::tempdir().unwrap();
    let fa = dir.path().join("ref.fa");
    let mut fa_bytes = b">ref\n".to_vec();
    let ref_bases: Vec<u8> = (0u64..512).map(|i| (i % 4) as u8).collect();
    for &b in &ref_bases {
        fa_bytes.push(b"ACGT"[b as usize]);
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("ref.fa.prmi");
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_with_config(&fa, &prefix, Some(16), Default::default(), 1, Some(cfg)).unwrap();
    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();
    (dir, cprefix, pack(&ref_bases), ref_bases.len() as u64)
}

#[test]
fn present_anchor_in_reference_returns_one() {
    let (dir, cprefix, pac, n) = fixture();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // A 40-base ACGT-repeat read: its leading 32-mer occurs in the ACGT×128 ref.
    let on: Vec<u8> = (0u64..40).map(|i| (i % 4) as u8).collect();
    assert_eq!(
        unsafe { prmi_present(handle, on.as_ptr(), on.len() as i32, pac.as_ptr(), n) },
        1,
        "in-ref → present"
    );

    // An all-A 40-mer: `AA` never occurs in an ACGT-repeat ref → absent.
    let off = [0u8; 40];
    assert_eq!(
        unsafe { prmi_present(handle, off.as_ptr(), off.len() as i32, pac.as_ptr(), n) },
        0,
        "absent → 0"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn present_short_read_or_all_n_is_absent() {
    let (dir, cprefix, pac, n) = fixture();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Reads with no full 32-mer window → no leading anchor → absent (0).
    let short: Vec<u8> = (0u64..31).map(|i| (i % 4) as u8).collect();
    assert_eq!(
        unsafe { prmi_present(handle, short.as_ptr(), short.len() as i32, pac.as_ptr(), n) },
        0,
        "len<32 → 0"
    );
    assert_eq!(
        unsafe { prmi_present(handle, ptr::null(), 0, pac.as_ptr(), n) },
        0,
        "empty → 0"
    );
    // All-N read: every window is skipped → absent (0).
    let alln = [4u8; 40];
    assert_eq!(
        unsafe { prmi_present(handle, alln.as_ptr(), alln.len() as i32, pac.as_ptr(), n) },
        0,
        "all-N → 0"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

#[test]
fn present_null_and_invalid_args() {
    let (dir, cprefix, pac, n) = fixture();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);
    let read = [0u8, 1, 2, 3];

    // null handle / null pac → -1.
    assert_eq!(
        unsafe { prmi_present(ptr::null(), read.as_ptr(), 4, pac.as_ptr(), n) },
        -1
    );
    assert_eq!(
        unsafe { prmi_present(handle as *const _, read.as_ptr(), 4, ptr::null(), n) },
        -1
    );
    // negative read_len → -2.
    assert_eq!(
        unsafe { prmi_present(handle as *const _, read.as_ptr(), -1, pac.as_ptr(), n) },
        -2
    );
    // mismatched pac_num_bases (≠ index l_pac) → -2; a smaller value keeps the
    // call memory-safe while pinning the ABI length contract.
    assert_eq!(
        unsafe { prmi_present(handle as *const _, read.as_ptr(), 4, pac.as_ptr(), n - 4) },
        -2
    );
    // null read with nonzero len → -1.
    assert_eq!(
        unsafe { prmi_present(handle as *const _, ptr::null(), 4, pac.as_ptr(), n) },
        -1
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}
