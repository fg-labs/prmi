// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI tests for the in-situ SA-probe counter: `prmi_probe_count_enabled`,
//! `prmi_probe_count_reset`, `prmi_probe_count_get`. The counter is the
//! consumer's in-situ confirmation that the `est_hint` (`prmi_isa_at`) launch
//! reduces per-call probes on real reseed calls.
//!
//! The reduction assertion only runs with the `spectrum-probe-count` feature:
//!   cargo test -p prmi-sys --features spectrum-probe-count

use prmi_sys::{prmi_probe_count_enabled, prmi_probe_count_get, prmi_probe_count_reset};

/// The counter API is always callable and ABI-stable regardless of the feature:
/// `enabled` reports 0/1, `reset`/`get` never crash, and without the feature
/// `get` reads 0.
#[test]
fn probe_count_api_is_always_callable() {
    let enabled = prmi_probe_count_enabled();
    assert!(
        enabled == 0 || enabled == 1,
        "enabled must be 0 or 1, got {enabled}"
    );
    prmi_probe_count_reset();
    let n = prmi_probe_count_get();
    if enabled == 0 {
        assert_eq!(
            n, 0,
            "probe count must read 0 when counting is compiled out"
        );
    }
}

/// With counting compiled in: the counter is thread-local and per-search (reset
/// → 0, a search increments it), and the `est_hint` hinted backward launch costs
/// no more probes than the cold model launch — the in-situ reduction the
/// consumer's `--seeding-only` runs measure.
#[cfg(feature = "spectrum-probe-count")]
#[test]
fn hinted_backward_costs_no_more_probes_than_cold() {
    use prmi::train::build_sidecar_with_config;
    use prmi::train::config::{MemoryMode, TrainerConfig};
    use prmi_sys::{
        prmi_close, prmi_isa_at, prmi_mem_search, prmi_mem_search_backward, prmi_open,
        PRMI_MEM_WANT_INTERVAL,
    };
    use std::ffi::CString;
    use std::ptr;

    assert_eq!(
        prmi_probe_count_enabled(),
        1,
        "feature build must report enabled"
    );

    // Build a small mode-2 sidecar with the `.isa` (reference = ACGT×128).
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
        .with_isa(true);
    build_sidecar_with_config(&fa, &prefix, Some(16), Default::default(), 1, Some(cfg)).unwrap();
    let cprefix = CString::new(prefix.to_str().unwrap()).unwrap();

    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // read == reference bases (ACGT repeating, 512 bases); right anchor at pivot.
    let read: Vec<u8> = (0u64..512).map(|i| (i % 4) as u8).collect();
    let pivot: i32 = 64;
    let pac = pack(&read);

    // reset → 0 sanity.
    prmi_probe_count_reset();
    assert_eq!(prmi_probe_count_get(), 0, "reset must zero the counter");

    // Right anchor for read[pivot..] (counts probes too — proves a search bumps it).
    let (mut a_ml, mut a_ss, mut a_occ) = (0u32, 0u64, 0u64);
    let rc = unsafe {
        prmi_mem_search(
            handle,
            read[pivot as usize..].as_ptr(),
            (read.len() - pivot as usize) as i32,
            pac.as_ptr(),
            read.len() as u64,
            0,
            PRMI_MEM_WANT_INTERVAL,
            &mut a_ml,
            &mut a_ss,
            &mut a_occ,
        )
    };
    assert_eq!(rc, 0);
    assert!(
        prmi_probe_count_get() > 0,
        "a forward search must bump the probe counter"
    );

    let mut hint: u64 = 0;
    assert_eq!(unsafe { prmi_isa_at(handle, pivot as u64, &mut hint) }, 0);

    let backward = |est_hint: u64, sa_start: u64, occ: u64| -> u64 {
        let (mut ml, mut ss, mut oc) = (0u32, 0u64, 0u64);
        prmi_probe_count_reset();
        let rc = unsafe {
            prmi_mem_search_backward(
                handle,
                sa_start,
                occ,
                u64::from(a_ml),
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
        assert_eq!(rc, 0);
        prmi_probe_count_get()
    };

    let cold_probes = backward(0, a_ss, a_occ);
    let hinted_probes = backward(hint, 0, 0);
    assert!(
        hinted_probes <= cold_probes,
        "hinted backward must not cost more probes than cold (hinted={hinted_probes}, cold={cold_probes})"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Pack unpacked bases (0..=3) into BWA-MEME bntpac 2-bit (MSB-first, 4/byte).
#[cfg(feature = "spectrum-probe-count")]
fn pack(bases: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bases.len().div_ceil(4)];
    for (i, &b) in bases.iter().enumerate() {
        out[i >> 2] |= (b & 3) << (6 - 2 * (i & 3));
    }
    out
}
