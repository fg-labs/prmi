// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Measure the per-call FFI wrapper overhead of `prmi_mem_search` — the cost the
//! consumer pays at the C->Rust boundary on EVERY seeding call (clear_last_error,
//! pointer validation, `catch_unwind`, out-ptr writes) that bwa-meme's native
//! inline seeding does not. A/B two tight loops over the SAME warm index + query:
//! the FFI path (`prmi_mem_search`) vs the direct Rust method (`idx.mem_search`).
//! Both do identical search work; the per-call delta IS the FFI overhead. Projected
//! across ~129 seeding calls/read, this is the candidate for the ~2.2x throughput
//! gap that probe-count does not explain.
//!
//! Run: `cargo run --release -p prmi-sys --example ffi_overhead`
//! Env: PRMI_HANDOFF (dir with chr22_A.fa.pac), PRMI_LEAVES/PRMI_FALLBACK (cache dir).

use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi_sys::{
    prmi_close, prmi_mem_search, prmi_mem_search_lean, prmi_open, PRMI_MEM_WANT_INTERVAL,
};
use std::ffi::CString;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn main() {
    // Local real-data benchmark — no machine-specific default path. Set
    // PRMI_HANDOFF to the directory holding chr22_A.fa.pac.
    let hoff = std::env::var("PRMI_HANDOFF").expect(
        "set PRMI_HANDOFF to a directory containing chr22_A.fa.pac \
         (ffi_overhead is a local real-data benchmark, not a portable example)",
    );
    let pac_path = format!("{hoff}/chr22_A.fa.pac");
    let leaves: u64 = std::env::var("PRMI_LEAVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_388_608);
    let fb_tag = std::env::var("PRMI_FALLBACK").unwrap_or_else(|_| "def".into());
    let prefix = format!("/tmp/prmi_chr22_replay_{leaves}_{fb_tag}/chr22.prmi");

    // Direct path: open the index via prmi's public API.
    let idx = LearnedIndex::open(Path::new(&prefix)).expect("open index (direct)");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(&pac_path).expect("read .pac");
    let enc = PacEncoding::Packed { num_bases: l_pac };

    // FFI path: open the SAME index behind an opaque handle.
    let cprefix = CString::new(prefix.clone()).unwrap();
    let mut handle = std::ptr::null_mut();
    let rc = unsafe { prmi_open(cprefix.as_ptr(), &mut handle) };
    assert_eq!(rc, 0, "prmi_open failed");

    // Representative queries: a length-1 reseed anchor, a mid maximal emit, a long
    // forward emit (digits 0..3 = A,C,G,T). Drawn so they actually match.
    let read: Vec<u8> = "GTGGAGATGGGATTTCACCATGTTGGCCAAGCTGGTCTCGAACACCTGACCTCAGGTGATCCACCCGCC"
        .bytes()
        .map(|b| match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 4,
        })
        .collect();
    let queries: Vec<&[u8]> = vec![&read[0..1], &read[0..20], &read[0..40], &read[10..60]];

    let iters: u64 = std::env::var("PRMI_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    println!(
        "[ffi_overhead] l_pac={l_pac} iters={iters} queries={}",
        queries.len()
    );

    for q in &queries {
        // Warm up both paths.
        for _ in 0..50_000 {
            black_box(idx.mem_search(q, &pac, enc));
        }
        // DIRECT: idx.mem_search
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            let m = idx.mem_search(black_box(q), &pac, enc);
            acc = acc.wrapping_add(m.match_len);
        }
        let t_direct = t0.elapsed().as_nanos() as f64 / iters as f64;
        black_box(acc);

        // FFI: prmi_mem_search (full C boundary).
        let (mut ml, mut sa, mut occ) = (0u32, 0u64, 0u64);
        let t1 = Instant::now();
        let mut acc2 = 0u64;
        // Accumulate the return code and assert AFTER the loop: release-visible
        // (unlike `debug_assert_eq!`, which compiles out in the documented
        // `--release` run) without putting a per-iteration assert in the timed
        // path. Any nonzero rc sticks.
        let mut rc_ok = 0i32;
        for _ in 0..iters {
            let rc = unsafe {
                prmi_mem_search(
                    handle,
                    q.as_ptr(),
                    q.len() as i32,
                    pac.as_ptr(),
                    l_pac,
                    0,
                    PRMI_MEM_WANT_INTERVAL,
                    &mut ml,
                    &mut sa,
                    &mut occ,
                )
            };
            rc_ok |= rc;
            acc2 = acc2.wrapping_add(ml as u64);
        }
        let t_ffi = t1.elapsed().as_nanos() as f64 / iters as f64;
        assert_eq!(rc_ok, 0, "prmi_mem_search returned nonzero in the FFI loop");
        black_box(acc2);

        // LEAN FFI: same boundary minus clear_last_error + catch_unwind.
        let t2 = Instant::now();
        let mut acc3 = 0u64;
        let mut rc_ok_lean = 0i32;
        for _ in 0..iters {
            let rc = unsafe {
                prmi_mem_search_lean(
                    handle,
                    q.as_ptr(),
                    q.len() as i32,
                    pac.as_ptr(),
                    l_pac,
                    0,
                    PRMI_MEM_WANT_INTERVAL,
                    &mut ml,
                    &mut sa,
                    &mut occ,
                )
            };
            rc_ok_lean |= rc;
            acc3 = acc3.wrapping_add(ml as u64);
        }
        let t_lean = t2.elapsed().as_nanos() as f64 / iters as f64;
        assert_eq!(
            rc_ok_lean, 0,
            "prmi_mem_search_lean returned nonzero in the FFI loop"
        );
        black_box(acc3);

        let overhead = t_ffi - t_direct;
        let guard = t_ffi - t_lean; // clear_last_error + catch_unwind cost
        println!(
            "len={:>2}: direct={:5.1} lean={:5.1} ffi={:5.1} ovh={:4.1} guard={:4.1} x129={:.2}us",
            q.len(),
            t_direct,
            t_lean,
            t_ffi,
            overhead,
            guard,
            overhead * 129.0 / 1000.0
        );
    }

    unsafe { prmi_close(handle) };
}
