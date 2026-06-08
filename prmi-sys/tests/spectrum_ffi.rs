// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FFI tests for `prmi_forward_spectrum`, `prmi_backward_spectrum`,
//! `prmi_smem_step_t`, `prmi_sa_positions_strided`, and the batch variants.

use prmi::train::build_sidecar;
use prmi_sys::{
    prmi_backward_spectrum, prmi_backward_spectrum_batch, prmi_bwd_task_t, prmi_close,
    prmi_forward_spectrum, prmi_forward_spectrum_batch, prmi_fwd_task_t, prmi_open,
    prmi_sa_positions, prmi_sa_positions_strided, prmi_smem_step_t,
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

/// Build a deterministic 256-base ACGT-repeat sidecar and return both the
/// tempdir (keeps the files alive) and the prefix string.
fn build_test_sidecar() -> (tempfile::TempDir, String, Vec<u8>) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("ref.fa");
    // ACGT × 64 = 256-base reference (unpacked: 0,1,2,3 repeating).
    let pac_unpacked: Vec<u8> = (0u64..256).map(|i| (i % 4) as u8).collect();
    let mut fa_bytes = b">ref\n".to_vec();
    for &b in &pac_unpacked {
        // ASCII: A=65, C=67, G=71, T=84
        fa_bytes.push(b"ACGT"[b as usize]);
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("ref.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let prefix_str = prefix.to_str().unwrap().to_owned();
    (dir, prefix_str, pac_unpacked)
}

/// Test that `prmi_forward_spectrum` returns 0 with at least one step and
/// that the deepest step has `match_len == query_len` for a query lifted
/// directly from the reference.
#[test]
fn forward_spectrum_query_from_reference_matches_fully() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Lift 32 bases from offset 8 of the forward reference.
    let query_len: usize = 32;
    let query: Vec<u8> = pac_unpacked[8..8 + query_len].to_vec();
    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;

    let mut out_steps: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        query_len
    ];
    let mut nsteps: u64 = 0;

    let rc = unsafe {
        prmi_forward_spectrum(
            handle,
            query.as_ptr(),
            query_len as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            out_steps.as_mut_ptr(),
            out_steps.len() as u64,
            &mut nsteps,
        )
    };

    assert_eq!(rc, 0, "prmi_forward_spectrum failed with rc={rc}");
    assert!(
        nsteps >= 1,
        "expected at least one step, got nsteps={nsteps}"
    );

    // The deepest (last) step must have match_len == query_len because the
    // query was lifted verbatim from the reference text.
    let deepest = &out_steps[(nsteps - 1) as usize];
    assert_eq!(
        deepest.match_len, query_len as u64,
        "deepest step match_len should equal query_len={query_len}, got {}",
        deepest.match_len
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test the -4 path: when max_steps == 0 but the query produces at least one
/// step, the function must return -4 and set *out_nsteps to the needed count.
#[test]
fn forward_spectrum_buffer_too_small_returns_minus_four() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let query_len: usize = 32;
    let query: Vec<u8> = pac_unpacked[8..8 + query_len].to_vec();
    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;

    // Provide a 1-element buffer but tell the function max_steps == 0.
    let mut dummy_step = prmi_smem_step_t {
        sa_start: 0,
        occ_count: 0,
        match_len: 0,
    };
    let mut nsteps: u64 = 0;

    let rc = unsafe {
        prmi_forward_spectrum(
            handle,
            query.as_ptr(),
            query_len as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            &mut dummy_step as *mut prmi_smem_step_t,
            0, // max_steps = 0 → too small
            &mut nsteps,
        )
    };

    assert_eq!(rc, -4, "expected -4 (buffer too small), got {rc}");
    assert!(
        nsteps > 0,
        "out_nsteps must be set to the needed count (>0), got {nsteps}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test that a null handle returns -1.
#[test]
fn forward_spectrum_null_handle_returns_minus_one() {
    let pac_packed = [0u8; 8];
    let query = [0u8; 32];
    let mut dummy_step = prmi_smem_step_t {
        sa_start: 0,
        occ_count: 0,
        match_len: 0,
    };
    let mut nsteps: u64 = 0;

    let rc = unsafe {
        prmi_forward_spectrum(
            ptr::null(),
            query.as_ptr(),
            query.len() as i32,
            pac_packed.as_ptr(),
            32,
            &mut dummy_step as *mut prmi_smem_step_t,
            1,
            &mut nsteps,
        )
    };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

/// Test that `prmi_backward_spectrum` left-extends a right-anchored interval.
///
/// Construction:
/// - Reference: ACGT×64 (256 bases, unpacked = 0,1,2,3 repeating).
/// - Anchor query: bases at offset 20 (`pac_unpacked[20..20+anchor_len]`).
///   Offset 20 % 4 = 0, so the anchor starts with base A(=0); the preceding
///   base at offset 19 is T(=3).  Because the ACGT pattern repeats every 4
///   bases, every 4th position in the forward reference text has a T preceding
///   it, so backward extension by one base (T) is always possible.
/// - `read` = `pac_unpacked[19..19+read_len]` (surrounding context including
///   the left base); `pivot = 1` (anchor starts at `read[1..]`).
/// - `forward_spectrum` on `read[pivot..]` (= `pac_unpacked[20..]`) gives us
///   the anchor SA interval; we feed its shallowest step into
///   `prmi_backward_spectrum` and assert rc==0 with nsteps>=1 and every step's
///   `match_len > anchor_len`.
#[test]
fn backward_spectrum_extends_left() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;

    // Anchor: bases from offset 20 (multiple of 4; A,C,G,T,A,C,...).
    // Preceding base at offset 19 = T = 3 (pattern repeats so T always precedes A).
    // Build the surrounding read: [offset 19 .. offset 19+read_len).
    let left_offset = 19usize;
    let anchor_offset = 20usize; // left_offset + 1
    let pivot: i32 = (anchor_offset - left_offset) as i32; // = 1
    let anchor_query_len = 20usize;
    // read = 1 left base + anchor_query_len right bases = 21 bytes
    let read_len = pivot as usize + anchor_query_len;
    let read: Vec<u8> = pac_unpacked[left_offset..left_offset + read_len].to_vec();

    // Run forward_spectrum on the anchor half of the read (read[pivot..]).
    let anchor_query: Vec<u8> = read[pivot as usize..].to_vec();
    let max_fwd: u64 = anchor_query.len() as u64;
    let mut fwd_steps: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        anchor_query.len()
    ];
    let mut fwd_nsteps: u64 = 0;
    let rc_fwd = unsafe {
        prmi_forward_spectrum(
            handle,
            anchor_query.as_ptr(),
            anchor_query.len() as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            fwd_steps.as_mut_ptr(),
            max_fwd,
            &mut fwd_nsteps,
        )
    };
    assert_eq!(rc_fwd, 0, "prmi_forward_spectrum failed (rc={rc_fwd})");
    assert!(
        fwd_nsteps >= 1,
        "need at least one forward step to anchor backward search"
    );

    // Use the shallowest (first) forward step as the anchor.
    let anchor = fwd_steps[0];
    let anchor_len = anchor.match_len;

    // Run backward_spectrum; capacity = pivot+1 is always sufficient.
    let max_bwd = pivot as u64 + 1;
    let mut bwd_steps: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        max_bwd as usize
    ];
    let mut bwd_nsteps: u64 = 0;
    let rc_bwd = unsafe {
        prmi_backward_spectrum(
            handle,
            anchor.sa_start,
            anchor.occ_count,
            anchor_len,
            read.as_ptr(),
            read_len as i32,
            pivot,
            pac_packed.as_ptr(),
            pac_num_bases,
            bwd_steps.as_mut_ptr(),
            max_bwd,
            &mut bwd_nsteps,
        )
    };
    assert_eq!(rc_bwd, 0, "prmi_backward_spectrum returned rc={rc_bwd}");
    // The reference has T at every position % 4 == 3, so extending left by one
    // (prepending T at read[pivot-1] = pac_unpacked[19] = 3) must succeed.
    assert!(
        bwd_nsteps >= 1,
        "expected at least one backward step (T precedes every anchor start at offset 20)"
    );
    for (i, step) in bwd_steps.iter().enumerate().take(bwd_nsteps as usize) {
        assert!(
            step.match_len > anchor_len,
            "backward step {i} match_len={} must exceed anchor_len={anchor_len}",
            step.match_len
        );
    }

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test that a null handle returns -1 for `prmi_backward_spectrum`.
#[test]
fn backward_spectrum_null_handle_returns_minus_one() {
    let pac_packed = [0u8; 8];
    let read = [0u8; 4];
    let mut dummy_step = prmi_smem_step_t {
        sa_start: 0,
        occ_count: 0,
        match_len: 0,
    };
    let mut nsteps: u64 = 0;

    let rc = unsafe {
        prmi_backward_spectrum(
            ptr::null(),
            /*sa_start=*/ 0,
            /*occ_count=*/ 1,
            /*anchor_len=*/ 1,
            read.as_ptr(),
            read.len() as i32,
            /*pivot=*/ 1,
            pac_packed.as_ptr(),
            /*pac_num_bases=*/ 32,
            &mut dummy_step as *mut prmi_smem_step_t,
            /*max_steps=*/ 1,
            &mut nsteps,
        )
    };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

/// Test that strided reads match the corresponding entries from a contiguous
/// read of the same range via `prmi_sa_positions`.
///
/// Reads k=2, step=3, n_out=4 via `prmi_sa_positions_strided`, then reads
/// a contiguous block [2, 2+4*3) via `prmi_sa_positions` and asserts
/// `strided[j] == contiguous[j * 3]` for each j.
#[test]
fn sa_positions_strided_matches_manual() {
    let (dir, prefix_str, _pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let k: u64 = 2;
    let step: u64 = 3;
    let n_out: u64 = 4;

    // Fetch strided positions.
    let mut strided_out = vec![0u64; n_out as usize];
    let rc = unsafe { prmi_sa_positions_strided(handle, k, step, n_out, strided_out.as_mut_ptr()) };
    assert_eq!(rc, 0, "prmi_sa_positions_strided returned rc={rc}");

    // Fetch a contiguous block covering all sampled indices: [k, k + n_out*step).
    let contiguous_count = n_out * step;
    let mut contiguous_out = vec![0u64; contiguous_count as usize];
    let rc2 =
        unsafe { prmi_sa_positions(handle, k, contiguous_count, contiguous_out.as_mut_ptr()) };
    assert_eq!(
        rc2, 0,
        "prmi_sa_positions (contiguous reference) returned rc={rc2}"
    );

    // strided[j] must equal contiguous[j * step].
    for j in 0..n_out as usize {
        assert_eq!(
            strided_out[j],
            contiguous_out[j * step as usize],
            "mismatch at j={j}: strided={} contiguous={}",
            strided_out[j],
            contiguous_out[j * step as usize]
        );
    }

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test that `prmi_sa_positions_strided` with `n_out == 0` returns 0 and
/// makes no writes.
#[test]
fn sa_positions_strided_n_out_zero_returns_zero() {
    let (dir, prefix_str, _pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // n_out == 0 → 0 with no writes, null out_positions is fine.
    let rc = unsafe { prmi_sa_positions_strided(handle, 0, 1, 0, ptr::null_mut()) };
    assert_eq!(rc, 0, "n_out=0 should return 0, got {rc}");

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test that a request whose last sampled index exceeds `sa_num` returns -4.
#[test]
fn sa_positions_strided_out_of_range_returns_minus_four() {
    let (dir, prefix_str, _pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    // Use a huge k that is certain to exceed sa_num.
    let huge_k: u64 = u64::MAX / 2;
    let mut buf = vec![0u64; 4];
    let rc = unsafe { prmi_sa_positions_strided(handle, huge_k, 1, 4, buf.as_mut_ptr()) };
    assert_eq!(
        rc, -4,
        "out-of-range strided read should return -4, got {rc}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Test that a null handle returns -1 for `prmi_sa_positions_strided`.
#[test]
fn sa_positions_strided_null_handle_returns_minus_one() {
    let mut buf = [0u64; 4];
    let rc = unsafe { prmi_sa_positions_strided(ptr::null(), 0, 1, 4, buf.as_mut_ptr()) };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

/// Test that a null `out_positions` with `n_out > 0` returns -2.
#[test]
fn sa_positions_strided_null_out_with_n_out_returns_minus_two() {
    let (dir, prefix_str, _pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let rc = unsafe { prmi_sa_positions_strided(handle, 0, 1, 4, ptr::null_mut()) };
    assert_eq!(
        rc, -2,
        "null out_positions with n_out>0 should return -2, got {rc}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

// ─── Batch variant tests ──────────────────────────────────────────────────────

/// Run a 2-task forward-spectrum batch over a shared arena, then verify that
/// each task's `out_nsteps[i]` and the written steps match the corresponding
/// single-query `prmi_forward_spectrum` result element-for-element.
#[test]
fn forward_spectrum_batch_matches_single() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;
    let query_len: usize = 32;

    // Two queries lifted from different offsets of the reference.
    let q0: Vec<u8> = pac_unpacked[0..query_len].to_vec();
    let q1: Vec<u8> = pac_unpacked[16..16 + query_len].to_vec();

    // Build a flat queries arena: [q0 || q1].
    let mut queries_arena = Vec::new();
    queries_arena.extend_from_slice(&q0);
    queries_arena.extend_from_slice(&q1);

    // Steps arena: 64 slots per task, tasks at steps_off 0 and 64.
    const MAX_STEPS: u32 = 64;
    let mut steps_arena: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        2 * MAX_STEPS as usize
    ];
    let mut out_nsteps = [0u64; 2];

    let tasks = [
        prmi_fwd_task_t {
            query_off: 0,
            query_len: query_len as u32,
            steps_off: 0,
            max_steps: MAX_STEPS,
        },
        prmi_fwd_task_t {
            query_off: query_len as u64,
            query_len: query_len as u32,
            steps_off: MAX_STEPS,
            max_steps: MAX_STEPS,
        },
    ];

    let rc = unsafe {
        prmi_forward_spectrum_batch(
            handle,
            queries_arena.as_ptr(),
            queries_arena.len() as u64,
            tasks.as_ptr(),
            2,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            steps_arena.len() as u64,
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "prmi_forward_spectrum_batch failed with rc={rc}");

    // Run single-query calls for both queries and compare element-for-element.
    for (qi, query) in [&q0, &q1].iter().enumerate() {
        let mut single_steps: Vec<prmi_smem_step_t> = vec![
            prmi_smem_step_t {
                sa_start: 0,
                occ_count: 0,
                match_len: 0
            };
            query_len
        ];
        let mut single_nsteps: u64 = 0;
        let rc_s = unsafe {
            prmi_forward_spectrum(
                handle,
                query.as_ptr(),
                query_len as i32,
                pac_packed.as_ptr(),
                pac_num_bases,
                single_steps.as_mut_ptr(),
                single_steps.len() as u64,
                &mut single_nsteps,
            )
        };
        assert_eq!(rc_s, 0, "single prmi_forward_spectrum (task {qi}) failed");
        assert_eq!(
            out_nsteps[qi], single_nsteps,
            "task {qi}: batch nsteps={} != single nsteps={single_nsteps}",
            out_nsteps[qi]
        );

        let task_steps_off = tasks[qi].steps_off as usize;
        let n = out_nsteps[qi] as usize;
        for k in 0..n {
            let batch_step = &steps_arena[task_steps_off + k];
            let single_step = &single_steps[k];
            assert_eq!(
                batch_step.sa_start, single_step.sa_start,
                "task {qi} step {k}: sa_start mismatch"
            );
            assert_eq!(
                batch_step.occ_count, single_step.occ_count,
                "task {qi} step {k}: occ_count mismatch"
            );
            assert_eq!(
                batch_step.match_len, single_step.match_len,
                "task {qi} step {k}: match_len mismatch"
            );
        }
    }

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// The batch overflow path: a task with `max_steps == 0` for a query that
/// produces at least one step must cause the batch to return -4, and that
/// task's `out_nsteps` must be set to the needed count (> 0).
#[test]
fn forward_spectrum_batch_overflow_returns_minus_four() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;
    let query_len: usize = 32;
    let query: Vec<u8> = pac_unpacked[0..query_len].to_vec();

    // Provide a real steps buffer but advertise max_steps=0 so the task overflows.
    let mut steps_arena: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        64
    ];
    let mut out_nsteps = [0u64; 1];

    let tasks = [prmi_fwd_task_t {
        query_off: 0,
        query_len: query_len as u32,
        steps_off: 0,
        max_steps: 0, // too small
    }];

    let rc = unsafe {
        prmi_forward_spectrum_batch(
            handle,
            query.as_ptr(),
            query.len() as u64,
            tasks.as_ptr(),
            1,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            steps_arena.len() as u64,
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(rc, -4, "expected -4 (overflow), got {rc}");
    assert!(
        out_nsteps[0] > 0,
        "out_nsteps[0] must be set to the needed count (>0), got {}",
        out_nsteps[0]
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Smoke test for the backward-spectrum batch: 2 tasks derived from forward
/// steps; assert rc==0 and each task's nsteps is sane (>= 0).
#[test]
fn backward_spectrum_batch_smoke() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;

    // Build two anchors using forward_spectrum (pivot at offsets 20 and 24,
    // both multiples of 4 so T always precedes them).
    let anchors = [(20usize, 1usize), (24usize, 1usize)];
    let anchor_query_len = 16usize;

    // Reads arena: each read = 1 left base + anchor_query_len bases.
    let read_len = 1 + anchor_query_len;
    let mut reads_arena: Vec<u8> = Vec::new();
    let mut fwd_anchor_sa_start = [0u64; 2];
    let mut fwd_anchor_occ_count = [0u64; 2];
    let mut fwd_anchor_len = [0u64; 2];

    for (idx, &(anchor_off, pivot)) in anchors.iter().enumerate() {
        let left_off = anchor_off - pivot;
        let read: Vec<u8> = pac_unpacked[left_off..left_off + read_len].to_vec();
        reads_arena.extend_from_slice(&read);

        let anchor_query: Vec<u8> = read[pivot..].to_vec();
        let mut fwd_steps: Vec<prmi_smem_step_t> = vec![
            prmi_smem_step_t {
                sa_start: 0,
                occ_count: 0,
                match_len: 0
            };
            anchor_query.len()
        ];
        let mut fwd_nsteps: u64 = 0;
        let rc_f = unsafe {
            prmi_forward_spectrum(
                handle,
                anchor_query.as_ptr(),
                anchor_query.len() as i32,
                pac_packed.as_ptr(),
                pac_num_bases,
                fwd_steps.as_mut_ptr(),
                fwd_steps.len() as u64,
                &mut fwd_nsteps,
            )
        };
        assert_eq!(rc_f, 0, "forward_spectrum failed for anchor {idx}");
        assert!(
            fwd_nsteps >= 1,
            "expected at least one forward step for anchor {idx}"
        );

        // Use the shallowest step as the anchor for backward search.
        fwd_anchor_sa_start[idx] = fwd_steps[0].sa_start;
        fwd_anchor_occ_count[idx] = fwd_steps[0].occ_count;
        fwd_anchor_len[idx] = fwd_steps[0].match_len;
    }

    // Steps arena: 32 slots per task.
    const MAX_STEPS: u32 = 32;
    let mut steps_arena: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        2 * MAX_STEPS as usize
    ];
    let mut out_nsteps = [0u64; 2];

    let tasks = [
        prmi_bwd_task_t {
            sa_start: fwd_anchor_sa_start[0],
            occ_count: fwd_anchor_occ_count[0],
            anchor_len: fwd_anchor_len[0],
            read_off: 0,
            read_len: read_len as u32,
            pivot: anchors[0].1 as u32,
            steps_off: 0,
            max_steps: MAX_STEPS,
        },
        prmi_bwd_task_t {
            sa_start: fwd_anchor_sa_start[1],
            occ_count: fwd_anchor_occ_count[1],
            anchor_len: fwd_anchor_len[1],
            read_off: read_len as u64,
            read_len: read_len as u32,
            pivot: anchors[1].1 as u32,
            steps_off: MAX_STEPS,
            max_steps: MAX_STEPS,
        },
    ];

    let rc = unsafe {
        prmi_backward_spectrum_batch(
            handle,
            reads_arena.as_ptr(),
            reads_arena.len() as u64,
            tasks.as_ptr(),
            2,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            steps_arena.len() as u64,
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "prmi_backward_spectrum_batch failed with rc={rc}");

    // Both tasks should produce at least one backward step (T precedes every
    // anchor at offsets 20 and 24 in the ACGT-repeat reference).
    for idx in 0..2 {
        assert!(
            out_nsteps[idx] >= 1,
            "task {idx}: expected at least one backward step, got {}",
            out_nsteps[idx]
        );
        let task_steps_off = tasks[idx].steps_off as usize;
        for k in 0..out_nsteps[idx] as usize {
            let step = &steps_arena[task_steps_off + k];
            assert!(
                step.match_len > fwd_anchor_len[idx],
                "task {idx} step {k}: match_len={} must exceed anchor_len={}",
                step.match_len,
                fwd_anchor_len[idx]
            );
        }
    }

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Null-handle returns -1 for `prmi_forward_spectrum_batch`.
#[test]
fn forward_spectrum_batch_null_handle_returns_minus_one() {
    let pac = [0u8; 8];
    let query = [0u8; 32];
    let task = prmi_fwd_task_t {
        query_off: 0,
        query_len: 32,
        steps_off: 0,
        max_steps: 64,
    };
    let mut steps = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        64
    ];
    let mut nsteps = [0u64; 1];
    let rc = unsafe {
        prmi_forward_spectrum_batch(
            ptr::null(),
            query.as_ptr(),
            query.len() as u64,
            &task as *const prmi_fwd_task_t,
            1,
            pac.as_ptr(),
            32,
            steps.as_mut_ptr(),
            steps.len() as u64,
            nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

/// Null-handle returns -1 for `prmi_backward_spectrum_batch`.
#[test]
fn backward_spectrum_batch_null_handle_returns_minus_one() {
    let pac = [0u8; 8];
    let read = [0u8; 4];
    let task = prmi_bwd_task_t {
        sa_start: 0,
        occ_count: 1,
        anchor_len: 1,
        read_off: 0,
        read_len: 4,
        pivot: 1,
        steps_off: 0,
        max_steps: 32,
    };
    let mut steps = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        32
    ];
    let mut nsteps = [0u64; 1];
    let rc = unsafe {
        prmi_backward_spectrum_batch(
            ptr::null(),
            read.as_ptr(),
            read.len() as u64,
            &task as *const prmi_bwd_task_t,
            1,
            pac.as_ptr(),
            32,
            steps.as_mut_ptr(),
            steps.len() as u64,
            nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(rc, -1, "null handle should return -1, got {rc}");
}

/// `ntasks == 0` returns 0 immediately for forward batch.
#[test]
fn forward_spectrum_batch_ntasks_zero_returns_zero() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);
    let pac_packed = pack_bases(&pac_unpacked);
    // All arena/task pointers null — ntasks==0 must still return 0.
    let rc = unsafe {
        prmi_forward_spectrum_batch(
            handle,
            ptr::null(),
            0,
            ptr::null(),
            0,
            pac_packed.as_ptr(),
            pac_unpacked.len() as u64,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "ntasks=0 should return 0, got {rc}");
    unsafe { prmi_close(handle) };
    drop(dir);
}

/// A task whose `query_off + query_len` exceeds `queries_arena_len` must return
/// -2 and must not crash (no out-of-bounds read).
#[test]
fn forward_spectrum_batch_out_of_arena_returns_minus_two() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;
    let query_len: usize = 32;
    let query: Vec<u8> = pac_unpacked[0..query_len].to_vec();

    let mut steps_arena: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        64
    ];
    let mut out_nsteps = [0u64; 1];

    // queries_arena_len is only 16, but task.query_len=32 — the window overflows.
    let tasks = [prmi_fwd_task_t {
        query_off: 0,
        query_len: query_len as u32,
        steps_off: 0,
        max_steps: 64,
    }];
    let rc = unsafe {
        prmi_forward_spectrum_batch(
            handle,
            query.as_ptr(),
            16, // deliberately too small — query window [0, 32) > 16
            tasks.as_ptr(),
            1,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            steps_arena.len() as u64,
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(
        rc, -2,
        "out-of-arena query window should return -2, got {rc}"
    );

    // Also test the steps arena overflow path: query arena is correct but
    // steps_arena_len is too small for the declared steps_off + max_steps.
    let tasks2 = [prmi_fwd_task_t {
        query_off: 0,
        query_len: query_len as u32,
        steps_off: 60, // 60 + 64 = 124 > 64 (steps_arena_len)
        max_steps: 64,
    }];
    let rc2 = unsafe {
        prmi_forward_spectrum_batch(
            handle,
            query.as_ptr(),
            query.len() as u64,
            tasks2.as_ptr(),
            1,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            64, // steps_arena_len=64, steps_off=60 + max_steps=64 > 64
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(
        rc2, -2,
        "out-of-arena steps window should return -2, got {rc2}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}

/// Backward batch: a task whose `read_off + read_len` exceeds `reads_arena_len`
/// must return -2 and must not crash.
#[test]
fn backward_spectrum_batch_out_of_arena_returns_minus_two() {
    let (dir, prefix_str, pac_unpacked) = build_test_sidecar();
    let cprefix = CString::new(prefix_str).unwrap();
    let mut handle = ptr::null_mut();
    assert_eq!(unsafe { prmi_open(cprefix.as_ptr(), &mut handle) }, 0);

    let pac_packed = pack_bases(&pac_unpacked);
    let pac_num_bases: u64 = pac_unpacked.len() as u64;

    // Build one anchor via forward_spectrum.
    let anchor_offset = 20usize;
    let anchor_query_len = 16usize;
    let anchor_query: Vec<u8> =
        pac_unpacked[anchor_offset..anchor_offset + anchor_query_len].to_vec();
    let mut fwd_steps: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        anchor_query_len
    ];
    let mut fwd_nsteps: u64 = 0;
    let rc_f = unsafe {
        prmi_forward_spectrum(
            handle,
            anchor_query.as_ptr(),
            anchor_query.len() as i32,
            pac_packed.as_ptr(),
            pac_num_bases,
            fwd_steps.as_mut_ptr(),
            fwd_steps.len() as u64,
            &mut fwd_nsteps,
        )
    };
    assert_eq!(rc_f, 0);
    assert!(fwd_nsteps >= 1);

    let left_offset = 19usize;
    let read_len = 1 + anchor_query_len;
    let reads_arena: Vec<u8> = pac_unpacked[left_offset..left_offset + read_len].to_vec();

    let mut steps_arena: Vec<prmi_smem_step_t> = vec![
        prmi_smem_step_t {
            sa_start: 0,
            occ_count: 0,
            match_len: 0
        };
        32
    ];
    let mut out_nsteps = [0u64; 1];

    // reads_arena_len is only 5, but read_len=17 — window overflows.
    let tasks = [prmi_bwd_task_t {
        sa_start: fwd_steps[0].sa_start,
        occ_count: fwd_steps[0].occ_count,
        anchor_len: fwd_steps[0].match_len,
        read_off: 0,
        read_len: read_len as u32,
        pivot: 1,
        steps_off: 0,
        max_steps: 32,
    }];
    let rc = unsafe {
        prmi_backward_spectrum_batch(
            handle,
            reads_arena.as_ptr(),
            5, // deliberately too small — read window [0, 17) > 5
            tasks.as_ptr(),
            1,
            pac_packed.as_ptr(),
            pac_num_bases,
            steps_arena.as_mut_ptr(),
            steps_arena.len() as u64,
            out_nsteps.as_mut_ptr(),
        )
    };
    assert_eq!(
        rc, -2,
        "out-of-arena read window should return -2, got {rc}"
    );

    unsafe { prmi_close(handle) };
    drop(dir);
}
