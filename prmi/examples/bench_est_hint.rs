// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Per-call benchmark: `mem_search` (model-launch, full nested narrowing) vs
//! `mem_search_from_hint` (the est_hint>0 / `no_search` confirm-only launch).
//!
//! This is the measurement gating Step 2 ISA: it isolates how much a confirm-only
//! launch from an EXACT inverse-SA hint saves per `prmi_mem_search` call, before
//! spending the +51 GB `.isa` storage or the consumer's carried-state rework. The
//! hint fed here is the true in-interval SA index from an initial unhinted call,
//! exactly what `prmi_isa_at(refpos)` would supply — so the timing reflects the
//! production fast path, and the harness asserts byte-identity as it goes.
//!
//! Run: `cargo run --release --example bench_est_hint`
//! Optional env: `PRMI_BENCH_REFLEN` (ref bases, default 2_000_000),
//! `PRMI_BENCH_QUERIES` (default 5000), `PRMI_BENCH_QLEN` (default 80),
//! `PRMI_BENCH_REPS` (timed passes over the query set, default 20).

use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi::train::mask::MaskConfig;
use std::io::Write;
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic ACGT bases via a PCG-style LCG (no external RNG, reproducible).
fn synth_bases(n: usize) -> Vec<u8> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 61) & 3) as u8
        })
        .collect()
}

fn main() {
    let ref_len = env_usize("PRMI_BENCH_REFLEN", 2_000_000);
    let n_queries = env_usize("PRMI_BENCH_QUERIES", 5_000);
    let qlen = env_usize("PRMI_BENCH_QLEN", 80);
    let reps = env_usize("PRMI_BENCH_REPS", 20);
    // The query/read windows index `bases[start..start+len]` and the start is
    // taken modulo `ref_len - len`, so the window length must be a positive
    // value strictly below the reference length or that math divides/mods by
    // zero (panic). The backward pass derives `bwd_qlen = qlen.max(60)`, so
    // guard against that derived length too.
    assert!(ref_len > 0, "PRMI_BENCH_REFLEN must be > 0");
    assert!(
        qlen > 0 && qlen < ref_len,
        "PRMI_BENCH_QLEN must be in 1..PRMI_BENCH_REFLEN (got {qlen}, ref_len {ref_len})"
    );
    assert!(
        qlen.max(60) < ref_len,
        "derived backward window length {} must be < PRMI_BENCH_REFLEN ({ref_len})",
        qlen.max(60)
    );

    eprintln!("[bench] generating {ref_len} synthetic bases ...");
    let bases = synth_bases(ref_len);

    // Write a FASTA (ACGT only, no N → SA order is well-defined) and build a
    // mode-2 sidecar (stored keys, the production search substrate).
    let tmp = std::env::temp_dir().join(format!("prmi_bench_est_hint_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");
    let fa = tmp.join("ref.fa");
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&fa).unwrap());
        writeln!(w, ">bench").unwrap();
        let alphabet = [b'A', b'C', b'G', b'T'];
        for chunk in bases.chunks(60) {
            let line: Vec<u8> = chunk.iter().map(|&b| alphabet[b as usize]).collect();
            w.write_all(&line).unwrap();
            w.write_all(b"\n").unwrap();
        }
    }
    let prefix = tmp.join("ref.prmi");
    eprintln!("[bench] building mode-2 sidecar ...");
    let t0 = Instant::now();
    build_sidecar_with_config(
        &fa,
        &prefix,
        None,
        MaskConfig::default(),
        0,
        Some(
            TrainerConfig::default()
                .with_memory_mode(MemoryMode::Mode2)
                .with_isa(true),
        ),
    )
    .expect("build sidecar");
    let idx = LearnedIndex::open(&prefix).expect("open sidecar");
    let enc = PacEncoding::Unpacked;
    eprintln!(
        "[bench] built + opened in {:.1}s · sa_num={} l_pac={} log2(sa_num)={:.1}",
        t0.elapsed().as_secs_f64(),
        idx.sa_num(),
        idx.l_pac(),
        (idx.sa_num() as f64).log2()
    );

    // Reference-lifted queries (windows of the forward bases) → each matches
    // maximally, exactly as a real seed extension does. Keep only matching
    // queries and record the true in-interval SA index as the launch hint.
    let span = ref_len - qlen;
    let stride = span.max(1) / n_queries.max(1);
    let mut work: Vec<(Vec<u8>, u64)> = Vec::with_capacity(n_queries);
    for k in 0..n_queries {
        let start = (k * stride) % span;
        let q = bases[start..start + qlen].to_vec();
        let m = idx.mem_search(&q, &bases, enc);
        if m.match_len > 0 && m.sa_start > 0 {
            // Byte-identity sanity for the hint path (asserted once per query).
            let h = idx.mem_search_from_hint(&q, m.sa_start, true, &bases, enc);
            assert_eq!(h, m, "hinted != unhinted while preparing bench corpus");
            work.push((q, m.sa_start));
        }
    }
    eprintln!(
        "[bench] corpus: {} matching queries (qlen={qlen})",
        work.len()
    );
    assert!(!work.is_empty(), "no matching queries");

    // Timed passes. `black_box` keeps the optimizer from hoisting the calls.
    let full = time_pass(reps, &work, |q, _hint| {
        std::hint::black_box(idx.mem_search(q, &bases, enc))
    });
    let hint_iv = time_pass(reps, &work, |q, hint| {
        std::hint::black_box(idx.mem_search_from_hint(q, hint, true, &bases, enc))
    });
    let hint_ml = time_pass(reps, &work, |q, hint| {
        std::hint::black_box(idx.mem_search_from_hint(q, hint, false, &bases, enc))
    });

    let per_call = |ns: f64| ns / (reps * work.len()) as f64;
    let (f, hi, hm) = (per_call(full), per_call(hint_iv), per_call(hint_ml));
    println!(
        "\n=== mem_search per-call (ns), {} queries × {reps} reps ===",
        work.len()
    );
    println!("  model-launch (full search)        : {f:8.1} ns/call");
    println!(
        "  est_hint + interval (no_search)   : {hi:8.1} ns/call   {:.2}× faster",
        f / hi
    );
    println!(
        "  est_hint, match_len only          : {hm:8.1} ns/call   {:.2}× faster",
        f / hm
    );
    println!(
        "\nConfirm-only launch removes the nested narrowing: {:.2}× on the interval path, \
         {:.2}× when only match_len is needed.",
        f / hi,
        f / hm
    );

    // ── backward (left extension) ──────────────────────────────────────────
    // Reference-lifted reads: a window of the forward bases at genomic `s`. The
    // right anchor is the forward one-shot of read[pivot..]; the hint is the
    // anchor's inverse-SA index at its natural locus (s + pivot).
    assert!(idx.has_isa(), "backward est_hint bench needs --with-isa");
    let bwd_qlen = qlen.max(60);
    let pivot = bwd_qlen / 2;
    let mut bwork: Vec<BwdItem> = Vec::with_capacity(n_queries);
    let bspan = ref_len - bwd_qlen;
    let bstride = bspan.max(1) / n_queries.max(1);
    for k in 0..n_queries {
        let s = (k * bstride) % bspan;
        let read = bases[s..s + bwd_qlen].to_vec();
        let fwd = idx.mem_search(&read[pivot..], &bases, enc);
        if fwd.match_len == 0 {
            continue;
        }
        let hint = idx.isa_at((s + pivot) as u64).expect("refpos in range");
        let full = idx.mem_search_backward(
            fwd.sa_start,
            fwd.occ,
            fwd.match_len,
            &read,
            pivot,
            &bases,
            enc,
        );
        let hinted =
            idx.mem_search_backward_from_hint(&read, pivot, fwd.match_len, hint, true, &bases, enc);
        assert_eq!(
            hinted, full,
            "backward hinted != from-scratch in bench prep"
        );
        bwork.push(BwdItem {
            read,
            pivot,
            anchor_len: fwd.match_len,
            sa_start: fwd.sa_start,
            occ: fwd.occ,
            hint,
        });
    }
    assert!(!bwork.is_empty(), "no backward corpus");

    let bwd_time = |f: &dyn Fn(&BwdItem)| -> f64 {
        let t = Instant::now();
        for _ in 0..reps {
            for it in &bwork {
                f(it);
            }
        }
        t.elapsed().as_nanos() as f64
    };
    let bf = bwd_time(&|it| {
        std::hint::black_box(idx.mem_search_backward(
            it.sa_start,
            it.occ,
            it.anchor_len,
            &it.read,
            it.pivot,
            &bases,
            enc,
        ));
    });
    let bhi = bwd_time(&|it| {
        std::hint::black_box(idx.mem_search_backward_from_hint(
            &it.read,
            it.pivot,
            it.anchor_len,
            it.hint,
            true,
            &bases,
            enc,
        ));
    });
    let bhm = bwd_time(&|it| {
        std::hint::black_box(idx.mem_search_backward_from_hint(
            &it.read,
            it.pivot,
            it.anchor_len,
            it.hint,
            false,
            &bases,
            enc,
        ));
    });
    let bper = |ns: f64| ns / (reps * bwork.len()) as f64;
    let (bf, bhi, bhm) = (bper(bf), bper(bhi), bper(bhm));
    println!(
        "\n=== mem_search_backward per-call (ns), {} items × {reps} reps ===",
        bwork.len()
    );
    println!("  model-launch (full search)        : {bf:8.1} ns/call");
    println!(
        "  est_hint + interval (no_search)   : {bhi:8.1} ns/call   {:.2}× faster",
        bf / bhi
    );
    println!(
        "  est_hint, match_len only          : {bhm:8.1} ns/call   {:.2}× faster",
        bf / bhm
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// One backward bench item (a reference-lifted read with its derived anchor + hint).
struct BwdItem {
    read: Vec<u8>,
    pivot: usize,
    anchor_len: u64,
    sa_start: u64,
    occ: u64,
    hint: u64,
}

/// Run `f` over every (query, hint) `reps` times; return total nanoseconds.
fn time_pass<F: Fn(&[u8], u64) -> prmi::index::spectrum::MemMatch>(
    reps: usize,
    work: &[(Vec<u8>, u64)],
    f: F,
) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        for (q, hint) in work {
            f(q, *hint);
        }
    }
    t.elapsed().as_nanos() as f64
}
