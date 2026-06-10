// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Criterion benchmarks for the FFI-facing one-shot search entry points that the
//! bwa-meme zigzag SMEM finder calls once per pivot/per-read:
//!
//! - `prmi_mem_search` → [`LearnedIndex::mem_search`] (model launch) and
//!   [`LearnedIndex::mem_search_from_hint`] (the `est_hint>0` / `no_search`
//!   confirm-only launch).
//! - `prmi_mem_search_backward` → [`LearnedIndex::mem_search_backward`] and
//!   [`LearnedIndex::mem_search_backward_from_hint`].
//!
//! These are the consumer's heaviest per-call paths; the lower-level
//! `forward_spectrum` / `backward_spectrum` primitives they bottom out in are
//! benched separately in `spectrum_bench.rs`. Each path is measured under two
//! occurrence regimes — a UNIQUE-occ corpus (reference-lifted reads, small SA
//! interval, the common aligner case) and a high-occ TANDEM-REPEAT corpus
//! (large interval, the D15 stress case) — because interval recovery cost
//! scales with occ.
//!
//! The hint fed to the `*_from_hint` benches is the true in-interval SA index
//! produced by an initial unhinted call — exactly what `prmi_isa_at(refpos)`
//! supplies in production — so the timing reflects the real fast path. The
//! setup asserts byte-identity (hinted == unhinted) once per corpus item, so a
//! regression that breaks the invariant fails the bench build rather than
//! silently mis-measuring.
//!
//! The index is built ONCE (outside every timed loop) and shared across all
//! groups. Reference size is env-tunable so the same harness covers a
//! laptop-fast default and a genomic-scale run:
//!   `PRMI_BENCH_REFLEN` (random-backbone bases, default 2_000_000).
//! For production numbers against a real chromosome, raise it (e.g. chr22 is
//! ~50M) — the build runs in criterion setup, not the timed loop.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use prmi::index::smem::PacEncoding;
use prmi::index::spectrum::MemMatch;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use tempfile::TempDir;

/// Read a `usize` from the environment, falling back to `default`.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Number of bases per query in the forward corpora.
const QUERY_LEN: usize = 80;
/// Number of reads per corpus.
const CORPUS_SIZE: usize = 256;
/// Tandem-repeat block: `REPEAT_COUNT × "ACGT"` inserted at the reference
/// midpoint to create a high-occurrence (large-interval) region.
const REPEAT_UNIT_ASCII: &[u8] = b"ACGT";
const REPEAT_UNIT_ENC: &[u8] = &[0, 1, 2, 3];
const REPEAT_COUNT: usize = 1024;

/// Deterministic ACGT bases (0..=3) via a PCG-style LCG — reproducible, no RNG dep.
fn synth_bases(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 61) & 3) as u8
        })
        .collect()
}

/// Built bench fixture: the tempdir (keeps mmap'd files alive), the unpacked pac
/// (one base/byte, 0..=3 — also the `bases` the queries are lifted from), the
/// index, and the genomic offset/length of the tandem-repeat block.
struct Fixture {
    _dir: TempDir,
    pac: Vec<u8>,
    idx: LearnedIndex,
    repeat_start: usize,
    repeat_len: usize,
    backbone_end: usize,
}

/// Build a mode-2 (stored-key) sidecar `--with-isa` over a synthetic reference:
/// a random backbone with a tandem-repeat block spliced in at the midpoint.
fn build_fixture() -> Fixture {
    let ref_len = env_usize("PRMI_BENCH_REFLEN", 2_000_000);
    let backbone = synth_bases(ref_len, 0x2545_F491_4F6C_DD1D);
    let half = ref_len / 2;
    let repeat_len = REPEAT_UNIT_ASCII.len() * REPEAT_COUNT;

    // Assemble unpacked bases: backbone[..half] + repeat + backbone[half..].
    let mut pac = Vec::with_capacity(ref_len + repeat_len);
    pac.extend_from_slice(&backbone[..half]);
    let repeat_start = pac.len();
    for _ in 0..REPEAT_COUNT {
        pac.extend_from_slice(REPEAT_UNIT_ENC);
    }
    pac.extend_from_slice(&backbone[half..]);
    // `backbone_end` is the last index of the FIRST backbone segment usable for
    // unique reads (keeps them clear of the repeat block).
    let backbone_end = repeat_start;

    // Write the matching FASTA and build the sidecar.
    let dir = tempfile::tempdir().expect("tempdir");
    let fa = dir.path().join("ref.fa");
    {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&fa).unwrap());
        writeln!(w, ">bench").unwrap();
        let alphabet = [b'A', b'C', b'G', b'T'];
        for chunk in pac.chunks(60) {
            let line: Vec<u8> = chunk.iter().map(|&b| alphabet[b as usize]).collect();
            w.write_all(&line).unwrap();
            w.write_all(b"\n").unwrap();
        }
    }
    let prefix = dir.path().join("ref.prmi");
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_isa(true);
    build_sidecar_with_config(&fa, &prefix, None, Default::default(), 0, Some(cfg))
        .expect("build sidecar");
    let idx = LearnedIndex::open(&prefix).expect("open index");
    assert!(idx.has_isa(), "fixture must build with .isa");

    Fixture {
        _dir: dir,
        pac,
        idx,
        repeat_start,
        repeat_len,
        backbone_end,
    }
}

// ── Forward corpora ─────────────────────────────────────────────────────────

/// A forward query plus the in-interval SA index hint that `prmi_isa_at` would
/// supply (the `sa_start` of its unhinted maximal match).
struct FwdItem {
    query: Vec<u8>,
    hint: u64,
}

/// Reference-lifted unique-occ forward corpus: windows of the random backbone,
/// each matching maximally with a small SA interval.
fn build_forward_unique(fx: &Fixture) -> Vec<FwdItem> {
    let enc = PacEncoding::Unpacked;
    let max_start = fx.backbone_end.saturating_sub(QUERY_LEN);
    assert!(max_start > 0, "backbone too short for QUERY_LEN");
    let stride = (max_start / CORPUS_SIZE).max(1);
    let mut out = Vec::with_capacity(CORPUS_SIZE);
    for k in 0..CORPUS_SIZE {
        let start = (k * stride) % max_start;
        let query = fx.pac[start..start + QUERY_LEN].to_vec();
        let m = fx.idx.mem_search(&query, &fx.pac, enc);
        if m.match_len == 0 || m.sa_start == 0 {
            continue;
        }
        // Byte-identity insurance: the hint launch must equal the model launch.
        let h = fx
            .idx
            .mem_search_from_hint(&query, m.sa_start, true, &fx.pac, enc);
        assert_eq!(h, m, "forward hinted != unhinted in corpus prep");
        out.push(FwdItem {
            query,
            hint: m.sa_start,
        });
    }
    assert!(!out.is_empty(), "no unique forward matches");
    out
}

/// High-occ forward corpus: queries drawn from the tandem-repeat block. Every
/// query matches the large repeat interval (`occ` ~ repeat length).
fn build_forward_repeat(fx: &Fixture) -> Vec<FwdItem> {
    let enc = PacEncoding::Unpacked;
    // The whole repeat block matches; lift QUERY_LEN windows from inside it so
    // each query is fully contained in the repeat region.
    let max_off = fx.repeat_len.saturating_sub(QUERY_LEN);
    assert!(max_off > 0, "repeat block too short for QUERY_LEN");
    let mut out = Vec::with_capacity(CORPUS_SIZE);
    for k in 0..CORPUS_SIZE {
        let off = k % max_off; // contiguous windows; phase varies vs the repeat unit
        let start = fx.repeat_start + off;
        let query = fx.pac[start..start + QUERY_LEN].to_vec();
        let m = fx.idx.mem_search(&query, &fx.pac, enc);
        if m.match_len == 0 || m.sa_start == 0 {
            continue;
        }
        out.push(FwdItem {
            query,
            hint: m.sa_start,
        });
    }
    assert!(!out.is_empty(), "no repeat forward matches");
    out
}

// ── Backward corpus ─────────────────────────────────────────────────────────

/// A backward item: a reference-lifted read, its pivot, the right anchor derived
/// from `read[pivot..]`, and the inverse-SA hint at the anchor's genomic locus.
struct BwdItem {
    read: Vec<u8>,
    pivot: usize,
    anchor_len: u64,
    sa_start: u64,
    occ: u64,
    hint: u64,
}

/// Reference-lifted backward corpus from the unique backbone: read = a window at
/// genomic `s`; the anchor is the forward one-shot of `read[pivot..]`; the hint
/// is `isa_at(s + pivot)` (the anchor's exact inverse-SA index).
fn build_backward_unique(fx: &Fixture) -> Vec<BwdItem> {
    let enc = PacEncoding::Unpacked;
    let bwd_len = QUERY_LEN.max(60);
    let pivot = bwd_len / 2;
    let max_start = fx.backbone_end.saturating_sub(bwd_len);
    assert!(max_start > 0, "backbone too short for backward reads");
    let stride = (max_start / CORPUS_SIZE).max(1);
    let mut out = Vec::with_capacity(CORPUS_SIZE);
    for k in 0..CORPUS_SIZE {
        let s = (k * stride) % max_start;
        let read = fx.pac[s..s + bwd_len].to_vec();
        let fwd = fx.idx.mem_search(&read[pivot..], &fx.pac, enc);
        if fwd.match_len == 0 {
            continue;
        }
        let hint = match fx.idx.isa_at((s + pivot) as u64) {
            Some(h) if h != 0 => h,
            _ => continue,
        };
        let full = fx.idx.mem_search_backward(
            fwd.sa_start,
            fwd.occ,
            fwd.match_len,
            &read,
            pivot,
            &fx.pac,
            enc,
        );
        let hinted = fx.idx.mem_search_backward_from_hint(
            &read,
            pivot,
            fwd.match_len,
            hint,
            true,
            &fx.pac,
            enc,
        );
        assert_eq!(
            hinted, full,
            "backward hinted != from-scratch in corpus prep"
        );
        out.push(BwdItem {
            read,
            pivot,
            anchor_len: fwd.match_len,
            sa_start: fwd.sa_start,
            occ: fwd.occ,
            hint,
        });
    }
    assert!(!out.is_empty(), "no backward corpus");
    out
}

// ── Benchmark groups ────────────────────────────────────────────────────────

fn run_forward(c: &mut Criterion, fx: &Fixture) {
    let enc = PacEncoding::Unpacked;
    let unique = build_forward_unique(fx);
    let repeat = build_forward_repeat(fx);

    let mut g = c.benchmark_group("mem_search_forward");
    g.throughput(Throughput::Elements(unique.len() as u64));

    g.bench_function("model_launch/unique", |b| {
        b.iter(|| {
            for it in &unique {
                black_box(fx.idx.mem_search(black_box(&it.query), &fx.pac, enc));
            }
        });
    });
    g.bench_function("model_launch/repeat", |b| {
        b.iter(|| {
            for it in &repeat {
                black_box(fx.idx.mem_search(black_box(&it.query), &fx.pac, enc));
            }
        });
    });
    g.bench_function("est_hint_interval/unique", |b| {
        b.iter(|| {
            for it in &unique {
                black_box(fx.idx.mem_search_from_hint(
                    black_box(&it.query),
                    black_box(it.hint),
                    true,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.bench_function("est_hint_match_len/unique", |b| {
        b.iter(|| {
            for it in &unique {
                black_box(fx.idx.mem_search_from_hint(
                    black_box(&it.query),
                    black_box(it.hint),
                    false,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    // High-occ (tandem-repeat) hinted variants, mirroring the unique ones above —
    // the est_hint path on large intervals, completing the forward matrix
    // (model_launch already covers both unique and repeat).
    g.bench_function("est_hint_interval/repeat", |b| {
        b.iter(|| {
            for it in &repeat {
                black_box(fx.idx.mem_search_from_hint(
                    black_box(&it.query),
                    black_box(it.hint),
                    true,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.bench_function("est_hint_match_len/repeat", |b| {
        b.iter(|| {
            for it in &repeat {
                black_box(fx.idx.mem_search_from_hint(
                    black_box(&it.query),
                    black_box(it.hint),
                    false,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.finish();
}

fn run_backward(c: &mut Criterion, fx: &Fixture) {
    let enc = PacEncoding::Unpacked;
    let bwork = build_backward_unique(fx);

    let mut g = c.benchmark_group("mem_search_backward");
    g.throughput(Throughput::Elements(bwork.len() as u64));

    g.bench_function("model_launch/unique", |b| {
        b.iter(|| {
            for it in &bwork {
                black_box(fx.idx.mem_search_backward(
                    black_box(it.sa_start),
                    black_box(it.occ),
                    black_box(it.anchor_len),
                    black_box(&it.read),
                    black_box(it.pivot),
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.bench_function("est_hint_interval/unique", |b| {
        b.iter(|| {
            for it in &bwork {
                black_box(fx.idx.mem_search_backward_from_hint(
                    black_box(&it.read),
                    black_box(it.pivot),
                    black_box(it.anchor_len),
                    black_box(it.hint),
                    true,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.bench_function("est_hint_match_len/unique", |b| {
        b.iter(|| {
            for it in &bwork {
                black_box(fx.idx.mem_search_backward_from_hint(
                    black_box(&it.read),
                    black_box(it.pivot),
                    black_box(it.anchor_len),
                    black_box(it.hint),
                    false,
                    &fx.pac,
                    enc,
                ));
            }
        });
    });
    g.finish();
}

fn bench_all(c: &mut Criterion) {
    // Build the index ONCE; every group shares it. (Avoids 6+ rebuilds.)
    let fx = build_fixture();
    eprintln!(
        "[mem_search_bench] sa_num={} l_pac={} log2(sa_num)={:.1} repeat=[{}..{}) ({}bp)",
        fx.idx.sa_num(),
        fx.idx.l_pac(),
        (fx.idx.sa_num() as f64).log2(),
        fx.repeat_start,
        fx.repeat_start + fx.repeat_len,
        fx.repeat_len,
    );
    // Touch MemMatch so the import is always used even if a group is edited out.
    let _ = MemMatch {
        match_len: 0,
        sa_start: 0,
        occ: 0,
    };
    run_forward(c, &fx);
    run_backward(c, &fx);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
