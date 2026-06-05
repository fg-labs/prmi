// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Criterion benchmarks for the spectrum primitives: forward_spectrum,
//! backward_spectrum, forward_spectrum_batch (per-query loop), and
//! sa_positions_strided. Also includes a high-occurrence anchor case
//! (tandem-repeat region) that represents the D15 scenario targeted by T7.
//!
//! Setup (sidecar build + pac preparation) is performed OUTSIDE the timed
//! loops — only the query primitives themselves are measured.
//!
//! # Reference size
//!
//! Default: 500 kbp deterministic synthetic ACGT + an embedded tandem-repeat
//! block (512 × "ACGT" = 2 048 bp) for the high-occ case. This builds in
//! ~2–5 s on a laptop (acceptable for a bench setup phase). To get production
//! numbers against a real chromosome, change `SYNTH_BASES` to 0 and point
//! `real_ref_fa()` at e.g. `~/work/references/hg38/chr21.fa`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use tempfile::TempDir;

// ── Reference size ────────────────────────────────────────────────────────────

/// Number of ACGT bases in the random backbone. Set to 0 to skip the random
/// backbone (useful when REPEAT_BLOCK alone is sufficient for testing).
/// To bench against a real chromosome, set this to 0 and provide a real FASTA
/// via `real_ref_fa()` instead of the synthetic generator.
const SYNTH_BASES: usize = 500_000;

/// A tandem repeat inserted into the synthetic reference. This creates a
/// region of high occurrence count (large SA interval), the D15 scenario.
/// 512 repetitions × 4 bp = 2 048 bp of tandem "ACGT" repeats.
/// Stored as ASCII for FASTA generation; the QUERY uses 0..=3 encoding.
const REPEAT_UNIT_ASCII: &[u8] = b"ACGT";
/// Same unit in 0..=3 encoding for query construction.
const REPEAT_UNIT_ENC: &[u8] = &[0, 1, 2, 3];
const REPEAT_COUNT: usize = 512;

/// Query length used for the forward/backward spectrum benchmarks.
const QUERY_LEN: usize = 75;

/// Number of queries in the corpus (forward/backward batches).
const CORPUS_SIZE: usize = 256;

/// Number of positions fetched in the strided SA bench.
const STRIDED_FETCH_N: usize = 64;

// ── FASTA / PAC helpers ───────────────────────────────────────────────────────

/// Build a deterministic synthetic FASTA: a random backbone with a tandem
/// repeat block inserted at the midpoint.
fn synthetic_fasta(seed: u64) -> Vec<u8> {
    let mut s = String::from(">synth\n");
    let half = SYNTH_BASES / 2;
    let mut x = seed;
    // First half of random backbone.
    for _ in 0..half {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(REPEAT_UNIT_ASCII[(x & 3) as usize] as char);
    }
    // Tandem repeat block (high-occ region).
    for _ in 0..REPEAT_COUNT {
        for &b in REPEAT_UNIT_ASCII {
            s.push(b as char);
        }
    }
    // Second half of random backbone.
    for _ in half..SYNTH_BASES {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(REPEAT_UNIT_ASCII[(x & 3) as usize] as char);
    }
    s.push('\n');
    s.into_bytes()
}

/// Build a sidecar over the synthetic reference. Returns `(TempDir, pac_bases,
/// l_pac, enc, LearnedIndex)`. `TempDir` keeps the files alive.
fn build_bench_index() -> (TempDir, Vec<u8>, u64, PacEncoding, LearnedIndex) {
    let dir = tempfile::tempdir().expect("tempdir");
    let fa_path = dir.path().join("synth.fa");
    let prefix = dir.path().join("synth.prmi");
    let fasta_bytes = synthetic_fasta(0xBEEF_C0DE_0001);
    std::fs::write(&fa_path, &fasta_bytes).expect("write fasta");
    // Build in mode 2 (stored 32-mer keys) so the spectrum query path uses the
    // stored-key compare fast path — this is the configuration the key-skip
    // optimization targets and benches.
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_with_config(&fa_path, &prefix, None, Default::default(), 0, Some(cfg))
        .expect("build sidecar");
    let idx = LearnedIndex::open(&prefix).expect("open index");

    // Build an unpacked pac from the FASTA bases (one byte per base, 0..=3).
    // The sidecar was also built from this FASTA (N→A mapping matches the
    // train path), so the unpacked pac is consistent with the SA.
    let total_bases = SYNTH_BASES + REPEAT_UNIT_ASCII.len() * REPEAT_COUNT;
    let l_pac = total_bases as u64;
    let pac = fasta_to_unpacked_pac(&fasta_bytes, total_bases);
    let enc = PacEncoding::Unpacked;
    (dir, pac, l_pac, enc, idx)
}

/// Extract base bytes (0..=3) from a single-record FASTA byte slice.
/// Skips the header line and newlines; maps ACGT → 0..=3, others → 0 (A).
fn fasta_to_unpacked_pac(fasta: &[u8], expected_bases: usize) -> Vec<u8> {
    let mut bases = Vec::with_capacity(expected_bases);
    let mut in_header = true;
    for &b in fasta {
        match b {
            b'>' => {
                in_header = true;
            }
            b'\n' => {
                in_header = false;
            }
            _ if !in_header => {
                bases.push(match b {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 0,
                });
            }
            _ => {}
        }
    }
    bases
}

// ── Corpus construction ───────────────────────────────────────────────────────

/// Build a corpus of `CORPUS_SIZE` queries lifted from random positions in the
/// reference pac. Queries are guaranteed non-empty and length `QUERY_LEN`.
fn build_query_corpus(pac: &[u8], l_pac: u64) -> Vec<Vec<u8>> {
    let max_start = l_pac.saturating_sub(QUERY_LEN as u64) as usize;
    if max_start == 0 {
        panic!("reference too short for QUERY_LEN={QUERY_LEN}");
    }
    let mut queries = Vec::with_capacity(CORPUS_SIZE);
    let mut x = 0xDEAD_BEEF_1234_5678u64;
    for _ in 0..CORPUS_SIZE {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = (x as usize) % max_start;
        queries.push(pac[start..start + QUERY_LEN].to_vec());
    }
    queries
}

/// Build a query from the tandem-repeat block (high-occ case).
/// Returns the repeat-unit query of length `QUERY_LEN` (repeating 0,1,2,3).
/// Uses 0..=3 encoding (matching the unpacked pac) so forward_spectrum can
/// match it against the embedded tandem-repeat region.
fn build_repeat_query() -> Vec<u8> {
    REPEAT_UNIT_ENC
        .iter()
        .cycle()
        .take(QUERY_LEN)
        .copied()
        .collect()
}

// ── Benchmark functions ───────────────────────────────────────────────────────

fn bench_forward_spectrum(c: &mut Criterion) {
    let (_dir, pac, l_pac, enc, idx) = build_bench_index();
    let queries = build_query_corpus(&pac, l_pac);

    let mut group = c.benchmark_group("forward_spectrum");
    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("corpus_256x75bp", |b| {
        b.iter(|| {
            for q in &queries {
                let steps = idx.forward_spectrum(black_box(q), black_box(&pac), enc);
                black_box(steps);
            }
        });
    });
    group.finish();
}

fn bench_backward_spectrum(c: &mut Criterion) {
    let (_dir, pac, l_pac, enc, idx) = build_bench_index();
    let queries = build_query_corpus(&pac, l_pac);

    // Run forward_spectrum once in setup to collect anchors. The pivot is placed
    // mid-query so the right-anchored span `q[pivot..pivot+anchor_len)` lies within the
    // query AND leaves bases to the left for backward extension; the anchor is derived
    // from `q[pivot..]` (backward_spectrum re-derives its interval from that span).
    let anchors: Vec<_> = queries
        .iter()
        .filter_map(|q| {
            let pivot = q.len() / 2;
            let steps = idx.forward_spectrum(&q[pivot..], &pac, enc);
            // Use the deepest step (last, highest match_len) as the anchor.
            steps.last().copied().map(|s| (s, q.clone(), pivot))
        })
        .collect();

    let mut group = c.benchmark_group("backward_spectrum");
    group.throughput(Throughput::Elements(anchors.len() as u64));
    group.bench_function("corpus_256x75bp", |b| {
        b.iter(|| {
            for (step, q, pivot) in &anchors {
                let steps = idx.backward_spectrum(
                    black_box(step.sa_start),
                    black_box(step.occ_count),
                    black_box(step.match_len),
                    black_box(q),
                    black_box(*pivot),
                    black_box(&pac),
                    enc,
                );
                black_box(steps);
            }
        });
    });
    group.finish();
}

fn bench_forward_spectrum_batch(c: &mut Criterion) {
    let (_dir, pac, l_pac, enc, idx) = build_bench_index();
    let queries = build_query_corpus(&pac, l_pac);

    // Bench a batch of 64 queries (representative aligner chunk size).
    let batch: Vec<_> = queries.iter().take(64).collect();

    let mut group = c.benchmark_group("forward_spectrum_batch");
    group.throughput(Throughput::Elements(batch.len() as u64));
    group.bench_with_input(BenchmarkId::new("batch", 64), &batch, |b, batch| {
        b.iter(|| {
            for q in batch.iter() {
                let steps = idx.forward_spectrum(black_box(*q), black_box(&pac), enc);
                black_box(steps);
            }
        });
    });
    group.finish();
}

fn bench_sa_positions_strided(c: &mut Criterion) {
    let (_dir, pac, l_pac, enc, idx) = build_bench_index();
    // Build an anchor with moderate occ to get a realistic interval.
    let queries = build_query_corpus(&pac, l_pac);
    let anchor = queries
        .iter()
        .filter_map(|q| {
            let steps = idx.forward_spectrum(q, &pac, enc);
            steps.last().copied()
        })
        // Pick the anchor with the largest occ_count (widest interval).
        .max_by_key(|s| s.occ_count);

    let anchor = anchor.expect("at least one anchor");
    let sa_start = anchor.sa_start;
    let sa_num = idx.sa_num();

    // Clamp fetch count to what the interval (and SA) can supply.
    let available = (sa_num.saturating_sub(sa_start)).min(anchor.occ_count) as usize;
    let fetch_n = STRIDED_FETCH_N.min(available).max(1);
    // step = 1 (contiguous) — representative for low-occ anchors.
    let step = 1u64;

    let mut out = vec![0u64; fetch_n];

    let mut group = c.benchmark_group("sa_positions_strided");
    group.throughput(Throughput::Elements(fetch_n as u64));
    group.bench_function("contiguous_64", |b| {
        b.iter(|| {
            idx.sa_positions_strided(black_box(sa_start), black_box(step), black_box(&mut out))
                .expect("strided fetch ok");
            black_box(&out);
        });
    });
    group.finish();
}

fn bench_high_occ_backward(c: &mut Criterion) {
    let (_dir, pac, _l_pac, enc, idx) = build_bench_index();

    // Query into the tandem-repeat block: high occ_count, large interval. The read is
    // the repeat-unit query; the pivot is placed mid-read so the right-anchored span
    // `read[pivot..pivot+anchor_len)` lies WITHIN the read AND there is room to the left
    // for backward extension. (backward_spectrum re-derives the interval from this span,
    // so the anchor must actually be present at `read[pivot..]`.)
    let repeat_query = build_repeat_query();
    let pivot = repeat_query.len() / 2;
    let pivot_query = repeat_query[pivot..].to_vec();
    let fwd_steps = idx.forward_spectrum(&pivot_query, &pac, enc);

    // Use the widest (first) forward step as the anchor — highest occ_count.
    let anchor = fwd_steps
        .iter()
        .max_by_key(|s| s.occ_count)
        .copied()
        .expect("forward spectrum non-empty on repeat query");

    eprintln!(
        "[spectrum_bench] high-occ anchor: sa_start={} occ_count={} match_len={} pivot={pivot}",
        anchor.sa_start, anchor.occ_count, anchor.match_len
    );

    let mut group = c.benchmark_group("high_occ_backward");
    group.sample_size(20); // high-occ backward is expensive; fewer samples ok
    group.bench_function("repeat_anchor", |b| {
        b.iter(|| {
            let steps = idx.backward_spectrum(
                black_box(anchor.sa_start),
                black_box(anchor.occ_count),
                black_box(anchor.match_len),
                black_box(&repeat_query),
                black_box(pivot),
                black_box(&pac),
                enc,
            );
            black_box(steps);
        });
    });
    group.finish();
}

// ── Criterion wiring ──────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_forward_spectrum,
    bench_backward_spectrum,
    bench_forward_spectrum_batch,
    bench_sa_positions_strided,
    bench_high_occ_backward,
);
criterion_main!(benches);
