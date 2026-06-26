// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Criterion benchmarks for the cheap O(1)/O(log) FFI primitives the bwa-meme
//! consumer calls outside the main search, one or more times per pivot:
//!
//! - `prmi_isa_at` → [`LearnedIndex::isa_at`] (inverse-SA, the launch-hint source
//!   for the `est_hint` path).
//! - `prmi_sa_positions` → [`LearnedIndex::sa_positions`] (resolve a contiguous SA
//!   interval → genomic positions).
//! - `prmi_reverse_complement_key` → [`prmi::encoding::reverse_complement_key`]
//!   (both-strand lookup, the word-level bit-swap).
//! - `prmi_reverse_complement_2bit` → [`prmi::encoding::reverse_complement_2bit`].
//! - `prmi_tokenize_32mer` → [`prmi::encoding::tokenize_32mer`].
//!
//! (`prmi_sa_positions_strided` and `prmi_lookup` are benched in
//! `spectrum_bench.rs` / `lookup_bench.rs` respectively.)
//!
//! Each primitive is too cheap to time one call at a time (criterion's per-iter
//! overhead would dominate), so every bench loops over a fixed corpus of inputs
//! and reports per-element throughput. The index-backed primitives use random
//! access into the SA to exercise realistic cache behavior; reference size is
//! env-tunable via `PRMI_BENCH_REFLEN` (default 2_000_000).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prmi::encoding::{reverse_complement_2bit, reverse_complement_key, tokenize_32mer, KMER_LEN};
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use tempfile::TempDir;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Corpus size for the loop-batched primitive benches.
const N: usize = 1024;

/// Deterministic ACGT bases (0..=3) via a PCG-style LCG.
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

/// Build a mode-2 `--with-isa` sidecar over a synthetic reference and return the
/// tempdir + index. `.isa` is required for the `isa_at` bench.
fn build_index() -> (TempDir, LearnedIndex) {
    let ref_len = env_usize("PRMI_BENCH_REFLEN", 2_000_000);
    let bases = synth_bases(ref_len, 0x2545_F491_4F6C_DD1D);
    let dir = tempfile::tempdir().expect("tempdir");
    let fa = dir.path().join("ref.fa");
    {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&fa).unwrap());
        writeln!(w, ">bench").unwrap();
        let alphabet = [b'A', b'C', b'G', b'T'];
        for chunk in bases.chunks(60) {
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
    assert!(idx.has_isa(), "primitives bench fixture needs .isa");
    (dir, idx)
}

// ── Index-backed primitives ─────────────────────────────────────────────────

fn bench_isa_at(c: &mut Criterion, idx: &LearnedIndex) {
    let sa_num = idx.sa_num();
    // A scattered set of reference positions (random access into the inverse SA).
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let refpos: Vec<u64> = (0..N)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            x % sa_num
        })
        .collect();

    let mut g = c.benchmark_group("isa_at");
    g.throughput(Throughput::Elements(refpos.len() as u64));
    g.bench_function("random", |b| {
        b.iter(|| {
            for &p in &refpos {
                black_box(idx.isa_at(black_box(p)));
            }
        });
    });
    g.finish();
}

fn bench_sa_positions(c: &mut Criterion, idx: &LearnedIndex) {
    let sa_num = idx.sa_num();
    // `sa_position_for`: scattered single lookups (the per-element cost the
    // consumer pays when resolving a refpos one at a time).
    let mut x = 0xD1B5_4A32_D192_ED03u64;
    let single: Vec<u64> = (0..N)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            x % sa_num
        })
        .collect();

    let mut g = c.benchmark_group("sa_positions");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("sa_position_for/scattered", |b| {
        b.iter(|| {
            for &i in &single {
                black_box(idx.sa_position_for(black_box(i)));
            }
        });
    });

    // `sa_positions`: resolve a contiguous block (an SA interval) in one call —
    // the shape used to materialize all occurrences of a seed.
    for &block in &[16usize, 256usize] {
        if sa_num < block as u64 {
            continue; // reference too small for this block size (e.g. tiny PRMI_BENCH_REFLEN)
        }
        let start = sa_num.saturating_sub(block as u64) / 3; // arbitrary in-range start
        let mut out = vec![0u64; block];
        g.throughput(Throughput::Elements(block as u64));
        g.bench_with_input(BenchmarkId::new("block", block), &block, |b, _| {
            b.iter(|| {
                idx.sa_positions(black_box(start), black_box(&mut out))
                    .expect("in range");
                black_box(&out);
            });
        });
    }
    g.finish();
}

// ── Encoding primitives (no index needed) ───────────────────────────────────

fn bench_reverse_complement_key(c: &mut Criterion) {
    // A corpus of random 64-bit words treated as packed 32-mer keys.
    let mut x = 0x0123_4567_89AB_CDEFu64;
    let keys: Vec<u64> = (0..N)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            x
        })
        .collect();

    let mut g = c.benchmark_group("reverse_complement_key");
    g.throughput(Throughput::Elements(keys.len() as u64));
    for &len in &[16usize, KMER_LEN] {
        g.bench_with_input(BenchmarkId::new("len", len), &len, |b, &len| {
            b.iter(|| {
                for &k in &keys {
                    black_box(reverse_complement_key(black_box(k), len));
                }
            });
        });
    }
    g.finish();
}

fn bench_reverse_complement_2bit(c: &mut Criterion) {
    let mut g = c.benchmark_group("reverse_complement_2bit");
    for &len in &[16usize, 32usize, 64usize] {
        // One representative base slice per length (values 0..=3).
        let bases = synth_bases(len, 0xABCD_1234_5678_9F01 ^ len as u64);
        g.throughput(Throughput::Elements(len as u64));
        g.bench_with_input(BenchmarkId::new("len", len), &bases, |b, bases| {
            b.iter(|| {
                black_box(reverse_complement_2bit(black_box(bases)));
            });
        });
    }
    g.finish();
}

fn bench_tokenize_32mer(c: &mut Criterion) {
    // A corpus of 32-base windows over a synthetic backbone.
    let backbone = synth_bases(N + KMER_LEN, 0x5DEE_CE66_D7A0_0DAFu64);
    let windows: Vec<&[u8]> = (0..N).map(|i| &backbone[i..i + KMER_LEN]).collect();

    let mut g = c.benchmark_group("tokenize_32mer");
    g.throughput(Throughput::Elements(windows.len() as u64));
    g.bench_function("len32", |b| {
        b.iter(|| {
            for w in &windows {
                black_box(tokenize_32mer(black_box(w), KMER_LEN));
            }
        });
    });
    g.finish();
}

// ── next_n: O(rlen²) → O(1) per-read distance precompute ───────────────────

// Both kernels are `#[inline(never)]` so the comparison is symmetric: each is
// measured across a real call boundary, and the optimizer cannot collapse the
// O(rlen²) baseline (e.g. by CSE-ing the repeated scans) into something faster
// than the per-pivot `fwd_qlen` calls it stands in for. The two prmi fns being
// mirrored here are private, so the bench reimplements rather than exports them.

/// Inline reimplementation of the old `fwd_qlen(read, p)` forward scan —
/// the baseline (O(rlen²) over all pivots).
#[inline(never)]
fn next_n_scan(read: &[u8], p: usize) -> usize {
    let rlen = read.len();
    for (i, &b) in read.iter().enumerate().skip(p) {
        if b >= 4 {
            return i - p;
        }
    }
    rlen - p
}

/// Inline reimplementation of `fill_next_n` (descending recurrence).
#[inline(never)]
fn fill_next_n_inline(read: &[u8], out: &mut Vec<u32>) {
    let rlen = read.len();
    out.clear();
    out.resize(rlen + 1, 0);
    for i in (0..rlen).rev() {
        out[i] = if read[i] >= 4 { 0 } else { out[i + 1] + 1 };
    }
}

fn bench_next_n(c: &mut Criterion) {
    // A corpus of reads with scattered ambiguous (>=4) bases. `v` is the top 3
    // bits (0..=7) and `v == 0` marks an N, so ~1 in 8 (12.5%) of bases are
    // ambiguous. This is conservative: a high N-density makes each forward scan
    // terminate early, shrinking the O(rlen²) baseline — realistic low-N reads
    // (long N-free runs, full-length scans) show a LARGER speedup than measured.
    let read_len = 150usize;
    let num_reads = N;
    let mut reads: Vec<Vec<u8>> = Vec::with_capacity(num_reads);
    let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
    for _ in 0..num_reads {
        let read: Vec<u8> = (0..read_len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = (state >> 61) as u8;
                // v in 0..=7; v == 0 (~1 in 8) is ambiguous (4), rest are 0..=3
                if v == 0 {
                    4u8
                } else {
                    v & 3
                }
            })
            .collect();
        reads.push(read);
    }

    let total_pivots: u64 = reads.iter().map(|r| r.len() as u64).sum();

    let mut g = c.benchmark_group("next_n");
    g.throughput(Throughput::Elements(total_pivots));

    // Baseline: O(rlen²) — repeated forward scan for every pivot.
    g.bench_function("scan_all_pivots", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for read in &reads {
                for p in 0..read.len() {
                    sum = sum.wrapping_add(next_n_scan(black_box(read), black_box(p)) as u64);
                }
            }
            black_box(sum)
        });
    });

    // Optimized: fill once per read (O(rlen)), then index O(1).
    let mut next_n: Vec<u32> = Vec::new();
    g.bench_function("fill_once_index", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for read in &reads {
                fill_next_n_inline(black_box(read), &mut next_n);
                for p in 0..read.len() {
                    sum = sum.wrapping_add(next_n[p] as u64);
                }
            }
            black_box(sum)
        });
    });

    g.finish();
}

// ── unpack_packed_forward: before/after middle-loop shapes ──────────────────
//
// `unpack_packed_forward` is private, so we inline-reimplement the two middle
// loop shapes (before / after PR4) and bench them over a packed-encoded corpus
// that is long enough (≥ 128 bases) to let the 32-base word loop dominate.
// This mirrors the pattern used by `bench_next_n` above.
//
// The LUT is the same 256-entry table as in spectrum.rs (duplicated here so
// the bench is self-contained). Each bench drives 32 packed words (32 × 32 =
// 1 024 bases per decode call) across a corpus of 256 encoded sequences.

/// 256-entry LUT: byte → four 2-bit bases as a little-endian u32.
/// Bit layout: base0 in bits 6-7 (MSB), base3 in bits 0-1 (LSB) of the source
/// byte. Entry `b` expands to `[base0, base1, base2, base3]` via `.to_le_bytes()`.
const BENCH_UNPACK_LUT: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut b = 0usize;
    while b < 256 {
        let byte = b as u32;
        t[b] = ((byte >> 6) & 0x3)
            | (((byte >> 4) & 0x3) << 8)
            | (((byte >> 2) & 0x3) << 16)
            | ((byte & 0x3) << 24);
        b += 1;
    }
    t
};

/// Corpus of packed sequences. Each is `PAC_BYTES` bytes = `PAC_BYTES * 4` bases.
const PAC_BYTES: usize = 256; // 1 024 bases = 32 full 32-base word-loop iterations
const CORPUS_COUNT: usize = 256;

fn make_packed_corpus() -> Vec<Vec<u8>> {
    let mut state = 0xFEED_FACE_DEAD_BEEFu64;
    (0..CORPUS_COUNT)
        .map(|_| {
            (0..PAC_BYTES)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 56) as u8
                })
                .collect()
        })
        .collect()
}

/// Before (PR3 baseline): 9 bounds checks per 32-base step — try_into().unwrap()
/// + 8× i+4*j slice indexing. Driven by the original while loop.
#[inline(never)]
fn unpack_middle_before(pac: &[u8], out: &mut [u8]) {
    let n = out.len();
    let mut i = 0usize;
    let mut pos = 0usize;
    while i + 32 <= n {
        let base = pos >> 2;
        let word = u64::from_le_bytes(pac[base..base + 8].try_into().unwrap());
        for j in 0..8 {
            let byte = (word >> (8 * j)) as u8;
            out[i + 4 * j..i + 4 * j + 4]
                .copy_from_slice(&BENCH_UNPACK_LUT[byte as usize].to_le_bytes());
        }
        i += 32;
        pos += 32;
    }
}

/// After (PR4): bounds checked once at the slice split via chunks_exact.
#[inline(never)]
fn unpack_middle_after(pac: &[u8], out: &mut [u8]) {
    let n = out.len();
    let i = 0usize;
    let pos = 0usize;
    let mid_words = (n - i) / 32;
    if mid_words > 0 {
        let src_byte = pos >> 2;
        let out_mid = &mut out[i..i + mid_words * 32];
        let src_mid = &pac[src_byte..src_byte + mid_words * 8];
        for (chunk_out, w) in out_mid.chunks_exact_mut(32).zip(src_mid.chunks_exact(8)) {
            let word = u64::from_le_bytes(w.try_into().unwrap());
            for (k, slot) in chunk_out.chunks_exact_mut(4).enumerate() {
                slot.copy_from_slice(
                    &BENCH_UNPACK_LUT[(word >> (8 * k)) as u8 as usize].to_le_bytes(),
                );
            }
        }
    }
}

fn bench_unpack_packed(c: &mut Criterion) {
    let corpus = make_packed_corpus();
    let out_len = PAC_BYTES * 4; // 1 024 unpacked bases per call
    let mut out = vec![0u8; out_len];
    let total_bases: u64 = (corpus.len() * out_len) as u64;

    let mut g = c.benchmark_group("unpack_packed_middle");
    g.throughput(Throughput::Elements(total_bases));

    g.bench_function("before_while_loop", |b| {
        b.iter(|| {
            for pac in &corpus {
                unpack_middle_before(black_box(pac), black_box(&mut out));
            }
            black_box(&out);
        });
    });

    g.bench_function("after_chunks_exact", |b| {
        b.iter(|| {
            for pac in &corpus {
                unpack_middle_after(black_box(pac), black_box(&mut out));
            }
            black_box(&out);
        });
    });

    g.finish();
}

// ── bit_at reduction: before (%) vs after (Lemire) — Bb2 ───────────────────
//
// `bit_at` is private, so we inline both shapes here. Each bench drives
// N=1024 `(h1, h2, i, num_bits)` probes; throughput is reported in probes.
// The before/after bench is the Bb2 targeted comparison.

/// `bit_at` baseline: combined % num_bits (true 64-bit division).
#[inline(never)]
fn bit_at_mod(h1: u64, h2: u64, i: u32, num_bits: u64) -> u64 {
    let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
    combined % num_bits
}

/// `bit_at` Lemire: (combined * num_bits) >> 64 (no division).
#[inline(never)]
fn bit_at_lemire(h1: u64, h2: u64, i: u32, num_bits: u64) -> u64 {
    let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
    (((combined as u128) * (num_bits as u128)) >> 64) as u64
}

fn bench_bit_at(c: &mut Criterion) {
    // A corpus of N (h1, h2, i) triples with a representative num_bits.
    // Use a realistic num_bits: BloomParams::for_keys(50_000, 0.01) ≈ 479,232 bits.
    let num_bits: u64 = 479_296; // multiple of 64, near optimal for 50k keys at 1%
    let mut state = 0xABCD_EF01_2345_6789u64;
    let probes: Vec<(u64, u64, u32)> = (0..N)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let h1 = state;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let h2 = state | 1;
            let i = (state >> 60) as u32; // 0..15
            (h1, h2, i)
        })
        .collect();

    let mut g = c.benchmark_group("bit_at_reduction");
    g.throughput(Throughput::Elements(probes.len() as u64));

    // Before: true 64-bit division (Bb2 baseline).
    g.bench_function("before_mod", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(h1, h2, i) in &probes {
                acc = acc.wrapping_add(bit_at_mod(
                    black_box(h1),
                    black_box(h2),
                    black_box(i),
                    black_box(num_bits),
                ));
            }
            black_box(acc)
        });
    });

    // After: Lemire multiply-shift (no division).
    g.bench_function("after_lemire", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(h1, h2, i) in &probes {
                acc = acc.wrapping_add(bit_at_lemire(
                    black_box(h1),
                    black_box(h2),
                    black_box(i),
                    black_box(num_bits),
                ));
            }
            black_box(acc)
        });
    });

    g.finish();
}

fn bench_all(c: &mut Criterion) {
    let (_dir, idx) = build_index();
    eprintln!(
        "[primitives_bench] sa_num={} l_pac={} has_isa={}",
        idx.sa_num(),
        idx.l_pac(),
        idx.has_isa(),
    );
    bench_isa_at(c, &idx);
    bench_sa_positions(c, &idx);
    bench_reverse_complement_key(c);
    bench_reverse_complement_2bit(c);
    bench_tokenize_32mer(c);
    bench_next_n(c);
    bench_unpack_packed(c);
    bench_bit_at(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
