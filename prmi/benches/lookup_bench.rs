// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Throughput benchmarks for the §4.4 lookup. Measures single-key lookup
//! latency on a built sidecar; the result is the per-call cost a downstream
//! aligner (e.g. bwa-mem3) will pay.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use std::path::PathBuf;
use tempfile::TempDir;

fn deterministic_fasta(n_bases: usize, seed: u64) -> Vec<u8> {
    let mut s = String::from(">bench\n");
    let mut x = seed;
    for _ in 0..n_bases {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(b"ACGT"[(x & 3) as usize] as char);
    }
    s.push('\n');
    s.into_bytes()
}

fn build_index(n_bases: usize) -> (TempDir, PathBuf, LearnedIndex) {
    let dir = tempfile::tempdir().unwrap();
    let fa: PathBuf = dir.path().join("bench.fa");
    std::fs::write(&fa, deterministic_fasta(n_bases, 0x000B_00B5_C0DE)).unwrap();
    let prefix = dir.path().join("bench.fa.prmi");
    build_sidecar(&fa, &prefix, None).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    (dir, fa, idx)
}

fn bench_lookup(c: &mut Criterion) {
    for &n in &[5386usize, 50_000usize] {
        let (_dir, fa, idx) = build_index(n);
        let sa_num = idx.sa_num();
        // Read the 2-bit bases once outside the benchmark loop.
        let (bases, _) = prmi::fasta::fasta_file_to_2bit(&fa).unwrap();
        // Pre-compute a deterministic set of 1024 keys to measure against.
        let mut keys = Vec::with_capacity(1024);
        for i in 0..1024u64 {
            let sa_idx = (i * sa_num.max(1) / 1024) % sa_num;
            let sa_pos = idx.sa_position_for(sa_idx);
            let avail = bases.len().saturating_sub(sa_pos as usize).min(32);
            keys.push(prmi::encoding::tokenize_32mer(
                &bases[sa_pos as usize..sa_pos as usize + avail],
                avail,
            ));
        }

        let mut group = c.benchmark_group(format!("lookup_{}bp", n));
        group.throughput(Throughput::Elements(keys.len() as u64));
        group.bench_function("hot", |b| {
            b.iter(|| {
                for &k in &keys {
                    let (pred, err) = idx.lookup(black_box(k));
                    black_box((pred, err));
                }
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
