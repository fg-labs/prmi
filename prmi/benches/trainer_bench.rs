// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! End-to-end build_sidecar wall time at increasing reference scales.
//! Quantitative baseline for the trainer's cost.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use prmi::train::build_sidecar;
use std::path::PathBuf;

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

fn bench_trainer(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_sidecar");
    group.sample_size(10); // build is expensive; 10 samples is enough
    for &n in &[5386usize, 50_000usize, 500_000usize] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let dir = tempfile::tempdir().unwrap();
                    let fa: PathBuf = dir.path().join("bench.fa");
                    std::fs::write(&fa, deterministic_fasta(n, 0x00C0_FFEE)).unwrap();
                    let prefix = dir.path().join("bench.fa.prmi");
                    (dir, fa, prefix)
                },
                |(dir, fa, prefix)| {
                    build_sidecar(&fa, &prefix, None, Default::default(), 0).unwrap();
                    black_box(dir);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_trainer);
criterion_main!(benches);
