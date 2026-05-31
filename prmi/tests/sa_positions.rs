// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for `LearnedIndex::sa_positions`.

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use std::path::PathBuf;
use tempfile::TempDir;

fn deterministic_fasta(n_bases: usize, seed: u64) -> Vec<u8> {
    let mut s = String::from(">sa_positions_test\n");
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

fn build_test_index() -> (TempDir, PathBuf, LearnedIndex) {
    let dir = tempfile::tempdir().unwrap();
    let fa = dir.path().join("g.fa");
    std::fs::write(&fa, deterministic_fasta(4096, 0xFADE_BEEF)).unwrap();
    let prefix = dir.path().join("g.fa.prmi");
    build_sidecar(&fa, &prefix, Some(64), Default::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    (dir, prefix, idx)
}

#[test]
fn sa_positions_reads_full_sa_correctly() {
    let (_dir, _prefix, idx) = build_test_index();
    let sa_num = idx.sa_num();
    let mut out = vec![0u64; sa_num as usize];
    idx.sa_positions(0, &mut out).unwrap();
    // Compare each slot against the singleton accessor.
    for i in 0..sa_num {
        assert_eq!(out[i as usize], idx.sa_position_for(i), "mismatch at i={i}");
    }
}

#[test]
fn sa_positions_count_zero_no_writes() {
    let (_dir, _prefix, idx) = build_test_index();
    let mut out: Vec<u64> = vec![];
    idx.sa_positions(0, &mut out).unwrap();
    // Also at non-zero k.
    idx.sa_positions(idx.sa_num() - 1, &mut out).unwrap();
}

#[test]
fn sa_positions_single_position() {
    let (_dir, _prefix, idx) = build_test_index();
    let mut out = [0u64; 1];
    idx.sa_positions(0, &mut out).unwrap();
    assert_eq!(out[0], idx.sa_position_for(0));
}

#[test]
fn sa_positions_middle_slice() {
    let (_dir, _prefix, idx) = build_test_index();
    let sa_num = idx.sa_num();
    let start = sa_num / 3;
    let len = sa_num / 5;
    let mut out = vec![0u64; len as usize];
    idx.sa_positions(start, &mut out).unwrap();
    for (i, &p) in out.iter().enumerate() {
        assert_eq!(p, idx.sa_position_for(start + i as u64));
    }
}

#[test]
fn sa_positions_out_of_range_returns_err() {
    let (_dir, _prefix, idx) = build_test_index();
    let sa_num = idx.sa_num();
    let mut out = vec![0u64; 5];
    let err = idx.sa_positions(sa_num - 2, &mut out).unwrap_err();
    let detail = format!("{err:?}");
    assert!(
        detail.contains("exceeds sa_num") || detail.contains("range"),
        "got: {detail}"
    );
}

#[test]
fn sa_positions_concurrent_threads() {
    use std::sync::Arc;
    let (_dir, _prefix, idx) = build_test_index();
    let idx = Arc::new(idx);
    let sa_num = idx.sa_num();
    let serial: Vec<u64> = (0..sa_num).map(|i| idx.sa_position_for(i)).collect();

    let mut handles = vec![];
    for tid in 0u64..8 {
        let idx = Arc::clone(&idx);
        let serial = serial.clone();
        handles.push(std::thread::spawn(move || {
            let chunk = sa_num / 8;
            let start = tid * chunk;
            let end = if tid == 7 { sa_num } else { start + chunk };
            let mut out = vec![0u64; (end - start) as usize];
            idx.sa_positions(start, &mut out).unwrap();
            for (i, &p) in out.iter().enumerate() {
                assert_eq!(p, serial[(start + i as u64) as usize]);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
