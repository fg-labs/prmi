// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use tempfile::tempdir;

fn synth_fasta() -> Vec<u8> {
    let mut v = b">chr1\n".to_vec();
    for _ in 0..64 {
        v.extend_from_slice(b"ACGT");
    }
    v.push(b'\n');
    v
}

#[test]
fn smem_range_finds_known_kmer() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("s.fa");
    std::fs::write(&fa, synth_fasta()).unwrap();
    let prefix = dir.path().join("s.fa.prmi");
    build_sidecar(&fa, &prefix, 16).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    // 2-bit-coded reference: ACGT repeating, 256 bases (length matches FASTA).
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();

    // 32-mer starting at offset 10.
    let query: Vec<u8> = bases[10..42].to_vec();
    let (k, l, s) = idx.smem_range(&query, &bases).unwrap();
    assert!(l > 0, "expected at least one match (k={k}, l={l}, s={s})");
    assert_eq!(s, 32, "expected 32-base exact match");
}

#[test]
fn smem_range_returns_empty_for_impossible_query() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("e.fa");
    std::fs::write(&fa, synth_fasta()).unwrap();
    let prefix = dir.path().join("e.fa.prmi");
    build_sidecar(&fa, &prefix, 16).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    // 32 A's: doesn't occur in ACGTACGT... pattern.
    let query = vec![0u8; 32];
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let (_k, l, _s) = idx.smem_range(&query, &bases).unwrap();
    assert_eq!(l, 0);
}
