// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for long-read seeding helpers:
//! `LearnedIndex::smem_range_long_read[_packed]` and
//! `prmi::encoding::minimizer_32mer`.

use prmi::encoding::minimizer_32mer;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// A synthetic 1 kb FASTA: ACGT repeating (256 periods × 4 = 1024 bases).
fn synth_fasta_1kb() -> Vec<u8> {
    let mut v = b">chr1\n".to_vec();
    let seq: Vec<u8> = (0..1024).map(|i| b"ACGT"[i % 4]).collect();
    for chunk in seq.chunks(60) {
        v.extend_from_slice(chunk);
        v.push(b'\n');
    }
    v
}

/// 2-bit unpacked bases for the 1 kb ACGT-repeat reference (values 0..=3).
fn synth_bases_1kb() -> Vec<u8> {
    (0..1024u32).map(|i| (i % 4) as u8).collect()
}

/// Pack 2-bit unpacked bases into BWA-MEME bntpac format (4 bases/byte, MSB-first).
fn pack_bases(bases: &[u8]) -> Vec<u8> {
    bases
        .chunks(4)
        .map(|c| {
            let mut b = 0u8;
            for (i, &base) in c.iter().enumerate() {
                b |= (base & 0x3) << (6 - 2 * i as u32);
            }
            b
        })
        .collect()
}

/// Build a sidecar and open it; returns the index.
fn open_test_idx(dir: &tempfile::TempDir, tag: &str) -> LearnedIndex {
    let fa = dir.path().join(format!("{tag}.fa"));
    std::fs::write(&fa, synth_fasta_1kb()).unwrap();
    let prefix = dir.path().join(format!("{tag}.fa.prmi"));
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    LearnedIndex::open(&prefix).unwrap()
}

// ---------------------------------------------------------------------------
// smem_range_long_read tests
// ---------------------------------------------------------------------------

/// Seed a 1 kb synthetic read at 32 evenly-spaced pivots (every 32 bases).
/// At least one pivot should hit a match in the matching reference.
#[test]
fn long_read_at_least_one_match_across_32_pivots() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_hits");
    let bases = synth_bases_1kb();

    // Pivots at offsets 0, 32, 64, …, 992 — all fit within 1024.
    let pivots: Vec<u64> = (0..32).map(|i| (i * 32) as u64).collect();
    let results = idx
        .smem_range_long_read(&bases, bases.len() as u64, &pivots, &bases)
        .unwrap();

    assert_eq!(results.len(), 32);
    let hits = results.iter().filter(|r| r.l > 0).count();
    assert!(
        hits > 0,
        "expected ≥1 match across 32 pivots on a matching reference"
    );
}

/// A pivot whose window runs off the end of the read should produce (0, 0, 0).
#[test]
fn long_read_pivot_past_end_returns_zero() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_past_end");
    let bases = synth_bases_1kb();

    // Pivot at offset 1000: 1000 + 32 = 1032 > 1024 → skip sentinel.
    // Pivot at offset 992: 992 + 32 = 1024 ≤ 1024 → valid.
    let pivots = vec![992u64, 1000u64];
    let results = idx
        .smem_range_long_read(&bases, bases.len() as u64, &pivots, &bases)
        .unwrap();

    assert_eq!(results.len(), 2);
    // The second pivot (1000) must be the skip sentinel.
    let sentinel = &results[1];
    assert_eq!(
        (sentinel.k, sentinel.l, sentinel.s),
        (0, 0, 0),
        "pivot past read end must return (0,0,0)"
    );
}

/// An empty pivot list must return an empty Vec without error.
#[test]
fn long_read_empty_pivot_list_returns_empty_vec() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_empty");
    let bases = synth_bases_1kb();

    let results = idx
        .smem_range_long_read(&bases, bases.len() as u64, &[], &bases)
        .unwrap();
    assert_eq!(results.len(), 0);
}

/// A single-pivot call must return a single result.
#[test]
fn long_read_single_pivot_returns_single_result() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_single");
    let bases = synth_bases_1kb();

    let pivots = vec![10u64]; // offset 10: 32-mer fits in 1024
    let results = idx
        .smem_range_long_read(&bases, bases.len() as u64, &pivots, &bases)
        .unwrap();

    assert_eq!(results.len(), 1);
}

// ---------------------------------------------------------------------------
// smem_range_long_read_packed tests
// ---------------------------------------------------------------------------

/// Packed variant: same pivots, should produce identical results to unpacked.
#[test]
fn long_read_packed_matches_unpacked() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_packed_match");
    let bases = synth_bases_1kb();
    let packed = pack_bases(&bases);
    let num_bases = bases.len() as u64;

    let pivots: Vec<u64> = (0..8).map(|i| (i * 64) as u64).collect();

    let unpacked_results = idx
        .smem_range_long_read(&bases, num_bases, &pivots, &bases)
        .unwrap();
    let packed_results = idx
        .smem_range_long_read_packed(&bases, num_bases, &pivots, &packed, num_bases)
        .unwrap();

    assert_eq!(unpacked_results.len(), packed_results.len());
    for (u, p) in unpacked_results.iter().zip(packed_results.iter()) {
        assert_eq!(
            (u.k, u.l, u.s),
            (p.k, p.l, p.s),
            "packed and unpacked long-read results must match"
        );
    }
}

// ---------------------------------------------------------------------------
// minimizer_32mer tests
// ---------------------------------------------------------------------------

/// A sequence shorter than 32 bases must return None.
#[test]
fn minimizer_32mer_returns_none_for_short_seq() {
    let bases: Vec<u8> = vec![0u8; 31];
    assert_eq!(minimizer_32mer(&bases), None);
    assert_eq!(minimizer_32mer(&[]), None);
    assert_eq!(minimizer_32mer(&[1, 2, 3]), None);
}

/// For exactly 32 bases there is one 32-mer; it must be returned at offset 0.
#[test]
fn minimizer_32mer_exact_32_bases_returns_offset_zero() {
    let bases: Vec<u8> = (0u8..32).map(|i| i % 4).collect();
    let (key, off) = minimizer_32mer(&bases).expect("expected Some for len=32");
    assert_eq!(off, 0, "only one 32-mer at offset 0");
    // Verify the key matches direct tokenization.
    let expected = prmi::encoding::tokenize_32mer(&bases, 32);
    assert_eq!(key, expected);
}

/// Inject a unique known-minimum 32-mer at a specific offset in a longer
/// sequence and verify `minimizer_32mer` finds it.
#[test]
fn minimizer_32mer_finds_known_minimum() {
    // Build a 100-base sequence of all T's (BASE_T = 3, key = u64::MAX).
    let mut bases = vec![3u8; 100];

    // Overwrite positions 40..72 with all A's (BASE_A = 0, key = 0x0000…0000).
    // This is guaranteed to be the lex-min 32-mer in the sequence.
    for b in &mut bases[40..72] {
        *b = 0;
    }

    let (key, off) = minimizer_32mer(&bases).expect("expected Some");
    assert_eq!(off, 40, "min 32-mer should start at offset 40");
    assert_eq!(key, 0u64, "all-A key should be 0");
}

/// When multiple 32-mers share the same key, the leftmost wins.
#[test]
fn minimizer_32mer_leftmost_tiebreak() {
    // All-A sequence: every 32-mer has key 0; leftmost offset should win.
    let bases = vec![0u8; 64];
    let (key, off) = minimizer_32mer(&bases).expect("expected Some");
    assert_eq!(off, 0, "leftmost offset wins on tie");
    assert_eq!(key, 0u64);
}
