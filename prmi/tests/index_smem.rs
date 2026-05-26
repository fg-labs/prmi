// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::smem::PacEncoding;
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
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
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
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    // 32 A's: doesn't occur in ACGTACGT... pattern.
    let query = vec![0u8; 32];
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let (_k, l, _s) = idx.smem_range(&query, &bases).unwrap();
    assert_eq!(l, 0);
}

/// Helper: open a test sidecar quickly.
fn open_test_idx(dir: &tempfile::TempDir, tag: &str) -> LearnedIndex {
    let fa = dir.path().join(format!("{tag}.fa"));
    std::fs::write(&fa, synth_fasta()).unwrap();
    let prefix = dir.path().join(format!("{tag}.fa.prmi"));
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    LearnedIndex::open(&prefix).unwrap()
}

fn pack_bases_for_test(bases: &[u8]) -> Vec<u8> {
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

#[test]
fn smem_range_rejects_query_shorter_than_32() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "q_short");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let short_query: Vec<u8> = bases[0..16].to_vec();
    assert!(
        idx.smem_range(&short_query, &bases).is_err(),
        "query shorter than 32 bases should return Err"
    );
}

#[test]
fn smem_range_rejects_query_longer_than_32() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "q_long");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let long_query: Vec<u8> = bases[0..36].to_vec();
    assert!(
        idx.smem_range(&long_query, &bases).is_err(),
        "query longer than 32 bases should return Err"
    );
}

#[test]
fn smem_range_packed_rejects_query_shorter_than_32() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "p_short");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let packed = pack_bases_for_test(&bases);
    let short_query: Vec<u8> = bases[0..16].to_vec();
    assert!(
        idx.smem_range_packed(&short_query, &packed, 256).is_err(),
        "packed query shorter than 32 bases should return Err"
    );
}

#[test]
fn smem_range_packed_rejects_query_longer_than_32() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "p_long");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let packed = pack_bases_for_test(&bases);
    let long_query: Vec<u8> = bases[0..36].to_vec();
    assert!(
        idx.smem_range_packed(&long_query, &packed, 256).is_err(),
        "packed query longer than 32 bases should return Err"
    );
}

// A packed `pac` shorter than `ceil(num_bases / 4)` must be rejected up front
// rather than panicking on an out-of-bounds byte access inside the decoder.
// These guard all three packed entry points.

#[test]
fn smem_range_packed_rejects_short_pac() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "p_short_pac");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let query: Vec<u8> = bases[0..32].to_vec();
    let full = pack_bases_for_test(&bases); // 64 bytes for 256 bases
    let truncated = &full[..4]; // far short of the required 64 bytes
    assert!(
        idx.smem_range_packed(&query, truncated, 256).is_err(),
        "packed pac shorter than ceil(num_bases/4) should return Err, not panic"
    );
}

#[test]
fn smem_range_enc_rejects_short_pac() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "enc_short_pac");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let query: Vec<u8> = bases[0..32].to_vec();
    let qref: &[u8] = &query;
    let full = pack_bases_for_test(&bases);
    let truncated = &full[..4];
    let res = idx.smem_range_enc(&[qref], truncated, PacEncoding::Packed { num_bases: 256 });
    assert!(
        res.is_err(),
        "smem_range_enc with a packed pac shorter than required should return Err"
    );
}

#[test]
fn smem_range_long_read_packed_rejects_short_pac() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "lr_short_pac");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let full = pack_bases_for_test(&bases);
    let truncated = &full[..4];
    let pivots = [0u64, 32];
    let res = idx.smem_range_long_read_packed(&bases, 256, &pivots, truncated, 256);
    assert!(
        res.is_err(),
        "smem_range_long_read_packed with a short pac should return Err"
    );
}

#[test]
fn smem_range_batch_rejects_short_query() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "b_short");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let short: Vec<u8> = bases[0..16].to_vec();
    let short_ref: &[u8] = &short;
    let result = idx.smem_range_batch(&[short_ref], &bases);
    assert!(
        result.is_err(),
        "batch query shorter than 32 bases should return Err"
    );
}

#[test]
fn smem_range_batch_rejects_long_query() {
    let dir = tempdir().unwrap();
    let idx = open_test_idx(&dir, "b_long");
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let long: Vec<u8> = bases[0..36].to_vec();
    let long_ref: &[u8] = &long;
    let result = idx.smem_range_batch(&[long_ref], &bases);
    assert!(
        result.is_err(),
        "batch query longer than 32 bases should return Err"
    );
}
