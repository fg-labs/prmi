// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the memory-mode menu (modes 1/2/3 and suffix-key-cache).
//!
//! Each test builds a small sidecar in a specific mode and verifies:
//! - The on-disk layout matches the expected bytes_per_entry.
//! - `key_at` / `isa_at` return expected values.
//! - `smem_range` produces identical results across all modes for the same corpus.

use prmi::encoding::tokenize_32mer;
use prmi::index::LearnedIndex;
use prmi::sidecar::sa_file::{BPE_MODE1, BPE_MODE2, BPE_MODE3, SA_FILE_HEADER_BYTES};
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi::train::mask::MaskConfig;
use tempfile::TempDir;

// ── small synthetic reference ──────────────────────────────────────────────────

/// Return a small synthetic FASTA for testing (64-base sequence).
/// We use a deterministic sequence to keep golden comparisons stable.
fn synthetic_fasta() -> String {
    // 64 bases: ACGT repeated 16x then a T-end for the sentinel.
    let seq = "ACGT".repeat(16);
    format!(">synthetic\n{}\n", seq)
}

/// Write a synthetic FASTA to a temp dir and return the path.
fn write_fasta(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("ref.fa");
    std::fs::write(&path, synthetic_fasta()).unwrap();
    path
}

/// Build a sidecar with the given mode. Returns the prefix path.
fn build_sidecar_mode(dir: &TempDir, mode: MemoryMode) -> std::path::PathBuf {
    let fa = write_fasta(dir);
    let prefix = dir.path().join("index");
    let config = TrainerConfig::default().with_memory_mode(mode);
    build_sidecar_with_config(
        &fa,
        &prefix,
        Some(16), // small l2_leaf_count for speed
        MaskConfig::default(),
        1, // single-threaded
        Some(config),
    )
    .unwrap();
    prefix
}

// ── mode 1 ────────────────────────────────────────────────────────────────────

#[test]
fn mode1_bytes_per_entry_is_5() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode1);
    let paths = SidecarPaths::from_prefix(&prefix);
    let file_size = std::fs::metadata(&paths.sa).unwrap().len();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = idx.sa_num();
    let expected_size = SA_FILE_HEADER_BYTES as u64 + n * BPE_MODE1 as u64;
    assert_eq!(file_size, expected_size, "mode 1 .sa size mismatch");
}

#[test]
fn mode1_key_at_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode1);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.key_at(0), None, "mode 1 should not store keys");
}

#[test]
fn mode1_isa_at_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode1);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.isa_at(0), None, "mode 1 should not store ISA");
}

#[test]
fn mode1_memory_mode_string_is_1() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode1);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.memory_mode(), "1");
}

// ── mode 2 ────────────────────────────────────────────────────────────────────

#[test]
fn mode2_bytes_per_entry_is_13() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode2);
    let paths = SidecarPaths::from_prefix(&prefix);
    let file_size = std::fs::metadata(&paths.sa).unwrap().len();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = idx.sa_num();
    let expected_size = SA_FILE_HEADER_BYTES as u64 + n * BPE_MODE2 as u64;
    assert_eq!(file_size, expected_size, "mode 2 .sa size mismatch");
}

#[test]
fn mode2_is_approximately_26x_larger_than_mode1() {
    // 13/5 = 2.6; allow tolerance of ±5%.
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let prefix1 = build_sidecar_mode(&dir1, MemoryMode::Mode1);
    let prefix2 = build_sidecar_mode(&dir2, MemoryMode::Mode2);
    let paths1 = SidecarPaths::from_prefix(&prefix1);
    let paths2 = SidecarPaths::from_prefix(&prefix2);
    let size1 = std::fs::metadata(&paths1.sa).unwrap().len();
    let size2 = std::fs::metadata(&paths2.sa).unwrap().len();

    // Body ratio (header is small): body2 / body1 ≈ 13/5 = 2.6
    let body1 = size1 - SA_FILE_HEADER_BYTES as u64;
    let body2 = size2 - SA_FILE_HEADER_BYTES as u64;
    let ratio = body2 as f64 / body1 as f64;
    assert!(
        (ratio - 2.6_f64).abs() < 0.05,
        "mode2/mode1 body ratio expected ~2.6, got {ratio:.3}"
    );
}

#[test]
fn mode2_key_at_is_some() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode2);
    let idx = LearnedIndex::open(&prefix).unwrap();
    // All entries should return Some(key).
    for i in 0..idx.sa_num().min(10) {
        assert!(idx.key_at(i).is_some(), "mode 2 key_at({i}) returned None");
    }
}

#[test]
fn mode2_stored_key_matches_tokenized_pac_key() {
    let dir = tempfile::tempdir().unwrap();
    let fa = write_fasta(&dir);
    let prefix = dir.path().join("index");
    let config = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_with_config(
        &fa,
        &prefix,
        Some(16),
        MaskConfig::default(),
        1,
        Some(config),
    )
    .unwrap();

    // Reconstruct the pac from the FASTA for comparison.
    let fasta_str = synthetic_fasta();
    let bases_raw: Vec<u8> = fasta_str
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.bytes())
        .map(|b| match b {
            b'A' | b'a' => 0u8,
            b'C' | b'c' => 1u8,
            b'G' | b'g' => 2u8,
            b'T' | b't' => 3u8,
            _ => 0u8,
        })
        .collect();

    let idx = LearnedIndex::open(&prefix).unwrap();
    // Verify the first few stored keys match what we'd compute from the pac.
    for i in 0..idx.sa_num().min(20) {
        let stored_key = idx.key_at(i).expect("mode 2 must store keys");
        let pos = idx.sa_position_for(i) as usize;
        let avail = (bases_raw.len().saturating_sub(pos)).min(32);
        let expected_key = tokenize_32mer(&bases_raw[pos..pos + avail], avail);
        assert_eq!(
            stored_key, expected_key,
            "mode 2: stored key at SA[{i}] (pos={pos}) doesn't match tokenized pac key"
        );
    }
}

#[test]
fn mode2_isa_at_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode2);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.isa_at(0), None, "mode 2 should not store ISA");
}

#[test]
fn mode2_memory_mode_string_is_2() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode2);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.memory_mode(), "2");
}

// ── mode 3 ────────────────────────────────────────────────────────────────────

#[test]
fn mode3_bytes_per_entry_is_21() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode3);
    let paths = SidecarPaths::from_prefix(&prefix);
    let file_size = std::fs::metadata(&paths.sa).unwrap().len();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = idx.sa_num();
    let expected_size = SA_FILE_HEADER_BYTES as u64 + n * BPE_MODE3 as u64;
    assert_eq!(file_size, expected_size, "mode 3 .sa size mismatch");
}

#[test]
fn mode3_key_at_is_some() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode3);
    let idx = LearnedIndex::open(&prefix).unwrap();
    for i in 0..idx.sa_num().min(10) {
        assert!(idx.key_at(i).is_some(), "mode 3 key_at({i}) returned None");
    }
}

#[test]
fn mode3_isa_at_is_some() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode3);
    let idx = LearnedIndex::open(&prefix).unwrap();
    for i in 0..idx.sa_num().min(10) {
        assert!(idx.isa_at(i).is_some(), "mode 3 isa_at({i}) returned None");
    }
}

#[test]
fn mode3_isa_is_inverse_of_sa() {
    // If sa[i] = pos, then isa[pos] = i. Verify a few entries.
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode3);
    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = idx.sa_num();
    // Build the ISA manually from the SA positions.
    let mut expected_isa = vec![0u64; n as usize];
    for i in 0..n {
        let pos = idx.sa_position_for(i) as usize;
        expected_isa[pos] = i;
    }
    // Check the stored ISA values match.
    for i in 0..n.min(30) {
        let pos = idx.sa_position_for(i) as usize;
        let stored_isa = idx.isa_at(i).expect("mode 3 must store ISA");
        let expected = expected_isa[pos];
        assert_eq!(
            stored_isa, expected,
            "mode 3: ISA at SA[{i}] (pos={pos}) expected {expected}, got {stored_isa}"
        );
    }
}

#[test]
fn mode3_memory_mode_string_is_3() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::Mode3);
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.memory_mode(), "3");
}

// ── suffix-key-cache ──────────────────────────────────────────────────────────

#[test]
fn suffix_key_cache_sa_is_mode1_size() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::SuffixKeyCache { cache_size: 10 });
    let paths = SidecarPaths::from_prefix(&prefix);
    let file_size = std::fs::metadata(&paths.sa).unwrap().len();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = idx.sa_num();
    let expected_size = SA_FILE_HEADER_BYTES as u64 + n * BPE_MODE1 as u64;
    assert_eq!(
        file_size, expected_size,
        "skc mode .sa should be same size as mode 1"
    );
}

#[test]
fn suffix_key_cache_skc_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::SuffixKeyCache { cache_size: 5 });
    let paths = SidecarPaths::from_prefix(&prefix);
    assert!(
        paths.skc.exists(),
        ".skc file should exist for suffix_key_cache mode"
    );
}

#[test]
fn suffix_key_cache_key_at_returns_some_for_cached_entries() {
    let cache_size = 8u64;
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::SuffixKeyCache { cache_size });
    let idx = LearnedIndex::open(&prefix).unwrap();
    // The first min(cache_size, sa_num) entries should be cached.
    let n_cached = cache_size.min(idx.sa_num());
    for i in 0..n_cached {
        assert!(
            idx.key_at(i).is_some(),
            "skc: key_at({i}) returned None for a cached entry"
        );
    }
}

#[test]
fn suffix_key_cache_key_at_returns_none_for_uncached_entries() {
    let cache_size = 5u64;
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::SuffixKeyCache { cache_size });
    let idx = LearnedIndex::open(&prefix).unwrap();
    // Entries beyond the cache should return None.
    let n = idx.sa_num();
    if n > cache_size {
        assert_eq!(
            idx.key_at(cache_size),
            None,
            "skc: key_at({cache_size}) should be None (beyond cache)"
        );
    }
}

#[test]
fn suffix_key_cache_memory_mode_string() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = build_sidecar_mode(&dir, MemoryMode::SuffixKeyCache { cache_size: 10 });
    let idx = LearnedIndex::open(&prefix).unwrap();
    assert_eq!(idx.memory_mode(), "suffix_key_cache");
}

// ── cross-mode smem_range equivalence ─────────────────────────────────────────
//
// Modes 1/2/3 and suffix-key-cache must produce identical smem_range results.

/// Build an unpacked PAC from the synthetic FASTA.
fn make_pac() -> Vec<u8> {
    let fasta_str = synthetic_fasta();
    fasta_str
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.bytes())
        .map(|b| match b {
            b'A' | b'a' => 0u8,
            b'C' | b'c' => 1u8,
            b'G' | b'g' => 2u8,
            b'T' | b't' => 3u8,
            _ => 0u8,
        })
        .collect()
}

/// Build a 32-base query from the first 32 bases of the PAC.
fn make_query(pac: &[u8]) -> Vec<u8> {
    pac[..32.min(pac.len())].to_vec()
}

#[test]
fn modes_1_2_3_smem_range_equivalent() {
    let pac = make_pac();
    let query = make_query(&pac);

    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let dir3 = tempfile::tempdir().unwrap();

    let prefix1 = build_sidecar_mode(&dir1, MemoryMode::Mode1);
    let prefix2 = build_sidecar_mode(&dir2, MemoryMode::Mode2);
    let prefix3 = build_sidecar_mode(&dir3, MemoryMode::Mode3);

    let idx1 = LearnedIndex::open(&prefix1).unwrap();
    let idx2 = LearnedIndex::open(&prefix2).unwrap();
    let idx3 = LearnedIndex::open(&prefix3).unwrap();

    let (k1, l1, s1) = idx1.smem_range(&query, &pac).unwrap();
    let (k2, l2, s2) = idx2.smem_range(&query, &pac).unwrap();
    let (k3, l3, s3) = idx3.smem_range(&query, &pac).unwrap();

    assert_eq!(
        (k1, l1, s1),
        (k2, l2, s2),
        "mode1 and mode2 smem_range differ for the same query"
    );
    assert_eq!(
        (k1, l1, s1),
        (k3, l3, s3),
        "mode1 and mode3 smem_range differ for the same query"
    );
}

#[test]
fn mode1_vs_suffix_key_cache_smem_range_equivalent() {
    let pac = make_pac();
    let query = make_query(&pac);

    let dir1 = tempfile::tempdir().unwrap();
    let dir_skc = tempfile::tempdir().unwrap();

    let prefix1 = build_sidecar_mode(&dir1, MemoryMode::Mode1);
    let prefix_skc = build_sidecar_mode(&dir_skc, MemoryMode::SuffixKeyCache { cache_size: 1000 });

    let idx1 = LearnedIndex::open(&prefix1).unwrap();
    let idx_skc = LearnedIndex::open(&prefix_skc).unwrap();

    let (k1, l1, s1) = idx1.smem_range(&query, &pac).unwrap();
    let (k_skc, l_skc, s_skc) = idx_skc.smem_range(&query, &pac).unwrap();

    assert_eq!(
        (k1, l1, s1),
        (k_skc, l_skc, s_skc),
        "mode1 and suffix_key_cache smem_range differ for the same query"
    );
}
