// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the memory-mode menu (modes 1 and 2).
//!
//! Each test builds a small sidecar in a specific mode and verifies:
//! - The on-disk layout matches the expected bytes_per_entry.
//! - `key_at` / `isa_at` return expected values.

use prmi::encoding::tokenize_32mer;
use prmi::index::LearnedIndex;
use prmi::sidecar::sa_file::{BPE_MODE1, BPE_MODE2, SA_FILE_HEADER_BYTES};
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
#[ignore = "forward-only primitive replaced by 2x spectrum in Plan 3"]
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
