// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the shared-memory loader (`prmi shm load` + `open_shm`).

use prmi::index::shm::{read_shm_blob, write_shm_blob};
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use std::path::PathBuf;
use tempfile::tempdir;

/// Build a small FASTA, train a sidecar, and return the tmpdir (kept alive by
/// caller) and the sidecar prefix.
fn build_test_sidecar() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("t.fa");
    let mut content = String::from(">chr1\n");
    for _ in 0..64 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("t.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    (dir, prefix)
}

// ── write_shm_blob / read_shm_blob round-trip ─────────────────────────────────

#[test]
fn shm_blob_write_read_round_trip() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("test.shm");

    // Write the blob.
    write_shm_blob(&prefix, &shm_path).expect("write_shm_blob should succeed");
    assert!(shm_path.exists(), "shm blob file should exist after write");

    // Read it back and validate the component layout.
    let blob = read_shm_blob(&shm_path).expect("read_shm_blob should succeed");
    assert!(blob.meta_len > 0, "meta component should be non-empty");
    assert!(blob.sa_len > 0, "sa component should be non-empty");
    assert!(blob.l1_len > 0, "l1 component should be non-empty");
    assert!(blob.l2_len > 0, "l2 component should be non-empty");

    // Components must not overlap and must be ordered.
    assert!(blob.meta_offset + blob.meta_len <= blob.sa_offset);
    assert!(blob.sa_offset + blob.sa_len <= blob.l1_offset);
    assert!(blob.l1_offset + blob.l1_len <= blob.l2_offset);

    // All offsets must be 4 KiB aligned.
    assert_eq!(
        blob.meta_offset % 4096,
        0,
        "meta_offset must be page-aligned"
    );
    assert_eq!(blob.sa_offset % 4096, 0, "sa_offset must be page-aligned");
    assert_eq!(blob.l1_offset % 4096, 0, "l1_offset must be page-aligned");
    assert_eq!(blob.l2_offset % 4096, 0, "l2_offset must be page-aligned");
}

// ── open_shm: basic open ──────────────────────────────────────────────────────

#[test]
fn open_shm_succeeds_and_matches_regular_open() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("basic.shm");

    write_shm_blob(&prefix, &shm_path).unwrap();

    let shm_idx = LearnedIndex::open_shm(&shm_path).expect("open_shm should succeed");
    let file_idx = LearnedIndex::open(&prefix).expect("regular open should succeed");

    // sa_num, max_error_bound, and bit_shift must be identical.
    assert_eq!(
        shm_idx.sa_num(),
        file_idx.sa_num(),
        "sa_num must match between shm and file-backed index"
    );
    assert_eq!(
        shm_idx.max_error_bound(),
        file_idx.max_error_bound(),
        "max_error_bound must match"
    );
    assert_eq!(
        shm_idx.bit_shift(),
        file_idx.bit_shift(),
        "bit_shift must match"
    );
    assert_eq!(
        shm_idx.format_version(),
        file_idx.format_version(),
        "format_version must match"
    );
}

// ── open_shm: lookup equivalence ─────────────────────────────────────────────

#[test]
fn open_shm_lookup_matches_regular_open() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("lookup.shm");

    write_shm_blob(&prefix, &shm_path).unwrap();

    let shm_idx = LearnedIndex::open_shm(&shm_path).unwrap();
    let file_idx = LearnedIndex::open(&prefix).unwrap();

    // Probe a range of keys and verify identical (pos, err) results.
    for key in [0u64, 1, 0x0123456789abcdef, u64::MAX / 2, u64::MAX] {
        let shm_result = shm_idx.lookup(key);
        let file_result = file_idx.lookup(key);
        assert_eq!(
            shm_result, file_result,
            "lookup({key}) differs: shm={shm_result:?} file={file_result:?}"
        );
    }
}

// ── open_shm: SA positions match ─────────────────────────────────────────────

#[test]
fn open_shm_sa_positions_match_regular_open() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("sa_pos.shm");

    write_shm_blob(&prefix, &shm_path).unwrap();

    let shm_idx = LearnedIndex::open_shm(&shm_path).unwrap();
    let file_idx = LearnedIndex::open(&prefix).unwrap();

    let n = shm_idx.sa_num().min(10);
    let mut shm_buf = vec![0u64; n as usize];
    let mut file_buf = vec![0u64; n as usize];
    shm_idx.sa_positions(0, &mut shm_buf).unwrap();
    file_idx.sa_positions(0, &mut file_buf).unwrap();
    assert_eq!(shm_buf, file_buf, "first {n} SA positions must match");
}

// ── open_shm: error cases ─────────────────────────────────────────────────────

#[test]
fn open_shm_missing_file_returns_error() {
    let result = LearnedIndex::open_shm("/nonexistent/path/prmi.shm");
    assert!(result.is_err(), "open_shm on missing file should fail");
}

#[test]
fn open_shm_corrupt_magic_returns_error() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("corrupt.shm");

    write_shm_blob(&prefix, &shm_path).unwrap();

    // Overwrite the first 16 bytes with garbage to corrupt the magic.
    let mut bytes = std::fs::read(&shm_path).unwrap();
    bytes[0..16].fill(0xff);
    std::fs::write(&shm_path, &bytes).unwrap();

    let result = LearnedIndex::open_shm(&shm_path);
    assert!(result.is_err(), "open_shm on corrupt magic should fail");
}

// ── read_shm_blob: wrapper-layout invariant enforcement ──────────────────────

/// Read a blob file's raw bytes, overwrite the little-endian u64 header field at
/// `[byte_off, byte_off+8)`, and write it back.
fn patch_header_u64(shm_path: &std::path::Path, byte_off: usize, value: u64) {
    let mut bytes = std::fs::read(shm_path).unwrap();
    bytes[byte_off..byte_off + 8].copy_from_slice(&value.to_le_bytes());
    std::fs::write(shm_path, &bytes).unwrap();
}

#[test]
fn read_shm_blob_rejects_offset_inside_header() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("inside_header.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();
    // meta_offset lives at header bytes [24..32]; 0 is inside the reserved header.
    patch_header_u64(&shm_path, 24, 0);
    assert!(
        read_shm_blob(&shm_path).is_err(),
        "a component offset inside the reserved header must be rejected"
    );
}

#[test]
fn read_shm_blob_rejects_misaligned_offset() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("misaligned.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();
    let sa_offset = read_shm_blob(&shm_path).unwrap().sa_offset as u64;
    // sa_offset lives at header bytes [40..48]; +1 makes it non-4KiB-aligned.
    patch_header_u64(&shm_path, 40, sa_offset + 1);
    assert!(
        read_shm_blob(&shm_path).is_err(),
        "a non-page-aligned component offset must be rejected"
    );
}

#[test]
fn read_shm_blob_rejects_overlapping_components() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("overlap.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();
    let sa_offset = read_shm_blob(&shm_path).unwrap().sa_offset as u64;
    // Point l1_offset (header bytes [56..64]) back at sa_offset: aligned, in
    // bounds, but overlaps the (non-empty) sa component.
    patch_header_u64(&shm_path, 56, sa_offset);
    assert!(
        read_shm_blob(&shm_path).is_err(),
        "overlapping / out-of-order components must be rejected"
    );
}

// ── LearnedIndex::write_shm helper ───────────────────────────────────────────

#[test]
fn write_shm_convenience_method_produces_valid_blob() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("helper.shm");

    LearnedIndex::write_shm(&prefix, &shm_path).expect("write_shm should succeed");
    let idx = LearnedIndex::open_shm(&shm_path).expect("open_shm should succeed");
    assert!(idx.sa_num() > 0, "index must have at least one SA entry");
}
