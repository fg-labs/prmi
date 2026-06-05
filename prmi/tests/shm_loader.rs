// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the shared-memory loader (`prmi shm load` + `open_shm`).

use prmi::index::shm::{read_shm_blob, write_shm_blob};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::TrainerConfig;
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

// ── LearnedIndex::write_shm helper ───────────────────────────────────────────

#[test]
fn write_shm_convenience_method_produces_valid_blob() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("helper.shm");

    LearnedIndex::write_shm(&prefix, &shm_path).expect("write_shm should succeed");
    let idx = LearnedIndex::open_shm(&shm_path).expect("open_shm should succeed");
    assert!(idx.sa_num() > 0, "index must have at least one SA entry");
}

// ── open_shm: .kmt carriage ──────────────────────────────────────────────────

/// A sidecar built WITHOUT `--kmer-table-k` produces a blob with no `.kmt`
/// component, and `open_shm` reports `has_kmt() == false` (it falls back to the
/// full forward search). This also guards the backward-compatible header: the
/// reserved `[88..104]` bytes read as zero.
#[test]
fn open_shm_without_kmt_has_no_table() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("no_kmt.shm");

    write_shm_blob(&prefix, &shm_path).unwrap();

    let blob = read_shm_blob(&shm_path).unwrap();
    assert_eq!(blob.kmt_len, 0, "no .kmt was built, so kmt_len must be 0");

    let idx = LearnedIndex::open_shm(&shm_path).unwrap();
    assert!(
        !idx.has_kmt(),
        "an shm blob without .kmt must not load a table"
    );
}

/// Build a sidecar WITH a `.kmt` k-mer table in `dir`, returning the sidecar
/// prefix and the unpacked reference bases (for forward-spectrum queries). The
/// reference is chosen to produce a non-empty forward spectrum.
fn build_kmt_sidecar(dir: &std::path::Path) -> (PathBuf, Vec<u8>) {
    let fa = dir.join("ref.fa");
    let pac_unpacked: Vec<u8> = (0u64..256).map(|i| ((i * 5 + 1) % 4) as u8).collect();
    let mut fa_bytes = b">ref\n".to_vec();
    for &b in &pac_unpacked {
        fa_bytes.push(b"ACGT"[b as usize]);
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();

    let prefix = dir.join("ref.fa.prmi");
    let cfg = TrainerConfig::default().with_kmer_table_k(6);
    build_sidecar_with_config(&fa, &prefix, Some(16), Default::default(), 1, Some(cfg)).unwrap();
    (prefix, pac_unpacked)
}

/// The file-backed full forward search over a 24-base query drawn from the
/// reference — the byte-identity reference output any accelerator must match.
fn full_search_reference(prefix: &std::path::Path, pac_unpacked: &[u8]) -> Vec<u8> {
    let file_idx = LearnedIndex::open(prefix).unwrap();
    let query: Vec<u8> = pac_unpacked[8..8 + 24].to_vec();
    let steps = file_idx.forward_spectrum(&query, pac_unpacked, PacEncoding::Unpacked);
    assert!(!steps.is_empty(), "expected a non-empty forward spectrum");
    // Flatten to a comparable byte vector via debug encoding of each step.
    format!("{steps:?}").into_bytes()
}

/// A sidecar built WITH `--kmer-table-k` carries the `.kmt` into the shm blob;
/// `open_shm` loads it (`has_kmt()`), and `forward_spectrum_auto` over the
/// shm-loaded table is byte-identical to the file-backed index's full forward
/// search. This proves the shm carriage neither drops nor corrupts the table.
#[test]
fn open_shm_with_kmt_is_byte_identical_to_full_search() {
    let dir = tempdir().unwrap();
    let (prefix, pac_unpacked) = build_kmt_sidecar(dir.path());

    let shm_path = dir.path().join("with_kmt.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();

    // The blob must carry the .kmt component.
    let blob = read_shm_blob(&shm_path).unwrap();
    assert!(blob.kmt_len > 0, ".kmt must be carried in the blob");
    // .kmt sits after .l2 and stays within the blob.
    assert!(blob.l2_offset + blob.l2_len <= blob.kmt_offset);
    assert_eq!(blob.kmt_offset % 4096, 0, "kmt_offset must be page-aligned");

    // shm-loaded index loads the table.
    let shm_idx = LearnedIndex::open_shm(&shm_path).unwrap();
    assert!(shm_idx.has_kmt(), "open_shm must load the carried .kmt");

    // Reference: the file-backed full forward search (the table is a pure
    // accelerator, so its tabled output must equal the untabled search).
    let query: Vec<u8> = pac_unpacked[8..8 + 24].to_vec();
    let file_idx = LearnedIndex::open(&prefix).unwrap();
    let reference = file_idx.forward_spectrum(&query, &pac_unpacked, PacEncoding::Unpacked);
    assert!(
        !reference.is_empty(),
        "expected a non-empty forward spectrum"
    );

    let via_shm_table = shm_idx.forward_spectrum_auto(&query, &pac_unpacked, PacEncoding::Unpacked);
    assert_eq!(
        via_shm_table, reference,
        "shm-table forward spectrum must be byte-identical to the full search"
    );
}

/// Best-effort, never-silently-wrong: a carried `.kmt` whose own header is
/// corrupt (magic clobbered) must be rejected by `from_shm_slice`, so `open_shm`
/// still succeeds with `has_kmt() == false`. The full forward search remains
/// correct (byte-identical to a clean file open). Exercises the corrupt-slice
/// fallback branch in the shm load path.
#[test]
fn open_shm_with_corrupt_kmt_falls_back() {
    let dir = tempdir().unwrap();
    let (prefix, pac_unpacked) = build_kmt_sidecar(dir.path());
    let reference = full_search_reference(&prefix, &pac_unpacked);

    let shm_path = dir.path().join("corrupt_kmt.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();

    // Clobber the carried .kmt's 4-byte magic (at the start of its component).
    let blob = read_shm_blob(&shm_path).unwrap();
    let kmt_off = blob.kmt_offset;
    assert!(blob.kmt_len > 0);
    drop(blob); // release the mmap before rewriting the file
    let mut bytes = std::fs::read(&shm_path).unwrap();
    bytes[kmt_off..kmt_off + 4].fill(0xff);
    std::fs::write(&shm_path, &bytes).unwrap();

    // open_shm must still succeed, ignore the corrupt table, and stay correct.
    let idx = LearnedIndex::open_shm(&shm_path).unwrap();
    assert!(
        !idx.has_kmt(),
        "a corrupt carried .kmt must be ignored, not loaded"
    );
    let query: Vec<u8> = pac_unpacked[8..8 + 24].to_vec();
    let via_fallback = idx.forward_spectrum_auto(&query, &pac_unpacked, PacEncoding::Unpacked);
    assert_eq!(
        format!("{via_fallback:?}").into_bytes(),
        reference,
        "fallback forward search must match the full search"
    );
}

/// Best-effort binding: a carried `.kmt` whose own header is structurally valid
/// but whose `ref_digest` no longer matches the sidecar must be rejected by the
/// `kmt_matches` reference-binding check (NOT by `from_shm_slice`), so `open_shm`
/// still succeeds with `has_kmt() == false`. This locks the reference-binding on
/// the shm path, mirroring the file path's `load_kmt_best_effort`.
#[test]
fn open_shm_with_ref_mismatched_kmt_falls_back() {
    let dir = tempdir().unwrap();
    let (prefix, pac_unpacked) = build_kmt_sidecar(dir.path());
    let reference = full_search_reference(&prefix, &pac_unpacked);

    let shm_path = dir.path().join("mismatch_kmt.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();

    // Flip one byte of the carried .kmt's `ref_digest` (kmt header bytes 24..56),
    // leaving its magic/version/sa_num intact so the slice opens but the digest
    // binding fails.
    let blob = read_shm_blob(&shm_path).unwrap();
    let digest_byte = blob.kmt_offset + 24;
    assert!(blob.kmt_len > 56);
    drop(blob);
    let mut bytes = std::fs::read(&shm_path).unwrap();
    bytes[digest_byte] ^= 0xff;
    std::fs::write(&shm_path, &bytes).unwrap();

    let idx = LearnedIndex::open_shm(&shm_path).unwrap();
    assert!(
        !idx.has_kmt(),
        "a ref-mismatched carried .kmt must be ignored, not loaded"
    );
    let query: Vec<u8> = pac_unpacked[8..8 + 24].to_vec();
    let via_fallback = idx.forward_spectrum_auto(&query, &pac_unpacked, PacEncoding::Unpacked);
    assert_eq!(
        format!("{via_fallback:?}").into_bytes(),
        reference,
        "fallback forward search must match the full search"
    );
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

#[test]
fn read_shm_blob_rejects_zero_length_core_component() {
    let (_dir, prefix) = build_test_sidecar();
    let shm_path = _dir.path().join("zero_meta.shm");
    write_shm_blob(&prefix, &shm_path).unwrap();
    // meta_len lives at header bytes [32..40]. Only `.kmt` may be absent; a
    // zero-length CORE component is a malformed blob and must be rejected, not
    // silently passed through as an empty slice.
    patch_header_u64(&shm_path, 32, 0);
    assert!(
        read_shm_blob(&shm_path).is_err(),
        "a zero-length core component (meta) must be rejected"
    );
}
