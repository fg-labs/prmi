// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::error::Error;
use prmi::index::LearnedIndex;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use tempfile::tempdir;

/// Build a 256-bp FASTA, return the tmpdir (kept alive by caller) and the sidecar prefix.
fn build_test_sidecar() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("t.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();
    let prefix = dir.path().join("t.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    (dir, prefix)
}

#[test]
fn opens_a_freshly_built_sidecar() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("t.fa");
    // 256 bp ACGT — same shape as Task 18 test.
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();

    let prefix = dir.path().join("t.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();
    assert!(idx.sa_num() > 0);
    assert_eq!(idx.format_version(), "PRMIv1");
    assert_eq!(idx.bit_shift(), 60);

    // Smoke-test a lookup — just verify it doesn't panic and returns within SA bounds.
    let (pred, _err) = idx.lookup(0);
    assert!(pred < idx.sa_num());
}

#[test]
fn rejects_missing_files() {
    let dir = tempdir().unwrap();
    let prefix = dir.path().join("absent");
    assert!(LearnedIndex::open(&prefix).is_err());
}

#[test]
fn open_rejects_unknown_strand_as_format_too_new() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("s.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();

    let prefix = dir.path().join("s.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    // Corrupt the .meta to set sa.strand = "forward_reverse" (a hypothetical v0.2 value).
    let meta_path = SidecarPaths::from_prefix(&prefix).meta;
    let meta_str = std::fs::read_to_string(&meta_path).unwrap();
    let corrupted = meta_str.replace(
        r#"strand = "forward_only""#,
        r#"strand = "forward_reverse""#,
    );
    assert!(
        corrupted.contains(r#"strand = "forward_reverse""#),
        "replacement did not take effect — check the .meta TOML format"
    );
    std::fs::write(&meta_path, corrupted).unwrap();

    // Opening should fail cleanly with Error::FormatTooNew { kind: "sa.strand=forward_reverse" }.
    let err = LearnedIndex::open(&prefix).unwrap_err();
    match err {
        Error::FormatTooNew { kind } => {
            assert_eq!(kind, "sa.strand=forward_reverse")
        }
        other => {
            panic!("expected FormatTooNew {{ kind: \"sa.strand=forward_reverse\" }}, got {other:?}")
        }
    }
}

#[test]
fn open_rejects_unknown_priors_type_as_format_too_new() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("e.fa");
    let mut content = String::from(">c\n");
    for _ in 0..32 {
        content.push_str("ACGTACGT");
    }
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();

    let prefix = dir.path().join("e.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    // Corrupt the .meta to set priors.type = "future_prior_type" (a hypothetical v0.2 value).
    let meta_path = SidecarPaths::from_prefix(&prefix).meta;
    let meta_str = std::fs::read_to_string(&meta_path).unwrap();
    let corrupted = meta_str.replace(r#"type = "uniform""#, r#"type = "future_prior_type""#);
    assert!(
        corrupted.contains(r#"type = "future_prior_type""#),
        "replacement did not take effect — check the .meta TOML format"
    );
    std::fs::write(&meta_path, corrupted).unwrap();

    // Opening should fail cleanly with Error::FormatTooNew { kind: "future_prior_type" }.
    let err = LearnedIndex::open(&prefix).unwrap_err();
    match err {
        Error::FormatTooNew { kind } => assert_eq!(kind, "future_prior_type"),
        other => panic!("expected FormatTooNew {{ kind: \"future_prior_type\" }}, got {other:?}"),
    }
}

#[test]
fn open_rejects_inconsistent_bit_shift() {
    let (_dir, prefix) = build_test_sidecar();

    // Read the .meta and corrupt bit_shift to 56, which would imply
    // l2_leaf_count=256 but the sidecar was built with l2_leaf_count=16
    // (bit_shift=60). cross_validate must catch this before any lookup
    // can cause an OOB mmap index.
    let meta_path = SidecarPaths::from_prefix(&prefix).meta;
    let meta_str = std::fs::read_to_string(&meta_path).unwrap();
    // The .meta has "bit_shift = 60"; replace with 56.
    let corrupted = meta_str.replace("bit_shift = 60", "bit_shift = 56");
    assert!(
        corrupted.contains("bit_shift = 56"),
        "replacement did not take effect — check the .meta TOML format (got: {meta_str})"
    );
    std::fs::write(&meta_path, corrupted).unwrap();

    let err = LearnedIndex::open(&prefix).unwrap_err();
    match err {
        Error::SidecarMismatch { detail, .. } => {
            assert!(
                detail.contains("bit_shift"),
                "expected bit_shift mention in detail, got: {detail}"
            );
        }
        other => panic!("expected SidecarMismatch, got {other:?}"),
    }
}
