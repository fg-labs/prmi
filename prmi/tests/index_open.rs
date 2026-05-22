// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::error::Error;
use prmi::index::LearnedIndex;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use tempfile::tempdir;

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
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

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
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

    // Corrupt the .meta to set sa.strand = "forward_reverse" (a hypothetical v0.2 value).
    let meta_path = SidecarPaths::from_prefix(&prefix).meta;
    let meta_str = std::fs::read_to_string(&meta_path).unwrap();
    let corrupted = meta_str.replace(r#"strand = "forward_only""#, r#"strand = "forward_reverse""#);
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
        other => panic!(
            "expected FormatTooNew {{ kind: \"sa.strand=forward_reverse\" }}, got {other:?}"
        ),
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
    build_sidecar(&fa, &prefix, Some(16)).unwrap();

    // Corrupt the .meta to set priors.type = "bed" (a hypothetical v0.2 value).
    let meta_path = SidecarPaths::from_prefix(&prefix).meta;
    let meta_str = std::fs::read_to_string(&meta_path).unwrap();
    let corrupted = meta_str.replace(r#"type = "uniform""#, r#"type = "bed""#);
    assert!(
        corrupted.contains(r#"type = "bed""#),
        "replacement did not take effect — check the .meta TOML format"
    );
    std::fs::write(&meta_path, corrupted).unwrap();

    // Opening should fail cleanly with Error::FormatTooNew { kind: "bed" }.
    let err = LearnedIndex::open(&prefix).unwrap_err();
    match err {
        Error::FormatTooNew { kind } => assert_eq!(kind, "bed"),
        other => panic!("expected FormatTooNew {{ kind: \"bed\" }}, got {other:?}"),
    }
}
