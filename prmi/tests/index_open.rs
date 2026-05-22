// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::LearnedIndex;
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
