// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::SidecarPaths;
use std::path::PathBuf;

#[test]
fn from_prefix_appends_suffixes() {
    let p = SidecarPaths::from_prefix(&PathBuf::from("/data/hg38.fa.prmi"));
    assert_eq!(p.meta, PathBuf::from("/data/hg38.fa.prmi.meta"));
    assert_eq!(p.sa, PathBuf::from("/data/hg38.fa.prmi.sa"));
    assert_eq!(p.l1, PathBuf::from("/data/hg38.fa.prmi.l1"));
    assert_eq!(p.l2, PathBuf::from("/data/hg38.fa.prmi.l2"));
    assert_eq!(p.isa, PathBuf::from("/data/hg38.fa.prmi.isa"));
}

#[test]
fn from_prefix_preserves_existing_extension() {
    // Verify we don't accidentally use Path::with_extension (which would
    // strip the `.prmi` from `hg38.fa.prmi` before appending).
    let p = SidecarPaths::from_prefix(&PathBuf::from("a.b.c"));
    assert_eq!(p.meta, PathBuf::from("a.b.c.meta"));
}
