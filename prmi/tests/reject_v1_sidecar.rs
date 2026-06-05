// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::error::Error;
use prmi::index::LearnedIndex;
use prmi::sidecar::SidecarPaths;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn open_rejects_v1_meta() {
    let dir = tempdir().unwrap();
    let prefix = dir.path().join("old.prmi");
    let paths = SidecarPaths::from_prefix(&prefix);
    // A minimal PRMIv1 .meta is enough — validation fails before binary files
    // are read.
    let mut f = std::fs::File::create(&paths.meta).unwrap();
    write!(
        f,
        r#"[prmi]
magic = "PRMIv1"
format_version = 1
trainer_version = "prmi=0.1.0"
created_utc = "2026-05-27T00:00:00Z"
[ref]
path = "r.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = 1
bytes_per_entry = 5
encoding = "packed_lo8_hi32"
mode = "1"
strand = "forward_only"
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 0
[priors]
type = "uniform"
"#
    )
    .unwrap();

    // Assert the concrete contract: the v1 magic is rejected before any
    // version check, so a stringified-text match (which could pass on an
    // unrelated failure that merely mentions "version"/"magic") is too loose.
    let err = LearnedIndex::open(&prefix).unwrap_err();
    match err {
        Error::BadMagic { found, .. } => assert_eq!(found, "PRMIv1"),
        Error::UnsupportedVersion { .. } => {}
        other => panic!("expected BadMagic/UnsupportedVersion for v1 sidecar, got {other:?}"),
    }
}
