// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::meta::Meta;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use prmi::train::mask::MaskConfig;
use std::io::Write;
use tempfile::tempdir;

fn write_fasta(dir: &std::path::Path, seq: &str) -> std::path::PathBuf {
    let p = dir.join("ref.fa");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, ">chr1\n{seq}").unwrap();
    p
}

#[test]
fn sidecar_is_2x_with_sentinel_row() {
    let dir = tempdir().unwrap();
    let seq = "ACGTACGTACGTACGTACGTACGTACGTACGTAA"; // 34 bases, no Ns
    let fa = write_fasta(dir.path(), seq);
    let prefix = dir.path().join("ref.fa.prmi");
    build_sidecar(&fa, &prefix, None, MaskConfig::default(), 1).unwrap();
    let paths = SidecarPaths::from_prefix(&prefix);
    let meta = Meta::read_file(&paths.meta).unwrap();
    let l_pac = seq.len() as u64;
    assert_eq!(meta.sa.strand, "forward_rc_2x");
    assert_eq!(meta.sa.num_entries, 2 * l_pac + 1);
}
