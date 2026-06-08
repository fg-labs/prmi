// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::error::Error;
use prmi::sidecar::meta::Meta;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::TrainerConfig;
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

#[test]
fn kmer_table_k_zero_is_rejected() {
    let dir = tempdir().unwrap();
    let fa = write_fasta(dir.path(), "ACGTACGTACGTACGTACGTACGTACGTACGTAA");
    let prefix = dir.path().join("ref.fa.prmi");
    let config = TrainerConfig::default().with_kmer_table_k(0);
    // k = 0 must fail as invalid input, not silently coerce to a 1-mer table.
    let err = build_sidecar_with_config(&fa, &prefix, None, MaskConfig::default(), 1, Some(config))
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "expected InvalidInput for kmer_table_k=0, got: {err:?}"
    );
}
