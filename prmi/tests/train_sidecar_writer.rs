// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use tempfile::tempdir;

/// Generate a pseudo-random DNA sequence of `len` bases using an LCG.
/// Used to produce diverse 32-mer keys across the full u64 key space,
/// so that the piecewise-linear top model routes entries into different
/// L2 leaves without collapsing to a constant prediction.
fn pseudo_random_dna(len: usize) -> String {
    let bases = b"ACGT";
    let mut state: u64 = 0x123456789abcdef0;
    let mut seq = String::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seq.push(bases[((state >> 30) & 3) as usize] as char);
    }
    seq
}

#[test]
fn writes_all_four_files_for_synthetic_fasta() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("test.fa");
    // 4096 bp pseudo-random sequence. build_sidecar pads the genome with 31
    // T-sentinel bases before suffix-array construction and then filters out
    // SA entries for the padding region, so the trained index covers exactly
    // genome_len = 4096 positions.
    let genome_len: usize = 4096;
    let mut content = String::from(">chr1\n");
    content.push_str(&pseudo_random_dna(genome_len));
    content.push('\n');
    std::fs::write(&fa, content.as_bytes()).unwrap();

    let prefix = dir.path().join("test.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();

    let paths = SidecarPaths::from_prefix(&prefix);
    assert!(paths.meta.exists(), "meta missing");
    assert!(paths.sa.exists(), "sa missing");
    assert!(paths.l1.exists(), "l1 missing");
    assert!(paths.l2.exists(), "l2 missing");

    // The .meta should round-trip parse cleanly.
    let meta = prmi::sidecar::meta::Meta::read_file(&paths.meta).unwrap();
    assert_eq!(meta.sa.bytes_per_entry, 5);
    assert_eq!(meta.sa.strand, "forward_only");
    assert_eq!(meta.priors.kind, "uniform");
    assert_eq!(meta.rmi.l2_leaf_count, 16);
    assert_eq!(meta.rmi.bit_shift, 60);
    assert!(meta.rmi.spec.starts_with("pwl4,linear,linear_spline"));
    assert_eq!(meta.ref_.sha256.len(), 64); // hex sha256
    assert_eq!(meta.sa.num_entries, genome_len as u64); // padding excluded

    // Default (no mask) → all mask fields false/None.
    assert!(!meta.sa.masked_n_runs);
    assert!(meta.sa.masked_homopolymers.is_none());
    assert!(meta.sa.masked_bed.is_none());
}
