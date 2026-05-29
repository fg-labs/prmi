// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the BED-prior trainer flag (`--prior-bed`).
//!
//! The BED prior biases the model fit toward pairs whose genome position falls
//! in the BED region, reducing per-leaf prediction error there at the cost of
//! potentially larger error outside.
//!
//! Tests here verify:
//! 1. Both uniform and BED-prior builds succeed and produce valid sidecars.
//! 2. The `.meta` TOML correctly records `[priors] type = "bed"`, `bed = <path>`,
//!    and `weight = 10.0`; uniform builds record `type = "uniform"` with no
//!    additional fields.
//! 3. The TOML round-trips cleanly.
//! 4. `prior_from_cli` rejects non-positive weights and accepts valid ones.

use prmi::sidecar::meta::Meta;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::TrainerConfig;
use prmi::train::mask::{parse_bed, BedInterval, MaskConfig};
use prmi::train::prior::Prior;
use prmi::train::prior_from_cli;
use std::io::Write;
use tempfile::tempdir;

/// Generate a deterministic pseudo-random DNA sequence of `len` bases.
fn pseudo_random_dna(len: usize, seed: u64) -> Vec<u8> {
    let bases = b"ACGT";
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bases[((state >> 30) & 3) as usize]
        })
        .collect()
}

/// Build a FASTA with `genome_len` pseudo-random bases.
fn build_fasta(genome_len: usize, seed: u64) -> Vec<u8> {
    let mut fa = b">chr1\n".to_vec();
    fa.extend(pseudo_random_dna(genome_len, seed));
    fa.push(b'\n');
    fa
}

// ---------------------------------------------------------------------------
// Test 1: BED-prior sidecar builds and records meta correctly
// ---------------------------------------------------------------------------

#[test]
fn bed_prior_builds_and_records_meta() {
    let dir = tempdir().unwrap();

    // 2048 bp pseudo-random genome.
    let genome_len = 2048usize;
    let fa = dir.path().join("test.fa");
    std::fs::write(&fa, build_fasta(genome_len, 0xDEAD_BEEF)).unwrap();

    // BED covers [256, 512) — ~12.5% of the genome.
    let bed_start = 256u64;
    let bed_end = 512u64;
    let bed = dir.path().join("prior.bed");
    {
        let mut f = std::fs::File::create(&bed).unwrap();
        writeln!(f, "chr1\t{bed_start}\t{bed_end}").unwrap();
    }

    // Build the uniform sidecar.
    let prefix_uniform = dir.path().join("uniform.prmi");
    build_sidecar_with_config(
        &fa,
        &prefix_uniform,
        Some(16),
        MaskConfig::default(),
        1,
        None,
    )
    .unwrap();

    // Build the BED-prior sidecar.
    let prefix_bed = dir.path().join("bed.prmi");
    let intervals = vec![BedInterval {
        start: bed_start,
        end: bed_end,
    }];
    let mut trainer_config = TrainerConfig::default();
    trainer_config.prior = Prior::Bed {
        intervals,
        weight: 10.0,
        path: Some(bed.clone()),
    };
    build_sidecar_with_config(
        &fa,
        &prefix_bed,
        Some(16),
        MaskConfig::default(),
        1,
        Some(trainer_config),
    )
    .unwrap();

    // Verify the meta TOML records the prior correctly for uniform.
    let meta_uniform = Meta::read_file(&SidecarPaths::from_prefix(&prefix_uniform).meta).unwrap();
    assert_eq!(meta_uniform.priors.kind, "uniform");
    assert!(
        meta_uniform.priors.bed.is_none(),
        "uniform prior should not record bed path"
    );
    assert!(
        meta_uniform.priors.weight.is_none(),
        "uniform prior should not record weight"
    );

    // Verify the meta TOML records the prior correctly for bed.
    let meta_bed = Meta::read_file(&SidecarPaths::from_prefix(&prefix_bed).meta).unwrap();
    assert_eq!(meta_bed.priors.kind, "bed");
    assert!(
        meta_bed.priors.bed.is_some(),
        "bed prior must record the BED path in [priors].bed"
    );
    assert!(
        (meta_bed.priors.weight.unwrap() - 10.0).abs() < 1e-9,
        "bed prior must record weight = 10.0"
    );

    // Both sidecars must produce finite error bounds.
    assert!(
        meta_uniform.rmi.max_error_bound > 0,
        "uniform sidecar should have non-zero error bound"
    );
    assert!(
        meta_bed.rmi.max_error_bound > 0,
        "BED-prior sidecar should have non-zero error bound"
    );

    // Sanity: the BED prior must not explode the global error bound more than
    // 3× the uniform bound. The prior redistributes error (tighter in-BED,
    // potentially looser out-of-BED), but the global bound should remain
    // well-controlled.
    assert!(
        meta_bed.rmi.max_error_bound <= meta_uniform.rmi.max_error_bound * 3,
        "BED prior must not explode global error bound \
         (uniform={}, bed={})",
        meta_uniform.rmi.max_error_bound,
        meta_bed.rmi.max_error_bound
    );

    // Verify all four sidecar files exist for the BED-prior build.
    let paths_bed = SidecarPaths::from_prefix(&prefix_bed);
    assert!(paths_bed.meta.exists(), ".meta missing");
    assert!(paths_bed.sa.exists(), ".sa missing");
    assert!(paths_bed.l1.exists(), ".l1 missing");
    assert!(paths_bed.l2.exists(), ".l2 missing");
}

// ---------------------------------------------------------------------------
// Test 2: Meta TOML round-trips for [priors] type = "bed"
// ---------------------------------------------------------------------------

#[test]
fn bed_prior_meta_toml_roundtrip() {
    let dir = tempdir().unwrap();

    let fa = dir.path().join("rt.fa");
    std::fs::write(&fa, build_fasta(1024, 0xABCD_1234)).unwrap();

    let bed = dir.path().join("rt.bed");
    {
        let mut f = std::fs::File::create(&bed).unwrap();
        writeln!(f, "chr1\t100\t300").unwrap();
    }

    let prefix = dir.path().join("rt.prmi");
    let intervals = parse_bed(&bed).unwrap();
    let mut trainer_config = TrainerConfig::default();
    trainer_config.prior = Prior::Bed {
        intervals,
        weight: 5.0,
        path: Some(bed.clone()),
    };
    build_sidecar_with_config(
        &fa,
        &prefix,
        Some(16),
        MaskConfig::default(),
        1,
        Some(trainer_config),
    )
    .unwrap();

    let paths = SidecarPaths::from_prefix(&prefix);
    let meta = Meta::read_file(&paths.meta).unwrap();

    assert_eq!(meta.priors.kind, "bed");
    assert!(meta.priors.bed.is_some());
    assert!((meta.priors.weight.unwrap() - 5.0).abs() < 1e-9);

    // Round-trip the TOML string.
    let toml_str = meta.to_toml().unwrap();
    let re_parsed = Meta::from_toml_str(&toml_str).unwrap();
    assert_eq!(re_parsed.priors.kind, "bed");
    assert_eq!(re_parsed.priors.bed, meta.priors.bed);
    assert!((re_parsed.priors.weight.unwrap() - 5.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Test 3: prior_from_cli accepts valid paths and rejects non-positive weight
// ---------------------------------------------------------------------------

#[test]
fn prior_from_cli_validates_weight() {
    let dir = tempdir().unwrap();
    let bed = dir.path().join("targets.bed");
    {
        let mut f = std::fs::File::create(&bed).unwrap();
        writeln!(f, "chr1\t0\t100").unwrap();
    }

    // Valid call: positive weight.
    let prior = prior_from_cli(Some(&bed), 10.0).unwrap();
    assert!(matches!(prior, Prior::Bed { .. }));

    // Valid call: no BED path → Uniform.
    let prior_none = prior_from_cli(None, 10.0).unwrap();
    assert!(matches!(prior_none, Prior::Uniform));

    // Invalid: zero weight.
    let err = prior_from_cli(Some(&bed), 0.0).unwrap_err();
    assert!(format!("{err}").contains("prior-bed-weight"));

    // Invalid: negative weight.
    let err2 = prior_from_cli(Some(&bed), -1.0).unwrap_err();
    assert!(format!("{err2}").contains("prior-bed-weight"));
}
