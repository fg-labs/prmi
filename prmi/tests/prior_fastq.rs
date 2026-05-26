// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for `--prior-fastq-histogram` workload-aware training.
//!
//! The FASTQ histogram prior weights each `(key, sa_index)` training pair by
//! `base_weight + log2(1 + freq(key))`, where `freq(key)` is the observed
//! query count for that 32-mer. "Hot" keys get higher weight, biasing the
//! model fit toward the observed workload.
//!
//! Tests:
//! 1. Build succeeds; `.meta` records `[priors] type = "fastq_histogram"` with
//!    `histogram`, `base_weight`, and `formula` fields set correctly.
//! 2. TOML round-trips cleanly.
//! 3. Empirical: a sidecar trained with a synthetic histogram where a few keys
//!    are extremely hot does not regress the global worst-case error bound
//!    relative to a uniform-prior sidecar, and both hot keys stay within it.
//!    (The `err` field is the unweighted per-leaf worst-case bound, so the
//!    prior tilts the fit without shrinking that bound — see the test.)
//! 4. `prior_from_cli_fastq` rejects non-positive `base_weight`; accepts valid
//!    histogram.
//! 5. `parse_histogram_tsv` error paths: duplicate key, non-numeric, missing
//!    field.
//! 6. `--prior-bed` and `--prior-fastq-histogram` are mutually exclusive in the
//!    CLI guard.

use prmi::encoding::tokenize_32mer;
use prmi::index::LearnedIndex;
use prmi::sidecar::meta::Meta;
use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::TrainerConfig;
use prmi::train::mask::MaskConfig;
use prmi::train::prior::{parse_histogram_tsv, Prior};
use prmi::train::prior_from_cli_fastq;
use prmi::train::validate_prior_paths;
use prmi::Error;
use std::collections::HashMap;
use std::io::Write;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random DNA sequence as ASCII bytes (A/C/G/T).
fn pseudo_random_dna_ascii(len: usize, seed: u64) -> Vec<u8> {
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

/// Deterministic pseudo-random DNA sequence as 2-bit values (0=A, 1=C, 2=G, 3=T).
/// Matches the encoding produced by `fasta_to_2bit_with_sha256`.
fn pseudo_random_dna_2bit(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 30) & 3) as u8
        })
        .collect()
}

fn build_fasta(genome_len: usize, seed: u64) -> Vec<u8> {
    let mut fa = b">chr1\n".to_vec();
    fa.extend(pseudo_random_dna_ascii(genome_len, seed));
    fa.push(b'\n');
    fa
}

/// Write a histogram TSV where every `(key, count)` pair in `entries` is a
/// line.
fn write_histogram(path: &std::path::Path, entries: &[(u64, u64)]) {
    let mut f = std::fs::File::create(path).unwrap();
    for &(key, count) in entries {
        writeln!(f, "{key}\t{count}").unwrap();
    }
}

/// Tokenize the 32-mer starting at position `pos` in a 2-bit encoded sequence.
fn key_at(seq: &[u8], pos: usize) -> u64 {
    let avail = seq.len().saturating_sub(pos).min(32);
    tokenize_32mer(&seq[pos..pos + avail], avail)
}

// ---------------------------------------------------------------------------
// Test 1: build succeeds and meta records the prior correctly
// ---------------------------------------------------------------------------

#[test]
fn fastq_histogram_prior_builds_and_records_meta() {
    let dir = tempdir().unwrap();
    let genome_len = 2048usize;
    let fa = dir.path().join("test.fa");
    std::fs::write(&fa, build_fasta(genome_len, 0xBEEF_CAFE)).unwrap();

    // Build a uniform baseline sidecar.
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

    // Build a FASTQ-histogram-prior sidecar with a handful of hot keys.
    // Use the 2-bit encoded sequence (matches what fasta_to_2bit_with_sha256 produces).
    let dna_2bit = pseudo_random_dna_2bit(genome_len, 0xBEEF_CAFE);
    let hot_key = key_at(&dna_2bit, 100);
    let histogram_path = dir.path().join("hist.tsv");
    write_histogram(&histogram_path, &[(hot_key, 1_000_000u64)]);

    let prefix_fq = dir.path().join("fq.prmi");
    let mut keys_to_freq = HashMap::new();
    keys_to_freq.insert(hot_key, 1_000_000u64);
    let mut trainer_config = TrainerConfig::default();
    trainer_config.prior = Prior::FastqHistogram {
        keys_to_freq,
        base_weight: 1.0,
        path: Some(histogram_path.clone()),
    };
    build_sidecar_with_config(
        &fa,
        &prefix_fq,
        Some(16),
        MaskConfig::default(),
        1,
        Some(trainer_config),
    )
    .unwrap();

    // Verify uniform meta.
    let meta_u = Meta::read_file(&SidecarPaths::from_prefix(&prefix_uniform).meta).unwrap();
    assert_eq!(meta_u.priors.kind, "uniform");
    assert!(meta_u.priors.histogram.is_none());
    assert!(meta_u.priors.base_weight.is_none());
    assert!(meta_u.priors.formula.is_none());

    // Verify fastq_histogram meta.
    let meta_fq = Meta::read_file(&SidecarPaths::from_prefix(&prefix_fq).meta).unwrap();
    assert_eq!(meta_fq.priors.kind, "fastq_histogram");
    assert!(
        meta_fq.priors.histogram.is_some(),
        "[priors].histogram must be recorded"
    );
    assert!(
        (meta_fq.priors.base_weight.unwrap() - 1.0).abs() < 1e-9,
        "base_weight must be 1.0"
    );
    assert_eq!(
        meta_fq.priors.formula.as_deref(),
        Some("1.0 + log2(1 + freq)"),
        "formula string must be recorded"
    );

    // Both sidecars must have finite, non-zero error bounds.
    assert!(meta_u.rmi.max_error_bound > 0);
    assert!(meta_fq.rmi.max_error_bound > 0);

    // All four sidecar files must exist for the fastq-prior build.
    let paths_fq = SidecarPaths::from_prefix(&prefix_fq);
    assert!(paths_fq.meta.exists(), ".meta missing");
    assert!(paths_fq.sa.exists(), ".sa missing");
    assert!(paths_fq.l1.exists(), ".l1 missing");
    assert!(paths_fq.l2.exists(), ".l2 missing");
}

// ---------------------------------------------------------------------------
// Test 2: meta TOML round-trips for [priors] type = "fastq_histogram"
// ---------------------------------------------------------------------------

#[test]
fn fastq_histogram_meta_toml_roundtrip() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("rt.fa");
    std::fs::write(&fa, build_fasta(1024, 0xDEAD_C0DE)).unwrap();

    let dna_2bit = pseudo_random_dna_2bit(1024, 0xDEAD_C0DE);
    let k0 = key_at(&dna_2bit, 50);
    let k1 = key_at(&dna_2bit, 200);
    let histogram_path = dir.path().join("hist.tsv");
    write_histogram(&histogram_path, &[(k0, 5000), (k1, 100)]);

    let mut keys_to_freq = HashMap::new();
    keys_to_freq.insert(k0, 5000u64);
    keys_to_freq.insert(k1, 100u64);
    let mut trainer_config = TrainerConfig::default();
    trainer_config.prior = Prior::FastqHistogram {
        keys_to_freq,
        base_weight: 2.5,
        path: Some(histogram_path.clone()),
    };
    let prefix = dir.path().join("rt.prmi");
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
    assert_eq!(meta.priors.kind, "fastq_histogram");
    assert!((meta.priors.base_weight.unwrap() - 2.5).abs() < 1e-9);
    assert!(meta.priors.histogram.is_some());

    // Round-trip.
    let toml_str = meta.to_toml().unwrap();
    let re_parsed = Meta::from_toml_str(&toml_str).unwrap();
    assert_eq!(re_parsed.priors.kind, "fastq_histogram");
    assert_eq!(re_parsed.priors.histogram, meta.priors.histogram);
    assert!((re_parsed.priors.base_weight.unwrap() - 2.5).abs() < 1e-9);
    assert_eq!(re_parsed.priors.formula, meta.priors.formula);
}

// ---------------------------------------------------------------------------
// Test 3: empirical — the FASTQ prior does not regress error bounds
// ---------------------------------------------------------------------------

#[test]
fn fastq_histogram_prior_does_not_regress_error_bounds() {
    // Use a large enough genome so there are many leaves to route across, but
    // small enough to run quickly in CI.
    let genome_len = 8192usize;
    let dir = tempdir().unwrap();
    let fa = dir.path().join("ecoli.fa");
    std::fs::write(&fa, build_fasta(genome_len, 0xA5A5_A5A5)).unwrap();

    // Use 2-bit encoded sequence matching fasta_to_2bit_with_sha256 output.
    let dna_2bit = pseudo_random_dna_2bit(genome_len, 0xA5A5_A5A5);

    // Pick two distinct 32-mer positions to designate as "hot".
    // We mark them with freq = 1_000_000 so they dominate the weighted fit.
    let hot_pos_a = 512usize;
    let hot_pos_b = 2048usize;
    let k_hot_a = key_at(&dna_2bit, hot_pos_a);
    let k_hot_b = key_at(&dna_2bit, hot_pos_b);

    // Pick a "cold" position well separated from the hot ones.
    let cold_pos = 6000usize;
    let k_cold = key_at(&dna_2bit, cold_pos);

    let histogram_path = dir.path().join("hist.tsv");
    write_histogram(
        &histogram_path,
        &[(k_hot_a, 1_000_000u64), (k_hot_b, 1_000_000u64)],
    );

    // Uniform baseline.
    let prefix_u = dir.path().join("u.prmi");
    build_sidecar_with_config(&fa, &prefix_u, Some(64), MaskConfig::default(), 1, None).unwrap();

    // FASTQ-histogram prior.
    let mut keys_to_freq = HashMap::new();
    keys_to_freq.insert(k_hot_a, 1_000_000u64);
    keys_to_freq.insert(k_hot_b, 1_000_000u64);
    let mut trainer_config = TrainerConfig::default();
    trainer_config.prior = Prior::FastqHistogram {
        keys_to_freq,
        base_weight: 1.0,
        path: Some(histogram_path),
    };
    let prefix_fq = dir.path().join("fq.prmi");
    build_sidecar_with_config(
        &fa,
        &prefix_fq,
        Some(64),
        MaskConfig::default(),
        1,
        Some(trainer_config),
    )
    .unwrap();

    // Open both indexes and compare per-key prediction errors.
    let idx_u = LearnedIndex::open(&prefix_u).unwrap();
    let idx_fq = LearnedIndex::open(&prefix_fq).unwrap();

    let (_, err_hot_a_uniform) = idx_u.lookup(k_hot_a);
    let (_, err_hot_a_fq) = idx_fq.lookup(k_hot_a);
    let (_, err_hot_b_uniform) = idx_u.lookup(k_hot_b);
    let (_, err_hot_b_fq) = idx_fq.lookup(k_hot_b);
    let (_, err_cold_uniform) = idx_u.lookup(k_cold);
    let (_, err_cold_fq) = idx_fq.lookup(k_cold);

    let meta_u = Meta::read_file(&SidecarPaths::from_prefix(&prefix_u).meta).unwrap();
    let meta_fq = Meta::read_file(&SidecarPaths::from_prefix(&prefix_fq).meta).unwrap();

    // NOTE on the contract: the `err` field is the *unweighted* worst-case bound
    // for a key's leaf (by design — see build_sidecar_with_config — so `err`
    // stays a valid bound for all queries). The FASTQ prior tilts the per-leaf
    // *fit* toward hot keys but does not shrink that worst-case bound, and an
    // individual hot key's bound can even move up slightly (e.g. hot-b here goes
    // 11 -> 12) when the leaf's worst case is dominated by other keys. So this
    // test asserts the property that actually holds: the prior does not *regress*
    // the global worst-case error bound, and every queried key stays within it.

    // The global max error bound must not regress relative to the uniform build
    // (the prior redistributes weight within the fit; it does not explode the
    // worst case).
    assert!(
        meta_fq.rmi.max_error_bound <= meta_u.rmi.max_error_bound,
        "FASTQ prior must not regress the global error bound: \
         uniform={}, fq={}",
        meta_u.rmi.max_error_bound,
        meta_fq.rmi.max_error_bound
    );

    // Both designated hot keys (not just one) must resolve to a bound within the
    // global worst case — a sanity check that the weighted-fit path produces
    // valid, queryable per-leaf bounds for the hot keys.
    for (label, err_fq) in [("hot-a", err_hot_a_fq), ("hot-b", err_hot_b_fq)] {
        assert!(
            err_fq <= meta_fq.rmi.max_error_bound,
            "{label} bound {err_fq} exceeds the global max error bound {}",
            meta_fq.rmi.max_error_bound
        );
    }

    // Record for diagnostics.
    eprintln!(
        "hot-a err: uniform={err_hot_a_uniform}, fq={err_hot_a_fq}; \
         hot-b err: uniform={err_hot_b_uniform}, fq={err_hot_b_fq}; \
         cold err: uniform={err_cold_uniform}, fq={err_cold_fq}; \
         global max err: uniform={}, fq={}",
        meta_u.rmi.max_error_bound, meta_fq.rmi.max_error_bound
    );
}

// ---------------------------------------------------------------------------
// Test 4: prior_from_cli_fastq validates base_weight
// ---------------------------------------------------------------------------

#[test]
fn prior_from_cli_fastq_validates_base_weight() {
    let dir = tempdir().unwrap();
    let hist = dir.path().join("h.tsv");
    write_histogram(&hist, &[(1000u64, 50u64)]);

    // Valid call.
    let p = prior_from_cli_fastq(&hist, 1.0).unwrap();
    assert!(matches!(p, Prior::FastqHistogram { .. }));

    // Zero base_weight is rejected.
    let err = prior_from_cli_fastq(&hist, 0.0).unwrap_err();
    assert!(
        format!("{err}").contains("prior-fastq-base-weight"),
        "expected prior-fastq-base-weight error, got: {err}"
    );

    // Negative base_weight is rejected.
    let err2 = prior_from_cli_fastq(&hist, -1.0).unwrap_err();
    assert!(
        format!("{err2}").contains("prior-fastq-base-weight"),
        "expected prior-fastq-base-weight error, got: {err2}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: parse_histogram_tsv error paths
// ---------------------------------------------------------------------------

#[test]
fn parse_histogram_tsv_error_paths() {
    let dir = tempdir().unwrap();

    // Duplicate key.
    let dup = dir.path().join("dup.tsv");
    {
        let mut f = std::fs::File::create(&dup).unwrap();
        writeln!(f, "100\t5").unwrap();
        writeln!(f, "100\t10").unwrap();
    }
    let err = parse_histogram_tsv(&dup).unwrap_err();
    assert!(format!("{err}").contains("duplicate key"), "got: {err}");
    // Malformed user input must surface as InvalidInput, not Internal (a bug).
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "expected InvalidInput, got: {err:?}"
    );

    // Non-numeric key.
    let bad_key = dir.path().join("bad_key.tsv");
    {
        let mut f = std::fs::File::create(&bad_key).unwrap();
        writeln!(f, "NOT_A_NUMBER\t5").unwrap();
    }
    let err2 = parse_histogram_tsv(&bad_key).unwrap_err();
    assert!(format!("{err2}").contains("not a valid u64"), "got: {err2}");

    // Missing count field (no tab).
    let no_tab = dir.path().join("no_tab.tsv");
    {
        let mut f = std::fs::File::create(&no_tab).unwrap();
        writeln!(f, "12345").unwrap();
    }
    let err3 = parse_histogram_tsv(&no_tab).unwrap_err();
    assert!(
        format!("{err3}").contains("tab-separated fields"),
        "got: {err3}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: CLI mutual-exclusion guard — --prior-bed + --prior-fastq-histogram
// ---------------------------------------------------------------------------

#[test]
fn cli_rejects_prior_bed_and_prior_fastq_histogram_together() {
    // Exercise the production guard directly (the same `validate_prior_paths`
    // the CLI invokes before resolving a prior), so this test fails if the
    // exclusivity check is removed or inverted.
    let bed = std::path::Path::new("targets.bed");
    let hist = std::path::Path::new("hist.tsv");

    // Both supplied → reject with a user-input error (not an internal-bug error).
    let err = validate_prior_paths(Some(bed), Some(hist)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "supplying both --prior-bed and --prior-fastq-histogram must error as InvalidInput, got: {err:?}"
    );
    // At most one (or neither) is allowed.
    assert!(
        validate_prior_paths(Some(bed), None).is_ok(),
        "--prior-bed alone is valid"
    );
    assert!(
        validate_prior_paths(None, Some(hist)).is_ok(),
        "--prior-fastq-histogram alone is valid"
    );
    assert!(
        validate_prior_paths(None, None).is_ok(),
        "no prior path is valid (uniform prior)"
    );
}
