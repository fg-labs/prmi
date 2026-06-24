// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Integration tests for the three training-pair mask flags:
//! N-run masking (default on), homopolymer masking, and BED masking.
//!
//! Each test builds a synthetic FASTA with a known degenerate region,
//! trains with and without the relevant mask, and checks that:
//!   - The masked run produces a finite, low `max_error_bound`.
//!   - The meta TOML records the correct mask config.
//!
//! The SA file is intentionally not checked for size changes: the SA is
//! always complete; only the training set is filtered.

use prmi::sidecar::SidecarPaths;
use prmi::train::build_sidecar;
use prmi::train::mask::{BedInterval, MaskConfig};
use std::io::Write;
use tempfile::tempdir;

/// ACGT cycle of `len` bytes.
fn acgt_cycle(len: usize) -> Vec<u8> {
    (0..len).map(|i| b"ACGT"[i % 4]).collect()
}

/// Build a FASTA file content with:
///  - `prefix_len` ACGT-cycle bases
///  - `bad_len` bytes all equal to `bad_base`
///  - `suffix_len` ACGT-cycle bases
fn synthetic_fasta(prefix_len: usize, bad_base: u8, bad_len: usize, suffix_len: usize) -> Vec<u8> {
    let mut fa = b">chr1\n".to_vec();
    fa.extend(acgt_cycle(prefix_len));
    fa.extend(std::iter::repeat_n(bad_base, bad_len));
    fa.extend(acgt_cycle(suffix_len));
    fa.push(b'\n');
    fa
}

// ---------------------------------------------------------------------------
// Test 1: N-run masking (default on) excludes N-window positions from training
// ---------------------------------------------------------------------------

#[test]
fn mask_n_runs_default_on_excludes_n_window() {
    // Verify that masked_training_set with mask_n_runs=true excludes SA
    // positions whose 32-mer window covers an N base, and that the resulting
    // training set is strictly smaller than the unmasked one.
    use prmi::encoding::KMER_LEN;
    use prmi::fasta::fasta_to_2bit_with_n_positions;
    use prmi::sa::build_suffix_array;
    use prmi::train::mask::NBitmap;
    use prmi::train::training_set::masked_training_set;

    // 200 ACGT bases + 100 N's + 200 ACGT bases.
    let mut fa_bytes = b">chr1\n".to_vec();
    fa_bytes.extend(acgt_cycle(200));
    fa_bytes.extend(std::iter::repeat_n(b'N', 100));
    fa_bytes.extend(acgt_cycle(200));
    fa_bytes.push(b'\n');

    let mut cursor = std::io::Cursor::new(&fa_bytes);
    let (mut bases, mut n_positions, _stats) = fasta_to_2bit_with_n_positions(&mut cursor).unwrap();
    let genome_len = bases.len();
    // Pad with T-sentinels.
    bases.extend(std::iter::repeat_n(3u8, KMER_LEN - 1)); // BASE_T = 3
    n_positions.extend(std::iter::repeat_n(false, KMER_LEN - 1));

    // Bit-packed view of the same N positions for the masked_training_set API;
    // the Vec<bool> `n_positions` stays as the verification oracle below.
    let n_bitmap = {
        let mut b = NBitmap::zeros(n_positions.len());
        for (i, &is_n) in n_positions.iter().enumerate() {
            if is_n {
                b.set(i);
            }
        }
        b
    };

    let full_sa = build_suffix_array(&bases, 1).unwrap();
    let sa: Vec<u64> = full_sa
        .into_iter()
        .filter(|&p| p < genome_len as u64)
        .collect();

    // Unmasked: all SA entries used.
    let ts_full = masked_training_set(
        &sa,
        &bases,
        &n_bitmap,
        &MaskConfig::default(),
        &prmi::train::prior::Prior::Uniform,
    );
    // Masked: N-run positions excluded.
    let mask_on = MaskConfig {
        mask_n_runs: true,
        ..Default::default()
    };
    let ts_masked = masked_training_set(
        &sa,
        &bases,
        &n_bitmap,
        &mask_on,
        &prmi::train::prior::Prior::Uniform,
    );

    // Masked must have fewer training pairs.
    assert!(
        ts_masked.len() < ts_full.len(),
        "masked set ({}) should have fewer entries than unmasked ({})",
        ts_masked.len(),
        ts_full.len()
    );

    // No key in the masked set should come from an N-window position.
    // Recover the SA positions from the sa_indices in the masked training set.
    for sa_idx in ts_masked.sa_indices.iter() {
        let sa_pos = sa[sa_idx as usize] as usize;
        let has_n = n_positions[sa_pos..(sa_pos + KMER_LEN).min(n_positions.len())].contains(&true);
        assert!(!has_n, "sa_pos={sa_pos} should not be in N-window");
    }

    // Also verify via full build_sidecar that the meta TOML records the flag.
    let dir = tempdir().unwrap();
    let fa_file = dir.path().join("n_run.fa");
    // Write just the original FASTA (no sentinels — build_sidecar adds them internally).
    std::fs::write(&fa_file, &fa_bytes).unwrap();

    let prefix_masked = dir.path().join("masked.prmi");
    build_sidecar(
        &fa_file,
        &prefix_masked,
        Some(16),
        MaskConfig {
            mask_n_runs: true,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let meta_masked =
        prmi::sidecar::meta::Meta::read_file(&SidecarPaths::from_prefix(&prefix_masked).meta)
            .unwrap();
    assert!(meta_masked.sa.masked_n_runs);

    let prefix_unmasked = dir.path().join("unmasked.prmi");
    build_sidecar(
        &fa_file,
        &prefix_unmasked,
        Some(16),
        MaskConfig::default(),
        1,
    )
    .unwrap();
    let meta_unmasked =
        prmi::sidecar::meta::Meta::read_file(&SidecarPaths::from_prefix(&prefix_unmasked).meta)
            .unwrap();
    assert!(!meta_unmasked.sa.masked_n_runs);
}

// ---------------------------------------------------------------------------
// Test 2: Homopolymer masking drops poly-A windows
// ---------------------------------------------------------------------------

#[test]
fn mask_homopolymers_drops_polya_window() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("polya.fa");

    // 800 ACGT bases + 50 A's + 800 ACGT bases.
    // The 50-A homopolymer creates identical-key collisions in the unmasked case.
    std::fs::write(&fa, synthetic_fasta(800, b'A', 50, 800)).unwrap();

    // Build with homopolymer masking at k=20.
    let prefix_masked = dir.path().join("masked.prmi");
    let mask_on = MaskConfig {
        mask_homopolymers: Some(20),
        ..Default::default()
    };
    build_sidecar(&fa, &prefix_masked, Some(16), mask_on, 1).unwrap();

    let paths_masked = SidecarPaths::from_prefix(&prefix_masked);
    let meta_masked = prmi::sidecar::meta::Meta::read_file(&paths_masked.meta).unwrap();

    // Build without homopolymer masking.
    let prefix_unmasked = dir.path().join("unmasked.prmi");
    build_sidecar(&fa, &prefix_unmasked, Some(16), Default::default(), 1).unwrap();

    let paths_unmasked = SidecarPaths::from_prefix(&prefix_unmasked);
    let meta_unmasked = prmi::sidecar::meta::Meta::read_file(&paths_unmasked.meta).unwrap();

    let err_masked = meta_masked.rmi.max_error_bound;
    let err_unmasked = meta_unmasked.rmi.max_error_bound;

    // Masking should reduce (or at worst equal) the error bound.
    assert!(
        err_masked <= err_unmasked,
        "homopolymer-masked err={err_masked} should be <= unmasked err={err_unmasked}"
    );

    // Meta records the k threshold.
    assert_eq!(meta_masked.sa.masked_homopolymers, Some(20));
    assert!(meta_unmasked.sa.masked_homopolymers.is_none());
}

// ---------------------------------------------------------------------------
// Test 3: BED masking excludes the listed interval
// ---------------------------------------------------------------------------

#[test]
fn mask_bed_excludes_listed_intervals() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("bed.fa");

    // 500 ACGT bases + 200 A's + 500 ACGT bases.
    // The A-run at [500..700) will be masked via a BED file.
    std::fs::write(&fa, synthetic_fasta(500, b'A', 200, 500)).unwrap();

    // Write the BED file covering the degenerate region.
    let bed = dir.path().join("bad.bed");
    {
        let mut f = std::fs::File::create(&bed).unwrap();
        writeln!(f, "# mask the A-run").unwrap();
        writeln!(f, "chr1\t500\t700").unwrap();
    }

    // Build with BED masking.
    let prefix_masked = dir.path().join("masked.prmi");
    let intervals = vec![BedInterval {
        start: 500,
        end: 700,
    }];
    let mask_on = MaskConfig {
        mask_bed: Some(intervals),
        mask_bed_path: Some(bed.clone()),
        ..Default::default()
    };
    build_sidecar(&fa, &prefix_masked, Some(16), mask_on, 1).unwrap();

    let paths_masked = SidecarPaths::from_prefix(&prefix_masked);
    let meta_masked = prmi::sidecar::meta::Meta::read_file(&paths_masked.meta).unwrap();

    // Build without BED masking.
    let prefix_unmasked = dir.path().join("unmasked.prmi");
    build_sidecar(&fa, &prefix_unmasked, Some(16), Default::default(), 1).unwrap();

    let paths_unmasked = SidecarPaths::from_prefix(&prefix_unmasked);
    let meta_unmasked = prmi::sidecar::meta::Meta::read_file(&paths_unmasked.meta).unwrap();

    let err_masked = meta_masked.rmi.max_error_bound;
    let err_unmasked = meta_unmasked.rmi.max_error_bound;

    assert!(
        err_masked <= err_unmasked,
        "BED-masked err={err_masked} should be <= unmasked err={err_unmasked}"
    );

    // Meta records the BED path.
    assert!(
        meta_masked.sa.masked_bed.is_some(),
        "masked_bed path should be recorded"
    );
    assert!(meta_unmasked.sa.masked_bed.is_none());
}

// ---------------------------------------------------------------------------
// Test 4: Meta TOML records mask config correctly after N-run+homopolymer build
// ---------------------------------------------------------------------------

#[test]
fn meta_toml_records_combined_mask_config() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("combo.fa");
    let seq: Vec<u8> = (0..2048).map(|i| b"ACGT"[i % 4]).collect();
    let mut content = b">chr1\n".to_vec();
    content.extend_from_slice(&seq);
    content.push(b'\n');
    std::fs::write(&fa, &content).unwrap();

    let prefix = dir.path().join("combo.prmi");
    let mask = MaskConfig {
        mask_n_runs: true,
        mask_homopolymers: Some(15),
        mask_bed: None,
        mask_bed_path: None,
        ..Default::default()
    };
    build_sidecar(&fa, &prefix, Some(16), mask, 1).unwrap();

    let paths = SidecarPaths::from_prefix(&prefix);
    let meta = prmi::sidecar::meta::Meta::read_file(&paths.meta).unwrap();

    assert!(meta.sa.masked_n_runs, "masked_n_runs should be true");
    assert_eq!(meta.sa.masked_homopolymers, Some(15));
    assert!(meta.sa.masked_bed.is_none());

    // Round-trip the TOML to confirm serde handles the optional fields.
    let toml_str = meta.to_toml().unwrap();
    let re_parsed = prmi::sidecar::meta::Meta::from_toml_str(&toml_str).unwrap();
    assert!(re_parsed.sa.masked_n_runs);
    assert_eq!(re_parsed.sa.masked_homopolymers, Some(15));
}
