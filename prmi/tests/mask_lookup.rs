// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! End-to-end correctness tests for masked-region lookup.
//!
//! The mask flag filters TRAINING data only; the SA on disk contains every
//! suffix. Queries against masked-position 32-mers must still succeed via
//! `smem_range`. This is the regression test for opus-pass2 finding #1.
//!
//! The commit `a604912` introduced two bugs in the LBC neighbor-correction
//! that together cause `smem_range` to return `l = 0` for queries against
//! masked-region 32-mers:
//!
//! **Bug 1 — Empty leaf case (main bug for this test fixture):**
//! When masking is active, keys from the masked region are absent from
//! training. If the masked region's keys lex-sort into a range where no other
//! training keys exist, the corresponding L2 leaf is empty. Empty leaves emit
//! `err = 0`, meaning `smem_range` searches only a single SA slot. The true
//! SA position of a masked-region query routing to this leaf is anywhere in
//! the gap `[0, next_sa_idx - 1]`, which the zero-error window misses.
//! Fix: set `err = next_sa_idx` for empty leaves so the window covers
//! `[0, 2 * next_sa_idx + 1)`.
//!
//! **Bug 2 — Sentinel check in `fit_direct_leaf`:**
//! The old check `if next_idx < ts_len` compared the SA-index value stored
//! by the LBC against the training-set length. Under masking, real SA indices
//! can be >= ts_len, causing the correction to be skipped for those leaves.
//! Fix: check `next_key != u64::MAX` (the LBC's sentinel key) instead.
//! Additionally, the correction itself used `ts.sa_indices[end-1]` (the last
//! TRAINING SA index) instead of `next_sa_idx - 1` (the position just below
//! the next leaf's first training pair); under masking the former is strictly
//! tighter and misses the masked gap.
//!
//! Both bugs are present in `a604912`; this regression test exercises both
//! by building a 5000-base reference, masking [2000..2050), and verifying
//! that `smem_range` returns a non-empty result for the distinctive 32-mer
//! at position 2000.

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use prmi::train::mask::{BedInterval, MaskConfig};
use std::io::Write as IoWrite;
use tempfile::tempdir;

/// Build a synthetic reference where:
///   [0..2000)     — deterministic ACGT-cycle pattern A (never repeats a 32-mer)
///   [2000..2050)  — a distinctive 50-base block unique in the reference
///   [2050..5000)  — deterministic CTAG-cycle pattern B (never repeats a 32-mer)
///
/// Mask [2000..2050) via BED. The distinctive 32-mer at position 2000 is
/// absent from training but present in the SA. `smem_range` must find it.
///
/// This test would FAIL against the regressed code (commit a604912) because
/// `ts.sa_indices[end - 1]` under-tightens the per-leaf err for the leaf whose
/// training range falls just before the masked region, making the search window
/// too narrow to cover SA positions in the gap.
///
/// **Coverage note (opus-pass3 finding #3):** this fixture exercises sub-bugs B
/// (sentinel check in `fit_direct_leaf`) and C (`next_sa_idx - 1` vs
/// `ts.sa_indices[end-1]` in the LBC correction) but NOT sub-bug A (empty-leaf
/// `err = 0`). The masked 32-mer at position 2000 starts with AAAA… (2-bit
/// top-4-bits = 0000), routing to L2 leaf 0. Leaf 0 is non-empty because
/// the ACGT-cycle pattern A contributes training keys starting with A to leaf 0.
/// For sub-bug A coverage see `smem_range_resolves_empty_leaf_query`.
#[test]
fn smem_range_resolves_masked_region_query() {
    let dir = tempdir().unwrap();
    let fa_path = dir.path().join("masked_query.fa");

    // Build a 5000-base reference in 2-bit integer encoding (0=A,1=C,2=G,3=T).
    let mut bases_2bit: Vec<u8> = Vec::with_capacity(5000);

    // [0..2000): ACGT cycle — unique 32-mers (period 4, length 2000 >> 32).
    for i in 0..2000usize {
        bases_2bit.push((i % 4) as u8);
    }

    // [2000..2050): distinctive block — AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC
    // This gives a unique 32-mer at position 2000 that will be masked.
    let distinctive_2bit: &[u8] = &[
        0, 0, 0, 0, // AAAA
        1, 1, 1, 1, // CCCC
        2, 2, 2, 2, // GGGG
        3, 3, 3, 3, // TTTT
        0, 0, 0, 0, // AAAA
        1, 1, 1, 1, // CCCC
        2, 2, 2, 2, // GGGG
        3, 3, 3, 3, // TTTT
        0, 0, 0, 0, // AAAA
        1, 1, 1, 1, // CCCC
        2, 2, // CC  (total: 42 bases — far more than 32, gives unique 32-mer)
    ];
    // Pad to exactly 50 bases with a distinct terminator pattern.
    let mut distinctive_block = distinctive_2bit.to_vec();
    while distinctive_block.len() < 50 {
        distinctive_block.push(3); // T
    }
    bases_2bit.extend_from_slice(&distinctive_block);
    assert_eq!(bases_2bit.len(), 2050);

    // [2050..5000): CTAG cycle — different from pattern A to avoid key collisions.
    for i in 0..(5000 - 2050) {
        bases_2bit.push([1u8, 3, 0, 2][i % 4]); // C T A G
    }
    assert_eq!(bases_2bit.len(), 5000);

    // Convert 2-bit to ASCII ACGT for FASTA writing.
    fn to_base(b: u8) -> u8 {
        b"ACGT"[b as usize]
    }
    let bases_ascii: Vec<u8> = bases_2bit.iter().map(|&b| to_base(b)).collect();

    // Write FASTA.
    {
        let mut f = std::fs::File::create(&fa_path).unwrap();
        writeln!(f, ">masked_query_test").unwrap();
        f.write_all(&bases_ascii).unwrap();
        writeln!(f).unwrap();
    }

    // Write BED file masking [2000, 2050).
    let bed_path = dir.path().join("mask.bed");
    {
        let mut bedf = std::fs::File::create(&bed_path).unwrap();
        writeln!(bedf, "masked_query_test\t2000\t2050").unwrap();
    }

    let prefix = dir.path().join("masked_query.fa.prmi");

    // Build the sidecar with BED masking.
    let mask = MaskConfig {
        mask_n_runs: false, // no N in this synthetic reference
        mask_homopolymers: None,
        mask_bed: Some(vec![BedInterval {
            start: 2000,
            end: 2050,
        }]),
        mask_bed_path: Some(bed_path),
    };
    // Use a small l2_leaf_count so that the masked gap is large relative to
    // leaf width — this maximises the chance that the masked 32-mer at position
    // 2000 falls in an inter-leaf gap rather than inside a leaf.
    build_sidecar(&fa_path, &prefix, Some(16), mask, 1).unwrap();

    // Open the sidecar.
    let idx = LearnedIndex::open(&prefix).unwrap();

    // Query: the distinctive 32-mer starting at position 2000 (masked region).
    let query_2bit: &[u8] = &distinctive_2bit[..32];

    // The pac passed to smem_range is 1-base-per-byte (values 0..=3).
    let pac: &[u8] = &bases_2bit;

    let (k, l, s) = idx.smem_range(query_2bit, pac).unwrap();

    // The query 32-mer exists in the SA (the SA is never masked). smem_range
    // must find it even though the position was excluded from training.
    assert!(
        l > 0,
        "smem_range returned l=0 for a masked-region query — \
         this is the opus-pass2 #1 regression bug \
         (ts.sa_indices[end-1] under-tightens the LBC neighbor-correction bound). \
         k={k} l={l} s={s}"
    );
    assert!(
        s >= 32,
        "matched seed length should be at least 32 (got s={s}); k={k} l={l}"
    );
}

/// Regression test for sub-bug A (opus-pass3 finding #3): empty-leaf `err = 0`.
///
/// Sub-bug A fires when the masked region's keys lex-sort into an inter-leaf
/// gap where NO other training keys exist — the corresponding L2 leaf is empty.
/// Before the fix, empty leaves emitted `err = 0`, producing a 1-slot search
/// window that misses the true SA position of masked-region queries routed there.
///
/// **Fixture design:** with `l2_leaf_count = 16` (bit_shift = 60):
///   - 2-bit key value 0 (all-A) routes to L2 leaf 0.
///   - 2-bit key value 1 (all-C) encodes as 0x5555_5555_5555_5555;
///     `0x5555…5555 >> 60 = 5`, so all-C 32-mers route to leaf 5.
///
/// Reference layout:
///   [0..4096)    — all A (key = 0, routes to leaf 0)
///   [4096..4128) — all C (32 bases, key → leaf 5; masked → leaf 5 is empty)
///   [4128..6144) — all A again (key = 0, routes to leaf 0)
///
/// With the C-block masked, leaf 5 has no training pairs (it is a true empty
/// leaf, not a trailing one) but the 32 positions [4096..4128) exist in the SA.
/// A query of 32 C's must produce `l > 0`.
///
/// This test would FAIL against the pre-fix code (empty-leaf `err = 0`) because
/// the search window would be [ts.len(), ts.len() + 1), which is outside the SA.
/// It passes with the pass-3 fix (`err = next_sa_idx`).
#[test]
fn smem_range_resolves_empty_leaf_query() {
    let dir = tempdir().unwrap();
    let fa_path = dir.path().join("empty_leaf.fa");

    // Build a 6144-base reference: 4096 A's, 32 C's (masked), 2016 A's.
    // The C-block at [4096..4128) gives a 32-mer of all-C at position 4096.
    // With l2_leaf_count=16 the all-C key routes to leaf 5; no other training
    // pair routes to leaf 5 (all A's → leaf 0), so leaf 5 is empty after masking.
    let mut bases_2bit: Vec<u8> = vec![0u8; 4096]; // A
    bases_2bit.resize(4128, 1u8); // C (32 bases)
    bases_2bit.resize(6144, 0u8); // A again
    assert_eq!(bases_2bit.len(), 6144);

    fn to_base(b: u8) -> u8 {
        b"ACGT"[b as usize]
    }
    let bases_ascii: Vec<u8> = bases_2bit.iter().map(|&b| to_base(b)).collect();

    {
        let mut f = std::fs::File::create(&fa_path).unwrap();
        writeln!(f, ">empty_leaf_test").unwrap();
        f.write_all(&bases_ascii).unwrap();
        writeln!(f).unwrap();
    }

    // Mask the C-block [4096..4128).
    let bed_path = dir.path().join("empty_leaf.bed");
    {
        let mut bedf = std::fs::File::create(&bed_path).unwrap();
        writeln!(bedf, "empty_leaf_test\t4096\t4128").unwrap();
    }

    let prefix = dir.path().join("empty_leaf.fa.prmi");

    let mask = MaskConfig {
        mask_n_runs: false,
        mask_homopolymers: None,
        mask_bed: Some(vec![BedInterval {
            start: 4096,
            end: 4128,
        }]),
        mask_bed_path: Some(bed_path),
    };
    // l2_leaf_count=16 → bit_shift=60 → all-A keys (0) → leaf 0;
    // all-C keys (0x5555…5555) → leaf 5 (0x5555…5555 >> 60 == 5).
    build_sidecar(&fa_path, &prefix, Some(16), mask, 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();

    // Query: 32 C's (key routes to the empty leaf 5).
    let query_c: Vec<u8> = vec![1u8; 32]; // 32 C's (2-bit 1)
    let pac: &[u8] = &bases_2bit;

    let (k, l, s) = idx.smem_range(&query_c, pac).unwrap();

    // The C-block positions [4096..4128) exist in the SA (the SA is never masked).
    // smem_range must find the unique 32-mer at position 4096 even though it was
    // excluded from training.
    assert!(
        l > 0,
        "smem_range returned l=0 for an empty-leaf query — \
         this is sub-bug A (opus-pass3 #3): empty-leaf emits err=0, \
         missing the SA positions [prev_last_sa+1, next_sa_idx-1]. \
         k={k} l={l} s={s}"
    );
    assert!(
        s >= 32,
        "matched seed length should be at least 32 (got s={s}); k={k} l={l}"
    );
}
