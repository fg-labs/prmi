// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Regression test for the trailing-empty-leaf correctness gap (opus-pass3 finding #2).
//!
//! When masking is active, 32-mers whose keys lex-sort ABOVE every training key
//! route to a trailing empty leaf — one whose `lbc.next` returns the sentinel
//! `(ts.len(), u64::MAX)` because no next non-empty leaf exists.
//!
//! Before the pass-3 fix the empty-leaf branch emitted `err = 0` for the
//! sentinel case, producing a 1-slot search window at `min(ts.len(), sa_num-1)`.
//! For high-key masked queries the true SA position is in `[prev_sa+1, sa_num-1]`,
//! which the 1-slot window misses.
//!
//! **Test strategy:** build a reference where almost all bases are `A` (2-bit 0,
//! key bits all zero, routing to L2 leaf 0) and a short tail of `T` bases at the
//! high-key end. Mask the `T`-tail. A 32-mer composed entirely of `T` (key =
//! `u64::MAX >> 2`, high bits all 1) routes to the highest L2 leaf with a small
//! `l2_leaf_count`. That leaf is empty in training (no `T`-tail keys were trained)
//! but `T`-tail SA positions exist in the complete SA. `smem_range` must find them.

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use prmi::train::mask::{BedInterval, MaskConfig};
use std::io::Write as IoWrite;
use tempfile::tempdir;

/// Build a reference whose HIGHEST-key region is masked so that trailing
/// empty leaves receive queries at runtime.
///
/// Reference layout (all 2-bit encoded, converted to ASCII ACGT for FASTA):
///   [0..4096)  — all `A` (2-bit 0) → every 32-mer has key 0 → routes to leaf 0
///   [4096..4160) — all `T` (2-bit 3) → 32-mer key = 0xFFFF_FFFF_FFFF_FFFF
///                  (after the first 32 T's are in the window)
///
/// We use `l2_leaf_count = 16` so the key space is split into 16 buckets.
/// All-A keys route to bucket 0; all-T keys route to bucket 15. Masking
/// `[4096..4160)` removes the only training pairs that would populate bucket 15,
/// leaving it as a trailing empty leaf.
///
/// A query of 32 T's (key = `u64::MAX`) must produce `l > 0`.
#[test]
fn smem_range_resolves_trailing_empty_leaf_query() {
    let dir = tempdir().unwrap();
    let fa_path = dir.path().join("trailing_mask.fa");

    // Build a 4160-base reference: 4096 A's followed by 64 T's.
    // Using 4096 A's gives plenty of unique 32-mers (they're all the same key =
    // 0 but different SA positions; the trainer will fit them into leaf 0).
    // Using 64 T's ensures at least one 32-mer window is entirely within the T
    // region (at position 4096+32 = 4128 the window [4128..4160) is all T's).
    let mut bases_2bit: Vec<u8> = vec![0u8; 4096]; // A
    bases_2bit.resize(4160, 3u8); // T
    assert_eq!(bases_2bit.len(), 4160);

    fn to_base(b: u8) -> u8 {
        b"ACGT"[b as usize]
    }
    let bases_ascii: Vec<u8> = bases_2bit.iter().map(|&b| to_base(b)).collect();

    {
        let mut f = std::fs::File::create(&fa_path).unwrap();
        writeln!(f, ">trailing_mask_test").unwrap();
        f.write_all(&bases_ascii).unwrap();
        writeln!(f).unwrap();
    }

    // Mask the entire T-tail [4096..4160) via BED.
    let bed_path = dir.path().join("trailing_mask.bed");
    {
        let mut bedf = std::fs::File::create(&bed_path).unwrap();
        writeln!(bedf, "trailing_mask_test\t4096\t4160").unwrap();
    }

    let prefix = dir.path().join("trailing_mask.fa.prmi");

    let mask = MaskConfig {
        mask_n_runs: false,
        mask_homopolymers: None,
        mask_bed: Some(vec![BedInterval {
            start: 4096,
            end: 4160,
        }]),
        mask_bed_path: Some(bed_path),
    };
    // l2_leaf_count = 16 → bit_shift = 60 → all-A keys (0) → leaf 0;
    // all-T keys (u64::MAX) → leaf 15 (bucket index = u64::MAX >> 60 = 15).
    build_sidecar(&fa_path, &prefix, Some(16), mask, 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();

    // Query: 32 T's (key routes to the trailing empty leaf).
    let query_t: Vec<u8> = vec![3u8; 32]; // 32 T's (2-bit 3)
    let pac: &[u8] = &bases_2bit;

    let (k, l, s) = idx.smem_range(&query_t, pac).unwrap();

    // The T-tail positions exist in the SA (the SA is never masked).
    // smem_range must find at least one of them.
    //
    // Note: `s` may be < 32 for T-tail suffixes near the end of the reference
    // (e.g. position 4159 has only 1 T, so its common prefix with a 32-T query
    // is 1). We only assert l > 0 — finding any SA entry in the T-tail is the
    // correctness property this test exercises.
    assert!(
        l > 0,
        "smem_range returned l=0 for a trailing-empty-leaf query — \
         this is the opus-pass3 #2 trailing-empty-leaf bug \
         (err=0 emitted for sentinel case, missing [prev_sa+1, sa_num-1]). \
         k={k} l={l} s={s}"
    );
    // The SA entries found must be in the T-tail region [4096..4160).
    // A non-zero `s` confirms at least a 1-base match.
    assert!(s >= 1, "s should be at least 1 (got {s}); k={k} l={l}");
}
