// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Scaled-up round-trip exercising the trainer's L1 fallback path on a
//! PhiX-sized reference (5386 bp, the actual PhiX-174 length).
//!
//! Smaller fixtures (4 KB synthetic) stay in `golden_roundtrip.rs`; this
//! file exercises one tier up.

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use tempfile::tempdir;

fn deterministic_fasta(n_bases: usize, seed: u64) -> Vec<u8> {
    let mut s = String::from(">phix_synth\n");
    let mut x = seed;
    for _ in 0..n_bases {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(b"ACGT"[(x & 3) as usize] as char);
    }
    s.push('\n');
    s.into_bytes()
}

#[test]
fn phix_scale_roundtrip_every_suffix_within_bound() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("phix.fa");
    std::fs::write(&fa, deterministic_fasta(5386, 0xCAFE_F00D)).unwrap();
    let prefix = dir.path().join("phix.fa.prmi");
    build_sidecar(&fa, &prefix, None, Default::default(), 1).unwrap(); // auto-scale l2_leaf_count

    let idx = LearnedIndex::open(&prefix).unwrap();
    let sa_num = idx.sa_num();
    let max_err = idx.max_error_bound();

    // Verify the auto-scale produced a non-trivial L2.
    assert!(
        idx.l2_leaf_count() >= 16,
        "expected l2_leaf_count >= 16, got {}",
        idx.l2_leaf_count()
    );

    // Every SA suffix's tokenized key must lookup within max_error_bound.
    // (Same property the smaller golden_roundtrip checks; doing it at
    // 5386 bp exercises L1 fallback on at least one dense leaf.)
    let (bases_seen, _) = prmi::fasta::fasta_file_to_2bit(&fa).unwrap();
    for i in 0..sa_num {
        let sa_pos = idx.sa_position_for(i);
        let avail = bases_seen.len().saturating_sub(sa_pos as usize).min(32);
        let key = prmi::encoding::tokenize_32mer(
            &bases_seen[sa_pos as usize..sa_pos as usize + avail],
            avail,
        );
        let (pred, _err) = idx.lookup(key);
        let dist = (pred as i64 - i as i64).unsigned_abs();
        assert!(
            dist <= max_err,
            "phix-scale: i={i} sa_pos={sa_pos} pred={pred} dist={dist} max_err={max_err}"
        );
    }
}

#[test]
fn phix_scale_smem_range_resolves_first_suffix() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("phix.fa");
    std::fs::write(&fa, deterministic_fasta(5386, 0xDEAD_BEEF_C0DE)).unwrap();
    let prefix = dir.path().join("phix.fa.prmi");
    build_sidecar(&fa, &prefix, None, Default::default(), 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();
    let (bases, _) = prmi::fasta::fasta_file_to_2bit(&fa).unwrap();

    // Take the first 32-base suffix; smem_range must return a non-empty
    // SA interval whose s (matched length) is at least 32.
    let query = bases[0..32].to_vec();
    let (k, l, s) = idx.smem_range(&query, &bases).unwrap();
    assert!(l > 0, "expected non-empty SA range at first suffix");
    assert!(s >= 32, "expected s >= 32, got {s}");
    assert!(
        k < idx.sa_num(),
        "k={k} out of range (sa_num={})",
        idx.sa_num()
    );
}
