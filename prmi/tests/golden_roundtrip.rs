// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::encoding::tokenize_32mer;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use tempfile::tempdir;

fn deterministic_fasta(n_bases: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut s = b">chr1\n".to_vec();
    for _ in 0..n_bases {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.push(b"ACGT"[(x & 3) as usize]);
    }
    s.push(b'\n');
    s
}

#[test]
#[ignore = "forward-only primitive replaced by 2x spectrum in Plan 3"]
fn every_suffix_predicted_within_error_bound() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("g.fa");
    std::fs::write(&fa, deterministic_fasta(4096, 0xDEAD_BEEF)).unwrap();
    let prefix = dir.path().join("g.fa.prmi");
    build_sidecar(&fa, &prefix, Some(64), Default::default(), 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();

    // Reconstruct the bases the trainer saw by re-reading the FASTA.
    let (bases_seen, _) = prmi::fasta::fasta_file_to_2bit(&fa).unwrap();

    let max_err = idx.max_error_bound();
    for i in 0..idx.sa_num() {
        let sa_pos = idx.sa_position_for(i);
        let avail = bases_seen.len().saturating_sub(sa_pos as usize).min(32);
        let key = tokenize_32mer(&bases_seen[sa_pos as usize..sa_pos as usize + avail], avail);
        let (pred, _err) = idx.lookup(key);
        let dist = (pred as i64 - i as i64).unsigned_abs();
        assert!(
            dist <= max_err,
            "i={i} sa_pos={sa_pos} dist={dist} max_err={max_err}"
        );
    }
}

/// Originally added as a regression guard against BWA-MEME-packed err
/// values (the upstream 4-field `min_flag<<62 | min_err<<32 | max_flag<<31
/// | max_err` packing that leaked into our sidecar pre-Phase-5-rev and
/// produced err values around 4.6e18). Under Phase 5-rev's Fulcrum
/// trainer the err is always a scalar symmetric radius, so this test
/// passes trivially. Kept as defense-in-depth: anything that
/// reintroduces packed err semantics (a BWA-MEME-style refactor, a
/// merge from a divergent fork, an accidental bit shift) will flunk
/// here long before downstream consumers notice runaway err.
#[test]
#[ignore = "forward-only primitive replaced by 2x spectrum in Plan 3"]
fn decoded_err_values_are_sane() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("e.fa");
    std::fs::write(&fa, deterministic_fasta(4096, 0xCAFE_BABE)).unwrap();
    let prefix = dir.path().join("e.fa.prmi");
    build_sidecar(&fa, &prefix, Some(64), Default::default(), 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();
    let sa_num = idx.sa_num();
    let max_err = idx.max_error_bound();

    // Reconstruct keys, run lookups, and assert each returned `err` is a
    // reasonable scalar — not BWA-MEME's packed 4.6e18 monster, and not
    // wildly larger than the global max_error_bound (which is the worst
    // observed error across the entire training set).
    let (bases_seen, _) = prmi::fasta::fasta_file_to_2bit(&fa).unwrap();
    for i in 0..sa_num {
        let sa_pos = idx.sa_position_for(i);
        let avail = bases_seen.len().saturating_sub(sa_pos as usize).min(32);
        let key = tokenize_32mer(&bases_seen[sa_pos as usize..sa_pos as usize + avail], avail);
        let (_pred, err) = idx.lookup(key);
        // err can legitimately reach max_error_bound * 2 in pathological cases
        // (asymmetric distributions), but should never approach 2^31. A bound
        // of 10x is generous slack while catching the 4.6e18 regression.
        assert!(
            err <= max_err.saturating_mul(10) || err < (1u64 << 31),
            "i={i} err={err} max_error_bound={max_err} — looks unpacked-BWA-MEME"
        );
    }
}
