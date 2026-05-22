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
fn every_suffix_predicted_within_error_bound() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("g.fa");
    std::fs::write(&fa, deterministic_fasta(4096, 0xDEAD_BEEF)).unwrap();
    let prefix = dir.path().join("g.fa.prmi");
    build_sidecar(&fa, &prefix, 64).unwrap();

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
