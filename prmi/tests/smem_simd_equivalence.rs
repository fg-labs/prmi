// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! SIMD-vs-scalar equivalence tests for `smem_range`.
//!
//! For every query in a corpus, asserts that the SIMD-dispatched path
//! (`resolve_one` via `smem_range`) and the pure-scalar path
//! (`resolve_one_scalar`) return identical `(k, l, s)` results.

use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Build a synthetic FASTA of `n_bases` from a deterministic LCG sequence.
fn deterministic_fasta(n_bases: usize, seed: u64) -> Vec<u8> {
    let mut s = b">chr1\n".to_vec();
    let mut x = seed;
    for _ in 0..n_bases {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(b"ACGT"[(x >> 32 & 3) as usize]);
    }
    s.push(b'\n');
    s
}

/// Build an unpacked-base (values 0..=3) pac from a FASTA.
fn fasta_to_unpacked(fasta: &[u8]) -> Vec<u8> {
    let mut bases = Vec::new();
    for &b in fasta {
        match b {
            b'A' | b'a' => bases.push(0),
            b'C' | b'c' => bases.push(1),
            b'G' | b'g' => bases.push(2),
            b'T' | b't' => bases.push(3),
            _ => {} // header / newline / N
        }
    }
    bases
}

/// Build a sidecar from the given FASTA bytes, return the opened index.
fn build_and_open(fasta: &[u8]) -> (tempfile::TempDir, LearnedIndex) {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("s.fa");
    std::fs::write(&fa, fasta).unwrap();
    let prefix = dir.path().join("s.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    (dir, idx)
}

// ---------------------------------------------------------------------------
// Main equivalence test
// ---------------------------------------------------------------------------

/// Run `smem_range` (SIMD path) and `resolve_one_scalar` over a corpus of
/// queries sampled from the SA and assert identical `(k, l, s)` for every
/// query.
fn run_equivalence(n_bases: usize, seed: u64) {
    let fasta = deterministic_fasta(n_bases, seed);
    let bases = fasta_to_unpacked(&fasta);
    let (_dir, idx) = build_and_open(&fasta);

    let sa_num = idx.sa_num();
    let enc = PacEncoding::Unpacked;

    // Sample up to 256 SA positions as query starting points.
    let n_queries = 256.min(sa_num as usize);
    let step = (sa_num as usize).max(1) / n_queries.max(1);

    for qi in 0..n_queries {
        let sa_idx = (qi * step) as u64 % sa_num;
        let sa_pos = idx.sa_position_for(sa_idx);
        let avail = bases.len().saturating_sub(sa_pos as usize).min(32);
        if avail < 32 {
            // Skip partial windows near the genome end (query must be exactly 32).
            continue;
        }
        let query = &bases[sa_pos as usize..sa_pos as usize + 32];

        // SIMD path (normal public API).
        let (k_simd, l_simd, s_simd) = idx.smem_range(query, &bases).expect("smem_range failed");

        // Scalar path (bypass SIMD).
        let sr_scalar = idx.resolve_one_scalar(query, &bases, enc, sa_num);

        assert_eq!(
            (k_simd, l_simd, s_simd),
            (sr_scalar.k, sr_scalar.l, sr_scalar.s),
            "SIMD/scalar mismatch at sa_idx={sa_idx} sa_pos={sa_pos}: \
             SIMD=(k={k_simd}, l={l_simd}, s={s_simd}) \
             SCALAR=(k={}, l={}, s={})",
            sr_scalar.k,
            sr_scalar.l,
            sr_scalar.s,
        );
    }
}

#[test]
fn simd_scalar_equivalence_small() {
    // ~256 bases — exercises the tail (< CHUNK candidates in the search window).
    run_equivalence(256, 0xDEAD_BEEF_0001u64);
}

#[test]
fn simd_scalar_equivalence_medium() {
    // 5 386-bp (PhiX-scale) — multiple full CHUNK passes.
    run_equivalence(5_386, 0xDEAD_BEEF_0002u64);
}

#[test]
fn simd_scalar_equivalence_large() {
    // 50 000-bp — exercises larger err bounds and more chunks.
    run_equivalence(50_000, 0xDEAD_BEEF_0003u64);
}

// ---------------------------------------------------------------------------
// Packed-pac equivalence
// ---------------------------------------------------------------------------

fn pack_bases(bases: &[u8]) -> (Vec<u8>, u64) {
    let n = bases.len();
    let mut out = vec![0u8; n.div_ceil(4)];
    for (i, &b) in bases.iter().enumerate() {
        let shift = 6 - 2 * ((i % 4) as u32);
        out[i / 4] |= (b & 0x3) << shift;
    }
    (out, n as u64)
}

#[test]
fn simd_scalar_equivalence_packed_pac() {
    // Verify that the packed-pac path also produces SIMD ↔ scalar equivalence.
    let fasta = deterministic_fasta(5_386, 0xDEAD_BEEF_0004u64);
    let bases = fasta_to_unpacked(&fasta);
    let (packed_pac, num_bases) = pack_bases(&bases);
    let (_dir, idx) = build_and_open(&fasta);

    let sa_num = idx.sa_num();
    let enc = PacEncoding::Packed { num_bases };
    let n_queries = 128.min(sa_num as usize);
    let step = (sa_num as usize).max(1) / n_queries.max(1);

    for qi in 0..n_queries {
        let sa_idx = (qi * step) as u64 % sa_num;
        let sa_pos = idx.sa_position_for(sa_idx);
        let avail = bases.len().saturating_sub(sa_pos as usize).min(32);
        if avail < 32 {
            continue;
        }
        let query = &bases[sa_pos as usize..sa_pos as usize + 32];

        // SIMD path (packed).
        let (k_simd, l_simd, s_simd) = idx
            .smem_range_packed(query, &packed_pac, num_bases)
            .expect("smem_range_packed failed");

        // Scalar path (packed).
        let sr_scalar = idx.resolve_one_scalar(query, &packed_pac, enc, sa_num);

        assert_eq!(
            (k_simd, l_simd, s_simd),
            (sr_scalar.k, sr_scalar.l, sr_scalar.s),
            "packed SIMD/scalar mismatch at sa_idx={sa_idx} sa_pos={sa_pos}"
        );
    }
}
