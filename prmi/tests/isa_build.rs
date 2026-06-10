// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! End-to-end ISA build/open: `prmi build --with-isa` emits a `.isa`, the index
//! loads it (`has_isa`), `isa_at` inverts the suffix array, and the inverse-SA
//! launch hint (`isa_at(refpos)` fed as `est_hint`) reproduces the model-launch
//! `mem_search` byte-for-byte.

use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi::train::mask::MaskConfig;
use std::io::Write;
use std::path::Path;

/// Deterministic ACGT bases (no N) via an LCG, written as a FASTA.
fn write_fasta(path: &Path, n: usize) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let bases: Vec<u8> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 61) & 3) as u8
        })
        .collect();
    let alphabet = [b'A', b'C', b'G', b'T'];
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(w, ">isa_test").unwrap();
    for chunk in bases.chunks(60) {
        let line: Vec<u8> = chunk.iter().map(|&b| alphabet[b as usize]).collect();
        w.write_all(&line).unwrap();
        w.write_all(b"\n").unwrap();
    }
    bases
}

fn build(dir: &Path, bases_len: usize, with_isa: bool) -> (LearnedIndex, Vec<u8>) {
    let fa = dir.join("ref.fa");
    let bases = write_fasta(&fa, bases_len);
    let prefix = dir.join("ref.prmi");
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_isa(with_isa);
    build_sidecar_with_config(&fa, &prefix, None, MaskConfig::default(), 1, Some(cfg)).unwrap();
    (LearnedIndex::open(&prefix).unwrap(), bases)
}

#[test]
fn isa_inverts_sa_and_hint_is_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let (idx, bases) = build(dir.path(), 3000, true);
    let e = PacEncoding::Unpacked;

    assert!(
        idx.has_isa(),
        "build --with-isa must produce a loadable .isa"
    );
    let sa_num = idx.sa_num();

    // isa_at inverts the SA: for every SA index i at reference position
    // p = sa_position_for(i), isa_at(p) == i.
    for i in (0..sa_num).step_by(7) {
        let refpos = idx.sa_position_for(i);
        assert_eq!(idx.isa_at(refpos), Some(i), "isa_at(sa_position_for({i}))");
    }
    // Out-of-range refpos → None.
    assert_eq!(idx.isa_at(sa_num), None);

    // ISA launch hint: for a matching query, isa_at(refpos of the maximal match)
    // round-trips to sa_start, and feeding it as est_hint is byte-identical.
    for start in (0..bases.len() - 60).step_by(53) {
        let q = &bases[start..start + 60];
        let m = idx.mem_search(q, &bases, e);
        if m.match_len == 0 {
            continue;
        }
        let refpos = idx.sa_position_for(m.sa_start);
        let hint = idx.isa_at(refpos).expect("refpos in range");
        assert_eq!(hint, m.sa_start, "isa_at(refpos) must recover the SA index");
        let hinted = idx.mem_search_from_hint(q, hint, true, &bases, e);
        assert_eq!(hinted, m, "est_hint=isa_at(refpos) must equal est_hint=0");
    }
}

#[test]
fn backward_hint_is_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let (idx, bases) = build(dir.path(), 3000, true);
    let e = PacEncoding::Unpacked;

    // Reference-lifted reads: read == a reference window, so the left walk at the
    // natural locus (s + pivot) follows the read↔reference alignment and the
    // confirm-only must reproduce the from-scratch backward extension exactly.
    for s in (0..bases.len() - 200).step_by(151) {
        let w = 120usize;
        let read = &bases[s..s + w];
        for &pivot in &[40usize, 60, 80] {
            // Right anchor for read[pivot..] via the forward one-shot.
            let fwd = idx.mem_search(&read[pivot..], &bases, e);
            if fwd.match_len == 0 {
                continue;
            }
            let anchor_len = fwd.match_len;
            // From-scratch (model-launch) backward extension — the oracle.
            let global =
                idx.mem_search_backward(fwd.sa_start, fwd.occ, anchor_len, read, pivot, &bases, e);

            // The hint is the SA index of the anchor at its natural genomic
            // position (s + pivot), obtained via the inverse SA.
            let hint = idx.isa_at((s + pivot) as u64).expect("refpos in range");
            let hinted =
                idx.mem_search_backward_from_hint(read, pivot, anchor_len, hint, true, &bases, e);
            assert_eq!(
                hinted, global,
                "backward hint != from-scratch at s={s} pivot={pivot}"
            );
            // match_len-only path reports the same total span.
            let ml =
                idx.mem_search_backward_from_hint(read, pivot, anchor_len, hint, false, &bases, e);
            assert_eq!(
                ml.match_len, global.match_len,
                "backward match_len-only mismatch"
            );
        }
    }
}

#[test]
fn default_build_has_no_isa() {
    let dir = tempfile::tempdir().unwrap();
    let (idx, _bases) = build(dir.path(), 1500, false);
    assert!(!idx.has_isa(), "default build must not emit a .isa");
    assert_eq!(idx.isa_at(0), None, "isa_at returns None without a .isa");
    // The .isa file must not exist.
    let isa_path = prmi::sidecar::SidecarPaths::from_prefix(&dir.path().join("ref.prmi")).isa;
    assert!(
        !isa_path.exists(),
        "no .isa file should be written by default"
    );
}
