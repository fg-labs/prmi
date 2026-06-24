// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Build a `.kmt` shallow-band accelerator for an EXISTING on-disk index,
//! without rebuilding the suffix array (measurement convenience). Computes the
//! length-1..=k SA-bound tables from the loaded `.sa` via `build_kmer_table`
//! and writes the sidecar bound to the reference by `sa_num` + `ref_digest`.
//!
//! Usage:
//!   build_kmt <index_prefix> <pac_path> <k> <ref_digest_hex>
//!
//! `ref_digest_hex` must be the index's `[sa] pac_sha256` (64 hex chars) so the
//! open path accepts the table; otherwise prmi silently ignores it. Before
//! writing, this tool refuses to clobber an existing sidecar unless BOTH hold:
//! `ref_digest_hex` matches the opened index's own binding digest (so the table
//! belongs to `index_prefix`), and the SHA-256 of `pac_path` matches it too (so
//! the table contents match the digest). Together a mismatched index or pac
//! can't yield — or overwrite a valid sidecar with — a loadable-but-wrong table.

use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::sidecar::kmt_file::KmtFileWriter;
use std::path::Path;

/// Decode a 64-char hex string into 32 bytes (panics on malformed input).
fn hex_decode_32(s: &str) -> [u8; 32] {
    let b = s.as_bytes();
    assert_eq!(
        b.len(),
        64,
        "ref_digest must be 64 hex chars, got {}",
        b.len()
    );
    let nib = |c: u8| -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex char in ref_digest"),
        }
    };
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (nib(b[2 * i]) << 4) | nib(b[2 * i + 1]);
    }
    out
}

/// Build a `.kmt` shallow-band table from the existing index at the given prefix
/// and write it next to the index, bound to `sa_num` + the supplied ref digest.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: build_kmt <index_prefix> <pac_path> <k> <ref_digest_hex>");
        std::process::exit(2);
    }
    let prefix = &args[1];
    let pac_path = &args[2];
    let k: u32 = args[3].parse().expect("k");
    let digest = hex_decode_32(&args[4]);

    let idx = LearnedIndex::open(Path::new(prefix)).expect("open index");

    // Identity guard #1 (index ↔ digest): reject a `ref_digest` that does not
    // match THIS index's own binding digest before doing any work. Without it a
    // self-consistent pac/digest pair belonging to a *different* index would be
    // built and written — clobbering any valid `.kmt` already next to `prefix`
    // before the post-write `has_kmt()` assert below could catch the mismatch.
    let index_digest = idx.ref_digest_hex();
    if !index_digest.eq_ignore_ascii_case(&args[4]) {
        eprintln!(
            "ref_digest {} does not match this index's digest {index_digest} — wrong index or digest",
            args[4]
        );
        std::process::exit(2);
    }

    let pac = std::fs::read(pac_path).expect("read pac");
    let enc = PacEncoding::Packed {
        num_bases: idx.l_pac(),
    };

    // Identity guard #2 (pac ↔ digest): the table is built from `pac` but
    // persisted under `ref_digest`. If the two disagree, the open path would
    // still accept a logically-wrong table (correct digest, wrong contents).
    // `ref_digest` is the index's `pac_sha256` (raw `.pac` bytes), so reject
    // any pac whose hash does not match the digest before writing. Combined with
    // guard #1 this transitively proves `pac` belongs to `prefix`.
    let pac_digest = prmi::pac::pac_sha256(Path::new(pac_path)).expect("hash pac");
    if !pac_digest.eq_ignore_ascii_case(&args[4]) {
        eprintln!(
            "pac sha256 {pac_digest} does not match ref_digest {} — wrong pac or digest",
            args[4]
        );
        std::process::exit(2);
    }

    eprintln!("building kmt k={k} over sa_num={} ...", idx.sa_num());
    let table = idx.build_kmer_table(k, &pac, enc);

    let (tk, tlo, thi) = table.parts();
    let kmt_path = format!("{prefix}.kmt");
    KmtFileWriter::write(Path::new(&kmt_path), tk, idx.sa_num(), &digest, tlo, thi)
        .expect("write kmt");
    eprintln!("wrote {kmt_path}");

    // Verify the freshly-written table actually binds and loads.
    let reopened = LearnedIndex::open(Path::new(prefix)).expect("reopen index");
    eprintln!("VERIFY has_kmt={}", reopened.has_kmt());
    assert!(
        reopened.has_kmt(),
        "kmt written but not loaded — binding mismatch"
    );
}
