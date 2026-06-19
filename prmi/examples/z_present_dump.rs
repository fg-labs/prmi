// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Dump the Design-Z present/absent partition for a FASTQ.
//!
//! Opens the tiered fast-path index (`PRMI_FAST`) and, for each read, runs the
//! same `present_anchor` pre-reject the production dispatch would use. Emits a
//! TSV (`qname<TAB>present`) where `present=1` means Design Z would serve this
//! read from the fast index and `present=0` means it falls back to whole genome.
//!
//! This is the join key for the Stage-1 alignment-equivalence diff: the
//! fast-vs-full alignment concordance is only meaningful over present reads
//! (absent reads are served by the fallback = full, so trivially concordant).
//!
//! ```text
//! PRMI_FAST=chr22.zh.prmi PRMI_PAC=chr22.fa.pac \
//!   cargo run --release --example z_present_dump -- on.fq > present.tsv
//! ```

fn main() {
    use prmi::index::smem::PacEncoding;
    use prmi::index::LearnedIndex;
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    let require = |k: &str| -> String {
        std::env::var(k).unwrap_or_else(|_| {
            eprintln!("z_present_dump: required env var {k} is not set");
            std::process::exit(2);
        })
    };
    let fast = LearnedIndex::open(Path::new(&require("PRMI_FAST"))).expect("open fast");
    let pac = std::fs::read(require("PRMI_PAC")).expect("read pac");
    // Reject a truncated or wrong-reference PRMI_PAC before it can make
    // `present_anchor` classify against the wrong bytes (or panic) while still
    // emitting a TSV (bntpac is 2 bits/base → `l_pac.div_ceil(4)` bytes).
    let l_pac = fast.l_pac();
    let expected_pac_bytes =
        usize::try_from(l_pac.div_ceil(4)).expect("l_pac does not fit in usize");
    assert_eq!(
        pac.len(),
        expected_pac_bytes,
        "PRMI_PAC length does not match PRMI_FAST l_pac"
    );
    let enc = PacEncoding::Packed { num_bases: l_pac };

    let files: Vec<String> = std::env::args().skip(1).collect();
    // No input paths → an empty partition with no join keys; surface the bad
    // invocation rather than silently emitting a header-only TSV. Validate BEFORE
    // the header `println!` so a failed run leaves no partial TSV on stdout.
    assert!(
        !files.is_empty(),
        "usage: z_present_dump <reads.fq> [reads2.fq ...]"
    );
    println!("qname\tpresent");
    // The emitted `qname` is the join key `scripts/z_aln_concordance.py` keys on,
    // and that consumer rejects empty/duplicate qnames — so enforce the same
    // contract here at the producer rather than emitting a TSV it will reject.
    let mut seen_qnames = HashSet::new();
    let (mut total, mut present) = (0u64, 0u64);
    for path in &files {
        let f = std::fs::File::open(path).unwrap_or_else(|_| panic!("open {path}"));
        let mut lines = BufReader::new(f).lines();
        while let Some(h) = lines.next() {
            // A file truncated after the sequence line must fail fast: classifying a
            // record with a missing `+`/quality line would still bump `total` and emit
            // a TSV row for an invalid read. Require all four FASTQ lines per record.
            let h = h.unwrap_or_else(|e| panic!("read {path}: {e}"));
            assert!(
                h.starts_with('@'),
                "invalid FASTQ in {path}: header must start with '@', got {h:?}"
            );
            let Some(seq) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing sequence after header");
            };
            let seq = seq.unwrap_or_else(|e| panic!("read {path}: {e}"));
            let Some(plus) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing '+' line after sequence");
            };
            let plus = plus.unwrap_or_else(|e| panic!("read {path}: {e}"));
            let Some(qual) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing quality line after '+'");
            };
            let qual = qual.unwrap_or_else(|e| panic!("read {path}: {e}"));
            assert!(
                plus.starts_with('+'),
                "invalid FASTQ in {path}: '+' line expected, got {plus:?}"
            );
            assert_eq!(
                qual.len(),
                seq.len(),
                "invalid FASTQ in {path}: seq/qual length mismatch"
            );
            // QNAME = header without leading '@', up to first whitespace.
            let qname = h
                .trim_start_matches('@')
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();
            assert!(!qname.is_empty(), "invalid FASTQ in {path}: empty QNAME");
            assert!(
                seen_qnames.insert(qname.clone()),
                "duplicate QNAME across inputs: {qname:?}"
            );
            let read: Vec<u8> = seq
                .bytes()
                .map(|b| prmi::encoding::base_to_2bit(b).unwrap_or(4))
                .collect();
            total += 1;
            let is_present = fast.present_anchor(&read, &pac, enc);
            if is_present {
                present += 1;
            }
            println!("{qname}\t{}", is_present as u8);
        }
    }
    eprintln!(
        "z_present_dump: {present}/{total} present ({:.1}%)",
        100.0 * present as f64 / total.max(1) as f64
    );
}
