// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

// Index loops are intentional: the rolling k-mer carries state across positions,
// and the interval/keep bitmaps are filled by position.
#![allow(clippy::needless_range_loop)]

//! Build the exome-plus keep-set BED: exome targets + a genome-count-capped
//! homology halo + flanks (Design Z, measurement-only; not shipped).
//!
//! Halo definition: every position of the reference whose `k`-mer (a) also occurs
//! at an exome position AND (b) has total reference count <= C. These are the
//! low-copy homologous copies of exonic k-mers — including them makes `occ` match
//! the whole-genome value for exonic seeds (the byte-identity condition), while
//! the count cap C drops high-copy repeats (which fall back regardless).
//!
//! Usage:
//!   build_halo_bed <ref.fa> <exome.bed> <k> <cap_C> <flank> > keepset.bed
//!
//! `ref.fa` must be a single-contig FASTA (e.g. chr22.fa). `exome.bed` is in that
//! contig's 0-based coordinates. Output BED = (exome ∪ halo) each ±flank, merged.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: build_halo_bed <ref.fa> <exome.bed> <k> <cap_C> <flank>");
        std::process::exit(2);
    }
    let fa = &a[1];
    let bed = &a[2];
    let k: usize = a[3].parse().expect("k");
    let cap: u32 = a[4].parse().expect("cap_C");
    let flank: i64 = a[5].parse().expect("flank");
    assert!((1..=32).contains(&k), "k must be 1..=32");

    // --- read single-contig FASTA into a 2-bit code vec (N -> 4 sentinel) ---
    let mut seq: Vec<u8> = Vec::new();
    let f = BufReader::new(std::fs::File::open(fa).expect("open fa"));
    for line in f.lines() {
        let line = line.unwrap();
        if line.starts_with('>') {
            continue;
        }
        // `BufRead::lines()` strips `\n` but not a trailing `\r`, so a CRLF FASTA
        // would otherwise push that `\r` through `_ => 4` as a phantom sentinel
        // base, inflating `n` and shifting every downstream coordinate. Trim it
        // (matching the BED loop's `raw.trim()`) before consuming the bytes.
        for b in line.trim_end().bytes() {
            seq.push(match b {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => 4,
            });
        }
    }
    let n = seq.len();
    eprintln!("ref len = {n}");

    // --- exome intervals (sorted, merged), 0-based half-open ---
    let mut ex: Vec<(usize, usize)> = Vec::new();
    let bf = BufReader::new(std::fs::File::open(bed).expect("open bed"));
    for line in bf.lines() {
        let line = line.unwrap();
        if line.is_empty() || line.starts_with('#') || line.starts_with("track") {
            continue;
        }
        let c: Vec<&str> = line.split_whitespace().collect();
        if c.len() < 3 {
            continue;
        }
        ex.push((c[1].parse().unwrap(), c[2].parse().unwrap()));
    }
    ex.sort_unstable();
    // membership bitmap over reference positions
    let mut in_exome = vec![false; n];
    for &(s, e) in &ex {
        for p in s..e.min(n) {
            in_exome[p] = true;
        }
    }

    // --- pass 1: count k-mers; flag those that occur at an exome position ---
    // k<=32 so a forward 2-bit code fits in u64. Windows containing N reset.
    let mask: u64 = if k == 32 {
        u64::MAX
    } else {
        (1u64 << (2 * k)) - 1
    };
    let mut count: HashMap<u64, u32> = HashMap::with_capacity(n);
    let mut in_ex_kmer: HashMap<u64, bool> = HashMap::with_capacity(n / 4);
    let mut code: u64 = 0;
    let mut valid = 0usize; // consecutive non-N bases
    for i in 0..n {
        let b = seq[i];
        if b >= 4 {
            valid = 0;
            code = 0;
            continue;
        }
        code = ((code << 2) | b as u64) & mask;
        valid += 1;
        if valid >= k {
            let start = i + 1 - k; // k-mer starting position
            let c = count.entry(code).or_insert(0);
            *c = c.saturating_add(1);
            if in_exome[start] {
                in_ex_kmer.insert(code, true);
            }
        }
    }
    eprintln!("distinct {k}-mers = {}", count.len());

    // --- halo k-mer set: in-exome AND total count <= cap ---
    let halo: std::collections::HashSet<u64> = in_ex_kmer
        .keys()
        .copied()
        .filter(|kmer| *count.get(kmer).unwrap_or(&0) <= cap)
        .collect();
    eprintln!("halo {k}-mers (in-exome, count<={cap}) = {}", halo.len());

    // --- pass 2: mark all reference positions of halo k-mers (the homologous
    // copies, exonic + off-exome). Cover the full k-mer span. ---
    let mut keep = vec![false; n];
    code = 0;
    valid = 0;
    for i in 0..n {
        let b = seq[i];
        if b >= 4 {
            valid = 0;
            code = 0;
            continue;
        }
        code = ((code << 2) | b as u64) & mask;
        valid += 1;
        if valid >= k {
            let start = i + 1 - k;
            if halo.contains(&code) {
                for p in start..start + k {
                    keep[p] = true;
                }
            }
        }
    }
    // union exome positions (so the targets themselves are always kept)
    for p in 0..n {
        if in_exome[p] {
            keep[p] = true;
        }
    }

    // --- emit merged intervals with ±flank ---
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut halo_bp = 0usize;
    let mut i = 0usize;
    let mut intervals: Vec<(i64, i64)> = Vec::new();
    while i < n {
        if !keep[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < n && keep[i] {
            i += 1;
        }
        halo_bp += i - s;
        let fs = (s as i64 - flank).max(0);
        let fe = (i as i64 + flank).min(n as i64);
        intervals.push((fs, fe));
    }
    // merge after flanking
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (s, e) in intervals {
        match merged.last_mut() {
            Some(last) if s <= last.1 => {
                if e > last.1 {
                    last.1 = e;
                }
            }
            _ => merged.push((s, e)),
        }
    }
    let mut kept_bp = 0i64;
    for (s, e) in &merged {
        writeln!(out, "chr22\t{s}\t{e}").unwrap();
        kept_bp += e - s;
    }
    eprintln!(
        "kept(pre-flank) bp = {halo_bp}; flanked+merged intervals = {}, bp = {kept_bp} ({:.2}% of ref)",
        merged.len(),
        100.0 * kept_bp as f64 / n as f64
    );
}
