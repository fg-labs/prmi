// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

// Index loops are intentional: the rolling k-mer carries state across positions,
// and the interval/keep bitmaps are filled by position.
#![allow(clippy::needless_range_loop)]

//! Build the exome-plus keep-set BED: exome targets + a genome-count-capped
//! homology halo + flanks (Design Z, measurement-only; not shipped).
//!
//! Halo definition: every position of the reference whose `k`-mer (a) overlaps an
//! exome position (its `k`-mer window intersects an exome base) AND (b) has total
//! reference count <= C (measured against the doubled `[Fwd || RC]` SA, so
//! palindromes count twice). These are the
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

/// Reverse-complement a right-aligned k-mer code (the `k` bases occupy the low
/// `2*k` bits, MSB-first, matching the rolling `code` built in both passes).
/// Canonicalizing each code to `min(code, revcomp_code(code, k))` collapses a
/// k-mer and its reverse complement into one key — required because the suffix
/// array is built over the doubled `[Fwd || RC]` text, so a seed's `occ` counts
/// both its forward copies and the forward copies of its reverse complement.
fn revcomp_code(mut code: u64, k: usize) -> u64 {
    let mut rc = 0u64;
    for _ in 0..k {
        rc = (rc << 2) | (3 - (code & 3));
        code >>= 2;
    }
    rc
}

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
    // Parse `flank` as `usize`: it is only ever added to / subtracted from
    // 0-based positions, so an unsigned type rejects negatives at parse time
    // and lets the interval math below use saturating (overflow-free) bounds.
    let flank: usize = a[5].parse().expect("flank");
    assert!((1..=32).contains(&k), "k must be 1..=32");

    // --- read single-contig FASTA into a 2-bit code vec (N -> 4 sentinel) ---
    let mut seq: Vec<u8> = Vec::new();
    // Contig name for the output BED, taken from the first FASTA header (not
    // hardcoded) so the tool is correct for any single-contig reference.
    let mut contig = String::new();
    let f = BufReader::new(std::fs::File::open(fa).expect("open fa"));
    for line in f.lines() {
        let line = line.unwrap();
        if let Some(rest) = line.strip_prefix('>') {
            if contig.is_empty() {
                contig = rest.split_whitespace().next().unwrap_or("seq").to_string();
            } else {
                // Single-contig tool: a second header would silently mislabel
                // the appended sequence with the first contig's name.
                eprintln!("error: FASTA must be single-contig; found additional header: {rest}");
                std::process::exit(2);
            }
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
        let raw = line.unwrap();
        let line = raw.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("track")
            || line.starts_with("browser")
        {
            continue;
        }
        let c: Vec<&str> = line.split_whitespace().collect();
        // Fail closed on malformed rows (parity with `MaskConfig::parse_bed`):
        // a silently-skipped row would shrink the keep-set without any warning.
        if c.len() < 3 {
            eprintln!("error: BED row needs at least 3 columns: {line:?}");
            std::process::exit(2);
        }
        // Fail closed on a BED contig that does not match the FASTA: the tool
        // emits intervals in the FASTA contig's coordinates, so a mismatched
        // (or extra-contig) BED would silently produce mislabeled intervals.
        if c[0] != contig {
            eprintln!(
                "error: BED contig {:?} does not match FASTA contig {:?}",
                c[0], contig
            );
            std::process::exit(2);
        }
        let s: usize = c[1].parse().expect("BED start");
        let e: usize = c[2].parse().expect("BED end");
        // Reject inverted/empty or out-of-reference intervals rather than
        // truncating them later: a half-open [s, e) must satisfy s < e <= n.
        if e <= s || e > n {
            eprintln!("error: BED interval [{s}, {e}) is invalid for reference length {n}");
            std::process::exit(2);
        }
        ex.push((s, e));
    }
    ex.sort_unstable();
    // membership bitmap over reference positions
    let mut in_exome = vec![false; n];
    for &(s, e) in &ex {
        // `e <= n` is enforced at parse time, so no clamping is needed here.
        for p in s..e {
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
            // Canonicalize to the smaller of the forward code and its reverse
            // complement so a k-mer and its RC share one key. The SA is built
            // over the doubled `[Fwd || RC]` text, so a seed's `occ` (the
            // "total reference count" the cap C is applied to) counts both
            // orientations — counting forward-only would undercount it.
            let start = i + 1 - k; // k-mer starting position
            let rc = revcomp_code(code, k);
            let canon = code.min(rc);
            // A palindromic k-mer (code == its own RC) maps to a single canonical
            // key, yet it occupies BOTH halves of the doubled `[Fwd || RC]` text
            // at each forward position, so it contributes 2 to `occ`; a
            // non-palindrome and its RC are distinct forward k-mers that each add
            // 1 under the shared key. Counting a palindrome once would halve its
            // `occ` and wrongly admit a high-copy palindrome past the cap C.
            let c = count.entry(canon).or_insert(0);
            *c = c.saturating_add(if code == rc { 2 } else { 1 });
            // Flag the k-mer as exonic if its window overlaps the exome at ANY
            // base, not only at `start`: a read covering the exome boundary
            // produces boundary-spanning seeds whose start is off-exome, and
            // those seeds still need their low-copy homologs kept for `occ`
            // byte-identity.
            if in_exome[start..start + k].iter().any(|&hit| hit) {
                in_ex_kmer.insert(canon, true);
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
            // Match the canonical key used when building `halo` so RC-homolog
            // positions (off-target copies that are reverse-complemented) are
            // kept too — they contribute to the exonic seed's `occ`.
            if halo.contains(&code.min(revcomp_code(code, k))) {
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
    let mut intervals: Vec<(usize, usize)> = Vec::new();
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
        // Saturating bounds keep the ±flank math in range for any `flank`
        // value (it is an unbounded CLI argument); `n` clamps the right edge.
        let fs = s.saturating_sub(flank);
        let fe = i.saturating_add(flank).min(n);
        intervals.push((fs, fe));
    }
    // merge after flanking
    intervals.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
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
    let mut kept_bp = 0usize;
    for (s, e) in &merged {
        writeln!(out, "{contig}\t{s}\t{e}").unwrap();
        kept_bp += e - s;
    }
    eprintln!(
        "kept(pre-flank) bp = {halo_bp}; flanked+merged intervals = {}, bp = {kept_bp} ({:.2}% of ref)",
        merged.len(),
        100.0 * kept_bp as f64 / n as f64
    );
}
