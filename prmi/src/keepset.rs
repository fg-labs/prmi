// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Keep-set construction (Design Z, item 2): turn a per-contig target BED plus a
//! reference into a flat-coordinate keep-set BED consumable by
//! `prmi build --keep-bed`.
//!
//! The keep-set is the union of:
//! - **Targets** — every position of each target interval (kept whole).
//! - **k-mer homology** — the START position of every reference k-mer that is a
//!   forward or reverse-complement copy of a k-mer occurring at a target
//!   position, provided that k-mer's reference count is `<= max_occ` (high-copy
//!   repeats are dropped — they fall back to the off-target engine regardless).
//!   Only the suffix-START position of each homologous copy is kept, because the
//!   tiered `.sa` retains an entry iff its suffix-start coordinate is in the
//!   keep-set; keeping the whole `[p, p+k)` span would over-keep on merge.
//!
//! Coordinate space: the output BED is in prmi's FLAT concatenated-genome
//! coordinate space (contigs concatenated in FASTA order, one position per base,
//! N included) — the same space `prmi build` operates in and `parse_bed`
//! consumes. The chrom column is a cosmetic label; positions are flat offsets.
//!
//! Output format: [`KeepsetFormat::Bed3`] emits one line per maximal kept run
//! (precise, but potentially many lines); [`KeepsetFormat::Bed12`] packs runs
//! into block-bearing lines (compact). Both decode to the identical kept
//! position set via `parse_bed`.
//!
//! Limitation: homology is detected with a single fixed `k` over exact matches
//! (forward + reverse-complement). It does not capture gapped/SNP-divergent
//! paralogy; that is by design — the off-target fallback serves those reads.

use crate::encoding::base_to_2bit;
use crate::error::{Error, Result};
use noodles_fasta::io::Reader;
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Write};
use std::path::Path;

/// A non-N base reads as 0..=3; this sentinel marks N / non-IUPAC, which resets
/// the rolling k-mer (no k-mer window may span an N).
const N_SENTINEL: u8 = 4;

/// Output BED flavor for the keep-set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepsetFormat {
    /// One BED3 line per maximal kept run.
    Bed3,
    /// BED12 lines packing up to `blocks_per_line` runs as blocks each.
    Bed12,
}

/// Options controlling keep-set construction.
#[derive(Debug, Clone)]
pub struct KeepsetOptions {
    /// k-mer length used to detect homology (1..=32). Default provisional
    /// pending the item-3 benchmark.
    pub k: usize,
    /// Drop target k-mers whose total reference count (forward + reverse
    /// complement, counted on the strand-canonical key) exceeds this cap —
    /// high-copy repeats, including inverted ones, fall back regardless. `0`
    /// means uncapped.
    pub max_occ: u64,
    /// Extend every kept run by this many bp on each side before merging. `0`
    /// keeps the set position-precise.
    pub flank: u64,
    /// Output BED flavor.
    pub format: KeepsetFormat,
    /// BED12 only: maximum blocks packed into a single line.
    pub blocks_per_line: usize,
    /// Cosmetic chrom label written to the (flat-coordinate) output BED.
    pub chrom_label: String,
}

impl Default for KeepsetOptions {
    fn default() -> Self {
        Self {
            k: 32,
            max_occ: 0,
            flank: 0,
            format: KeepsetFormat::Bed3,
            blocks_per_line: 1000,
            chrom_label: "genome".to_string(),
        }
    }
}

/// Summary statistics returned by [`build_keepset`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeepsetStats {
    /// Flat genome length (sum of contig lengths).
    pub genome_len: u64,
    /// Number of target intervals mapped to flat coordinates.
    pub target_intervals: usize,
    /// Number of maximal kept runs emitted (before BED12 packing).
    pub kept_runs: usize,
    /// Total kept base pairs across all runs.
    pub kept_bp: u64,
    /// Number of BED lines written.
    pub lines_written: usize,
}

/// One contig's position in the flat concatenated genome.
#[derive(Debug, Clone)]
struct Contig {
    name: String,
    offset: u64,
    len: u64,
}

/// Read a FASTA into the flat 2-bit genome (ACGT→0..3, N→[`N_SENTINEL`]) and the
/// contig table. Contigs are concatenated in FASTA record order, one position
/// per base, matching `prmi build`'s flat coordinate space (which encodes N→A
/// but still occupies one position, so flat offsets are identical).
fn read_fasta_contigs(path: &Path) -> Result<(Vec<u8>, Vec<Contig>)> {
    let file = std::fs::File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = Reader::new(BufReader::new(file));
    let mut bases: Vec<u8> = Vec::new();
    let mut contigs: Vec<Contig> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    for record in reader.records() {
        let record = record.map_err(|e| Error::Fasta {
            file: path.to_path_buf(),
            detail: e.to_string(),
        })?;
        let name = String::from_utf8_lossy(record.name()).into_owned();
        // `parse_target_bed_to_flat` collapses contigs into a name->offset map;
        // a duplicate name would silently overwrite the earlier offset and map
        // its targets to the wrong contig, so reject duplicates up front.
        if !seen_names.insert(name.clone()) {
            return Err(Error::InvalidInput {
                detail: format!("reference FASTA contains duplicate contig name {name:?}"),
            });
        }
        let offset = bases.len() as u64;
        let mut len = 0u64;
        for &b in record.sequence().as_ref() {
            bases.push(base_to_2bit(b).unwrap_or(N_SENTINEL));
            len += 1;
        }
        contigs.push(Contig { name, offset, len });
    }
    Ok((bases, contigs))
}

/// Parse a per-contig BED3 target file and map each interval into flat
/// coordinates using the contig table. The chrom column must name a contig in
/// the reference; `end` must not exceed the contig length.
fn parse_target_bed_to_flat(
    path: &Path,
    contigs: &[Contig],
) -> Result<Vec<(u64, u64)>> {
    let by_name: HashMap<&str, &Contig> =
        contigs.iter().map(|c| (c.name.as_str(), c)).collect();
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("track")
            || line.starts_with("browser")
        {
            continue;
        }
        // Require exact BED3. A wider record (e.g. BED12 with exon blocks) would
        // have its block structure silently dropped and be widened to the outer
        // [start, end) span — unlike the keep-set OUTPUT, which `parse_bed`
        // decodes block-precise. Reject it so the two paths cannot disagree.
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() != 3 {
            return Err(Error::InvalidInput {
                detail: format!(
                    "target BED line {}: expected BED3 with exactly 3 columns, got {}",
                    lineno + 1,
                    cols.len()
                ),
            });
        }
        let chrom = cols[0];
        let parse = |s: &str, what: &str| -> Result<u64> {
            s.parse::<u64>().map_err(|_| Error::InvalidInput {
                detail: format!("target BED line {}: bad {} {:?}", lineno + 1, what, s),
            })
        };
        let start = parse(cols[1], "start")?;
        let end = parse(cols[2], "end")?;
        if end <= start {
            return Err(Error::InvalidInput {
                detail: format!(
                    "target BED line {}: end <= start ({} <= {})",
                    lineno + 1,
                    end,
                    start
                ),
            });
        }
        let contig = by_name.get(chrom).ok_or_else(|| Error::InvalidInput {
            detail: format!(
                "target BED line {}: chrom {:?} not found in reference",
                lineno + 1,
                chrom
            ),
        })?;
        if end > contig.len {
            return Err(Error::InvalidInput {
                detail: format!(
                    "target BED line {}: end {} exceeds contig {:?} length {}",
                    lineno + 1,
                    end,
                    chrom,
                    contig.len
                ),
            });
        }
        out.push((contig.offset + start, contig.offset + end));
    }
    Ok(out)
}

/// Invoke `f(start, code)` for every full k-mer window in `bases`, where
/// `start` is the window's first position and `code` is its 2-bit packed code.
/// The rolling code resets at any [`N_SENTINEL`] base, so no window spans an N.
/// `k` must be in 1..=32 (caller-validated).
fn for_each_kmer(bases: &[u8], k: usize, mut f: impl FnMut(usize, u64)) {
    let mask: u64 = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    let mut code: u64 = 0;
    let mut valid: usize = 0;
    for (i, &b) in bases.iter().enumerate() {
        if b >= N_SENTINEL {
            valid = 0;
            code = 0;
            continue;
        }
        code = ((code << 2) | b as u64) & mask;
        valid += 1;
        if valid >= k {
            f(i + 1 - k, code);
        }
    }
}

/// Reverse-complement a `for_each_kmer` code: a k-mer packed LOW-aligned into the
/// low `2*k` bits (base 0 in the high `2*(k-1)..2*k` slot). Complements each base
/// (2-bit XOR `0b11`) and reverses base order, returning the RC in the same
/// low-aligned encoding.
///
/// NOTE: this is *not* [`crate::encoding::reverse_complement_key`], which operates
/// on HIGH-aligned, T-padded 32-mer keys; feeding a low-aligned `for_each_kmer`
/// code to that function mis-aligns the bases for `k < 32` and silently drops RC
/// homologs. Keep the two encodings from crossing.
fn reverse_complement_low(mut code: u64, k: usize) -> u64 {
    let mut rc = 0u64;
    for _ in 0..k {
        rc = (rc << 2) | ((!code) & 0b11);
        code >>= 2;
    }
    rc
}

/// The strand-canonical key for a low-aligned k-mer code: the lexicographic min
/// of the code and its reverse complement. A k-mer and its RC share one key, so
/// forward/RC homologs compare equal and inverted repeats count toward `max_occ`.
fn canonical_low_key(code: u64, k: usize) -> u64 {
    code.min(reverse_complement_low(code, k))
}

/// Compute the merged, flat-coordinate kept runs for the keep-set.
///
/// `bases` is the flat 2-bit genome (values 0..=3, [`N_SENTINEL`] for N).
/// `targets` are flat half-open target intervals (need not be sorted/merged).
/// Returns sorted, merged `[start, end)` runs.
///
/// Returns [`Error::InvalidInput`] if `k` is outside `1..=32`, or if any target
/// interval is empty/inverted (`end <= start`) or extends past `bases.len()`.
///
/// Pure (no I/O) so the homology logic can be unit-tested on synthetic input.
pub fn compute_keep_runs(
    bases: &[u8],
    targets: &[(u64, u64)],
    k: usize,
    max_occ: u64,
    flank: u64,
) -> Result<Vec<(u64, u64)>> {
    if !(1..=32).contains(&k) {
        return Err(Error::InvalidInput {
            detail: format!("keep-set k must be in 1..=32, got {k}"),
        });
    }
    let n = bases.len();

    // Target-position bitmap (whole intervals are kept). Validate caller input
    // rather than silently clipping: an out-of-range or inverted interval is
    // malformed (and `s.min(n) > e.min(n)` would panic the slice below), so
    // reject it up front.
    let mut in_target = vec![false; n];
    for &(s, e) in targets {
        if e <= s {
            return Err(Error::InvalidInput {
                detail: format!("keep-set target end <= start ({e} <= {s})"),
            });
        }
        if e > n as u64 {
            return Err(Error::InvalidInput {
                detail: format!("keep-set target end {e} exceeds genome length {n}"),
            });
        }
        for slot in &mut in_target[s as usize..e as usize] {
            *slot = true;
        }
    }

    // Pass 1: collect the strand-canonical keys of k-mers that START at a target
    // position. Canonicalizing here (min of forward and reverse-complement code)
    // means a target k-mer and its RC share one key, so inverted homologs match
    // and count together later. We deliberately do NOT count all genome k-mers —
    // on a whole genome that map would hold billions of entries (~140 GB on
    // hg38). Only target k-mers can contribute to the keep-set, so only they are
    // tracked.
    let mut target_keys: HashSet<u64> = HashSet::new();
    for_each_kmer(bases, k, |start, code| {
        if in_target[start] {
            target_keys.insert(canonical_low_key(code, k));
        }
    });

    // Homology k-mer set: target k-mers under the count cap. When uncapped
    // (max_occ == 0) every target k-mer qualifies and no counting is needed;
    // otherwise count genome-wide occurrences of TARGET keys ONLY (the map is
    // bounded by the number of distinct target k-mers, not the whole genome).
    // Counting on the canonical key sums forward and reverse-complement
    // occurrences, so inverted high-copy repeats cannot slip under the cap.
    let halo: HashSet<u64> = if max_occ == 0 {
        target_keys
    } else {
        let mut count: HashMap<u64, u64> = HashMap::with_capacity(target_keys.len());
        for_each_kmer(bases, k, |_start, code| {
            let key = canonical_low_key(code, k);
            if target_keys.contains(&key) {
                let c = count.entry(key).or_insert(0);
                *c = c.saturating_add(1);
            }
        });
        target_keys
            .into_iter()
            .filter(|key| *count.get(key).unwrap_or(&0) <= max_occ)
            .collect()
    };

    // Keep bitmap: targets (whole) ∪ homology START positions (length-1). A
    // position is a homologous anchor if its k-mer's canonical key matches a
    // target k-mer's — which holds for both forward and reverse-complement
    // copies (catches inverted paralogs); only the suffix-start position is kept.
    let mut keep = in_target; // reuse the target bitmap as the base
    for_each_kmer(bases, k, |start, code| {
        if halo.contains(&canonical_low_key(code, k)) {
            keep[start] = true;
        }
    });

    Ok(runs_from_bitmap(&keep, flank))
}

/// Extract maximal `true` runs from `keep`, extend each by `flank` on both sides
/// (clamped to `[0, len]`), then sort and merge overlapping runs.
fn runs_from_bitmap(keep: &[bool], flank: u64) -> Vec<(u64, u64)> {
    let n = keep.len();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !keep[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < n && keep[i] {
            i += 1;
        }
        let (mut a, mut b) = (s as u64, i as u64);
        if flank > 0 {
            a = a.saturating_sub(flank);
            b = (b + flank).min(n as u64);
        }
        runs.push((a, b));
    }
    if flank == 0 || runs.len() < 2 {
        return runs;
    }
    runs.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
    for (s, e) in runs {
        match merged.last_mut() {
            Some(last) if s <= last.1 => {
                if e > last.1 {
                    last.1 = e;
                }
            }
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// Write the kept runs as one BED3 line per run.
fn emit_bed3<W: Write>(out: &mut W, chrom: &str, runs: &[(u64, u64)]) -> Result<usize> {
    for &(s, e) in runs {
        writeln!(out, "{chrom}\t{s}\t{e}").map_err(io_err)?;
    }
    Ok(runs.len())
}

/// Write the kept runs as BED12 lines, packing up to `blocks_per_line` runs as
/// blocks each. Each line's span is `[first_run_start, last_run_end)`; block `i`
/// is `(blockSizes[i], blockStarts[i] = run.start - span_start)`.
fn emit_bed12<W: Write>(
    out: &mut W,
    chrom: &str,
    runs: &[(u64, u64)],
    blocks_per_line: usize,
) -> Result<usize> {
    let mut lines = 0usize;
    for chunk in runs.chunks(blocks_per_line.max(1)) {
        let span_start = chunk[0].0;
        let span_end = chunk[chunk.len() - 1].1;
        let sizes: Vec<String> = chunk.iter().map(|&(s, e)| (e - s).to_string()).collect();
        let starts: Vec<String> =
            chunk.iter().map(|&(s, _)| (s - span_start).to_string()).collect();
        writeln!(
            out,
            "{chrom}\t{span_start}\t{span_end}\tkeep\t0\t+\t{span_start}\t{span_end}\t0\t{}\t{}\t{}",
            chunk.len(),
            sizes.join(","),
            starts.join(","),
        )
        .map_err(io_err)?;
        lines += 1;
    }
    Ok(lines)
}

fn io_err(source: std::io::Error) -> Error {
    Error::Io {
        path: std::path::PathBuf::from("<keepset output>"),
        source,
    }
}

/// Build a keep-set BED from a reference FASTA and a per-contig target BED,
/// writing it to `out`. Returns summary statistics.
pub fn build_keepset<W: Write>(
    ref_fa: &Path,
    target_bed: &Path,
    opts: &KeepsetOptions,
    out: &mut W,
) -> Result<KeepsetStats> {
    let (bases, contigs) = read_fasta_contigs(ref_fa)?;
    let genome_len = bases.len() as u64;
    let targets = parse_target_bed_to_flat(target_bed, &contigs)?;
    let runs = compute_keep_runs(&bases, &targets, opts.k, opts.max_occ, opts.flank)?;
    let kept_bp: u64 = runs.iter().map(|&(s, e)| e - s).sum();
    let lines_written = match opts.format {
        KeepsetFormat::Bed3 => emit_bed3(out, &opts.chrom_label, &runs)?,
        KeepsetFormat::Bed12 => {
            emit_bed12(out, &opts.chrom_label, &runs, opts.blocks_per_line)?
        }
    };
    Ok(KeepsetStats {
        genome_len,
        target_intervals: targets.len(),
        kept_runs: runs.len(),
        kept_bp,
        lines_written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::mask::{covered_by_bed, parse_bed};
    use tempfile::NamedTempFile;

    // Build a base vector from an ACGTN string (N -> sentinel).
    fn bases(s: &str) -> Vec<u8> {
        s.bytes()
            .map(|b| match b {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => N_SENTINEL,
            })
            .collect()
    }

    fn covered(runs: &[(u64, u64)], p: u64) -> bool {
        runs.iter().any(|&(s, e)| p >= s && p < e)
    }

    #[test]
    fn targets_are_kept_whole() {
        // Target [2,5) kept entirely with no homology halo. The target k-mers
        // (AACG/ACGG/CGGT at starts 2/3/4) and their reverse complements do not
        // recur elsewhere, so the immediate neighbors at 1 (TAAC) and 5 (GGTC)
        // stay out. (A periodic sequence like ACGTACGT would make position 1 an
        // RC homolog of a target k-mer and thus legitimately kept.)
        let b = bases("TTAACGGTCC");
        let runs = compute_keep_runs(&b, &[(2, 5)], 4, 0, 0).unwrap();
        assert!(covered(&runs, 2) && covered(&runs, 4));
        assert!(!covered(&runs, 1) && !covered(&runs, 5));
    }

    #[test]
    fn off_target_homolog_start_is_kept() {
        // Two exact copies of "ACGTAC" at 0 and 12, separated by a unique GGGGGG
        // filler. Target is the single start position 0, so the only target
        // k-mer is "ACGTAC"; k=6 -> the copy's START (12) is a forward homolog.
        let b = bases("ACGTACGGGGGGACGTACGGGG");
        let runs = compute_keep_runs(&b, &[(0, 1)], 6, 0, 0).unwrap();
        assert!(covered(&runs, 0), "target start kept");
        assert!(covered(&runs, 12), "homolog start at 12 must be kept");
        // Mid-kmer position 13 ("CGTACG") is not a target k-mer start, so the
        // start-only keep leaves it out (we do not keep the whole [p,p+k) span).
        assert!(!covered(&runs, 13), "only the suffix-start is kept");
        // The GGGGGG filler is neither target nor homolog.
        assert!(!covered(&runs, 7), "unique filler not kept");
    }

    #[test]
    fn reverse_complement_homolog_is_kept() {
        // Target k-mer "AACCGG" (at 0); its reverse complement "CCGGTT" planted
        // at 7, isolated from the target by an N so the rolling window cannot
        // also surface "CCGGTT" as a *forward* target k-mer. The only way to keep
        // position 7 is therefore the reverse-complement path: it exercises RC
        // homology in isolation (regression for the low- vs high-aligned key
        // mismatch that previously dropped RC homologs for k < 32).
        // Layout: 0..6 = target "AACCGG", 6 = N, 7..13 = RC copy "CCGGTT".
        let b = bases("AACCGGNCCGGTT");
        let runs = compute_keep_runs(&b, &[(0, 1)], 6, 0, 0).unwrap();
        assert!(covered(&runs, 0), "target start kept");
        assert!(covered(&runs, 7), "reverse-complement homolog start kept");
    }

    #[test]
    fn high_copy_kmer_dropped_by_cap() {
        // "AC" repeated -> the 2-mer "AC" occurs many times. With cap=2 it is
        // dropped, so only the target interval itself is kept (no homology halo).
        let b = bases("ACACACACACACACAC");
        let capped = compute_keep_runs(&b, &[(0, 2)], 2, 2, 0).unwrap();
        let kept_bp: u64 = capped.iter().map(|&(s, e)| e - s).sum();
        assert_eq!(kept_bp, 2, "over-cap homology dropped; only target kept");
        // Uncapped: the homologous starts are kept too (more than 2 bp).
        let uncapped = compute_keep_runs(&b, &[(0, 2)], 2, 0, 0).unwrap();
        let kept_bp_u: u64 = uncapped.iter().map(|&(s, e)| e - s).sum();
        assert!(kept_bp_u > 2, "uncapped keeps homologous AC starts");
    }

    #[test]
    fn inverted_repeat_counts_toward_cap() {
        // The target k-mer "AC" (at 0) occurs once forward, but its reverse
        // complement "GT" recurs four times in the "GTGTGTGT" tail. Because the
        // cap is applied on the strand-canonical key, the AC/GT family count is
        // 1 + 4 = 5, so an inverted high-copy repeat cannot slip under the cap.
        let b = bases("ACGTGTGTGT");
        // cap = 3 < 5 -> the whole family is dropped; only the target is kept.
        let capped = compute_keep_runs(&b, &[(0, 1)], 2, 3, 0).unwrap();
        let kept_bp: u64 = capped.iter().map(|&(s, e)| e - s).sum();
        assert_eq!(kept_bp, 1, "inverted high-copy repeat dropped by the cap");
        // Uncapped, the four reverse-complement homolog starts are kept too.
        let uncapped = compute_keep_runs(&b, &[(0, 1)], 2, 0, 0).unwrap();
        let kept_bp_u: u64 = uncapped.iter().map(|&(s, e)| e - s).sum();
        assert_eq!(kept_bp_u, 5, "uncapped keeps the RC homolog starts (0,2,4,6,8)");
    }

    #[test]
    fn flank_extends_and_merges_runs() {
        // Exercise the run-extraction + flank + merge logic directly on a
        // bitmap (isolated from homology). Single kept position 5.
        let mut keep = vec![false; 12];
        keep[5] = true;
        assert_eq!(runs_from_bitmap(&keep, 0), vec![(5, 6)]);
        assert_eq!(runs_from_bitmap(&keep, 1), vec![(4, 7)], "flank=1 widens to [4,7)");
        // Two nearby positions whose flanks overlap must merge into one run.
        let mut keep2 = vec![false; 12];
        keep2[3] = true;
        keep2[5] = true;
        assert_eq!(runs_from_bitmap(&keep2, 1), vec![(2, 7)], "overlapping flanks merge");
        // Flank clamps at the genome bounds.
        let mut keep3 = vec![false; 5];
        keep3[0] = true;
        keep3[4] = true;
        assert_eq!(runs_from_bitmap(&keep3, 10), vec![(0, 5)], "flank clamps to [0,len]");
    }

    #[test]
    fn invalid_k_rejected() {
        let b = bases("ACGT");
        assert!(compute_keep_runs(&b, &[], 0, 0, 0).is_err());
        assert!(compute_keep_runs(&b, &[], 33, 0, 0).is_err());
    }

    #[test]
    fn bed3_and_bed12_decode_to_identical_keepset() {
        // Emit the same runs as BED3 and BED12, parse both back through the
        // builder's parse_bed, and assert identical covered positions.
        let runs = vec![(10u64, 12u64), (20, 21), (100, 130)];
        let mut f3 = NamedTempFile::new().unwrap();
        emit_bed3(&mut f3, "genome", &runs).unwrap();
        f3.flush().unwrap();
        let mut f12 = NamedTempFile::new().unwrap();
        emit_bed12(&mut f12, "genome", &runs, 2).unwrap();
        f12.flush().unwrap();
        let iv3 = parse_bed(f3.path()).unwrap();
        let iv12 = parse_bed(f12.path()).unwrap();
        for p in 0..140u64 {
            assert_eq!(
                covered_by_bed(&iv3, p),
                covered_by_bed(&iv12, p),
                "BED3 and BED12 disagree at flat position {p}"
            );
        }
        // Spot-check the keep-set content itself.
        assert!(covered_by_bed(&iv3, 10) && covered_by_bed(&iv3, 11));
        assert!(!covered_by_bed(&iv3, 12));
        assert!(covered_by_bed(&iv3, 20) && !covered_by_bed(&iv3, 21));
        assert!(covered_by_bed(&iv3, 129) && !covered_by_bed(&iv3, 130));
    }

    #[test]
    fn target_bed_maps_per_contig_to_flat() {
        // Two contigs of length 10 and 8; a target on the second contig maps to
        // flat offset 10 + local.
        let contigs = vec![
            Contig { name: "chr1".into(), offset: 0, len: 10 },
            Contig { name: "chr2".into(), offset: 10, len: 8 },
        ];
        let mut bed = NamedTempFile::new().unwrap();
        writeln!(bed, "chr2\t2\t5").unwrap();
        bed.flush().unwrap();
        let flat = parse_target_bed_to_flat(bed.path(), &contigs).unwrap();
        assert_eq!(flat, vec![(12, 15)]);
    }

    #[test]
    fn duplicate_fasta_contig_names_rejected() {
        // Duplicate names would collapse in the name->offset map and misroute
        // targets, so reading the reference must fail with InvalidInput.
        let mut fa = NamedTempFile::new().unwrap();
        writeln!(fa, ">chr1\nACGT\n>chr1\nTTTT").unwrap();
        fa.flush().unwrap();
        let err = read_fasta_contigs(fa.path()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput for duplicate contig name, got {err:?}"
        );
        // A unique-named reference still reads cleanly.
        let mut ok = NamedTempFile::new().unwrap();
        writeln!(ok, ">chr1\nACGT\n>chr2\nTTTT").unwrap();
        ok.flush().unwrap();
        let (bases, contigs) = read_fasta_contigs(ok.path()).unwrap();
        assert_eq!(bases.len(), 8);
        assert_eq!(contigs.len(), 2);
    }

    #[test]
    fn canonical_low_key_unifies_strands() {
        // A k-mer and its reverse complement share one canonical key.
        let acgtac = 0b00_01_10_11_00_01u64; // A C G T A C, low-aligned, k=6
        let rc = reverse_complement_low(acgtac, 6);
        assert_eq!(canonical_low_key(acgtac, 6), canonical_low_key(rc, 6));
        // Reverse complement is an involution on the low-aligned encoding.
        assert_eq!(reverse_complement_low(rc, 6), acgtac);
    }

    #[test]
    fn target_bed_rejects_unknown_chrom_and_overrun() {
        let contigs = vec![Contig { name: "chr1".into(), offset: 0, len: 10 }];
        let mut bad_chrom = NamedTempFile::new().unwrap();
        writeln!(bad_chrom, "chrX\t0\t5").unwrap();
        bad_chrom.flush().unwrap();
        assert!(parse_target_bed_to_flat(bad_chrom.path(), &contigs).is_err());
        let mut overrun = NamedTempFile::new().unwrap();
        writeln!(overrun, "chr1\t0\t20").unwrap();
        overrun.flush().unwrap();
        assert!(parse_target_bed_to_flat(overrun.path(), &contigs).is_err());
    }

    #[test]
    fn target_bed_requires_exact_bed3() {
        // A wider record (BED6, BED12) must be rejected rather than silently
        // collapsed to its outer [start, end) span, since the keep-set output is
        // decoded block-precise by parse_bed.
        let contigs = vec![Contig {
            name: "chr1".into(),
            offset: 0,
            len: 100,
        }];
        // BED6: 6 columns -> rejected.
        let mut bed6 = NamedTempFile::new().unwrap();
        writeln!(bed6, "chr1\t0\t50\tname\t0\t+").unwrap();
        bed6.flush().unwrap();
        assert!(matches!(
            parse_target_bed_to_flat(bed6.path(), &contigs),
            Err(Error::InvalidInput { .. })
        ));
        // BED12: an exon-block transcript -> rejected (not widened to [10,90)).
        let mut bed12 = NamedTempFile::new().unwrap();
        writeln!(bed12, "chr1\t10\t90\ttx\t0\t+\t10\t90\t0\t2\t5,5,\t0,75,").unwrap();
        bed12.flush().unwrap();
        assert!(matches!(
            parse_target_bed_to_flat(bed12.path(), &contigs),
            Err(Error::InvalidInput { .. })
        ));
        // Exact BED3 still parses.
        let mut bed3 = NamedTempFile::new().unwrap();
        writeln!(bed3, "chr1\t10\t20").unwrap();
        bed3.flush().unwrap();
        assert_eq!(
            parse_target_bed_to_flat(bed3.path(), &contigs).unwrap(),
            vec![(10, 20)]
        );
    }

    #[test]
    fn compute_keep_runs_rejects_invalid_targets() {
        // compute_keep_runs is public and returns Result, so malformed intervals
        // must be rejected, not silently clipped (an inverted range would also
        // panic the bitmap slice). Genome below is length 10.
        let b = bases("ACGTACGTAC");
        // end > genome length.
        assert!(matches!(
            compute_keep_runs(&b, &[(0, 20)], 4, 0, 0),
            Err(Error::InvalidInput { .. })
        ));
        // end <= start.
        assert!(matches!(
            compute_keep_runs(&b, &[(5, 5)], 4, 0, 0),
            Err(Error::InvalidInput { .. })
        ));
        assert!(matches!(
            compute_keep_runs(&b, &[(8, 3)], 4, 0, 0),
            Err(Error::InvalidInput { .. })
        ));
        // A valid in-range interval still succeeds.
        assert!(compute_keep_runs(&b, &[(2, 5)], 4, 0, 0).is_ok());
    }
}
