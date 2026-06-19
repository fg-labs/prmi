// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Training-pair masks. Drop SA positions whose 32-mer key is degenerate
//! (N-run, homopolymer, or user-supplied BED interval). The SA file
//! itself is unaffected; the masks only narrow the (key, sa_index) set
//! the RMI is fit and verified against.

use crate::error::{Error, Result};
use std::path::Path;

/// Configuration for which training positions to mask out before fitting.
///
/// All masks default to off. The trainer enables `mask_n_runs` by default
/// (i.e., `mask_n_runs: true`) unless the user passes `--no-mask-n-runs`.
#[derive(Debug, Clone, Default)]
pub struct MaskConfig {
    /// Skip positions whose 32-mer window covers any N base. ON by default.
    pub mask_n_runs: bool,
    /// If `Some(k)`, skip positions whose 32-mer window contains a run
    /// of the same base of length >= k. (k must be >= 2 to be meaningful;
    /// k > 32 effectively disables the check.)
    pub mask_homopolymers: Option<u32>,
    /// If `Some(intervals)`, skip positions falling in any of these
    /// reference intervals (0-based, half-open, sorted by start).
    pub mask_bed: Option<Vec<BedInterval>>,
    /// Source path of the BED file used to build `mask_bed`, stored for
    /// provenance in the `.meta` TOML. `None` when no BED was provided.
    pub mask_bed_path: Option<std::path::PathBuf>,
    /// If `Some(intervals)`, build a *tiered* (position-filtered) `.sa`:
    /// retain ONLY suffix-array entries whose forward reference coordinate
    /// falls in one of these intervals (0-based, half-open, sorted, merged).
    ///
    /// This is the opposite polarity of [`MaskConfig::mask_bed`] — a KEEP-mask
    /// — and, unlike `mask_bed` (which narrows only the RMI training pairs), it
    /// filters the `.sa` file ITSELF so the on-disk suffix array shrinks to the
    /// keep-set (Design Z). The full genome text/`.pac` is unchanged, so
    /// `match_len` is still computed against real genome bases and SA positions
    /// remain native genome coordinates. The keep-set is applied
    /// RC-symmetrically to the doubled `[Fwd||RC]` text via [`keep_doubled_pos`].
    /// `None` = the normal full build.
    pub keep_bed: Option<Vec<BedInterval>>,
    /// Source path of the BED used to build `keep_bed`, for `.meta` provenance.
    pub keep_bed_path: Option<std::path::PathBuf>,
}

/// A single half-open reference interval `[start, end)` from a BED file.
#[derive(Debug, Clone)]
pub struct BedInterval {
    /// 0-based start position (inclusive).
    pub start: u64,
    /// 0-based end position (exclusive).
    pub end: u64,
}

/// Parse a BED file into a sorted, merged list of `BedInterval`s.
///
/// Comment lines (`#`), track/browser header lines, and blank lines are
/// silently skipped. Requires at least 3 whitespace-separated columns
/// (chrom, start, end). The chromosome column is parsed but not stored —
/// intervals are concatenated genome-wide, matching the flat-genome
/// coordinate space used by the trainer.
///
/// Per-line BED3 vs BED12 (decided by column count):
/// - **BED3** (3–11 columns): keep the whole `[start, end)` interval. This is
///   the canonical target / coarse-homology shape and is unchanged.
/// - **BED12** (≥12 columns): keep ONLY the blocks. Columns 10/11/12 are
///   `blockCount` / comma-separated `blockSizes` / comma-separated
///   `blockStarts` (offsets relative to `start`); block `i` contributes
///   `[start + blockStarts[i], start + blockStarts[i] + blockSizes[i])`. This
///   lets the keep-set encode position-precise homology (e.g. length-1 / short
///   contiguous-run k-mer start positions) compactly, one BED line per region.
///
/// Returns an error if any data line has fewer than 3 columns, if start/end or
/// any block field is not a valid integer, if `end <= start`, or if a BED12
/// line's blocks are malformed (count mismatch, zero-length, or extending past
/// `end`).
///
/// The returned vector is sorted by `start` and overlapping intervals merged.
pub fn parse_bed(path: &Path) -> Result<Vec<BedInterval>> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("track")
            || line.starts_with("browser")
        {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            return Err(Error::InvalidInput {
                detail: format!(
                    "BED parse error at line {}: need 3 columns, got {}",
                    lineno + 1,
                    cols.len()
                ),
            });
        }
        let start: u64 = cols[1].parse().map_err(|_| Error::InvalidInput {
            detail: format!(
                "BED parse error at line {}: bad start {:?}",
                lineno + 1,
                cols[1]
            ),
        })?;
        let end: u64 = cols[2].parse().map_err(|_| Error::InvalidInput {
            detail: format!(
                "BED parse error at line {}: bad end {:?}",
                lineno + 1,
                cols[2]
            ),
        })?;
        if end <= start {
            return Err(Error::InvalidInput {
                detail: format!(
                    "BED parse error at line {}: end <= start ({} <= {})",
                    lineno + 1,
                    end,
                    start
                ),
            });
        }
        // A 12+-column line carries BED12 block fields (cols 10/11/12): keep
        // only the blocks. Anything narrower is BED3: keep the whole interval.
        if cols.len() >= 12 {
            push_bed12_blocks(&mut out, &cols, start, end, lineno)?;
        } else {
            out.push(BedInterval { start, end });
        }
    }
    out.sort_by_key(|i| i.start);

    // Merge overlapping / adjacent intervals so that covered_by_bed's binary
    // search (which only checks the immediately-preceding interval) is correct.
    // Without merging, an interval like [10, 100) followed by [20, 30) would
    // leave p=50 uncovered: binary search finds start=20, p < 30 is false.
    if out.len() > 1 {
        let mut merged: Vec<BedInterval> = Vec::with_capacity(out.len());
        for iv in out {
            match merged.last_mut() {
                Some(last) if iv.start < last.end => {
                    // Overlapping or touching — extend the current interval.
                    if iv.end > last.end {
                        last.end = iv.end;
                    }
                }
                _ => merged.push(iv),
            }
        }
        out = merged;
    }

    Ok(out)
}

/// Parse the BED12 block fields of a single line and append one `BedInterval`
/// per block to `out`.
///
/// `cols` is the whitespace-split line (already known to have ≥12 columns);
/// `start`/`end` are the line's chromStart/chromEnd. Per the BED12 spec, column
/// 10 (`cols[9]`) is `blockCount`, column 11 (`cols[10]`) is comma-separated
/// `blockSizes`, and column 12 (`cols[11]`) is comma-separated `blockStarts`
/// (offsets relative to `start`). A trailing comma (UCSC-style) is tolerated.
/// Block `i` contributes `[start + starts[i], start + starts[i] + sizes[i])`.
///
/// Errors (all `InvalidInput`, with the 1-based line number) on a non-integer
/// field, `blockCount == 0`, a size/start-count mismatch, a zero-length block,
/// or a block extending past `end`.
fn push_bed12_blocks(
    out: &mut Vec<BedInterval>,
    cols: &[&str],
    start: u64,
    end: u64,
    lineno: usize,
) -> Result<()> {
    let err = |detail: String| Error::InvalidInput { detail };
    let block_count: usize = cols[9].parse().map_err(|_| {
        err(format!(
            "BED parse error at line {}: bad blockCount {:?}",
            lineno + 1,
            cols[9]
        ))
    })?;
    if block_count == 0 {
        return Err(err(format!(
            "BED parse error at line {}: blockCount is 0 (a BED12 line must have >=1 block)",
            lineno + 1
        )));
    }
    // Split a comma-separated u64 list, tolerating a single trailing comma.
    let split_u64 = |field: &str, what: &str| -> Result<Vec<u64>> {
        field
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<u64>().map_err(|_| {
                    err(format!(
                        "BED parse error at line {}: bad {} entry {:?}",
                        lineno + 1,
                        what,
                        s
                    ))
                })
            })
            .collect()
    };
    let sizes = split_u64(cols[10], "blockSizes")?;
    let starts = split_u64(cols[11], "blockStarts")?;
    if sizes.len() != block_count || starts.len() != block_count {
        return Err(err(format!(
            "BED parse error at line {}: blockCount {} but got {} blockSizes and {} blockStarts",
            lineno + 1,
            block_count,
            sizes.len(),
            starts.len()
        )));
    }
    for i in 0..block_count {
        if sizes[i] == 0 {
            return Err(err(format!(
                "BED parse error at line {}: block {} has zero length",
                lineno + 1,
                i
            )));
        }
        // Checked arithmetic: a malformed line with huge blockStart/blockSize
        // could otherwise wrap and slip past the `be > end` bound check below.
        let bs = start.checked_add(starts[i]);
        let be = bs.and_then(|bs| bs.checked_add(sizes[i]));
        let (bs, be) = match (bs, be) {
            (Some(bs), Some(be)) => (bs, be),
            _ => {
                return Err(err(format!(
                    "BED parse error at line {}: block {} coordinates overflow",
                    lineno + 1,
                    i
                )))
            }
        };
        if be > end {
            return Err(err(format!(
                "BED parse error at line {}: block {} [{}, {}) extends past chromEnd {}",
                lineno + 1,
                i,
                bs,
                be,
                end
            )));
        }
        out.push(BedInterval { start: bs, end: be });
    }
    Ok(())
}

/// Decide whether a doubled-text suffix-array position should be RETAINED by a
/// tiered keep-mask (Design Z).
///
/// `pos` is a position in the doubled `[Fwd||RC]+sentinel` coordinate space
/// (`0..=2*l_pac`); `l_pac` is the forward genome length. The forward reference
/// coordinate is recovered as:
/// - `pos < l_pac`            → forward strand, coordinate `pos`;
/// - `l_pac <= pos < 2*l_pac` → reverse strand, coordinate `2*l_pac - 1 - pos`;
/// - `pos == 2*l_pac`         → the sentinel/empty-suffix row, always kept;
/// - `pos > 2*l_pac`          → out of range; rejected (fail closed).
///
/// Testing the *forward* coordinate against the keep-set for BOTH strands makes
/// the mask RC-symmetric: a forward position `p` and its reverse image
/// `2*l_pac - 1 - p` are kept together, which the RC-span/zigzag search logic
/// relies on. Returns `true` to keep the entry.
#[inline]
pub fn keep_doubled_pos(keep: &[BedInterval], pos: u64, l_pac: u64) -> bool {
    let doubled = match l_pac.checked_mul(2) {
        Some(v) => v,
        None => return false,
    };
    let fwd_coord = if pos < l_pac {
        pos
    } else if pos < doubled {
        doubled - 1 - pos
    } else if pos == doubled {
        // Sentinel/empty-suffix row: always retained so the SA's terminator
        // semantics (and the doubled-coordinate invariants) are preserved.
        return true;
    } else {
        // Out-of-range doubled coordinate: reject rather than silently retain a
        // malformed position.
        return false;
    };
    covered_by_bed(keep, fwd_coord)
}

/// Return `true` if position `p` is covered by any sorted, half-open interval.
///
/// Performs binary search — O(log n) per query.
pub fn covered_by_bed(intervals: &[BedInterval], p: u64) -> bool {
    // Find the largest interval whose start <= p.
    match intervals.binary_search_by_key(&p, |i| i.start) {
        // Exact hit on a start position.
        Ok(_) => true,
        Err(i) => {
            if i == 0 {
                false
            } else {
                let prev = &intervals[i - 1];
                p >= prev.start && p < prev.end
            }
        }
    }
}

/// A bit-packed boolean vector marking N positions. Stores one bit per position
/// instead of one byte (as `Vec<bool>` does) — an 8× memory reduction for the
/// genome-scale doubled N bitmap on the materialized training path.
///
/// Bits beyond `len` (in the final word) are always clear, so [`NBitmap::any`]
/// can scan whole words without masking.
#[derive(Debug, Clone, Default)]
pub struct NBitmap {
    words: Vec<u64>,
    len: usize,
}

impl NBitmap {
    /// An all-clear (no N) bitmap covering `len` positions.
    pub fn zeros(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
            len,
        }
    }

    /// Number of positions the bitmap covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitmap covers no positions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mark position `i` as N (set its bit).
    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(
            i < self.len,
            "NBitmap::set index {i} out of range (len={})",
            self.len
        );
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    /// Returns `true` if position `i` is marked N.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(
            i < self.len,
            "NBitmap::get index {i} out of range (len={})",
            self.len
        );
        (self.words[i >> 6] >> (i & 63)) & 1 != 0
    }

    /// Returns `true` if any position is marked N. Scans 64 positions per word.
    pub fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }
}

/// Return `true` if the window `bases[p..p+32]` contains any N position.
///
/// Uses the doubled-text [`NBitmap`] of N positions (built from the FASTA
/// parser's N bitmap in `build_sidecar_core`). Handles short windows at the end
/// of the reference gracefully by clamping.
#[inline]
pub fn n_in_window(n_positions: &NBitmap, p: usize) -> bool {
    let end = (p + 32).min(n_positions.len());
    (p..end).any(|i| n_positions.get(i))
}

/// Return `true` if `bases[p..p+32]` contains a run of the same base of
/// length >= `k`. Short windows at the end of the reference are handled by
/// clamping.
///
/// Returns `false` when `k > 32` (no 32-mer can contain a run >= 33).
#[inline]
pub fn homopolymer_in_window(bases: &[u8], p: usize, k: u32) -> bool {
    if k as usize > 32 {
        return false;
    }
    let end = (p + 32).min(bases.len());
    let window = &bases[p..end];
    if window.is_empty() {
        return false;
    }
    let mut run = 1u32;
    for w in window.windows(2) {
        if w[0] == w[1] {
            run += 1;
            if run >= k {
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // --- parse_bed -----------------------------------------------------------

    #[test]
    fn parse_bed_basic() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(
            f,
            "# comment\ntrack name=test\nbrowser position chr1:1-100\n\nchr1\t10\t20\nchr1\t30\t50"
        )
        .unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start, 10);
        assert_eq!(intervals[0].end, 20);
        assert_eq!(intervals[1].start, 30);
        assert_eq!(intervals[1].end, 50);
    }

    #[test]
    fn parse_bed_sorts_by_start() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t50\t60\nchr1\t10\t20\nchr1\t30\t40").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals[0].start, 10);
        assert_eq!(intervals[1].start, 30);
        assert_eq!(intervals[2].start, 50);
    }

    #[test]
    fn parse_bed_rejects_too_few_columns() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t10").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("3 columns"));
    }

    #[test]
    fn parse_bed_rejects_end_le_start() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t20\t10").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("end <= start"));
    }

    #[test]
    fn parse_bed_empty_file_ok() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "# only comments\n").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert!(intervals.is_empty());
    }

    #[test]
    fn parse_bed_merges_overlapping_intervals() {
        // Three intervals: [10, 50), [20, 30), [100, 200).
        // [10, 50) subsumes [20, 30); after merge: [10, 50) and [100, 200).
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t10\t50\nchr1\t20\t30\nchr1\t100\t200").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2, "overlapping intervals must be merged");
        assert_eq!(intervals[0].start, 10);
        assert_eq!(intervals[0].end, 50);
        assert_eq!(intervals[1].start, 100);
        assert_eq!(intervals[1].end, 200);

        // p=50 (exclusive end of first merged interval) is NOT covered.
        assert!(!covered_by_bed(&intervals, 50));
        // p=49 IS covered (last position in first merged interval).
        assert!(covered_by_bed(&intervals, 49));
    }

    #[test]
    fn covered_by_bed_overlapping_correctness() {
        // Without interval merging, covered_by_bed([10,100), [20,30)) would
        // fail to cover p=50: binary search lands on start=20, p < 30 is false.
        // After merging, the interval is [10, 100) and p=50 is correctly covered.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t10\t100\nchr1\t20\t30").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert!(
            covered_by_bed(&intervals, 50),
            "p=50 must be covered by the merged [10, 100) interval"
        );
    }

    // --- BED12 block parsing -------------------------------------------------

    #[test]
    fn parse_bed12_keeps_only_blocks_not_whole_span() {
        // Line spans [100, 200) but declares 2 blocks: [100,110) and [150,170).
        // BED12 keeps ONLY the blocks; the [110,150) and [170,200) gaps are out.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t2\t10,20\t0,50").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2, "two disjoint blocks, not the whole span");
        assert_eq!((intervals[0].start, intervals[0].end), (100, 110));
        assert_eq!((intervals[1].start, intervals[1].end), (150, 170));
        assert!(covered_by_bed(&intervals, 100));
        assert!(covered_by_bed(&intervals, 109));
        assert!(!covered_by_bed(&intervals, 110), "block1 end is exclusive");
        assert!(!covered_by_bed(&intervals, 120), "inter-block gap not kept");
        assert!(covered_by_bed(&intervals, 150));
        assert!(covered_by_bed(&intervals, 169));
        assert!(!covered_by_bed(&intervals, 170), "block2 end is exclusive");
        assert!(!covered_by_bed(&intervals, 199), "span tail past last block not kept");
    }

    #[test]
    fn parse_bed12_tolerates_trailing_comma() {
        // UCSC writes blockSizes/blockStarts with a trailing comma.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t2\t10,20,\t0,50,").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!((intervals[0].start, intervals[0].end), (100, 110));
        assert_eq!((intervals[1].start, intervals[1].end), (150, 170));
    }

    #[test]
    fn parse_bed12_length1_blocks_are_position_precise() {
        // The k-mer-homology generator's shape: single-position blocks marking
        // exact suffix-start positions. Two length-1 blocks at offsets 0 and 5.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t1000\t1010\tr\t0\t+\t1000\t1010\t0\t2\t1,1\t0,5").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!((intervals[0].start, intervals[0].end), (1000, 1001));
        assert_eq!((intervals[1].start, intervals[1].end), (1005, 1006));
        assert!(covered_by_bed(&intervals, 1000));
        assert!(!covered_by_bed(&intervals, 1001));
        assert!(covered_by_bed(&intervals, 1005));
        assert!(!covered_by_bed(&intervals, 1004));
    }

    #[test]
    fn parse_bed_mixed_bed3_and_bed12_lines() {
        // BED3 line kept whole; BED12 line kept by blocks; both coexist + merge.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t0\t50").unwrap();
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t1\t10\t0").unwrap();
        let intervals = parse_bed(f.path()).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!((intervals[0].start, intervals[0].end), (0, 50));
        assert_eq!((intervals[1].start, intervals[1].end), (100, 110));
    }

    #[test]
    fn parse_bed12_rejects_block_count_mismatch() {
        let mut f = NamedTempFile::new().unwrap();
        // blockCount=2 but only one size/start each.
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t2\t10\t0").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("blockCount"));
    }

    #[test]
    fn parse_bed12_rejects_block_past_chrom_end() {
        let mut f = NamedTempFile::new().unwrap();
        // Block [100, 260) extends past chromEnd 200.
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t1\t160\t0").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("extends past chromEnd"));
    }

    #[test]
    fn parse_bed12_rejects_zero_block_count() {
        let mut f = NamedTempFile::new().unwrap();
        // 12 columns so the BED12 branch is taken, but blockCount is 0.
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t0\t1\t0").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("blockCount is 0"));
    }

    #[test]
    fn parse_bed12_rejects_overflowing_block_coords() {
        let mut f = NamedTempFile::new().unwrap();
        // blockStart near u64::MAX would wrap start+blockStart; must be rejected
        // rather than slipping past the chromEnd bound via wraparound.
        writeln!(
            f,
            "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t1\t10\t18446744073709551610"
        )
        .unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("overflow"));
    }

    #[test]
    fn parse_bed12_rejects_zero_length_block() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "chr1\t100\t200\tr\t0\t+\t100\t200\t0\t1\t0\t0").unwrap();
        let err = parse_bed(f.path()).unwrap_err();
        assert!(format!("{err}").contains("zero length"));
    }

    // --- covered_by_bed ------------------------------------------------------

    #[test]
    fn covered_by_bed_boundaries() {
        // interval [10, 20)
        let intervals = vec![BedInterval { start: 10, end: 20 }];
        assert!(!covered_by_bed(&intervals, 9), "p=9 should be outside");
        assert!(
            covered_by_bed(&intervals, 10),
            "p=10 should be inside (start)"
        );
        assert!(covered_by_bed(&intervals, 15), "p=15 should be inside");
        assert!(
            covered_by_bed(&intervals, 19),
            "p=19 should be inside (last)"
        );
        assert!(
            !covered_by_bed(&intervals, 20),
            "p=20 should be outside (exclusive end)"
        );
    }

    #[test]
    fn covered_by_bed_empty() {
        assert!(!covered_by_bed(&[], 5));
    }

    #[test]
    fn covered_by_bed_multiple_intervals() {
        let intervals = vec![
            BedInterval { start: 10, end: 20 },
            BedInterval { start: 30, end: 50 },
            BedInterval {
                start: 100,
                end: 200,
            },
        ];
        assert!(!covered_by_bed(&intervals, 5));
        assert!(covered_by_bed(&intervals, 10));
        assert!(!covered_by_bed(&intervals, 25));
        assert!(covered_by_bed(&intervals, 30));
        assert!(covered_by_bed(&intervals, 49));
        assert!(!covered_by_bed(&intervals, 50));
        assert!(covered_by_bed(&intervals, 150));
    }

    // --- n_in_window ---------------------------------------------------------

    #[test]
    fn n_in_window_no_n() {
        let n_pos = NBitmap::zeros(64);
        assert!(!n_in_window(&n_pos, 0));
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_n_at_start() {
        let mut n_pos = NBitmap::zeros(64);
        n_pos.set(0);
        assert!(n_in_window(&n_pos, 0));
        // position 32 doesn't cover position 0
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_n_at_end_of_window() {
        let mut n_pos = NBitmap::zeros(64);
        n_pos.set(31); // last base of the window [0..32)
        assert!(n_in_window(&n_pos, 0));
        // window [32..64): position 31 is outside
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_clamps_at_end() {
        // Short reference: 10 bases, all N
        let mut n_pos = NBitmap::zeros(10);
        for i in 0..10 {
            n_pos.set(i);
        }
        assert!(n_in_window(&n_pos, 0)); // window is [0..10) — all N
        assert!(n_in_window(&n_pos, 5)); // window is [5..10) — all N
    }

    #[test]
    fn nbitmap_get_set_any() {
        let mut b = NBitmap::zeros(130); // spans 3 words (64+64+2)
        assert_eq!(b.len(), 130);
        assert!(!b.any());
        assert!(!b.get(0) && !b.get(63) && !b.get(64) && !b.get(129));
        b.set(63);
        b.set(64);
        b.set(129);
        assert!(b.any());
        assert!(b.get(63) && b.get(64) && b.get(129));
        assert!(!b.get(62) && !b.get(65) && !b.get(128));
        // The empty bitmap reports no N and is empty.
        let e = NBitmap::zeros(0);
        assert!(e.is_empty() && !e.any());
    }

    // --- homopolymer_in_window -----------------------------------------------

    #[test]
    fn homopolymer_no_run() {
        // ACGT repeated — no run of same base >= 2
        let bases: Vec<u8> = (0u8..4).cycle().take(64).collect();
        assert!(!homopolymer_in_window(&bases, 0, 2));
    }

    #[test]
    fn homopolymer_poly_a_run() {
        // 64 A's
        let bases = vec![0u8; 64]; // BASE_A = 0
        assert!(homopolymer_in_window(&bases, 0, 2));
        assert!(homopolymer_in_window(&bases, 0, 20));
        assert!(homopolymer_in_window(&bases, 0, 32));
        // k=33 > 32 — should always return false
        assert!(!homopolymer_in_window(&bases, 0, 33));
    }

    #[test]
    fn homopolymer_run_exactly_at_threshold() {
        // Build a 64-byte sequence where the window starting at position 0 has
        // exactly 10 A's at the start, followed by a C (to break the run),
        // then ACGT cycle for the remainder.
        // bases[0..10] = A (10 in a row), bases[10] = C, bases[11..] = ACGT cycle.
        let mut bases = vec![0u8; 64];
        for b in bases[0..10].iter_mut() {
            *b = 0; // BASE_A
        }
        bases[10] = 1; // BASE_C — breaks the A-run
        for (idx, b) in bases[11..].iter_mut().enumerate() {
            *b = (idx % 4) as u8; // ACGT cycle starting from A
        }
        // The ACGT cycle at [11..] produces A at positions 11,15,19,...
        // Each A is isolated by the cycle, so no run >= 2 in that region.
        // k=10 from position 0 should find the A-run (10 A's >= 10)
        assert!(homopolymer_in_window(&bases, 0, 10));
        // k=11 should NOT (the A-run is exactly 10; no other run in [0..32) is >= 11)
        assert!(!homopolymer_in_window(&bases, 0, 11));
    }

    #[test]
    fn homopolymer_k_greater_than_32_disabled() {
        let bases = vec![0u8; 64]; // all A
        assert!(!homopolymer_in_window(&bases, 0, 33));
        assert!(!homopolymer_in_window(&bases, 0, 100));
    }
}
