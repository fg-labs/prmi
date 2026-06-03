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
}

/// A single half-open reference interval `[start, end)` from a BED file.
#[derive(Debug, Clone)]
pub struct BedInterval {
    /// 0-based start position (inclusive).
    pub start: u64,
    /// 0-based end position (exclusive).
    pub end: u64,
}

/// Parse a BED file into a sorted list of `BedInterval`s.
///
/// Comment lines (`#`), track/browser header lines, and blank lines are
/// silently skipped. Requires at least 3 whitespace-separated columns
/// (chrom, start, end). The chromosome column is parsed but not stored —
/// intervals are concatenated genome-wide, matching the flat-genome
/// coordinate space used by the trainer.
///
/// Returns an error if any data line has fewer than 3 columns, or if
/// start/end are not valid integers, or if `end <= start`.
///
/// The returned vector is sorted by `start`.
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
        out.push(BedInterval { start, end });
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

/// Return `true` if the window `bases[p..p+32]` contains any N position.
///
/// Uses the `n_positions` bitmap produced during FASTA parsing. Handles
/// short windows at the end of the reference gracefully by clamping.
#[inline]
pub fn n_in_window(n_positions: &[bool], p: usize) -> bool {
    let end = (p + 32).min(n_positions.len());
    n_positions[p..end].iter().any(|&b| b)
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
        let n_pos = vec![false; 64];
        assert!(!n_in_window(&n_pos, 0));
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_n_at_start() {
        let mut n_pos = vec![false; 64];
        n_pos[0] = true;
        assert!(n_in_window(&n_pos, 0));
        // position 32 doesn't cover position 0
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_n_at_end_of_window() {
        let mut n_pos = vec![false; 64];
        n_pos[31] = true; // last base of the window [0..32)
        assert!(n_in_window(&n_pos, 0));
        // window [32..64): position 31 is outside
        assert!(!n_in_window(&n_pos, 32));
    }

    #[test]
    fn n_in_window_clamps_at_end() {
        // Short reference: 10 bases, all N
        let n_pos = vec![true; 10];
        assert!(n_in_window(&n_pos, 0)); // window is [0..10) — all N
        assert!(n_in_window(&n_pos, 5)); // window is [5..10) — all N
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
