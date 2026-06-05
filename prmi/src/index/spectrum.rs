// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Forward/backward SMEM "spectrum" primitives over the 2× (Fwd||RC) SA.
//! The reference base at a doubled-coordinate position is reconstructed from the
//! caller's FORWARD pac; the sentinel (position 2*l_pac) sorts smallest.

use crate::index::smem::{pac_base_at, validate_packed_pac, PacEncoding};
use crate::index::LearnedIndex;

/// Reference base (0..=3) at doubled-coordinate position `q` over the
/// `[Fwd(l_pac) || RC(l_pac)]` text, reconstructed from the forward `pac`:
/// - `q < l_pac`            -> `fwd(q)`
/// - `l_pac <= q < 2*l_pac` -> `3 - fwd(2*l_pac - 1 - q)`  (reverse-complement)
/// - `q >= 2*l_pac`         -> `None` (sentinel / end-of-text; sorts smallest)
#[inline]
pub(crate) fn doubled_base_at(pac: &[u8], enc: PacEncoding, l_pac: u64, q: u64) -> Option<u8> {
    if q >= 2 * l_pac {
        return None; // sentinel / past end
    }
    if q < l_pac {
        pac_base_at(pac, q, enc)
    } else {
        let fwd_pos = 2 * l_pac - 1 - q;
        pac_base_at(pac, fwd_pos, enc).map(|b| 3 - b)
    }
}

/// Compare `query` (bases 0..=3) against the reference suffix starting at
/// doubled-coordinate position `sa_pos`. Returns `(ref_less, lcp)`:
/// - `ref_less == true`  => the reference suffix is lexicographically < query.
/// - `lcp` = length of the common prefix.
///
/// Sentinel rule: if the reference runs out (`doubled_base_at` -> None) before a
/// mismatch, the reference is the smallest symbol, so `ref < query` (`ref_less =
/// true`). This matches the GSA `0`-sentinel ordering the SA was built with.
/// (Do NOT use the build-side `text_value_to_base` mapping here.)
#[inline]
pub(crate) fn compare_query_vs_suffix_2x(
    query: &[u8],
    sa_pos: u64,
    pac: &[u8],
    enc: PacEncoding,
    l_pac: u64,
) -> (bool, u32) {
    let mut lcp: u32 = 0;
    for (j, &qb) in query.iter().enumerate() {
        match doubled_base_at(pac, enc, l_pac, sa_pos + j as u64) {
            None => return (true, lcp), // ref exhausted -> ref < query
            Some(rb) if rb == qb => lcp += 1,
            Some(rb) => return (rb < qb, lcp), // mismatch -> order by base
        }
    }
    (false, lcp) // full query matched a prefix of the (longer-or-equal) ref suffix
}

/// One breakpoint of an SMEM spectrum: the SA interval `[sa_start, sa_start+occ_count)`
/// matching the query to length `match_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmemStep {
    /// SA interval start index (raw 2× SA).
    pub sa_start: u64,
    /// Occurrence count = SA interval size (both strands, native to the 2× SA).
    pub occ_count: u64,
    /// Match length (LCP) at this breakpoint.
    pub match_len: u64,
}

impl LearnedIndex {
    /// Forward spectrum from a pivot: the breakpoint trace of the narrowing SA
    /// interval, in ascending `match_len`, up to the maximal forward match.
    /// `query = read[pivot..]` (bases 0..=3). `pac` is the FORWARD pac; reference
    /// bases on the 2× text are reconstructed via `doubled_base_at`.
    ///
    /// Correctness-first: each prefix interval is found by binary search WITHIN
    /// the previous (wider) prefix interval — the intervals are nested, and
    /// interval(0) is the whole SA. We do NOT clamp to the learned-model error
    /// window (that window is only valid for the full-length key and would
    /// undercount short/wide prefixes). Model-accelerated launch (seeding the
    /// deep interval from `lookup()` and expanding outward) is a tracked perf
    /// optimization, deliberately deferred to keep this byte-identity-critical
    /// path provably correct.
    pub fn forward_spectrum(&self, query: &[u8], pac: &[u8], enc: PacEncoding) -> Vec<SmemStep> {
        let mut steps = Vec::new();
        if query.is_empty() {
            return steps;
        }
        // A packed pac that cannot hold its declared base count must fail closed
        // before any walker work: a truncated buffer would otherwise be misread
        // as a sentinel and yield a wrong interval. (`pac_base_at` is hardened to
        // return `None`, but truncation must not silently extend.)
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "forward_spectrum").is_err() {
                return steps;
            }
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        // Current interval = interval(m-1); starts as the whole SA (m=0).
        let mut lo = 0u64;
        let mut hi = sa_num;
        let mut prev_occ = u64::MAX;

        for m in 1..=query.len() {
            let qm = &query[..m];
            // Lower bound of qm within [lo, hi): first index whose suffix is >= qm.
            let mut a = lo;
            let mut b = hi;
            while a < b {
                let mid = a + (b - a) / 2;
                let pos = self.sa_position_for(mid);
                let (ref_less, _) = compare_query_vs_suffix_2x(qm, pos, pac, enc, l_pac);
                if ref_less {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            let k = a;
            // Upper bound within [k, hi): first index NOT sharing the full qm prefix.
            let mut c = k;
            let mut d = hi;
            while c < d {
                let mid = c + (d - c) / 2;
                let pos = self.sa_position_for(mid);
                let (_, lcp) = compare_query_vs_suffix_2x(qm, pos, pac, enc, l_pac);
                if (lcp as usize) >= qm.len() {
                    c = mid + 1;
                } else {
                    d = mid;
                }
            }
            let occ = c - k;
            if occ == 0 {
                break; // no suffix matches this prefix; maximal match is m-1
            }
            if occ != prev_occ {
                steps.push(SmemStep {
                    sa_start: k,
                    occ_count: occ,
                    match_len: m as u64,
                });
                prev_occ = occ;
            } else if let Some(last) = steps.last_mut() {
                last.match_len = m as u64;
            }
            // Narrow for the next, deeper prefix.
            lo = k;
            hi = c;
        }
        steps
    }

    /// Backward spectrum: refine the right-anchored interval `[sa_start,
    /// sa_start+occ_count)` (matching `read[pivot..pivot+anchor_len)`) leftward.
    /// Each emitted step's `match_len` is the TOTAL span (`anchor_len + left_ext`).
    /// `pac` is the FORWARD pac. Requires the sidecar to carry an `.isa` (returns
    /// an empty trace if `isa_for_refpos` is unavailable).
    // The argument list mirrors the spec/FFI contract for the backward primitive
    // (the right-anchored interval, the anchor span, and the read/pivot/pac/enc the
    // 2× compare needs); grouping them into a struct would only obscure the FFI shape.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum(
        &self,
        sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let mut steps = Vec::new();
        if self.isa_for_refpos(0).is_none() || occ_count == 0 {
            return steps;
        }
        // Fail closed on out-of-range inputs rather than panicking inside the
        // walk: `read[pivot - 1 - ..]` requires `pivot <= read.len()`, and
        // `sa_position_for(i)` requires the whole `[sa_start, sa_start+occ_count)`
        // interval to lie within the SA.
        if pivot > read.len() {
            return steps;
        }
        match sa_start.checked_add(occ_count) {
            Some(end) if end <= self.sa_num() => {}
            _ => return steps,
        }
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "backward_spectrum").is_err() {
                return steps;
            }
        }
        let l_pac = self.l_pac();
        let mut cur_start = sa_start;
        let mut cur_occ = occ_count;
        let mut left_ext: u64 = 0;

        while (left_ext as usize) < pivot {
            let c = read[pivot - 1 - left_ext as usize];
            if c >= 4 {
                break; // ambiguous read base
            }
            // Map each interval position's left neighbor through the inverse SA;
            // keep those whose left-neighbor base == c. They form a contiguous
            // SA sub-run (the c·Q interval).
            let mut new_min = u64::MAX;
            let mut new_max = 0u64;
            let mut count = 0u64;
            for i in cur_start..cur_start + cur_occ {
                let pos = self.sa_position_for(i);
                if pos == 0 {
                    continue; // no left neighbor
                }
                let q = pos - 1;
                if doubled_base_at(pac, enc, l_pac, q) == Some(c) {
                    let isa_idx = self.isa_for_refpos(q).expect("isa present");
                    new_min = new_min.min(isa_idx);
                    new_max = new_max.max(isa_idx);
                    count += 1;
                }
            }
            if count == 0 {
                break; // cannot extend further left
            }
            // Hard guard (not debug_assert): on the byte-identity-critical path, a
            // contiguity violation means a corrupt SA/ISA — fail loudly rather than
            // emit wrong seeds. O(1) on top of the O(occ) loop already run.
            // (Plan 4's FFI wrapper must catch_unwind / convert to an error code.)
            assert_eq!(
                new_max - new_min + 1,
                count,
                "c·Q interval not contiguous (corrupt SA/ISA): min={new_min} max={new_max} count={count}"
            );
            cur_start = new_min;
            cur_occ = count;
            left_ext += 1;
            steps.push(SmemStep {
                sa_start: cur_start,
                occ_count: cur_occ,
                match_len: anchor_len + left_ext,
            });
        }
        steps
    }
}

#[cfg(test)]
mod compare_tests {
    use super::*;

    fn pac() -> (Vec<u8>, PacEncoding, u64) {
        // forward ACGTAC = 0,1,2,3,0,1 ; l_pac = 6.
        (vec![0, 1, 2, 3, 0, 1], PacEncoding::Unpacked, 6)
    }

    #[test]
    fn compare_exact_prefix() {
        let (p, e, l) = pac();
        // query AC matches suffix at pos 0 (A C G T ...): lcp=2, ref not < query.
        let (ref_less, lcp) = compare_query_vs_suffix_2x(&[0, 1], 0, &p, e, l);
        assert_eq!((ref_less, lcp), (false, 2));
    }

    #[test]
    fn compare_mismatch_orders_by_base() {
        let (p, e, l) = pac();
        // query AG vs suffix at 0 (A C ...): mismatch at idx1, ref C(1) < query G(2) => ref_less.
        let (ref_less, lcp) = compare_query_vs_suffix_2x(&[0, 2], 0, &p, e, l);
        assert_eq!((ref_less, lcp), (true, 1));
    }

    #[test]
    fn compare_sentinel_is_smallest() {
        let (p, e, l) = pac();
        // Suffix at the last RC base (pos 2*l-1 = 11): one base then sentinel.
        // A 2-base query sharing that first base then hitting the sentinel =>
        // ref exhausted => ref_less = true, lcp = 1.
        let last = 2 * l - 1;
        let b = doubled_base_at(&p, e, l, last).unwrap();
        let (ref_less, lcp) = compare_query_vs_suffix_2x(&[b, b], last, &p, e, l);
        assert_eq!((ref_less, lcp), (true, 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubled_base_fwd_rc_sentinel() {
        // forward = ACGT = 0,1,2,3, l_pac = 4 (one-base-per-byte unpacked pac).
        let pac = [0u8, 1, 2, 3];
        let enc = PacEncoding::Unpacked;
        let l = 4u64;
        // forward half:
        assert_eq!(doubled_base_at(&pac, enc, l, 0), Some(0));
        assert_eq!(doubled_base_at(&pac, enc, l, 3), Some(3));
        // RC half: q=4 -> fwd(2*4-1-4)=fwd(3)=3 -> 3-3=0; q=7 -> fwd(0)=0 -> 3.
        assert_eq!(doubled_base_at(&pac, enc, l, 4), Some(0));
        assert_eq!(doubled_base_at(&pac, enc, l, 7), Some(3));
        // sentinel:
        assert_eq!(doubled_base_at(&pac, enc, l, 8), None);
        assert_eq!(doubled_base_at(&pac, enc, l, 99), None);
    }
}
