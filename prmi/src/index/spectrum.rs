// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Forward/backward SMEM "spectrum" primitives over the 2× (Fwd||RC) SA.
//! The reference base at a doubled-coordinate position is reconstructed from the
//! caller's FORWARD pac; the sentinel (position 2*l_pac) sorts smallest.

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::index::smem::{pac_base_at, validate_packed_pac, PacEncoding};
use crate::index::LearnedIndex;
use crate::sidecar::kmt_file::KmerBounds;
use rayon::prelude::*;

/// Reference base (0..=3) at doubled-coordinate position `q` over the
/// `[Fwd(l_pac) || RC(l_pac)]` text, reconstructed from the forward `pac`:
/// - `q < l_pac`            -> `fwd(q)`
/// - `l_pac <= q < 2*l_pac` -> `3 - fwd(2*l_pac - 1 - q)`  (reverse-complement)
/// - `q >= 2*l_pac`         -> `None` (sentinel / end-of-text; sorts smallest)
///
/// The vectorized [`compare_query_vs_suffix_2x`] reads the doubled text in bulk
/// via `fill_doubled_chunk`; this per-base accessor backs the scalar oracle and
/// the single-base leftward walk in
/// [`mem_search_backward_from_hint`](LearnedIndex::mem_search_backward_from_hint).
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
///
/// Scalar one-base-at-a-time reference implementation. Retained as the source of
/// truth for the vectorized [`compare_query_vs_suffix_2x`]; the two are asserted
/// byte-identical by a proptest over both encodings. Test-only: the production
/// path uses the vectorized version exclusively.
#[cfg(test)]
#[inline]
pub(crate) fn compare_query_vs_suffix_2x_scalar(
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

/// Number of unpacked reference bases buffered per fetch in
/// [`compare_query_vs_suffix_2x`]. A multiple of 8 so the inner word-at-a-time
/// loop consumes whole `u64` lanes with at most one short tail per chunk.
const CHUNK_BASES: usize = 32;

/// Fill up to `out.len()` unpacked reference bases (each `0..=3`) of the doubled
/// `[Fwd || RC]` text starting at doubled position `q`, writing into `out[..n]`.
/// Returns the count `n` actually filled. `n < out.len()` iff the fill stopped
/// early because it reached a region boundary (forward→RC at `l_pac`) or the
/// sentinel/end-of-text (`2*l_pac`); the caller re-invokes for the next region
/// or treats a short fill at the sentinel as the reference being exhausted.
///
/// Vectorized common cases (`Unpacked` forward / RC) copy or mirror contiguous
/// byte runs; `Packed` decodes per base. A single call never crosses the
/// `l_pac` boundary — it fills only within the region containing `q`.
fn fill_doubled_chunk(pac: &[u8], enc: PacEncoding, l_pac: u64, q: u64, out: &mut [u8]) -> usize {
    if out.is_empty() || q >= 2 * l_pac {
        return 0;
    }
    if q < l_pac {
        // Forward region: bases are the reference at [q, l_pac).
        let avail = (l_pac - q) as usize;
        let n = avail.min(out.len());
        match enc {
            PacEncoding::Unpacked => {
                let start = q as usize;
                out[..n].copy_from_slice(&pac[start..start + n]);
            }
            PacEncoding::Packed { .. } => {
                for (i, slot) in out[..n].iter_mut().enumerate() {
                    // In-bounds by construction: q + i < l_pac <= num_bases.
                    *slot = pac_base_at(pac, q + i as u64, enc).unwrap();
                }
            }
        }
        n
    } else {
        // RC region: base at doubled position p is `fwd(2*l_pac-1-p) ^ 3`.
        // The forward positions descend as p ascends; stop at the sentinel.
        let avail = (2 * l_pac - q) as usize;
        let n = avail.min(out.len());
        match enc {
            PacEncoding::Unpacked => {
                // Forward positions covered: mirror(q) down to mirror(q+n-1).
                // mirror(q) = 2*l_pac-1-q (descending), so the slice is
                // pac[hi-n+1 ..= hi] read in reverse, each XORed with 3.
                let hi = (2 * l_pac - 1 - q) as usize; // mirror of q (highest fwd pos)
                let lo = hi + 1 - n; // mirror of q+n-1
                let src = &pac[lo..=hi];
                for (i, slot) in out[..n].iter_mut().enumerate() {
                    *slot = src[n - 1 - i] ^ 3;
                }
            }
            PacEncoding::Packed { .. } => {
                for (i, slot) in out[..n].iter_mut().enumerate() {
                    let fwd_pos = 2 * l_pac - 1 - (q + i as u64);
                    *slot = pac_base_at(pac, fwd_pos, enc).unwrap() ^ 3;
                }
            }
        }
        n
    }
}

/// Vectorized (word-at-a-time) re-implementation of
/// [`compare_query_vs_suffix_2x_scalar`]. Reads the doubled reference in
/// [`CHUNK_BASES`]-base chunks via [`fill_doubled_chunk`] and compares each
/// chunk against the query 8 bases (one `u64`) at a time, locating the first
/// mismatching base within a word from the XOR's trailing-zero count. The
/// observable contract — `(ref_less, lcp)` and the sentinel/exhaustion rule — is
/// byte-identical to the scalar version (asserted by proptest over both
/// encodings).
#[inline]
pub(crate) fn compare_query_vs_suffix_2x(
    query: &[u8],
    sa_pos: u64,
    pac: &[u8],
    enc: PacEncoding,
    l_pac: u64,
) -> (bool, u32) {
    let mut lcp: u32 = 0;
    let mut q_off: usize = 0; // bases of `query` already matched
    let mut buf = [0u8; CHUNK_BASES];

    while q_off < query.len() {
        let want = (query.len() - q_off).min(CHUNK_BASES);
        let n = fill_doubled_chunk(pac, enc, l_pac, sa_pos + q_off as u64, &mut buf[..want]);
        if n == 0 {
            // Ref exhausted (sentinel/end) with query bases remaining.
            return (true, lcp);
        }
        let qchunk = &query[q_off..q_off + n];
        let rchunk = &buf[..n];

        // Word-at-a-time over the filled bases; final partial word handled below.
        let mut k = 0usize;
        while k + 8 <= n {
            let qw = u64::from_le_bytes(qchunk[k..k + 8].try_into().unwrap());
            let rw = u64::from_le_bytes(rchunk[k..k + 8].try_into().unwrap());
            let xor = qw ^ rw;
            if xor != 0 {
                let byte = (xor.trailing_zeros() / 8) as usize;
                let idx = k + byte;
                return (rchunk[idx] < qchunk[idx], lcp + idx as u32);
            }
            k += 8;
        }
        // Tail (< 8 bases) of this chunk, one base at a time.
        while k < n {
            if rchunk[k] != qchunk[k] {
                return (rchunk[k] < qchunk[k], lcp + k as u32);
            }
            k += 1;
        }

        lcp += n as u32;
        q_off += n;
        // If `n < want` the fill stopped at the forward→RC boundary; the next
        // iteration fills the next region. A short fill that is actually the
        // sentinel/end is detected next round (`fill_doubled_chunk` returns 0).
    }
    (false, lcp) // full query matched a prefix of the (longer-or-equal) ref suffix
}

/// Precompute the keyed-compare `(nbases, mask)` from the query length. The mask
/// keeps the high `nbases * 2` bits (MSB-first) so only the active bases compare;
/// `nbases == 0` yields a zero mask that the compare never reads (it returns
/// early). Loop callers compute this once per prefix depth and pass it to
/// [`compare_query_vs_suffix_2x_keyed_with_mask`] so the inner probe loop does
/// not recompute the invariant per probe.
#[inline]
pub(crate) fn keyed_compare_mask(query_len: usize) -> (usize, u64) {
    let nbases = query_len.min(KMER_LEN);
    let mask: u64 = if nbases == 0 {
        0
    } else {
        // `bits` is in 2..=64, so the shift is well-defined.
        let bits = nbases * 2;
        if bits >= 64 {
            u64::MAX
        } else {
            !((1u64 << (64 - bits)) - 1)
        }
    };
    (nbases, mask)
}

/// Key-aware variant of [`compare_query_vs_suffix_2x`]. Uses a precomputed
/// query 32-mer key (`query_key = tokenize_32mer(query, min(32, query.len()))`)
/// and the stored suffix key (`stored_key`, from `LearnedIndex::key_at`) to
/// resolve the first `min(32, query.len())` bases of the compare from two `u64`
/// XORs — no per-base pac reads or forward/RC demux for that prefix. This is
/// BWA-MEME's `suffixarray_uint64` trick.
///
/// MUST produce a `(ref_less, lcp)` byte-identical to
/// [`compare_query_vs_suffix_2x_scalar`].
///
/// # Sentinel guard (correctness-critical)
///
/// The stored key was produced by `key_for_position_2x`, which **T-pads** a
/// suffix shorter than 32 real doubled bases (sentinel `0` → T=3). T-pad sorts
/// LARGE, but the compare's contract treats the sentinel / end-of-reference as
/// SMALLEST. Those orderings are OPPOSITE, so the key gives the WRONG ordering
/// for any suffix within 32 bases of the doubled-text end. The key path is
/// therefore taken ONLY when the suffix has ≥32 real doubled bases —
/// `sa_pos + 32 <= 2 * l_pac` — guaranteeing the stored key is a true 32-mer
/// with no T-pad. Otherwise the safe vectorized pac compare is used (it handles
/// the sentinel correctly).
///
/// `stored_key` is `None` for a mode-1 sidecar (no inline keys); that also
/// routes to the vectorized fallback.
#[inline]
pub(crate) fn compare_query_vs_suffix_2x_keyed(
    query: &[u8],
    query_key: u64,
    stored_key: Option<u64>,
    sa_pos: u64,
    pac: &[u8],
    enc: PacEncoding,
    l_pac: u64,
) -> (bool, u32) {
    let (nbases, mask) = keyed_compare_mask(query.len());
    compare_query_vs_suffix_2x_keyed_with_mask(
        query, query_key, stored_key, sa_pos, pac, enc, l_pac, nbases, mask,
    )
}

/// As [`compare_query_vs_suffix_2x_keyed`], but with the loop-invariant
/// `(nbases, mask)` precomputed by the caller via [`keyed_compare_mask`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn compare_query_vs_suffix_2x_keyed_with_mask(
    query: &[u8],
    query_key: u64,
    stored_key: Option<u64>,
    sa_pos: u64,
    pac: &[u8],
    enc: PacEncoding,
    l_pac: u64,
    nbases: usize,
    mask: u64,
) -> (bool, u32) {
    // Key path is valid only with a stored key AND a full 32 real doubled bases
    // at `sa_pos` (no T-pad / no sentinel within the first 32). Otherwise fall
    // back to the safe vectorized compare, which honours the sentinel rule.
    let stored_key = match stored_key {
        Some(k) if sa_pos + KMER_LEN as u64 <= 2 * l_pac => k,
        _ => return compare_query_vs_suffix_2x(query, sa_pos, pac, enc, l_pac),
    };

    if nbases == 0 {
        // Empty query matched (a prefix of) the ref suffix; scalar returns this.
        return (false, 0);
    }
    let xor = (query_key ^ stored_key) & mask;

    if xor != 0 {
        // First differing 2-bit field (MSB-first): leading_zeros/2 is its index.
        let idx = (xor.leading_zeros() / 2) as usize;
        // Order by that base. Extract the 2-bit field from each key.
        let shift = 2 * (KMER_LEN - 1 - idx) as u32;
        let qb = (query_key >> shift) & 0x3;
        let rb = (stored_key >> shift) & 0x3;
        return (rb < qb, idx as u32);
    }

    // First `nbases` bases are equal.
    if query.len() <= KMER_LEN {
        // Query fully consumed within the first 32 bases and matched a prefix of
        // the (longer-or-equal) ref suffix.
        (false, query.len() as u32)
    } else {
        // Query is longer than 32 and the first 32 matched. Continue from base 32
        // via the vectorized pac compare on the remaining query vs `sa_pos + 32`.
        let (ref_less, tail_lcp) = compare_query_vs_suffix_2x(
            &query[KMER_LEN..],
            sa_pos + KMER_LEN as u64,
            pac,
            enc,
            l_pac,
        );
        (ref_less, KMER_LEN as u32 + tail_lcp)
    }
}

/// Test/profiling-only SA-probe counter (a cold `sa_position_for` read per probe).
///
/// Enabled by the `spectrum-probe-count` feature; OFF by default so the production
/// hot path carries no counter. Used by `examples/profile_spectrum.rs` to report
/// median/p99 probes per backward left step before vs after the model launch.
#[cfg(feature = "spectrum-probe-count")]
pub mod probe_count {
    use std::cell::{Cell, RefCell};

    /// Max prefix depth tracked in the per-depth histogram (queries are ≤ ~100 bp).
    pub const MAX_DEPTH: usize = 256;

    thread_local! {
        static PROBES: Cell<u64> = const { Cell::new(0) };
        /// Current prefix depth `m`, set by the forward search before each step.
        static DEPTH: Cell<usize> = const { Cell::new(0) };
        /// Per-depth probe counts (`DEPTH_PROBES[m]`), accumulated across resets.
        static DEPTH_PROBES: RefCell<[u64; MAX_DEPTH]> = const { RefCell::new([0u64; MAX_DEPTH]) };
    }

    /// Reset the per-thread probe counter to zero.
    pub fn reset() {
        PROBES.with(|p| p.set(0));
    }

    /// Read the current per-thread probe count.
    pub fn get() -> u64 {
        PROBES.with(|p| p.get())
    }

    /// Set the current prefix depth `m` (clamped) for per-depth bucketing.
    #[inline]
    pub fn set_depth(m: usize) {
        DEPTH.with(|d| d.set(m.min(MAX_DEPTH - 1)));
    }

    /// Zero the per-depth probe histogram (totals accumulate until reset).
    pub fn reset_depth_probes() {
        DEPTH_PROBES.with(|d| *d.borrow_mut() = [0u64; MAX_DEPTH]);
    }

    /// Snapshot the per-depth probe histogram (`[m] = probes at prefix depth m`).
    pub fn depth_probes() -> Vec<u64> {
        DEPTH_PROBES.with(|d| d.borrow().to_vec())
    }

    /// Increment the per-thread probe counter (one cold SA position read) and
    /// the current depth's bucket.
    #[inline]
    pub(crate) fn bump() {
        PROBES.with(|p| p.set(p.get() + 1));
        let m = DEPTH.with(|d| d.get());
        DEPTH_PROBES.with(|d| d.borrow_mut()[m] += 1);
    }
}

/// No-op when the probe counter feature is disabled (production builds).
#[cfg(not(feature = "spectrum-probe-count"))]
#[inline(always)]
fn bump_probe() {}

/// Count one SA probe when the `spectrum-probe-count` feature is enabled.
#[cfg(feature = "spectrum-probe-count")]
#[inline]
fn bump_probe() {
    probe_count::bump();
}

/// Set the current prefix depth for per-depth probe bucketing (no-op in
/// production builds).
#[cfg(not(feature = "spectrum-probe-count"))]
#[inline(always)]
fn set_probe_depth(_m: usize) {}

/// Set the current prefix depth for per-depth probe bucketing.
#[cfg(feature = "spectrum-probe-count")]
#[inline]
fn set_probe_depth(m: usize) {
    probe_count::set_depth(m);
}

/// Result of [`LearnedIndex::mem_search`]: the maximal exact forward match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemMatch {
    /// Maximal exact-match length (0 if `query[0]` does not occur).
    pub match_len: u64,
    /// SA-interval start at `match_len` (0 when `match_len == 0`).
    pub sa_start: u64,
    /// Occurrence count at `match_len` (0 when `match_len == 0`).
    pub occ: u64,
}

/// One breakpoint of an SMEM spectrum: the SA interval `[sa_start, sa_start+occ_count)`
/// matching the query to length `match_len`.
///
/// `#[repr(C)]` with fields in this exact order lets the FFI (`prmi-sys`) fill
/// its `prmi_smem_step_t` output buffer in place via `*_fill`, with a
/// compile-time layout assertion guarding the equivalence. Do not reorder the
/// fields without updating `prmi_smem_step_t` and that assertion.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmemStep {
    /// SA interval start index (raw 2× SA).
    pub sa_start: u64,
    /// Occurrence count = SA interval size (both strands, native to the 2× SA).
    pub occ_count: u64,
    /// Match length (LCP) at this breakpoint.
    pub match_len: u64,
}

/// Capacity hint for a spectrum's step vector. A spectrum emits one step per
/// occurrence-count change (coalesced), which is a small handful even for long
/// queries, so this preallocation avoids the first few regrowth reallocations
/// without meaningfully over-allocating. It is a hint only and never affects
/// the contents.
const SPECTRUM_STEPS_HINT: usize = 16;

/// One backward-extension request (the lockstep analogue of the serial
/// `backward_spectrum` arguments). `read` is borrowed for the driver's lifetime.
pub struct BwdTask<'a> {
    /// SA interval start (from the forward search that seeded this anchor).
    pub sa_start: u64,
    /// SA interval size (occurrence count).
    pub occ_count: u64,
    /// Length of the right-anchored match (`read[pivot..pivot+anchor_len)`).
    pub anchor_len: u64,
    /// The full read.
    pub read: &'a [u8],
    /// Pivot index: the backward search extends left from `read[pivot-1]`.
    pub pivot: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum FbSub {
    GallopLeft,
    GallopRight,
    Binary,
}

/// `find_boundary` as a probe-driven state machine over `[dlo, dhi)`.
struct FbState {
    dlo: u64,
    dhi: u64,
    lo: u64,
    hi: u64,
    span: u64,
    sub: FbSub,
    done_val: Option<u64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Which {
    Lower,
    Upper,
}

impl FbState {
    fn new(dlo: u64, dhi: u64, seed_lo: u64, seed_hi: u64) -> Self {
        let mut lo = seed_lo.clamp(dlo, dhi);
        let mut hi = seed_hi.clamp(lo, dhi);
        let mut done_val = None;
        if lo == hi {
            if hi < dhi {
                hi += 1;
            } else if lo > dlo {
                lo -= 1;
            } else {
                done_val = Some(dlo);
            }
        }
        let span = (hi - lo).max(1);
        FbState {
            dlo,
            dhi,
            lo,
            hi,
            span,
            sub: FbSub::GallopLeft,
            done_val,
        }
    }

    /// The SA index this boundary wants to probe next, or `None` if the boundary
    /// is resolved (then `result()` holds the answer).
    fn next_probe(&mut self) -> Option<u64> {
        if self.done_val.is_some() {
            return None;
        }
        loop {
            match self.sub {
                FbSub::GallopLeft => {
                    if self.lo > self.dlo {
                        return Some(self.lo - 1);
                    }
                    self.sub = FbSub::GallopRight;
                }
                FbSub::GallopRight => {
                    if self.hi < self.dhi {
                        return Some(self.hi - 1);
                    }
                    self.sub = FbSub::Binary;
                }
                FbSub::Binary => {
                    if self.lo < self.hi {
                        return Some(self.lo + (self.hi - self.lo) / 2);
                    }
                    self.done_val = Some(self.lo);
                    return None;
                }
            }
        }
    }

    /// Feed `go_right(probed_index)` back; updates state. `mid` is the index that
    /// was probed (recomputed identically to `next_probe`).
    fn feed(&mut self, r: bool) {
        match self.sub {
            FbSub::GallopLeft => {
                if !r {
                    self.lo = self.lo.saturating_sub(self.span).max(self.dlo);
                    self.span = self.span.saturating_mul(2);
                } else {
                    self.sub = FbSub::GallopRight;
                }
            }
            FbSub::GallopRight => {
                if r {
                    self.hi = self.hi.saturating_add(self.span).min(self.dhi);
                    self.span = self.span.saturating_mul(2);
                    self.sub = FbSub::GallopLeft;
                } else {
                    self.sub = FbSub::Binary;
                }
            }
            FbSub::Binary => {
                let mid = self.lo + (self.hi - self.lo) / 2;
                if r {
                    self.lo = mid + 1;
                } else {
                    self.hi = mid;
                }
            }
        }
    }

    fn result(&self) -> Option<u64> {
        self.done_val
    }
}

struct BwdStepper<'a> {
    read: &'a [u8],
    pivot: usize,
    anchor_len: u64,
    sa_num: u64,
    p_start: usize,
    p_key: u64,
    win_hi: u64,
    which: Which,
    lower: u64,
    left_ext: u64,
    fb: FbState,
    steps: Vec<SmemStep>,
    finished: bool,
}

impl<'a> BwdStepper<'a> {
    fn p_len(&self) -> usize {
        self.pivot + self.anchor_len as usize - self.p_start
    }

    /// Initialize the Lower search for the current `left_ext`; returns false if the
    /// extension cannot continue (left_ext==pivot or ambiguous base). On true, `fb`
    /// is primed (its `next_probe` gives the first probe, possibly None if the
    /// window resolved degenerately — caller must then drive the boundary handoff).
    fn begin_left_step(&mut self, idx: &LearnedIndex) -> bool {
        if self.left_ext as usize >= self.pivot {
            return false;
        }
        let c = self.read[self.pivot - 1 - self.left_ext as usize];
        if c >= 4 {
            return false;
        }
        self.p_start = self.pivot - 1 - self.left_ext as usize;
        let p_slice = &self.read[self.p_start..self.pivot + self.anchor_len as usize];
        self.p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));
        let (pred, err) = idx.lookup(self.p_key);
        let win_lo = pred.saturating_sub(err);
        self.win_hi = pred.saturating_add(err).saturating_add(1).min(self.sa_num);
        self.which = Which::Lower;
        self.fb = FbState::new(0, self.sa_num, win_lo, self.win_hi);
        true
    }

    /// The SA index to probe next, or None if the whole task is finished. Drives
    /// across boundary/left-step handoffs that need no probe (degenerate windows).
    fn next_probe(&mut self, idx: &LearnedIndex) -> Option<u64> {
        loop {
            if self.finished {
                return None;
            }
            if let Some(mid) = self.fb.next_probe() {
                return Some(mid);
            }
            // Current boundary resolved with no (further) probe -> handoff.
            if !self.advance_boundary(idx) {
                return None;
            }
        }
    }

    /// One probe result: compute (ref_less,lcp), feed the active boundary, and if
    /// it just resolved, perform the boundary/left-step handoff.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        idx: &LearnedIndex,
        pos: u64,
        key: Option<u64>,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) {
        let (ref_less, lcp) = idx.bwd_compare(self, pos, key, pac, enc, l_pac);
        let r = match self.which {
            Which::Lower => ref_less,
            Which::Upper => (lcp as usize) >= self.p_len(),
        };
        self.fb.feed(r);
    }

    /// Move from a resolved boundary to the next state. Returns false when the
    /// whole task is finished. Lower-resolved -> start Upper. Upper-resolved ->
    /// emit the step (or stop), advance left_ext, begin the next left step.
    fn advance_boundary(&mut self, idx: &LearnedIndex) -> bool {
        let b = match self.fb.result() {
            Some(b) => b,
            None => return true, // not resolved; keep probing
        };
        match self.which {
            Which::Lower => {
                self.lower = b;
                self.which = Which::Upper;
                self.fb = FbState::new(self.lower, self.sa_num, self.win_hi, self.win_hi);
                true
            }
            Which::Upper => {
                let upper = b;
                if upper <= self.lower {
                    self.finished = true;
                    return false;
                }
                self.left_ext += 1;
                self.steps.push(SmemStep {
                    sa_start: self.lower,
                    occ_count: upper - self.lower,
                    match_len: self.anchor_len + self.left_ext,
                });
                if !self.begin_left_step(idx) {
                    self.finished = true;
                    return false;
                }
                true
            }
        }
    }
}

/// Precomputed per-length SA lower-bound tables for accelerating the shallow
/// (`m <= k`) bands of [`LearnedIndex::forward_spectrum`]. See
/// [`LearnedIndex::build_kmer_table`].
pub struct KmerTable {
    /// Max prefix length covered.
    pub k: u32,
    /// `lo[m-1][w]` / `hi[m-1][w]` = SA lower/upper bound of the length-`m` mer
    /// `w` (lex index), `4^m` entries each. Separate per-length tables (not a
    /// single padded k-mer table) and an explicit upper bound (not `lo[w+1]`),
    /// because short text-end suffixes are placed by the exact compare —
    /// derivation mis-orders a bare length-`<m` suffix against `qm`·A…A.
    lo: Vec<Vec<u64>>,
    hi: Vec<Vec<u64>>,
}

impl KmerTable {
    /// Borrow the table's components `(k, lo, hi)` for serialization to a
    /// `.kmt` file. `lo[m-1]` / `hi[m-1]` are the length-`m` bound arrays.
    pub(crate) fn parts(&self) -> (u32, &[Vec<u64>], &[Vec<u64>]) {
        (self.k, &self.lo, &self.hi)
    }
}

impl KmerBounds for KmerTable {
    #[inline]
    fn k(&self) -> u32 {
        self.k
    }
    #[inline]
    fn lo(&self, m: usize, w: u64) -> u64 {
        self.lo[m - 1][w as usize]
    }
    #[inline]
    fn hi(&self, m: usize, w: u64) -> u64 {
        self.hi[m - 1][w as usize]
    }
}

/// Sink for spectrum breakpoints. Lets the search core target a full `Vec`
/// (`forward_spectrum` / `backward_spectrum`), a single maximal step with no
/// allocation (`mem_search`), or a caller-provided slice (the FFI, no
/// intermediate `Vec`) without changing the search logic. The push-new-vs-extend
/// coalescing decision lives in [`push_step`]; a sink only stores.
pub(crate) trait StepSink {
    /// Append a new breakpoint.
    fn push_new(&mut self, step: SmemStep);
    /// Extend the most recently pushed breakpoint's `match_len` (same interval,
    /// deeper prefix). A no-op if nothing has been pushed.
    fn extend_last_match_len(&mut self, match_len: u64);
}

impl StepSink for Vec<SmemStep> {
    #[inline]
    fn push_new(&mut self, step: SmemStep) {
        self.push(step);
    }
    #[inline]
    fn extend_last_match_len(&mut self, match_len: u64) {
        if let Some(last) = self.last_mut() {
            last.match_len = match_len;
        }
    }
}

/// A [`StepSink`] that retains only the most recent breakpoint — exactly the
/// `.last()` of the full spectrum — so `mem_search` gets the maximal match
/// without allocating the intermediate `Vec`.
#[derive(Default)]
pub(crate) struct LastStepSink {
    last: Option<SmemStep>,
}

impl LastStepSink {
    #[inline]
    fn last(&self) -> Option<SmemStep> {
        self.last
    }
}

impl StepSink for LastStepSink {
    #[inline]
    fn push_new(&mut self, step: SmemStep) {
        self.last = Some(step);
    }
    #[inline]
    fn extend_last_match_len(&mut self, match_len: u64) {
        if let Some(last) = self.last.as_mut() {
            last.match_len = match_len;
        }
    }
}

/// A [`StepSink`] that writes breakpoints into a caller-provided slice (the FFI
/// output buffer), counting the TOTAL number emitted — including any beyond the
/// slice's capacity, which are counted but not written so the caller can detect
/// overflow and retry with a larger buffer.
pub(crate) struct SliceStepSink<'a> {
    out: &'a mut [SmemStep],
    count: usize,
}

impl<'a> SliceStepSink<'a> {
    #[inline]
    fn new(out: &'a mut [SmemStep]) -> Self {
        Self { out, count: 0 }
    }
    /// Total steps emitted (the value to report as `out_nsteps`).
    #[inline]
    fn count(&self) -> usize {
        self.count
    }
}

impl StepSink for SliceStepSink<'_> {
    #[inline]
    fn push_new(&mut self, step: SmemStep) {
        if let Some(slot) = self.out.get_mut(self.count) {
            *slot = step;
        }
        self.count += 1;
    }
    #[inline]
    fn extend_last_match_len(&mut self, match_len: u64) {
        // The most recently pushed step is at `count - 1`; update it only if it
        // was actually written (within capacity).
        if let Some(last) = self.count.checked_sub(1).and_then(|i| self.out.get_mut(i)) {
            last.match_len = match_len;
        }
    }
}

/// Emit/coalesce one forward breakpoint: push a new step when `occ` changes,
/// else extend the previous step's `match_len`. Matches `forward_spectrum`'s
/// inline coalescing so the tabled variant produces an identical trace.
#[inline]
fn push_step<S: StepSink + ?Sized>(
    sink: &mut S,
    prev_occ: &mut u64,
    sa_start: u64,
    occ: u64,
    match_len: u64,
) {
    if occ != *prev_occ {
        sink.push_new(SmemStep {
            sa_start,
            occ_count: occ,
            match_len,
        });
        *prev_occ = occ;
    } else {
        sink.extend_last_match_len(match_len);
    }
}

/// Lex index of the first `m` bases (`0..=3`) as a k-mer, MSB-first — the same
/// convention [`LearnedIndex::build_kmer_table`] uses for its `w` axis
/// (`bases[j] == (w >> (2 * (m - 1 - j))) & 0b11`). Used to look a left-extended
/// backward pattern's `m`-mer prefix up in the k-mer table for a probe-free seed.
#[inline]
fn kmer_lex_index(bases: &[u8], m: usize) -> u64 {
    let mut w = 0u64;
    for &b in &bases[..m] {
        w = (w << 2) | (u64::from(b) & 0b11);
    }
    w
}

impl LearnedIndex {
    /// Emit the maximal forward match of a UNIQUE suffix as one coalesced step,
    /// using a single SA probe. Precondition: the active narrowing interval is
    /// exactly `[uniq, uniq + 1)` — `query`'s prefix-so-far matches the lone
    /// suffix at `sa_position_for(uniq)`. Its full forward match is the LCP of
    /// the WHOLE `query` against that one suffix (`compare_query_vs_suffix_2x_keyed`
    /// returns it directly), so this replaces the two boundary binary searches
    /// the narrowing loop spends per remaining depth re-confirming a 1-wide
    /// interval. Byte-identical: the loop would emit exactly one `occ == 1` step
    /// extended to the same LCP, which `push_step`'s same-`occ` coalescing
    /// reproduces. `uniq` supplies `sa_start` only if this is the FIRST
    /// `occ == 1` step (a fresh push); once an `occ == 1` step already exists
    /// (`*prev_occ == 1`) `push_step` only extends its `match_len` and ignores
    /// `uniq` — and in that case `uniq` equals the existing step's `sa_start`
    /// anyway (the lone suffix's index does not change as the match deepens).
    #[allow(clippy::too_many_arguments)]
    fn push_unique_suffix_tail<S: StepSink + ?Sized>(
        &self,
        sink: &mut S,
        prev_occ: &mut u64,
        uniq: u64,
        query: &[u8],
        query_key: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) {
        let pos = self.sa_position_for(uniq);
        bump_probe();
        let key = self.key_at(uniq);
        let (_, lcp) =
            compare_query_vs_suffix_2x_keyed(query, query_key, key, pos, pac, enc, l_pac);
        push_step(sink, prev_occ, uniq, 1, u64::from(lcp));
    }

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
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        self.forward_spectrum_into(query, pac, enc, &mut steps);
        steps
    }

    /// Generic core of [`forward_spectrum`]: drives the narrowing search and
    /// emits each breakpoint into `sink`. The `Vec`-returning entry point, the
    /// allocation-free `mem_search` (via [`LastStepSink`]), and the FFI slice
    /// fill all share this one implementation.
    pub(crate) fn forward_spectrum_into<S: StepSink + ?Sized>(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        sink: &mut S,
    ) {
        if query.is_empty() {
            return;
        }
        // A packed pac that cannot hold its declared base count must fail closed
        // before any walk: a truncated buffer would otherwise be misread as a
        // sentinel and yield a wrong interval. (`pac_base_at` is hardened to
        // return `None`, but truncation must not silently extend.)
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "forward_spectrum").is_err() {
                return;
            }
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        // Precompute the query's 32-mer key ONCE: the query is fixed across all
        // prefix steps. The keyed compare masks this to the active prefix length
        // `qm.len()`, so the high `qm.len()` 2-bit fields (which equal the first
        // `qm.len()` bases) are reused for every `m` — no per-step recompute.
        let query_key = tokenize_32mer(query, query.len().min(KMER_LEN));
        // Current interval = interval(m-1); starts as the whole SA (m=0).
        let mut lo = 0u64;
        let mut hi = sa_num;
        let mut prev_occ = u64::MAX;

        for m in 1..=query.len() {
            // Bucket the probes of this prefix step under depth `m` (profiling
            // only; no-op without the `spectrum-probe-count` feature).
            set_probe_depth(m);
            // occ==1 fast path: a unique suffix's remaining forward match is one
            // direct compare, not a boundary search at every remaining depth.
            if hi - lo == 1 {
                self.push_unique_suffix_tail(
                    sink,
                    &mut prev_occ,
                    lo,
                    query,
                    query_key,
                    pac,
                    enc,
                    l_pac,
                );
                return;
            }
            let qm = &query[..m];
            // The keyed-compare mask depends only on `qm.len()`, invariant across
            // both inner binary searches at this depth — compute it once here
            // rather than per probe.
            let (qm_nbases, qm_mask) = keyed_compare_mask(qm.len());
            // Lower bound of qm within [lo, hi): first index whose suffix is >= qm.
            let mut a = lo;
            let mut b = hi;
            while a < b {
                self.prefetch_bsearch(a, b);
                let mid = a + (b - a) / 2;
                let pos = self.sa_position_for(mid);
                bump_probe();
                let key = self.key_at(mid);
                let (ref_less, _) = compare_query_vs_suffix_2x_keyed_with_mask(
                    qm, query_key, key, pos, pac, enc, l_pac, qm_nbases, qm_mask,
                );
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
                self.prefetch_bsearch(c, d);
                let mid = c + (d - c) / 2;
                let pos = self.sa_position_for(mid);
                bump_probe();
                let key = self.key_at(mid);
                let (_, lcp) = compare_query_vs_suffix_2x_keyed_with_mask(
                    qm, query_key, key, pos, pac, enc, l_pac, qm_nbases, qm_mask,
                );
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
            push_step(sink, &mut prev_occ, k, occ, m as u64);
            // Narrow for the next, deeper prefix.
            lo = k;
            hi = c;
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Lower bound of prefix `qm` within `[a, b)`: first SA index whose suffix
    /// is `>= qm`. Mirrors `forward_spectrum`'s lower-bound search exactly (same
    /// keyed compare), so a table built from it matches the search.
    fn lower_bound_prefix(
        &self,
        qm: &[u8],
        qm_key: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
        mut a: u64,
        mut b: u64,
    ) -> u64 {
        while a < b {
            let mid = a + (b - a) / 2;
            let pos = self.sa_position_for(mid);
            bump_probe();
            let key = self.key_at(mid);
            let (ref_less, _) =
                compare_query_vs_suffix_2x_keyed(qm, qm_key, key, pos, pac, enc, l_pac);
            if ref_less {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        a
    }

    #[allow(clippy::too_many_arguments)]
    /// Upper bound of prefix `qm` within `[c, b)`: first SA index (>= the lower
    /// bound) whose suffix does NOT have `qm` as a prefix. Mirrors
    /// `forward_spectrum`'s upper-bound search (lcp-based), so it orders short
    /// text-end suffixes identically — `lo[w+1]` cannot (a bare length-`<m`
    /// suffix sorts below `(qm+1)`·A…A but is still the true upper bound).
    fn upper_bound_prefix(
        &self,
        qm: &[u8],
        qm_key: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
        mut c: u64,
        mut d: u64,
    ) -> u64 {
        while c < d {
            let mid = c + (d - c) / 2;
            let pos = self.sa_position_for(mid);
            bump_probe();
            let key = self.key_at(mid);
            let (_, lcp) = compare_query_vs_suffix_2x_keyed(qm, qm_key, key, pos, pac, enc, l_pac);
            if (lcp as usize) >= qm.len() {
                c = mid + 1;
            } else {
                d = mid;
            }
        }
        c
    }

    /// Build the per-length SA lower/upper-bound tables (PROTOTYPE; `k <= 12`
    /// recommended).
    ///
    /// For each length `m = 1..=k`, `lo[m-1][w]` and `hi[m-1][w]` are the SA
    /// lower/upper bounds of the length-`m` mer `w` (`4^m` entries each),
    /// computed by the same lower-/upper-bound search [`forward_spectrum`] uses
    /// (so short text-end suffixes are ordered identically). The `m`-prefix
    /// interval of a query is then `[lo[m-1][p], hi[m-1][p])` for `p` = the
    /// query's `m`-mer index — pure index lookups, no SA probes. An explicit
    /// `hi` is required (not `lo[p+1]`): a bare length-`<m` text-end suffix can
    /// sort below `(qm+1)·A…A`, so the next mer's lower bound is not the upper
    /// bound of `qm`.
    ///
    /// Memory: `sum_{m=1}^{k} 4^m` entries × 2 (lo+hi) ≈ `2.67 · 4^k`
    /// (k=12 → ~44.7 M × 8 B ≈ 358 MB). A production version can collapse this
    /// to a single k-mer table plus a small text-end short-suffix correction
    /// (~134 MB).
    pub fn build_kmer_table(&self, k: u32, pac: &[u8], enc: PacEncoding) -> KmerTable {
        assert!((1..=16).contains(&k), "k-mer table k must be in 1..=16");
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        let mut lo: Vec<Vec<u64>> = Vec::with_capacity(k as usize);
        let mut hi: Vec<Vec<u64>> = Vec::with_capacity(k as usize);
        for m in 1..=k {
            let nm: u64 = 1u64 << (2 * m);
            let pairs: Vec<(u64, u64)> = (0..nm)
                .into_par_iter()
                .map(|w| {
                    let mut bases = [0u8; 16];
                    for (j, slot) in bases.iter_mut().enumerate().take(m as usize) {
                        *slot = ((w >> (2 * (m as usize - 1 - j))) & 0b11) as u8;
                    }
                    let qm = &bases[..m as usize];
                    let qm_key = tokenize_32mer(qm, m as usize);
                    let l = self.lower_bound_prefix(qm, qm_key, pac, enc, l_pac, 0, sa_num);
                    let h = self.upper_bound_prefix(qm, qm_key, pac, enc, l_pac, l, sa_num);
                    (l, h)
                })
                .collect();
            let (lm, hm): (Vec<u64>, Vec<u64>) = pairs.into_iter().unzip();
            lo.push(lm);
            hi.push(hm);
        }
        KmerTable { k, lo, hi }
    }

    /// Table-accelerated forward spectrum (PROTOTYPE): the shallow bands
    /// (`m <= table.k`) come from the K-mer table with zero SA probes; the deep
    /// bands (`m > table.k`) nested-narrow within the table's `m=k` interval,
    /// exactly as [`forward_spectrum`] does. Produces a byte-identical
    /// `SmemStep` trace.
    pub fn forward_spectrum_tabled(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        table: &impl KmerBounds,
    ) -> Vec<SmemStep> {
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        self.forward_spectrum_tabled_into(query, pac, enc, table, &mut steps);
        steps
    }

    /// Generic core of [`forward_spectrum_tabled`]; see [`Self::forward_spectrum_into`].
    pub(crate) fn forward_spectrum_tabled_into<S: StepSink + ?Sized>(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        table: &impl KmerBounds,
        sink: &mut S,
    ) {
        if query.is_empty() {
            return;
        }
        // Mirror `forward_spectrum_into`/`forward_spectrum_auto_into`: a packed
        // pac that cannot hold its declared base count must fail closed before
        // any walk (the `_tabled` entry point is reachable directly, not only via
        // `_auto`).
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "forward_spectrum_tabled").is_err() {
                return;
            }
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        let k = table.k() as usize;
        let query_key = tokenize_32mer(query, query.len().min(KMER_LEN));
        let mut prev_occ = u64::MAX;
        let mut lo = 0u64;
        let mut hi = sa_num;

        // ── shallow bands m=1..=min(k, len): O(1) from the per-length tables ──
        let shallow = k.min(query.len());
        let mut prefix: u64 = 0; // running m-mer lex index
        for m in 1..=shallow {
            set_probe_depth(m);
            prefix = (prefix << 2) | query[m - 1] as u64;
            let kk = table.lo(m, prefix);
            let cc = table.hi(m, prefix);
            let occ = cc.saturating_sub(kk);
            if occ == 0 {
                return; // maximal match is m-1 (mirrors forward_spectrum)
            }
            push_step(sink, &mut prev_occ, kk, occ, m as u64);
            lo = kk;
            hi = cc;
        }

        // ── deep bands m=k+1..=len: nested-narrow with SA probes ──────────────
        for m in (shallow + 1)..=query.len() {
            set_probe_depth(m);
            // occ==1 fast path (mirrors `forward_spectrum`): once the shallow
            // table bands or a prior deep step narrow to a single suffix, finish
            // with one direct compare instead of per-depth boundary searches.
            if hi - lo == 1 {
                self.push_unique_suffix_tail(
                    sink,
                    &mut prev_occ,
                    lo,
                    query,
                    query_key,
                    pac,
                    enc,
                    l_pac,
                );
                return;
            }
            let qm = &query[..m];
            let (qm_nbases, qm_mask) = keyed_compare_mask(qm.len());
            let mut a = lo;
            let mut b = hi;
            while a < b {
                self.prefetch_bsearch(a, b);
                let mid = a + (b - a) / 2;
                let pos = self.sa_position_for(mid);
                bump_probe();
                let key = self.key_at(mid);
                let (ref_less, _) = compare_query_vs_suffix_2x_keyed_with_mask(
                    qm, query_key, key, pos, pac, enc, l_pac, qm_nbases, qm_mask,
                );
                if ref_less {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            let kk = a;
            let mut c = kk;
            let mut d = hi;
            while c < d {
                self.prefetch_bsearch(c, d);
                let mid = c + (d - c) / 2;
                let pos = self.sa_position_for(mid);
                bump_probe();
                let key = self.key_at(mid);
                let (_, lcp) = compare_query_vs_suffix_2x_keyed_with_mask(
                    qm, query_key, key, pos, pac, enc, l_pac, qm_nbases, qm_mask,
                );
                if (lcp as usize) >= qm.len() {
                    c = mid + 1;
                } else {
                    d = mid;
                }
            }
            let occ = c - kk;
            if occ == 0 {
                break;
            }
            push_step(sink, &mut prev_occ, kk, occ, m as u64);
            lo = kk;
            hi = c;
        }
    }

    /// Forward spectrum using the loaded `.kmt` k-mer table when present
    /// (shallow bands resolved with zero SA probes), else the full search.
    /// Byte-identical to [`forward_spectrum`] either way; this is the
    /// transparent entry point the FFI calls.
    pub fn forward_spectrum_auto(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        self.forward_spectrum_auto_into(query, pac, enc, &mut steps);
        steps
    }

    /// Generic core of [`forward_spectrum_auto`]: dispatches to the tabled or
    /// full search and emits breakpoints into `sink`.
    pub(crate) fn forward_spectrum_auto_into<S: StepSink + ?Sized>(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        sink: &mut S,
    ) {
        // The packed-pac guard lives in the two delegates below
        // (`forward_spectrum_tabled_into` / `forward_spectrum_into`); both run it
        // before any walk, so a truncated buffer is rejected on every path here
        // (including the one-shot `mem_search` sink) without re-validating twice.
        match &self.kmt {
            Some(table) => self.forward_spectrum_tabled_into(query, pac, enc, table, sink),
            None => self.forward_spectrum_into(query, pac, enc, sink),
        }
    }

    /// Fill `out` with the forward spectrum's breakpoints and return the TOTAL
    /// number of steps. If the return value exceeds `out.len()`, the spectrum
    /// overflowed the buffer: the first `out.len()` steps were written and the
    /// caller should retry with a larger buffer. Lets the FFI write straight into
    /// its output buffer with no intermediate `Vec`.
    pub fn forward_spectrum_auto_fill(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        out: &mut [SmemStep],
    ) -> usize {
        let mut sink = SliceStepSink::new(out);
        self.forward_spectrum_auto_into(query, pac, enc, &mut sink);
        sink.count()
    }

    /// One LCP probe: how many leading bases of `query` match the suffix at SA
    /// index `mid` (`lcp(query, suffix[mid])`, capped at `query.len()`). One cold
    /// SA-position read plus the keyed 2× compare — the boundary-neighbour probe
    /// the hinted-spectrum parent walk uses to find the next breakpoint depth.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn lcp_query_suffix(
        &self,
        query: &[u8],
        query_key: u64,
        mid: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) -> u64 {
        bump_probe();
        let pos = self.sa_position_for(mid);
        let key = self.key_at(mid);
        let (_, lcp) =
            compare_query_vs_suffix_2x_keyed(query, query_key, key, pos, pac, enc, l_pac);
        u64::from(lcp)
    }

    /// PROTOTYPE — forward spectrum launched from an EXACT SA-index `hint`,
    /// producing the SAME [`SmemStep`] breakpoint trace as [`forward_spectrum`]
    /// but via a parent-interval (suffix-tree-parent) walk instead of cold
    /// per-depth narrowing. This is the trace-returning twin of
    /// [`mem_search_from_hint`](Self::mem_search_from_hint): that method keeps
    /// only the maximal interval; this keeps EVERY breakpoint, so a caller can
    /// drive `min_intv`-gated multi-anchor emission (BWA-MEME reseed) from the
    /// `no_search` launch.
    ///
    /// Mechanism — deep → shallow, `O(number of breakpoints)` probes (not
    /// `O(match_len)`, the trap a naive per-depth re-narrowing falls into by
    /// re-probing the unique tail):
    /// 1. Confirm `L = LCP(query, suffix[hint])` and recover the maximal interval
    ///    `[lo, hi)` by two boundary gallops seeded at `hint` — exactly as
    ///    [`mem_search_from_hint`].
    /// 2. Emit the band `(lo, hi - lo, L)`. The parent band's deepest depth is
    ///    `m' = max(LCP(query, suffix[lo - 1]), LCP(query, suffix[hi]))` — the two
    ///    boundary-neighbour suffixes, two probes. Gallop `[lo, hi)` outward at
    ///    `query[..m']` to the parent interval and repeat until `m' == 0` (the
    ///    depth-1 band — no shallower breakpoint).
    ///
    /// Bands are collected deep→shallow then reversed to ascending `match_len`,
    /// matching [`forward_spectrum`]'s order. Nested narrowing makes `occ`
    /// strictly decreasing, so one band == one `push_step` breakpoint: the traces
    /// are byte-identical when `hint` lies in the maximal interval (the same
    /// contract as [`mem_search_from_hint`] — a wrong hint yields a shorter
    /// divergent trace the caller detects via continued-match; verified by
    /// `forward_spectrum_from_hint_equals_cold`).
    ///
    /// Correctness does NOT depend on the gallop seeds (they only set probe
    /// count): every boundary is the deterministic flip of the same
    /// `ref_less`/`shares_prefix` predicates [`forward_spectrum`] uses, searched
    /// over the same global SA, so the located intervals are identical.
    ///
    /// Falls back to the cold [`forward_spectrum`] for `hint == 0`, an
    /// out-of-range hint, or an empty query.
    pub fn forward_spectrum_from_hint(
        &self,
        query: &[u8],
        hint: u64,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let sa_num = self.sa_num();
        if query.is_empty() || hint == 0 || hint >= sa_num {
            return self.forward_spectrum(query, pac, enc);
        }
        let l_pac = self.l_pac();
        let query_key = tokenize_32mer(query, query.len().min(KMER_LEN));

        // (1) Confirm the maximal match length at the hint.
        let max_len = {
            let pos = self.sa_position_for(hint);
            bump_probe();
            let key = self.key_at(hint);
            let (_, lcp) =
                compare_query_vs_suffix_2x_keyed(query, query_key, key, pos, pac, enc, l_pac);
            lcp as usize
        };
        if max_len == 0 {
            return Vec::new(); // hint shares no prefix with the query (contract violation)
        }

        // Recover the maximal interval [lo, hi) at depth max_len, seeded at hint.
        let qm = &query[..max_len];
        let qm_key = tokenize_32mer(qm, qm.len().min(KMER_LEN));
        let mut lo = self.find_boundary(0, sa_num, hint, hint + 1, |mid| {
            self.ref_less(qm, qm_key, mid, pac, enc, l_pac)
        });
        let mut hi = self.find_boundary(lo, sa_num, hint, hint + 1, |mid| {
            self.shares_prefix(qm, qm_key, mid, pac, enc, l_pac)
        });
        let mut band_deepest = max_len as u64;

        // (2) Deep → shallow parent-interval walk: one band per breakpoint.
        let mut bands: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        loop {
            bands.push(SmemStep {
                sa_start: lo,
                occ_count: hi - lo,
                match_len: band_deepest,
            });

            // Next-shallower breakpoint depth = max LCP of the two boundary
            // neighbours (the suffixes just outside [lo, hi)).
            let lcp_left = if lo > 0 {
                self.lcp_query_suffix(query, query_key, lo - 1, pac, enc, l_pac)
            } else {
                0
            };
            let lcp_right = if hi < sa_num {
                self.lcp_query_suffix(query, query_key, hi, pac, enc, l_pac)
            } else {
                0
            };
            let m_parent = lcp_left.max(lcp_right);
            if m_parent == 0 {
                break; // current band's shallowest depth is 1: no shallower breakpoint
            }

            // Parent interval at depth m_parent: gallop [lo, hi) outward. The
            // parent contains the current interval, so its lower bound is ≤ lo
            // (domain [0, lo + 1)) and its upper bound is ≥ hi.
            let pqm = &query[..m_parent as usize];
            let pqm_key = tokenize_32mer(pqm, pqm.len().min(KMER_LEN));
            let new_lo = self.find_boundary(0, lo + 1, lo, lo + 1, |mid| {
                self.ref_less(pqm, pqm_key, mid, pac, enc, l_pac)
            });
            let new_hi = self.find_boundary(new_lo, sa_num, hi.saturating_sub(1), hi, |mid| {
                self.shares_prefix(pqm, pqm_key, mid, pac, enc, l_pac)
            });
            lo = new_lo;
            hi = new_hi;
            band_deepest = m_parent;
        }

        bands.reverse(); // ascending match_len, matching forward_spectrum's order
        bands
    }

    /// One-shot maximal exact forward match: the longest prefix of `query` that
    /// occurs in the reference, with its SA interval. Equals the MAXIMAL (deepest)
    /// step of [`forward_spectrum_auto`] — byte-identical by construction; reuses
    /// the k-mer table + occ==1 fast path. `match_len == 0` (and `sa_start/occ ==
    /// 0`) when `query` is empty or `query[0]` does not occur.
    pub fn mem_search(&self, query: &[u8], pac: &[u8], enc: PacEncoding) -> MemMatch {
        // Track only the maximal step — no intermediate Vec for this one-shot call.
        let mut sink = LastStepSink::default();
        self.forward_spectrum_auto_into(query, pac, enc, &mut sink);
        match sink.last() {
            Some(s) => MemMatch {
                match_len: s.match_len,
                sa_start: s.sa_start,
                occ: s.occ_count,
            },
            None => MemMatch {
                match_len: 0,
                sa_start: 0,
                occ: 0,
            },
        }
    }

    /// One-shot maximal forward match launched from an EXACT SA index hint
    /// (`hint`), skipping the model lookup AND the nested interval-narrowing —
    /// BWA-MEME's `no_search=true` / `mem_search_tradeoff` fast path. `hint` MUST
    /// be a SA index whose suffix lies in the query's maximal-match interval
    /// (the caller obtains it from the inverse SA, `prmi_isa_at(refpos)`); the
    /// result is then byte-identical to [`mem_search`] (the launch changes speed,
    /// never the answer — see `mem_search_hint_equals_unhinted`). A wrong hint is
    /// a caller contract violation and yields an undefined (non-equal) answer.
    ///
    /// Mechanism: one keyed compare gives `match_len = LCP(query, suffix[hint])`
    /// (every suffix in the maximal interval has this same LCP — none matches
    /// deeper, or the interval would not be maximal). When `want_interval`, two
    /// boundary searches gallop outward from `hint` to recover `[sa_start, sa_start
    /// + occ)` at depth `match_len`; otherwise only `match_len` is computed.
    ///
    /// `hint` must satisfy `0 < hint < sa_num` (0 is the no-hint sentinel and the
    /// sentinel row; out of range is a caller error the FFI rejects).
    pub fn mem_search_from_hint(
        &self,
        query: &[u8],
        hint: u64,
        want_interval: bool,
        pac: &[u8],
        enc: PacEncoding,
    ) -> MemMatch {
        let zero = MemMatch {
            match_len: 0,
            sa_start: 0,
            occ: 0,
        };
        // Fail closed on an undersized packed pac (as the non-hint paths do): a
        // truncated buffer would otherwise be misread inside the compare helpers
        // (a missing sentinel byte → wrong LCP, or an out-of-bounds unwrap).
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "mem_search_from_hint").is_err() {
                return zero;
            }
        }
        if query.is_empty() || hint == 0 || hint >= self.sa_num() {
            return zero;
        }
        let l_pac = self.l_pac();
        let query_key = tokenize_32mer(query, query.len().min(KMER_LEN));
        // Confirm: the maximal match length is the LCP of the query against the
        // hinted suffix (one probe instead of the whole narrowing search).
        let pos = self.sa_position_for(hint);
        bump_probe();
        let key = self.key_at(hint);
        let (_, lcp) =
            compare_query_vs_suffix_2x_keyed(query, query_key, key, pos, pac, enc, l_pac);
        let match_len = u64::from(lcp);
        if match_len == 0 {
            return zero;
        }
        if !want_interval {
            return MemMatch {
                match_len,
                sa_start: 0,
                occ: 0,
            };
        }
        // Recover the maximal interval `[lower, upper)` of `query[..match_len]`,
        // seeding the boundary searches at the hint (which lies inside it).
        let sa_num = self.sa_num();
        let qm = &query[..match_len as usize];
        let qm_key = tokenize_32mer(qm, qm.len().min(KMER_LEN));
        let lower = self.find_boundary(0, sa_num, hint, hint + 1, |mid| {
            self.ref_less(qm, qm_key, mid, pac, enc, l_pac)
        });
        let upper = self.find_boundary(lower, sa_num, hint, hint + 1, |mid| {
            self.shares_prefix(qm, qm_key, mid, pac, enc, l_pac)
        });
        MemMatch {
            match_len,
            sa_start: lower,
            occ: upper.saturating_sub(lower),
        }
    }

    /// One-shot maximal exact BACKWARD (leftward) match: the deepest left
    /// extension of the right anchor `[sa_start, sa_start+occ_count)` (matching
    /// `read[pivot..pivot+anchor_len)`), with its SA interval. The backward twin
    /// of [`mem_search`]. `match_len` is the TOTAL matched span
    /// (`anchor_len + left_ext`).
    ///
    /// - When [`backward_spectrum`](Self::backward_spectrum) produces ≥1 left
    ///   step, the result equals its MAXIMAL (deepest) step — byte-identical by
    ///   construction.
    /// - When no left extension is possible (`pivot == 0`, an ambiguous left
    ///   base, or the anchor cannot extend), the maximal match is the anchor
    ///   itself: `(sa_start, occ_count, anchor_len)`.
    /// - When `occ_count == 0` (the anchor does not occur), all-zero.
    #[allow(clippy::too_many_arguments)]
    pub fn mem_search_backward(
        &self,
        sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
    ) -> MemMatch {
        if occ_count == 0 {
            return MemMatch {
                match_len: 0,
                sa_start: 0,
                occ: 0,
            };
        }
        // Track only the maximal left step — no intermediate Vec for this
        // one-shot call (production `seed_override` is always `None`).
        let mut sink = LastStepSink::default();
        self.backward_spectrum_inner_into(
            sa_start, occ_count, anchor_len, read, pivot, pac, enc, None, &mut sink,
        );
        match sink.last() {
            Some(s) => MemMatch {
                match_len: s.match_len,
                sa_start: s.sa_start,
                occ: s.occ_count,
            },
            // No left extension: the anchor itself is the maximal backward match.
            None => MemMatch {
                match_len: anchor_len,
                sa_start,
                occ: occ_count,
            },
        }
    }

    /// One-shot maximal LEFT extension launched from an EXACT SA index hint —
    /// the backward twin of [`mem_search_from_hint`](Self::mem_search_from_hint),
    /// BWA-MEME's `no_search` for the left direction. `hint` is the SA index whose
    /// suffix is the right anchor (`read[pivot..pivot+anchor_len]`); the caller
    /// obtains it from the inverse SA (`prmi_isa_at(refpos)`). No prior anchor
    /// interval is needed — `hint` replaces it (this drops the caller's separate
    /// length-1 anchor search).
    ///
    /// Mechanism: `p_anchor = sa_position_for(hint)` is the anchor's genomic
    /// position; the left extension is a direct reference walk
    /// (`read[pivot-1-k]` vs `doubled_base_at(p_anchor-1-k)`) — no SA search, no
    /// per-step model launch — giving `left_ext` in `O(left_ext)`. When the
    /// interval is requested, the maximal pattern `read[pivot-left_ext ..
    /// pivot+anchor_len]` (which starts at text position `p_anchor - left_ext`) is
    /// resolved by two boundary searches tight-seeded from
    /// `isa_at(p_anchor - left_ext)` (or a correct loose seed without `.isa`).
    ///
    /// `*match_len` is ALWAYS the TOTAL anchored span `anchor_len + left_ext` —
    /// the caller's gating signal: a short span means the hint diverged from the
    /// read, so fall back to the model launch. The result equals
    /// [`mem_search_backward`](Self::mem_search_backward) (`est_hint == 0`)
    /// whenever `hint` is at a maximal-extension locus (the launch changes speed,
    /// never the answer).
    ///
    /// `hint` must satisfy `0 < hint < sa_num`.
    #[allow(clippy::too_many_arguments)]
    pub fn mem_search_backward_from_hint(
        &self,
        read: &[u8],
        pivot: usize,
        anchor_len: u64,
        hint: u64,
        want_interval: bool,
        pac: &[u8],
        enc: PacEncoding,
    ) -> MemMatch {
        let zero = MemMatch {
            match_len: 0,
            sa_start: 0,
            occ: 0,
        };
        // Fail closed on an undersized packed pac (as the non-hint paths do): a
        // truncated buffer would otherwise be misread inside `doubled_base_at`.
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "mem_search_backward_from_hint").is_err() {
                return zero;
            }
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        // Checked: a large caller-provided `anchor_len` must not wrap when
        // narrowed to usize or added to `pivot` (a wrapped window bound would
        // pass the `> read.len()` guard and read out of range).
        let anchor_len_usize = match usize::try_from(anchor_len) {
            Ok(v) => v,
            Err(_) => return zero,
        };
        let anchor_end = match pivot.checked_add(anchor_len_usize) {
            Some(v) => v,
            None => return zero,
        };
        if hint == 0 || hint >= sa_num || anchor_end > read.len() {
            return zero;
        }
        // Genomic position the anchor aligns to (text start of `read[pivot..]`).
        let p_anchor = self.sa_position_for(hint);

        // Direct leftward reference walk: extend while the read base matches the
        // reference base one position further left. Stops at an ambiguous read
        // base, a read/text boundary, a sentinel, or a mismatch.
        let mut left_ext: u64 = 0;
        while (left_ext as usize) < pivot && p_anchor > left_ext {
            let read_base = read[pivot - 1 - left_ext as usize];
            if read_base >= 4 {
                break;
            }
            let ref_pos = p_anchor - 1 - left_ext;
            match doubled_base_at(pac, enc, l_pac, ref_pos) {
                Some(b) if b == read_base => left_ext += 1,
                _ => break,
            }
        }
        // Checked: the FFI casts `match_len` to u32 downstream, so a u64 wrap
        // here would surface as a silently truncated match length.
        let match_len = match anchor_len.checked_add(left_ext) {
            Some(v) => v,
            None => return zero,
        };

        if !want_interval {
            return MemMatch {
                match_len,
                sa_start: 0,
                occ: 0,
            };
        }
        // Interval of the maximal pattern P = read[pivot-left_ext .. anchor_end],
        // which occurs at text position `p_anchor - left_ext`.
        let p_start = pivot - left_ext as usize;
        let p_slice = &read[p_start..anchor_end];
        if p_slice.is_empty() {
            // Degenerate (anchor_len == 0, no left extension): empty pattern.
            return MemMatch {
                match_len,
                sa_start: 0,
                occ: 0,
            };
        }
        let p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));
        // Tight seed from the inverse SA when present (the exact SA index of P at
        // this locus); otherwise the hint itself — `find_boundary` expands on a
        // miss, so the seed only affects speed, never the boundary it returns.
        let seed = self.isa_at(p_anchor - left_ext).unwrap_or(hint);
        let lower = self.find_boundary(0, sa_num, seed, seed + 1, |mid| {
            self.ref_less(p_slice, p_key, mid, pac, enc, l_pac)
        });
        let upper = self.find_boundary(lower, sa_num, seed, seed + 1, |mid| {
            self.shares_prefix(p_slice, p_key, mid, pac, enc, l_pac)
        });
        MemMatch {
            match_len,
            sa_start: lower,
            occ: upper.saturating_sub(lower),
        }
    }

    /// PROTOTYPE — cold backward spectrum whose per-step seed window comes from
    /// the k-mer table instead of the learned-model `lookup`. Produces the SAME
    /// `SmemStep` trace as [`backward_spectrum`] — byte-identical by
    /// [`find_boundary`](Self::find_boundary)'s seed-independence (the seed only
    /// sets probe count, never the boundary it returns) — but works WITHOUT a
    /// hint, the bwa-meme reseed case (first left-extension of each anchor, no
    /// carried refpos). The question it answers: does the table's exact k-mer
    /// interval beat the model window on the un-accelerated backward direction,
    /// the way `.kmt` accelerated forward?
    ///
    /// Identical loop to [`backward_spectrum_inner_into`](Self::backward_spectrum_inner_into)
    /// except the seed source; `forward_spectrum_from_hint_equals_cold`'s backward
    /// twin (`backward_spectrum_tabled_equals_cold`) is the divergence guard.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum_tabled(
        &self,
        _sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
        table: &impl KmerBounds,
    ) -> Vec<SmemStep> {
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        if occ_count == 0 {
            return steps;
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        // Checked: a large caller-provided `anchor_len` must not wrap when
        // narrowed to usize or added to `pivot` (the `read[p_start..anchor_end]`
        // slice below would otherwise panic on an out-of-range bound). Fail
        // closed, mirroring `mem_search_backward_from_hint`.
        let anchor_len_usize = match usize::try_from(anchor_len) {
            Ok(v) => v,
            Err(_) => return steps,
        };
        let anchor_end = match pivot.checked_add(anchor_len_usize) {
            Some(v) => v,
            None => return steps,
        };
        if anchor_end > read.len() {
            return steps;
        }
        let k = table.k() as usize;
        let mut left_ext: u64 = 0;

        while (left_ext as usize) < pivot {
            let c = read[pivot - 1 - left_ext as usize];
            if c >= 4 {
                break; // ambiguous read base
            }
            let p_start = pivot - 1 - left_ext as usize;
            let p_slice = &read[p_start..anchor_end];
            let p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));

            // Seed window from the k-mer table: the EXACT SA interval of P's first
            // `min(|P|, k)` bases (a superset of P's interval — P extends it), with
            // zero SA probes. Replaces the model `lookup`'s `[pred-err, pred+err+1)`.
            let m = p_slice.len().min(k);
            let w = kmer_lex_index(p_slice, m);
            let (win_lo, win_hi) = (table.lo(m, w), table.hi(m, w));

            let lower = self.find_boundary(0, sa_num, win_lo, win_hi, |mid| {
                self.ref_less(p_slice, p_key, mid, pac, enc, l_pac)
            });
            let upper = self.find_boundary(lower, sa_num, win_hi, win_hi, |mid| {
                self.shares_prefix(p_slice, p_key, mid, pac, enc, l_pac)
            });

            if upper <= lower {
                break; // cannot extend further left
            }
            left_ext += 1;
            steps.push(SmemStep {
                sa_start: lower,
                occ_count: upper - lower,
                match_len: anchor_len + left_ext,
            });
        }
        steps
    }

    /// PROTOTYPE — backward spectrum launched from an EXACT SA-index `hint` (the
    /// anchor's inverse-SA index), producing the SAME `SmemStep` trace as
    /// [`backward_spectrum`] but via a direct leftward reference walk with each
    /// left step's interval seeded exactly from the inverse SA — the full-trace
    /// twin of [`mem_search_backward_from_hint`](Self::mem_search_backward_from_hint),
    /// which keeps only the maximal step.
    ///
    /// Mechanism: walk left while `read[pivot-1-k]` matches the reference base at
    /// `p_anchor-1-k` (no SA search for the walk itself); at each extended step
    /// recover the interval of `P_k = read[pivot-k .. pivot+anchor_len]` with two
    /// boundary searches tight-seeded from `isa_at(p_anchor-k)` (the exact SA index
    /// of `P_k` at this locus; a loose seed = `hint` without `.isa`). Emits one
    /// step per left base, exactly as cold `backward_spectrum`.
    ///
    /// Byte-identical to [`backward_spectrum`] when `hint` is at the maximal-
    /// extension locus (same contract as [`mem_search_backward_from_hint`]): the
    /// walk then extends exactly as far as cold's interval-non-empty loop, and each
    /// step recovers the same pattern's interval (`find_boundary` is seed-
    /// independent). A diverged hint yields a shorter trace the caller detects via
    /// continued-match. Verified by `backward_spectrum_from_hint_equals_cold`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum_from_hint(
        &self,
        read: &[u8],
        pivot: usize,
        anchor_len: u64,
        hint: u64,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        let sa_num = self.sa_num();
        // Checked: fail closed instead of wrapping `pivot + anchor_len` (a wrapped
        // bound would slip past the `> read.len()` guard). Mirrors
        // `mem_search_backward_from_hint`.
        let anchor_len_usize = match usize::try_from(anchor_len) {
            Ok(v) => v,
            Err(_) => return steps,
        };
        let anchor_end = match pivot.checked_add(anchor_len_usize) {
            Some(v) => v,
            None => return steps,
        };
        if hint == 0 || hint >= sa_num || anchor_end > read.len() {
            return steps;
        }
        let l_pac = self.l_pac();
        let p_anchor = self.sa_position_for(hint);

        let mut left_ext: u64 = 0;
        while (left_ext as usize) < pivot && p_anchor > left_ext {
            let read_base = read[pivot - 1 - left_ext as usize];
            if read_base >= 4 {
                break; // ambiguous read base
            }
            // Extend only while the read matches the reference at the hinted locus.
            let ref_pos = p_anchor - 1 - left_ext;
            match doubled_base_at(pac, enc, l_pac, ref_pos) {
                Some(b) if b == read_base => {}
                _ => break,
            }
            left_ext += 1;

            // Interval of P = read[pivot-left_ext .. anchor_end], at text position
            // `p_anchor - left_ext`, tight-seeded from the inverse SA.
            let p_start = pivot - left_ext as usize;
            let p_slice = &read[p_start..anchor_end];
            let p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));
            let seed = self.isa_at(p_anchor - left_ext).unwrap_or(hint);
            let lower = self.find_boundary(0, sa_num, seed, seed + 1, |mid| {
                self.ref_less(p_slice, p_key, mid, pac, enc, l_pac)
            });
            let upper = self.find_boundary(lower, sa_num, seed, seed + 1, |mid| {
                self.shares_prefix(p_slice, p_key, mid, pac, enc, l_pac)
            });
            if upper <= lower {
                break; // diverged hint: pattern absent (shorter trace, caller's gate)
            }
            steps.push(SmemStep {
                sa_start: lower,
                occ_count: upper - lower,
                match_len: anchor_len + left_ext,
            });
        }
        steps
    }

    /// Start a stepper; performs the first model lookup + Lower `FbState` init, or
    /// finishes immediately (occ==0, or no left base, or a degenerate first window
    /// that resolves with no probe).
    fn bwd_stepper_new<'a>(&self, t: &BwdTask<'a>) -> BwdStepper<'a> {
        let sa_num = self.sa_num();
        let mut s = BwdStepper {
            read: t.read,
            pivot: t.pivot,
            anchor_len: t.anchor_len,
            sa_num,
            p_start: 0,
            p_key: 0,
            win_hi: 0,
            which: Which::Lower,
            lower: 0,
            left_ext: 0,
            fb: FbState::new(0, 0, 0, 0),
            steps: Vec::with_capacity(SPECTRUM_STEPS_HINT),
            finished: false,
        };
        // Fail closed on an out-of-range anchor window, mirroring the serial
        // `backward_spectrum_inner`'s `pivot + anchor_len <= read.len()` guard —
        // otherwise a malformed lockstep task panics here while the serial path
        // returns empty, breaking the byte-identical-strategy contract.
        let valid_window = t
            .pivot
            .checked_add(t.anchor_len as usize)
            .map(|end| end <= t.read.len())
            .unwrap_or(false);
        if t.occ_count == 0 || !valid_window || !s.begin_left_step(self) {
            s.finished = true;
        }
        s
    }

    #[allow(clippy::too_many_arguments)]
    fn bwd_compare(
        &self,
        s: &BwdStepper,
        pos: u64,
        key: Option<u64>,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) -> (bool, u32) {
        // One SA probe per call, matching the serial `ref_less`/`shares_prefix`
        // so probe-count profiling of the lockstep path is consistent (no-op in
        // production builds).
        bump_probe();
        let p_slice = &s.read[s.p_start..s.pivot + s.anchor_len as usize];
        compare_query_vs_suffix_2x_keyed(p_slice, s.p_key, key, pos, pac, enc, l_pac)
    }

    /// Drive ONE stepper to completion serially. Equivalent to `backward_spectrum`;
    /// exists to prove the stepper is byte-identical before lockstepping.
    #[allow(dead_code)]
    fn backward_spectrum_via_stepper(
        &self,
        t: &BwdTask,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let l_pac = self.l_pac();
        let mut s = self.bwd_stepper_new(t);
        while let Some(mid) = s.next_probe(self) {
            let pos = self.sa_position_for(mid);
            let key = self.key_at(mid);
            s.advance(self, pos, key, pac, enc, l_pac);
        }
        s.steps
    }

    /// Drive N backward steppers in lockstep. Each round batch-loads every active
    /// task's current probe (`sa_position_for` + `key_at` — independent loads the
    /// CPU keeps in flight = MLP), then advances every task. Output[i] is byte-
    /// identical to `backward_spectrum(tasks[i]…)`.
    pub fn backward_spectrum_lockstep(
        &self,
        tasks: &[BwdTask],
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<Vec<SmemStep>> {
        let l_pac = self.l_pac();
        let mut steppers: Vec<BwdStepper> = tasks.iter().map(|t| self.bwd_stepper_new(t)).collect();
        let mut mids: Vec<Option<u64>> = steppers.iter_mut().map(|s| s.next_probe(self)).collect();
        // Per-round scratch buffers, allocated ONCE and reused (entries for
        // inactive tasks hold stale values but are never read — the advance loop
        // guards on `mids[i].is_some()`).
        let mut posv: Vec<u64> = vec![0; steppers.len()];
        let mut keyv: Vec<Option<u64>> = vec![None; steppers.len()];
        loop {
            // Phase 1: batch the cold loads (independent -> memory-level parallelism).
            let mut any = false;
            for i in 0..steppers.len() {
                if let Some(mid) = mids[i] {
                    any = true;
                    posv[i] = self.sa_position_for(mid);
                    keyv[i] = self.key_at(mid);
                }
            }
            if !any {
                break;
            }
            // Phase 2: advance every active task and fetch its next probe.
            for i in 0..steppers.len() {
                if mids[i].is_some() {
                    steppers[i].advance(self, posv[i], keyv[i], pac, enc, l_pac);
                    mids[i] = steppers[i].next_probe(self);
                }
            }
        }
        steppers.into_iter().map(|s| s.steps).collect()
    }

    /// Backward spectrum: refine the right-anchored interval `[sa_start,
    /// sa_start+occ_count)` (matching `read[pivot..pivot+anchor_len)`) leftward.
    /// Each emitted step's `match_len` is the TOTAL span (`anchor_len + left_ext`).
    /// `pac` is the FORWARD pac.
    ///
    /// Each step prepends one read base `c = read[pivot-1-left_ext]` to the current
    /// matched span and re-derives the SA interval of the LEFT-EXTENDED query
    /// `P = read[pivot-(left_ext+1) .. pivot+anchor_len)`. Prepending a base moves the
    /// interval into the `c`-block, so — unlike forward narrowing — the `c·Q` interval
    /// is NOT contained in the previous interval's SA range and cannot be bounded by it.
    /// The suffixes sharing `P` are contiguous in the SA, so the interval is fully
    /// described by its `[lower, upper)` boundaries — no member enumeration, making
    /// each step `O(log N · |P|)` regardless of occupancy.
    ///
    /// # Model-accelerated launch (window as a HINT, never a clamp)
    ///
    /// Rather than binary-searching the full `[0, sa_num)` for each boundary, the
    /// learned model seeds a small window: `key = tokenize_32mer(P, min(32, |P|))`
    /// and `(pred, err) = lookup(key)` bracket where `P`'s 32-mer prefix sorts, so the
    /// true `[lower, upper)` lies near `[pred - err, pred + err]`. We binary-search
    /// WITHIN that window, then **expand on miss**: if a boundary search converges at
    /// the window's edge (and that edge is not already 0 / `sa_num`), the true boundary
    /// may lie outside the window, so the range is exponentially galloped outward and
    /// re-searched until the boundary is strictly interior or the SA end is reached.
    /// The window is therefore only a starting hint — a wrong `pred` or `err = 0` still
    /// yields the TRUE interval via expansion (asserted oracle-identical by proptest,
    /// including a deliberately-wrong-seed / `err = 0` test). On a real chromosome this
    /// cuts the cold SA probes per left step from `~log2(sa_num)` to `~log2(2·err)`.
    ///
    /// The `sa_start`/`occ_count` inputs are only consulted for the empty/initial
    /// early-out; the interval is re-derived from `read`/`pivot`/`anchor_len`. The ISA
    /// sidecar is not needed.
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
        self.backward_spectrum_inner(sa_start, occ_count, anchor_len, read, pivot, pac, enc, None)
    }

    /// Fill `out` with the backward spectrum's left-extension steps and return
    /// the TOTAL number of steps (overflow semantics as
    /// [`forward_spectrum_auto_fill`](Self::forward_spectrum_auto_fill)). Lets the
    /// FFI write straight into its output buffer with no intermediate `Vec`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum_fill(
        &self,
        sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
        out: &mut [SmemStep],
    ) -> usize {
        let mut sink = SliceStepSink::new(out);
        self.backward_spectrum_inner_into(
            sa_start, occ_count, anchor_len, read, pivot, pac, enc, None, &mut sink,
        );
        sink.count()
    }

    /// Shared driver for [`backward_spectrum`] and its test-only reference. The
    /// `seed_override` hook (test-only) forces a `(pred, err)` window in place of
    /// the model's `lookup`, so the proptest can prove the expand-on-miss recovery
    /// holds for a deliberately-wrong seed and for `err = 0`. In production
    /// `seed_override` is always `None` and the real model launch is used.
    #[allow(clippy::too_many_arguments)]
    fn backward_spectrum_inner(
        &self,
        sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
        seed_override: Option<fn(u64) -> (u64, u64)>,
    ) -> Vec<SmemStep> {
        let mut steps: Vec<SmemStep> = Vec::with_capacity(SPECTRUM_STEPS_HINT);
        self.backward_spectrum_inner_into(
            sa_start,
            occ_count,
            anchor_len,
            read,
            pivot,
            pac,
            enc,
            seed_override,
            &mut steps,
        );
        steps
    }

    /// Locate the SA interval `[lower, upper)` of one left-extended backward
    /// pattern `p_slice` (the matched span with one more base prepended), via two
    /// boundary searches. Returns `None` when the interval is empty (the
    /// extension cannot continue). Drives the per-base backward TRACE
    /// ([`backward_spectrum_inner_into`]); the one-shot maximal
    /// ([`mem_search_backward`]) runs that same per-base loop with a last-step
    /// sink and keeps only its deepest step.
    ///
    /// The 32-mer key is recomputed per call: a base is prepended each step, so
    /// the first 32 bases change; the keyed compare masks it to `p_slice.len()`.
    /// [`find_boundary`](Self::find_boundary) is seed-independent — the seed sets
    /// only the probe count, never the interval returned — so each seed is chosen
    /// for the fewest probes: the LOWER search uses the model window
    /// `[pred - err, pred + err + 1)` (binary search ~log2(2·err)); the UPPER
    /// search is unit-seeded at `lower` and gallops right (~2·log2(occ) probes).
    /// `err` therefore sizes only the lower seed, not the returned interval.
    #[allow(clippy::too_many_arguments)]
    fn backward_locate_step(
        &self,
        p_slice: &[u8],
        sa_num: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
        seed_override: Option<fn(u64) -> (u64, u64)>,
    ) -> Option<(u64, u64)> {
        let p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));
        let (pred, err) = match seed_override {
            Some(f) => f(p_key),
            None => self.lookup(p_key),
        };
        // Lower bound: first SA index whose suffix is >= P (where `ref_less` turns
        // false). Seed with the model error window `[pred - err, pred + err + 1)`.
        // The one-shot fast path now peels off unique/collapsing anchors, so this
        // loop runs only while the interval is genuinely WIDE — where the error
        // window brackets the boundary directly (binary search ~log2(2·err))
        // and a unit gallop from `pred` would instead step out across the same
        // distance. Upper bound: first index NOT sharing the full P prefix,
        // unit-seeded at `lower` and galloped right (~2·log2(occ) probes, beating a
        // gallop back from the model's loose right edge).
        let win_lo = pred.saturating_sub(err);
        let win_hi = pred.saturating_add(err).saturating_add(1).min(sa_num);
        let lower = self.find_boundary(0, sa_num, win_lo, win_hi, |mid| {
            self.ref_less(p_slice, p_key, mid, pac, enc, l_pac)
        });
        let upper = self.find_boundary(lower, sa_num, lower, lower, |mid| {
            self.shares_prefix(p_slice, p_key, mid, pac, enc, l_pac)
        });
        if upper <= lower {
            None
        } else {
            Some((lower, upper))
        }
    }

    /// Generic core of [`backward_spectrum_inner`]: emits each left-extension step
    /// into `sink` (no coalescing — every left step is a distinct step). Drives the
    /// full backward TRACE. The one-shot maximal
    /// ([`mem_search_backward`](Self::mem_search_backward)) reuses this same
    /// per-base loop with a last-step sink, keeping only the deepest step.
    #[allow(clippy::too_many_arguments)]
    fn backward_spectrum_inner_into<S: StepSink + ?Sized>(
        &self,
        _sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
        seed_override: Option<fn(u64) -> (u64, u64)>,
        sink: &mut S,
    ) {
        if occ_count == 0 {
            return;
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        let anchor_len_usize = anchor_len as usize;
        // Fail closed on out-of-range inputs rather than panicking inside the
        // walk: the left-extension reads `read[pivot-1-..]` and the pattern slice
        // spans `read[.. pivot + anchor_len)`, so the whole anchored window must
        // lie within `read`. A packed pac must also hold its declared base count
        // before any `pac_base_at` read.
        match pivot.checked_add(anchor_len_usize) {
            Some(end) if end <= read.len() => {}
            _ => return,
        }
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "backward_spectrum").is_err() {
                return;
            }
        }
        let mut left_ext: u64 = 0;

        while (left_ext as usize) < pivot {
            let c = read[pivot - 1 - left_ext as usize];
            if c >= 4 {
                break; // ambiguous read base
            }
            // Left-extended query P: prepend the new base to the matched span.
            // Start index is `pivot - (left_ext + 1)` (>= 0 by the loop condition),
            // end is `pivot + anchor_len` (exclusive); length = anchor_len+left_ext+1.
            let p_slice = &read[pivot - 1 - left_ext as usize..pivot + anchor_len_usize];
            // A structurally contiguous interval (`upper > lower`) or `None` when
            // the extension cannot continue — `Some` guarantees a non-empty step.
            let (lower, upper) =
                match self.backward_locate_step(p_slice, sa_num, pac, enc, l_pac, seed_override) {
                    Some(interval) => interval,
                    None => break, // cannot extend further left
                };
            left_ext += 1;
            sink.push_new(SmemStep {
                sa_start: lower,
                occ_count: upper - lower,
                match_len: anchor_len + left_ext,
            });
        }
    }

    /// Prefetch the two SA entries a binary search over `[lo, hi)` would probe
    /// next (the children of the current midpoint), so the next cold DRAM read is
    /// already in flight while the current keyed compare runs. Advisory — it
    /// never changes a search result, only its latency; out-of-range indices are
    /// ignored by `prefetch_sa`. Called at the top of every SA binary-search loop.
    #[inline(always)]
    fn prefetch_bsearch(&self, lo: u64, hi: u64) {
        if lo >= hi {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        self.prefetch_sa(lo + (mid - lo) / 2); // next mid if we branch toward lo
        self.prefetch_sa(mid + 1 + (hi - mid - 1) / 2); // next mid if we branch toward hi
    }

    /// Find the boundary index in `[domain_lo, domain_hi)` — the first index `i`
    /// where the monotone predicate `go_right(i)` is `false` — seeded from the model
    /// window `[seed_lo, seed_hi)` with exponential expand-on-miss.
    ///
    /// `go_right` MUST be monotone over `[domain_lo, domain_hi)`: `true` on a (possibly
    /// empty) prefix `[domain_lo, boundary)` and `false` on `[boundary, domain_hi)`.
    /// The returned value equals what a plain binary search over the full
    /// `[domain_lo, domain_hi)` would return; the seed window only changes the number
    /// of probes, never the answer (the gallop always re-brackets the true boundary).
    ///
    /// Bracketing invariant on entry to the final binary search: `go_right(lo-1)` is
    /// `true` (or `lo == domain_lo`) and `go_right(hi-1)` is `false` (or `hi ==
    /// domain_hi`), so `[lo, hi)` straddles the boundary.
    #[inline]
    fn find_boundary(
        &self,
        domain_lo: u64,
        domain_hi: u64,
        seed_lo: u64,
        seed_hi: u64,
        mut go_right: impl FnMut(u64) -> bool,
    ) -> u64 {
        // Clamp the seed window into the domain; an empty/degenerate window collapses
        // to a single point inside the domain so the gallop can still expand from it.
        let mut lo = seed_lo.clamp(domain_lo, domain_hi);
        let mut hi = seed_hi.clamp(lo, domain_hi);
        if lo == hi {
            // Degenerate window: nudge to a unit interval inside the domain so both
            // edge probes below are well-defined (hi-1 == lo).
            if hi < domain_hi {
                hi += 1;
            } else if lo > domain_lo {
                lo -= 1;
            } else {
                return domain_lo; // domain is empty
            }
        }
        let mut span = (hi - lo).max(1);
        // Each `go_right` is a cold SA probe. The left- and right-edge probes
        // (`go_right(lo-1)` / `go_right(hi-1)`) only need re-evaluating when that
        // edge actually moves, so cache each result and invalidate the entry for
        // the edge that galloped. `go_right` is a deterministic function of the
        // index, so a cached value equals what re-probing would return — same
        // bracket, fewer probes.
        let mut left_edge: Option<bool> = None; // cached go_right(lo - 1)
        let mut right_edge: Option<bool> = None; // cached go_right(hi - 1)
        loop {
            // Left edge too far right: `go_right(lo-1)` is false => boundary is at or
            // left of `lo`. Gallop left so the bracket includes it.
            if lo > domain_lo {
                let at_or_right = *left_edge.get_or_insert_with(|| go_right(lo - 1));
                if !at_or_right {
                    lo = lo.saturating_sub(span).max(domain_lo);
                    span = span.saturating_mul(2);
                    left_edge = None; // `lo` moved; the cached probe is stale.
                    continue;
                }
            }
            // Right edge too far left: `go_right(hi-1)` is true => boundary is at or
            // right of `hi`. Gallop right.
            if hi < domain_hi {
                let at_or_left = *right_edge.get_or_insert_with(|| go_right(hi - 1));
                if at_or_left {
                    hi = hi.saturating_add(span).min(domain_hi);
                    span = span.saturating_mul(2);
                    right_edge = None; // `hi` moved; the cached probe is stale.
                    continue;
                }
            }
            break;
        }
        // `[lo, hi)` now straddles the boundary; standard binary search. Every
        // `go_right(mid)` probes SA entry `mid` (a cold DRAM read); prefetch the
        // two possible next-probe entries so that read is already in flight while
        // this iteration's keyed compare runs. Advisory only — it never changes
        // the boundary returned, just the probe latency. (Every caller's
        // predicate probes SA entry `mid`, so the prefetch targets are SA
        // indices; out-of-range indices are ignored by `prefetch_sa`.)
        while lo < hi {
            self.prefetch_bsearch(lo, hi);
            let mid = lo + (hi - lo) / 2;
            if go_right(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// One lower-bound probe: is the suffix at SA index `mid` lexicographically less
    /// than `p_slice`? Reads `sa_position_for(mid)` (the cold DRAM hit) and the stored
    /// key, then runs the keyed 2× compare.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn ref_less(
        &self,
        p_slice: &[u8],
        p_key: u64,
        mid: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) -> bool {
        bump_probe();
        let pos = self.sa_position_for(mid);
        let key = self.key_at(mid);
        let (ref_less, _) =
            compare_query_vs_suffix_2x_keyed(p_slice, p_key, key, pos, pac, enc, l_pac);
        ref_less
    }

    /// One upper-bound probe: does the suffix at SA index `mid` share the full
    /// `|p_slice|`-length prefix with `p_slice` (lcp >= |P|)?
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn shares_prefix(
        &self,
        p_slice: &[u8],
        p_key: u64,
        mid: u64,
        pac: &[u8],
        enc: PacEncoding,
        l_pac: u64,
    ) -> bool {
        bump_probe();
        let pos = self.sa_position_for(mid);
        let key = self.key_at(mid);
        let (_, lcp) = compare_query_vs_suffix_2x_keyed(p_slice, p_key, key, pos, pac, enc, l_pac);
        (lcp as usize) >= p_slice.len()
    }

    /// Test-only: drive [`backward_spectrum`] with a forced `(pred, err)` window in
    /// place of the model's `lookup`, so the equality proptest can prove the
    /// expand-on-miss recovery holds for a deliberately-wrong seed and for `err = 0`.
    /// Hidden from the public API; production callers use [`backward_spectrum`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum_with_seed(
        &self,
        sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
        seed: fn(u64) -> (u64, u64),
    ) -> Vec<SmemStep> {
        self.backward_spectrum_inner(
            sa_start,
            occ_count,
            anchor_len,
            read,
            pivot,
            pac,
            enc,
            Some(seed),
        )
    }

    /// Test-only reference: the model-free backward spectrum that binary-searches the
    /// FULL SA `[0, sa_num)` for both interval boundaries on every left step. This is
    /// the pre-model-launch implementation, retained as the source of truth the
    /// equality proptest pins the model-launched [`backward_spectrum`] against. Hidden
    /// from the public API; not used on the production hot path.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn backward_spectrum_reference(
        &self,
        _sa_start: u64,
        occ_count: u64,
        anchor_len: u64,
        read: &[u8],
        pivot: usize,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let mut steps = Vec::new();
        if occ_count == 0 {
            return steps;
        }
        let sa_num = self.sa_num();
        let l_pac = self.l_pac();
        let anchor_len_usize = anchor_len as usize;
        // Fail closed on out-of-range inputs rather than panicking inside the
        // walk: the left-extension reads `read[pivot-1-..]` and the pattern slice
        // spans `read[.. pivot + anchor_len)`, so the whole anchored window must
        // lie within `read`. A packed pac must also hold its declared base count
        // before any `pac_base_at` read.
        match pivot.checked_add(anchor_len_usize) {
            Some(end) if end <= read.len() => {}
            _ => return steps,
        }
        if let PacEncoding::Packed { num_bases } = enc {
            if validate_packed_pac(pac, num_bases, "backward_spectrum").is_err() {
                return steps;
            }
        }
        let mut left_ext: u64 = 0;

        while (left_ext as usize) < pivot {
            let c = read[pivot - 1 - left_ext as usize];
            if c >= 4 {
                break;
            }
            let p_start = pivot - 1 - left_ext as usize;
            let p_end = pivot + anchor_len_usize;
            let p_slice = &read[p_start..p_end];
            let p_key = tokenize_32mer(p_slice, p_slice.len().min(KMER_LEN));

            // Lower bound over the FULL [0, sa_num).
            let mut lo = 0u64;
            let mut hi = sa_num;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.ref_less(p_slice, p_key, mid, pac, enc, l_pac) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let lower = lo;
            // Upper bound over [lower, sa_num).
            let mut c_lo = lower;
            let mut c_hi = sa_num;
            while c_lo < c_hi {
                let mid = c_lo + (c_hi - c_lo) / 2;
                if self.shares_prefix(p_slice, p_key, mid, pac, enc, l_pac) {
                    c_lo = mid + 1;
                } else {
                    c_hi = mid;
                }
            }
            let upper = c_lo;

            if upper <= lower {
                break;
            }
            left_ext += 1;
            steps.push(SmemStep {
                sa_start: lower,
                occ_count: upper - lower,
                match_len: anchor_len + left_ext,
            });
        }
        steps
    }
}

#[cfg(test)]
mod compare_tests {
    use super::*;
    use proptest::prelude::*;

    fn pac() -> (Vec<u8>, PacEncoding, u64) {
        // forward ACGTAC = 0,1,2,3,0,1 ; l_pac = 6.
        (vec![0, 1, 2, 3, 0, 1], PacEncoding::Unpacked, 6)
    }

    /// Pack an unpacked base vector (each `0..=3`) into the BWA `bntpac`
    /// MSB-first 2-bit form (4 bases/byte), returning `(packed, num_bases)`.
    fn pack_bntpac(bases: &[u8]) -> (Vec<u8>, u64) {
        let mut packed = vec![0u8; bases.len().div_ceil(4)];
        for (i, &b) in bases.iter().enumerate() {
            let shift = 6 - 2 * ((i % 4) as u32);
            packed[i / 4] |= (b & 0x3) << shift;
        }
        (packed, bases.len() as u64)
    }

    /// Materialize the doubled `[Fwd || RC]` text as one base per byte, for
    /// reading the reference suffix directly in a test oracle.
    fn doubled_text(fwd: &[u8]) -> Vec<u8> {
        let mut t = fwd.to_vec();
        t.extend(fwd.iter().rev().map(|&b| b ^ 3));
        t
    }

    /// Build a query of length `qlen` from the doubled text starting at
    /// `sa_pos`, then mutate it so we exercise match / mismatch / overrun cases.
    /// Returns the query bases.
    fn query_from(
        text: &[u8],
        sa_pos: usize,
        qlen: usize,
        twist_at: Option<(usize, u8)>,
    ) -> Vec<u8> {
        let mut q: Vec<u8> = (0..qlen)
            .map(|i| text.get(sa_pos + i).copied().unwrap_or(0))
            .collect();
        if let Some((pos, val)) = twist_at {
            if pos < q.len() {
                q[pos] = val % 4;
            }
        }
        q
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

    // ── Region / boundary unit tests for the vectorized compare ──────────────

    /// Reference long enough to exercise multi-word (>= 16 base) matches, a
    /// forward→RC mid-word crossing, and the sentinel mid-word. l_pac = 20.
    fn long_ref() -> Vec<u8> {
        // 20 forward bases (values cycle 0..=3), so the doubled text is 40 bases.
        (0..20u8).map(|i| i % 4).collect()
    }

    /// Run the assertion against both encodings: vectorized == scalar, and (for
    /// extra confidence) report the agreed value.
    fn check_both_encodings(fwd: &[u8], query: &[u8], sa_pos: u64) -> (bool, u32) {
        let l_pac = fwd.len() as u64;
        let (vec_u, scal_u) = {
            let e = PacEncoding::Unpacked;
            (
                compare_query_vs_suffix_2x(query, sa_pos, fwd, e, l_pac),
                compare_query_vs_suffix_2x_scalar(query, sa_pos, fwd, e, l_pac),
            )
        };
        assert_eq!(vec_u, scal_u, "Unpacked: vec != scalar (sa_pos={sa_pos})");
        let (packed, num_bases) = pack_bntpac(fwd);
        let e = PacEncoding::Packed { num_bases };
        let vec_p = compare_query_vs_suffix_2x(query, sa_pos, &packed, e, l_pac);
        let scal_p = compare_query_vs_suffix_2x_scalar(query, sa_pos, &packed, e, l_pac);
        assert_eq!(vec_p, scal_p, "Packed: vec != scalar (sa_pos={sa_pos})");
        assert_eq!(
            vec_u, vec_p,
            "Unpacked vs Packed disagree (sa_pos={sa_pos})"
        );
        vec_u
    }

    #[test]
    fn forward_exact_match_full_query() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        // Query = first 17 bases of the forward text -> full query matched.
        let q = query_from(&text, 0, 17, None);
        let (ref_less, lcp) = check_both_encodings(&fwd, &q, 0);
        assert_eq!((ref_less, lcp), (false, 17));
    }

    #[test]
    fn forward_mismatch_at_each_byte_position() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        // Force a mismatch at byte positions 0..8 within the first word.
        for pos in 0..8usize {
            let twisted = (text[pos] + 1) % 4; // guaranteed != text[pos]
            let q = query_from(&text, 0, 12, Some((pos, twisted)));
            let (_, lcp) = check_both_encodings(&fwd, &q, 0);
            assert_eq!(lcp, pos as u32, "mismatch should be located at byte {pos}");
        }
    }

    #[test]
    fn forward_multiword_match_spanning_16_plus() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        // 20-base query matching the whole forward half (spans >2 words),
        // mismatch only at base 19 vs the RC region's first base afterwards.
        let q = query_from(&text, 0, 20, None);
        let (ref_less, lcp) = check_both_encodings(&fwd, &q, 0);
        assert_eq!((ref_less, lcp), (false, 20));
    }

    #[test]
    fn rc_region_suffix() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        let l_pac = fwd.len() as u64;
        // sa_pos in the RC half; match several bases then mismatch.
        let sa_pos = l_pac + 2; // RC region
        let q = query_from(&text, sa_pos as usize, 10, Some((4, 9)));
        check_both_encodings(&fwd, &q, sa_pos);
    }

    #[test]
    fn window_crosses_l_pac_mid_word() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        let l_pac = fwd.len() as u64;
        // Start a few bases before l_pac so the window straddles forward→RC.
        let sa_pos = l_pac - 3;
        let q = query_from(&text, sa_pos as usize, 12, None);
        let (ref_less, lcp) = check_both_encodings(&fwd, &q, sa_pos);
        // 3 forward + 9 RC bases all match the materialized text.
        assert_eq!((ref_less, lcp), (false, 12));
    }

    #[test]
    fn suffix_reaches_sentinel_mid_word() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        let l_pac = fwd.len() as u64;
        // Start 5 bases before the end of the doubled text; query is longer,
        // so the reference is exhausted mid-query -> ref_less = true.
        let sa_pos = 2 * l_pac - 5;
        let q = query_from(&text, sa_pos as usize, 12, None);
        let (ref_less, lcp) = check_both_encodings(&fwd, &q, sa_pos);
        assert_eq!((ref_less, lcp), (true, 5));
    }

    #[test]
    fn query_longer_than_whole_text() {
        let fwd = long_ref();
        let text = doubled_text(&fwd);
        // Query = entire doubled text + extra base -> ref exhausted at the end.
        let mut q = text.clone();
        q.push(0);
        let (ref_less, lcp) = check_both_encodings(&fwd, &q, 0);
        assert_eq!((ref_less, lcp), (true, text.len() as u32));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// Primary correctness gate: the vectorized compare must be byte-identical
        /// to the scalar reference for both encodings, over random references,
        /// random `sa_pos` spanning forward/RC/sentinel, and random queries.
        #[test]
        fn vectorized_equals_scalar(
            fwd in prop::collection::vec(0u8..=3, 1..400),
            // sa_pos spans [0, 2*l_pac] inclusive (the sentinel boundary).
            sa_pos_frac in 0.0f64..=1.0,
            query in prop::collection::vec(0u8..=3, 0..64),
        ) {
            let l_pac = fwd.len() as u64;
            let sa_pos = ((2 * l_pac) as f64 * sa_pos_frac).round() as u64;

            // Unpacked.
            let e = PacEncoding::Unpacked;
            let got = compare_query_vs_suffix_2x(&query, sa_pos, &fwd, e, l_pac);
            let want = compare_query_vs_suffix_2x_scalar(&query, sa_pos, &fwd, e, l_pac);
            prop_assert_eq!(got, want, "Unpacked mismatch: sa_pos={}", sa_pos);

            // Packed (bntpac).
            let (packed, num_bases) = pack_bntpac(&fwd);
            let e = PacEncoding::Packed { num_bases };
            let got = compare_query_vs_suffix_2x(&query, sa_pos, &packed, e, l_pac);
            let want = compare_query_vs_suffix_2x_scalar(&query, sa_pos, &packed, e, l_pac);
            prop_assert_eq!(got, want, "Packed mismatch: sa_pos={}", sa_pos);
        }
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

/// Correctness gate for the stored-key fast path: drives a real mode-2 sidecar
/// (so `key_at` is populated) and asserts the key-aware compare is byte-identical
/// to the scalar reference for random `mid` SA indices — INCLUDING indices whose
/// `sa_position_for(mid)` lands within 32 of `2*l_pac` (the near-sentinel
/// fallback). Lives in its own module because it pulls in the trainer + temp-file
/// machinery the lighter compare tests do not need.
#[cfg(test)]
mod keyed_tests {
    use super::*;
    use crate::index::LearnedIndex;
    use crate::train::config::{MemoryMode, TrainerConfig};
    use crate::train::{build_sidecar_from_pac_with_config, mask::MaskConfig};
    use proptest::prelude::*;
    use std::io::Write;

    /// Write an unpacked base vector (each `0..=3`) as a bwa-format `.pac`
    /// (MSB-first, 4 bases/byte, trailing `l_pac % 4` byte) the trainer reads.
    fn write_pac(path: &std::path::Path, bases: &[u8]) {
        let l = bases.len();
        let mut buf = vec![0u8; l / 4 + 1];
        for (i, &b) in bases.iter().enumerate() {
            buf[i >> 2] |= b << ((3 - (i & 3)) * 2);
        }
        buf.push((l % 4) as u8);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    /// Build a mode-2 sidecar from `fwd` and return the opened index + tempdir
    /// (kept alive for the mmap lifetime).
    fn build_mode2(fwd: &[u8]) -> (tempfile::TempDir, LearnedIndex) {
        let dir = tempfile::tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, fwd);
        let prefix = dir.path().join("r.prmi");
        let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
        build_sidecar_from_pac_with_config(
            &pac,
            &prefix,
            None,
            MaskConfig::default(),
            1,
            Some(cfg),
        )
        .unwrap();
        let idx = LearnedIndex::open(&prefix).unwrap();
        (dir, idx)
    }

    /// Independent brute-force forward spectrum: for each prefix length `m`,
    /// LINEAR-SCAN every SA entry and take the contiguous run whose suffix has
    /// `query[..m]` as a prefix (scalar compare = ground truth), emitting a step
    /// on each `occ` change. Shares no code with the binary-search / fast-path
    /// logic under test, so it is a true oracle. `O(sa_num * query_len)` — small
    /// references only. Relies on SA sortedness (matching suffixes are
    /// contiguous), which is verified independently by `sa-verify`.
    fn forward_spectrum_oracle(
        idx: &LearnedIndex,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<SmemStep> {
        let sa_num = idx.sa_num();
        let l_pac = idx.l_pac();
        let mut steps = Vec::new();
        let mut prev_occ = u64::MAX;
        for m in 1..=query.len() {
            let qm = &query[..m];
            let mut lo = sa_num;
            let mut hi = 0u64;
            let mut found = false;
            #[allow(clippy::needless_range_loop)]
            for i in 0..sa_num {
                let pos = idx.sa_position_for(i);
                let (_, lcp) = compare_query_vs_suffix_2x_scalar(qm, pos, pac, enc, l_pac);
                if lcp as usize >= m {
                    if !found {
                        lo = i;
                        found = true;
                    }
                    hi = i + 1;
                }
            }
            if !found {
                break;
            }
            push_step(&mut steps, &mut prev_occ, lo, hi - lo, m as u64);
        }
        steps
    }

    #[test]
    fn mode2_sidecar_populates_keys() {
        // Sanity: a mode-2 build must expose stored keys via key_at.
        let fwd: Vec<u8> = (0..80u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let (_dir, idx) = build_mode2(&fwd);
        assert_eq!(idx.memory_mode(), "2");
        assert!(idx.key_at(0).is_some(), "mode-2 sidecar must store keys");
    }

    /// An undersized packed pac must fail closed (zero match) in both forward
    /// hint paths, exactly as the non-hint paths do — never reach the compare
    /// helpers, where a missing byte would mis-decode or panic.
    #[test]
    fn mem_search_from_hint_rejects_undersized_packed_pac() {
        let fwd: Vec<u8> = (0..80u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let (_dir, idx) = build_mode2(&fwd);
        let zero = MemMatch {
            match_len: 0,
            sa_start: 0,
            occ: 0,
        };
        // `num_bases` claims 4096 bases (needs ≥1024 packed bytes); `tiny_pac`
        // holds 4. Validation runs before the hint check, so any in-range hint
        // still fails closed.
        let enc = PacEncoding::Packed { num_bases: 4096 };
        let tiny_pac = [0u8; 4];
        let q = [0u8, 1, 2, 3];
        assert_eq!(idx.mem_search_from_hint(&q, 1, true, &tiny_pac, enc), zero);
        assert_eq!(idx.mem_search_from_hint(&q, 1, false, &tiny_pac, enc), zero);
    }

    /// A pathological `anchor_len` must not overflow `pivot + anchor_len`; the
    /// backward hint path fails closed instead of panicking on the cast/add.
    #[test]
    fn mem_search_backward_from_hint_rejects_overflowing_anchor_len() {
        let fwd: Vec<u8> = (0..120u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let (_dir, idx) = build_mode2(&fwd);
        let zero = MemMatch {
            match_len: 0,
            sa_start: 0,
            occ: 0,
        };
        let read = &fwd[..50];
        let got = idx.mem_search_backward_from_hint(
            read,
            24,
            u64::MAX,
            1,
            true,
            &fwd,
            PacEncoding::Unpacked,
        );
        assert_eq!(got, zero);
    }

    /// The k-mer table built from the in-memory unpacked `bases` must be
    /// byte-identical to the table built from the packed `.pac` form of the
    /// same reference. This is the invariant that lets `build_sidecar_core`
    /// build the table against the in-memory `bases` (which it already holds)
    /// instead of re-reading the entire `.pac` from disk.
    #[test]
    fn kmer_table_packed_pac_equals_unpacked_bases() {
        let fwd: Vec<u8> = (0..200u32).map(|i| ((i * 7 + 3) % 4) as u8).collect();
        let (_dir, idx) = build_mode2(&fwd);
        let k = 6;
        let unpacked = idx.build_kmer_table(k, &fwd, PacEncoding::Unpacked);
        // Pack `fwd` into the BWA bntpac MSB-first 2-bit form (4 bases/byte).
        let mut packed = vec![0u8; fwd.len().div_ceil(4)];
        for (i, &b) in fwd.iter().enumerate() {
            packed[i / 4] |= (b & 0x3) << (6 - 2 * ((i % 4) as u32));
        }
        let num_bases = fwd.len() as u64;
        let packed_tbl = idx.build_kmer_table(k, &packed, PacEncoding::Packed { num_bases });
        // Guard against a vacuous pass: at least one band must hold a non-empty
        // interval, so the equality below compares real bounds.
        let (_, lo, hi) = unpacked.parts();
        let any_non_empty = lo
            .iter()
            .zip(hi)
            .any(|(lm, hm)| lm.iter().zip(hm).any(|(l, h)| h > l));
        assert!(any_non_empty, "expected at least one non-empty k-mer band");
        assert_eq!(
            unpacked.parts(),
            packed_tbl.parts(),
            "packed-pac and unpacked-bases k-mer tables must be byte-identical"
        );
    }

    proptest! {
        // A sidecar build is costly, so keep the case count modest; each case
        // exercises many random `mid` indices against the same sidecar.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Key-aware compare == scalar reference over a real mode-2 sidecar, for
        /// random queries and random SA indices (which include positions within
        /// 32 of the doubled-text end -> the near-sentinel fallback path).
        #[test]
        fn keyed_equals_scalar_mode2(
            fwd in prop::collection::vec(0u8..=3, 40..220),
            query in prop::collection::vec(0u8..=3, 0..64),
            mid_fracs in prop::collection::vec(0.0f64..=1.0, 24),
        ) {
            let l_pac = fwd.len() as u64;
            let (_dir, idx) = build_mode2(&fwd);
            let sa_num = idx.sa_num();
            prop_assert_eq!(idx.l_pac(), l_pac);

            let e = PacEncoding::Unpacked;
            let query_key = tokenize_32mer(&query, query.len().min(KMER_LEN));

            for frac in mid_fracs {
                // mid in [0, sa_num); bias the sampling so near-sentinel SA
                // positions are hit (we additionally scan deterministically below).
                let mid = ((sa_num.saturating_sub(1)) as f64 * frac).round() as u64;
                let pos = idx.sa_position_for(mid);
                let key = idx.key_at(mid);

                let got = compare_query_vs_suffix_2x_keyed(
                    &query, query_key, key, pos, &fwd, e, l_pac,
                );
                let want = compare_query_vs_suffix_2x_scalar(&query, pos, &fwd, e, l_pac);
                prop_assert_eq!(
                    got, want,
                    "keyed != scalar: mid={} pos={} (l_pac={}, 2*l_pac={})",
                    mid, pos, l_pac, 2 * l_pac
                );
            }

            // Deterministically sweep EVERY SA index to guarantee the
            // near-sentinel positions (pos + 32 > 2*l_pac) are covered — these are
            // exactly the entries where the stored (T-padded) key must NOT be used.
            for mid in 0..sa_num {
                let pos = idx.sa_position_for(mid);
                let key = idx.key_at(mid);
                let got = compare_query_vs_suffix_2x_keyed(
                    &query, query_key, key, pos, &fwd, e, l_pac,
                );
                let want = compare_query_vs_suffix_2x_scalar(&query, pos, &fwd, e, l_pac);
                prop_assert_eq!(
                    got, want,
                    "keyed != scalar (full sweep): mid={} pos={} near_sentinel={}",
                    mid, pos, pos + KMER_LEN as u64 > 2 * l_pac
                );
            }
        }

        /// Table-accelerated `forward_spectrum_tabled` == reference
        /// `forward_spectrum`, for random small references (so short text-end
        /// suffixes are a large fraction of the SA) and a small `k`. This is the
        /// hardest stress for the `m < k` short-suffix boundary edge.
        #[test]
        fn forward_tabled_equals_reference(
            fwd in prop::collection::vec(0u8..=3, 40..220),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..40), 1..6),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let table = idx.build_kmer_table(6, &fwd, e);
            for q in &queries {
                prop_assert_eq!(
                    idx.forward_spectrum_tabled(q, &fwd, e, &table),
                    idx.forward_spectrum(q, &fwd, e),
                    "tabled != reference, query={:?}",
                    q
                );
            }
        }

    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// `forward_spectrum` (no table) == the independent brute-force oracle,
        /// for random references and random queries. This is the correctness
        /// anchor the fast path must preserve.
        #[test]
        fn forward_spectrum_equals_oracle(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..50), 1..5),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            for q in &queries {
                prop_assert_eq!(
                    idx.forward_spectrum(q, &fwd, e),
                    forward_spectrum_oracle(&idx, q, &fwd, e),
                    "forward_spectrum != oracle, query={:?}", q
                );
            }
        }

        /// `forward_spectrum_tabled` == the independent oracle (k=5 table).
        #[test]
        fn forward_tabled_equals_oracle(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..50), 1..5),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let table = idx.build_kmer_table(5, &fwd, e);
            for q in &queries {
                prop_assert_eq!(
                    idx.forward_spectrum_tabled(q, &fwd, e, &table),
                    forward_spectrum_oracle(&idx, q, &fwd, e),
                    "forward_spectrum_tabled != oracle, query={:?}", q
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// Same as above but the table is written to a `.kmt` and read back via
        /// the mmap `KmtFileReader` — covers the production (file-backed)
        /// `KmerBounds` impl through `forward_spectrum_tabled`.
        #[test]
        fn forward_tabled_via_kmt_file_equals_reference(
            fwd in prop::collection::vec(0u8..=3, 40..220),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..40), 1..6),
        ) {
            use crate::sidecar::kmt_file::{KmtFileReader, KmtFileWriter};
            let (dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let table = idx.build_kmer_table(6, &fwd, e);
            let (k, lo, hi) = table.parts();
            let kpath = dir.path().join("t.kmt");
            KmtFileWriter::write(&kpath, k, idx.sa_num(), &[0u8; 32], lo, hi).unwrap();
            let reader = KmtFileReader::open(&kpath).unwrap();
            for q in &queries {
                prop_assert_eq!(
                    idx.forward_spectrum_tabled(q, &fwd, e, &reader),
                    idx.forward_spectrum(q, &fwd, e),
                    "tabled-via-file != reference, query={:?}",
                    q
                );
            }
        }
    }

    /// End-to-end: a sidecar built with `--kmer-table-k` loads the `.kmt` on
    /// open and `forward_spectrum_auto` dispatches through it byte-identically.
    #[test]
    fn build_with_kmt_dispatches_byte_identically() {
        let fwd: Vec<u8> = (0..200u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let dir = tempfile::tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, &fwd);
        let prefix = dir.path().join("r.prmi");
        let cfg = TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_kmer_table_k(6);
        build_sidecar_from_pac_with_config(
            &pac,
            &prefix,
            None,
            MaskConfig::default(),
            1,
            Some(cfg),
        )
        .unwrap();
        let idx = LearnedIndex::open(&prefix).unwrap();
        assert!(idx.kmt.is_some(), ".kmt should be loaded on open");
        let e = PacEncoding::Unpacked;
        for q in [
            vec![1u8, 2, 3, 0, 1, 2, 3],
            vec![0u8, 0, 1],
            vec![3u8, 3, 3, 3, 2, 1, 0, 2],
        ] {
            assert_eq!(
                idx.forward_spectrum_auto(&q, &fwd, e),
                idx.forward_spectrum(&q, &fwd, e),
                "auto != reference for {q:?}"
            );
        }
    }

    /// With the occ==1 fast path, a deep UNIQUE forward match costs ONE SA probe
    /// in the deep bands (m>k), not ~2 per depth. Pre-fast-path baseline for this
    /// input is ~148 deep probes; this asserts the collapse. Gated on the probe
    /// counter feature (run with `--features spectrum-probe-count`).
    #[cfg(feature = "spectrum-probe-count")]
    #[test]
    fn occ1_fast_path_collapses_deep_probes() {
        use crate::index::spectrum::probe_count;
        // PCG-style LCG (seed 0xfeedface12345678): the 80-mer at position 500
        // is unique in the doubled reference already at m==6, so the k==6 table
        // hands the deep loop an interval of width 1. Without the fast path the
        // deep band runs 74 iterations of two binary searches each (~148 probes);
        // with the fast path the first deep iteration fires the occ==1 short-cut
        // and the whole tail costs exactly 1 probe.
        let mut state: u64 = 0xfeedface12345678;
        let mut fwd = Vec::with_capacity(2000);
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(0xda3e39cb94b95bdb);
            fwd.push((state >> 62) as u8);
        }
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        let k = 6u32;
        let table = idx.build_kmer_table(k, &fwd, e);
        let q = &fwd[500..580]; // 80 bp, reference-lifted -> unique by m==6
        probe_count::reset_depth_probes();
        let _ = idx.forward_spectrum_tabled(q, &fwd, e, &table);
        let hist = probe_count::depth_probes();
        let deep: u64 = hist[(k as usize + 1)..].iter().sum();
        assert!(
            deep <= 6,
            "deep-band (m>{k}) probes should collapse with the fast path, got {deep}"
        );
    }

    /// A `.kmt` whose reference digest no longer matches the sidecar is ignored
    /// (best-effort load) and the forward search falls back to the full,
    /// always-correct path — never silently-wrong SMEMs.
    #[test]
    fn mismatched_kmt_is_ignored_and_falls_back() {
        let fwd: Vec<u8> = (0..200u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let dir = tempfile::tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, &fwd);
        let prefix = dir.path().join("r.prmi");
        let cfg = TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_kmer_table_k(6);
        build_sidecar_from_pac_with_config(
            &pac,
            &prefix,
            None,
            MaskConfig::default(),
            1,
            Some(cfg),
        )
        .unwrap();
        assert!(LearnedIndex::open(&prefix).unwrap().kmt.is_some());

        // Flip a byte in the `.kmt` ref_digest region (header[24..56]); same size,
        // valid magic — so it opens, but the digest no longer matches `.meta`.
        let kpath = crate::sidecar::SidecarPaths::from_prefix(&prefix).kmt;
        let mut bytes = std::fs::read(&kpath).unwrap();
        bytes[30] ^= 0xff;
        std::fs::write(&kpath, &bytes).unwrap();

        let idx = LearnedIndex::open(&prefix).unwrap();
        assert!(idx.kmt.is_none(), "mismatched .kmt must be ignored");
        let e = PacEncoding::Unpacked;
        let q = vec![1u8, 2, 3, 0, 1, 2, 3];
        assert_eq!(
            idx.forward_spectrum_auto(&q, &fwd, e),
            idx.forward_spectrum(&q, &fwd, e),
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// `backward_spectrum_via_stepper` == serial `backward_spectrum`, for
        /// anchors derived from forward over random refs/reads.
        #[test]
        fn bwd_stepper_equals_serial(
            fwd in prop::collection::vec(0u8..=3, 60..240),
            read in prop::collection::vec(0u8..=3, 30..120),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            // pivots across the read; each forward step is an anchor.
            for pivot in (5..read.len()).step_by(7) {
                for a in idx.forward_spectrum(&read[pivot..], &fwd, e) {
                    let t = BwdTask {
                        sa_start: a.sa_start,
                        occ_count: a.occ_count,
                        anchor_len: a.match_len,
                        read: &read,
                        pivot,
                    };
                    prop_assert_eq!(
                        idx.backward_spectrum_via_stepper(&t, &fwd, e),
                        idx.backward_spectrum(t.sa_start, t.occ_count, t.anchor_len, &read, pivot, &fwd, e),
                        "stepper != serial at pivot {}", pivot
                    );
                }
            }
        }
    }

    /// Map a backward spectrum's maximal step to a `MemMatch`, with the anchor as
    /// the result when there is no left extension and zero when `occ == 0` — the
    /// exact contract `mem_search_backward` implements.
    fn maximal_backward(steps: &[SmemStep], sa_start: u64, occ: u64, anchor_len: u64) -> MemMatch {
        if occ == 0 {
            return MemMatch {
                match_len: 0,
                sa_start: 0,
                occ: 0,
            };
        }
        match steps.last() {
            Some(s) => MemMatch {
                match_len: s.match_len,
                sa_start: s.sa_start,
                occ: s.occ_count,
            },
            None => MemMatch {
                match_len: anchor_len,
                sa_start,
                occ,
            },
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// `mem_search_backward` == the maximal step of the INDEPENDENT model-free
        /// backward reference (and of production `backward_spectrum`), mapped via
        /// the anchor/zero contract — for anchors derived from forward.
        #[test]
        fn mem_search_backward_equals_maximal_backward_step(
            fwd in prop::collection::vec(0u8..=3, 60..240),
            read in prop::collection::vec(0u8..=3, 30..120),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            for pivot in (5..read.len()).step_by(7) {
                for a in idx.forward_spectrum(&read[pivot..], &fwd, e) {
                    let (s, o, al) = (a.sa_start, a.occ_count, a.match_len);
                    let got = idx.mem_search_backward(s, o, al, &read, pivot, &fwd, e);
                    // Independent oracle: the model-free backward reference.
                    let ref_steps =
                        idx.backward_spectrum_reference(s, o, al, &read, pivot, &fwd, e);
                    prop_assert_eq!(
                        got,
                        maximal_backward(&ref_steps, s, o, al),
                        "mem_search_backward != independent backward oracle at pivot {}", pivot
                    );
                    // And the production spectrum's maximal step.
                    let prod_steps = idx.backward_spectrum(s, o, al, &read, pivot, &fwd, e);
                    prop_assert_eq!(
                        got,
                        maximal_backward(&prod_steps, s, o, al),
                        "mem_search_backward != backward_spectrum maximal at pivot {}", pivot
                    );
                }
            }
        }
    }

    /// `mem_search_backward` edge cases: `occ_count == 0` → all-zero; `pivot == 0`
    /// (no left base) → the anchor itself (`match_len == anchor_len`).
    #[test]
    fn mem_search_backward_zero_and_anchor_cases() {
        let fwd: Vec<u8> = (0..200u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 9) & 3) as u8)
            .collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        // occ_count == 0 -> all zero.
        assert_eq!(
            idx.mem_search_backward(0, 0, 5, &fwd, 10, &fwd, e),
            MemMatch {
                match_len: 0,
                sa_start: 0,
                occ: 0
            }
        );
        // pivot == 0: no left extension -> the anchor is returned unchanged.
        let a = idx.forward_spectrum(&fwd, &fwd, e).last().copied().unwrap();
        let m = idx.mem_search_backward(a.sa_start, a.occ_count, a.match_len, &fwd, 0, &fwd, e);
        assert_eq!(
            m,
            MemMatch {
                match_len: a.match_len,
                sa_start: a.sa_start,
                occ: a.occ_count
            }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]
        /// `backward_spectrum_lockstep` over a batch of anchors == per-anchor
        /// serial `backward_spectrum`, element for element.
        #[test]
        fn bwd_lockstep_equals_serial(
            fwd in prop::collection::vec(0u8..=3, 60..240),
            reads in prop::collection::vec(prop::collection::vec(0u8..=3, 30..120), 1..4),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            // Gather a batch of anchors across all reads/pivots.
            let mut tasks = Vec::new();
            let mut want = Vec::new();
            for read in &reads {
                for pivot in (5..read.len()).step_by(9) {
                    for a in idx.forward_spectrum(&read[pivot..], &fwd, e) {
                        tasks.push(BwdTask {
                            sa_start: a.sa_start,
                            occ_count: a.occ_count,
                            anchor_len: a.match_len,
                            read,
                            pivot,
                        });
                        want.push(idx.backward_spectrum(
                            a.sa_start, a.occ_count, a.match_len, read, pivot, &fwd, e,
                        ));
                    }
                }
            }
            let got = idx.backward_spectrum_lockstep(&tasks, &fwd, e);
            prop_assert_eq!(got, want, "lockstep batch != serial");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// `mem_search` == the maximal step of the independent brute-force forward
        /// oracle (and of `forward_spectrum_auto`), for random refs/queries.
        #[test]
        fn mem_search_equals_maximal_forward_step(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 0..50), 1..6),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let table = idx.build_kmer_table(5, &fwd, e);
            for q in &queries {
                let got = idx.mem_search(q, &fwd, e);
                // Oracle: maximal step (or zero match).
                let oracle = forward_spectrum_oracle(&idx, q, &fwd, e);
                let want = match oracle.last() {
                    Some(s) => MemMatch { match_len: s.match_len, sa_start: s.sa_start, occ: s.occ_count },
                    None => MemMatch { match_len: 0, sa_start: 0, occ: 0 },
                };
                prop_assert_eq!(got, want, "mem_search != oracle maximal, query={:?}", q);
                // Also equals forward_spectrum_auto's last step (table path).
                let auto = idx.forward_spectrum_tabled(q, &fwd, e, &table);
                let want_auto = match auto.last() {
                    Some(s) => MemMatch { match_len: s.match_len, sa_start: s.sa_start, occ: s.occ_count },
                    None => MemMatch { match_len: 0, sa_start: 0, occ: 0 },
                };
                prop_assert_eq!(got, want_auto, "mem_search != forward_spectrum_tabled last, query={:?}", q);
            }
        }

        /// The ISA-launch (`mem_search_from_hint`) MUST be byte-identical to the
        /// from-scratch `mem_search` for EVERY SA index in the maximal interval —
        /// the launch hint changes speed, never the answer. This is the hard gate
        /// for the est_hint>0 / no_search fast path.
        #[test]
        fn mem_search_hint_equals_unhinted(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..50), 1..6),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            for q in &queries {
                let unhinted = idx.mem_search(q, &fwd, e);
                if unhinted.match_len == 0 {
                    continue; // no match → no valid hint exists
                }
                // Every SA index in the maximal interval is a valid launch hint
                // and must reproduce the identical (sa_start, occ, match_len).
                for off in 0..unhinted.occ {
                    let hint = unhinted.sa_start + off;
                    let hinted = idx.mem_search_from_hint(q, hint, true, &fwd, e);
                    prop_assert_eq!(
                        hinted, unhinted,
                        "hinted != unhinted at hint={}, query={:?}", hint, q
                    );
                    // want_interval=false must still report the same match_len.
                    let ml_only = idx.mem_search_from_hint(q, hint, false, &fwd, e);
                    prop_assert_eq!(ml_only.match_len, unhinted.match_len, "match_len-only mismatch");
                }
            }
        }

        /// The hinted forward SPECTRUM (`forward_spectrum_from_hint`) MUST be
        /// byte-identical to the cold `forward_spectrum` trace for EVERY SA index
        /// in the maximal interval — the parent-interval walk changes speed, never
        /// the trace. The hard gate for the trace-returning no_search path that
        /// the bwa-meme reseed needs (min_intv-gated multi-anchor emission).
        #[test]
        fn forward_spectrum_from_hint_equals_cold(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            queries in prop::collection::vec(prop::collection::vec(0u8..=3, 1..50), 1..6),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            for q in &queries {
                let cold = idx.forward_spectrum(q, &fwd, e);
                let maximal = idx.mem_search(q, &fwd, e);
                if maximal.match_len == 0 {
                    // No match → no interior hint exists; hint==0 falls back to cold.
                    prop_assert_eq!(idx.forward_spectrum_from_hint(q, 0, &fwd, e), cold);
                    continue;
                }
                // Every SA index in the maximal interval is a valid launch hint and
                // must reproduce the identical breakpoint trace.
                for off in 0..maximal.occ {
                    let hint = maximal.sa_start + off;
                    let hinted = idx.forward_spectrum_from_hint(q, hint, &fwd, e);
                    prop_assert_eq!(
                        hinted, cold.clone(),
                        "hinted spectrum != cold at hint={}, query={:?}", hint, q
                    );
                }
            }
        }

        /// Backward `mem_search_backward_from_hint` == from-scratch
        /// `mem_search_backward` when the hint is at the maximal-extension locus.
        /// `build_mode2` builds NO `.isa`, so this also exercises the loose-seed
        /// fallback (`isa_at(..).unwrap_or(hint)`) — `find_boundary`'s
        /// seed-independence must still yield the identical interval.
        #[test]
        fn mem_search_backward_hint_equals_unhinted(
            fwd in prop::collection::vec(0u8..=3, 80..220),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let sa_num = idx.sa_num();
            // Reference-lifted reads: read == a window of fwd at genomic `s`, so
            // the anchor's natural locus is `s + pivot` and its left walk follows
            // the read↔reference alignment (the maximal-extension locus).
            let w = 50usize;
            for s in (0..fwd.len().saturating_sub(w)).step_by(23) {
                let read = &fwd[s..s + w];
                for pivot in [12usize, 24, 36] {
                    let anchor = idx.mem_search(&read[pivot..], &fwd, e);
                    if anchor.match_len == 0 {
                        continue;
                    }
                    let global = idx.mem_search_backward(
                        anchor.sa_start, anchor.occ, anchor.match_len, read, pivot, &fwd, e,
                    );
                    // Inverse SA at genomic `s + pivot` (no `.isa` here → scan).
                    let refpos = (s + pivot) as u64;
                    let hint = (0..sa_num).find(|&i| idx.sa_position_for(i) == refpos).unwrap();
                    let hinted = idx.mem_search_backward_from_hint(
                        read, pivot, anchor.match_len, hint, true, &fwd, e,
                    );
                    prop_assert_eq!(
                        hinted, global,
                        "backward hinted != from-scratch at s={}, pivot={}", s, pivot
                    );
                }
            }
        }

        /// The hinted backward SPECTRUM (`backward_spectrum_from_hint`, full trace)
        /// MUST be byte-identical to cold `backward_spectrum` when the hint is at
        /// the maximal-extension locus — every left step, not just the maximal.
        /// The hard gate for the trace-returning hinted left extension.
        #[test]
        fn backward_spectrum_from_hint_equals_cold(
            fwd in prop::collection::vec(0u8..=3, 80..220),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let sa_num = idx.sa_num();
            let w = 50usize;
            for s in (0..fwd.len().saturating_sub(w)).step_by(23) {
                let read = &fwd[s..s + w];
                for pivot in [12usize, 24, 36] {
                    let anchor = idx.mem_search(&read[pivot..], &fwd, e);
                    if anchor.match_len == 0 {
                        continue;
                    }
                    let cold = idx.backward_spectrum(
                        anchor.sa_start, anchor.occ, anchor.match_len, read, pivot, &fwd, e,
                    );
                    let refpos = (s + pivot) as u64;
                    let hint = (0..sa_num).find(|&i| idx.sa_position_for(i) == refpos).unwrap();
                    let hinted = idx.backward_spectrum_from_hint(
                        read, pivot, anchor.match_len, hint, &fwd, e,
                    );
                    prop_assert_eq!(
                        hinted, cold,
                        "backward hinted spectrum != cold at s={}, pivot={}", s, pivot
                    );
                }
            }
        }

        /// STALE-HINT SAFETY (the gating contract for the bwa-meme reseed): a
        /// WRONG `hint` to `mem_search_backward_from_hint` can only UNDER-extend or
        /// return `occ == 0` — it can NEVER return a full-length-but-wrong interval
        /// that would pass a length check. Two invariants, over random wrong hints:
        ///   (a) self-consistency: when `occ > 0`, the returned `(sa_start, occ)` is
        ///       EXACTLY the true SA interval of the read substring of length
        ///       `match_len` (`mem_search` of that substring), so `match_len` and
        ///       the interval can never disagree; and
        ///   (b) never-longer: `match_len <= ` the cold (`est_hint = 0`) maximal.
        /// Together ⇒ a consumer LENGTH check (hinted `match_len == expected`) is a
        /// sufficient stale-hint gate: equal length forces the identical substring,
        /// hence the identical, correct interval.
        #[test]
        fn backward_stale_hint_is_self_consistent_and_not_longer(
            fwd in prop::collection::vec(0u8..=3, 80..220),
            hint_frac in 0.0f64..1.0,
            pivot in 12usize..40,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let sa_num = idx.sa_num();
            let w = 50usize;
            prop_assume!(fwd.len() >= w);
            // A read lifted from the reference (so the length-1 anchor occurs), fed
            // a DELIBERATELY WRONG hint: a random SA index, not read[pivot]'s locus.
            let s = pivot % (fwd.len() - w + 1);
            let read = &fwd[s..s + w];
            let anchor_len: u64 = 1;
            let anchor_end = pivot + anchor_len as usize;
            let wrong_hint = 1 + (hint_frac * (sa_num - 1) as f64) as u64; // in [1, sa_num)

            let hinted = idx.mem_search_backward_from_hint(
                read, pivot, anchor_len, wrong_hint, true, &fwd, e,
            );

            // (a) Self-consistency: interval == true interval of the implied substring.
            if hinted.occ > 0 {
                let left_ext = hinted.match_len - anchor_len;
                let p_start = pivot - left_ext as usize;
                let p = &read[p_start..anchor_end];
                let truth = idx.mem_search(p, &fwd, e);
                prop_assert_eq!(truth.match_len as usize, p.len(),
                    "implied substring must occur fully (p={:?})", p);
                prop_assert_eq!(
                    (hinted.sa_start, hinted.occ), (truth.sa_start, truth.occ),
                    "interval disagrees with match_len's substring at hint={}", wrong_hint
                );
            }

            // (b) Never longer than the cold maximal of the same length-1 anchor.
            let anchor = idx.mem_search(&read[pivot..anchor_end], &fwd, e);
            let cold = idx.mem_search_backward(
                anchor.sa_start, anchor.occ, anchor_len, read, pivot, &fwd, e,
            );
            prop_assert!(
                hinted.match_len <= cold.match_len,
                "stale hint over-extended: hinted={} cold={} hint={}",
                hinted.match_len, cold.match_len, wrong_hint
            );
        }

        /// The `.kmt`-seeded cold backward spectrum (`backward_spectrum_tabled`)
        /// MUST be byte-identical to the model-seeded cold `backward_spectrum` —
        /// the seed source changes probe count, never the trace (the divergence
        /// guard for the parallel loop copy).
        #[test]
        fn backward_spectrum_tabled_equals_cold(
            fwd in prop::collection::vec(0u8..=3, 80..220),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let e = PacEncoding::Unpacked;
            let table = idx.build_kmer_table(5, &fwd, e);
            let w = 50usize;
            for s in (0..fwd.len().saturating_sub(w)).step_by(23) {
                let read = &fwd[s..s + w];
                for pivot in [12usize, 24, 36] {
                    let anchor = idx.mem_search(&read[pivot..], &fwd, e);
                    if anchor.match_len == 0 {
                        continue;
                    }
                    let cold = idx.backward_spectrum(
                        anchor.sa_start, anchor.occ, anchor.match_len, read, pivot, &fwd, e,
                    );
                    let tabled = idx.backward_spectrum_tabled(
                        anchor.sa_start, anchor.occ, anchor.match_len, read, pivot, &fwd, e, &table,
                    );
                    prop_assert_eq!(
                        tabled, cold,
                        "backward tabled != model-seeded cold at s={}, pivot={}", s, pivot
                    );
                }
            }
        }
    }

    /// Both backward-spectrum PROTOTYPES must fail closed (empty trace) on a
    /// pathological `anchor_len` rather than panicking on the `pivot + anchor_len`
    /// cast/add — mirroring `mem_search_backward_from_hint`'s guards.
    #[test]
    fn backward_spectrum_prototypes_reject_overflowing_anchor_len() {
        let fwd: Vec<u8> = (0..120u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        let read = &fwd[..50];
        let table = idx.build_kmer_table(5, &fwd, e);
        // `anchor_len = u64::MAX` overflows `pivot + anchor_len`; both prototypes
        // must return an empty trace instead of panicking.
        assert!(idx
            .backward_spectrum_from_hint(read, 24, u64::MAX, 1, &fwd, e)
            .is_empty());
        assert!(idx
            .backward_spectrum_tabled(1, 1, u64::MAX, read, 24, &fwd, e, &table)
            .is_empty());
    }

    #[test]
    fn mem_search_deep_unique_and_empty() {
        let fwd: Vec<u8> = (0..400u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 11) & 3) as u8)
            .collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        // Reference-lifted 60-mer => matches to full length (unique deep).
        // fwd[116..176] is unique in the doubled (fwd||RC) text for this LCG seed.
        let q = &fwd[116..176];
        let m = idx.mem_search(q, &fwd, e);
        assert_eq!(m.match_len, 60, "reference-lifted query must match fully");
        assert_eq!(m.occ, 1, "a 60-mer is unique in this 400-base ref");
        // Empty query => zero match.
        assert_eq!(
            idx.mem_search(&[], &fwd, e),
            MemMatch {
                match_len: 0,
                sa_start: 0,
                occ: 0
            }
        );
        // A single base that does occur => match_len >= 1.
        assert!(idx.mem_search(&[fwd[0]], &fwd, e).match_len >= 1);
    }

    #[test]
    fn forward_spectrum_fill_matches_vec_and_reports_overflow() {
        let fwd: Vec<u8> = (0..500u32)
            .map(|i| ((i.wrapping_mul(40_503) >> 7) & 3) as u8)
            .collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        // A query with several breakpoints (wide shallow → narrow deep).
        let q = &fwd[40..130];
        let want = idx.forward_spectrum_auto(q, &fwd, e);

        // Ample buffer: fill writes exactly the Vec steps and returns the count.
        let mut buf = vec![
            SmemStep {
                sa_start: 0,
                occ_count: 0,
                match_len: 0
            };
            want.len() + 4
        ];
        let n = idx.forward_spectrum_auto_fill(q, &fwd, e, &mut buf);
        assert_eq!(n, want.len());
        assert_eq!(&buf[..n], &want[..]);

        // Undersized buffer: returns the full count (> capacity) and writes the
        // prefix it could fit; the count is what the caller uses to retry.
        if !want.is_empty() {
            let cap = want.len() - 1;
            let mut small = vec![
                SmemStep {
                    sa_start: 0,
                    occ_count: 0,
                    match_len: 0
                };
                cap
            ];
            let n2 = idx.forward_spectrum_auto_fill(q, &fwd, e, &mut small);
            assert_eq!(n2, want.len(), "fill must report the total step count");
            assert!(n2 > cap, "this case must overflow the buffer");
        }
    }

    #[test]
    fn backward_spectrum_fill_matches_vec_and_reports_overflow() {
        let fwd: Vec<u8> = (0..500u32)
            .map(|i| ((i.wrapping_mul(2_246_822_519) >> 9) & 3) as u8)
            .collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        // Derive an anchor from a forward step, then left-extend it.
        let read = &fwd;
        let pivot = 120usize;
        let anchor = idx.forward_spectrum(&read[pivot..], &fwd, e);
        let a = *anchor.last().expect("anchor exists");
        let want =
            idx.backward_spectrum(a.sa_start, a.occ_count, a.match_len, read, pivot, &fwd, e);

        // Ample buffer: fill writes exactly the Vec steps and returns the count.
        let mut buf = vec![
            SmemStep {
                sa_start: 0,
                occ_count: 0,
                match_len: 0
            };
            want.len() + 4
        ];
        let n = idx.backward_spectrum_fill(
            a.sa_start,
            a.occ_count,
            a.match_len,
            read,
            pivot,
            &fwd,
            e,
            &mut buf,
        );
        assert_eq!(n, want.len());
        assert_eq!(&buf[..n], &want[..]);

        // Undersized buffer reports the total count even though it can't fit.
        if !want.is_empty() {
            let cap = want.len() - 1;
            let mut small = vec![
                SmemStep {
                    sa_start: 0,
                    occ_count: 0,
                    match_len: 0
                };
                cap
            ];
            let n2 = idx.backward_spectrum_fill(
                a.sa_start,
                a.occ_count,
                a.match_len,
                read,
                pivot,
                &fwd,
                e,
                &mut small,
            );
            assert_eq!(n2, want.len(), "fill must report the total step count");
            assert!(n2 > cap, "this case must overflow the buffer");
        }
    }

    /// Reference-lifted queries extend to full depth and become unique, so they
    /// drive the occ==1 fast path hard. Both forward paths must stay byte-
    /// identical to the independent oracle across many lift offsets.
    #[test]
    fn forward_fast_path_deep_unique_matches_oracle() {
        let fwd: Vec<u8> = (0..600u32)
            .map(|i| ((i.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 8) & 3) as u8)
            .collect();
        let (_dir, idx) = build_mode2(&fwd);
        let e = PacEncoding::Unpacked;
        let table = idx.build_kmer_table(6, &fwd, e);
        for start in (0..fwd.len() - 60).step_by(37) {
            let q = &fwd[start..start + 60];
            let oracle = forward_spectrum_oracle(&idx, q, &fwd, e);
            assert_eq!(
                idx.forward_spectrum(q, &fwd, e),
                oracle,
                "forward_spectrum != oracle at lift {start}"
            );
            assert_eq!(
                idx.forward_spectrum_tabled(q, &fwd, e, &table),
                oracle,
                "forward_spectrum_tabled != oracle at lift {start}"
            );
        }
    }
}
