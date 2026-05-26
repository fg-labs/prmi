// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `smem_range` — resolve the SA range matching a query via bounded local
//! search anchored by the §4.4 lookup prediction.

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::error::Result;
use crate::index::smem_simd::{tokenize_4_at_once, tokenize_4_scalar};
use crate::index::LearnedIndex;

/// An SA range result: start index `k`, length `l`, and common prefix length `s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmemRange {
    /// SA index of the first matching entry.
    pub k: u64,
    /// Number of consecutive SA entries matching the query key.
    pub l: u64,
    /// Length of the longest common prefix shared by all entries in the range.
    pub s: u64,
}

/// Encoding of the caller-owned pac slice handed to `smem_range`.
///
/// - `Unpacked`: one base per byte, values `0..=3` (A=0, C=1, G=2, T=3).
///   `pac.len()` equals the number of bases.
/// - `Packed`: two bits per base, MSB-first within each byte (base 0 in
///   bits 6-7, base 1 in bits 4-5, base 2 in bits 2-3, base 3 in bits 0-1
///   — matching BWA / BWA-MEME's `bntpac` convention). `pac.len()` equals
///   `ceil(num_bases / 4)`. The `num_bases` field gives the exact base
///   count (a final byte may carry padding).
#[derive(Debug, Clone, Copy)]
pub enum PacEncoding {
    /// One base per byte; `pac.len()` is the number of bases.
    Unpacked,
    /// Two bits per base, MSB-first (BWA-MEME `bntpac`). `pac.len()` must
    /// be at least `ceil(num_bases / 4)`.
    Packed {
        /// Exact number of bases represented in the packed slice.
        num_bases: u64,
    },
}

/// Copy up to 32 bases starting at `pos` from `pac` (in whichever encoding)
/// into `out`. Returns the number of bases actually written. Out-of-range
/// positions yield 0 bases.
///
/// `pub(crate)` so that `smem_simd` can call it directly; the public surface
/// is only `smem_range` / `smem_range_packed`.
#[inline]
pub(crate) fn read_unpacked_window_pub(
    pac: &[u8],
    pos: u64,
    enc: PacEncoding,
    out: &mut [u8; 32],
) -> usize {
    read_unpacked_window(pac, pos, enc, out)
}

/// Validate that a 2-bit packed `pac` slice is long enough to hold `num_bases`
/// bases (4 per byte). Returns `Err(Error::Internal)` rather than letting a
/// short slice panic on an out-of-bounds byte access inside
/// [`read_unpacked_window`]. `ctx` names the calling entry point for the error.
fn validate_packed_pac(pac: &[u8], num_bases: u64, ctx: &str) -> Result<()> {
    let required =
        usize::try_from(num_bases.div_ceil(4)).map_err(|_| crate::error::Error::Internal {
            detail: format!("{ctx}: num_bases={num_bases} is too large for this platform"),
        })?;
    if pac.len() < required {
        return Err(crate::error::Error::Internal {
            detail: format!(
                "{ctx}: pac.len()={} is smaller than the required packed length {required} \
                 for num_bases={num_bases}",
                pac.len(),
            ),
        });
    }
    Ok(())
}

#[inline]
fn read_unpacked_window(pac: &[u8], pos: u64, enc: PacEncoding, out: &mut [u8; 32]) -> usize {
    match enc {
        PacEncoding::Unpacked => {
            let start = pos as usize;
            if start >= pac.len() {
                return 0;
            }
            let avail = (pac.len() - start).min(32);
            out[..avail].copy_from_slice(&pac[start..start + avail]);
            avail
        }
        PacEncoding::Packed { num_bases } => {
            if pos >= num_bases {
                return 0;
            }
            let avail = (num_bases - pos).min(32) as usize;
            for (i, slot) in out[..avail].iter_mut().enumerate() {
                let p = pos as usize + i;
                let byte = pac[p / 4];
                let shift = 6 - 2 * ((p % 4) as u32);
                *slot = (byte >> shift) & 0x3;
            }
            avail
        }
    }
}

impl LearnedIndex {
    /// Resolve the SA range matching `query` against the supplied `pac`
    /// (1-base-per-byte; values 0..=3). Returns `(k, l, s)`: SA start index,
    /// match count, and common prefix length.
    ///
    /// **`query` must be exactly 32 bases long.** Shorter queries are not
    /// supported in v0.1: the T-padding used for sub-32-mer keys does not
    /// preserve lex order relative to longer SA suffixes, so the `err` bound
    /// in the learned index does not cover the discrepancy. Longer queries are
    /// silently truncated to the first 32 bases by `resolve_one`, which is
    /// consistent with the 32-mer contract but surprising for callers who pass
    /// longer slices. Use exactly 32 bases to avoid ambiguity.
    ///
    /// Returns `Err(Error::Internal)` if `query.len() != 32`.
    ///
    /// This single-key path calls `resolve_one` directly — no Vec allocation.
    pub fn smem_range(&self, query: &[u8], pac: &[u8]) -> Result<(u64, u64, u64)> {
        if query.len() != KMER_LEN {
            return Err(crate::error::Error::Internal {
                detail: format!(
                    "smem_range: query must be exactly {} bases, got {}",
                    KMER_LEN,
                    query.len(),
                ),
            });
        }
        let sa_num = self.sa_num();
        let SmemRange { k, l, s } = self.resolve_one(query, pac, PacEncoding::Unpacked, sa_num);
        Ok((k, l, s))
    }

    /// Resolve the SA range matching `query` against a 2-bit packed `pac`
    /// (BWA / BWA-MEME `bntpac` convention: 4 bases per byte, MSB-first).
    /// Returns `(k, l, s)`: SA start index, match count, and common prefix length.
    ///
    /// `num_bases` is the exact number of bases encoded in `pac`; `pac.len()`
    /// must be at least `ceil(num_bases / 4)`.
    ///
    /// **`query` must be exactly 32 bases long.** See [`smem_range`](Self::smem_range)
    /// for the rationale — the same constraint applies here.
    ///
    /// Returns `Err(Error::Internal)` if `query.len() != 32`.
    ///
    /// This single-key path calls `resolve_one` directly — no Vec allocation.
    pub fn smem_range_packed(
        &self,
        query: &[u8],
        pac: &[u8],
        num_bases: u64,
    ) -> Result<(u64, u64, u64)> {
        if query.len() != KMER_LEN {
            return Err(crate::error::Error::Internal {
                detail: format!(
                    "smem_range_packed: query must be exactly {} bases, got {}",
                    KMER_LEN,
                    query.len(),
                ),
            });
        }
        validate_packed_pac(pac, num_bases, "smem_range_packed")?;
        let sa_num = self.sa_num();
        let SmemRange { k, l, s } =
            self.resolve_one(query, pac, PacEncoding::Packed { num_bases }, sa_num);
        Ok((k, l, s))
    }

    /// Batch-friendly variant. The C API in v0.1 calls into the single
    /// version, but internals stay batch-shaped so v0.2 can expose a batch
    /// FFI as an additive change.
    ///
    /// This is the original unpacked API kept for backward compatibility.
    pub fn smem_range_batch(&self, queries: &[&[u8]], pac: &[u8]) -> Result<Vec<SmemRange>> {
        self.smem_range_enc(queries, pac, PacEncoding::Unpacked)
    }

    /// Encoding-aware batch variant. Prefer `smem_range_batch` for unpacked
    /// or `smem_range_packed` for the single-query packed path.
    ///
    /// Every query in `queries` must be exactly 32 bases long. The first
    /// query with wrong length causes an `Err(Error::Internal)` return; no
    /// further queries are processed.
    pub fn smem_range_enc(
        &self,
        queries: &[&[u8]],
        pac: &[u8],
        enc: PacEncoding,
    ) -> Result<Vec<SmemRange>> {
        if let PacEncoding::Packed { num_bases } = enc {
            validate_packed_pac(pac, num_bases, "smem_range_enc")?;
        }
        let sa_num = self.sa_num();
        let mut out = Vec::with_capacity(queries.len());
        for (idx, &q) in queries.iter().enumerate() {
            if q.len() != KMER_LEN {
                return Err(crate::error::Error::Internal {
                    detail: format!(
                        "smem_range_batch: query at index {idx} must be exactly {} bases, got {}",
                        KMER_LEN,
                        q.len(),
                    ),
                });
            }
            out.push(self.resolve_one(q, pac, enc, sa_num));
        }
        Ok(out)
    }

    fn resolve_one(&self, query: &[u8], pac: &[u8], enc: PacEncoding, sa_num: u64) -> SmemRange {
        let qlen = query.len().min(KMER_LEN);
        let qkey = tokenize_32mer(query, qlen);
        let (pred, err) = self.lookup(qkey);

        // §4.4: `err` IS the bound the caller must search within. Do not
        // widen with `max_error_bound`; that would search the whole SA and
        // defeat the learned index.
        let lo = pred.saturating_sub(err);
        let hi = pred.saturating_add(err).saturating_add(1).min(sa_num);

        // Determine if this index has stored keys (modes 2/3 or suffix_key_cache).
        // When stored keys are available, we can skip tokenize_4_at_once entirely.
        let has_stored_keys =
            self.sa().bytes_per_entry() > 5 || self.memory_mode() == "suffix_key_cache";

        let mut window = [0u8; 32];
        let mut k = 0u64;
        let mut l = 0u64;
        let mut in_run = false;
        let mut first_sa_pos = 0u64;
        let mut last_sa_pos = 0u64;

        if has_stored_keys {
            // Fast path: use stored keys; skip pac reads for key comparison.
            for sa_idx in lo..hi {
                let sa_pos = self.sa().position(sa_idx);
                let candidate = match self.key_at(sa_idx) {
                    Some(k) => k,
                    None => {
                        // Cache miss (suffix_key_cache mode). Fall back to pac.
                        let avail = read_unpacked_window(pac, sa_pos, enc, &mut window);
                        tokenize_32mer(&window[..avail], avail)
                    }
                };
                if candidate == qkey {
                    if !in_run {
                        k = sa_idx;
                        in_run = true;
                        first_sa_pos = sa_pos;
                    }
                    l += 1;
                    last_sa_pos = sa_pos;
                } else if in_run {
                    break;
                }
            }
        } else {
            // Normal path (mode 1): SIMD-chunked tokenization from pac.
            // SIMD chunked loop: process 4 candidates at a time.
            const CHUNK: usize = 4;
            let mut i = lo;
            'outer: while i + CHUNK as u64 <= hi {
                let mut positions = [0u64; CHUNK];
                for (j, pos) in positions.iter_mut().enumerate() {
                    *pos = self.sa().position(i + j as u64);
                }
                let mut keys = [0u64; CHUNK];
                tokenize_4_at_once(pac, enc, &positions, &mut keys);

                for j in 0..CHUNK {
                    let candidate = keys[j];
                    if candidate == qkey {
                        if !in_run {
                            k = i + j as u64;
                            in_run = true;
                            first_sa_pos = positions[j];
                        }
                        l += 1;
                        last_sa_pos = positions[j];
                    } else if in_run {
                        break 'outer;
                    }
                }
                i += CHUNK as u64;
            }

            // Scalar tail for the remaining (hi - i) < CHUNK candidates.
            // Only run the tail if the chunked loop exhausted all full chunks
            // (i.e., we didn't break out of 'outer mid-chunk on a run termination).
            // When `break 'outer` fires, `i` has not been advanced, so
            // `i + CHUNK <= hi` still holds. When the while condition fails normally,
            // `i + CHUNK > hi`.
            if i + CHUNK as u64 > hi {
                for j_offset in 0..(hi - i) {
                    let sa_idx = i + j_offset;
                    let sa_pos = self.sa().position(sa_idx);
                    let avail = read_unpacked_window(pac, sa_pos, enc, &mut window);
                    let candidate = tokenize_32mer(&window[..avail], avail);
                    if candidate == qkey {
                        if !in_run {
                            k = sa_idx;
                            in_run = true;
                            first_sa_pos = sa_pos;
                        }
                        l += 1;
                        last_sa_pos = sa_pos;
                    } else if in_run {
                        break;
                    }
                }
            }
        }

        if l == 0 {
            return SmemRange { k: 0, l: 0, s: 0 };
        }

        // `s` is the length common to ALL entries in [k, k+l). Because the
        // SA is lex-sorted, the range's common prefix equals the prefix
        // shared by the boundary suffixes (first and last) against the
        // query — never max over the run.
        let avail_first = read_unpacked_window(pac, first_sa_pos, enc, &mut window);
        let s_first = common_prefix_len(query, &window[..avail_first], qlen);
        let avail_last = read_unpacked_window(pac, last_sa_pos, enc, &mut window);
        let s_last = common_prefix_len(query, &window[..avail_last], qlen);
        let s = s_first.min(s_last) as u64;
        SmemRange { k, l, s }
    }

    /// Seed a single long read at multiple pivot offsets in one call.
    ///
    /// For each pivot `i`:
    /// - If `pivot_offsets[i] + 32 > read_len` the result is `(0, 0, 0)` — the
    ///   pivot's 32-mer window extends past the end of the read and is skipped.
    /// - Otherwise the 32 bases starting at `pivot_offsets[i]` are tokenised and
    ///   looked up via `smem_range`. `pac` is in 1-base-per-byte form (values
    ///   `0..=3`); `read_bases.len()` must equal `read_len`.
    ///
    /// Returns a `Vec<SmemRange>` with exactly `pivot_offsets.len()` entries.
    /// Skipped pivots produce `SmemRange { k: 0, l: 0, s: 0 }`.
    ///
    /// Returns `Err(Error::Internal)` if `read_bases.len() != read_len as usize`.
    pub fn smem_range_long_read(
        &self,
        read_bases: &[u8],
        read_len: u64,
        pivot_offsets: &[u64],
        pac: &[u8],
    ) -> crate::error::Result<Vec<SmemRange>> {
        if read_bases.len() != read_len as usize {
            return Err(crate::error::Error::Internal {
                detail: format!(
                    "smem_range_long_read: read_bases.len()={} != read_len={}",
                    read_bases.len(),
                    read_len,
                ),
            });
        }
        self.smem_range_long_read_inner(
            read_bases,
            read_len,
            pivot_offsets,
            pac,
            PacEncoding::Unpacked,
        )
    }

    /// Seed a single long read at multiple pivot offsets, with 2-bit packed pac.
    ///
    /// Same semantics as [`smem_range_long_read`](Self::smem_range_long_read)
    /// except the reference is in BWA / BWA-MEME `bntpac` format (2 bits per
    /// base, MSB-first within each byte). `pac_num_bases` is the exact number of
    /// bases encoded in `pac`; `pac.len()` must be at least
    /// `ceil(pac_num_bases / 4)`.
    pub fn smem_range_long_read_packed(
        &self,
        read_bases: &[u8],
        read_len: u64,
        pivot_offsets: &[u64],
        pac: &[u8],
        pac_num_bases: u64,
    ) -> crate::error::Result<Vec<SmemRange>> {
        if read_bases.len() != read_len as usize {
            return Err(crate::error::Error::Internal {
                detail: format!(
                    "smem_range_long_read_packed: read_bases.len()={} != read_len={}",
                    read_bases.len(),
                    read_len,
                ),
            });
        }
        validate_packed_pac(pac, pac_num_bases, "smem_range_long_read_packed")?;
        self.smem_range_long_read_inner(
            read_bases,
            read_len,
            pivot_offsets,
            pac,
            PacEncoding::Packed {
                num_bases: pac_num_bases,
            },
        )
    }

    /// Shared implementation for the two `smem_range_long_read*` variants.
    fn smem_range_long_read_inner(
        &self,
        read_bases: &[u8],
        read_len: u64,
        pivot_offsets: &[u64],
        pac: &[u8],
        enc: PacEncoding,
    ) -> crate::error::Result<Vec<SmemRange>> {
        let sa_num = self.sa_num();
        let mut results = Vec::with_capacity(pivot_offsets.len());
        for &off in pivot_offsets {
            // Skip pivots where the 32-base window would run off the read end.
            if off.saturating_add(KMER_LEN as u64) > read_len {
                results.push(SmemRange { k: 0, l: 0, s: 0 });
                continue;
            }
            let start = off as usize;
            let q = &read_bases[start..start + KMER_LEN];
            results.push(self.resolve_one(q, pac, enc, sa_num));
        }
        Ok(results)
    }

    /// Pure-scalar variant of `resolve_one` that bypasses SIMD dispatch.
    ///
    /// Exposed as `pub` so integration tests can call it directly for
    /// SIMD-vs-scalar equivalence checks. Not part of the stable public API.
    #[doc(hidden)]
    pub fn resolve_one_scalar(
        &self,
        query: &[u8],
        pac: &[u8],
        enc: PacEncoding,
        sa_num: u64,
    ) -> SmemRange {
        let qlen = query.len().min(KMER_LEN);
        let qkey = tokenize_32mer(query, qlen);
        let (pred, err) = self.lookup(qkey);

        let lo = pred.saturating_sub(err);
        let hi = pred.saturating_add(err).saturating_add(1).min(sa_num);

        let mut window = [0u8; 32];
        let mut k = 0u64;
        let mut l = 0u64;
        let mut in_run = false;
        let mut first_sa_pos = 0u64;
        let mut last_sa_pos = 0u64;

        const CHUNK: usize = 4;
        let mut i = lo;
        'outer: while i + CHUNK as u64 <= hi {
            let mut positions = [0u64; CHUNK];
            for (j, pos) in positions.iter_mut().enumerate() {
                *pos = self.sa().position(i + j as u64);
            }
            let mut keys = [0u64; CHUNK];
            tokenize_4_scalar(pac, enc, &positions, &mut keys);

            for j in 0..CHUNK {
                let candidate = keys[j];
                if candidate == qkey {
                    if !in_run {
                        k = i + j as u64;
                        in_run = true;
                        first_sa_pos = positions[j];
                    }
                    l += 1;
                    last_sa_pos = positions[j];
                } else if in_run {
                    break 'outer;
                }
            }
            i += CHUNK as u64;
        }

        if i + CHUNK as u64 > hi {
            for j_offset in 0..(hi - i) {
                let sa_idx = i + j_offset;
                let sa_pos = self.sa().position(sa_idx);
                let avail = read_unpacked_window(pac, sa_pos, enc, &mut window);
                let candidate = tokenize_32mer(&window[..avail], avail);
                if candidate == qkey {
                    if !in_run {
                        k = sa_idx;
                        in_run = true;
                        first_sa_pos = sa_pos;
                    }
                    l += 1;
                    last_sa_pos = sa_pos;
                } else if in_run {
                    break;
                }
            }
        }

        if l == 0 {
            return SmemRange { k: 0, l: 0, s: 0 };
        }

        let avail_first = read_unpacked_window(pac, first_sa_pos, enc, &mut window);
        let s_first = common_prefix_len(query, &window[..avail_first], qlen);
        let avail_last = read_unpacked_window(pac, last_sa_pos, enc, &mut window);
        let s_last = common_prefix_len(query, &window[..avail_last], qlen);
        let s = s_first.min(s_last) as u64;
        SmemRange { k, l, s }
    }
}

fn common_prefix_len(a: &[u8], b: &[u8], cap: usize) -> usize {
    let n = a.len().min(b.len()).min(cap);
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a slice of unpacked bases (0..=3, one per byte) into 2-bit
    /// packed format (BWA-MEME bntpac convention).
    fn pack_bases(bases: &[u8]) -> Vec<u8> {
        let n = bases.len();
        let mut out = vec![0u8; n.div_ceil(4)];
        for (i, &b) in bases.iter().enumerate() {
            let shift = 6 - 2 * ((i % 4) as u32);
            out[i / 4] |= (b & 0x3) << shift;
        }
        out
    }

    #[test]
    fn pack_bases_round_trips() {
        // 8 bases: ACGTACGT → two bytes
        let bases: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = pack_bases(&bases);
        assert_eq!(packed.len(), 2);

        // byte 0: A=00 C=01 G=10 T=11 → 0b00011011 = 0x1B
        assert_eq!(packed[0], 0x1B);
        // byte 1: same pattern
        assert_eq!(packed[1], 0x1B);

        // Verify round-trip via read_unpacked_window
        let enc = PacEncoding::Packed { num_bases: 8 };
        let mut out = [0u8; 32];
        let n = read_unpacked_window(&packed, 0, enc, &mut out);
        assert_eq!(n, 8);
        assert_eq!(&out[..8], bases.as_slice());
    }

    #[test]
    fn read_unpacked_window_packed_partial_last_byte() {
        // 5 bases: ACGTA — last byte carries only one base (A in bits 6-7)
        let bases: Vec<u8> = vec![0, 1, 2, 3, 0];
        let packed = pack_bases(&bases);
        assert_eq!(packed.len(), 2);

        let enc = PacEncoding::Packed { num_bases: 5 };
        let mut out = [0u8; 32];
        let n = read_unpacked_window(&packed, 0, enc, &mut out);
        assert_eq!(n, 5);
        assert_eq!(&out[..5], bases.as_slice());
    }

    #[test]
    fn read_unpacked_window_out_of_range_returns_zero() {
        let bases: Vec<u8> = vec![0, 1, 2, 3];
        let unpacked_enc = PacEncoding::Unpacked;
        let packed = pack_bases(&bases);
        let packed_enc = PacEncoding::Packed { num_bases: 4 };

        let mut out = [0u8; 32];

        // pos == len for unpacked → 0
        assert_eq!(read_unpacked_window(&bases, 4, unpacked_enc, &mut out), 0);
        // pos beyond num_bases for packed → 0
        assert_eq!(read_unpacked_window(&packed, 4, packed_enc, &mut out), 0);
    }

    #[test]
    fn packed_window_starting_mid_sequence() {
        // Positions 2..6: G T A C
        let bases: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let packed = pack_bases(&bases);
        let enc = PacEncoding::Packed { num_bases: 8 };

        let mut out = [0u8; 32];
        let n = read_unpacked_window(&packed, 2, enc, &mut out);
        assert_eq!(n, 6); // 8 - 2 = 6 bases remaining, capped at min(6, 32)
        assert_eq!(&out[..6], &bases[2..8]);
    }
}
