// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `smem_range` — resolve the SA range matching a query via bounded local
//! search anchored by the §4.4 lookup prediction.

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::error::Result;
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

impl LearnedIndex {
    /// Resolve the SA range matching the query against the supplied pac
    /// (1-base-per-byte; values 0..=3). See v0.1 brief §5 for the C API
    /// shape this backs.
    pub fn smem_range(&self, query: &[u8], pac: &[u8]) -> Result<(u64, u64, u64)> {
        let SmemRange { k, l, s } = self.smem_range_batch(&[query], pac)?[0];
        Ok((k, l, s))
    }

    /// Batch-friendly variant. The C API in v0.1 calls into the single
    /// version, but internals stay batch-shaped so v0.2 can expose a batch
    /// FFI as an additive change.
    pub fn smem_range_batch(&self, queries: &[&[u8]], pac: &[u8]) -> Result<Vec<SmemRange>> {
        let sa_num = self.sa_num();
        let mut out = Vec::with_capacity(queries.len());
        for &q in queries {
            out.push(self.resolve_one(q, pac, sa_num));
        }
        Ok(out)
    }

    fn resolve_one(&self, query: &[u8], pac: &[u8], sa_num: u64) -> SmemRange {
        let qlen = query.len().min(KMER_LEN);
        let qkey = tokenize_32mer(query, qlen);
        let (pred, err) = self.lookup(qkey);

        // §4.4: `err` IS the bound the caller must search within. Do not
        // widen with `max_error_bound`; that would search the whole SA and
        // defeat the learned index.
        let lo = pred.saturating_sub(err);
        let hi = pred.saturating_add(err).saturating_add(1).min(sa_num);

        let mut k = 0u64;
        let mut l = 0u64;
        let mut in_run = false;
        let mut first_sa_pos = 0u64;
        let mut last_sa_pos = 0u64;
        for i in lo..hi {
            let sa_pos = self.sa().position(i);
            let candidate = sa_anchored_key(pac, sa_pos);
            if candidate == qkey {
                if !in_run {
                    k = i;
                    in_run = true;
                    first_sa_pos = sa_pos;
                }
                l += 1;
                last_sa_pos = sa_pos;
            } else if in_run {
                break;
            }
        }
        if l == 0 {
            return SmemRange { k: 0, l: 0, s: 0 };
        }

        // `s` is the length common to ALL entries in [k, k+l). Because the
        // SA is lex-sorted, the range's common prefix equals the prefix
        // shared by the boundary suffixes (first and last) against the
        // query — never max over the run.
        let s_first = common_prefix_len(query, &pac[first_sa_pos as usize..], qlen);
        let s_last = common_prefix_len(query, &pac[last_sa_pos as usize..], qlen);
        let s = s_first.min(s_last) as u64;
        SmemRange { k, l, s }
    }
}

#[inline]
fn sa_anchored_key(pac: &[u8], sa_pos: u64) -> u64 {
    let start = sa_pos as usize;
    let avail = pac.len().saturating_sub(start).min(KMER_LEN);
    tokenize_32mer(&pac[start..start + avail], avail)
}

fn common_prefix_len(a: &[u8], b: &[u8], cap: usize) -> usize {
    let n = a.len().min(b.len()).min(cap);
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}
