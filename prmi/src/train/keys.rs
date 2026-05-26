// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Key generation from a suffix array: [`sa_to_keys`].

use crate::encoding::{tokenize_32mer, KMER_LEN};

/// For each SA position in `sa`, tokenize the 32-mer starting at that
/// position in `bases`. Returns one key per SA entry, in SA order.
pub fn sa_to_keys(sa: &[u64], bases: &[u8]) -> Vec<u64> {
    let n = bases.len();
    sa.iter()
        .map(|&pos| {
            let start = pos as usize;
            let avail = n.saturating_sub(start).min(KMER_LEN);
            tokenize_32mer(&bases[start..start + avail], avail)
        })
        .collect()
}
