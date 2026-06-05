// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! SA-order certification: prove prmi's 2× GSA equals the unique lexicographic
//! suffix ordering of the doubled `[Fwd||RC]+sentinel` text.

use crate::fasta::fasta_to_2bit_with_sha256;
use crate::sa::{build_doubled_2x_text, build_gsa};
use std::path::Path;

/// Independent oracle: the suffix array of `text` computed by a plain
/// comparison sort of all suffix start positions. O(N² log N) — for small
/// references only. Returns start positions sorted by suffix.
pub fn oracle_suffix_array(text: &[u8]) -> Vec<u64> {
    let mut idx: Vec<u64> = (0..text.len() as u64).collect();
    idx.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    idx
}

/// Failure modes of [`certify_sa_order`]: either the SA build itself failed, or
/// prmi's SA disagreed with the oracle at a specific index.
#[derive(Debug)]
pub enum SaCertError {
    /// The underlying `build_gsa` call failed.
    Build(crate::error::Error),
    /// prmi's SA differs from the oracle at `index`.
    Mismatch {
        /// SA index of the first disagreement.
        index: usize,
        /// Position prmi placed at `index`.
        prmi_pos: u64,
        /// Position the oracle placed at `index`.
        oracle_pos: u64,
    },
}

/// Build prmi's 2× GSA over `fwd` and assert it matches the oracle entry by
/// entry. Returns `Ok(num_entries)`, or an [`SaCertError`] describing either a
/// build failure or the first mismatch. This is public SA certification, so a
/// build failure is propagated as a typed error rather than aborting the process.
pub fn certify_sa_order(fwd: &[u8], threads: usize) -> Result<u64, SaCertError> {
    let text = build_doubled_2x_text(fwd);
    let prmi_sa = build_gsa(&text, threads).map_err(SaCertError::Build)?;
    let oracle = oracle_suffix_array(&text);
    for (i, (&p, &o)) in prmi_sa.iter().zip(oracle.iter()).enumerate() {
        if p != o {
            return Err(SaCertError::Mismatch {
                index: i,
                prmi_pos: p,
                oracle_pos: o,
            });
        }
    }
    Ok(prmi_sa.len() as u64)
}

/// Build the 2× SA from a FASTA and certify its order against the independent
/// oracle (exhaustive O(N²) — small references only). Returns the number of
/// certified entries, or an error describing the first mismatch.
pub fn sa_verify_fasta(ref_fa: &Path, threads: usize) -> crate::error::Result<u64> {
    let (bases, _n, _stats, _sha, _sz) = fasta_to_2bit_with_sha256(ref_fa)?;
    match certify_sa_order(&bases, threads) {
        Ok(n) => Ok(n),
        Err(SaCertError::Build(e)) => Err(e),
        Err(SaCertError::Mismatch {
            index,
            prmi_pos,
            oracle_pos,
        }) => Err(crate::error::Error::Internal {
            detail: format!("SA mismatch at index {index}: prmi={prmi_pos} oracle={oracle_pos}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certify_small_reference() {
        let fwd = [2u8, 0, 1, 3, 0, 1, 2, 3, 0, 0, 1, 2];
        assert!(certify_sa_order(&fwd, 1).is_ok());
    }

    #[test]
    fn certify_homopolymer_and_palindrome() {
        let fwd = [0u8, 0, 0, 0, 1, 2, 3, 3, 3]; // AAAACGTTT
        assert!(certify_sa_order(&fwd, 1).is_ok());
    }
}
