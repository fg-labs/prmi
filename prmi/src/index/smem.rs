// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Pac decoding primitives shared by the query path.
//!
//! These helpers decode a caller-owned reference (`pac`) in either
//! 1-base-per-byte (`Unpacked`) or 2-bit-packed (`Packed`, BWA / BWA-MEME
//! `bntpac`) form. The 2×-aware spectrum query path (`spectrum.rs`) routes
//! every reference read through [`pac_base_at`] (via `doubled_base_at`).

use crate::error::Result;

/// Encoding of a caller-owned pac slice.
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

/// Decode a single base at position `pos` from `pac`. Returns `None` if
/// `pos` is out of range. Used by the spectrum query path's per-position
/// reference reads (via `doubled_base_at`).
#[inline]
pub fn pac_base_at(pac: &[u8], pos: u64, enc: PacEncoding) -> Option<u8> {
    match enc {
        PacEncoding::Unpacked => {
            let p = pos as usize;
            if p >= pac.len() {
                None
            } else {
                Some(pac[p])
            }
        }
        PacEncoding::Packed { num_bases } => {
            if pos >= num_bases {
                None
            } else {
                let p = pos as usize;
                // `num_bases` is caller-supplied and may disagree with `pac.len()`
                // now that this helper is public; index defensively so a
                // truncated/inconsistent buffer yields `None` instead of panicking.
                let byte = *pac.get(p / 4)?;
                let shift = 6 - 2 * ((p % 4) as u32);
                Some((byte >> shift) & 0x3)
            }
        }
    }
}

/// Validate that a 2-bit packed `pac` slice is long enough to hold `num_bases`
/// bases (4 per byte). Returns `Err(Error::Internal)` rather than letting a
/// short slice panic on an out-of-bounds byte access inside the spectrum walk.
/// `ctx` names the calling entry point for the error.
pub(crate) fn validate_packed_pac(pac: &[u8], num_bases: u64, ctx: &str) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Copy up to 32 bases starting at `pos` from `pac` (in whichever
    /// encoding) into `out`. Returns the number of bases actually written.
    /// Test-only helper exercising the `pac_base_at` decoding convention in
    /// bulk; the production query path reads one base at a time.
    fn read_unpacked_window(pac: &[u8], pos: u64, enc: PacEncoding, out: &mut [u8; 32]) -> usize {
        let mut n = 0;
        for slot in out.iter_mut() {
            match pac_base_at(pac, pos + n as u64, enc) {
                Some(b) => {
                    *slot = b;
                    n += 1;
                    if n == 32 {
                        break;
                    }
                }
                None => break,
            }
        }
        n
    }

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

    #[test]
    fn pac_base_at_unpacked() {
        let bases: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        assert_eq!(pac_base_at(&bases, 0, PacEncoding::Unpacked), Some(0));
        assert_eq!(pac_base_at(&bases, 3, PacEncoding::Unpacked), Some(3));
        assert_eq!(pac_base_at(&bases, 5, PacEncoding::Unpacked), Some(1));
        assert_eq!(pac_base_at(&bases, 6, PacEncoding::Unpacked), None);
    }

    #[test]
    fn pac_base_at_packed() {
        let bases: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        let packed = pack_bases(&bases);
        let enc = PacEncoding::Packed { num_bases: 6 };
        assert_eq!(pac_base_at(&packed, 0, enc), Some(0));
        assert_eq!(pac_base_at(&packed, 3, enc), Some(3));
        assert_eq!(pac_base_at(&packed, 5, enc), Some(1));
        assert_eq!(pac_base_at(&packed, 6, enc), None);
    }

    #[test]
    fn pac_base_at_packed_inconsistent_buffer_is_none() {
        // num_bases claims 8 (needs 2 bytes) but only 1 byte is supplied;
        // a position whose byte index is past the slice must yield None, not panic.
        let packed = [0x1Bu8]; // one byte = bases 0..4
        let enc = PacEncoding::Packed { num_bases: 8 };
        assert_eq!(pac_base_at(&packed, 4, enc), None);
    }

    #[test]
    fn validate_packed_pac_rejects_short_slice() {
        // num_bases=8 needs ceil(8/4)=2 bytes; a 1-byte slice is too short.
        assert!(validate_packed_pac(&[0u8], 8, "test").is_err());
        // Exactly enough bytes is accepted.
        assert!(validate_packed_pac(&[0u8, 0u8], 8, "test").is_ok());
    }
}
