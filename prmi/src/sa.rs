// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Suffix-array construction and 5-byte position packing. The packed
//! representation supports positions up to 2^40 - 1 (~1.1 trillion bases),
//! well past the human genome's ~3.1 Gbp.

use byteorder::{ByteOrder, LittleEndian};
use libsais::{SuffixArrayConstruction, ThreadCount};

use crate::error::{Error, Result};

/// Bytes used to store one packed SA position on disk (5-byte uint40).
pub const BYTES_PER_PACKED_ENTRY: usize = 5;
/// Maximum SA position that fits in the 5-byte on-disk representation (~1.1 Tbp).
pub const MAX_PACKED_POSITION: u64 = (1u64 << 40) - 1;

/// Pack a `u64` position into the on-disk 5-byte layout:
/// `bytes[0..4] = LE u32 position_hi (upper 32 bits)`,
/// `bytes[4]    = u8 position_lo (lowest 8 bits)`.
///
/// `position_hi` holds bits 8..40 (the upper 32); `position_lo` holds bits
/// 0..8 (the lowest 8). Both ranges are half-open (`a..b` means bits `a`
/// through `b-1` inclusive).
/// The full uint40 reconstruction is `(hi as u64) << 8 | lo as u64`.
///
/// Panics if `pos > MAX_PACKED_POSITION`.
#[inline]
pub fn pack_position(pos: u64) -> [u8; BYTES_PER_PACKED_ENTRY] {
    assert!(
        pos <= MAX_PACKED_POSITION,
        "SA position {pos} exceeds 40-bit packing limit"
    );
    let mut out = [0u8; BYTES_PER_PACKED_ENTRY];
    let hi = (pos >> 8) as u32;
    let lo = (pos & 0xff) as u8;
    LittleEndian::write_u32(&mut out[0..4], hi);
    out[4] = lo;
    out
}

/// Inverse of [`pack_position`].
#[inline]
pub fn unpack_position(bytes: &[u8; BYTES_PER_PACKED_ENTRY]) -> u64 {
    let hi = LittleEndian::read_u32(&bytes[0..4]) as u64;
    let lo = bytes[4] as u64;
    (hi << 8) | lo
}

/// Construct the suffix array for a 2-bit–encoded byte sequence.
///
/// `bases` is a slice of 2-bit–coded bases (values 0–3). An empty slice
/// returns an empty `Vec`. On success, each element of the returned `Vec` is
/// the 0-based starting position of the corresponding sorted suffix, stored as
/// `u64`.
///
/// `threads` controls the number of OpenMP threads used during construction:
/// - `0` — auto (OpenMP picks a thread count, typically the CPU count).
/// - `1` — single-threaded (no OpenMP overhead).
/// - `N > 1` — use exactly N threads.
///
/// # Errors
///
/// Returns `Error::SaConstruction` if the underlying `libsais` call fails.
pub fn build_suffix_array(bases: &[u8], threads: usize) -> Result<Vec<u64>> {
    if bases.is_empty() {
        return Ok(Vec::new());
    }

    let sa_i64: Vec<i64> = if threads == 1 {
        SuffixArrayConstruction::for_text(bases)
            .in_owned_buffer64()
            .single_threaded()
            .run()
            .map_err(|e| Error::SaConstruction {
                detail: format!("{e}"),
            })?
            .into_vec()
    } else {
        // threads == 0 → openmp_default() (let OpenMP choose); threads > 1 →
        // clamp to u16::MAX and pass as a fixed count.
        let tc = if threads == 0 {
            ThreadCount::openmp_default()
        } else {
            ThreadCount::fixed(threads.min(u16::MAX as usize) as u16)
        };
        SuffixArrayConstruction::for_text(bases)
            .in_owned_buffer64()
            .multi_threaded(tc)
            .run()
            .map_err(|e| Error::SaConstruction {
                detail: format!("{e}"),
            })?
            .into_vec()
    };

    let sa: Vec<u64> = sa_i64.into_iter().map(|v| v as u64).collect();
    Ok(sa)
}

/// Build the generalized-suffix-array input text for the 2× (forward+RC) SA,
/// byte-identical to bwa-mem3's `write_doubled_pac` recipe.
///
/// `fwd` is the forward reference, one base per byte, values 0..=3
/// (A=0, C=1, G=2, T=3). The returned text has length `2*fwd.len() + 1`:
/// `[Fwd || RC]` in the `b+1` alphabet (`$=0, A=1, C=2, G=3, T=4`) followed
/// by a single `0` sentinel. The RC half is `complement(fwd[l_pac-1-i])`,
/// where `complement = 3 - base`, computed on the 0..=3 values before the
/// `+1` offset.
///
/// **Byte-identity caveat:** this assumes `fwd` uses the SAME ambiguous-base
/// encoding as bwa's `.pac`. bwa maps `N` to a RANDOM base (`lrand48()&3`) at
/// pack time, whereas prmi's FASTA loader currently maps `N` to `A`.
/// Byte-identity with bwa-mem3's FM-index therefore holds only for N-free
/// references; reconciling N handling (e.g. by building from bwa's existing
/// `.pac` rather than re-deriving from FASTA) is tracked for a later milestone.
pub fn build_doubled_2x_text(fwd: &[u8]) -> Vec<u8> {
    let l_pac = fwd.len();
    let mut text = Vec::with_capacity(2 * l_pac + 1);
    for &b in fwd {
        debug_assert!(b <= 3, "base {b} out of range 0..=3");
        text.push(b + 1);
    }
    for &b in fwd.iter().rev() {
        debug_assert!(b <= 3, "base {b} out of range 0..=3");
        text.push((3 - b) + 1);
    }
    text.push(0);
    text
}

/// Map a `b+1`-alphabet text value back to a 0..=3 base. The sentinel `0`
/// maps to `T` (3) — it only appears once, at the end of the text, past any
/// real 32-mer window, so treating it as a T-pad terminator is safe.
/// This is the inverse of the `base + 1` offset applied by
/// [`build_doubled_2x_text`].
#[inline]
pub fn text_value_to_base(v: u8) -> u8 {
    if v == 0 {
        crate::encoding::BASE_T
    } else {
        v - 1
    }
}

/// Construct the generalized suffix array of `text` (which MUST end in a `0`
/// sentinel — see [`build_doubled_2x_text`]). Returns `N+1` SA entries, where
/// `N+1 == text.len()`, including the empty-suffix/sentinel row at index 0.
///
/// Uses the same underlying `libsais` GSA routine (`libsais64_gsa_omp`) that
/// bwa-mem3 calls, so the resulting order is byte-identical given identical
/// input bytes. `threads`: 0 = auto, 1 = single-threaded, N>1 = exactly N.
///
/// # Errors
/// Returns `Error::InvalidInput` if `text` is non-empty but does not end in a
/// `0` sentinel, and `Error::SaConstruction` if the underlying `libsais` call
/// fails.
pub fn build_gsa(text: &[u8], threads: usize) -> Result<Vec<u64>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    if *text.last().unwrap() != 0 {
        return Err(Error::InvalidInput {
            detail: "GSA text must end in a 0 sentinel".into(),
        });
    }

    let sa_i64: Vec<i64> = if threads == 1 {
        SuffixArrayConstruction::for_text(text)
            .in_owned_buffer64()
            .single_threaded()
            .generalized_suffix_array()
            .run()
            .map_err(|e| Error::SaConstruction {
                detail: format!("{e}"),
            })?
            .into_vec()
    } else {
        let tc = if threads == 0 {
            ThreadCount::openmp_default()
        } else {
            ThreadCount::fixed(threads.min(u16::MAX as usize) as u16)
        };
        SuffixArrayConstruction::for_text(text)
            .in_owned_buffer64()
            .multi_threaded(tc)
            .generalized_suffix_array()
            .run()
            .map_err(|e| Error::SaConstruction {
                detail: format!("{e}"),
            })?
            .into_vec()
    };

    Ok(sa_i64.into_iter().map(|v| v as u64).collect())
}

#[cfg(test)]
mod gsa_tests {
    use super::*;

    #[test]
    fn gsa_length_and_sentinel_row() {
        let fwd = [0u8, 1, 2, 3]; // ACGT
        let text = build_doubled_2x_text(&fwd);
        let sa = build_gsa(&text, 1).unwrap();
        assert_eq!(sa.len(), text.len());
        // The trailing 0 is the unique smallest symbol, so the empty/sentinel
        // suffix (start position N = text.len()-1) sorts first.
        assert_eq!(sa[0] as usize, text.len() - 1);
    }

    #[test]
    fn gsa_is_sorted_lexicographically() {
        let fwd = [2u8, 0, 1, 3, 0, 1]; // GACTAC
        let text = build_doubled_2x_text(&fwd);
        let sa = build_gsa(&text, 1).unwrap();
        for w in sa.windows(2) {
            let a = &text[w[0] as usize..];
            let b = &text[w[1] as usize..];
            assert!(a <= b, "SA not sorted at {:?}", w);
        }
    }

    #[test]
    fn gsa_rejects_text_without_sentinel() {
        // A non-empty text that does not end in a 0 sentinel violates the
        // documented precondition and must be rejected at runtime (in release
        // builds too), not silently mis-built.
        let text = [1u8, 2, 3, 4];
        let err = build_gsa(&text, 1).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn gsa_empty_text_is_ok() {
        // The empty-text fast path returns before the sentinel check.
        assert!(build_gsa(&[], 1).unwrap().is_empty());
    }
}

#[cfg(test)]
mod doubled_text_tests {
    use super::*;

    #[test]
    fn doubled_text_layout_and_sentinel() {
        let fwd = [0u8, 1, 2, 3]; // ACGT
        let text = build_doubled_2x_text(&fwd);
        assert_eq!(text.len(), 2 * fwd.len() + 1);
        assert_eq!(&text[0..4], &[1, 2, 3, 4]);
        // reverse(ACGT)=TGCA=3,2,1,0; complement(3-b)=0,1,2,3; +1 => 1,2,3,4.
        assert_eq!(&text[4..8], &[1, 2, 3, 4]);
        assert_eq!(text[8], 0);
    }

    #[test]
    fn doubled_text_rc_is_reverse_complement() {
        let fwd = [0u8, 0, 1, 2]; // AACG
        let text = build_doubled_2x_text(&fwd);
        assert_eq!(&text[0..4], &[1, 1, 2, 3]); // fwd +1
                                                // complement(0,0,1,2)=(3,3,2,1); reverse => (1,2,3,3); +1 => (2,3,4,4).
        assert_eq!(&text[4..8], &[2, 3, 4, 4]);
        assert_eq!(*text.last().unwrap(), 0);
    }

    #[test]
    fn doubled_text_empty() {
        assert_eq!(build_doubled_2x_text(&[]), vec![0u8]);
    }

    // `build_doubled_2x_text` rejects out-of-range bases via `debug_assert!`, which
    // is compiled out at `opt-level=3`, so this `should_panic` test only holds where
    // debug assertions are active. Gating it on `cfg(debug_assertions)` keeps
    // `cargo test --release` green (the release profile gained `lto`/`codegen-units`
    // for the perf work; the assertion semantics are unchanged in debug builds).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "out of range")]
    fn doubled_text_rejects_out_of_range_base() {
        // `4` is outside the documented 0..=3 alphabet; the reverse-complement
        // `3 - b` would underflow, so a debug build must trip the assertion.
        let _ = build_doubled_2x_text(&[0u8, 4]);
    }
}
