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
