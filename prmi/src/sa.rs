// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Suffix-array construction and 5-byte position packing. The packed
//! representation supports positions up to 2^40 - 1 (~1.1 trillion bases),
//! well past the human genome's ~3.1 Gbp.

use byteorder::{ByteOrder, LittleEndian};

pub const BYTES_PER_PACKED_ENTRY: usize = 5;
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
    assert!(pos <= MAX_PACKED_POSITION,
            "SA position {pos} exceeds 40-bit packing limit");
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
