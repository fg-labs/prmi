// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! 2-bit base encoding and 32-mer tokenization. The tokenization rule is
//! authoritative — the trainer (key generation from the SA) and the reader
//! (key generation from query input) must call the same function to stay
//! interoperable. See the v0.1 handoff brief §6 for the spec.

pub const BASE_A: u8 = 0;
pub const BASE_C: u8 = 1;
pub const BASE_G: u8 = 2;
pub const BASE_T: u8 = 3;

/// 32 bases per `u64` key.
pub const KMER_LEN: usize = 32;

/// Map an ASCII base byte to its 2-bit code. Returns `None` for N or any
/// non-IUPAC byte; the caller decides how to substitute (the trainer maps
/// N → A for v0.1).
#[inline]
pub fn base_to_2bit(b: u8) -> Option<u8> {
    match b {
        b'A' | b'a' => Some(BASE_A),
        b'C' | b'c' => Some(BASE_C),
        b'G' | b'g' => Some(BASE_G),
        b'T' | b't' => Some(BASE_T),
        _ => None,
    }
}

/// Build a 32-mer key from a slice of 2-bit-coded bases (one base per byte,
/// values 0..=3). MSB-first: `bases[0]` lands in bits 63..62. If `len < 32`
/// the remaining low-bit slots are padded with `BASE_T` (0b11) so that keys
/// sort identically to their underlying sequences as `u64`.
///
/// Panics in debug if any byte in `bases[..len]` is > 3.
#[inline]
pub fn tokenize_32mer(bases: &[u8], len: usize) -> u64 {
    let len = len.min(KMER_LEN);
    let mut key: u64 = 0;
    for i in 0..KMER_LEN {
        let shift = 2 * (KMER_LEN - 1 - i);
        let b = if i < len {
            let v = bases[i];
            debug_assert!(v <= 3, "tokenize_32mer: base byte {v} > 3");
            v as u64
        } else {
            BASE_T as u64
        };
        key |= b << shift;
    }
    key
}
