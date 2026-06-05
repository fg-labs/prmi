// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! 2-bit base encoding and 32-mer tokenization. The tokenization rule is
//! authoritative — the trainer (key generation from the SA) and the reader
//! (key generation from query input) must call the same function to stay
//! interoperable. See the v0.1 handoff brief §6 for the spec.

/// 2-bit code for adenine (A).
pub const BASE_A: u8 = 0;
/// 2-bit code for cytosine (C).
pub const BASE_C: u8 = 1;
/// 2-bit code for guanine (G).
pub const BASE_G: u8 = 2;
/// 2-bit code for thymine (T).
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
/// The `len` parameter controls how many bases are active: pass `len < bases.len()`
/// to use only a prefix of the slice (e.g., for SA positions near the genome end);
/// `bases[..len]` is the active window. The caller may pass any value of `len`;
/// it is clamped to `min(KMER_LEN, bases.len())` internally before any bases are read.
///
/// Panics in debug if any byte in `bases[..min(len, bases.len(), 32)]` is > 3
/// (i.e., the actual active window after clamping, not the raw `len` argument).
#[inline]
pub fn tokenize_32mer(bases: &[u8], len: usize) -> u64 {
    let len = len.min(KMER_LEN).min(bases.len());
    let mut key: u64 = 0;
    for (i, shift) in (0..KMER_LEN).map(|i| (i, 2 * (KMER_LEN - 1 - i))) {
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

/// Reverse-complement a 32-mer key (2-bit packed, MSB-first; same convention
/// as `tokenize_32mer`). The query has `len` valid bases occupying the high
/// `2*len` bits; the trailing `(32 - len) * 2` low bits are T-pads.
///
/// Returns the rev-comp key in the same MSB-first packed form with the same
/// `len` (low bits T-padded). Useful when an aligner needs to look up a
/// query on both strands.
pub fn reverse_complement_key(key: u64, len: usize) -> u64 {
    let len = len.min(KMER_LEN);

    // Step 1: complement every 2-bit base (XOR all-ones). The T-pad bits are
    // overwritten at the end, so their complemented value is irrelevant.
    let complemented = key ^ u64::MAX;

    // Step 2: reverse the order of the 32 2-bit base fields, preserving each
    // field's value. A full bit-reverse reverses field order but also swaps the
    // two bits within each field, so undo that intra-field swap with a single
    // adjacent-pair exchange. This replaces the prior 32-iteration shift loop
    // with a handful of branch-free word operations.
    let r = complemented.reverse_bits();
    let reversed = ((r & 0x5555_5555_5555_5555) << 1) | ((r >> 1) & 0x5555_5555_5555_5555);

    // Step 3: re-align the valid bases into the high `2*len` bits and set the low
    // `(32 - len) * 2` bits to the T-pad. `valid_shift == pad_bits` is in 0..=64;
    // guard the shift-by-64 case (len == 0) explicitly to avoid UB.
    let valid_shift = (KMER_LEN - len) * 2;
    if valid_shift >= 64 {
        // len == 0: no valid bases, the whole key is T-pad.
        u64::MAX
    } else {
        let shifted_high = reversed << valid_shift;
        let pad_mask = (1u64 << valid_shift) - 1;
        shifted_high | pad_mask
    }
}

/// Reverse-complement a slice of 2-bit unpacked bases (1 base per byte,
/// values `0..=3`). Returns a new Vec; in-place would be possible but the
/// allocation cost is negligible for the short keys this is used on.
pub fn reverse_complement_2bit(bases: &[u8]) -> Vec<u8> {
    bases.iter().rev().map(|&b| (b & 0x3) ^ 0x3).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rc_acgt_is_palindrome() {
        let key = tokenize_32mer(&[0, 1, 2, 3], 4);
        assert_eq!(reverse_complement_key(key, 4), key);
    }

    #[test]
    fn rc_aaaa_is_tttt() {
        let aaaa = tokenize_32mer(&[0, 0, 0, 0], 4);
        let tttt = tokenize_32mer(&[3, 3, 3, 3], 4);
        assert_eq!(reverse_complement_key(aaaa, 4), tttt);
    }

    #[test]
    fn rc_double_is_identity() {
        // For arbitrary len in 1..=32 and arbitrary bases, rev-comp of rev-comp is identity.
        for len in 1..=32 {
            let bases: Vec<u8> = (0..len).map(|i| ((i * 7 + 3) & 0x3) as u8).collect();
            let k = tokenize_32mer(&bases, len);
            let rc = reverse_complement_key(k, len);
            let rcrc = reverse_complement_key(rc, len);
            assert_eq!(rcrc, k, "len={len}");
        }
    }

    #[test]
    fn rc_32mer_matches_2bit_path() {
        // Long form: rev-comp the byte array and tokenize, should match rev-comp of the key.
        let bases: Vec<u8> = (0..32).map(|i| ((i * 11 + 5) & 0x3) as u8).collect();
        let k = tokenize_32mer(&bases, 32);
        let rc_bases = reverse_complement_2bit(&bases);
        let k_from_bases = tokenize_32mer(&rc_bases, 32);
        let k_from_key = reverse_complement_key(k, 32);
        assert_eq!(k_from_bases, k_from_key);
    }

    #[test]
    fn rc_short_query_padding() {
        // For len < 32, pad bits should be T's (binary 11) in the LOW positions.
        let key = tokenize_32mer(&[0, 1, 2], 3);
        let rc = reverse_complement_key(key, 3);
        // [A, C, G] reverse → [G, C, A] complement → [C, G, T] = [1, 2, 3]
        let expected = tokenize_32mer(&[1, 2, 3], 3);
        assert_eq!(rc, expected);
        // Low (32-3)*2 = 58 bits should be all 1s (T-pad).
        let low_mask = (1u64 << 58) - 1;
        assert_eq!(rc & low_mask, low_mask);
    }

    #[test]
    fn rc_2bit_array_short() {
        let bases = vec![0, 1, 2, 3, 0]; // ACGTA
        let rc = reverse_complement_2bit(&bases);
        // Reverse: ATGCA → complement: TACGT → [3, 0, 1, 2, 3]
        assert_eq!(rc, vec![3, 0, 1, 2, 3]);
    }

    /// Independent byte-path reference for [`reverse_complement_key`]: unpack the
    /// high `len` 2-bit bases (MSB-first), reverse-complement the base list, then
    /// repack MSB-first with a T-pad via [`tokenize_32mer`]. Shares no code with
    /// the word-level bit-swap under test, so it is a faithful oracle.
    fn rc_key_reference(key: u64, len: usize) -> u64 {
        let len = len.min(KMER_LEN);
        let mut bases: Vec<u8> = Vec::with_capacity(len);
        for j in 0..len {
            let shift = 2 * (KMER_LEN - 1 - j);
            bases.push(((key >> shift) & 0x3) as u8);
        }
        let rc: Vec<u8> = bases.iter().rev().map(|&b| b ^ 0x3).collect();
        tokenize_32mer(&rc, len)
    }

    #[test]
    fn rc_key_single_base_complement() {
        // A↔T, C↔G at len 1: the complemented base sits in the high 2 bits and
        // the remaining 62 bits are T-pad.
        for b in 0u8..4 {
            let k = tokenize_32mer(&[b], 1);
            let expected = tokenize_32mer(&[b ^ 0x3], 1);
            assert_eq!(reverse_complement_key(k, 1), expected, "base {b}");
        }
    }

    #[test]
    fn rc_key_len_zero_is_all_t() {
        // No valid bases → the whole key is the T-pad (all ones), matching
        // tokenize_32mer(&[], 0) and the byte-path reference.
        assert_eq!(reverse_complement_key(0, 0), u64::MAX);
        assert_eq!(reverse_complement_key(0xDEAD_BEEF_1234_5678, 0), u64::MAX);
        assert_eq!(reverse_complement_key(0, 0), rc_key_reference(0, 0));
    }

    #[test]
    fn rc_key_len_over_32_clamps() {
        // `len > 32` is clamped to 32; the result must equal the len-32 result.
        let bases: Vec<u8> = (0..32).map(|i| ((i * 13 + 1) & 0x3) as u8).collect();
        let k = tokenize_32mer(&bases, 32);
        let at_32 = reverse_complement_key(k, 32);
        assert_eq!(reverse_complement_key(k, 33), at_32);
        assert_eq!(reverse_complement_key(k, 40), at_32);
    }

    #[test]
    fn rc_key_hand_computed_32mer() {
        // ACGT repeated 8× (32 bases). Reverse-complement of (ACGT)*8 is (ACGT)*8
        // because RC(ACGT)=ACGT and reversing the tiling of a palindrome unit of
        // even multiplicity returns the same sequence.
        let unit = [0u8, 1, 2, 3]; // A C G T
        let bases: Vec<u8> = unit.iter().copied().cycle().take(32).collect();
        let k = tokenize_32mer(&bases, 32);
        assert_eq!(reverse_complement_key(k, 32), k);
        // AAAA…(32) → TTTT…(32): all-zero key → all-ones key.
        assert_eq!(reverse_complement_key(0, 32), u64::MAX);
    }

    proptest! {
        /// Across all keys and lengths (including 0 and >32), the word-level
        /// implementation must equal the independent byte-path oracle.
        #[test]
        fn rc_key_matches_byte_path_reference(key in any::<u64>(), len in 0usize..=40) {
            prop_assert_eq!(reverse_complement_key(key, len), rc_key_reference(key, len));
        }

        /// Reverse-complement is an involution on the valid (high `len`) bases.
        #[test]
        fn rc_key_double_is_identity_for_valid_bases(key in any::<u64>(), len in 1usize..=32) {
            // Normalize the input so its T-pad already matches a real len-`len`
            // key (tokenize via the reference round-trip), then RC∘RC == identity.
            let normalized = rc_key_reference(rc_key_reference(key, len), len);
            prop_assert_eq!(
                reverse_complement_key(reverse_complement_key(normalized, len), len),
                normalized
            );
        }
    }
}
