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
    let len = len.min(32);
    if len == 0 {
        // Empty query: zero valid bases, so all 32 slots are T-pad. This both
        // matches the `tokenize_32mer(&[], 0) == u64::MAX` convention and avoids
        // the shift-by-64 (`reversed << 64`, `1u64 << 64`) panic below.
        return u64::MAX;
    }
    // Step 1: complement every 2-bit base by XOR-ing with the all-ones
    // pattern (0xFFFF_FFFF_FFFF_FFFF). T-pad bits stay valid since 11 XOR 11 = 00 (A);
    // but we'll overwrite the pad region with T's at the end, so the XOR's
    // effect on the pad region is irrelevant.
    let complemented = key ^ 0xFFFF_FFFF_FFFF_FFFF;

    // Step 2: reverse the order of the 32 2-bit chunks. Bit-pair reverse.
    // Standard bit-reverse, but at 2-bit granularity.
    let mut reversed = 0u64;
    for i in 0..32 {
        // Source chunk i lives in bits (62 - 2*i)..(64 - 2*i).
        let chunk = (complemented >> (62 - 2 * i)) & 0x3;
        // Destination after reverse: chunk i goes to position (31 - i).
        reversed |= chunk << (62 - 2 * (31 - i));
    }

    // Step 3: the valid bases of the rev-comp are now in the LOW 2*len bits
    // (since the original valid bases were in the HIGH 2*len bits, and we
    // reversed). Shift them up to the high 2*len bits and T-pad the rest.
    let valid_shift = (32 - len) * 2;
    let shifted_high = reversed << valid_shift;
    // Closed-form T-pad mask: `(32 - len) * 2` low bits all set to 1.
    // When len == 32, no padding needed and 1u64 << 0 - 1 = 0 is correct;
    // avoid the shift-by-64 UB case with an explicit branch.
    let pad_mask: u64 = if len == 32 {
        0
    } else {
        let pad_bits = (32 - len) * 2;
        (1u64 << pad_bits) - 1
    };
    shifted_high | pad_mask
}

/// Reverse-complement a slice of 2-bit unpacked bases (1 base per byte,
/// values `0..=3`). Returns a new Vec; in-place would be possible but
/// the allocation cost is dwarfed by smem_range's local-search loop.
pub fn reverse_complement_2bit(bases: &[u8]) -> Vec<u8> {
    bases.iter().rev().map(|&b| (b & 0x3) ^ 0x3).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
