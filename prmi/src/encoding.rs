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

/// Build the per-position rolling forward 32-mer key cache for a 2-bit-coded read.
///
/// `keys[i]` is byte-identical to `tokenize_32mer(&read[i..], min(KMER_LEN, read.len() - i))`
/// for every `i` in `0..read.len()`, but the whole vector is produced in a single
/// `O(read.len())` right-to-left pass instead of one 32-iteration `tokenize_32mer`
/// per position. This lets a forward search at pivot `i` read `keys[i]` instead of
/// re-tokenizing `read[i..]`.
///
/// Recurrence (MSB-first, matching [`tokenize_32mer`]): `keys[i]` is `keys[i + 1]`
/// shifted right by one 2-bit field, with `read[i]` inserted into the top field
/// (bits 63..62) and the new lowest field (bits 1..0) set to `read[i + 31]` when
/// that base exists, else the `BASE_T` pad. `read[i + 1]`'s field, which was the
/// top field of `keys[i + 1]`, correctly lands one field lower.
///
/// This function is **panic-free on `N` (and any byte `>= 4`)**: every base is masked
/// with `& 0b11` before it enters a key, so an `N` position produces a defined but
/// arbitrary key value rather than panicking. That is exactly what the collect hot
/// path needs: it builds the cache over the WHOLE read (which may contain `N`), but
/// a key `keys[i]` is only ever consumed when the forward query at pivot `i` has
/// `len >= KMER_LEN`, which guarantees the window `read[i..i + 32]` is `N`-free (a
/// forward query is clamped at the next `N`, so `len >= 32` implies no `N` in the
/// first 32 bases). For any position whose 32-mer window straddles an `N`, the key
/// is garbage — but the caller's `len >= KMER_LEN` guard means that key is never read.
///
/// For an `N`-free window, `keys[i]` is byte-identical to
/// `tokenize_32mer(&read[i..], min(KMER_LEN, read.len() - i))`; the masking only
/// changes the (unused) values at `N`-straddling positions.
///
/// Returns an empty vector for an empty read.
#[inline]
pub fn rolling_forward_keys(bases: &[u8]) -> Vec<u64> {
    let mut keys = Vec::new();
    rolling_forward_keys_into(bases, &mut keys);
    keys
}

/// In-place variant of [`rolling_forward_keys`] that fills `keys` rather than
/// allocating a fresh `Vec`.
///
/// `keys` is cleared and resized to `bases.len()` before filling; when its
/// capacity already covers the read (the steady state on a reused scratch
/// buffer) no allocation occurs, so the collect hot path stays allocation-free
/// once warmed up. The written contents are byte-identical to
/// `rolling_forward_keys(bases)` — see that function for the recurrence and the
/// `N`-safety contract.
#[inline]
pub fn rolling_forward_keys_into(bases: &[u8], keys: &mut Vec<u64>) {
    // The all-T pad field (value 0b11) used for positions with no base (past the read end).
    const T_FIELD: u64 = BASE_T as u64;
    let n = bases.len();
    // Reuse the existing allocation: `clear` + `resize` grows to `n` zeros without
    // reallocating whenever `keys.capacity() >= n`.
    keys.clear();
    keys.resize(n, 0u64);
    if n == 0 {
        return;
    }
    // Seed the last position directly. `read[n-1]` may be an `N`: mask to 2 bits so the
    // top field is defined (garbage for `N`, but only positions with a full `N`-free
    // 32-mer window are ever consumed by the guarded caller). The low 62 bits are the
    // T-pad, matching `tokenize_32mer(&read[n-1..], 1)` for a real base.
    let top_shift = 2 * (KMER_LEN - 1);
    let t_pad_low = (1u64 << top_shift) - 1;
    keys[n - 1] = (((bases[n - 1] & 0b11) as u64) << top_shift) | t_pad_low;
    for i in (0..n - 1).rev() {
        // Mask to 2 bits: `N` (4) and any stray byte become a defined base so the
        // rolling build never panics. Keys at `N`-straddling positions are unused.
        let b0 = (bases[i] & 0b11) as u64;
        // Shift the successor's window down by one field; the successor's top field
        // (`read[i+1]`) moves into bits 61..60, its lowest field falls off.
        let shifted = keys[i + 1] >> 2;
        // Insert `read[i]` into the top field (bits 63..62).
        let top = b0 << (2 * (KMER_LEN - 1));
        // The new lowest field is `read[i+31]` if it exists (window is full), else T-pad.
        let low_field = if i + KMER_LEN <= n {
            (bases[i + KMER_LEN - 1] & 0b11) as u64
        } else {
            T_FIELD
        };
        // `shifted` has its low field zeroed by the >>2; OR in the correct low base.
        keys[i] = top | (shifted & !0b11u64) | low_field;
    }
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

    #[test]
    fn rolling_forward_keys_empty_is_empty() {
        assert!(rolling_forward_keys(&[]).is_empty());
    }

    #[test]
    fn rolling_forward_keys_hand_computed_short() {
        // Read shorter than 32: every position is a T-padded prefix.
        let read = [0u8, 1, 2, 3, 0]; // A C G T A
        let keys = rolling_forward_keys(&read);
        assert_eq!(keys.len(), read.len());
        for i in 0..read.len() {
            let want = tokenize_32mer(&read[i..], KMER_LEN.min(read.len() - i));
            assert_eq!(keys[i], want, "position {i}");
        }
    }

    #[test]
    fn rolling_forward_keys_into_matches_allocating_variant() {
        // The `_into` variant is byte-identical to `rolling_forward_keys` for reads
        // shorter than, equal to, and longer than the 32-mer window (incl. empty).
        for len in [0usize, 1, 5, 32, 33, 80] {
            let read: Vec<u8> = (0..len).map(|i| ((i * 7 + 1) & 0x3) as u8).collect();
            let mut buf = Vec::new();
            rolling_forward_keys_into(&read, &mut buf);
            assert_eq!(buf, rolling_forward_keys(&read), "len {len}");
        }
    }

    #[test]
    fn rolling_forward_keys_into_reuses_buffer_without_reallocating() {
        // A warmed buffer (capacity from the first, longest read) must not reallocate
        // when refilled for equal-or-shorter reads: the collect hot path relies on the
        // scratch staying allocation-free once warmed up.
        let long: Vec<u8> = (0..80).map(|i| ((i * 3 + 1) & 0x3) as u8).collect();
        let mut buf = Vec::new();
        rolling_forward_keys_into(&long, &mut buf);
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        for len in [80usize, 64, 32, 1, 0] {
            let read: Vec<u8> = (0..len).map(|i| ((i * 5 + 2) & 0x3) as u8).collect();
            rolling_forward_keys_into(&read, &mut buf);
            assert_eq!(buf.len(), len);
            assert_eq!(buf.capacity(), cap, "capacity grew at len {len}");
            assert_eq!(buf.as_ptr(), ptr, "buffer reallocated at len {len}");
        }
    }

    #[test]
    fn rolling_forward_keys_exactly_32() {
        // A read of exactly 32 bases: keys[0] is the full window, later positions
        // shrink and T-pad.
        let read: Vec<u8> = (0..32).map(|i| ((i * 7 + 1) & 0x3) as u8).collect();
        let keys = rolling_forward_keys(&read);
        for i in 0..read.len() {
            let want = tokenize_32mer(&read[i..], KMER_LEN.min(read.len() - i));
            assert_eq!(keys[i], want, "position {i}");
        }
        // keys[0] must equal a direct full-window tokenize.
        assert_eq!(keys[0], tokenize_32mer(&read, 32));
    }

    proptest! {
        /// The rolling cache is byte-identical to a fresh `tokenize_32mer` at every
        /// position, for reads both shorter and longer than the 32-mer window,
        /// including offsets right up against the read end. This is the byte-identity
        /// contract the collect hot path relies on to substitute `keys[pivot]` for a
        /// re-tokenize of `read[pivot..]`.
        #[test]
        fn rolling_forward_keys_equals_tokenize_at_every_offset(
            read in proptest::collection::vec(0u8..4, 0usize..80)
        ) {
            let keys = rolling_forward_keys(&read);
            prop_assert_eq!(keys.len(), read.len());
            for i in 0..read.len() {
                let want = tokenize_32mer(&read[i..], KMER_LEN.min(read.len() - i));
                prop_assert_eq!(keys[i], want, "offset {} of len {}", i, read.len());
            }
        }

        /// With `N` (`4`) present the build must (a) never panic and (b) stay
        /// byte-identical to `tokenize_32mer` at every position whose full 32-mer
        /// window is `N`-free — which is exactly the set the guarded collect caller
        /// consumes (`query.len() >= KMER_LEN` implies no `N` in the first 32 bases).
        /// Positions whose 32-mer window straddles an `N` are allowed to differ; the
        /// caller never reads those keys.
        #[test]
        fn rolling_forward_keys_n_safe_and_identical_on_n_free_windows(
            read in proptest::collection::vec(0u8..5, 0usize..90)
        ) {
            // (a) no panic even with N bytes.
            let keys = rolling_forward_keys(&read);
            prop_assert_eq!(keys.len(), read.len());
            let n = read.len();
            for i in 0..n {
                let window_len = KMER_LEN.min(n - i);
                // The window `read[i..i+window_len]` is what a forward query at pivot `i`
                // would tokenize. If it is N-free, the guarded caller may use keys[i]; it
                // must then equal a fresh tokenize.
                let n_free = read[i..i + window_len].iter().all(|&b| b < 4);
                // The caller only ever uses keys[i] when the query fills the window
                // (window_len == KMER_LEN); restrict the identity check to that guarded set.
                if n_free && window_len == KMER_LEN {
                    let want = tokenize_32mer(&read[i..i + window_len], window_len);
                    prop_assert_eq!(
                        keys[i], want,
                        "N-free full window at offset {} of len {} must match", i, n
                    );
                }
            }
        }
    }
}
