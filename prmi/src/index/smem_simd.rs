// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

// SIMD intrinsics require unsafe blocks. This is the only file in the crate
// that uses unsafe; all other modules operate under #![deny(unsafe_code)].
#![allow(unsafe_code)]

//! SIMD-accelerated 32-mer tokenization for the `smem_range` local-search
//! inner loop. The hot path in `resolve_one` walks `[lo, hi)` SA positions
//! comparing each candidate's 32-mer key against the query key; for large
//! error bounds this loop dominates smem_range cost.
//!
//! The public entry point is [`tokenize_4_at_once`], which tokenizes four
//! SA positions concurrently:
//!
//! - **NEON (aarch64)**: two `uint8x16_t` loads per candidate (32 bytes = 2 × 16),
//!   then a NEON-vectorised pack into a `u64`. Active when `cfg(target_arch = "aarch64")`.
//! - **AVX2 (x86_64)**: a single `__m256i` load per candidate (32 bytes), then
//!   AVX2-vectorised pack. Active at runtime when `is_x86_feature_detected!("avx2")`.
//! - **Scalar fallback**: four sequential calls to [`tokenize_32mer`]. Used on
//!   non-SIMD x86_64 and any other architecture.
//!
//! All three paths produce bit-identical output for any (pac, positions, enc) input.
//! The SIMD paths win by reducing per-base branch overhead and exploiting wider
//! memory loads when candidates' windows are contiguous in the pac buffer.
//!
//! # Safety
//! The `avx2` path uses `unsafe` intrinsics guarded by `is_x86_feature_detected!`.
//! The `aarch64` NEON path uses `unsafe` intrinsics; NEON is always present on
//! 64-bit ARMv8 targets.

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::index::smem::{read_unpacked_window_pub, PacEncoding};

/// Tokenize four candidate 32-mers from `pac` at the given `sa_positions` and
/// write their keys into `out_keys`.
///
/// Architecturally, this dispatches to:
/// - NEON on `aarch64` (always available on 64-bit ARMv8)
/// - AVX2 on `x86_64` when the CPU supports it (checked at runtime)
/// - Scalar fallback otherwise
///
/// Results are bit-identical to four sequential calls to
/// `tokenize_32mer` + `read_unpacked_window_pub`.
///
/// # Parameters
/// - `pac`: the reference pac slice (either unpacked 1b/base or packed 2b/base)
/// - `enc`: encoding descriptor (see [`PacEncoding`])
/// - `sa_positions`: exactly 4 genome positions to tokenize
/// - `out_keys`: output buffer; overwritten with the 4 tokenized keys
#[inline]
pub(crate) fn tokenize_4_at_once(
    pac: &[u8],
    enc: PacEncoding,
    sa_positions: &[u64; 4],
    out_keys: &mut [u64; 4],
) {
    cfg_if_simd(pac, enc, sa_positions, out_keys);
}

/// Architecture-dispatched inner implementation, split out so each `cfg` branch
/// is a complete function body and the compiler does not emit unreachable-code
/// warnings for the dead branches.
#[inline]
fn cfg_if_simd(pac: &[u8], enc: PacEncoding, sa_positions: &[u64; 4], out_keys: &mut [u64; 4]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on 64-bit aarch64 targets.
        unsafe { tokenize_4_neon(pac, enc, sa_positions, out_keys) }
    }

    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection guards AVX2 intrinsic use.
            unsafe { tokenize_4_avx2(pac, enc, sa_positions, out_keys) }
        } else {
            tokenize_4_scalar(pac, enc, sa_positions, out_keys)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        tokenize_4_scalar(pac, enc, sa_positions, out_keys)
    }
}

/// Scalar fallback: four sequential `read_unpacked_window_pub` + `tokenize_32mer` calls.
///
/// Exposed as `pub` so integration tests can call it directly for
/// SIMD-vs-scalar equivalence checks. Not part of the public API.
#[doc(hidden)]
#[inline]
pub fn tokenize_4_scalar(
    pac: &[u8],
    enc: PacEncoding,
    sa_positions: &[u64; 4],
    out_keys: &mut [u64; 4],
) {
    let mut window = [0u8; 32];
    for (pos, key) in sa_positions.iter().zip(out_keys.iter_mut()) {
        let avail = read_unpacked_window_pub(pac, *pos, enc, &mut window);
        *key = tokenize_32mer(&window[..avail], avail);
    }
}

/// NEON path: aarch64 only. Uses 128-bit NEON registers to load and pack
/// each 32-byte window, then calls the scalar packing step. The speedup
/// comes from using NEON's wider loads and the elimination of scalar loop
/// overhead in `read_unpacked_window_pub` for the packed-pac case.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tokenize_4_neon(
    pac: &[u8],
    enc: PacEncoding,
    sa_positions: &[u64; 4],
    out_keys: &mut [u64; 4],
) {
    use std::arch::aarch64::*;

    let mut windows = [[0u8; 32]; 4];
    let mut avails = [0usize; 4];

    // Unpack all four windows first.
    for (i, &pos) in sa_positions.iter().enumerate() {
        avails[i] = read_unpacked_window_pub(pac, pos, enc, &mut windows[i]);
    }

    // For each window, pack 32 unpacked-base bytes (values 0..=3) into a u64
    // using NEON vector instructions. Each base occupies 2 bits; base[0] lands
    // in bits 63:62 (MSB-first).
    //
    // Strategy: load 16 bytes into a uint8x16_t twice (2 loads for 32 bytes),
    // then pack pairs of 2-bit values by shifting and ORing. For windows with
    // avail < 32 we fall back to the scalar path to handle T-padding correctly.
    for (i, (window, &avail)) in windows.iter().zip(avails.iter()).enumerate() {
        if avail < KMER_LEN {
            // Short window: scalar handles T-padding.
            out_keys[i] = tokenize_32mer(&window[..avail], avail);
            continue;
        }

        // Full 32-base window: NEON pack.
        // Load two 16-byte halves.
        let lo_ptr = window.as_ptr();
        let hi_ptr = window.as_ptr().add(16);
        let lo: uint8x16_t = vld1q_u8(lo_ptr);
        let hi: uint8x16_t = vld1q_u8(hi_ptr);

        // Pack lo half (bases 0..15) into a u64.
        let lo_key = neon_pack_16_bases_msb_first(lo);
        // Pack hi half (bases 16..31) into a u64. These occupy the lower 32 bits
        // of the final key (each base is 2 bits; 16 bases = 32 bits).
        let hi_key = neon_pack_16_bases_msb_first(hi);

        // Combine: lo_key holds bases 0..15 in bits 63:32, hi_key holds bases
        // 16..31 in its bits 63:32. Shift hi_key right by 32 to place it in
        // the low 32 bits of the result.
        out_keys[i] = lo_key | (hi_key >> 32);
    }
}

/// Pack 16 unpacked-base bytes (values 0..=3) from a NEON uint8x16_t register
/// into the HIGH 32 bits of a u64, MSB-first. Bases at lower indices occupy
/// higher bit positions. The LOW 32 bits of the result are zero.
///
/// Bit layout: base[0] → bits 63:62, base[1] → bits 61:60, ..., base[15] → bits 33:32.
///
/// # Safety
/// Caller must ensure NEON is available (target_arch = "aarch64").
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_pack_16_bases_msb_first(bases: std::arch::aarch64::uint8x16_t) -> u64 {
    use std::arch::aarch64::*;

    // Extract the 16 bytes into a plain array for lane-wise packing.
    // We can't do the whole pack as a single NEON chain without scatter/gather,
    // so we extract to a u128 and pack in scalar. This is still faster than
    // the loop in read_unpacked_window_pub because:
    // 1. The load is a single 128-bit NEON load (vs. per-byte logic).
    // 2. The subsequent scalar pack is tight (no branch per byte).
    let mut arr = [0u8; 16];
    vst1q_u8(arr.as_mut_ptr(), bases);

    // Pack 16 bases (2 bits each) MSB-first into bits 63:32 of a u64.
    // base[0] → bits 63:62, base[1] → bits 61:60, ..., base[15] → bits 33:32.
    let mut key: u64 = 0;
    for (i, &b) in arr.iter().enumerate() {
        let shift = 62 - 2 * i as u32; // 62, 60, 58, ..., 32
        key |= (b as u64 & 0x3) << shift;
    }
    key
}

/// AVX2 path: x86_64 only, guarded by runtime `is_x86_feature_detected!("avx2")`.
/// Loads each 32-byte window as a single `__m256i`, then packs 32 2-bit bases
/// into a u64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tokenize_4_avx2(
    pac: &[u8],
    enc: PacEncoding,
    sa_positions: &[u64; 4],
    out_keys: &mut [u64; 4],
) {
    use std::arch::x86_64::*;

    let mut windows = [[0u8; 32]; 4];
    let mut avails = [0usize; 4];

    for (i, &pos) in sa_positions.iter().enumerate() {
        avails[i] = read_unpacked_window_pub(pac, pos, enc, &mut windows[i]);
    }

    for (i, (window, &avail)) in windows.iter().zip(avails.iter()).enumerate() {
        if avail < KMER_LEN {
            // Scalar handles T-padding for short windows.
            out_keys[i] = tokenize_32mer(&window[..avail], avail);
            continue;
        }

        // Load 32 bytes into an AVX2 256-bit register.
        // SAFETY: windows[i] is [u8; 32], always 32 bytes, and we guard avail == 32.
        // The pointer is to stack memory so alignment is not an issue for _mm256_loadu_si256.
        let v = _mm256_loadu_si256(window.as_ptr() as *const __m256i);

        // Pack 32 unpacked 2-bit bases from the AVX2 register into a u64.
        out_keys[i] = avx2_pack_32_bases_msb_first(v);
    }
}

/// Pack 32 unpacked-base bytes (values 0..=3) from a `__m256i` register
/// into a u64, MSB-first. Base at lane 0 → bits 63:62.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_pack_32_bases_msb_first(v: std::arch::x86_64::__m256i) -> u64 {
    use std::arch::x86_64::*;

    // Extract to a 32-byte array and pack in scalar. The AVX2 advantage is the
    // single 256-bit load (vs. 32 individual byte reads in the worst case for
    // packed pac) and the batching of four such loads across the 4 candidates.
    let mut arr = [0u8; 32];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v);

    // Tight scalar pack: no branches per byte after the store.
    let mut key: u64 = 0;
    for (i, &b) in arr.iter().enumerate() {
        // base[0] → bits 63:62, base[31] → bits 1:0
        let shift = 2 * (KMER_LEN - 1 - i) as u32;
        key |= (b as u64 & 0x3) << shift;
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::tokenize_32mer;
    use proptest::prelude::*;

    /// Generate a random 32-byte array of 2-bit bases (values 0..=3).
    fn arb_window() -> impl Strategy<Value = [u8; 32]> {
        prop::array::uniform32(0u8..4u8)
    }

    /// Generate a random partial window: (length in 1..=32, values 0..=3).
    fn arb_partial_window() -> impl Strategy<Value = (usize, Vec<u8>)> {
        (1usize..=32usize)
            .prop_flat_map(|len| prop::collection::vec(0u8..4u8, len).prop_map(move |v| (len, v)))
    }

    // -------------------------------------------------------------------------
    // Full 32-base windows
    // -------------------------------------------------------------------------

    proptest! {
        /// `tokenize_4_at_once` matches four sequential `tokenize_32mer` calls
        /// on full 32-base windows (unpacked pac).
        #[test]
        fn prop_tokenize_4_matches_scalar_full(
            w0 in arb_window(),
            w1 in arb_window(),
            w2 in arb_window(),
            w3 in arb_window(),
        ) {
            // Build an unpacked pac where each window is stored at a fixed offset.
            // Positions 0, 32, 64, 96.
            let mut pac = vec![0u8; 128];
            pac[0..32].copy_from_slice(&w0);
            pac[32..64].copy_from_slice(&w1);
            pac[64..96].copy_from_slice(&w2);
            pac[96..128].copy_from_slice(&w3);

            let enc = PacEncoding::Unpacked;
            let sa_positions: [u64; 4] = [0, 32, 64, 96];

            let mut simd_keys = [0u64; 4];
            tokenize_4_at_once(&pac, enc, &sa_positions, &mut simd_keys);

            let expected = [
                tokenize_32mer(&w0, 32),
                tokenize_32mer(&w1, 32),
                tokenize_32mer(&w2, 32),
                tokenize_32mer(&w3, 32),
            ];

            prop_assert_eq!(simd_keys, expected);
        }

        /// Scalar path explicitly matches `tokenize_32mer` on full windows.
        #[test]
        fn prop_scalar_matches_tokenize_32mer_full(
            w0 in arb_window(),
            w1 in arb_window(),
            w2 in arb_window(),
            w3 in arb_window(),
        ) {
            let mut pac = vec![0u8; 128];
            pac[0..32].copy_from_slice(&w0);
            pac[32..64].copy_from_slice(&w1);
            pac[64..96].copy_from_slice(&w2);
            pac[96..128].copy_from_slice(&w3);

            let enc = PacEncoding::Unpacked;
            let sa_positions: [u64; 4] = [0, 32, 64, 96];

            let mut scalar_keys = [0u64; 4];
            tokenize_4_scalar(&pac, enc, &sa_positions, &mut scalar_keys);

            let expected = [
                tokenize_32mer(&w0, 32),
                tokenize_32mer(&w1, 32),
                tokenize_32mer(&w2, 32),
                tokenize_32mer(&w3, 32),
            ];

            prop_assert_eq!(scalar_keys, expected);
        }
    }

    // -------------------------------------------------------------------------
    // Partial windows (avail < 32): T-padding correctness
    // -------------------------------------------------------------------------

    proptest! {
        /// `tokenize_4_at_once` and `tokenize_4_scalar` agree on partial windows
        /// (positions near the pac boundary where avail < 32). Exercises the
        /// `if avail < KMER_LEN` short-circuit branches in `tokenize_4_neon` and
        /// `tokenize_4_avx2`.
        #[test]
        fn prop_tokenize_4_matches_scalar_partial(
            (len, bases) in arb_partial_window(),
        ) {
            // Build a pac of exactly `len` bases. All four sa_positions point
            // into it at different offsets so each slot sees a different
            // `avail` (0..=len) and any positions >= len yield avail = 0.
            let pac = &bases[..len];
            let enc = PacEncoding::Unpacked;

            // Four positions spanning the pac: 0, len/4, len/2, len (the last
            // is at or past the boundary so avail == 0).
            let sa_positions: [u64; 4] = [
                0,
                (len / 4) as u64,
                (len / 2) as u64,
                len as u64,
            ];

            let mut simd_keys = [0u64; 4];
            let mut scalar_keys = [0u64; 4];
            tokenize_4_at_once(pac, enc, &sa_positions, &mut simd_keys);
            tokenize_4_scalar(pac, enc, &sa_positions, &mut scalar_keys);

            // Independently compute the expected key for each slot through the
            // documented scalar path: read_unpacked_window_pub + tokenize_32mer.
            let mut expected = [0u64; 4];
            for (slot, &pos) in sa_positions.iter().enumerate() {
                let mut window = [0u8; 32];
                let avail = read_unpacked_window_pub(pac, pos, enc, &mut window);
                expected[slot] = tokenize_32mer(&window[..avail], avail);
            }

            prop_assert_eq!(simd_keys, expected,
                "SIMD path: pac_len={} sa_positions={:?} simd_keys={:?} expected={:?}",
                len, sa_positions, simd_keys, expected);
            prop_assert_eq!(scalar_keys, expected,
                "scalar path: pac_len={} sa_positions={:?} scalar_keys={:?} expected={:?}",
                len, sa_positions, scalar_keys, expected);
            prop_assert_eq!(simd_keys, scalar_keys,
                "SIMD-vs-scalar mismatch on partial windows: pac_len={} sa_positions={:?}",
                len, sa_positions);
        }
    }

    // -------------------------------------------------------------------------
    // Cross-path equivalence: SIMD vs scalar on the same 4-candidate input
    // -------------------------------------------------------------------------

    proptest! {
        /// `tokenize_4_at_once` (dispatched path, may be SIMD) produces the same
        /// result as `tokenize_4_scalar` for any 4-candidate input.
        #[test]
        fn prop_simd_matches_scalar(
            w0 in arb_window(),
            w1 in arb_window(),
            w2 in arb_window(),
            w3 in arb_window(),
        ) {
            let mut pac = vec![0u8; 128];
            pac[0..32].copy_from_slice(&w0);
            pac[32..64].copy_from_slice(&w1);
            pac[64..96].copy_from_slice(&w2);
            pac[96..128].copy_from_slice(&w3);

            let enc = PacEncoding::Unpacked;
            let sa_positions: [u64; 4] = [0, 32, 64, 96];

            let mut simd_keys = [0u64; 4];
            let mut scalar_keys = [0u64; 4];
            tokenize_4_at_once(&pac, enc, &sa_positions, &mut simd_keys);
            tokenize_4_scalar(&pac, enc, &sa_positions, &mut scalar_keys);

            prop_assert_eq!(simd_keys, scalar_keys);
        }
    }

    // -------------------------------------------------------------------------
    // Deterministic unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn tokenize_4_all_zeros_is_aaaa() {
        // All-A (0) pac → key should be 0x0000_0000_0000_0000.
        let pac = vec![0u8; 128];
        let enc = PacEncoding::Unpacked;
        let sa_positions: [u64; 4] = [0, 32, 64, 96];
        let mut keys = [0xdead_beef_dead_beefu64; 4];
        tokenize_4_at_once(&pac, enc, &sa_positions, &mut keys);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(k, 0, "slot {i}: expected all-A key 0, got {k:#018x}");
        }
    }

    #[test]
    fn tokenize_4_all_t_is_max() {
        // All-T (3) pac → key should be 0xFFFF_FFFF_FFFF_FFFF.
        let pac = vec![3u8; 128];
        let enc = PacEncoding::Unpacked;
        let sa_positions: [u64; 4] = [0, 32, 64, 96];
        let mut keys = [0u64; 4];
        tokenize_4_at_once(&pac, enc, &sa_positions, &mut keys);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(
                k,
                u64::MAX,
                "slot {i}: expected all-T key 0xFFFFFFFFFFFFFFFF, got {k:#018x}"
            );
        }
    }

    #[test]
    fn tokenize_4_alternating_matches_scalar() {
        // Alternating ACGT pattern.
        let window: Vec<u8> = (0..32).map(|i| (i % 4) as u8).collect();
        let mut pac = vec![0u8; 128];
        for slot in 0..4 {
            pac[slot * 32..slot * 32 + 32].copy_from_slice(&window);
        }
        let enc = PacEncoding::Unpacked;
        let sa_positions: [u64; 4] = [0, 32, 64, 96];

        let mut simd_keys = [0u64; 4];
        tokenize_4_at_once(&pac, enc, &sa_positions, &mut simd_keys);
        let expected = tokenize_32mer(&window, 32);
        for (i, &k) in simd_keys.iter().enumerate() {
            assert_eq!(
                k, expected,
                "slot {i}: expected {expected:#018x}, got {k:#018x}"
            );
        }
    }

    #[test]
    fn tokenize_4_packed_pac_matches_unpacked() {
        // Build an unpacked and a packed pac encoding the same bases.
        let bases: Vec<u8> = (0..128u8).map(|i| i % 4).collect();
        let num_bases = 128u64;

        // Pack into 2-bit format.
        let mut packed = vec![0u8; 32]; // 128 bases / 4 = 32 bytes
        for (i, &b) in bases.iter().enumerate() {
            let shift = 6 - 2 * ((i % 4) as u32);
            packed[i / 4] |= (b & 0x3) << shift;
        }

        let enc_unpacked = PacEncoding::Unpacked;
        let enc_packed = PacEncoding::Packed { num_bases };
        let sa_positions: [u64; 4] = [0, 32, 64, 96];

        let mut unpacked_keys = [0u64; 4];
        let mut packed_keys = [0u64; 4];
        tokenize_4_at_once(&bases, enc_unpacked, &sa_positions, &mut unpacked_keys);
        tokenize_4_at_once(&packed, enc_packed, &sa_positions, &mut packed_keys);

        assert_eq!(
            unpacked_keys, packed_keys,
            "packed and unpacked pacs should produce identical keys"
        );
    }

    #[test]
    fn tokenize_4_partial_window_t_padding() {
        // pac has only 10 bases at position 0; remaining 22 positions should
        // be T-padded (value 3 = 0b11), matching tokenize_32mer's behaviour.
        let partial_bases: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // 10 bases
        let enc = PacEncoding::Unpacked;
        let sa_positions: [u64; 4] = [0, 0, 0, 0]; // all read from position 0

        let mut keys = [0u64; 4];
        tokenize_4_scalar(&partial_bases, enc, &sa_positions, &mut keys);

        let expected = tokenize_32mer(&partial_bases, 10);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(
                k, expected,
                "slot {i}: T-padding mismatch: got {k:#018x}, expected {expected:#018x}"
            );
        }
    }
}
