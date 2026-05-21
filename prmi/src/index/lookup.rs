// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use crate::sidecar::model_file::{ModelEntry, ModelFileReader};

/// The §4.4 lookup, parameterized over readers — used by both the runtime
/// path and the trainer's verification pass.
///
/// `l1` / `l2` are the L1 leaf array and the L2 routing-layer array
/// respectively. `bit_shift = 64 - log2(l2.len())`. `sa_num` is the total
/// SA length used to clamp the final prediction.
#[inline]
pub fn lookup_core<A: Layer + ?Sized, B: Layer + ?Sized>(
    key: u64,
    l1: &A,
    l2: &B,
    bit_shift: u32,
    sa_num: u64,
) -> (u64, u64) {
    // `key >> 64` is undefined behavior on u64. The only legal case for
    // bit_shift == 64 is a single-leaf L2 (trivial routing).
    let l2_idx = if bit_shift >= 64 {
        0
    } else {
        (key >> bit_shift) as usize
    };
    let l2e = l2.entry(l2_idx);
    let mut fpred = l2e.alpha + l2e.beta * (key as f64);
    let mut err = l2e.err;

    if (err >> 63) != 0 {
        let partial_start = ((err >> 32) & 0x7fff_ffff) as usize;
        let partial_num = (err & 0xffff_ffff) as usize;
        debug_assert!(partial_num > 0);
        let local = clamp_to_int(fpred, 0.0, (partial_num - 1) as f64);
        let l1e = l1.entry(partial_start + local);
        fpred = l1e.alpha + l1e.beta * (key as f64);
        err = l1e.err;
    }

    let pos = clamp_to_int(fpred, 0.0, sa_num.saturating_sub(1) as f64) as u64;
    (pos, err)
}

#[inline]
fn clamp_to_int(v: f64, lo: f64, hi: f64) -> usize {
    if v.is_nan() {
        return lo as usize;
    }
    v.clamp(lo, hi) as usize
}

/// Layer abstraction: same lookup math runs against in-memory slices
/// (training/verify) or an mmap-backed reader (runtime).
pub trait Layer {
    fn entry(&self, i: usize) -> ModelEntry;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Layer for [ModelEntry] {
    #[inline]
    fn entry(&self, i: usize) -> ModelEntry {
        self[i]
    }
    #[inline]
    fn len(&self) -> usize {
        <[ModelEntry]>::len(self)
    }
}

impl Layer for ModelFileReader {
    #[inline]
    fn entry(&self, i: usize) -> ModelEntry {
        ModelFileReader::entry(self, i)
    }
    #[inline]
    fn len(&self) -> usize {
        ModelFileReader::len(self)
    }
}

/// In-memory entry-point used by training/verify. Passes slices straight
/// through to `lookup_core`; **no per-call allocation**. The trainer's
/// brute-force verification pass calls this once per SA entry (millions
/// of times on real genomes), so the slice path matters.
#[inline]
pub fn lookup_with_components(
    key: u64,
    l1: &[ModelEntry],
    l2: &[ModelEntry],
    bit_shift: u32,
    sa_num: u64,
) -> (u64, u64) {
    lookup_core::<[ModelEntry], [ModelEntry]>(key, l1, l2, bit_shift, sa_num)
}
