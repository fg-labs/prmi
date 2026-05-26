// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Runtime index over a trained sidecar.
//!
//! This commit introduces only the §4.4 lookup math ([`lookup`]), which is
//! shared by the trainer's verification pass (`train::verify`) and the
//! runtime `LearnedIndex`. The `LearnedIndex` handle, `smem_range`, the SIMD
//! inner loop, and the shared-memory loader land in PRs #5a/#5b/#5c.

/// §4.4 lookup math: parameterized over mmap-backed readers or in-memory slices.
pub mod lookup;
