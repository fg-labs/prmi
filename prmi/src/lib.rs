// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

// unsafe_code is deny (not forbid) so that the single unsafe island in
// sidecar::sa_file (mmap) can opt in with #[allow(unsafe_code)].
#![deny(unsafe_code)]

//! prmi — Piecewise Recursive Model Index for sorted-key lookup, with a
//! genomics-oriented trainer over the suffix array of a reference genome.
//!
//! See `docs/superpowers/handoff/2026-05-20-prmi-v0.1-brief.md` for the
//! v0.1 sidecar format spec and the C ABI contract.

pub mod cli;
pub mod encoding;
pub mod error;
pub mod fasta;
pub mod index;
pub mod sa;
pub mod sidecar;
pub mod train;
pub use error::{Error, Result};

pub mod upstream;
