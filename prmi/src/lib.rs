// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! prmi — Piecewise Recursive Model Index over genomic suffix arrays.
//!
//! This commit introduces the cleanroom trainer ([`train`]) and the shared
//! §4.4 lookup math ([`index::lookup`]). The runtime `LearnedIndex`, priors,
//! memory modes, CLI, and FFI land in subsequent PRs.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod encoding;
pub mod error;
pub mod fasta;
pub mod index;
pub mod sa;
pub mod sidecar;
pub mod train;
pub use error::{Error, Result};

// Upstream code carries its own (often minimal) docs from Marcus 2020.
#[allow(missing_docs)]
pub mod upstream;
