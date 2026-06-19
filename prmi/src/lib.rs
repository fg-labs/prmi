// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! prmi — Piecewise Recursive Model Index for sorted-key lookup, with a
//! genomics-oriented trainer over the suffix array of a reference genome.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod cli;
pub mod encoding;
pub mod error;
pub mod fasta;
pub mod histogram;
pub mod index;
pub mod inspect;
pub mod keepset;
pub mod pac;
pub mod sa;
pub mod sidecar;
pub mod train;
pub mod verify_sa;
pub use error::{Error, Result};

// Upstream code carries its own (often minimal) docs from Marcus 2020.
#[allow(missing_docs)]
pub mod upstream;
