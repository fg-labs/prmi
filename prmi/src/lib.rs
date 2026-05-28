// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! prmi — Piecewise Recursive Model Index over genomic suffix arrays.
//!
//! The full crate surface lands across the v0.1 stack. This commit
//! introduces the Fulcrum-authored utility modules
//! ([`encoding`], [`fasta`], [`sa`], [`error`], [`sidecar::magic`])
//! that the trainer, sidecar reader, and index runtime consume.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod encoding;
pub mod error;
pub mod fasta;
pub mod sa;
pub mod sidecar;
pub use error::{Error, Result};

// Upstream code carries its own (often minimal) docs from Marcus 2020.
#[allow(missing_docs)]
pub mod upstream;
