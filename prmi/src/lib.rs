// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! prmi — Piecewise Recursive Model Index over genomic suffix arrays.
//!
//! This commit establishes the workspace skeleton and relocates Marcus's
//! `learnedsystems/RMI` primitives into [`upstream`]. Fulcrum-authored
//! crate content (encoding, fasta, sa, sidecar, train, index, cli, FFI)
//! lands in subsequent PRs in the v0.1 stack.

#![deny(unsafe_code)]
#![warn(missing_docs)]

// Upstream code carries its own (often minimal) docs from Marcus 2020.
#[allow(missing_docs)]
pub mod upstream;
