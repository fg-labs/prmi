// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Error types for the prmi crate.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("i/o error on {path}: {source}")]
    Io { path: PathBuf, #[source] source: io::Error },

    #[error("invalid sidecar magic in {file}: found {found:?}, expected {expected:?}")]
    BadMagic { file: PathBuf, found: String, expected: String },

    #[error("sidecar format version {found} not supported (this crate handles version {expected})")]
    UnsupportedVersion { found: u32, expected: u32 },

    #[error("sidecar component {file} has inconsistent size: {detail}")]
    SizeMismatch { file: PathBuf, detail: String },

    #[error("unsupported encoding {encoding:?} in {file} (v0.1 only supports \"packed_lo8_hi32\")")]
    UnsupportedEncoding { file: PathBuf, encoding: String },

    #[error("companion sidecar files {file} disagree: {detail}")]
    SidecarMismatch { file: PathBuf, detail: String },

    #[error("priors type {kind:?} requires a newer prmi (format too new)")]
    FormatTooNew { kind: String },

    #[error("malformed TOML in {file}: {source}")]
    TomlParse { file: PathBuf, #[source] source: toml::de::Error },

    #[error("malformed FASTA in {file}: {detail}")]
    Fasta { file: PathBuf, detail: String },

    #[error("invalid base byte {byte:#x} at position {pos}")]
    InvalidBase { byte: u8, pos: u64 },

    #[error("suffix array construction failed: {detail}")]
    SaConstruction { detail: String },

    #[error("internal invariant violated: {detail}")]
    Internal { detail: String },
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
impl Error {
    /// Construct a `BadMagic` error with an empty file path for use in unit tests.
    pub fn bad_magic_str(found: impl Into<String>, expected: impl Into<String>) -> Self {
        Error::BadMagic {
            file: PathBuf::new(),
            found: found.into(),
            expected: expected.into(),
        }
    }
}
