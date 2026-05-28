// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Error types for the prmi crate.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Error variants returned by all fallible prmi operations.
#[derive(Debug, Error)]
pub enum Error {
    /// An I/O operation failed on the given path.
    #[error("i/o error on {path}: {source}")]
    Io {
        /// File path involved in the failed I/O operation.
        path: PathBuf,
        #[source]
        /// Underlying I/O error.
        source: io::Error,
    },

    /// A sidecar file had an unexpected magic string.
    #[error("invalid sidecar magic in {file}: found {found:?}, expected {expected:?}")]
    BadMagic {
        /// File that contained the bad magic value.
        file: PathBuf,
        /// Magic value that was found.
        found: String,
        /// Magic value that was expected.
        expected: String,
    },

    /// The sidecar's on-disk format version is not supported by this build.
    #[error(
        "sidecar format version {found} not supported (this crate handles version {expected})"
    )]
    UnsupportedVersion {
        /// Version found in the file.
        found: u32,
        /// Version this crate supports.
        expected: u32,
    },

    /// A sidecar component's on-disk size disagrees with its header.
    #[error("sidecar component {file} has inconsistent size: {detail}")]
    SizeMismatch {
        /// File with the inconsistent size.
        file: PathBuf,
        /// Human-readable description of the mismatch.
        detail: String,
    },

    /// The `.sa` file uses an encoding the v0.1 reader does not support.
    #[error(
        "unsupported encoding {encoding:?} in {file} (v0.1 only supports \"packed_lo8_hi32\")"
    )]
    UnsupportedEncoding {
        /// File that contains the unsupported encoding.
        file: PathBuf,
        /// Encoding string found in the file.
        encoding: String,
    },

    /// Two companion sidecar files contain inconsistent metadata.
    #[error("companion sidecar files {file} disagree: {detail}")]
    SidecarMismatch {
        /// File where the mismatch was detected.
        file: PathBuf,
        /// Human-readable description of the inconsistency.
        detail: String,
    },

    /// The priors `type` field names a kind introduced by a newer prmi version.
    #[error("priors type {kind:?} requires a newer prmi (format too new)")]
    FormatTooNew {
        /// The unrecognised priors kind string.
        kind: String,
    },

    /// The `.meta` file contains malformed TOML.
    #[error("malformed TOML in {file}: {source}")]
    TomlParse {
        /// File that could not be parsed.
        file: PathBuf,
        #[source]
        /// TOML parse error.
        source: toml::de::Error,
    },

    /// The reference FASTA could not be parsed.
    #[error("malformed FASTA in {file}: {detail}")]
    Fasta {
        /// Path to the malformed FASTA.
        file: PathBuf,
        /// Human-readable detail about the parse failure.
        detail: String,
    },

    /// A sequence byte was not A/C/G/T and could not be encoded.
    #[error("invalid base byte {byte:#x} at position {pos}")]
    InvalidBase {
        /// The offending byte value.
        byte: u8,
        /// 0-based position in the concatenated sequence.
        pos: u64,
    },

    /// The underlying `libsais` call to build the suffix array failed.
    #[error("suffix array construction failed: {detail}")]
    SaConstruction {
        /// Detail message from the SA builder.
        detail: String,
    },

    /// An internal invariant was violated — always indicates a bug.
    #[error("internal invariant violated: {detail}")]
    Internal {
        /// Description of the violated invariant.
        detail: String,
    },
}

/// Convenience alias: `Result<T>` is `std::result::Result<T, Error>`.
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
