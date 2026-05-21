// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! TOML header for the sidecar. See v0.1 brief §4.1 for the spec.

use crate::error::{Error, Result};
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level sidecar metadata struct, corresponding to the `.meta` TOML file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub prmi: Prmi,
    #[serde(rename = "ref")]
    pub ref_: Ref,
    pub sa: Sa,
    pub rmi: RmiSpec,
    pub priors: Priors,
}

/// Sidecar identity and versioning fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prmi {
    pub magic: String,
    pub format_version: u32,
    pub trainer_version: String,
    pub created_utc: String,
}

/// Reference genome provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ref {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Suffix-array layout fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sa {
    pub num_entries: u64,
    pub bytes_per_entry: u8,
    pub encoding: String,
}

/// RMI architecture spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiSpec {
    pub spec: String,
    pub l2_leaf_count: u64,
    pub bit_shift: u32,
    pub max_error_bound: u64,
}

/// Prior distribution spec.
///
/// `kind` is a raw `String` rather than an enum so that unknown future values
/// surface in `Error::FormatTooNew { kind }` by name instead of failing
/// opaquely during TOML deserialisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Priors {
    /// `type` field. v0.1 ships only `"uniform"`. Captured as a raw String
    /// so unknown values reach Error::FormatTooNew by name.
    #[serde(rename = "type")]
    pub kind: String,
}

/// Canonical v0.1 prior kinds. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub(crate) const KNOWN_PRIORS_V01: &[&str] = &["uniform"];

impl Meta {
    /// Serialize this `Meta` to a pretty-printed TOML string.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Internal {
            detail: format!("toml serialize: {e}"),
        })
    }

    /// Parse and validate a `Meta` from a TOML string.
    ///
    /// File-path context is absent; errors report an empty path. Use
    /// [`read_file`] when reading from disk so errors include the path.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        Self::from_toml_str_with_file(s, None)
    }

    /// Read, parse, and validate a `Meta` from the given path.
    pub fn read_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read_to_string(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_toml_str_with_file(&bytes, Some(path))
    }

    /// Validate and write this `Meta` as TOML to the given path.
    pub fn write_file(&self, path: &Path) -> Result<()> {
        self.validate_with_file(path)?;
        let s = self.to_toml()?;
        std::fs::write(path, s).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })
    }

    fn from_toml_str_with_file(s: &str, src: Option<&Path>) -> Result<Self> {
        let file = src.map(Path::to_path_buf).unwrap_or_default();
        let parsed: Meta = toml::from_str(s).map_err(|e| Error::TomlParse {
            file: file.clone(),
            source: e,
        })?;
        parsed.validate_with_file(file.as_path())?;
        Ok(parsed)
    }

    fn validate_with_file(&self, file: &Path) -> Result<()> {
        if self.prmi.magic != META_MAGIC {
            return Err(Error::BadMagic {
                file: file.to_path_buf(),
                found: self.prmi.magic.clone(),
                expected: META_MAGIC.to_string(),
            });
        }
        if self.prmi.format_version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: self.prmi.format_version,
                expected: FORMAT_VERSION,
            });
        }
        if !KNOWN_PRIORS_V01.contains(&self.priors.kind.as_str()) {
            return Err(Error::FormatTooNew {
                kind: self.priors.kind.clone(),
            });
        }
        if self.sa.bytes_per_entry != 5 {
            return Err(Error::SizeMismatch {
                file: file.to_path_buf(),
                detail: format!(
                    "bytes_per_entry={} (v0.1 only supports 5)",
                    self.sa.bytes_per_entry
                ),
            });
        }
        if self.sa.encoding != "packed_lo8_hi32" {
            return Err(Error::UnsupportedEncoding {
                file: file.to_path_buf(),
                encoding: self.sa.encoding.clone(),
            });
        }
        Ok(())
    }
}
