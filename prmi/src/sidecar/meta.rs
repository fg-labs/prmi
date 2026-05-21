// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! TOML header for the sidecar. See v0.1 brief §4.1 for the spec.

use crate::error::{Error, Result};
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub prmi: Prmi,
    #[serde(rename = "ref")]
    pub ref_: Ref,
    pub sa: Sa,
    pub rmi: RmiSpec,
    pub priors: Priors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prmi {
    pub magic: String,
    pub format_version: u32,
    pub trainer_version: String,
    pub created_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ref {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sa {
    pub num_entries: u64,
    pub bytes_per_entry: u8,
    pub encoding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiSpec {
    pub spec: String,
    pub l2_leaf_count: u64,
    pub bit_shift: u32,
    pub max_error_bound: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Priors {
    /// `type` field. v0.1 ships only `"uniform"`. Captured as a raw String
    /// so unknown values reach Error::FormatTooNew by name.
    #[serde(rename = "type")]
    pub kind: String,
}

/// Canonical v0.1 prior kinds. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub const KNOWN_PRIORS_V01: &[&str] = &["uniform"];

impl Meta {
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Internal {
            detail: format!("toml serialize: {e}"),
        })
    }

    pub fn from_toml_str(s: &str) -> Result<Self> {
        let parsed: Meta = toml::from_str(s).map_err(|e| Error::TomlParse {
            file: std::path::PathBuf::new(),
            source: e,
        })?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn read_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read_to_string(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_toml_str(&bytes)
    }

    pub fn write_file(&self, path: &Path) -> Result<()> {
        let s = self.to_toml()?;
        std::fs::write(path, s).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.prmi.magic != META_MAGIC {
            return Err(Error::BadMagic {
                file: std::path::PathBuf::new(),
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
            return Err(Error::FormatTooNew { kind: self.priors.kind.clone() });
        }
        if self.sa.bytes_per_entry != 5 {
            return Err(Error::SizeMismatch {
                file: std::path::PathBuf::new(),
                detail: format!("bytes_per_entry={} (v0.1 only supports 5)", self.sa.bytes_per_entry),
            });
        }
        Ok(())
    }
}
