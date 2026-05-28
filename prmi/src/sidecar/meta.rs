// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! TOML header for the sidecar. See v0.1 brief §4.1 for the spec.

use crate::error::{Error, Result};
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level sidecar metadata struct, corresponding to the `.meta` TOML file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    /// Sidecar identity and version fields.
    pub prmi: Prmi,
    #[serde(rename = "ref")]
    /// Reference genome provenance.
    pub ref_: Ref,
    /// Suffix-array layout metadata.
    pub sa: Sa,
    /// RMI architecture specification.
    pub rmi: RmiSpec,
    /// Prior distribution specification.
    pub priors: Priors,
}

/// Sidecar identity and versioning fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prmi {
    /// Magic string identifying this as a prmi sidecar (e.g. `"PRMIv1"`).
    pub magic: String,
    /// On-disk format version integer.
    pub format_version: u32,
    /// Semver of the `prmi` crate that produced this sidecar.
    pub trainer_version: String,
    /// RFC 3339 UTC timestamp at which the sidecar was created.
    pub created_utc: String,
}

/// Reference genome provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ref {
    /// Filesystem path of the reference FASTA at build time.
    pub path: String,
    /// SHA-256 hex digest of the reference FASTA.
    pub sha256: String,
    /// Size of the reference FASTA in bytes.
    pub size_bytes: u64,
}

/// Suffix-array layout fields.
///
/// Note: `deny_unknown_fields` is intentionally omitted here. The `[sa]`
/// section has several `#[serde(default)]` fields for backward compatibility
/// with older sidecars. Adding `deny_unknown_fields` to a section with
/// `serde(default)` fields works correctly, but the `[sa]` section is most
/// likely to gain new optional fields in future versions (e.g., additional
/// masking metadata). Unknown fields in `[sa]` trigger `Error::FormatTooNew`
/// via the `format_version` check rather than at deserialization time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sa {
    /// Number of suffix-array entries (equals the genome length in bases).
    pub num_entries: u64,
    /// Bytes per packed SA position. Varies by memory mode:
    /// - mode 1 (default): 5
    /// - mode 2: 13 (position + 8-byte key)
    /// - mode 3: 21 (position + 8-byte key + 8-byte ISA)
    /// - suffix_key_cache: 5 (keys live in separate `.skc` file)
    pub bytes_per_entry: u8,
    /// Encoding name for the SA positions. Mode-dependent:
    /// - mode 1 / suffix_key_cache: `"packed_lo8_hi32"`
    /// - mode 2: `"packed_lo8_hi32_key64"`
    /// - mode 3: `"packed_lo8_hi32_key64_isa64"`
    pub encoding: String,
    /// Memory mode for this sidecar. One of `"1"`, `"2"`, `"3"`, or
    /// `"suffix_key_cache"`. Defaults to `"1"` for backward compatibility
    /// with sidecars built before the memory-mode menu was introduced.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// For `mode = "suffix_key_cache"`: number of (sa_index, key) pairs
    /// cached in the companion `.skc` file. `None` for all other modes.
    #[serde(default)]
    pub skc_cache_size: Option<u64>,
    /// Which strand the suffix array was built over. v0.1 supports only
    /// `"forward_only"` (no reverse-complement concatenation). Defaults to
    /// `"forward_only"` when absent so that sidecars built before this field
    /// was added still load cleanly.
    #[serde(default = "default_strand")]
    pub strand: String,
    /// Whether N-run positions were excluded from the training set.
    /// `true` means any SA position whose 32-mer window contained at least
    /// one N base was skipped during model fitting.
    /// Defaults to `false` for backward compatibility with older sidecars.
    #[serde(default)]
    pub masked_n_runs: bool,
    /// If `Some(k)`, SA positions whose 32-mer window contained a homopolymer
    /// run of length >= `k` were excluded from the training set.
    /// `None` means homopolymer masking was not applied.
    #[serde(default)]
    pub masked_homopolymers: Option<u32>,
    /// Filesystem path of the BED file used to mask training positions, if any.
    /// `None` means no BED masking was applied.
    #[serde(default)]
    pub masked_bed: Option<String>,
}

/// Default value for [`Sa::mode`] when deserialising older sidecars that
/// pre-date the memory-mode menu. Mode 1 (position-only) is the historic
/// and only mode for such sidecars.
fn default_mode() -> String {
    "1".to_string()
}

/// Default value for [`Sa::strand`] when deserialising older sidecars that
/// pre-date the field.
fn default_strand() -> String {
    "forward_only".to_string()
}

/// RMI architecture spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiSpec {
    /// RMI architecture string (e.g. `"linear,linear"`).
    pub spec: String,
    /// Number of L2 leaf nodes (must be a power of two).
    pub l2_leaf_count: u64,
    /// Bit-shift used to route a key into the L2 layer: `64 - log2(l2_leaf_count)`.
    pub bit_shift: u32,
    /// Global maximum prediction error across all keys in the training set.
    pub max_error_bound: u64,
}

/// Prior distribution spec.
///
/// `kind` is a raw `String` rather than an enum so that unknown future values
/// surface in `Error::FormatTooNew { kind }` by name instead of failing
/// opaquely during TOML deserialisation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Priors {
    /// `type` field. v0.1 ships `"uniform"`, `"bed"`, and `"fastq_histogram"`.
    /// Captured as a raw String so unknown values reach Error::FormatTooNew by name.
    #[serde(rename = "type")]
    pub kind: String,
    /// Filesystem path of the BED file used as a training-weight prior.
    /// Present only when `type = "bed"`. Defaults to `None` for backward
    /// compatibility with sidecars built before this field was added.
    #[serde(default)]
    pub bed: Option<String>,
    /// Weight multiplier applied to in-BED training pairs during model fit.
    /// Present only when `type = "bed"`. Defaults to `None` for backward
    /// compatibility.
    #[serde(default)]
    pub weight: Option<f64>,
    /// Filesystem path of the FASTQ k-mer frequency histogram TSV used as a
    /// training-weight prior. Present only when `type = "fastq_histogram"`.
    /// Defaults to `None` for backward compatibility.
    #[serde(default)]
    pub histogram: Option<String>,
    /// Base weight for k-mers absent from the histogram.
    /// Present only when `type = "fastq_histogram"`. Defaults to `None` for
    /// backward compatibility (readers should treat absent as 1.0).
    #[serde(default)]
    pub base_weight: Option<f64>,
    /// Weight formula string, recorded for provenance. Defaults to `None` for
    /// backward compatibility.
    ///
    /// v0.1 records `"1.0 + log2(1 + freq)"` for `type = "fastq_histogram"`.
    #[serde(default)]
    pub formula: Option<String>,
}

/// Canonical v0.1 prior kinds. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub(crate) const KNOWN_PRIORS_V01: &[&str] = &["uniform", "bed", "fastq_histogram"];

/// Canonical v0.1 SA strand values. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub(crate) const KNOWN_STRANDS_V01: &[&str] = &["forward_only"];

/// Canonical v0.1 SA mode values. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub(crate) const KNOWN_MODES_V01: &[&str] = &["1", "2", "3", "suffix_key_cache"];

/// Expected `bytes_per_entry` for each mode. Must be consistent with
/// [`crate::train::config::MemoryMode::bytes_per_entry`].
///
/// Returns `None` for an unknown mode (which would already have been rejected
/// by the `KNOWN_MODES_V01` check before this is called).
pub(crate) fn expected_bytes_per_entry_for_mode(mode: &str) -> Option<u8> {
    match mode {
        "1" | "suffix_key_cache" => Some(5),
        "2" => Some(13),
        "3" => Some(21),
        _ => None,
    }
}

/// Expected `encoding` name for each mode.
pub(crate) fn expected_encoding_for_mode(mode: &str) -> Option<&'static str> {
    match mode {
        "1" | "suffix_key_cache" => Some("packed_lo8_hi32"),
        "2" => Some("packed_lo8_hi32_key64"),
        "3" => Some("packed_lo8_hi32_key64_isa64"),
        _ => None,
    }
}

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
    /// [`Meta::read_file`] when reading from disk so errors include the path.
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
        if !KNOWN_STRANDS_V01.contains(&self.sa.strand.as_str()) {
            return Err(Error::FormatTooNew {
                kind: format!("sa.strand={}", self.sa.strand),
            });
        }
        // Validate mode.
        if !KNOWN_MODES_V01.contains(&self.sa.mode.as_str()) {
            return Err(Error::FormatTooNew {
                kind: format!("sa.mode={}", self.sa.mode),
            });
        }
        // Validate skc_cache_size consistency with mode. The companion `.skc`
        // file (and hence a cache size) is meaningful only for the
        // `suffix_key_cache` mode; every other mode must leave it unset.
        match (self.sa.mode.as_str(), self.sa.skc_cache_size) {
            ("suffix_key_cache", None) => {
                return Err(Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: "skc_cache_size is required for mode=suffix_key_cache".to_string(),
                });
            }
            ("suffix_key_cache", Some(0)) => {
                return Err(Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: "skc_cache_size must be > 0 for mode=suffix_key_cache".to_string(),
                });
            }
            (mode, Some(_)) if mode != "suffix_key_cache" => {
                return Err(Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: format!("skc_cache_size must be unset for mode={mode}"),
                });
            }
            _ => {}
        }
        // Validate bytes_per_entry consistency with mode.
        let expected_bpe =
            expected_bytes_per_entry_for_mode(&self.sa.mode).expect("mode already validated");
        if self.sa.bytes_per_entry != expected_bpe {
            return Err(Error::SizeMismatch {
                file: file.to_path_buf(),
                detail: format!(
                    "bytes_per_entry={} inconsistent with mode={} (expected {})",
                    self.sa.bytes_per_entry, self.sa.mode, expected_bpe
                ),
            });
        }
        // Validate encoding consistency with mode.
        let expected_enc =
            expected_encoding_for_mode(&self.sa.mode).expect("mode already validated");
        if self.sa.encoding != expected_enc {
            return Err(Error::UnsupportedEncoding {
                file: file.to_path_buf(),
                encoding: self.sa.encoding.clone(),
            });
        }
        Ok(())
    }
}
