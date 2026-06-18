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
    /// Magic string identifying this as a prmi sidecar (e.g. `"PRMIv2"`).
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
    pub bytes_per_entry: u8,
    /// Encoding name for the SA positions. Mode-dependent:
    /// - mode 1: `"packed_lo8_hi32"`
    /// - mode 2: `"packed_lo8_hi32_key64"`
    pub encoding: String,
    /// Memory mode for this sidecar. One of `"1"` or `"2"`. Defaults to `"1"`
    /// for backward compatibility with sidecars built before the memory-mode
    /// menu was introduced.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Legacy field retained for backward-compatible deserialization of older
    /// sidecars. No longer written by the trainer; always `None` for current
    /// builds.
    #[serde(default)]
    pub skc_cache_size: Option<u64>,
    /// Which strand the suffix array was built over. v2 requires
    /// `"forward_rc_2x"` (forward + reverse-complement concatenation).
    /// The `#[serde(default)]` falls back to `"forward_only"` only so that
    /// pre-v2 sidecars deserialize without error — they are then rejected by
    /// validation.
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
    /// Number of forward reference bases (`l_pac`). The 2× SA has
    /// `2*l_pac + 1` entries. Provenance + verification only; prmi does not
    /// use this at query time. `None` for sidecars built before v2.
    #[serde(default)]
    pub l_pac: Option<u64>,
    /// Whether the `.sa` stores a 32-mer key alongside each position
    /// (`--store-keys`; 13 B vs 5 B per entry). `None` for pre-v2 sidecars.
    #[serde(default)]
    pub stored_keys: Option<bool>,
    /// Whether this is a tiered (position-filtered, Design Z) `.sa` that retains
    /// only a keep-set subset of suffix positions. When `Some(true)`,
    /// `num_entries` is legitimately `< 2*l_pac+1` while `l_pac` stays the full
    /// forward genome length (positions remain native genome coordinates).
    /// `None`/`Some(false)` => a full 2× SA (`num_entries == 2*l_pac+1`).
    #[serde(default)]
    pub tiered: Option<bool>,
    /// Filesystem path of the keep-set BED used to build the tiered (Design Z)
    /// `.sa`, recorded so a tiered sidecar can identify which keep-set produced
    /// the position-filtered suffix array. `None` for full (non-tiered) builds.
    #[serde(default)]
    pub keep_bed: Option<String>,
    /// SHA-256 (hex) of the bwa `.pac` consumed, for `.pac`-sourced builds.
    /// `None` for FASTA-sourced (non-byte-identical) builds.
    #[serde(default)]
    pub pac_sha256: Option<String>,
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
    /// P50 of the error-bound distribution over the training set (probes-per-lookup driver).
    /// `None` for sidecars built before v2 recorded percentiles.
    #[serde(default)]
    pub err_p50: Option<u64>,
    /// P90 of the error-bound distribution over the training set.
    /// `None` for sidecars built before v2 recorded percentiles.
    #[serde(default)]
    pub err_p90: Option<u64>,
    /// P99 of the error-bound distribution over the training set.
    /// `None` for sidecars built before v2 recorded percentiles.
    #[serde(default)]
    pub err_p99: Option<u64>,
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

/// Canonical v2 SA strand values. v2 builds only the forward+RC 2× SA.
/// Anything outside this list triggers `Error::FormatTooNew { kind }`.
pub(crate) const KNOWN_STRANDS: &[&str] = &["forward_rc_2x"];

/// Canonical v0.1 SA mode values. Anything outside this list triggers
/// `Error::FormatTooNew { kind }` at validation time.
pub(crate) const KNOWN_MODES_V01: &[&str] = &["1", "2"];

/// Expected `bytes_per_entry` for each mode. Must be consistent with
/// [`crate::train::config::MemoryMode::bytes_per_entry`].
///
/// Returns `None` for an unknown mode (which would already have been rejected
/// by the `KNOWN_MODES_V01` check before this is called).
pub(crate) fn expected_bytes_per_entry_for_mode(mode: &str) -> Option<u8> {
    match mode {
        "1" => Some(5),
        "2" => Some(13),
        _ => None,
    }
}

/// Expected `encoding` name for each mode.
pub(crate) fn expected_encoding_for_mode(mode: &str) -> Option<&'static str> {
    match mode {
        "1" => Some("packed_lo8_hi32"),
        "2" => Some("packed_lo8_hi32_key64"),
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
        if !KNOWN_STRANDS.contains(&self.sa.strand.as_str()) {
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
        // Validate the v2 2× invariants when present. A FULL 2× SA has exactly
        // `2*l_pac + 1` entries. A tiered (Design Z) keep-masked `.sa` retains
        // only a position-filtered subset, so it legitimately has FEWER entries
        // while `l_pac` stays the full forward genome length (positions remain
        // native genome coordinates). The `tiered` flag authorizes that: when
        // set, `1 <= num_entries <= 2*l_pac+1`; otherwise exactly `2*l_pac+1`.
        // More than `2*l_pac+1` is impossible either way (the doubled text has
        // exactly that many suffixes) and indicates corruption.
        if let Some(l_pac) = self.sa.l_pac {
            let expected_entries = l_pac
                .checked_mul(2)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: format!("l_pac overflow: {l_pac}"),
                })?;
            let tiered = self.sa.tiered == Some(true);
            let ok = if tiered {
                self.sa.num_entries >= 1 && self.sa.num_entries <= expected_entries
            } else {
                self.sa.num_entries == expected_entries
            };
            if !ok {
                return Err(Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: format!(
                        "num_entries={} inconsistent with l_pac={} (tiered={}; \
                         expected {} 2*l_pac+1={})",
                        self.sa.num_entries,
                        l_pac,
                        tiered,
                        if tiered { "1..=" } else { "==" },
                        expected_entries
                    ),
                });
            }
        } else if self.sa.tiered == Some(true) {
            // A tiered sidecar's `num_entries` invariants (and the forward-length
            // derivation) are all defined relative to `l_pac`; without it the
            // tiered claim is unverifiable, so reject rather than silently skip.
            return Err(Error::SizeMismatch {
                file: file.to_path_buf(),
                detail: "tiered=true requires sa.l_pac to be present".to_string(),
            });
        }
        if let Some(stored_keys) = self.sa.stored_keys {
            let expected_stored_keys = self.sa.mode == "2";
            if stored_keys != expected_stored_keys {
                return Err(Error::SizeMismatch {
                    file: file.to_path_buf(),
                    detail: format!(
                        "stored_keys={} inconsistent with mode={} (expected {})",
                        stored_keys, self.sa.mode, expected_stored_keys
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn rejects_v1_format_version() {
        let toml = r#"
[prmi]
magic = "PRMIv1"
format_version = 1
trainer_version = "prmi=0.1.0"
created_utc = "2026-05-27T00:00:00Z"
[ref]
path = "ref.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = 1
bytes_per_entry = 5
encoding = "packed_lo8_hi32"
mode = "1"
strand = "forward_only"
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 0
[priors]
type = "uniform"
"#;
        let err = Meta::from_toml_str(toml).unwrap_err();
        assert!(matches!(
            err,
            Error::BadMagic { .. } | Error::UnsupportedVersion { .. }
        ));
    }

    #[test]
    fn rejects_forward_only_strand_in_v2() {
        // End-to-end: a PRMIv2 sidecar with strand="forward_only" must be rejected.
        let toml = r#"
[prmi]
magic = "PRMIv2"
format_version = 2
trainer_version = "prmi=0.2.0"
created_utc = "2026-05-28T00:00:00Z"
[ref]
path = "ref.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = 1
bytes_per_entry = 5
encoding = "packed_lo8_hi32"
mode = "1"
strand = "forward_only"
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 0
[priors]
type = "uniform"
"#;
        let err = Meta::from_toml_str(toml).unwrap_err();
        assert!(
            matches!(err, Error::FormatTooNew { .. }),
            "expected FormatTooNew, got {err:?}"
        );
        // Supplemental sanity checks on the constant itself.
        assert!(!KNOWN_STRANDS.contains(&"forward_only"));
        assert!(KNOWN_STRANDS.contains(&"forward_rc_2x"));
    }

    #[test]
    fn pac_sha256_and_err_percentiles_roundtrip() {
        let toml = r#"
[prmi]
magic = "PRMIv2"
format_version = 2
trainer_version = "prmi=0.1.0"
created_utc = "2026-05-28T00:00:00Z"
[ref]
path = "ref.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = 9
bytes_per_entry = 13
encoding = "packed_lo8_hi32_key64"
mode = "2"
strand = "forward_rc_2x"
pac_sha256 = "ab12"
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 80
err_p50 = 3
err_p90 = 12
err_p99 = 80
[priors]
type = "uniform"
"#;
        let meta = Meta::from_toml_str(toml).unwrap();
        assert_eq!(meta.sa.pac_sha256, Some("ab12".into()));
        assert_eq!(meta.rmi.err_p50, Some(3));
        assert_eq!(meta.rmi.err_p90, Some(12));
        assert_eq!(meta.rmi.err_p99, Some(80));
    }

    #[test]
    fn sa_section_roundtrips_l_pac_and_stored_keys() {
        let toml = r#"
[prmi]
magic = "PRMIv2"
format_version = 2
trainer_version = "prmi=0.1.0"
created_utc = "2026-05-28T00:00:00Z"
[ref]
path = "ref.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = 9
bytes_per_entry = 13
encoding = "packed_lo8_hi32_key64"
mode = "2"
strand = "forward_rc_2x"
l_pac = 4
stored_keys = true
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 0
[priors]
type = "uniform"
"#;
        let meta = Meta::from_toml_str(toml).unwrap();
        assert_eq!(meta.sa.l_pac, Some(4));
        assert_eq!(meta.sa.stored_keys, Some(true));
    }

    /// A tiered (position-filtered, Design Z) sidecar legitimately has FEWER than
    /// `2*l_pac+1` entries (only keep-set positions are retained) while `l_pac`
    /// stays the full forward genome length. The `tiered` flag authorizes that.
    fn tiered_toml(num_entries: u64, tiered_line: &str) -> String {
        // mode "2": 13 B/entry, stored 32-mer keys.
        tiered_toml_mode(
            num_entries,
            tiered_line,
            "2",
            13,
            "packed_lo8_hi32_key64",
            true,
        )
    }

    /// Build a `[sa]` meta TOML for the given memory mode so the tiered
    /// `num_entries` invariants can be round-tripped across modes (this is an
    /// on-disk format/validation change, not a mode-2-only one).
    fn tiered_toml_mode(
        num_entries: u64,
        tiered_line: &str,
        mode: &str,
        bytes_per_entry: u8,
        encoding: &str,
        stored_keys: bool,
    ) -> String {
        format!(
            r#"
[prmi]
magic = "PRMIv2"
format_version = 2
trainer_version = "prmi=0.1.0"
created_utc = "2026-05-28T00:00:00Z"
[ref]
path = "ref.fa"
sha256 = "00"
size_bytes = 1
[sa]
num_entries = {num_entries}
bytes_per_entry = {bytes_per_entry}
encoding = "{encoding}"
mode = "{mode}"
strand = "forward_rc_2x"
l_pac = 4
stored_keys = {stored_keys}
{tiered_line}
[rmi]
spec = "pwl4,linear,linear_spline"
l2_leaf_count = 16
bit_shift = 60
max_error_bound = 0
[priors]
type = "uniform"
"#
        )
    }

    /// mode "1": 5 B/entry, position-only (no stored keys).
    fn tiered_toml_mode1(num_entries: u64, tiered_line: &str) -> String {
        tiered_toml_mode(num_entries, tiered_line, "1", 5, "packed_lo8_hi32", false)
    }

    #[test]
    fn tiered_sa_allows_fewer_entries_than_full() {
        // num_entries = 5 < 2*l_pac+1 = 9, authorized by tiered = true.
        let m = Meta::from_toml_str(&tiered_toml(5, "tiered = true")).unwrap();
        assert_eq!(m.sa.tiered, Some(true));
        assert_eq!(m.sa.l_pac, Some(4));
    }

    #[test]
    fn non_tiered_rejects_fewer_entries() {
        // Same but without the tiered flag: a short full SA is corruption.
        assert!(Meta::from_toml_str(&tiered_toml(5, "")).is_err());
    }

    #[test]
    fn num_entries_above_max_rejected_even_when_tiered() {
        // 11 > 2*l_pac+1 = 9 is impossible (the doubled text has 9 suffixes).
        assert!(Meta::from_toml_str(&tiered_toml(11, "tiered = true")).is_err());
    }

    // The same tiered `num_entries` invariants must hold for mode "1" — the
    // validation is mode-independent, so cover it across both memory modes.

    #[test]
    fn tiered_sa_allows_fewer_entries_than_full_mode1() {
        let m = Meta::from_toml_str(&tiered_toml_mode1(5, "tiered = true")).unwrap();
        assert_eq!(m.sa.tiered, Some(true));
        assert_eq!(m.sa.l_pac, Some(4));
        assert_eq!(m.sa.mode, "1");
        assert_eq!(m.sa.stored_keys, Some(false));
    }

    #[test]
    fn non_tiered_rejects_fewer_entries_mode1() {
        assert!(Meta::from_toml_str(&tiered_toml_mode1(5, "")).is_err());
    }

    #[test]
    fn num_entries_above_max_rejected_even_when_tiered_mode1() {
        assert!(Meta::from_toml_str(&tiered_toml_mode1(11, "tiered = true")).is_err());
    }

    #[test]
    fn tiered_without_l_pac_rejected() {
        // tiered=true with no l_pac is unverifiable and must be rejected (the
        // num_entries bounds are all defined relative to l_pac). Drop the
        // `l_pac = 4` line from an otherwise-valid tiered mode-2 meta.
        let toml = tiered_toml(5, "tiered = true").replace("l_pac = 4\n", "");
        assert!(Meta::from_toml_str(&toml).is_err());
    }
}
