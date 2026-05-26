// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Training-pair priors for target-aware model fitting.
//!
//! A [`Prior`] assigns a non-negative weight to each `(key, sa_index)` training
//! pair. The [`Uniform`] prior gives weight 1.0 to all pairs (the default). The
//! [`Bed`] prior gives weight `weight` (> 1.0) to pairs whose SA position falls
//! inside any BED interval and weight 1.0 to all others, biasing the model fit
//! toward regions of the reference that matter more for the workload. The
//! [`FastqHistogram`] prior gives higher weight to pairs whose 32-mer key is
//! frequently queried, biasing the model fit toward the observed workload.
//!
//! The prior affects **model fit** (weighted SLR) but not **verification** — the
//! max-error-bound verification pass always uses uniform weights against the true
//! SA position for every key.

use crate::error::{Error, Result};
use crate::train::mask::BedInterval;
use std::collections::HashMap;
use std::path::Path;

/// A training-pair prior.
///
/// `#[non_exhaustive]` so future variants can be added without breaking
/// pattern matches.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum Prior {
    /// All training pairs are weighted equally (weight = 1.0).
    #[default]
    Uniform,
    /// Pairs whose SA position falls inside any BED interval receive `weight`;
    /// all other pairs receive 1.0. `intervals` must be sorted by start and
    /// have no overlapping ranges (the invariant maintained by [`parse_bed`]).
    ///
    /// [`parse_bed`]: crate::train::mask::parse_bed
    Bed {
        /// Sorted, non-overlapping BED intervals (0-based, half-open).
        intervals: Vec<BedInterval>,
        /// Weight assigned to in-BED pairs. Must be > 0.0.
        weight: f64,
        /// Source path of the BED file, for provenance. `None` when constructed
        /// programmatically without a backing file.
        path: Option<std::path::PathBuf>,
    },
    /// Pairs whose 32-mer key appears in the FASTQ histogram receive a higher
    /// weight proportional to that key's frequency. Keys absent from the
    /// histogram receive `base_weight`.
    ///
    /// Weight formula: `base_weight + log2(1.0 + freq)` where `freq` is the
    /// per-key count from the histogram TSV. Keys with `freq = 0` (absent)
    /// receive exactly `base_weight`.
    ///
    /// The log2 transformation compresses a heavy-tailed frequency distribution
    /// (e.g. repeat k-mers that appear millions of times) into a range that
    /// emphasises hot keys without completely overwhelming cold ones.
    FastqHistogram {
        /// Map from 32-mer key (u64 tokenisation) to observed frequency.
        /// Keys absent from the map are treated as having frequency 0.
        keys_to_freq: HashMap<u64, u64>,
        /// Weight assigned to pairs whose key is absent from the histogram.
        /// Must be > 0.0. Default is 1.0.
        base_weight: f64,
        /// Source path of the histogram TSV, for provenance. `None` when
        /// constructed programmatically without a backing file.
        path: Option<std::path::PathBuf>,
    },
}

/// Return the training weight for a pair whose SA position is `sa_pos` and
/// whose 32-mer tokenization is `key`.
///
/// For [`Prior::Uniform`] this is always `1.0`. For [`Prior::Bed`] this is
/// `weight` if `sa_pos` is covered by any interval, else `1.0`. For
/// [`Prior::FastqHistogram`] this is `base_weight + log2(1.0 + freq(key))`.
#[inline]
pub fn weight_for_pair(prior: &Prior, key: u64, sa_pos: u64) -> f64 {
    match prior {
        Prior::Uniform => 1.0,
        Prior::Bed {
            intervals, weight, ..
        } => {
            if crate::train::mask::covered_by_bed(intervals, sa_pos) {
                *weight
            } else {
                1.0
            }
        }
        Prior::FastqHistogram {
            keys_to_freq,
            base_weight,
            ..
        } => {
            let freq = keys_to_freq.get(&key).copied().unwrap_or(0);
            base_weight + (1.0 + freq as f64).log2()
        }
    }
}

/// Convenience wrapper that calls [`weight_for_pair`] using only `sa_pos`.
///
/// Retained for callers that only have `sa_pos` available (e.g. the BED
/// prior path in `masked_training_set` before key computation). The `key`
/// argument is ignored by BED and Uniform priors; for the FASTQ histogram
/// prior, `key` must also be supplied — use [`weight_for_pair`] directly
/// when both are available.
#[inline]
#[doc(hidden)]
pub fn weight_for(prior: &Prior, sa_pos: u64) -> f64 {
    weight_for_pair(prior, 0, sa_pos)
}

/// Parse a 32-mer frequency histogram TSV into a `HashMap<u64, u64>`.
///
/// Each non-empty, non-comment line must have exactly two tab-separated
/// columns: `key_u64\tcount_u64`. Lines beginning with `#` are treated as
/// comments and ignored. Blank lines are ignored. Duplicate keys are rejected.
///
/// # Format
///
/// ```text
/// # Optional comment
/// 1152921504606846976    42
/// 2305843009213693952    1000000
/// ```
///
/// The key column is the 32-mer tokenised as a `u64` (MSB-first, 2-bit-encoded,
/// matching prmi's `tokenize_32mer` function). The count column is the
/// number of times that k-mer appeared in the FASTQ.
///
/// Keys with `count = 0` are parsed and stored; they effectively contribute
/// no lift above `base_weight` and may be omitted for efficiency.
///
/// # Errors
///
/// Returns `Err` if:
/// - The file cannot be read.
/// - Any data line does not have exactly two fields.
/// - Either field is not a valid `u64`.
/// - A key appears more than once.
pub fn parse_histogram_tsv(path: &Path) -> Result<HashMap<u64, u64>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut map: HashMap<u64, u64> = HashMap::new();

    for (line_num, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        // Skip blank lines and comment lines.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut fields = trimmed.splitn(3, '\t');
        let key_str = fields.next().ok_or_else(|| Error::InvalidInput {
            detail: format!(
                "{}:{}: expected exactly two tab-separated fields",
                path.display(),
                line_num + 1
            ),
        })?;
        let count_str = fields.next().ok_or_else(|| Error::InvalidInput {
            detail: format!(
                "{}:{}: expected exactly two tab-separated fields, found only one",
                path.display(),
                line_num + 1
            ),
        })?;
        if fields.next().is_some() {
            return Err(Error::InvalidInput {
                detail: format!(
                    "{}:{}: expected exactly two tab-separated fields, found three or more",
                    path.display(),
                    line_num + 1
                ),
            });
        }

        let key: u64 = key_str.trim().parse().map_err(|_| Error::InvalidInput {
            detail: format!(
                "{}:{}: key '{}' is not a valid u64",
                path.display(),
                line_num + 1,
                key_str.trim()
            ),
        })?;
        let count: u64 = count_str.trim().parse().map_err(|_| Error::InvalidInput {
            detail: format!(
                "{}:{}: count '{}' is not a valid u64",
                path.display(),
                line_num + 1,
                count_str.trim()
            ),
        })?;

        if map.contains_key(&key) {
            return Err(Error::InvalidInput {
                detail: format!(
                    "{}:{}: duplicate key {} in histogram",
                    path.display(),
                    line_num + 1,
                    key
                ),
            });
        }

        map.insert(key, count);
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::mask::BedInterval;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn uniform_prior_always_weight_one() {
        let prior = Prior::Uniform;
        assert_eq!(weight_for_pair(&prior, 0, 0), 1.0);
        assert_eq!(weight_for_pair(&prior, 12345, 1_000_000), 1.0);
        assert_eq!(weight_for_pair(&prior, u64::MAX, u64::MAX), 1.0);
    }

    #[test]
    fn bed_prior_in_bed_gets_higher_weight() {
        let prior = Prior::Bed {
            intervals: vec![BedInterval {
                start: 100,
                end: 200,
            }],
            weight: 10.0,
            path: None,
        };
        // Inside — key is ignored by BED prior.
        assert_eq!(weight_for_pair(&prior, 0, 100), 10.0);
        assert_eq!(weight_for_pair(&prior, 0, 150), 10.0);
        assert_eq!(weight_for_pair(&prior, 0, 199), 10.0);
        // Outside
        assert_eq!(weight_for_pair(&prior, 0, 99), 1.0);
        assert_eq!(weight_for_pair(&prior, 0, 200), 1.0);
        assert_eq!(weight_for_pair(&prior, 0, 500), 1.0);
    }

    #[test]
    fn bed_prior_empty_intervals_is_all_uniform() {
        let prior = Prior::Bed {
            intervals: vec![],
            weight: 10.0,
            path: None,
        };
        assert_eq!(weight_for_pair(&prior, 0, 50), 1.0);
        assert_eq!(weight_for_pair(&prior, 0, 0), 1.0);
    }

    #[test]
    fn prior_default_is_uniform() {
        let prior = Prior::default();
        assert!(matches!(prior, Prior::Uniform));
    }

    #[test]
    fn fastq_histogram_absent_key_returns_base_weight() {
        let prior = Prior::FastqHistogram {
            keys_to_freq: HashMap::new(),
            base_weight: 1.0,
            path: None,
        };
        // key absent → base_weight + log2(1 + 0) = 1.0 + 0.0 = 1.0
        assert!((weight_for_pair(&prior, 42, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fastq_histogram_freq_1_gives_weight_2() {
        let mut m = HashMap::new();
        m.insert(100u64, 1u64);
        let prior = Prior::FastqHistogram {
            keys_to_freq: m,
            base_weight: 1.0,
            path: None,
        };
        // freq=1 → base_weight + log2(2) = 1.0 + 1.0 = 2.0
        assert!((weight_for_pair(&prior, 100, 0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn fastq_histogram_hot_key_higher_than_cold_key() {
        let mut m = HashMap::new();
        m.insert(999u64, 1_000_000u64); // very hot
        m.insert(1u64, 2u64); // barely warm
        let prior = Prior::FastqHistogram {
            keys_to_freq: m,
            base_weight: 1.0,
            path: None,
        };
        let w_hot = weight_for_pair(&prior, 999, 0);
        let w_cold = weight_for_pair(&prior, 1, 0);
        let w_absent = weight_for_pair(&prior, 0, 0);
        assert!(
            w_hot > w_cold,
            "hot key weight {w_hot} should be > cold key weight {w_cold}"
        );
        assert!(
            w_cold > w_absent,
            "cold key weight {w_cold} should be > absent key weight {w_absent}"
        );
        // hot: 1 + log2(1 + 10^6) ≈ 1 + 19.93 ≈ 20.93
        assert!((w_hot - (1.0 + (1.0 + 1_000_000f64).log2())).abs() < 1e-6);
    }

    #[test]
    fn parse_histogram_tsv_basic() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "# comment").unwrap();
        writeln!(f, "1000\t42").unwrap();
        writeln!(f, "2000\t1000000").unwrap();
        writeln!(f).unwrap(); // blank line
        let map = parse_histogram_tsv(f.path()).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1000u64], 42u64);
        assert_eq!(map[&2000u64], 1_000_000u64);
    }

    #[test]
    fn parse_histogram_tsv_rejects_duplicate_keys() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "100\t5").unwrap();
        writeln!(f, "100\t10").unwrap();
        let err = parse_histogram_tsv(f.path()).unwrap_err();
        assert!(
            format!("{err}").contains("duplicate key"),
            "expected duplicate-key error, got: {err}"
        );
    }

    #[test]
    fn parse_histogram_tsv_rejects_non_numeric_key() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "not_a_number\t5").unwrap();
        let err = parse_histogram_tsv(f.path()).unwrap_err();
        assert!(
            format!("{err}").contains("not a valid u64"),
            "expected numeric error, got: {err}"
        );
    }

    #[test]
    fn parse_histogram_tsv_rejects_missing_count_field() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "1234").unwrap(); // no tab, no count
        let err = parse_histogram_tsv(f.path()).unwrap_err();
        assert!(
            format!("{err}").contains("tab-separated fields"),
            "expected field-count error, got: {err}"
        );
    }

    #[test]
    fn parse_histogram_tsv_empty_file_is_ok() {
        let f = NamedTempFile::new().unwrap();
        let map = parse_histogram_tsv(f.path()).unwrap();
        assert!(map.is_empty());
    }
}
