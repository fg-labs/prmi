// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Brute-force verification pass — computes the global maximum absolute
//! prediction error across a training set. Stored in `.meta` as
//! `max_error_bound`; runtime uses it to size local-search bounds.

use crate::index::lookup::lookup_with_components;
use crate::train::prmi::PrmiModel;
use crate::train::training_set::TrainingSet;

/// Compute the absolute per-key prediction error for every training pair.
///
/// For each `(key, target)` in `ts`, the model prediction is obtained via
/// [`lookup_with_components`] and the error is
/// `(pred as i64 - target as i64).unsigned_abs()`.
///
/// This is the shared kernel used by both [`compute_max_error_bound`] and
/// [`compute_error_distribution`] so the two functions cannot drift apart.
fn per_key_errors(model: &PrmiModel, ts: &TrainingSet) -> Vec<u64> {
    let sa_num = ts.sa_num;
    ts.keys
        .iter()
        .zip(ts.sa_indices.iter())
        .map(|(k, target)| {
            let (pred, _err) =
                lookup_with_components(*k, &model.l1, &model.l2, model.bit_shift, sa_num);
            (pred as i64 - *target as i64).unsigned_abs()
        })
        .collect()
}

/// Brute-force pass: predict every training key, return the max absolute
/// prediction error. Becomes `max_error_bound` in the sidecar `.meta` header.
///
/// For v0.1 uniform priors, `ts.sa_indices[i] == i`, and the error is
/// `|pred - i|`. For future v0.2/v0.3 priors with non-identity sa_indices,
/// this function correctly compares against `ts.sa_indices[i]`.
pub fn compute_max_error_bound(model: &PrmiModel, ts: &TrainingSet) -> u64 {
    per_key_errors(model, ts).into_iter().max().unwrap_or(0)
}

/// Error-bound distribution over the training set: `(p50, p90, p99, max)` of
/// the absolute per-key prediction error. Drives probes-per-lookup and thus
/// speed. Uses the SAME per-key error as [`compute_max_error_bound`].
pub fn compute_error_distribution(model: &PrmiModel, ts: &TrainingSet) -> (u64, u64, u64, u64) {
    let mut errs = per_key_errors(model, ts);
    if errs.is_empty() {
        return (0, 0, 0, 0);
    }
    errs.sort_unstable();
    let pct = |p: f64| errs[(((errs.len() - 1) as f64) * p).round() as usize];
    (pct(0.50), pct(0.90), pct(0.99), *errs.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::config::TrainerConfig;
    use crate::train::trainer::train_with_config;
    use crate::train::training_set::TrainingSet;

    fn make_ts(keys: Vec<u64>, sa_indices: Vec<u64>) -> TrainingSet {
        let sa_num = sa_indices.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        TrainingSet {
            keys,
            sa_indices,
            sa_num,
            weights: None,
        }
    }

    #[test]
    fn error_distribution_ordering_and_max_matches_max_error_bound() {
        // 32 training pairs with a perfect linear layout so the model fits
        // well and errors are small but non-trivially varied.
        let n = 32usize;
        let key_base = 1u64 << 50;
        let key_stride = 1u64 << 46; // all keys route to leaf 0 of a 16-leaf L2
        let keys: Vec<u64> = (0..n as u64).map(|i| key_base + i * key_stride).collect();
        let sa_indices: Vec<u64> = (0..n as u64).collect();
        let ts = make_ts(keys, sa_indices);

        let config = TrainerConfig::default();
        let model = train_with_config(&ts, 16, &config).unwrap();

        let (p50, p90, p99, max) = compute_error_distribution(&model, &ts);

        // Ordering invariant.
        assert!(p50 <= p90, "p50={p50} must be <= p90={p90}");
        assert!(p90 <= p99, "p90={p90} must be <= p99={p99}");
        assert!(p99 <= max, "p99={p99} must be <= max={max}");

        // max from distribution must equal compute_max_error_bound.
        let expected_max = compute_max_error_bound(&model, &ts);
        assert_eq!(
            max, expected_max,
            "distribution max={max} must equal compute_max_error_bound={expected_max}"
        );
    }

    #[test]
    fn error_distribution_empty_training_set() {
        // An empty training set should return all zeros without panicking.
        let ts = make_ts(vec![], vec![]);
        let keys: Vec<u64> = vec![];
        let sa_indices: Vec<u64> = vec![];
        let ts2 = TrainingSet {
            keys,
            sa_indices,
            sa_num: 0,
            weights: None,
        };
        // We need a model; build one from a minimal non-empty ts, then test
        // compute_error_distribution against the empty ts.
        let ts_small = make_ts(vec![1u64 << 60], vec![0]);
        let config = TrainerConfig::default();
        let model = train_with_config(&ts_small, 16, &config).unwrap();
        let _ = ts; // suppress unused warning
        assert_eq!(compute_error_distribution(&model, &ts2), (0, 0, 0, 0));
        assert_eq!(compute_max_error_bound(&model, &ts2), 0);
    }
}
