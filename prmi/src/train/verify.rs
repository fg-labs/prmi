// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Brute-force verification pass — computes the global maximum absolute
//! prediction error across a training set. Stored in `.meta` as
//! `max_error_bound`; runtime uses it to size local-search bounds.

use crate::index::lookup::lookup_with_components;
use crate::train::prmi::PrmiModel;
use crate::train::training_set::TrainingSet;
use rayon::prelude::*;

/// Upper bound on the dense histogram size in [`compute_error_distribution`].
///
/// The per-key error is structurally bounded only by `sa_num - 1` (≈ 6.4e9 at
/// hg38 scale), so sizing a dense count array to `max_err + 1` could allocate
/// tens of GB on a pathologically ill-fit model — reintroducing the very array
/// this function exists to remove. We therefore cap the dense region at
/// `HIST_DENSE_CAP` and spill the rare `err > cap` into a small overflow list.
/// Real builds have a tiny `max_err` (hg38: ~3.4e5), so the dense array is a
/// few MB and the overflow is empty.
const HIST_DENSE_CAP: u64 = 256 << 20;

/// Above this dense-histogram length the per-worker buffers of the parallel
/// fold (`dense_len * rayon_threads * 8` bytes) would dwarf a single buffer, so
/// the histogram falls back to a serial single-buffer pass. `1 << 20` entries =
/// 8 MiB per buffer; below it the parallel path's `× threads` is cheap, above it
/// memory — not CPU — is the bottleneck and one buffer is the safe choice.
/// Real builds (hg38 `max_err` ~3.4e5) stay well under this and run parallel.
const HIST_PARALLEL_DENSE_LIMIT: usize = 1 << 20;

/// Absolute prediction error of training pair `i`.
///
/// The SA rank is both the regression target and the (possibly streamed) key's
/// source position; it is computed once and reused. Each call is independent,
/// so the verify passes parallelise without any cross-thread accumulation.
#[inline]
fn error_at(model: &PrmiModel, ts: &TrainingSet, i: usize) -> u64 {
    let rank = ts.sa_indices.get(i);
    let key = ts.keys.at(i, rank);
    let (pred, _err) =
        lookup_with_components(key, &model.l1, &model.l2, model.bit_shift, ts.sa_num);
    (pred as i64 - rank as i64).unsigned_abs()
}

/// Brute-force pass: predict every training key, return the max absolute
/// prediction error. Becomes `max_error_bound` in the sidecar `.meta` header.
///
/// For v0.1 uniform priors, `ts.sa_indices[i] == i`, and the error is
/// `|pred - i|`. For future v0.2/v0.3 priors with non-identity sa_indices,
/// this function correctly compares against `ts.sa_indices[i]`.
///
/// Parallel `max` is order-independent → bit-identical to the serial reduction.
pub fn compute_max_error_bound(model: &PrmiModel, ts: &TrainingSet) -> u64 {
    (0..ts.len())
        .into_par_iter()
        .map(|i| error_at(model, ts, i))
        .max()
        .unwrap_or(0)
}

/// Error-bound distribution over the training set: `(p50, p90, p99, max)` of
/// the absolute per-key prediction error. Drives probes-per-lookup and thus
/// speed. Uses the SAME per-key error as [`compute_max_error_bound`].
///
/// Computed by a streaming histogram rather than by sorting a per-key error
/// vector: two streaming passes (max+count, then bucket counts) over a dense
/// count array capped at [`HIST_DENSE_CAP`] plus a small overflow list. This
/// avoids both the ~51.5 GB error vector and its sort at hg38 scale.
///
/// The returned percentiles are **bit-identical** to the previous
/// sort-then-index implementation: the value at sorted rank
/// `idx = round((n-1)·p)` equals the smallest histogram value whose inclusive
/// cumulative count first exceeds `idx`. The returned `max` equals
/// [`compute_max_error_bound`].
pub fn compute_error_distribution(model: &PrmiModel, ts: &TrainingSet) -> (u64, u64, u64, u64) {
    let n = ts.len();
    if n == 0 {
        return (0, 0, 0, 0);
    }

    // Pass 1 (parallel): global max. Order-independent.
    let max = (0..n)
        .into_par_iter()
        .map(|i| error_at(model, ts, i))
        .max()
        .unwrap();

    // Pass 2: a dense histogram (covering [0, dense_cap]) plus an overflow list
    // for the rare err > dense_cap. The parallel path allocates one dense buffer
    // per rayon worker, so above `HIST_PARALLEL_DENSE_LIMIT` it would multiply a
    // large buffer by the thread count; there a serial single-buffer pass caps
    // memory. Both produce identical counts (order-independent), so percentiles
    // stay bit-identical.
    let dense_cap = max.min(HIST_DENSE_CAP);
    let parallel = (dense_cap as usize + 1) <= HIST_PARALLEL_DENSE_LIMIT;
    let (dense, mut overflow) = error_histogram(model, ts, n, dense_cap, parallel);
    overflow.sort_unstable();

    let idx = |p: f64| (((n - 1) as f64) * p).round() as usize;
    (
        select_rank(&dense, &overflow, idx(0.50)),
        select_rank(&dense, &overflow, idx(0.90)),
        select_rank(&dense, &overflow, idx(0.99)),
        max,
    )
}

/// Build the dense error histogram `[0, dense_cap]` plus an overflow list of the
/// `err > dense_cap` values, over the `n` training pairs. `parallel` selects the
/// per-worker fold (fast, but `dense_len × threads` memory) or a serial
/// single-buffer pass (memory-bounded). Both return identical counts — the only
/// difference is execution strategy. The overflow list is returned unsorted.
fn error_histogram(
    model: &PrmiModel,
    ts: &TrainingSet,
    n: usize,
    dense_cap: u64,
    parallel: bool,
) -> (Vec<u64>, Vec<u64>) {
    let dense_len = dense_cap as usize + 1;
    if parallel {
        (0..n)
            .into_par_iter()
            .fold(
                || (vec![0u64; dense_len], Vec::<u64>::new()),
                |(mut dense, mut ov), i| {
                    let e = error_at(model, ts, i);
                    if e <= dense_cap {
                        dense[e as usize] += 1;
                    } else {
                        ov.push(e);
                    }
                    (dense, ov)
                },
            )
            .reduce(
                || (vec![0u64; dense_len], Vec::<u64>::new()),
                |(mut d1, mut o1), (d2, o2)| {
                    for (a, b) in d1.iter_mut().zip(d2.iter()) {
                        *a += *b;
                    }
                    o1.extend(o2);
                    (d1, o1)
                },
            )
    } else {
        // Serial: one dense buffer, no `× threads` amplification.
        let mut dense = vec![0u64; dense_len];
        let mut overflow = Vec::<u64>::new();
        for i in 0..n {
            let e = error_at(model, ts, i);
            if e <= dense_cap {
                dense[e as usize] += 1;
            } else {
                overflow.push(e);
            }
        }
        (dense, overflow)
    }
}

/// Value at sorted rank `idx` of a multiset stored as a dense histogram
/// (`dense[v]` = count of value `v`) plus a sorted `overflow` list of values
/// larger than every dense index.
///
/// Returns the smallest value whose inclusive cumulative count first exceeds
/// `idx` — equivalent to `sorted_values[idx]`. Callers guarantee
/// `idx < total_count`, so the walk always returns before the end.
fn select_rank(dense: &[u64], overflow_sorted: &[u64], idx: usize) -> u64 {
    let mut cum = 0usize;
    for (v, &c) in dense.iter().enumerate() {
        cum += c as usize;
        if cum > idx {
            return v as u64;
        }
    }
    for &v in overflow_sorted {
        cum += 1;
        if cum > idx {
            return v;
        }
    }
    // Unreachable for valid `idx < total_count`; return the largest value.
    overflow_sorted
        .last()
        .copied()
        .unwrap_or_else(|| dense.len().saturating_sub(1) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::config::TrainerConfig;
    use crate::train::trainer::train_with_config;
    use crate::train::training_set::{Keys, SaIndices, TrainingSet};
    use std::sync::Arc;

    fn make_ts(keys: Vec<u64>, sa_indices: Vec<u64>) -> TrainingSet {
        let sa_num = sa_indices.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        TrainingSet {
            keys: Keys::Materialized(Arc::new(keys)),
            sa_indices: SaIndices::Materialized(Arc::new(sa_indices)),
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
            keys: Keys::Materialized(Arc::new(keys)),
            sa_indices: SaIndices::Materialized(Arc::new(sa_indices)),
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

    /// Sort-then-index reference: the exact computation the histogram path
    /// replaced. Used to prove byte-for-byte equivalence.
    fn reference_distribution(model: &PrmiModel, ts: &TrainingSet) -> (u64, u64, u64, u64) {
        let sa_num = ts.sa_num;
        let mut errs: Vec<u64> = (0..ts.len())
            .map(|i| {
                let rank = ts.sa_indices.get(i);
                let key = ts.keys.at(i, rank);
                let (pred, _) =
                    lookup_with_components(key, &model.l1, &model.l2, model.bit_shift, sa_num);
                (pred as i64 - rank as i64).unsigned_abs()
            })
            .collect();
        if errs.is_empty() {
            return (0, 0, 0, 0);
        }
        errs.sort_unstable();
        let pct = |p: f64| errs[(((errs.len() - 1) as f64) * p).round() as usize];
        (pct(0.50), pct(0.90), pct(0.99), *errs.last().unwrap())
    }

    #[test]
    fn error_distribution_equals_sorted_reference() {
        // The streaming histogram must return bit-identical (p50,p90,p99,max)
        // to the sort-then-index reference across model sizes.
        let config = TrainerConfig::default();
        for &n in &[16usize, 100, 1000, 4096] {
            let keys: Vec<u64> = (0..n as u64).map(|i| (i + 1) * (1u64 << 44)).collect();
            let sa_indices: Vec<u64> = (0..n as u64).collect();
            let ts = make_ts(keys, sa_indices);
            let model = train_with_config(&ts, 16, &config).unwrap();
            assert_eq!(
                compute_error_distribution(&model, &ts),
                reference_distribution(&model, &ts),
                "histogram distribution must match sorted reference for n={n}"
            );
        }
    }

    #[test]
    fn error_histogram_serial_equals_parallel() {
        // The serial single-buffer fallback (used above HIST_PARALLEL_DENSE_LIMIT
        // to cap memory) must produce identical counts to the parallel fold.
        let config = TrainerConfig::default();
        let keys: Vec<u64> = (0..1000u64).map(|i| (i + 1) * (1u64 << 44)).collect();
        let sa_indices: Vec<u64> = (0..1000u64).collect();
        let ts = make_ts(keys, sa_indices);
        let model = train_with_config(&ts, 16, &config).unwrap();
        let n = ts.len();
        let dense_cap = compute_max_error_bound(&model, &ts).min(HIST_DENSE_CAP);
        let par = error_histogram(&model, &ts, n, dense_cap, true);
        let ser = error_histogram(&model, &ts, n, dense_cap, false);
        let mut par_ov = par.1.clone();
        let mut ser_ov = ser.1.clone();
        par_ov.sort_unstable();
        ser_ov.sort_unstable();
        assert_eq!(par.0, ser.0, "dense histograms differ between strategies");
        assert_eq!(par_ov, ser_ov, "overflow lists differ between strategies");
    }

    #[test]
    fn select_rank_matches_sorted_indexing_with_overflow() {
        // Deterministic LCG. A small dense cap forces values into the overflow
        // list, exercising the dense→overflow boundary and ties. select_rank
        // must equal sorted_values[idx] for every probed rank.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let cap: u64 = 100;
        for trial in 0..300 {
            let len = 1 + (next() as usize % 600);
            let errs: Vec<u64> = (0..len).map(|_| next() % 250).collect();
            let mut sorted = errs.clone();
            sorted.sort_unstable();

            let maxv = *errs.iter().max().unwrap();
            let dense_cap = maxv.min(cap);
            let mut dense = vec![0u64; dense_cap as usize + 1];
            let mut overflow: Vec<u64> = Vec::new();
            for &e in &errs {
                if e <= dense_cap {
                    dense[e as usize] += 1;
                } else {
                    overflow.push(e);
                }
            }
            overflow.sort_unstable();

            for &idx in &[0usize, len / 4, len / 2, (3 * len) / 4, len - 1] {
                assert_eq!(
                    select_rank(&dense, &overflow, idx),
                    sorted[idx],
                    "trial={trial} len={len} idx={idx}"
                );
            }
        }
    }
}
