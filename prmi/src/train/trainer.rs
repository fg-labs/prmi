// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Fulcrum-authored P-RMI training driver built over Marcus's RMI primitives.
//!
//! The public entry point is [`train_with_config`]. It composes:
//! - [`LinearModel`] for L2 direct-leaf fits and fallback routing models
//! - [`LinearSplineModel`] for L1 sub-leaf fits
//! - [`RMITrainingData`] for training-data iteration and per-leaf rescaling
//! - [`LowerBoundCorrection`] for empty-leaf boundary handling
//!
//! Algorithm: spec §5.10 in `docs/superpowers/specs/2026-05-21-prmi-cleanroom-trainer.md`.
//! Constant defaults: audit memo `docs/superpowers/research/2026-05-21-bwa-meme-audit.md`.
//! Lookup contract: brief §4.4 in `docs/superpowers/handoff/2026-05-20-prmi-v0.1-brief.md`.

use crate::error::{Error, Result};
use crate::sidecar::model_file::ModelEntry;
use crate::train::config::TrainerConfig;
use crate::train::prmi::PrmiModel;
use crate::train::training_set::TrainingSet;
use crate::upstream::train::lower_bound_correction::LowerBoundCorrection;
use crate::upstream::{LinearModel, LinearSplineModel, Model, ModelParam, RMITrainingData};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Compute a reasonable default `l2_leaf_count` from the SA size.
///
/// Targets ~12 SA entries per L2 leaf (BWA-MEME's empirical human-genome
/// ratio: 3.1 Gbp / 2^28 ≈ 12). Rounds down to a power of two and clamps to
/// `[2^4, 2^28]`: the lower bound is the pwl<4 floor (`l2_leaf_count < 16` is
/// rejected by the trainer); the upper bound is BWA-MEME's largest published
/// configuration.
///
/// # Examples
///
/// ```ignore
/// default_l2_leaf_count(5_000)           // phiX (~5 kb) → 256 (2^8)
/// default_l2_leaf_count(5_000_000)       // E. coli (~5 Mbp) → 2^18
/// default_l2_leaf_count(3_100_000_000)   // hg38 (~3.1 Gbp) → 2^28
/// ```
pub fn default_l2_leaf_count(sa_num: usize) -> u64 {
    let target = (sa_num / 12).max(16);
    let pow2 = (target.next_power_of_two() >> 1).max(16);
    (pow2 as u64).clamp(16, 1 << 28)
}

/// Extract (alpha, beta) from a fitted model and guard against non-finite
/// values. Every `LinearModel::new` / `LinearSplineModel::new` result must
/// pass through here before being written to a `ModelEntry`.
fn alpha_beta(m: &dyn Model) -> Result<(f64, f64)> {
    let params = m.params();
    let alpha = match params.first() {
        Some(ModelParam::Float(v)) => *v,
        _ => {
            return Err(Error::Internal {
                detail: "model alpha not f64".into(),
            })
        }
    };
    let beta = match params.get(1) {
        Some(ModelParam::Float(v)) => *v,
        _ => {
            return Err(Error::Internal {
                detail: "model beta not f64".into(),
            })
        }
    };
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::Internal {
            detail: format!("non-finite model params: alpha={alpha}, beta={beta}"),
        });
    }
    Ok((alpha, beta))
}

/// Encode a fallback L2 `err` field. Bit 63 is set to signal the fallback
/// path; bits 32-62 hold `partial_start`; bits 0-31 hold `partial_num`.
///
/// Constraints: `partial_start < 2^31` (31 bits), `partial_num < 2^32` (32 bits).
/// Source: brief §4.4.
#[inline]
fn encode_fallback_err(partial_start: u64, partial_num: u64) -> u64 {
    debug_assert!(partial_start < (1 << 31), "partial_start overflow");
    debug_assert!(partial_num < (1 << 32), "partial_num overflow");
    (1u64 << 63) | ((partial_start & 0x7fff_ffff) << 32) | (partial_num & 0xffff_ffff)
}

/// Mirror the runtime's `lookup_core` exactly: `pred = clamp(alpha + beta*key,
/// 0, sa_num-1)` with truncation (matching `index::lookup::clamp_to_int`).
/// Returning i64 so callers can take signed differences against sa_idx
/// without overflow. NaN maps to 0 (same as runtime).
#[inline]
fn predict_clamped(alpha: f64, beta: f64, key: u64, sa_num: u64) -> i64 {
    let raw = alpha + beta * key as f64;
    if raw.is_nan() {
        return 0;
    }
    raw.clamp(0.0, sa_num.saturating_sub(1) as f64) as i64
}

// ── main entry point ─────────────────────────────────────────────────────────

/// Train a P-RMI using `config` on the supplied training set.
///
/// This is the algorithmic core. [`crate::train::prmi::train_prmi`] delegates
/// here with a default config; callers that need non-default thresholds (e.g.
/// tests that want a small `fallback_threshold`) can call this directly.
pub fn train_with_config(
    ts: &TrainingSet,
    l2_leaf_count: u64,
    config: &TrainerConfig,
) -> Result<PrmiModel> {
    // ── step 1: validate ──────────────────────────────────────────────────
    if !l2_leaf_count.is_power_of_two() || l2_leaf_count < 2 {
        return Err(Error::Internal {
            detail: format!("l2_leaf_count={l2_leaf_count} must be a power of two ≥ 2"),
        });
    }
    let pwl_bits = l2_leaf_count.trailing_zeros();
    if pwl_bits < 4 {
        return Err(Error::Internal {
            detail: format!(
                "l2_leaf_count={l2_leaf_count} would use pwl{pwl_bits}; \
                 minimum supported value is 16 (pwl4)"
            ),
        });
    }
    if ts.is_empty() {
        return Err(Error::Internal {
            detail: "empty training set".into(),
        });
    }

    // ── step 2: derived constants ─────────────────────────────────────────
    let bit_shift = 64 - pwl_bits;
    let n = ts.len();

    // ── step 3: build RMITrainingData ─────────────────────────────────────
    // Build Vec<(u64, usize)> from the training set. The `RMITrainingData`
    // scale machinery applies a scale factor to the `usize` coordinate.
    let pairs: Vec<(u64, usize)> = ts
        .keys
        .iter()
        .zip(ts.sa_indices.iter())
        .map(|(&k, &s)| (k, s as usize))
        .collect();
    let data = RMITrainingData::<u64>::new(Box::new(pairs));

    // ── step 4: LowerBoundCorrection over full data ───────────────────────
    // `pred_func` is the L2 routing function: key >> bit_shift, clamped to
    // [0, l2_leaf_count). Used by LBC to compute per-leaf first/last/next/prev.
    let bs = bit_shift; // capture by value
    let lbc: LowerBoundCorrection<u64> =
        LowerBoundCorrection::new(|k: u64| k >> bs, l2_leaf_count, &data);

    // ── step 5: partition keys into L2 leaves ─────────────────────────────
    // One pass over the sorted keys (monotone routing → leaves fill left-to-right).
    // `leaf_ranges[i] = (start_idx, end_idx)` into the flat keys/sa_indices vecs.
    let mut leaf_ranges: Vec<(usize, usize)> = vec![(0, 0); l2_leaf_count as usize];
    {
        let mut i = 0usize;
        for (leaf_idx, range) in leaf_ranges.iter_mut().enumerate() {
            let start = i;
            while i < n && (ts.keys[i] >> bit_shift) as usize == leaf_idx {
                i += 1;
            }
            *range = (start, i);
        }
    }

    // ── step 6: build L1 and L2 arrays ───────────────────────────────────
    let mut l1: Vec<ModelEntry> = Vec::new();
    let mut l2: Vec<ModelEntry> = Vec::with_capacity(l2_leaf_count as usize);

    for (leaf_idx, &(start, end)) in leaf_ranges.iter().enumerate() {
        let leaf_len = end - start;

        if leaf_len == 0 {
            // ── case a: empty leaf ────────────────────────────────────────
            // Emit a constant model returning the next non-empty leaf's first
            // SA index. `lbc.next_index(leaf_idx)` is the SA position of the
            // first key in the next non-empty leaf (or `data.len()` if none).
            let const_pred = lbc.next_index(leaf_idx) as f64;
            l2.push(ModelEntry {
                alpha: const_pred,
                beta: 0.0,
                err: 0,
            });
            continue;
        }

        if leaf_len <= config.fallback_threshold {
            // ── case b: direct leaf (DIRECT path) ────────────────────────
            l2.push(fit_direct_leaf(
                &lbc, ts, start, end, leaf_idx, bit_shift, n,
            )?);
        } else {
            // ── case c: large leaf — try fallback, downgrade if needed ───
            let start_y = ts.sa_indices[start] as usize;
            let end_y = ts.sa_indices[end - 1] as usize;

            if end_y == start_y {
                // DBZ guard: all keys in this leaf map to the same SA position.
                // Downgrade to direct fit. [A] audit, §Issues "Division by zero".
                l2.push(fit_direct_leaf(
                    &lbc, ts, start, end, leaf_idx, bit_shift, n,
                )?);
            } else {
                l2.push(fit_fallback_leaf(
                    ts, start, end, start_y, end_y, &mut l1, config, n as u64,
                )?);
            }
        }
    }

    // ── step 8: assert shape ──────────────────────────────────────────────
    assert_eq!(l2.len(), l2_leaf_count as usize);

    // ── step 9: cap L1 size ───────────────────────────────────────────────
    if l1.len() as u64 > config.max_l1_entries {
        return Err(Error::Internal {
            detail: format!(
                "l1 array size {} exceeds max_l1_entries {} (brief §4.4 bit-width constraint)",
                l1.len(),
                config.max_l1_entries
            ),
        });
    }

    Ok(PrmiModel { l1, l2, bit_shift })
}

// ── per-leaf helpers ─────────────────────────────────────────────────────────

/// Fit a direct (non-fallback) L2 leaf entry.
///
/// 1. Fit a `LinearModel` on the leaf's training pairs.
/// 2. NaN-guard the resulting (alpha, beta).
/// 3. Compute `err` = max absolute prediction error, widened by the LBC
///    neighbor-correction term (spec §5.10 step 7b).
fn fit_direct_leaf(
    lbc: &LowerBoundCorrection<u64>,
    ts: &TrainingSet,
    start: usize,
    end: usize,
    leaf_idx: usize,
    bit_shift: u32,
    n: usize,
) -> Result<ModelEntry> {
    // Build a soft-copy restricted to this leaf's range.
    // We build a new Vec for the leaf slice. This is O(leaf_len) per leaf —
    // acceptable since leaves are O(fallback_threshold) = O(1000) entries.
    let leaf_pairs: Vec<(u64, usize)> = ts.keys[start..end]
        .iter()
        .zip(ts.sa_indices[start..end].iter())
        .map(|(&k, &s)| (k, s as usize))
        .collect();
    let leaf_data = RMITrainingData::<u64>::new(Box::new(leaf_pairs));

    let model = LinearModel::new(&leaf_data);
    let (alpha, beta) = alpha_beta(&model)?;

    // Compute in-leaf error: max |predict(k) - sa_idx| over all leaf keys.
    let mut err = 0u64;
    for (k, sa_idx) in ts.keys[start..end]
        .iter()
        .zip(ts.sa_indices[start..end].iter())
    {
        let pred = predict_clamped(alpha, beta, *k, n as u64);
        let d = (pred - *sa_idx as i64).unsigned_abs();
        if d > err {
            err = d;
        }
    }

    // LBC neighbor-correction: widen err by the distance from this leaf's
    // model to the first key of the NEXT non-empty leaf, measured against
    // the LAST sa_index of THIS leaf.
    //
    // A query key k between this leaf's last key and the next leaf's first key
    // routes to this leaf (k >> bit_shift == leaf_idx), so this leaf's model
    // must predict close enough to the last SA position in this leaf.
    //
    // Skip when next_index == data.len() (no next leaf exists; no inter-leaf gap).
    let next_idx = lbc.next_index(leaf_idx);
    if next_idx < n {
        // `lbc.next(leaf_idx)` = (next_sa_index, next_key)
        let (_, next_key) = lbc.next(leaf_idx);
        // Only apply if the next key routes here too (i.e. same L2 bucket).
        // In practice this can only happen when bit_shift >= 64 (impossible
        // for valid configs), but guard it anyway for robustness.
        let next_key_bucket = if bit_shift >= 64 {
            0
        } else {
            next_key >> bit_shift
        } as usize;
        if next_key_bucket == leaf_idx {
            // next key routes to this same leaf — no inter-leaf gap to correct.
        } else {
            // last SA index belonging to this leaf = next_idx - 1
            let last_sa = (next_idx - 1) as i64;
            let pred_next = predict_clamped(alpha, beta, next_key, n as u64);
            let d = (pred_next - last_sa).unsigned_abs();
            if d > err {
                err = d;
            }
        }
    }

    Ok(ModelEntry { alpha, beta, err })
}

/// Fit a fallback L2 leaf entry: a routing model + L1 sub-leaf array.
///
/// Algorithm (spec §5.10 step 7c):
/// 1. Compute `partial_num = ceil(leaf_len / partial_target_size)`.
/// 2. Build a rescaled view of the leaf data for the routing model.
/// 3. Fit a `LinearModel` routing model on the rescaled view.
/// 4. Partition the leaf into `partial_num` sub-leaves via routing predictions.
/// 5. For each sub-leaf: fit `LinearSplineModel` (≥2 keys), constant (1 key),
///    or constant-to-next (empty).
/// 6. Append sub-leaf entries to `l1`; emit routing L2 entry.
#[allow(clippy::too_many_arguments)]
fn fit_fallback_leaf(
    ts: &TrainingSet,
    start: usize,
    end: usize,
    start_y: usize,
    end_y: usize,
    l1: &mut Vec<ModelEntry>,
    config: &TrainerConfig,
    sa_num: u64,
) -> Result<ModelEntry> {
    let leaf_len = end - start;

    // partial_num = ceil(leaf_len / partial_target_size). [A] audit §"Algorithmic notes".
    let partial_num = (leaf_len as u64).div_ceil(config.partial_target_size) as usize;

    // Build a scaled copy of the leaf for routing-model training.
    // Scale = (partial_num - 1) / (end_y - start_y) maps sa_indices in
    // [start_y, end_y] to the range [0, partial_num - 1].
    // We fold the offset (subtracting start_y) directly into the scaled Vec
    // since Marcus's reverted RMITrainingData has no set_offset API.
    // [A] audit §"Design decision #6 Scale/offset rescaling".
    let scale = (partial_num - 1) as f64 / (end_y - start_y) as f64;
    let scaled_pairs: Vec<(u64, usize)> = ts.keys[start..end]
        .iter()
        .zip(ts.sa_indices[start..end].iter())
        .map(|(&k, &s)| {
            let shifted = (s as usize).saturating_sub(start_y);
            let scaled = (shifted as f64 * scale).round() as usize;
            (k, scaled.min(partial_num - 1))
        })
        .collect();
    let scaled_data = RMITrainingData::<u64>::new(Box::new(scaled_pairs));

    // Fit the routing LinearModel on the scaled data.
    let routing_model = LinearModel::new(&scaled_data);
    let (routing_alpha, routing_beta) = alpha_beta(&routing_model)?;

    // ── partition leaf into sub-leaves ────────────────────────────────────
    // `sub_leaf_items[s]` = Vec of (key, sa_index) pairs routed to sub-leaf s.
    // Routing: clamp(routing_alpha + routing_beta * key, 0, partial_num - 1) using
    // truncation (not rounding) to match the runtime's `clamp_to_int` behavior.
    let mut sub_leaf_items: Vec<Vec<(u64, usize)>> = vec![Vec::new(); partial_num];
    for (&k, &sa_idx) in ts.keys[start..end]
        .iter()
        .zip(ts.sa_indices[start..end].iter())
    {
        let raw = routing_alpha + routing_beta * k as f64;
        let sub_idx = if raw.is_nan() {
            0
        } else {
            raw.clamp(0.0, (partial_num - 1) as f64) as usize
        };
        sub_leaf_items[sub_idx].push((k, sa_idx as usize));
    }

    // ── compute "next non-empty sub-leaf first sa_index" for empty sub-leaves.
    // Forward scan: for each empty sub-leaf, record the first SA index of the
    // next non-empty sub-leaf (or end_y + 1 if none exists after).
    // [A] audit §"Design decision #7 LBC empty-partial-leaf edge case".
    let mut next_non_empty_sa: Vec<usize> = vec![end_y + 1; partial_num];
    {
        let mut fill = end_y + 1;
        for s in (0..partial_num).rev() {
            if !sub_leaf_items[s].is_empty() {
                fill = sub_leaf_items[s][0].1;
            }
            next_non_empty_sa[s] = fill;
        }
    }

    // ── record where this leaf's L1 entries begin ─────────────────────────
    let partial_start = l1.len() as u64;
    debug_assert!(partial_start < (1 << 31), "partial_start overflow");

    // ── fit each sub-leaf and append to l1 ───────────────────────────────
    for s in 0..partial_num {
        let items = &sub_leaf_items[s];
        let entry = match items.len() {
            0 => {
                // Empty sub-leaf: constant pointing at next non-empty first SA index.
                ModelEntry {
                    alpha: next_non_empty_sa[s] as f64,
                    beta: 0.0,
                    err: 0,
                }
            }
            1 => {
                // Single-key sub-leaf: constant at that key's SA index.
                ModelEntry {
                    alpha: items[0].1 as f64,
                    beta: 0.0,
                    err: 0,
                }
            }
            _ => {
                // Two or more keys: fit LinearSplineModel. [A] audit §Q4 decision.
                let sub_pairs: Vec<(u64, usize)> = items.clone();
                let sub_data = RMITrainingData::<u64>::new(Box::new(sub_pairs));
                let sub_model = LinearSplineModel::new(&sub_data);
                let (sub_alpha, sub_beta) = alpha_beta(&sub_model)?;

                // Compute err: max |pred - sa_idx| over sub-leaf keys.
                let mut sub_err = 0u64;
                for &(k, sa_idx) in items {
                    let pred = predict_clamped(sub_alpha, sub_beta, k, sa_num);
                    let d = (pred - sa_idx as i64).unsigned_abs();
                    if d > sub_err {
                        sub_err = d;
                    }
                }
                ModelEntry {
                    alpha: sub_alpha,
                    beta: sub_beta,
                    err: sub_err,
                }
            }
        };
        l1.push(entry);
    }

    // L2 entry: routing model params + encoded fallback pointer.
    Ok(ModelEntry {
        alpha: routing_alpha,
        beta: routing_beta,
        err: encode_fallback_err(partial_start, partial_num as u64),
    })
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::lookup::lookup_with_components;
    use crate::train::config::TrainerConfig;
    use crate::train::training_set::TrainingSet;

    fn make_ts(keys: Vec<u64>, sa_indices: Vec<u64>) -> TrainingSet {
        TrainingSet { keys, sa_indices }
    }

    // ── default_l2_leaf_count tests ───────────────────────────────────────

    #[test]
    fn default_l2_for_phix_scale() {
        assert_eq!(default_l2_leaf_count(5_000), 256);
    }

    #[test]
    fn default_l2_for_ecoli_scale() {
        assert_eq!(default_l2_leaf_count(5_000_000), 1 << 18);
    }

    #[test]
    fn default_l2_for_human_scale() {
        assert_eq!(default_l2_leaf_count(3_100_000_000), 1 << 27);
    }

    #[test]
    fn default_l2_floor_tiny_input() {
        assert_eq!(default_l2_leaf_count(0), 16);
    }

    #[test]
    fn default_l2_floor_small_input() {
        assert_eq!(default_l2_leaf_count(100), 16);
    }

    #[test]
    fn default_l2_ceiling_huge_input() {
        assert_eq!(default_l2_leaf_count(usize::MAX), 1 << 28);
    }

    // ── encode_fallback_err tests ─────────────────────────────────────────

    #[test]
    fn encode_fallback_err_sets_high_bit() {
        let enc = encode_fallback_err(0, 1);
        assert_ne!(enc >> 63, 0, "high bit must be set");
    }

    #[test]
    fn encode_fallback_err_round_trips() {
        let partial_start: u64 = 0x1234_5678;
        let partial_num: u64 = 0xABCD;
        let enc = encode_fallback_err(partial_start, partial_num);
        let decoded_start = (enc >> 32) & 0x7fff_ffff;
        let decoded_num = enc & 0xffff_ffff;
        assert_eq!(decoded_start, partial_start, "partial_start round-trip");
        assert_eq!(decoded_num, partial_num, "partial_num round-trip");
    }

    #[test]
    fn encode_fallback_err_zero_partial_start() {
        let enc = encode_fallback_err(0, 42);
        assert_ne!(
            enc >> 63,
            0,
            "high bit must be set even when partial_start=0"
        );
        let decoded_start = (enc >> 32) & 0x7fff_ffff;
        assert_eq!(decoded_start, 0);
        let decoded_num = enc & 0xffff_ffff;
        assert_eq!(decoded_num, 42);
    }

    // ── fit_l2_direct_perfect_line ─────────────────────────────────────────

    #[test]
    fn fit_l2_direct_perfect_line() {
        // Perfect linear dataset: sa_indices[i] = i. Predictions and the
        // runtime's clamp range both live in [0, n-1], so a well-fit
        // LinearModel should produce err ≈ 0.
        //
        // Keys start at 1 << 50 so that minus_epsilon never underflows.
        let n = 32usize;
        let key_base = 1u64 << 50;
        let key_stride = 1u64 << 46; // all keys remain in bucket 0 of a 16-leaf L2
        let keys: Vec<u64> = (0..n as u64).map(|i| key_base + i * key_stride).collect();
        let sa_indices: Vec<u64> = (0..n as u64).collect();
        let ts = make_ts(keys, sa_indices);

        // Verify all keys route to leaf 0 with bit_shift = 60.
        for &k in &ts.keys {
            assert_eq!(k >> 60, 0, "key should route to leaf 0");
        }

        let config = TrainerConfig::default();
        let model = train_with_config(&ts, 16, &config).unwrap();

        assert_eq!(model.l2.len(), 16);
        assert_eq!(model.bit_shift, 60);
        let l2_entry = model.l2[0];
        assert!(l2_entry.alpha.is_finite(), "alpha should be finite");
        assert!(l2_entry.beta.is_finite(), "beta should be finite");
        // err should be tiny for a perfect linear fit on in-range targets.
        assert!(
            l2_entry.err < 10,
            "err should be tiny for a perfect linear dataset, got alpha={} beta={} err={}",
            l2_entry.alpha,
            l2_entry.beta,
            l2_entry.err
        );
    }

    // ── tiny_training_set_one_key_per_leaf ───────────────────────────────────

    #[test]
    fn tiny_training_set_one_key_per_leaf() {
        // 16 keys, each in a different leaf of a 16-leaf L2.
        let bit_shift = 60u32;
        let keys: Vec<u64> = (0u64..16).map(|i| i << bit_shift).collect();
        let sa_indices: Vec<u64> = (0u64..16).collect();
        let ts = make_ts(keys.clone(), sa_indices.clone());

        let config = TrainerConfig::default();
        let model = train_with_config(&ts, 16, &config).unwrap();

        assert_eq!(model.l2.len(), 16);

        // Each training key must be predictable within its own err bound.
        for (k, sa_idx) in keys.iter().zip(sa_indices.iter()) {
            let (pred, err) = lookup_with_components(*k, &model.l1, &model.l2, model.bit_shift, 16);
            let d = (pred as i64 - *sa_idx as i64).unsigned_abs();
            assert!(
                d <= err,
                "key={k} pred={pred} sa_idx={sa_idx} d={d} err={err}"
            );
        }
    }

    // ── large_uniform_leaf_uses_fallback ─────────────────────────────────────

    #[test]
    fn large_uniform_leaf_uses_fallback() {
        // 20 keys all routing to leaf 0, fallback_threshold=10 → triggers fallback.
        let n = 20usize;
        let key_base = 1u64 << 50;
        let key_stride = 1u64 << 46; // all keys remain in bucket 0 of a 16-leaf L2
        let keys: Vec<u64> = (0..n as u64).map(|i| key_base + i * key_stride).collect();
        // sa_indices must span a range so end_y != start_y (avoids DBZ downgrade).
        let sa_indices: Vec<u64> = (0..n as u64).collect();
        let ts = make_ts(keys, sa_indices);

        let config = TrainerConfig {
            fallback_threshold: 10, // small enough that 20 keys triggers fallback
            ..TrainerConfig::default()
        };

        let model = train_with_config(&ts, 16, &config).unwrap();

        // The L2 entry for leaf 0 should have the fallback high-bit set.
        let l2_entry = model.l2[0];
        assert_ne!(l2_entry.err >> 63, 0, "leaf 0 should be a fallback leaf");

        // L1 array should be non-empty.
        assert!(
            !model.l1.is_empty(),
            "L1 array should have sub-leaf entries"
        );
    }
}
