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
/// Targets ~12–24 SA entries per L2 leaf, derived from BWA-MEME's empirical
/// human-genome ratio (3.1 Gbp / 2^28 ≈ 12). Rounds *down* to a power of two
/// (so the realized ratio is up to ~2x the target) and clamps to `[2^4, 2^28]`:
/// the lower bound is the pwl<4 floor (`l2_leaf_count < 16` is rejected by the
/// trainer); the upper bound is BWA-MEME's largest published configuration.
///
/// # Examples
///
/// ```ignore
/// default_l2_leaf_count(5_000)           // phiX (~5 kb) → 256 (2^8)
/// default_l2_leaf_count(5_000_000)       // E. coli (~5 Mbp) → 2^18
/// default_l2_leaf_count(3_100_000_000)   // hg38 (~3.1 Gbp) → 2^27
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
///
/// Returns an error if `partial_start >= 2^31` or `partial_num >= 2^32` to
/// catch overflow in release builds, not just debug builds.
#[inline]
fn encode_fallback_err(partial_start: u64, partial_num: u64) -> Result<u64> {
    if partial_start >= (1u64 << 31) {
        return Err(Error::Internal {
            detail: format!("partial_start={partial_start} exceeds 2^31 - 1 cap (brief §4.4)"),
        });
    }
    if partial_num >= (1u64 << 32) {
        return Err(Error::Internal {
            detail: format!("partial_num={partial_num} exceeds 2^32 - 1 cap (brief §4.4)"),
        });
    }
    Ok((1u64 << 63) | ((partial_start & 0x7fff_ffff) << 32) | (partial_num & 0xffff_ffff))
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
    if config.partial_target_size == 0 {
        return Err(Error::Internal {
            detail: "TrainerConfig.partial_target_size must be > 0".into(),
        });
    }
    if config.fallback_threshold == 0 {
        return Err(Error::Internal {
            detail: "TrainerConfig.fallback_threshold must be > 0".into(),
        });
    }
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
    // The full SA size governs prediction clamping. When masking is active,
    // ts.sa_num > ts.len(), and using ts.sa_num ensures predictions are not
    // clamped too aggressively (which would inflate the error bound).
    let sa_num = ts.sa_num;

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
            // SA index. `lbc.next_real(leaf_idx)` is the SA position of the
            // first key in the next non-empty leaf (or None if none exists).
            //
            // In the unmasked (dense) case no query routes to an empty leaf
            // because every key in the reference has a training pair. Under
            // masking, the masked region's keys are absent from training, so
            // empty leaves CAN receive queries at runtime.
            //
            // For such queries the true SA position is somewhere in the SA
            // range [prev_last_sa + 1, next_first_sa - 1]. The model predicts
            // next_first_sa (= const_pred). To guarantee the search window
            // covers the entire gap, set err = const_pred. This makes
            // lo = 0 (conservative but correct) and hi = 2 * const_pred + 1.
            if let Some((next_sa_idx, _)) = lbc.next_real(leaf_idx) {
                // Normal empty leaf: a next non-empty leaf exists.
                // err = next_sa_idx guarantees the window [0, 2*next_sa_idx+1)
                // covers all masked SA positions in [prev_last_sa+1, next_sa_idx-1].
                let const_pred = next_sa_idx as f64;
                let err = next_sa_idx as u64;
                l2.push(ModelEntry {
                    alpha: const_pred,
                    beta: 0.0,
                    err,
                });
            } else {
                // ── case a-trailing: no next non-empty leaf exists ────────
                // This is the trailing-empty-leaf case: masked 32-mers whose
                // keys lex-sort above every training key route here. Their true
                // SA ranks lie in [prev_last_sa + 1, sa_num - 1].
                //
                // Setting err = 0 (as the old code did) produces a 1-slot
                // window that misses the valid SA tail. Instead, emit a
                // centred constant model that covers the entire valid range.
                let (lo, hi) = if let Some((prev_sa_idx, _)) = lbc.prev_real(leaf_idx) {
                    // A previous non-empty leaf gives us a tight lower bound.
                    let lo = prev_sa_idx as u64 + 1;
                    let hi = sa_num.saturating_sub(1);
                    (lo, hi)
                } else {
                    // No preceding leaf either: cover the whole SA.
                    (0u64, sa_num.saturating_sub(1))
                };
                let mid = (lo + hi) / 2;
                let radius = hi.saturating_sub(lo).saturating_add(1).div_ceil(2);
                l2.push(ModelEntry {
                    alpha: mid as f64,
                    beta: 0.0,
                    err: radius,
                });
            }
            continue;
        }

        if leaf_len <= config.fallback_threshold {
            // ── case b: direct leaf (DIRECT path) ────────────────────────
            l2.push(fit_direct_leaf(
                &lbc, ts, start, end, leaf_idx, bit_shift, sa_num,
            )?);
        } else {
            // ── case c: large leaf — try fallback, downgrade if needed ───
            let start_y = ts.sa_indices[start] as usize;
            let end_y = ts.sa_indices[end - 1] as usize;

            if end_y == start_y {
                // DBZ guard: all keys in this leaf map to the same SA position.
                // Downgrade to direct fit. [A] audit, §Issues "Division by zero".
                l2.push(fit_direct_leaf(
                    &lbc, ts, start, end, leaf_idx, bit_shift, sa_num,
                )?);
            } else {
                l2.push(fit_fallback_leaf(
                    ts, start, end, start_y, end_y, &mut l1, config, sa_num,
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
    // Full SA size (for prediction clamping — may exceed ts.len() when masking).
    sa_num: u64,
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
        let pred = predict_clamped(alpha, beta, *k, sa_num);
        let d = (pred - *sa_idx as i64).unsigned_abs();
        if d > err {
            err = d;
        }
    }

    // LBC neighbor-correction: widen err by the distance from the model's
    // prediction at the next leaf's first key vs the last SA position reachable
    // by any query routing to THIS leaf.
    //
    // Queries whose key k lex-sorts in (this_leaf_last_key, next_leaf_first_key)
    // route to this leaf (k >> bit_shift == leaf_idx). For masked references the
    // SA is complete but the training set is sparse, so that key gap can contain
    // 32-mers whose underlying genome positions were excluded from training. Those
    // positions still exist in the SA and must be found by `smem_range`.
    //
    // The correct upper bound for the correction is the SA index just BELOW the
    // next leaf's first training pair: `next_sa_idx - 1`. Using the current
    // leaf's last TRAINING SA index (`ts.sa_indices[end-1]`) would be strictly
    // tighter under masking — it misses the masked SA positions in the gap.
    //
    // Use `next_real` rather than `next` + `is_next_real`: the `Option` return
    // makes the sentinel encoding self-evident. The old sentinel stored
    // `(num_keys, T::max_value())`, but `T::max_value() == u64::MAX` is also a
    // valid tokenisation output for a real all-T (TTTT…T) 32-mer; comparing by
    // value alone would be ambiguous. `next_real` returns `None` exclusively for
    // the trailing group, `Some` for every leaf that has a real successor.
    if let Some((next_sa_idx, next_key)) = lbc.next_real(leaf_idx) {
        // Only apply if the next key routes to a different L2 bucket (otherwise
        // there is no inter-leaf gap — both keys share the same leaf).
        let next_key_bucket = (if bit_shift >= 64 {
            0u64
        } else {
            next_key >> bit_shift
        }) as usize;
        if next_key_bucket != leaf_idx {
            // `next_sa_idx` is the SA index of the next leaf's first training
            // pair. The highest SA index any query routing to THIS leaf can have
            // is `next_sa_idx - 1` (the position just below). Use that as the
            // reference point for the distance calculation.
            //
            // See the `smem_range_resolves_masked_region_query` regression test
            // (opus-pass2 finding #1) for the end-to-end proof that this bound
            // is necessary.
            let last_routable_sa_idx = next_sa_idx.saturating_sub(1) as i64;
            let pred_next = predict_clamped(alpha, beta, next_key, sa_num);
            let d = (pred_next - last_routable_sa_idx).unsigned_abs();
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
    // Runtime overflow check — encode_fallback_err will also catch this,
    // but checking here gives a clearer error before we append any sub-leaf
    // entries to l1.
    if partial_start >= (1u64 << 31) {
        return Err(Error::Internal {
            detail: format!("partial_start={partial_start} exceeds 2^31 - 1 cap (brief §4.4)"),
        });
    }

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
        err: encode_fallback_err(partial_start, partial_num as u64)?,
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
        let sa_num = sa_indices.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        TrainingSet {
            keys,
            sa_indices,
            sa_num,
        }
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
        let enc = encode_fallback_err(0, 1).unwrap();
        assert_ne!(enc >> 63, 0, "high bit must be set");
    }

    #[test]
    fn encode_fallback_err_round_trips() {
        let partial_start: u64 = 0x1234_5678;
        let partial_num: u64 = 0xABCD;
        let enc = encode_fallback_err(partial_start, partial_num).unwrap();
        let decoded_start = (enc >> 32) & 0x7fff_ffff;
        let decoded_num = enc & 0xffff_ffff;
        assert_eq!(decoded_start, partial_start, "partial_start round-trip");
        assert_eq!(decoded_num, partial_num, "partial_num round-trip");
    }

    #[test]
    fn encode_fallback_err_zero_partial_start() {
        let enc = encode_fallback_err(0, 42).unwrap();
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

    #[test]
    fn encode_fallback_err_rejects_overflow_partial_start() {
        let result = encode_fallback_err(1u64 << 31, 1);
        assert!(result.is_err(), "partial_start=2^31 should return Err");
    }

    #[test]
    fn encode_fallback_err_rejects_overflow_partial_num() {
        let result = encode_fallback_err(0, 1u64 << 32);
        assert!(result.is_err(), "partial_num=2^32 should return Err");
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

    // ── TrainerConfig validation ──────────────────────────────────────────────

    #[test]
    fn train_with_config_rejects_zero_partial_target_size() {
        let ts = make_ts(vec![0u64, 1 << 60], vec![0u64, 1]);
        let config = TrainerConfig {
            partial_target_size: 0,
            ..TrainerConfig::default()
        };
        let result = train_with_config(&ts, 16, &config);
        assert!(result.is_err(), "partial_target_size=0 should return Err");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("partial_target_size"),
            "error should mention partial_target_size, got: {msg}"
        );
    }

    #[test]
    fn train_with_config_rejects_zero_fallback_threshold() {
        let ts = make_ts(vec![0u64, 1 << 60], vec![0u64, 1]);
        let config = TrainerConfig {
            fallback_threshold: 0,
            ..TrainerConfig::default()
        };
        let result = train_with_config(&ts, 16, &config);
        assert!(result.is_err(), "fallback_threshold=0 should return Err");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("fallback_threshold"),
            "error should mention fallback_threshold, got: {msg}"
        );
    }
}
