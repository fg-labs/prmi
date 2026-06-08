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
use crate::train::training_set::{Keys, SaIndices, TrainingSet};
use crate::upstream::train::lower_bound_correction::LowerBoundCorrection;
use crate::upstream::{
    weighted_slr, KeyType, LinearModel, LinearSplineModel, Model, ModelParam, RMITrainingData,
    RMITrainingDataIteratorProvider,
};
use rayon::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Zero-copy [`RMITrainingDataIteratorProvider`] over a [`TrainingSet`]'s
/// `Arc`-shared key/index vectors.
///
/// It exists only to feed [`LowerBoundCorrection::new`]'s single sequential
/// pass. Sharing the `Arc`s avoids materialising a `Vec<(u64, usize)>` copy of
/// the entire training set (≈16 B/pair → ~100 GB at hg38 scale, on top of the
/// vectors it duplicates). The `(key, sa_index)` sequence it yields is identical
/// to the `Vec<(u64, usize)>` provider it replaces, so every fitted model
/// parameter — and thus the on-disk sidecar — is byte-for-byte unchanged.
struct KeySaProvider {
    keys: Keys,
    sa_indices: SaIndices,
}

impl RMITrainingDataIteratorProvider for KeySaProvider {
    type InpType = u64;

    fn len(&self) -> usize {
        self.sa_indices.len()
    }

    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (u64, usize)> + '_> {
        Box::new((0..self.sa_indices.len()).map(move |i| {
            let rank = self.sa_indices.get(i);
            (self.keys.at(i, rank), rank as usize)
        }))
    }

    fn key_type(&self) -> KeyType {
        KeyType::U64
    }

    fn get(&self, idx: usize) -> Option<(u64, usize)> {
        if idx < self.sa_indices.len() {
            let rank = self.sa_indices.get(idx);
            Some((self.keys.at(idx, rank), rank as usize))
        } else {
            None
        }
    }
}

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
    // `LowerBoundCorrection::new` (step 4) makes a single sequential pass over
    // the (key, sa_index) pairs and is the only consumer of `data`. Feed it a
    // zero-copy provider that shares the training set's `Arc`-backed vectors
    // rather than materialising a `Vec<(u64, usize)>` (≈16 B/pair, ~100 GB at
    // hg38 scale). The iteration order is identical to the old `Vec` provider,
    // so every downstream model parameter — and the sidecar — is unchanged.
    let data = RMITrainingData::<u64>::new(Box::new(KeySaProvider {
        keys: ts.keys.clone(),
        sa_indices: ts.sa_indices.clone(),
    }));

    // Resolve the global weight vector. `None` means uniform (weight=1.0).
    // Stored as a reference to avoid cloning when not needed.
    let global_weights: Option<&Vec<f64>> = ts.weights.as_ref();

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
            while i < n && (ts.key(i) >> bit_shift) as usize == leaf_idx {
                i += 1;
            }
            *range = (start, i);
        }
    }

    // ── step 6: fit leaves in parallel, then assemble serially ────────────
    // Each leaf fits independently from immutable state (`lbc`, `ts`,
    // `global_weights`), so the fits run over rayon. The order-dependent L1
    // concatenation and `partial_start` stamping happen afterward in
    // `assemble_model`, reproducing the sequential L1 layout byte-for-byte.
    let fits: Vec<LeafFit> = leaf_ranges
        .par_iter()
        .enumerate()
        .map(|(leaf_idx, &(start, end))| {
            fit_leaf(
                &lbc,
                ts,
                leaf_idx,
                start,
                end,
                bit_shift,
                sa_num,
                config,
                global_weights,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let (l1, l2) = assemble_model(fits)?;

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
/// 1. Fit a `LinearModel` on the leaf's training pairs (weighted if `weights`
///    is `Some`; uniform otherwise).
/// 2. NaN-guard the resulting (alpha, beta).
/// 3. Compute `err` = max absolute prediction error, widened by the LBC
///    neighbor-correction term (spec §5.10 step 7b).
///
/// The verify pass (error computation) always uses uniform measurement against
/// each pair's true SA index — weights affect only the fit, not the error bound.
#[allow(clippy::too_many_arguments)]
fn fit_direct_leaf(
    lbc: &LowerBoundCorrection<u64>,
    ts: &TrainingSet,
    start: usize,
    end: usize,
    leaf_idx: usize,
    bit_shift: u32,
    // Full SA size (for prediction clamping — may exceed ts.len() when masking).
    sa_num: u64,
    // Global per-pair weights. `None` means uniform (1.0).
    weights: Option<&Vec<f64>>,
) -> Result<ModelEntry> {
    // Materialise this leaf's SA-index targets once (bounded by leaf size).
    // Identical values to the former `ts.sa_indices[start..end]` slice.
    let leaf_sa: Vec<u64> = (start..end).map(|i| ts.sa_indices.get(i)).collect();
    let leaf_keys: Vec<u64> = (start..end).map(|i| ts.key(i)).collect();
    let (alpha, beta) = if let Some(ws) = weights {
        // Weighted fit using per-pair weights from the training set.
        let leaf_pairs: Vec<(f64, f64)> = leaf_keys
            .iter()
            .zip(leaf_sa.iter())
            .map(|(&k, &s)| (k as f64, s as f64))
            .collect();
        let leaf_weights: Vec<f64> = ws[start..end].to_vec();
        let (a, b) = weighted_slr(&leaf_pairs, &leaf_weights);
        (a, b)
    } else {
        // Unweighted fit (original path).
        let leaf_pairs: Vec<(u64, usize)> = leaf_keys
            .iter()
            .zip(leaf_sa.iter())
            .map(|(&k, &s)| (k, s as usize))
            .collect();
        let leaf_data = RMITrainingData::<u64>::new(Box::new(leaf_pairs));
        let model = LinearModel::new(&leaf_data);
        // Validate and extract (alpha, beta) from the model.
        let (a, b) = alpha_beta(&model)?;
        (a, b)
    };

    // Validate params (the weighted_slr path returns finite values by
    // construction, but check anyway for defence-in-depth).
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::Internal {
            detail: format!("non-finite model params: alpha={alpha}, beta={beta}"),
        });
    }

    // Compute in-leaf error: max |predict(k) - sa_idx| over all leaf keys.
    let mut err = 0u64;
    for (k, sa_idx) in leaf_keys.iter().zip(leaf_sa.iter()) {
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
    // positions still exist in the SA and must be findable by the query path.
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
            // This masked-region correction (opus-pass2 finding #1) ensures the
            // per-leaf `err` bound covers masked SA positions in the inter-leaf
            // key gap, so the query path can still resolve them.
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
/// 3. Fit a `LinearModel` routing model on the rescaled view (weighted if
///    `weights` is `Some`).
/// 4. Partition the leaf into `partial_num` sub-leaves via routing predictions.
/// 5. For each sub-leaf: fit `LinearSplineModel` (≥2 keys; weighted if
///    `weights` is `Some`), constant (1 key), or constant-to-next (empty).
/// 6. Append sub-leaf entries to `l1`; emit routing L2 entry.
///
/// The error bound computation (steps 5d and 6d) always uses unweighted
/// max-absolute-error so that the sidecar's `err` field remains a valid
/// worst-case bound for all queries, regardless of their BED-coverage status.
#[allow(clippy::too_many_arguments)]
fn fit_fallback_leaf(
    ts: &TrainingSet,
    start: usize,
    end: usize,
    start_y: usize,
    end_y: usize,
    config: &TrainerConfig,
    sa_num: u64,
    weights: Option<&Vec<f64>>,
) -> Result<FallbackFit> {
    let leaf_len = end - start;
    // This leaf's SA-index targets (bounded by leaf size); identical values to
    // the former `ts.sa_indices[start..end]` slice.
    let leaf_sa: Vec<u64> = (start..end).map(|i| ts.sa_indices.get(i)).collect();
    let leaf_keys: Vec<u64> = (start..end).map(|i| ts.key(i)).collect();

    // partial_num = ceil(leaf_len / partial_target_size). [A] audit §"Algorithmic notes".
    let partial_num = (leaf_len as u64).div_ceil(config.partial_target_size) as usize;

    // Build a scaled copy of the leaf for routing-model training.
    // Scale = (partial_num - 1) / (end_y - start_y) maps sa_indices in
    // [start_y, end_y] to the range [0, partial_num - 1].
    // We fold the offset (subtracting start_y) directly into the scaled Vec
    // since Marcus's reverted RMITrainingData has no set_offset API.
    // [A] audit §"Design decision #6 Scale/offset rescaling".
    let scale = (partial_num - 1) as f64 / (end_y - start_y) as f64;
    let scaled_f64_pairs: Vec<(f64, f64)> = leaf_keys
        .iter()
        .zip(leaf_sa.iter())
        .map(|(&k, &s)| {
            let shifted = (s as usize).saturating_sub(start_y);
            let scaled = (shifted as f64 * scale).round();
            (k as f64, scaled.min((partial_num - 1) as f64))
        })
        .collect();

    // Fit the routing LinearModel on the scaled data (weighted if prior active).
    let (routing_alpha, routing_beta) = if let Some(ws) = weights {
        let leaf_weights: Vec<f64> = ws[start..end].to_vec();
        let (a, b) = weighted_slr(&scaled_f64_pairs, &leaf_weights);
        if !a.is_finite() || !b.is_finite() {
            return Err(Error::Internal {
                detail: format!("non-finite routing model params: alpha={a}, beta={b}"),
            });
        }
        (a, b)
    } else {
        // Unweighted: use original RMITrainingData path.
        let scaled_pairs: Vec<(u64, usize)> = scaled_f64_pairs
            .iter()
            .map(|&(k, v)| (k as u64, v as usize))
            .collect();
        let scaled_data = RMITrainingData::<u64>::new(Box::new(scaled_pairs));
        let routing_model = LinearModel::new(&scaled_data);
        alpha_beta(&routing_model)?
    };

    // ── partition leaf into sub-leaves ────────────────────────────────────
    // `sub_leaf_items[s]` = Vec of (key, sa_index, weight) triples routed to sub-leaf s.
    // Routing: clamp(routing_alpha + routing_beta * key, 0, partial_num - 1) using
    // truncation (not rounding) to match the runtime's `clamp_to_int` behavior.
    // Weight is always 1.0 when no prior is active; this keeps the sub-leaf
    // fitting path uniform regardless of whether weights are present.
    let mut sub_leaf_items: Vec<Vec<(u64, usize, f64)>> = vec![Vec::new(); partial_num];
    for (i, (&k, &sa_idx)) in leaf_keys.iter().zip(leaf_sa.iter()).enumerate() {
        let w = weights.map_or(1.0, |ws| ws[start + i]);
        let raw = routing_alpha + routing_beta * k as f64;
        let sub_idx = if raw.is_nan() {
            0
        } else {
            raw.clamp(0.0, (partial_num - 1) as f64) as usize
        };
        sub_leaf_items[sub_idx].push((k, sa_idx as usize, w));
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

    // ── fit each sub-leaf into this leaf's own L1 buffer ──────────────────
    // `partial_start` (the offset into the global L1 array) is NOT read here:
    // it depends on every prior leaf's L1 count, which is only known in the
    // serial assembly phase. Computing it here would force this fit to run in
    // leaf order. Instead the entries are buffered and concatenated later.
    let mut l1_entries: Vec<ModelEntry> = Vec::with_capacity(partial_num);
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
                // Two or more keys: fit LinearSplineModel (weighted if prior active).
                // [A] audit §Q4 decision.
                let sub_f64_pairs: Vec<(f64, f64)> = items
                    .iter()
                    .map(|&(k, sa, _)| (k as f64, sa as f64))
                    .collect();
                let sub_weights: Vec<f64> = items.iter().map(|&(_, _, w)| w).collect();
                let sub_model = LinearSplineModel::new_weighted(&sub_f64_pairs, &sub_weights);
                let sub_params = sub_model.params();
                let sub_alpha = sub_params[0].as_float();
                let sub_beta = sub_params[1].as_float();
                if !sub_alpha.is_finite() || !sub_beta.is_finite() {
                    return Err(Error::Internal {
                        detail: format!(
                            "non-finite sub-leaf model params: alpha={sub_alpha}, beta={sub_beta}"
                        ),
                    });
                }

                // Compute err: max |pred - sa_idx| over sub-leaf keys.
                // Error is always computed without weights (uniform verification truth).
                let mut sub_err = 0u64;
                for &(k, sa_idx, _) in items {
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
        l1_entries.push(entry);
    }

    Ok(FallbackFit {
        routing_alpha,
        routing_beta,
        partial_num,
        l1_entries,
    })
}

/// A fitted fallback leaf, minus the global L1 offset (`partial_start`), which
/// is assigned during serial assembly. See [`fit_fallback_leaf`].
struct FallbackFit {
    routing_alpha: f64,
    routing_beta: f64,
    partial_num: usize,
    l1_entries: Vec<ModelEntry>,
}

/// Result of fitting one L2 leaf, ready for serial assembly into the model.
enum LeafFit {
    /// A final L2 entry (empty or direct leaf) with no L1 entries.
    Entry(ModelEntry),
    /// A fallback leaf whose L2 `err` (encoding `partial_start`) and L1 entries
    /// are stitched in during serial assembly.
    Fallback(FallbackFit),
}

/// Fit a single L2 leaf `[start, end)` — empty, direct, or fallback — into a
/// [`LeafFit`].
///
/// Reads only immutable state (`lbc`, `ts`, `weights`), so leaves fit
/// independently and in parallel. All per-leaf floating-point math runs on one
/// thread, so the result is bit-identical to the serial fit regardless of
/// scheduling; the order-dependent L1 assembly is deferred to
/// [`assemble_model`].
#[allow(clippy::too_many_arguments)]
fn fit_leaf(
    lbc: &LowerBoundCorrection<u64>,
    ts: &TrainingSet,
    leaf_idx: usize,
    start: usize,
    end: usize,
    bit_shift: u32,
    sa_num: u64,
    config: &TrainerConfig,
    weights: Option<&Vec<f64>>,
) -> Result<LeafFit> {
    let leaf_len = end - start;

    // ── case a: empty leaf ────────────────────────────────────────────────
    if leaf_len == 0 {
        let entry = if let Some((next_sa_idx, _)) = lbc.next_real(leaf_idx) {
            // Normal empty leaf: err = next_sa_idx makes the window
            // [0, 2*next_sa_idx+1) cover all masked SA positions in the gap.
            ModelEntry {
                alpha: next_sa_idx as f64,
                beta: 0.0,
                err: next_sa_idx as u64,
            }
        } else {
            // Trailing empty leaf: centre a constant model over the valid tail
            // [prev_last_sa + 1, sa_num - 1] (or the whole SA if no predecessor).
            let (lo, hi) = if let Some((prev_sa_idx, _)) = lbc.prev_real(leaf_idx) {
                (prev_sa_idx as u64 + 1, sa_num.saturating_sub(1))
            } else {
                (0u64, sa_num.saturating_sub(1))
            };
            let mid = (lo + hi) / 2;
            // `mid` is the floor midpoint, so the farthest valid position is
            // `ceil((hi - lo) / 2)` away — no `+1` (that overstated the radius by
            // one for odd-length tails, widening the search window needlessly).
            let radius = hi.saturating_sub(lo).div_ceil(2);
            ModelEntry {
                alpha: mid as f64,
                beta: 0.0,
                err: radius,
            }
        };
        return Ok(LeafFit::Entry(entry));
    }

    // ── case b: direct leaf ───────────────────────────────────────────────
    if leaf_len <= config.fallback_threshold {
        return Ok(LeafFit::Entry(fit_direct_leaf(
            lbc, ts, start, end, leaf_idx, bit_shift, sa_num, weights,
        )?));
    }

    // ── case c: large leaf — fallback, downgrading to direct on a DBZ ─────
    let start_y = ts.sa_indices.get(start) as usize;
    let end_y = ts.sa_indices.get(end - 1) as usize;
    if end_y == start_y {
        // All keys map to the same SA position; a fallback rescale would divide
        // by zero, so fit a direct leaf instead. [A] audit §Issues "DBZ".
        return Ok(LeafFit::Entry(fit_direct_leaf(
            lbc, ts, start, end, leaf_idx, bit_shift, sa_num, weights,
        )?));
    }
    Ok(LeafFit::Fallback(fit_fallback_leaf(
        ts, start, end, start_y, end_y, config, sa_num, weights,
    )?))
}

/// Serially assemble fitted leaves into the `(l1, l2)` model arrays in leaf
/// order, stamping each fallback leaf's `partial_start` (its offset into the
/// growing L1 array). This reproduces the exact L1 byte layout of the original
/// sequential fit, so the model is byte-for-byte identical.
fn assemble_model(fits: Vec<LeafFit>) -> Result<(Vec<ModelEntry>, Vec<ModelEntry>)> {
    let mut l1: Vec<ModelEntry> = Vec::new();
    let mut l2: Vec<ModelEntry> = Vec::with_capacity(fits.len());
    for fit in fits {
        match fit {
            LeafFit::Entry(entry) => l2.push(entry),
            LeafFit::Fallback(fb) => {
                let partial_start = l1.len() as u64;
                // Overflow check on the final offset (encode_fallback_err also
                // catches this, but this gives a clearer error first).
                if partial_start >= (1u64 << 31) {
                    return Err(Error::Internal {
                        detail: format!(
                            "partial_start={partial_start} exceeds 2^31 - 1 cap (brief §4.4)"
                        ),
                    });
                }
                l1.extend(fb.l1_entries);
                l2.push(ModelEntry {
                    alpha: fb.routing_alpha,
                    beta: fb.routing_beta,
                    err: encode_fallback_err(partial_start, fb.partial_num as u64)?,
                });
            }
        }
    }
    Ok((l1, l2))
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::lookup::lookup_with_components;
    use crate::train::config::TrainerConfig;
    use crate::train::training_set::TrainingSet;
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
        for k in ts.keys_iter() {
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
