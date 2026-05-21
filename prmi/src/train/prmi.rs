// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! P-RMI training driver. Wraps Marcus's upstream + BWA-MEME's
//! `pwl,linear,linear_spline` trainer and reshapes the resulting trained
//! parameters into the (L1 fallback array, L2 routing layer) layout the
//! prmi reader consumes — see v0.1 brief §4 / §4.4.

use crate::error::{Error, Result};
use crate::sidecar::model_file::ModelEntry;
use crate::train::training_set::TrainingSet;
use crate::upstream::{train, Model, ModelParam, RMITrainingData, TrainedRMI};

/// An in-memory P-RMI model ready to be serialized into the `.l1` / `.l2`
/// sidecar files.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PrmiModel {
    /// Flat L1 fallback array. Empty when no L2 leaf required a fallback.
    pub l1: Vec<ModelEntry>,
    /// L2 routing layer. Length always equals `l2_leaf_count`.
    pub l2: Vec<ModelEntry>,
    /// `bit_shift = 64 - log2(l2_leaf_count)`. Reader uses this to compute
    /// the L2 index from a key.
    pub bit_shift: u32,
}

/// Train a P-RMI on the supplied training set. `l2_leaf_count` must be a
/// power of two ≥ 2. The training-set keys must be non-decreasing.
pub fn train_prmi(ts: &TrainingSet, l2_leaf_count: u64) -> Result<PrmiModel> {
    if !l2_leaf_count.is_power_of_two() || l2_leaf_count < 2 {
        return Err(Error::Internal {
            detail: format!("l2_leaf_count={l2_leaf_count} must be a power of two ≥ 2"),
        });
    }
    if ts.is_empty() {
        return Err(Error::Internal {
            detail: "empty training set".into(),
        });
    }
    let bit_shift = 64 - l2_leaf_count.trailing_zeros();
    let pwl_bits = l2_leaf_count.trailing_zeros();

    // Build the upstream training data: (key, sa_index) pairs.
    let data: Vec<(u64, usize)> = ts
        .keys
        .iter()
        .zip(ts.sa_indices.iter())
        .map(|(&k, &idx)| (k, idx as usize))
        .collect();
    let td = RMITrainingData::new(Box::new(data));

    // The spec string "pwlN,linear,linear_spline" — N must match the bit-width.
    let spec = format!("pwl{pwl_bits},linear,linear_spline");
    let trained = train(&td, &spec, l2_leaf_count);

    let (l1, l2) = unpack_trained_to_l1_l2(&trained, l2_leaf_count as usize)?;
    Ok(PrmiModel { l1, l2, bit_shift })
}

/// Walk the trained 3-layer (top + partial_3rd_models + sec_models) RMI and
/// emit our `.l2` (routing layer) and `.l1` (flat fallback array) entries.
fn unpack_trained_to_l1_l2(
    trained: &TrainedRMI,
    l2_leaf_count: usize,
) -> Result<(Vec<ModelEntry>, Vec<ModelEntry>)> {
    if trained.rmi.len() != 3 {
        return Err(Error::Internal {
            detail: format!(
                "expected trained.rmi.len() == 3 (top + L1 + L2), got {}",
                trained.rmi.len()
            ),
        });
    }
    let sec_models = &trained.rmi[2];
    if sec_models.len() != l2_leaf_count {
        return Err(Error::Internal {
            detail: format!(
                "expected sec_models.len() == {l2_leaf_count}, got {}",
                sec_models.len()
            ),
        });
    }
    if trained.last_layer_max_l1s.len() != l2_leaf_count {
        return Err(Error::Internal {
            detail: format!(
                "expected last_layer_max_l1s.len() == {l2_leaf_count}, got {}",
                trained.last_layer_max_l1s.len()
            ),
        });
    }

    // L2: one entry per sec_model with err from last_layer_max_l1s.
    let mut l2: Vec<ModelEntry> = Vec::with_capacity(l2_leaf_count);
    for (i, model) in sec_models.iter().enumerate() {
        let (alpha, beta) = extract_alpha_beta(model.as_ref())?;
        let err = trained.last_layer_max_l1s[i];
        l2.push(ModelEntry { alpha, beta, err });
    }

    // L1: flat fallback array, only populated if any L2 leaf has the fallback
    // bit (bit 63) set in last_layer_max_l1s. When trained.rmi[1] is the dummy
    // ([dummy_model]) case, third_layer_max_l1s is empty and we emit no L1
    // entries.
    let mut l1: Vec<ModelEntry> = Vec::new();
    let has_fallback = trained.last_layer_max_l1s.iter().any(|&e| e >> 63 != 0);
    if has_fallback {
        let partial_models = &trained.rmi[1];
        if trained.third_layer_max_l1s.len() != partial_models.len() {
            return Err(Error::Internal {
                detail: format!(
                    "third_layer_max_l1s.len()={} but rmi[1].len()={} — \
                     trainer produced inconsistent L1 shape",
                    trained.third_layer_max_l1s.len(),
                    partial_models.len()
                ),
            });
        }
        l1.reserve(partial_models.len());
        for (j, model) in partial_models.iter().enumerate() {
            let (alpha, beta) = extract_alpha_beta(model.as_ref())?;
            let err = trained.third_layer_max_l1s[j];
            l1.push(ModelEntry { alpha, beta, err });
        }
    }

    Ok((l1, l2))
}

/// Extract the (alpha, beta) parameters from a LinearModel or
/// LinearSplineModel. Both expose them as the first two `ModelParam::Float`
/// entries from `Model::params()`.
fn extract_alpha_beta(model: &dyn Model) -> Result<(f64, f64)> {
    let params = model.params();
    if params.len() < 2 {
        return Err(Error::Internal {
            detail: format!(
                "expected at least 2 ModelParams for L1/L2 model, got {}",
                params.len()
            ),
        });
    }
    let alpha = match &params[0] {
        ModelParam::Float(v) => *v,
        other => {
            return Err(Error::Internal {
                detail: format!("expected ModelParam::Float for alpha, got {other:?}"),
            });
        }
    };
    let beta = match &params[1] {
        ModelParam::Float(v) => *v,
        other => {
            return Err(Error::Internal {
                detail: format!("expected ModelParam::Float for beta, got {other:?}"),
            });
        }
    };
    Ok((alpha, beta))
}
