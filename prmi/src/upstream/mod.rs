// Copyright Ryan Marcus 2020
// Modified by Fulcrum Genomics 2026

#![allow(clippy::all)]
#![allow(unused_imports, unused_variables, unused_assignments, dead_code)]
#![allow(missing_docs)] // upstream code carries its own (often minimal) docs from Marcus 2020 + BWA-MEME

mod models;
pub mod train;

pub(crate) use models::weighted_slr;
pub use models::KeyType;
pub use models::{
    LinearModel, LinearSplineModel, Model, ModelInput, ModelParam, RMITrainingData,
    RMITrainingDataIteratorProvider,
};
// `train` and `TrainedRMI` are from Marcus's original two_layer.rs. They are
// only used via the Fulcrum trainer in train/trainer.rs (through
// LowerBoundCorrection). The TrainingKey::minus_epsilon impl for u64/u32 has
// an underflow bug at key=0 (wraps to u64::MAX in release, panics in debug).
// Gate the whole module pub(crate) so it is not reachable from downstream
// users of the prmi crate.
pub(crate) use train::{train, TrainedRMI};
