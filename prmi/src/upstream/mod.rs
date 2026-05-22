// Copyright Ryan Marcus 2020
// Modified by Fulcrum Genomics 2026

#![allow(clippy::all)]
#![allow(unused_imports, unused_variables, unused_assignments, dead_code)]
#![allow(missing_docs)] // upstream code carries its own (often minimal) docs from Marcus 2020 + BWA-MEME

#[allow(
    clippy::all,
    unused_imports,
    unused_variables,
    unused_assignments,
    dead_code
)]
mod models;
#[allow(
    clippy::all,
    unused_imports,
    unused_variables,
    unused_assignments,
    dead_code
)]
pub mod train;

pub use models::KeyType;
pub use models::{
    LinearModel, LinearSplineModel, Model, ModelInput, ModelParam, RMITrainingData,
    RMITrainingDataIteratorProvider,
};
pub use train::{train, TrainedRMI};
