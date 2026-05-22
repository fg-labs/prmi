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
mod cache_fix;
#[allow(
    clippy::all,
    unused_imports,
    unused_variables,
    unused_assignments,
    dead_code
)]
mod codegen;
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
mod train;

#[allow(
    clippy::all,
    unused_imports,
    unused_variables,
    unused_assignments,
    dead_code
)]
pub mod optimizer;
pub use codegen::output_rmi;
pub use codegen::rmi_size;
pub use models::KeyType;
pub use models::{Model, ModelInput, ModelParam, RMITrainingData, RMITrainingDataIteratorProvider};
pub use optimizer::find_pareto_efficient_configs;
pub use train::{train, train_bounded, train_for_size, TrainedRMI};
