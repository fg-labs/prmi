// Copyright Ryan Marcus 2020          (origin: learnedsystems/RMI)
// Copyright 2022 Youngmok Jung et al. (origin: kaist-ina/BWA-MEME RMI fork)
// Modified by Fulcrum Genomics 2026
// SPDX-License-Identifier: MIT

use super::*;

pub struct PiecewiselinearModel {
    params: u64,
}

impl PiecewiselinearModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>, params: u64) -> PiecewiselinearModel {
        let _ = data;
        return PiecewiselinearModel { params };
    }
}

impl Model for PiecewiselinearModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let kmer = self.params;

        return (inp.as_int() >> (64 - kmer)) as f64;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.into()];
    }

    fn code(&self) -> String {
        return format!(
            "
inline uint64_t pwl(uint64_t kmer, uint64_t inp) {{

    return inp >> (64-kmer);
}}"
        );
    }

    fn function_name(&self) -> String {
        return String::from("pwl");
    }

    fn set_to_constant_model(&mut self, _constant: u64) -> bool {
        return false;
    }
}
