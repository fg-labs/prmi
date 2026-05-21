// < begin copyright >
// Copyright Ryan Marcus 2020
// Modified by Fulcrum Genomics 2026
//
// See root directory of this project for license terms.
//
// < end copyright >

use super::*;

#[allow(clippy::float_cmp)]
fn linear_splines<T: TrainingKey>(data: &RMITrainingData<T>) -> (f64, f64) {
    if data.len() == 0 {
        return (0.0, 0.0);
    }

    if data.len() == 1 {
        return (data.get(0).1 as f64, 0.0);
    }

    let first_pt = data.get(0);
    let last_pt = data.get(data.len() - 1);

    if first_pt.0 == last_pt.0 {
        // data is all duplicates!
        return (data.get(0).1 as f64, 0.0);
    }

    let slope =
        (first_pt.1 as f64 - last_pt.1 as f64) / (first_pt.0.as_float() - last_pt.0.as_float());
    let intercept = first_pt.1 as f64 - slope * first_pt.0.as_float();

    return (intercept, slope);
}

pub struct LinearSplineModel {
    params: (f64, f64),
}

impl LinearSplineModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> LinearSplineModel {
        return LinearSplineModel {
            params: linear_splines(data),
        };
    }
}

impl Model for LinearSplineModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let (alpha, beta) = self.params;
        return beta.mul_add(inp.as_float(), alpha);
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("linear");
    }

    fn set_to_constant_model(&mut self, constant: u64) -> bool {
        self.params = (constant as f64, 0.0);
        return true;
    }
}
