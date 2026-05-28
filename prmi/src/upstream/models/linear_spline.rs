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

    // Two distinct integer keys can still collide in f64 once the gap is
    // smaller than the f64 ULP at that magnitude — for u64 this kicks in
    // near 2^53 and degrades further toward 2^64. A T-rich 32-mer key
    // (bases packed MSB-first as binary 11s) sits near u64::MAX, where
    // ~4096 distinct u64 values share one f64 representation. Without
    // this guard the slope below divides by 0.0 and emits ±inf.
    if first_pt.0.as_float() == last_pt.0.as_float() {
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

    /// Fit a `LinearSplineModel` using per-pair weights.
    ///
    /// Because `linear_splines` is a two-point fit (first key, last key), the
    /// weighting strategy is: among all pairs with weight > 0, pick the one
    /// with the smallest key as the first anchor and the one with the largest
    /// key as the last anchor, then use Marcus's two-point slope formula on
    /// those anchors.
    ///
    /// In practice all pairs have weight >= 1.0 under the BED prior, so the
    /// anchors are always the first and last elements of the sorted training
    /// slice — identical to the unweighted path. The method is kept general for
    /// future callers that might zero-out pairs.
    ///
    /// `pairs` is `(key_f64, sa_index_f64)` in non-decreasing key order.
    /// `weights` must have the same length as `pairs`.
    pub fn new_weighted(pairs: &[(f64, f64)], weights: &[f64]) -> LinearSplineModel {
        assert_eq!(pairs.len(), weights.len(), "pairs and weights must match");

        // Find first and last pair with weight > 0.
        let first = pairs
            .iter()
            .zip(weights.iter())
            .find(|(_, &w)| w > 0.0)
            .map(|(p, _)| *p);
        let last = pairs
            .iter()
            .zip(weights.iter())
            .rfind(|(_, &w)| w > 0.0)
            .map(|(p, _)| *p);

        let params = match (first, last) {
            (None, _) | (_, None) => (0.0, 0.0),
            (Some(f), Some(l)) if f == l => (f.1, 0.0),
            (Some((fx, fy)), Some((lx, ly))) => {
                if fx == lx {
                    // f64 key collision — constant model.
                    (fy, 0.0)
                } else {
                    let slope = (fy - ly) / (fx - lx);
                    let intercept = fy - slope * fx;
                    if !slope.is_finite() || !intercept.is_finite() {
                        (fy, 0.0)
                    } else {
                        (intercept, slope)
                    }
                }
            }
        };
        LinearSplineModel { params }
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
