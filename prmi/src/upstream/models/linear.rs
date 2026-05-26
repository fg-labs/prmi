// < begin copyright >
// Copyright Ryan Marcus 2020
// Modified by Fulcrum Genomics 2026
//
// See root directory of this project for license terms.
//
// < end copyright >

use super::*;

fn slr<T: Iterator<Item = (f64, f64)>>(loc_data: T) -> (f64, f64) {
    // compute the covariance of x and y as well as the variance of x in
    // a single pass.

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n: u64 = 0;
    let mut m2 = 0.0;

    let mut data_size = 0;
    for (x, y) in loc_data {
        n += 1;
        let dx = x - mean_x;
        mean_x += dx / (n as f64);
        mean_y += (y - mean_y) / (n as f64);
        c += dx * (y - mean_y);

        let dx2 = x - mean_x;
        m2 += dx * dx2;
        data_size += 1;
    }

    // special case when we have 0 or 1 items
    if data_size == 0 {
        return (0.0, 0.0);
    }

    if data_size == 1 {
        return (mean_y, 0.0);
    }

    let cov = c / ((n - 1) as f64);
    let var = m2 / ((n - 1) as f64);
    assert!(
        var >= 0.0,
        "variance of model with {} data items was negative",
        n
    );

    if var == 0.0 {
        // variance is zero. pick the mean (only) value.
        return (mean_y, 0.0);
    }

    // Pre-division variance guard. Even when var > 0.0, a subnormal var
    // (~10^-308) produces a wildly amplified beta.
    //
    // Threshold: `f64::EPSILON * mean_x.abs().max(1.0)` is the round-off floor
    // of `mean_x` itself. A variance below that floor means the per-sample
    // deviations `(x_i - mean_x)` are at or below `mean_x`'s representation
    // error — equivalently, all keys collapse to a single f64 representation
    // and the effective key range is zero in f64 precision. The clamp to 1.0
    // keeps the threshold sensible when `mean_x` is near zero. Treat that
    // case as a constant model rather than dividing by a near-zero `var`.
    if !var.is_finite() || var < f64::EPSILON * mean_x.abs().max(1.0) {
        return (mean_y, 0.0);
    }

    let beta: f64 = cov / var;
    // Post-division guard: NaN/Inf from f64 ops still possible if cov was
    // abnormal despite the pre-division check above. Defence-in-depth.
    if !beta.is_finite() {
        return (mean_y, 0.0);
    }

    // Magnitude cap. Any SA address is < 2^60 (~10^18), so a beta with
    // |beta| > 1e18 means pred = alpha + beta*key saturates at clamp time for
    // any positive key. Emitting a constant model avoids wasted f64 work in
    // the lookup path and keeps err bounds tight.
    if beta.abs() > 1e18 {
        return (mean_y, 0.0);
    }
    let alpha = mean_y - beta * mean_x;

    return (alpha, beta);
}

/// Weighted simple linear regression using a numerically stable two-pass
/// algorithm.
///
/// Weights are non-negative. The weighted mean is computed first, then the
/// weighted covariance and variance in a second pass. This avoids Kahan
/// cancellation issues that can arise in a single-pass formulation when weights
/// vary by a large factor (e.g. 1.0 vs 10.0 for BED-prior training).
///
/// Returns `(intercept, slope)` using the same fall-through guards as [`slr`]:
/// zero total weight, zero variance, non-finite intermediates, and extreme
/// slopes all collapse to a constant model.
pub(crate) fn weighted_slr(pairs: &[(f64, f64)], weights: &[f64]) -> (f64, f64) {
    assert_eq!(
        pairs.len(),
        weights.len(),
        "pairs and weights must have the same length"
    );

    // --- pass 1: weighted means -----------------------------------------------
    let mut w_sum = 0.0f64;
    let mut wx_sum = 0.0f64;
    let mut wy_sum = 0.0f64;
    for (&(x, y), &w) in pairs.iter().zip(weights.iter()) {
        w_sum += w;
        wx_sum += w * x;
        wy_sum += w * y;
    }

    if w_sum == 0.0 || pairs.is_empty() {
        return (0.0, 0.0);
    }

    if pairs.len() == 1 {
        return (pairs[0].1, 0.0);
    }

    let mean_x = wx_sum / w_sum;
    let mean_y = wy_sum / w_sum;

    // --- pass 2: weighted covariance and variance of x -----------------------
    let mut wcov = 0.0f64;
    let mut wvar = 0.0f64;
    for (&(x, y), &w) in pairs.iter().zip(weights.iter()) {
        let dx = x - mean_x;
        wcov += w * dx * (y - mean_y);
        wvar += w * dx * dx;
    }

    if wvar == 0.0 {
        return (mean_y, 0.0);
    }

    // Pre-division guard: same rationale as in `slr` — subnormal wvar.
    if !wvar.is_finite() || wvar < f64::EPSILON * mean_x.abs().max(1.0) * w_sum {
        return (mean_y, 0.0);
    }

    let beta = wcov / wvar;
    if !beta.is_finite() {
        return (mean_y, 0.0);
    }
    if beta.abs() > 1e18 {
        return (mean_y, 0.0);
    }

    let alpha = mean_y - beta * mean_x;
    (alpha, beta)
}

fn loglinear_slr<T: TrainingKey>(data: &RMITrainingData<T>) -> (f64, f64) {
    // log all of the outputs, omit any item that doesn't have a valid log
    let transformed_data: Vec<(f64, f64)> = data
        .iter()
        .map(|(x, y)| (x.as_float(), (y as f64).ln()))
        .filter(|(_, y)| y.is_finite())
        .collect();

    // TODO this currently creates a copy of the data and then calls
    // slr... we can probably do better by moving the log into the slr.
    return slr(transformed_data.into_iter());
}

pub struct LinearModel {
    params: (f64, f64),
}

impl LinearModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> LinearModel {
        let params = slr(data
            .iter()
            .map(|(inp, offset)| (inp.as_float(), offset as f64)));
        return LinearModel { params };
    }

    /// Fit a `LinearModel` using per-pair weights.
    ///
    /// `weights` must have the same length as `data`. Each entry is a
    /// non-negative weight; higher values pull the fit toward those pairs.
    /// When all weights are equal the result is identical to [`Self::new`].
    ///
    /// Used by the BED-prior trainer path: pairs whose SA position falls in a
    /// BED interval receive a higher weight so the model minimizes weighted
    /// squared error, producing tighter predictions in the target region.
    pub fn new_weighted(pairs: &[(f64, f64)], weights: &[f64]) -> LinearModel {
        let params = weighted_slr(pairs, weights);
        LinearModel { params }
    }
}

impl Model for LinearModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let (intercept, slope) = self.params;
        return slope.mul_add(inp.as_float(), intercept);
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

pub struct LogLinearModel {
    params: (f64, f64),
}

fn exp1(inp: f64) -> f64 {
    let mut x = inp;
    x = 1.0 + x / 64.0;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}

impl LogLinearModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> LogLinearModel {
        return LogLinearModel {
            params: loglinear_slr(&data),
        };
    }
}

impl Model for LogLinearModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let (alpha, beta) = self.params;
        return exp1(beta.mul_add(inp.as_float(), alpha));
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
inline double loglinear(double alpha, double beta, double inp) {
    return exp1(std::fma(beta, inp, alpha));
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("loglinear");
    }
    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::EXP1);
        return to_r;
    }
}

pub struct RobustLinearModel {
    params: (f64, f64),
}

impl RobustLinearModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> RobustLinearModel {
        let total_items = data.len();
        if data.len() == 0 {
            return RobustLinearModel { params: (0.0, 0.0) };
        }

        let bnd = usize::max(1, ((total_items as f64) * 0.0001) as usize);
        assert!(bnd * 2 + 1 < data.len());

        let iter = data.iter().skip(bnd).take(data.len() - 2 * bnd);

        let robust_params = slr(iter.map(|(inp, offset)| (inp.as_float(), offset as f64)));

        return RobustLinearModel {
            params: robust_params,
        };
    }
}

impl Model for RobustLinearModel {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_model(pairs: Vec<(u64, usize)>) -> LinearModel {
        let data = RMITrainingData::<u64>::new(Box::new(pairs));
        LinearModel::new(&data)
    }

    /// When all keys are within the same f64 ULP (i.e. their f64 representations
    /// all round to the same value), the empirical variance is effectively zero
    /// in f64 arithmetic. Without the pre-division guard the code would divide
    /// by a subnormal var and produce a wildly inflated beta.
    ///
    /// Construction: 10 keys near u64::MAX, spaced 1 apart. At that magnitude
    /// (~1.8e19), the f64 ULP is 2048, so all 10 keys round to the same f64
    /// value. The computed `m2` is therefore 0.0 (or subnormal), and after
    /// dividing by (n-1) the var is 0.0 or subnormal — well below the
    /// `f64::EPSILON * mean_x.abs()` threshold. The guard must return
    /// `(mean_y, 0.0)` rather than an inflated beta.
    #[test]
    fn slr_subnormal_var_returns_constant_model() {
        // Keys near u64::MAX; all 10 should have the same f64 representation.
        let base: u64 = u64::MAX - 9;
        let pairs: Vec<(u64, usize)> = (0..10u64).map(|i| (base + i, (i * 100) as usize)).collect();

        // Sanity: all keys should map to the same f64 at this magnitude.
        let f0 = base as f64;
        for i in 0..10u64 {
            let fi = (base + i) as f64;
            // At u64::MAX the f64 ULP is 2048; 9 apart < 2048 → same f64.
            assert_eq!(f0, fi, "keys should map to the same f64 at this magnitude");
        }

        let model = make_linear_model(pairs);
        let params = model.params();
        let beta = match params.get(1) {
            Some(ModelParam::Float(v)) => *v,
            _ => panic!("expected Float beta"),
        };
        assert_eq!(
            beta, 0.0,
            "subnormal-var keys should produce beta=0 (constant model), got beta={beta}"
        );
    }

    /// Sanity: a simple 2-point dataset with meaningful variance should still
    /// produce a non-zero beta (the guard must not over-fire).
    #[test]
    fn slr_normal_variance_produces_nonzero_beta() {
        // key 0 → SA 0, key (1<<48) → SA 1000. Large gap → var is large.
        let pairs: Vec<(u64, usize)> = vec![(0u64, 0), (1u64 << 48, 1000)];
        let model = make_linear_model(pairs);
        let params = model.params();
        let beta = match params.get(1) {
            Some(ModelParam::Float(v)) => *v,
            _ => panic!("expected Float beta"),
        };
        assert!(
            beta != 0.0 && beta.is_finite(),
            "normal variance should yield non-zero finite beta, got {beta}"
        );
    }
}
