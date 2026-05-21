// Copyright Ryan Marcus 2020          (origin: learnedsystems/RMI)
// Copyright 2022 Youngmok Jung et al. (origin: kaist-ina/BWA-MEME RMI fork)
// Modified by Fulcrum Genomics 2026
// SPDX-License-Identifier: MIT

use super::super::models::TrainingKey;
use super::super::models::*;
use super::lower_bound_correction::LowerBoundCorrection;
use super::{train_model, validate, TrainedRMI};
use log::*;

fn error_between(v1: u64, v2: u64, max_pred: u64) -> u64 {
    let pred1 = u64::min(v1, max_pred);
    let pred2 = u64::min(v2, max_pred);
    return u64::max(pred1, pred2) - u64::min(pred1, pred2);
}

fn build_models_from<T: TrainingKey>(
    data: &RMITrainingData<T>,
    top_model: &Box<dyn Model>,
    model_type: &str,
    start_idx: usize,
    end_idx: usize,
    first_model_idx: usize,
    num_models: usize,
) -> Vec<Box<dyn Model>> {
    assert!(
        end_idx > start_idx,
        "start index was {} but end index was {}",
        start_idx,
        end_idx
    );
    assert!(end_idx <= data.len());
    assert!(start_idx <= data.len());

    let dummy_md = RMITrainingData::<T>::empty();
    let mut leaf_models: Vec<Box<dyn Model>> = Vec::with_capacity(num_models as usize);
    let mut second_layer_data = Vec::with_capacity((end_idx - start_idx) / num_models as usize);
    let mut last_target = first_model_idx;

    let bounded_it = data.iter().skip(start_idx).take(end_idx - start_idx);

    for (x, y) in bounded_it {
        let model_pred = top_model.predict_to_int(&x.to_model_input()) as usize;
        assert!(
            top_model.needs_bounds_check() || model_pred < first_model_idx + num_models,
            "Top model gave an index of {} which is out of bounds of {}. \
                Subset range: {} to {}",
            model_pred,
            start_idx + num_models,
            start_idx,
            end_idx
        );
        let target = usize::min(first_model_idx + num_models - 1, model_pred);
        assert!(target >= last_target);

        if target > last_target {
            // this is the first datapoint for the next leaf model.
            // train the previous leaf model.

            // include the first point of the next leaf node to
            // support lower bound searches (not required, but reduces error)
            let last_item = second_layer_data.last().copied();
            second_layer_data.push((x, y));

            let container = RMITrainingData::new(Box::new(second_layer_data));
            let leaf_model = train_model(model_type, &container);
            leaf_models.push(leaf_model);

            // leave empty models for any we skipped.
            for _skipped_idx in (last_target + 1)..target {
                leaf_models.push(train_model(model_type, &dummy_md));
            }
            assert_eq!(leaf_models.len() + first_model_idx, target);

            second_layer_data = Vec::new();

            // include the last item of this leaf in the next leaf
            // to support lower bound searches.
            if let Some(v) = last_item {
                second_layer_data.push(v);
            }
        }

        second_layer_data.push((x, y));
        last_target = target;
    }

    // train the last remaining model
    assert!(!second_layer_data.is_empty());
    let container = RMITrainingData::new(Box::new(second_layer_data));
    let leaf_model = train_model(model_type, &container);
    leaf_models.push(leaf_model);
    assert!(leaf_models.len() <= num_models);

    // add models at the end with nothing mapped into them
    for _skipped_idx in (last_target + 1)..(first_model_idx + num_models) as usize {
        leaf_models.push(train_model(model_type, &dummy_md));
    }
    assert_eq!(num_models as usize, leaf_models.len());
    return leaf_models;
}

// called in build_partial_3layer_models_from (Task 15.5e)
#[allow(dead_code)]
fn build_partial_models_from<T: TrainingKey>(
    data: &RMITrainingData<T>,
    top_model: &Box<dyn Model>,
    model_type: &str,
    start_idx: usize,
    end_idx: usize,
    first_model_idx: usize,
    num_models: usize,
    top_model_offset: u64,
) -> Vec<Box<dyn Model>> {
    // reserved for future use; matches BWA-MEME signature for call compatibility
    let _ = top_model_offset;

    assert!(
        end_idx > start_idx,
        "start index was {} but end index was {}",
        start_idx,
        end_idx
    );
    assert!(end_idx <= data.len());
    assert!(start_idx <= data.len());

    let dummy_md = RMITrainingData::<T>::empty();
    let mut leaf_models: Vec<Box<dyn Model>> = Vec::with_capacity(num_models as usize);
    let mut second_layer_data = Vec::with_capacity((end_idx - start_idx) / num_models as usize);
    let mut last_target = first_model_idx;

    let bounded_it = data.iter().skip(start_idx).take(end_idx - start_idx);

    for (x, y) in bounded_it {
        let model_pred = top_model.predict_to_int(&x.to_model_input()) as usize;
        assert!(
            top_model.needs_bounds_check() || model_pred < first_model_idx + num_models,
            "Top model gave an index of {} which is out of bounds of {}. \
                Subset range: {} to {}",
            model_pred,
            start_idx + num_models,
            start_idx,
            end_idx
        );
        let target = usize::min(first_model_idx + num_models - 1, model_pred);
        assert!(target >= last_target);

        if target > last_target {
            // this is the first datapoint for the next leaf model.
            // train the previous leaf model.

            // include the first point of the next leaf node to
            // support lower bound searches (not required, but reduces error)
            let last_item = second_layer_data.last().copied();
            second_layer_data.push((x, y));

            let container = RMITrainingData::new(Box::new(second_layer_data));
            let leaf_model = train_model(model_type, &container);
            leaf_models.push(leaf_model);

            // leave empty models for any we skipped.
            for _skipped_idx in (last_target + 1)..target {
                leaf_models.push(train_model(model_type, &dummy_md));
            }
            assert_eq!(leaf_models.len() + first_model_idx, target);

            second_layer_data = Vec::new();

            // include the last item of this leaf in the next leaf
            // to support lower bound searches.
            if let Some(v) = last_item {
                second_layer_data.push(v);
            }
        }

        second_layer_data.push((x, y));
        last_target = target;
    }

    // train the last remaining model
    assert!(!second_layer_data.is_empty());
    let container = RMITrainingData::new(Box::new(second_layer_data));
    let leaf_model = train_model(model_type, &container);
    leaf_models.push(leaf_model);
    assert!(leaf_models.len() <= num_models);

    // add models at the end with nothing mapped into them
    for _skipped_idx in (last_target + 1)..(first_model_idx + num_models) as usize {
        leaf_models.push(train_model(model_type, &dummy_md));
    }
    assert_eq!(num_models as usize, leaf_models.len());
    return leaf_models;
}

// called in train_partial_three_layer (Task 15.5f)
#[allow(dead_code)]
fn build_partial_3layer_models_from<T: TrainingKey>(
    data: &RMITrainingData<T>,
    top_model: &Box<dyn Model>,
    model_type: &str,
    model_type_partial: &str,
    start_idx: usize,
    end_idx: usize,
    first_model_idx: usize,
    num_models: usize,
) -> (
    Vec<Box<dyn Model>>,
    Vec<(usize, usize)>,
    Vec<LowerBoundCorrection<T>>,
    Vec<Box<dyn Model>>,
    usize,
) {
    assert!(
        end_idx > start_idx,
        "start index was {} but end index was {}",
        start_idx,
        end_idx
    );
    assert!(end_idx <= data.len());
    assert!(start_idx <= data.len());

    let dummy_md = RMITrainingData::<T>::empty();
    let mut leaf_models: Vec<Box<dyn Model>> = Vec::with_capacity(num_models as usize);
    let mut partial_3rd_models: Vec<Box<dyn Model>> = Vec::new();
    let mut partial_3rd_idx: Vec<(usize, usize)> = Vec::new();
    let mut partial_3rd_lb_corrs: Vec<LowerBoundCorrection<T>> = Vec::new();
    let mut second_layer_data: Vec<(T, usize)> =
        Vec::with_capacity((end_idx - start_idx) / num_models as usize);
    let mut last_target = first_model_idx;

    let bounded_it = data.iter().skip(start_idx).take(end_idx - start_idx);

    let mut third_layer_num: usize = 0;

    let make_partial_threshold = 1000;
    let average_partial_model_num = 20;

    for (x, y) in bounded_it {
        let model_pred = top_model.predict_to_int(&x.to_model_input()) as usize;
        assert!(
            top_model.needs_bounds_check() || model_pred < first_model_idx + num_models,
            "Top model gave an index of {} which is out of bounds of {}. \
                Subset range: {} to {}",
            model_pred,
            start_idx + num_models,
            start_idx,
            end_idx
        );
        let target = usize::min(first_model_idx + num_models - 1, model_pred);
        assert!(target >= last_target);
        if target > last_target {
            // this is the first datapoint for the next leaf model.
            // train the previous leaf model.
            // include the first point of the next leaf node to
            // support lower bound searches (not required, but reduces error)
            let last_item = second_layer_data.last().copied();
            second_layer_data.push((x, y));
            let mut container = RMITrainingData::new(Box::new(second_layer_data));
            if container.len() > make_partial_threshold {
                let curr_third_layer_num =
                    (container.len() as f64 / average_partial_model_num as f64).round() as u64;
                let start_y = container.get(0).1;
                let end_y = container.get(container.len() - 1).1;

                container.set_offset(container.get(0).1);
                container.set_scale((curr_third_layer_num - 1) as f64 / (end_y - start_y) as f64);

                let leaf_model = train_model(model_type, &container);

                // check if only single partial model is used, and exclude

                partial_3rd_idx.append(&mut vec![(third_layer_num, curr_third_layer_num as usize)]);
                container.set_offset(0);
                container.set_scale(1.0);

                let mut curr_partial_3rd_model = build_partial_models_from(
                    &container,
                    &leaf_model,
                    model_type_partial,
                    0,
                    container.len(),
                    0,
                    curr_third_layer_num as usize,
                    third_layer_num as u64,
                );

                let lb_corrections = LowerBoundCorrection::new(
                    |x| leaf_model.predict_to_int(&x.to_model_input()),
                    curr_third_layer_num,
                    &container,
                );
                for idx in 0..(curr_third_layer_num as usize) {
                    assert_eq!(
                        lb_corrections.first_key(idx).is_none(),
                        lb_corrections.last_key(idx).is_none()
                    );
                    if lb_corrections.last_key(idx).is_none() {
                        // model is empty!
                        let mut upper_bound = lb_corrections.next_index(idx);
                        // next index is 0 if first and single model is used in 3rd layer
                        // no need to use partial layer in this case, need fix
                        if lb_corrections.first_non_empty_model() == 0
                            && lb_corrections.first_non_empty_model()
                                == lb_corrections.last_non_empty_model()
                        {
                            upper_bound = end_y + 1;
                        }
                        if !curr_partial_3rd_model[idx].set_to_constant_model(upper_bound as u64) {
                            panic!();
                        }
                    }
                }
                leaf_models.push(leaf_model);
                partial_3rd_lb_corrs.append(&mut vec![lb_corrections]);
                partial_3rd_models.append(&mut curr_partial_3rd_model);
                third_layer_num += curr_third_layer_num as usize;
                assert!(partial_3rd_models.len() == third_layer_num);
            } else {
                let leaf_model = train_model(model_type, &container);
                partial_3rd_idx.append(&mut vec![(0, 0)]);
                leaf_models.push(leaf_model);
            }
            // leave empty models for any we skipped.
            for _skipped_idx in (last_target + 1)..target {
                leaf_models.push(train_model(model_type, &dummy_md));
                partial_3rd_idx.append(&mut vec![(0, 0)]);
            }
            assert_eq!(leaf_models.len() + first_model_idx, target);

            second_layer_data = Vec::new();

            // include the last item of this leaf in the next leaf
            // to support lower bound searches.
            if let Some(v) = last_item {
                second_layer_data.push(v);
            }
        }
        second_layer_data.push((x, y));
        last_target = target;
    }

    // train the last remaining model
    assert!(!second_layer_data.is_empty());
    let mut container = RMITrainingData::new(Box::new(second_layer_data));
    if container.len() > make_partial_threshold {
        let curr_third_layer_num =
            (container.len() as f64 / average_partial_model_num as f64).round() as usize;
        let start_y = container.get(0).1;
        let end_y = container.get(container.len() - 1).1;

        container.set_offset(container.get(0).1);
        container.set_scale((curr_third_layer_num - 1) as f64 / (end_y - start_y) as f64);

        let leaf_model = train_model(model_type, &container);
        partial_3rd_idx.append(&mut vec![(third_layer_num, curr_third_layer_num)]);

        container.set_offset(0);
        container.set_scale(1.0);
        // build partial 3 layer model with calculated number of models to build
        let mut curr_partial_3rd_model = build_partial_models_from(
            &container,
            &leaf_model,
            model_type_partial,
            0,
            container.len(),
            0,
            curr_third_layer_num,
            third_layer_num as u64,
        );
        // Do lowerboundcorrection
        let lb_corrections = LowerBoundCorrection::new(
            |x| leaf_model.predict_to_int(&x.to_model_input()),
            curr_third_layer_num as u64,
            &container,
        );
        // set empty models to constant model
        for idx in 0..curr_third_layer_num {
            assert_eq!(
                lb_corrections.first_key(idx).is_none(),
                lb_corrections.last_key(idx).is_none()
            );
            if lb_corrections.last_key(idx).is_none() {
                // model is empty!
                let mut upper_bound = lb_corrections.next_index(idx);
                // if data is all inside single and first model of partial models, upper bound should be end_y + 1
                // lowerbound have next set to 0 in this case
                if lb_corrections.first_non_empty_model() == 0
                    && lb_corrections.first_non_empty_model()
                        == lb_corrections.last_non_empty_model()
                {
                    upper_bound = end_y + 1;
                }
                if !curr_partial_3rd_model[idx].set_to_constant_model(upper_bound as u64) {
                    panic!();
                }
            }
        }
        leaf_models.push(leaf_model);
        partial_3rd_lb_corrs.append(&mut vec![lb_corrections]);
        partial_3rd_models.append(&mut curr_partial_3rd_model);

        third_layer_num += curr_third_layer_num;
        assert!(partial_3rd_models.len() == third_layer_num);
    } else {
        let leaf_model = train_model(model_type, &container);
        partial_3rd_idx.append(&mut vec![(0, 0)]);
        leaf_models.push(leaf_model);
    }
    assert!(leaf_models.len() <= num_models);

    // add models at the end with nothing mapped into them
    for _skipped_idx in (last_target + 1)..(first_model_idx + num_models) as usize {
        leaf_models.push(train_model(model_type, &dummy_md));
        partial_3rd_idx.append(&mut vec![(0, 0)]);
    }
    assert_eq!(num_models as usize, leaf_models.len());
    return (
        leaf_models,
        partial_3rd_idx,
        partial_3rd_lb_corrs,
        partial_3rd_models,
        third_layer_num,
    );
}

pub fn train_two_layer<T: TrainingKey>(
    md_container: &mut RMITrainingData<T>,
    layer1_model: &str,
    layer2_model: &str,
    num_leaf_models: u64,
) -> TrainedRMI {
    validate(&[String::from(layer1_model), String::from(layer2_model)]);

    let num_rows = md_container.len();

    trace!("Training top-level {} model layer", layer1_model);
    md_container.set_scale(num_leaf_models as f64 / num_rows as f64);
    let top_model = train_model(layer1_model, &md_container);

    // Check monotonicity if in debug mode
    #[cfg(debug_assertions)]
    {
        let mut last_pred = 0;
        for (x, _y) in md_container.iter_model_input() {
            let prediction = top_model.predict_to_int(&x);
            debug_assert!(
                prediction >= last_pred,
                "Top model {} was non-monotonic on input {:?}",
                layer1_model,
                x
            );
            last_pred = prediction;
        }
        trace!("Top model was monotonic.");
    }

    trace!(
        "Training second-level {} model layer (num models = {})",
        layer2_model,
        num_leaf_models
    );
    md_container.set_scale(1.0);

    // find a prediction boundary near the middle
    let midpoint_model = num_leaf_models / 2;
    let split_idx = md_container.lower_bound_by(|x| {
        let model_idx = top_model.predict_to_int(&x.0.to_model_input());
        let model_target = u64::min(num_leaf_models - 1, model_idx);
        return model_target.cmp(&midpoint_model);
    });

    // make sure the split point that we got is valid
    if split_idx > 0 && split_idx < md_container.len() {
        let key_at = top_model.predict_to_int(&md_container.get_key(split_idx).to_model_input());
        let key_pr =
            top_model.predict_to_int(&md_container.get_key(split_idx - 1).to_model_input());
        assert!(key_at > key_pr);
    }

    let mut leaf_models = if split_idx >= md_container.len() {
        build_models_from(
            &md_container,
            &top_model,
            layer2_model,
            0,
            md_container.len(),
            0,
            num_leaf_models as usize,
        )
    } else {
        let split_idx_target = u64::min(
            num_leaf_models - 1,
            top_model.predict_to_int(&md_container.get_key(split_idx).to_model_input()),
        ) as usize;

        let first_half_models = split_idx_target as usize;
        let second_half_models = num_leaf_models as usize - split_idx_target as usize;

        let (mut hf1, mut hf2) = rayon::join(
            || {
                build_models_from(
                    &md_container,
                    &top_model,
                    layer2_model,
                    0,
                    split_idx,
                    0,
                    first_half_models,
                )
            },
            || {
                build_models_from(
                    &md_container,
                    &top_model,
                    layer2_model,
                    split_idx + 1,
                    md_container.len(),
                    split_idx_target,
                    second_half_models,
                )
            },
        );

        let mut leaf_models = Vec::new();
        leaf_models.append(&mut hf1);
        leaf_models.append(&mut hf2);
        leaf_models
    };

    trace!("Computing lower bound stats...");
    let lb_corrections = LowerBoundCorrection::new(
        |x| top_model.predict_to_int(&x.to_model_input()),
        num_leaf_models,
        md_container,
    );

    trace!("Fixing empty models...");
    // replace any empty model with a model that returns the correct constant
    // (for LB predictions), if the underlying model supports it.
    let mut could_not_replace = false;
    for idx in 0..(num_leaf_models as usize) - 1 {
        assert_eq!(
            lb_corrections.first_key(idx).is_none(),
            lb_corrections.last_key(idx).is_none()
        );

        if lb_corrections.last_key(idx).is_none() {
            // model is empty!
            let upper_bound = lb_corrections.next_index(idx);
            if !leaf_models[idx].set_to_constant_model(upper_bound as u64) {
                could_not_replace = true;
            }
        }
    }

    if could_not_replace {
        warn!(
            "Some empty models could not be replaced with constants, \
               negative lookup performance may be poor."
        );
    }

    trace!("Computing last level errors...");
    // evaluate model, compute last level errors
    let mut last_layer_max_l1s = vec![(0, 0); num_leaf_models as usize];
    for (x, y) in md_container.iter_model_input() {
        let leaf_idx = top_model.predict_to_int(&x);
        let target = u64::min(num_leaf_models - 1, leaf_idx) as usize;

        let pred = leaf_models[target].predict_to_int(&x);
        let err = error_between(pred, y as u64, md_container.len() as u64);

        let cur_val = last_layer_max_l1s[target];
        last_layer_max_l1s[target] = (cur_val.0 + 1, u64::max(err, cur_val.1));
    }

    // for lower bound searches, we need to make sure that:
    //   (1) a query for the first key in the next leaf minus one
    //       includes the key in the next leaf. (upper error)
    //   (2) a query for the last key in the previous leaf plus one
    //       includes the first key after the previous leaf (lower error)
    //       (normally, the first key after the previous leaf is the first
    //        key in this leaf, but not in the case where this leaf has no keys)
    let mut large_corrections = 0;
    for leaf_idx in 0..num_leaf_models as usize {
        let curr_err = last_layer_max_l1s[leaf_idx].1;
        let upper_error = {
            let (idx_of_next, key_of_next) = lb_corrections.next(leaf_idx);
            let pred =
                leaf_models[leaf_idx].predict_to_int(&key_of_next.minus_epsilon().to_model_input());
            error_between(pred, idx_of_next as u64 + 1, md_container.len() as u64)
        };

        let lower_error = {
            let first_key_before = lb_corrections.prev_key(leaf_idx);

            let prev_idx = if leaf_idx == 0 { 0 } else { leaf_idx - 1 };
            let first_idx = lb_corrections.next_index(prev_idx);

            let pred = leaf_models[leaf_idx]
                .predict_to_int(&first_key_before.plus_epsilon().to_model_input());
            error_between(pred, first_idx as u64, md_container.len() as u64)
        };

        let new_err = *(&[curr_err, upper_error, lower_error])
            .iter()
            .max()
            .unwrap()
            + lb_corrections.longest_run(leaf_idx);

        let num_items_in_leaf = last_layer_max_l1s[leaf_idx].0;
        last_layer_max_l1s[leaf_idx] = (num_items_in_leaf, new_err);

        if new_err - curr_err > 512 && num_items_in_leaf > 100 {
            large_corrections += 1;
        }
    }

    if large_corrections > 1 {
        trace!(
            "Of {} models, {} needed large lower bound corrections.",
            num_leaf_models,
            large_corrections
        );
    }

    trace!("Evaluating two-layer RMI...");
    let (m_idx, m_err) = last_layer_max_l1s
        .iter()
        .enumerate()
        .max_by_key(|(_idx, &x)| x.1)
        .unwrap();

    let model_max_error = m_err.1;
    let model_max_error_idx = m_idx;

    let model_avg_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| n * err)
        .sum::<u64>() as f64
        / num_rows as f64;

    let model_avg_l2_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| ((n * err) as f64).powf(2.0) / num_rows as f64)
        .sum::<f64>();

    let model_avg_log2_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| (*n as f64) * ((2 * err + 2) as f64).log2())
        .sum::<f64>()
        / num_rows as f64;

    let model_max_log2_error: f64 = (model_max_error as f64).log2();

    let final_errors = last_layer_max_l1s
        .into_iter()
        .map(|(_n, err)| err)
        .collect();

    return TrainedRMI {
        num_rmi_rows: md_container.len(),
        num_data_rows: md_container.len(),
        model_avg_error,
        model_avg_l2_error,
        model_avg_log2_error,
        model_max_error,
        model_max_error_idx,
        model_max_log2_error,
        last_layer_max_l1s: final_errors,
        third_layer_max_l1s: Vec::new(),
        rmi: vec![vec![top_model], leaf_models],
        models: format!("{},{}", layer1_model, layer2_model),
        branching_factor: num_leaf_models,
        cache_fix: None,
        build_time: 0,
    };
}

pub fn train_partial_three_layer<T: TrainingKey>(
    md_container: &mut RMITrainingData<T>,
    layer1_model: &str,
    layer2_model: &str,
    layer3_model: &str,
    num_leaf_models: u64,
) -> TrainedRMI {
    validate(&[
        String::from(layer1_model),
        String::from(layer2_model),
        String::from(layer3_model),
    ]);

    let num_rows = md_container.len();

    let second_model_num = num_leaf_models;

    log::debug!("Training top-level {} model layer", layer1_model);
    md_container.set_scale(second_model_num as f64 / num_rows as f64);
    let top_model = train_model(layer1_model, &md_container);

    // Check monotonicity if in debug mode
    #[cfg(debug_assertions)]
    {
        let mut last_pred = 0;
        for (x, _y) in md_container.iter_model_input() {
            let prediction = top_model.predict_to_int(&x);
            debug_assert!(
                prediction >= last_pred,
                "Top model {} was non-monotonic on input {:?}",
                layer1_model,
                x
            );
            last_pred = prediction;
        }
        trace!("Top model was monotonic.");
    }

    log::debug!(
        "Training second-level {} model layer (num models = {})",
        layer2_model,
        second_model_num
    );
    md_container.set_scale(1.0);
    let first_idx_target = 0;
    let (
        mut sec_models,
        partial_3rd_idx,
        partial_3rd_lb_corrs,
        partial_3rd_models,
        third_layer_num,
    ) = build_partial_3layer_models_from(
        &md_container,
        &top_model,
        layer2_model,
        layer3_model,
        0,
        md_container.len(),
        first_idx_target,
        second_model_num as usize,
    );

    log::debug!("[2nd layer]Computing lower bound stats...");
    let lb_corrections = LowerBoundCorrection::new(
        |x| top_model.predict_to_int(&x.to_model_input()),
        second_model_num,
        md_container,
    );

    log::debug!("[2nd layer]Fixing empty models...");
    // replace any empty model with a model that returns the correct constant
    // (for LB predictions), if the underlying model supports it.
    let mut could_not_replace = false;
    for idx in 0..(second_model_num as usize) {
        assert_eq!(
            lb_corrections.first_key(idx).is_none(),
            lb_corrections.last_key(idx).is_none()
        );

        if lb_corrections.last_key(idx).is_none() {
            // model is empty!
            let upper_bound = lb_corrections.next_index(idx);
            if !sec_models[idx].set_to_constant_model(upper_bound as u64) {
                could_not_replace = true;
            }
        }
    }
    if could_not_replace {
        warn!(
            "[2nd layer]Some empty models could not be replaced with constants, \
            negative lookup performance may be poor."
        );
        panic!();
    }
    log::debug!("Computing last level errors...");
    // evaluate model, compute last level errors

    let mut last_layer_max_l1s = vec![(0, 0); num_leaf_models as usize];
    let mut third_layer_max_l1s = vec![(0, 0); third_layer_num as usize];

    for (x, y) in md_container.iter_model_input() {
        let leaf_idx = top_model.predict_to_int(&x);
        let target = u64::min(num_leaf_models - 1, leaf_idx) as usize;

        let pred;
        if partial_3rd_idx[target] == (0, 0) {
            pred = sec_models[target].predict_to_int(&x);
            let err = error_between(pred, y as u64, md_container.len() as u64);
            let cur_val = last_layer_max_l1s[target];

            // cur_val stores minimum error (can be minus value) and maximum error (can be minus also)
            // first 32bit is for minimum err, next 32bit for maximum error
            // first bit is used for sign, 1 for minus, 0 for plus

            let mut min_err = (cur_val.1 >> 32) & 0x000000003fffffffu64; // left most bit is used for partial model representation
            let mut min_flag = ((cur_val.1 >> 32) & 0x0000000040000000u64) >> 30;
            let mut max_err = (cur_val.1) & 0x000000007fffffffu64;
            let mut max_flag = ((cur_val.1) & 0x0000000080000000u64) >> 31;

            if pred > y as u64 {
                // err is minus err
                if min_err < err || min_flag == 0 {
                    min_err = err;
                    min_flag = 1;
                }
                if max_err > err && max_flag == 1 {
                    max_err = err;
                    max_flag = 1;
                }
            } else {
                // pred is smaller than y, plus err
                if min_err > err && min_flag == 0 {
                    min_err = err;
                    min_flag = 0;
                }
                if max_err < err || max_flag == 1 {
                    max_err = err;
                    max_flag = 0;
                }
            }

            last_layer_max_l1s[target] = (
                cur_val.0 + 1,
                min_flag << 62 | min_err << 32 | max_flag << 31 | max_err,
            );
            assert!(last_layer_max_l1s[target].1 >> 63 == 0); // flag for partial model should be 0
        } else {
            // partial_3rd_idx.0 has start of 3rd model list, partial_3rd_idx.1 has number of models
            let mut target_third =
                sec_models[target].predict_to_int(&x) + partial_3rd_idx[target].0 as u64;
            target_third = u64::min(
                (partial_3rd_idx[target].0 + partial_3rd_idx[target].1 - 1) as u64,
                target_third,
            );
            target_third = u64::max(partial_3rd_idx[target].0 as u64, target_third);
            pred = partial_3rd_models[target_third as usize].predict_to_int(&x);

            let cur_val = last_layer_max_l1s[target];
            // put number of cumulative partial models in 32 most significant bits, number of models in 32 least significant bits
            // first bit is set as 1 if partial model is used
            assert!(partial_3rd_idx[target].0 as u64 <= 0x000000007fffffffu64);
            assert!(partial_3rd_idx[target].1 as u64 <= 0x00000000ffffffffu64);
            last_layer_max_l1s[target] = (
                cur_val.0 + 1,
                ((partial_3rd_idx[target].0 as u64 | 0x0000000080000000u64) << 32)
                    | partial_3rd_idx[target].1 as u64,
            );

            let err = error_between(pred, y as u64, md_container.len() as u64);
            let cur_val = third_layer_max_l1s[target_third as usize];

            let mut min_err = (cur_val.1 >> 32) & 0x000000003fffffffu64; // left most bit is used for partial model representation
            let mut min_flag = ((cur_val.1 >> 32) & 0x0000000040000000u64) >> 30;
            let mut max_err = (cur_val.1) & 0x000000007fffffffu64;
            let mut max_flag = ((cur_val.1) & 0x0000000080000000u64) >> 31;

            if pred > y as u64 {
                // err is minus err
                if min_err < err || min_flag == 0 {
                    min_err = err;
                    min_flag = 1;
                }
                if max_err > err && max_flag == 1 {
                    max_err = err;
                    max_flag = 1;
                }
            } else {
                // pred is smaller than y, plus err
                if min_err > err && min_flag == 0 {
                    min_err = err;
                    min_flag = 0;
                }
                if max_err < err || max_flag == 1 {
                    max_err = err;
                    max_flag = 0;
                }
            }
            third_layer_max_l1s[target_third as usize] = (
                cur_val.0 + 1,
                min_flag << 62 | min_err << 32 | max_flag << 31 | max_err,
            );
        }
    }

    let large_corrections = 0;
    let mut partial_third_num = 0;
    let report_error_threshold = 100000;
    for leaf_idx in 0..num_leaf_models as usize {
        if partial_3rd_idx[leaf_idx] != (0, 0) {
            for third_idx in 0..partial_3rd_idx[leaf_idx].1 {
                let curr_err = third_layer_max_l1s[third_idx + partial_3rd_idx[leaf_idx].0].1;

                let mut min_err = (curr_err >> 32) & 0x000000003fffffffu64; // left most bit is used for partial model representation
                let mut min_flag = ((curr_err >> 32) & 0x0000000040000000u64) >> 30;
                let mut max_err = curr_err & 0x000000007fffffffu64;
                let mut max_flag = (curr_err & 0x0000000080000000u64) >> 31;

                let mut upper_flag = false;
                let mut lower_flag = false;
                let upper_error = {
                    if third_idx
                        >= partial_3rd_lb_corrs[partial_third_num].last_non_empty_model() as usize
                    {
                        let (idx_of_next, key_of_next) = lb_corrections.next(leaf_idx);
                        let pred = partial_3rd_models[third_idx + partial_3rd_idx[leaf_idx].0]
                            .predict_to_int(&key_of_next.minus_epsilon().to_model_input());

                        upper_flag = pred > idx_of_next as u64;
                        error_between(pred, idx_of_next as u64 + 1, md_container.len() as u64)
                    } else {
                        let (idx_of_next, key_of_next) =
                            partial_3rd_lb_corrs[partial_third_num].next(third_idx);
                        let pred = partial_3rd_models[third_idx + partial_3rd_idx[leaf_idx].0]
                            .predict_to_int(&key_of_next.minus_epsilon().to_model_input());

                        upper_flag = pred > idx_of_next as u64;
                        error_between(pred, idx_of_next as u64 + 1, md_container.len() as u64)
                    }
                };
                let lower_error = {
                    let first_key_before;
                    let prev_idx;
                    let first_idx;
                    // if empty models
                    if third_idx
                        <= partial_3rd_lb_corrs[partial_third_num].first_non_empty_model() as usize
                    {
                        first_key_before = lb_corrections.prev_key(leaf_idx);
                        prev_idx = if leaf_idx == 0 { 0 } else { leaf_idx - 1 };
                        first_idx = lb_corrections.next_index(prev_idx);
                        let (prev_idx_inner, first_idx_inner) = if leaf_idx == 0 {
                            let p = if third_idx == 0 { 0 } else { third_idx - 1 };
                            (p, partial_3rd_lb_corrs[partial_third_num].next_index(p))
                        } else {
                            (prev_idx, first_idx)
                        };
                        let pred = partial_3rd_models[third_idx + partial_3rd_idx[leaf_idx].0]
                            .predict_to_int(&first_key_before.plus_epsilon().to_model_input());

                        lower_flag = pred > first_idx_inner as u64;
                        error_between(pred, first_idx_inner as u64, md_container.len() as u64)
                    } else {
                        first_key_before =
                            partial_3rd_lb_corrs[partial_third_num].prev_key(third_idx);
                        prev_idx = if third_idx == 0 { 0 } else { third_idx - 1 };
                        first_idx = partial_3rd_lb_corrs[partial_third_num].next_index(prev_idx);
                        let (eff_prev_idx, eff_first_idx) =
                            if partial_3rd_lb_corrs[partial_third_num].first_non_empty_model() == 0
                                && partial_3rd_lb_corrs[partial_third_num].first_non_empty_model()
                                    == partial_3rd_lb_corrs[partial_third_num]
                                        .last_non_empty_model()
                            {
                                let p = if leaf_idx == 0 { 0 } else { leaf_idx - 1 };
                                (p, lb_corrections.next_index(p))
                            } else {
                                (prev_idx, first_idx)
                            };
                        let pred = partial_3rd_models[third_idx + partial_3rd_idx[leaf_idx].0]
                            .predict_to_int(&first_key_before.plus_epsilon().to_model_input());

                        lower_flag = pred > eff_first_idx as u64;
                        error_between(pred, eff_first_idx as u64, md_container.len() as u64)
                    }
                };

                // (debug trace removed for upper_error > report_error_threshold)
                // (debug trace removed for lower_error > report_error_threshold)

                if upper_flag {
                    // err is minus err
                    if min_err < upper_error || min_flag == 0 {
                        min_err = upper_error;
                        min_flag = 1;
                    }
                    if max_err > upper_error && max_flag == 1 {
                        max_err = upper_error;
                        max_flag = 1;
                    }
                } else {
                    // pred is smaller than y, plus err
                    if min_err > upper_error && min_flag == 0 {
                        min_err = upper_error;
                        min_flag = 0;
                    }
                    if max_err < upper_error || max_flag == 1 {
                        max_err = upper_error;
                        max_flag = 0;
                    }
                }

                if lower_flag {
                    // err is minus err
                    if min_err < lower_error || min_flag == 0 {
                        min_err = lower_error;
                        min_flag = 1;
                    }
                    if max_err > lower_error && max_flag == 1 {
                        max_err = lower_error;
                        max_flag = 1;
                    }
                } else {
                    // pred is smaller than y, plus err
                    if min_err > lower_error && min_flag == 0 {
                        min_err = lower_error;
                        min_flag = 0;
                    }
                    if max_err < lower_error || max_flag == 1 {
                        max_err = lower_error;
                        max_flag = 0;
                    }
                }
                let num_items_in_leaf =
                    third_layer_max_l1s[third_idx + partial_3rd_idx[leaf_idx].0].0;

                third_layer_max_l1s[third_idx + partial_3rd_idx[leaf_idx].0] = (
                    num_items_in_leaf,
                    min_flag << 62 | min_err << 32 | max_flag << 31 | max_err,
                );
            }
            partial_third_num += 1;

            continue;
        }

        let curr_err = last_layer_max_l1s[leaf_idx].1;
        let mut min_err = (curr_err >> 32) & 0x000000003fffffffu64; // left most bit is used for partial model representation
        let mut min_flag = ((curr_err >> 32) & 0x0000000040000000u64) >> 30;
        let mut max_err = curr_err & 0x000000007fffffffu64;
        let mut max_flag = (curr_err & 0x0000000080000000u64) >> 31;

        let mut upper_flag = false;
        let mut lower_flag = false;

        let upper_error = {
            let (idx_of_next, key_of_next) = lb_corrections.next(leaf_idx);
            let pred =
                sec_models[leaf_idx].predict_to_int(&key_of_next.minus_epsilon().to_model_input());
            upper_flag = pred > idx_of_next as u64;
            error_between(pred, idx_of_next as u64 + 1, md_container.len() as u64)
        };

        let lower_error = {
            let first_key_before = lb_corrections.prev_key(leaf_idx);

            let prev_idx = if leaf_idx == 0 { 0 } else { leaf_idx - 1 };
            let first_idx = lb_corrections.next_index(prev_idx);

            let pred = sec_models[leaf_idx]
                .predict_to_int(&first_key_before.plus_epsilon().to_model_input());
            lower_flag = pred > first_idx as u64;
            error_between(pred, first_idx as u64, md_container.len() as u64)
        };

        if upper_flag {
            // err is minus err
            if min_err < upper_error || min_flag == 0 {
                min_err = upper_error;
                min_flag = 1;
            }
            if max_err > upper_error && max_flag == 1 {
                max_err = upper_error;
                max_flag = 1;
            }
        } else {
            // pred is smaller than y, plus err
            if min_err > upper_error && min_flag == 0 {
                min_err = upper_error;
                min_flag = 0;
            }
            if max_err < upper_error || max_flag == 1 {
                max_err = upper_error;
                max_flag = 0;
            }
        }

        if lower_flag {
            // err is minus err
            if min_err < lower_error || min_flag == 0 {
                min_err = lower_error;
                min_flag = 1;
            }
            if max_err > lower_error && max_flag == 1 {
                max_err = lower_error;
                max_flag = 1;
            }
        } else {
            // pred is smaller than y, plus err
            if min_err > lower_error && min_flag == 0 {
                min_err = lower_error;
                min_flag = 0;
            }
            if max_err < lower_error || max_flag == 1 {
                max_err = lower_error;
                max_flag = 0;
            }
        }
        let num_items_in_leaf = last_layer_max_l1s[leaf_idx].0;

        last_layer_max_l1s[leaf_idx] = (
            num_items_in_leaf,
            min_flag << 62 | min_err << 32 | max_flag << 31 | max_err,
        );
    }

    if large_corrections > 1 {
        trace!(
            "Of {} models, {} needed large lower bound corrections.",
            num_leaf_models,
            large_corrections
        );
    }

    trace!("Evaluating Second layer of RMI...");

    log::debug!(
        "[INFO] Number of leaf and partial models: {}, leaf: {}, partial: {}, leaf models that have partial models:{}",
        num_leaf_models as usize + third_layer_num as usize,
        num_leaf_models,
        third_layer_num,
        partial_3rd_lb_corrs.len()
    );

    let (m_idx, m_err) = last_layer_max_l1s
        .iter()
        .enumerate()
        .max_by_key(|(_idx, &x)| (x.1 & 0x7fffffff) + ((x.1 >> 32) & 0x3fffffff))
        .unwrap();

    let model_max_error = (m_err.1 & 0x7fffffff) + ((m_err.1 >> 32) & 0x3fffffff);
    let model_max_error_idx = m_idx;

    let model_avg_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| n * ((err & 0x7fffffff) + ((err >> 32) & 0x3fffffff)))
        .sum::<u64>() as f64
        / num_rows as f64;

    let model_avg_l2_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| {
            (((err & 0x7fffffff) + ((err >> 32) & 0x3fffffff)) as f64).powf(2.0) / num_rows as f64
        })
        .sum::<f64>();

    let model_avg_log2_error: f64 = last_layer_max_l1s
        .iter()
        .map(|(n, err)| {
            (*n as f64) * (((err & 0x7fffffff) + ((err >> 32) & 0x3fffffff) + 2) as f64).log2()
        })
        .sum::<f64>()
        / num_rows as f64;

    let model_max_log2_error: f64 = (model_max_error as f64).log2();

    let final_errors = last_layer_max_l1s
        .into_iter()
        .map(|(_n, err)| err)
        .collect();

    let final_third_errors = third_layer_max_l1s
        .into_iter()
        .map(|(_n, err)| err)
        .collect();

    let rmi = if third_layer_num > 0 {
        vec![vec![top_model], partial_3rd_models, sec_models]
    } else {
        let dummy_model = train_model("pwl", &md_container);
        vec![vec![top_model], vec![dummy_model], sec_models]
    };

    return TrainedRMI {
        num_rmi_rows: md_container.len(),
        num_data_rows: md_container.len(),
        model_avg_error,
        model_avg_l2_error,
        model_avg_log2_error,
        model_max_error,
        model_max_error_idx,
        model_max_log2_error,
        last_layer_max_l1s: final_errors,
        third_layer_max_l1s: final_third_errors,
        rmi,
        // Note: layer1, layer3, layer2 order is BWA-MEME's spec-string serialization convention
        models: format!("{},{},{}", layer1_model, layer3_model, layer2_model),
        branching_factor: num_leaf_models,
        cache_fix: None,
        build_time: 0,
    };
}
