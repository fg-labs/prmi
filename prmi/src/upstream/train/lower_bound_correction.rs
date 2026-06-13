// Copyright Ryan Marcus 2020          (origin: learnedsystems/RMI)
// Modified by Fulcrum Genomics 2026
// SPDX-License-Identifier: MIT

use super::super::models::*;

fn find_first_below<T: Copy>(data: &[Option<T>], idx: usize) -> Option<(usize, T)> {
    assert!(idx < data.len());
    if idx == 0 {
        return None;
    }

    let mut i = idx - 1;
    loop {
        if let Some(v) = data[i] {
            return Some((i, v));
        }
        if i == 0 {
            return None;
        }
        i -= 1;
    }
}

fn find_first_above<T: Copy>(data: &[Option<T>], idx: usize) -> Option<(usize, T)> {
    assert!(idx < data.len());
    if idx == data.len() - 1 {
        return None;
    }

    let mut i = idx + 1;
    loop {
        if let Some(v) = data[i] {
            return Some((i, v));
        }
        if i == data.len() - 1 {
            return None;
        }
        i += 1;
    }
}

// next_for_leaf[i] stores the (key index, key) pairs for the first key in the
// leaf model after leaf i. next_for_leaf[last leaf index] stores the maximum possible key.
//
// next_is_real[i] is true when next_for_leaf[i] is a real training pair and
// false when it is the sentinel (num_keys, T::max_value()) meaning no next
// non-empty leaf exists. The sentinel encoding is ambiguous for pathological
// inputs where a real all-T (TTTT…T) 32-mer is the first key of some leaf
// (T::max_value() == u64::MAX is a valid tokenisation output). Tracking a
// dedicated bool avoids the false-positive sentinel detection that would
// otherwise occur in that corner case.
fn compute_next_for_leaf<T: TrainingKey>(
    num_leaf_models: u64,
    num_keys: usize,
    first_key_for_leaf: &[Option<(usize, T)>],
) -> (Vec<(usize, T)>, Vec<bool>) {
    let mut next_for_leaf = vec![(0, T::zero_value()); num_leaf_models as usize];
    let mut next_is_real = vec![false; num_leaf_models as usize];
    let mut idx: usize = 0;
    while idx < num_leaf_models as usize {
        match find_first_above(&first_key_for_leaf, idx as usize) {
            Some((next_leaf_idx, val)) => {
                assert!(next_leaf_idx > idx);
                for i in idx..next_leaf_idx {
                    next_for_leaf[i] = val;
                    next_is_real[i] = true;
                }
                idx = next_leaf_idx;
            }
            None => {
                for i in idx..num_leaf_models as usize {
                    next_for_leaf[i] = (num_keys, T::max_value());
                    // next_is_real[i] stays false — sentinel case.
                }
                break;
            }
        }
    }

    (next_for_leaf, next_is_real)
}

// prev_for_leaf[i] stores the (sa_index, key) of the LAST training pair in
// the most recent non-empty leaf before leaf i.
//
// prev_is_real[i] is true when a previous non-empty leaf exists and false when
// leaf i has no predecessor (i.e. it is the first non-empty leaf or all leaves
// before it are empty). This avoids the ambiguity where (0, 0) could be either
// a real pair or the zero-initialized sentinel.
fn compute_prev_for_leaf<T: TrainingKey>(
    num_leaf_models: u64,
    last_key_for_leaf: &[Option<(usize, T)>],
) -> (Vec<(usize, T)>, Vec<bool>) {
    let mut prev_for_leaf: Vec<(usize, T)> = vec![(0, T::zero_value()); num_leaf_models as usize];
    let mut prev_is_real = vec![false; num_leaf_models as usize];
    let mut idx: usize = num_leaf_models as usize - 1;
    while idx > 0 {
        match find_first_below(&last_key_for_leaf, idx as usize) {
            Some((prev_leaf_idx, val)) => {
                assert!(prev_leaf_idx < idx);
                for i in prev_leaf_idx + 1..idx + 1 {
                    prev_for_leaf[i] = val;
                    prev_is_real[i] = true;
                }
                idx = prev_leaf_idx;
            }
            None => {
                break;
            }
        }
    }

    (prev_for_leaf, prev_is_real)
}

pub struct LowerBoundCorrection<T> {
    first: Vec<Option<(usize, T)>>,
    last: Vec<Option<(usize, T)>>,
    next: Vec<(usize, T)>,
    next_is_real: Vec<bool>,
    prev: Vec<(usize, T)>,
    prev_is_real: Vec<bool>,
    run_lengths: Vec<u64>,
}

impl<T: TrainingKey> LowerBoundCorrection<T> {
    pub fn new<F>(
        pred_func: F,
        num_leaf_models: u64,
        data: &RMITrainingData<T>,
    ) -> LowerBoundCorrection<T>
    where
        F: Fn(T) -> u64,
    {
        let mut first_key_for_leaf: Vec<Option<(usize, T)>> = vec![None; num_leaf_models as usize];
        let mut last_key_for_leaf: Vec<Option<(usize, T)>> = vec![None; num_leaf_models as usize];
        let mut max_run_length: Vec<u64> = vec![0; num_leaf_models as usize];

        let mut last_target = 0;
        let mut current_run_length = 0;
        // Guard empty input: `data.get_key(0)` panics on an empty dataset and
        // there are no runs to record. The iterator below is also empty, so
        // first/last/next/prev stay correctly unpopulated.
        if data.len() > 0 {
            let mut current_run_key = data.get_key(0);
            for (x, y) in data.iter() {
                let leaf_idx = pred_func(x.into());
                let target = u64::min(num_leaf_models - 1, leaf_idx) as usize;

                if target == last_target && x == current_run_key {
                    current_run_length += 1;
                } else if target != last_target || x != current_run_key {
                    max_run_length[last_target] =
                        u64::max(max_run_length[last_target], current_run_length);

                    current_run_length = 1;
                    current_run_key = x;
                    last_target = target;
                }

                if first_key_for_leaf[target].is_none() {
                    first_key_for_leaf[target] = Some((y, x));
                }
                last_key_for_leaf[target] = Some((y, x));
            }

            // Flush the final run. The loop only commits a run to
            // `max_run_length` on a transition, so the last run — and the only
            // run for a single-run dataset — would otherwise never be recorded,
            // making `longest_run` underreport (returning 0 for single-run data).
            max_run_length[last_target] =
                u64::max(max_run_length[last_target], current_run_length);
        }

        let (next_for_leaf, next_is_real) =
            compute_next_for_leaf(num_leaf_models, data.len(), &first_key_for_leaf);
        let (prev_for_leaf, prev_is_real) =
            compute_prev_for_leaf(num_leaf_models, &last_key_for_leaf);

        return LowerBoundCorrection {
            first: first_key_for_leaf,
            last: last_key_for_leaf,
            next: next_for_leaf,
            next_is_real,
            prev: prev_for_leaf,
            prev_is_real,
            run_lengths: max_run_length,
        };
    }

    pub fn first_key(&self, leaf_idx: usize) -> Option<T> {
        return self.first[leaf_idx].map(|x| x.1);
    }

    pub fn last_key(&self, leaf_idx: usize) -> Option<T> {
        return self.last[leaf_idx].map(|x| x.1);
    }

    /// Returns the raw next `(sa_index, key)` entry for `leaf_idx`.
    ///
    /// **Caller must check `is_next_real` first**; otherwise the returned value
    /// may be the sentinel encoding `(num_keys, T::max_value())` rather than a
    /// real training pair. Prefer [`next_real`](Self::next_real), which returns
    /// `Option` and is self-evident.
    pub fn next(&self, leaf_idx: usize) -> (usize, T) {
        return self.next[leaf_idx];
    }

    /// Returns the raw next `sa_index` for `leaf_idx`.
    ///
    /// **Caller must check `is_next_real` first**; otherwise the returned value
    /// may be the sentinel `num_keys`. Prefer [`next_index_real`](Self::next_index_real).
    pub fn next_index(&self, leaf_idx: usize) -> usize {
        return self.next[leaf_idx].0;
    }

    /// Returns `true` when `next(leaf_idx)` is a real training pair (a non-empty
    /// leaf follows `leaf_idx`) and `false` when it is the sentinel value meaning
    /// no next non-empty leaf exists.
    ///
    /// Prefer this over comparing the next key against `u64::MAX`: a real all-T
    /// 32-mer encodes as `u64::MAX` and is indistinguishable from the sentinel by
    /// key value alone.
    pub fn is_next_real(&self, leaf_idx: usize) -> bool {
        return self.next_is_real[leaf_idx];
    }

    /// Returns `Some((sa_index, key))` when a next non-empty leaf follows
    /// `leaf_idx`, or `None` when `leaf_idx` is in the trailing group with no
    /// real successor.
    ///
    /// This is the preferred alternative to calling `next` + `is_next_real`
    /// separately: the `Option` return makes the sentinel encoding self-evident
    /// and prevents callers from accidentally using the sentinel value as if it
    /// were a real training pair.
    pub fn next_real(&self, leaf_idx: usize) -> Option<(usize, T)> {
        if self.next_is_real[leaf_idx] {
            Some(self.next[leaf_idx])
        } else {
            None
        }
    }

    /// Returns `Some(sa_index)` when a next non-empty leaf follows `leaf_idx`,
    /// or `None` otherwise.
    ///
    /// See [`next_real`](Self::next_real) for rationale.
    pub fn next_index_real(&self, leaf_idx: usize) -> Option<usize> {
        if self.next_is_real[leaf_idx] {
            Some(self.next[leaf_idx].0)
        } else {
            None
        }
    }

    /// Returns the raw previous key for `leaf_idx`.
    ///
    /// **Caller must check `is_prev_real` first**; otherwise the returned value
    /// is the zero-initialized default `T::zero_value()` rather than a real
    /// predecessor key. Prefer [`prev_real`](Self::prev_real), which returns
    /// `Option` and is self-evident.
    pub fn prev_key(&self, leaf_idx: usize) -> T {
        return self.prev[leaf_idx].1;
    }

    /// Returns `true` when a previous non-empty leaf exists before `leaf_idx`.
    ///
    /// When `false`, `leaf_idx` is the first non-empty leaf or all preceding
    /// leaves are empty — there is no valid `prev_sa_index`.
    pub fn is_prev_real(&self, leaf_idx: usize) -> bool {
        return self.prev_is_real[leaf_idx];
    }

    /// Returns `Some((sa_index, key))` of the last training pair in the previous
    /// non-empty leaf, or `None` when `leaf_idx` has no real predecessor.
    ///
    /// This is the preferred alternative to calling `prev_key` / `prev_sa_index`
    /// + `is_prev_real` separately.
    pub fn prev_real(&self, leaf_idx: usize) -> Option<(usize, T)> {
        if self.prev_is_real[leaf_idx] {
            Some(self.prev[leaf_idx])
        } else {
            None
        }
    }

    /// Returns `Some(sa_index)` of the last training pair in the previous
    /// non-empty leaf, or `None` when `leaf_idx` has no real predecessor.
    ///
    /// See [`prev_real`](Self::prev_real) for rationale.
    pub fn prev_index_real(&self, leaf_idx: usize) -> Option<usize> {
        if self.prev_is_real[leaf_idx] {
            Some(self.prev[leaf_idx].0)
        } else {
            None
        }
    }

    /// Returns the SA index of the last training pair in the previous non-empty
    /// leaf. Only meaningful when `is_prev_real(leaf_idx)` is `true`.
    pub fn prev_sa_index(&self, leaf_idx: usize) -> usize {
        return self.prev[leaf_idx].0;
    }

    pub fn longest_run(&self, leaf_idx: usize) -> u64 {
        return self.run_lengths[leaf_idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upstream::models::RMITrainingData;

    /// Build a `LowerBoundCorrection` from a simple flat `(key, sa_index)` list
    /// and a fixed leaf count, using key-based routing.
    fn build_lbc(pairs: Vec<(u64, usize)>, num_leaves: u64) -> LowerBoundCorrection<u64> {
        let data = RMITrainingData::<u64>::new(Box::new(pairs));
        LowerBoundCorrection::new(|k: u64| k >> 60, num_leaves, &data)
    }

    /// Regression: when keys include u64::MAX (the same bit-pattern as the
    /// sentinel), `next_real` must still return `None` for leaves that are
    /// genuinely trailing and `Some` for leaves that have a real next neighbour.
    ///
    /// With 16 leaves (bit_shift = 60), u64::MAX routes to leaf 15.
    /// Leaf 0 contains a key that resolves to the all-T 32-mer (u64::MAX),
    /// leaf 15 contains the same key.  Leaf 15 is the last non-empty leaf so
    /// its `next` entry is the sentinel.  Without the `next_is_real` bool, a
    /// naive comparison against `u64::MAX` would incorrectly treat leaf 0's
    /// "next" (which points at leaf 15's real first key, which happens to BE
    /// u64::MAX) as a sentinel.
    #[test]
    fn next_real_distinguishes_sentinel_from_all_t_key() {
        // Leaf 0 key: 0 (routes to leaf 0 via >> 60).
        // Leaf 15 key: u64::MAX (routes to leaf 15 via >> 60; also == sentinel value).
        let pairs: Vec<(u64, usize)> = vec![(0u64, 0), (u64::MAX, 1)];
        let lbc = build_lbc(pairs, 16);

        // Leaf 0 has a next (leaf 15 with key u64::MAX) — must be Some.
        let next0 = lbc.next_real(0);
        assert!(
            next0.is_some(),
            "leaf 0 has a real next (leaf 15); next_real should be Some, got None"
        );
        let (next_sa, next_key) = next0.unwrap();
        assert_eq!(
            next_key,
            u64::MAX,
            "next key from leaf 0 should be u64::MAX"
        );
        assert_eq!(next_sa, 1, "next SA index from leaf 0 should be 1");

        // Leaf 15 is the trailing leaf — must be None.
        let next15 = lbc.next_real(15);
        assert!(
            next15.is_none(),
            "leaf 15 is trailing; next_real should be None, got {:?}",
            next15
        );

        // Leaves 1–14 are empty but have leaf 15 as their next non-empty
        // leaf, so `next_real` returns `Some` for each of them.
        for li in 1usize..15 {
            assert!(
                lbc.next_real(li).is_some(),
                "leaf {li} should have leaf 15 as its next (Some), got None"
            );
        }

        // prev_real for leaf 15: leaf 0 is a real predecessor.
        let prev15 = lbc.prev_real(15);
        assert!(
            prev15.is_some(),
            "leaf 15 has a real prev (leaf 0); prev_real should be Some"
        );

        // prev_real for leaf 0: no predecessor.
        let prev0 = lbc.prev_real(0);
        assert!(
            prev0.is_none(),
            "leaf 0 has no predecessor; prev_real should be None"
        );
    }

    /// Regression: a single-run dataset (all entries share one key/leaf, so the
    /// run-length loop never sees a transition) must still report the full run
    /// length. Before the final-run flush, `longest_run` returned 0 here.
    ///
    /// All iterated entries fall in one run, so `longest_run(0)` must equal the
    /// number of entries the trainer iterates. (We derive the expected count
    /// from `data.iter()` rather than hard-coding it.)
    #[test]
    fn single_run_reports_full_length() {
        let pairs: Vec<(u64, usize)> = vec![(5u64, 0), (5u64, 1), (5u64, 2)];
        let data = RMITrainingData::<u64>::new(Box::new(pairs));
        let expected = data.iter().count() as u64;
        assert!(expected > 0, "test setup: dataset must be non-empty");
        let lbc = LowerBoundCorrection::new(|k: u64| k >> 60, 16, &data);
        assert_eq!(
            lbc.longest_run(0),
            expected,
            "single-run dataset should report longest_run equal to its iterated length"
        );
    }

    /// Regression: building over an empty dataset must not panic (the run-length
    /// loop previously called `data.get_key(0)` unconditionally).
    #[test]
    fn empty_data_does_not_panic() {
        let lbc = build_lbc(Vec::new(), 16);
        assert_eq!(lbc.longest_run(0), 0);
        assert!(lbc.first_key(0).is_none());
        assert!(lbc.next_real(0).is_none());
    }
}
