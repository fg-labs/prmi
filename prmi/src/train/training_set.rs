// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Training-set construction for the P-RMI trainer.
//!
//! A [`TrainingSet`] is a sorted list of `(key, sa_index)` pairs the trainer
//! learns to predict: given a 32-mer key, return the SA index of the matching
//! suffix array entry. For v0.1, [`uniform_training_set`] (unmasked) and
//! [`masked_training_set`] (mask-filtered) are shipped. v0.2 will add
//! `bed_filtered_training_set` (filters to on-target intervals) and v0.3 will
//! add `fastq_weighted_training_set` (weights by query-side frequency).

use crate::encoding::{tokenize_32mer, KMER_LEN};
use crate::train::keys::sa_to_keys;
use crate::train::mask::{
    covered_by_bed, homopolymer_in_window, keep_doubled_pos, n_in_window, BedInterval, MaskConfig,
    NBitmap,
};
use crate::train::prior::Prior;
use std::sync::Arc;

/// SA-index targets for a training set — the `y` value the model learns to
/// predict from each key.
///
/// Materialising one `u64` per training pair costs ~51.5 GB at hg38 scale. On
/// the byte-identical `.pac` production path the targets are simply the SA ranks
/// `0..sa_num` minus a tiny set of skipped ranks (the short-window suffixes that
/// can't form a full 32-mer), so they are represented as a dense range plus a
/// small sorted skip list instead. Heavier masking / non-uniform priors keep the
/// explicit `Materialized` form.
///
/// `Dense` and `Materialized` yield bit-identical target sequences for the same
/// input, so the trained model is unchanged by the representation.
#[derive(Debug, Clone)]
pub enum SaIndices {
    /// Targets are `[0, len + skips.len())` with the sorted SA ranks in `skips`
    /// removed. The `i`-th target is the `i`-th rank not present in `skips`.
    Dense {
        /// Number of kept targets (== number of training pairs).
        len: usize,
        /// Sorted, ascending SA ranks excluded from the training set.
        skips: Arc<Vec<u64>>,
    },
    /// Explicit per-pair target list.
    Materialized(Arc<Vec<u64>>),
}

impl Default for SaIndices {
    fn default() -> Self {
        SaIndices::Materialized(Arc::new(Vec::new()))
    }
}

impl SaIndices {
    /// Number of training pairs (targets).
    pub fn len(&self) -> usize {
        match self {
            SaIndices::Dense { len, .. } => *len,
            SaIndices::Materialized(v) => v.len(),
        }
    }

    /// `true` if there are no targets.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`-th SA-index target. For `Dense`, the `i`-th rank in `[0, sa_num)`
    /// not present in `skips`, found by a fixed-point walk over the (tiny) skip
    /// list: the answer `r` satisfies `r = i + |{s in skips : s <= r}|`.
    pub fn get(&self, i: usize) -> u64 {
        match self {
            SaIndices::Materialized(v) => v[i],
            SaIndices::Dense { skips, .. } => {
                let i = i as u64;
                let mut d = 0u64;
                loop {
                    let nd = skips.partition_point(|&s| s <= i + d) as u64;
                    if nd == d {
                        return i + d;
                    }
                    d = nd;
                }
            }
        }
    }

    /// Sequential iterator over the targets in ascending order — identical to
    /// the `Materialized` vector's order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        match self {
            SaIndices::Materialized(v) => SaIndicesIter::Mat(v.iter()),
            SaIndices::Dense { len, skips } => SaIndicesIter::Dense {
                v: 0,
                sa_num: (*len + skips.len()) as u64,
                skips,
                si: 0,
            },
        }
    }
}

/// Concrete iterator backing [`SaIndices::iter`] (hidden behind `impl
/// Iterator`). Static dispatch keeps the hot LBC/verify passes
/// branch-predictable.
enum SaIndicesIter<'a> {
    // Walks `0..sa_num`, skipping ranks present in the sorted `skips`.
    Dense {
        v: u64,
        sa_num: u64,
        skips: &'a [u64],
        si: usize,
    },
    // Wraps the materialized slice iterator.
    Mat(std::slice::Iter<'a, u64>),
}

impl Iterator for SaIndicesIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            SaIndicesIter::Mat(it) => it.next().copied(),
            SaIndicesIter::Dense {
                v,
                sa_num,
                skips,
                si,
            } => {
                while *v < *sa_num {
                    if *si < skips.len() && skips[*si] == *v {
                        *si += 1;
                        *v += 1;
                        continue;
                    }
                    let r = *v;
                    *v += 1;
                    return Some(r);
                }
                None
            }
        }
    }
}

/// 32-mer training keys — the `x` value fed to the model.
///
/// Materialising one `u64` per training pair costs ~51.5 GB at hg38 scale. On
/// the dense build path the key for pair `i` is fully determined by the suffix
/// array and the 2× text (`tokenize_32mer` of the base window at
/// `sa[sa_index(i)]`), so it is recomputed on demand from shared `Arc`s instead
/// — which also removes the need to materialise the separate `text_bases` array.
/// Heavier masking / non-uniform priors keep the explicit `Materialized` form.
///
/// `Streamed` and `Materialized` yield bit-identical keys for the same input
/// (the streamed map matches `text_value_to_base`), so the model is unchanged.
#[derive(Debug, Clone)]
pub enum Keys {
    /// Explicit per-pair key list.
    Materialized(Arc<Vec<u64>>),
    /// Recomputed on demand: `key(i) = tokenize_32mer(base window at
    /// sa[rank])`, where `rank` is the pair's SA rank and `text` is the
    /// `b+1`-alphabet 2× text (mapped to `0..=3` bases inline).
    Streamed {
        /// The full 2× suffix array (doubled coordinates).
        sa: Arc<Vec<u64>>,
        /// The `b+1`-alphabet 2× text (`1..=4` + sentinel `0`).
        text: Arc<Vec<u8>>,
    },
}

impl Default for Keys {
    fn default() -> Self {
        Keys::Materialized(Arc::new(Vec::new()))
    }
}

impl Keys {
    /// The key for training pair `i`, whose SA rank is `rank`. For `Streamed`,
    /// tokenises the 32-base window of the 2× text starting at `sa[rank]`
    /// (every kept pair has a full window). `Materialized` ignores `rank`.
    #[inline]
    pub fn at(&self, i: usize, rank: u64) -> u64 {
        match self {
            Keys::Materialized(v) => v[i],
            Keys::Streamed { sa, text } => {
                let pos = sa[rank as usize] as usize;
                let mut window = [0u8; KMER_LEN];
                for (slot, &v) in window.iter_mut().zip(&text[pos..pos + KMER_LEN]) {
                    *slot = crate::sa::text_value_to_base(v);
                }
                tokenize_32mer(&window, KMER_LEN)
            }
        }
    }
}

/// A sorted set of `(key, sa_index)` training pairs for the P-RMI trainer.
///
/// `keys[i]` and `sa_indices[i]` together form the i-th training pair: the
/// trainer learns to predict `sa_indices[i]` from `keys[i]`. The two vectors
/// always have the same length, and `keys` is non-decreasing.
///
/// `keys` is wrapped in [`Arc`] so the trainer can share it (zero-copy) with the
/// lower-bound-correction pass instead of materialising a `Vec<(u64, usize)>`
/// duplicate of the entire set — a ~100 GB saving at hg38 scale. `sa_indices`
/// uses [`SaIndices`], which on the `.pac` path avoids materialising a second
/// ~51.5 GB vector. Both yield the same values per index, so the model is
/// unchanged by the representation.
///
/// `sa_num` is the total size of the **full** SA (not the number of training
/// pairs). It may be larger than `keys.len()` when masking is active, and is
/// used by the trainer to clamp predictions to the correct range.
///
/// `weights` is an optional per-pair weight vector. When `Some`, it has the
/// same length as `keys` and `sa_indices`. When `None`, all pairs are
/// implicitly weighted 1.0. The BED prior path populates this field.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TrainingSet {
    /// 32-mer keys in non-decreasing order (materialised or streamed).
    pub keys: Keys,
    /// SA-index target for each key — what the model learns to predict.
    pub sa_indices: SaIndices,
    /// Total number of entries in the full SA (may exceed the pair count when
    /// masking is active). The trainer uses this as the prediction clamp bound.
    pub sa_num: u64,
    /// Optional per-pair training weights. `None` means uniform weight (1.0).
    /// When `Some`, the length equals the number of pairs.
    pub weights: Option<Vec<f64>>,
}

impl TrainingSet {
    /// Number of training pairs. The SA-index targets are authoritative;
    /// materialised keys must match (checked in debug builds).
    pub fn len(&self) -> usize {
        if let Keys::Materialized(v) = &self.keys {
            debug_assert_eq!(v.len(), self.sa_indices.len());
        }
        self.sa_indices.len()
    }

    /// `true` if no training pairs.
    pub fn is_empty(&self) -> bool {
        self.sa_indices.is_empty()
    }

    /// The 32-mer key of training pair `i`.
    #[inline]
    pub fn key(&self, i: usize) -> u64 {
        self.keys.at(i, self.sa_indices.get(i))
    }

    /// Sequential iterator over all training keys in pair order.
    pub fn keys_iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len()).map(move |i| self.key(i))
    }
}

/// Produce a uniform training set: one `(key, sa_index)` pair per SA entry.
///
/// The SA must be the lex-sorted suffix array of `bases`. The returned
/// `TrainingSet` has `keys = sa_to_keys(sa, bases)` and
/// `sa_indices = 0..sa.len()`.
pub fn uniform_training_set(sa: &[u64], bases: &[u8]) -> TrainingSet {
    let keys = sa_to_keys(sa, bases);
    let sa_num = sa.len() as u64;
    // One pair per SA entry: targets are exactly 0..sa_num (no skips).
    TrainingSet {
        keys: Keys::Materialized(Arc::new(keys)),
        sa_indices: SaIndices::Dense {
            len: sa.len(),
            skips: Arc::new(Vec::new()),
        },
        sa_num,
        weights: None,
    }
}

/// Produce a mask-filtered training set: one `(key, sa_index)` pair per
/// unmasked SA entry.
///
/// The SA must be the lex-sorted suffix array of `bases`. For each SA entry
/// at index `sa_idx`, the reference position `sa[sa_idx]` is checked against
/// the enabled mask predicates; if any predicate fires, that entry is excluded
/// from the returned `TrainingSet`. The SA file on disk is **not** affected —
/// only the set of (key, SA-index) pairs used to fit and evaluate the model is
/// filtered.
///
/// `n_positions` must have the same length as `bases` and marks positions that
/// were originally N in the reference FASTA. Short windows at the end of the
/// reference (where `sa_pos + 32 > bases.len()`) are also excluded, consistent
/// with the sentinel-padding strategy in the trainer.
///
/// When `prior` is [`Prior::Bed`], the returned `TrainingSet::weights` field
/// is `Some(weights)` with each entry set to the BED weight for pairs inside
/// the BED region and 1.0 for pairs outside. When `prior` is
/// [`Prior::FastqHistogram`], `weights` is `Some(weights)` with each entry set
/// to `base_weight + log2(1 + freq(key))`. When `prior` is [`Prior::Uniform`],
/// `weights` is `None` (uniform weights are implicit).
pub fn masked_training_set(
    sa: &[u64],
    bases: &[u8],
    n_positions: &NBitmap,
    mask: &MaskConfig,
    prior: &Prior,
) -> TrainingSet {
    let n = bases.len();
    let sa_num = sa.len() as u64;

    // Fast path (the `.pac` production build): a uniform prior with no
    // homopolymer/BED mask and no effective N-mask means the ONLY excluded
    // entries are the short-window suffixes (`p + 32 > n`). Their SA ranks are
    // recorded as a small skip list and `sa_indices` is left virtual — avoiding
    // a ~51.5 GB target vector. Keys are identical to the materialized path
    // (every kept entry has `avail == KMER_LEN`, so the same tokenisation).
    let no_n_effect = !mask.mask_n_runs || !n_positions.any();
    let virtualize = matches!(prior, Prior::Uniform)
        && mask.mask_homopolymers.is_none()
        && mask.mask_bed.is_none()
        && no_n_effect;
    if virtualize {
        let mut keys: Vec<u64> = Vec::with_capacity(sa.len());
        let mut skips: Vec<u64> = Vec::new();
        for (sa_idx, &sa_pos) in sa.iter().enumerate() {
            let p = sa_pos as usize;
            if p + KMER_LEN > n {
                skips.push(sa_idx as u64);
                continue;
            }
            keys.push(tokenize_32mer(&bases[p..p + KMER_LEN], KMER_LEN));
        }
        let len = keys.len();
        return TrainingSet {
            keys: Keys::Materialized(Arc::new(keys)),
            sa_indices: SaIndices::Dense {
                len,
                skips: Arc::new(skips),
            },
            sa_num,
            weights: None,
        };
    }

    let mut keys: Vec<u64> = Vec::with_capacity(sa.len());
    let mut sa_indices: Vec<u64> = Vec::with_capacity(sa.len());
    let use_weights = !matches!(prior, Prior::Uniform);
    let mut weights: Vec<f64> = if use_weights {
        Vec::with_capacity(sa.len())
    } else {
        Vec::new()
    };

    for (sa_idx, &sa_pos) in sa.iter().enumerate() {
        let p = sa_pos as usize;

        // Skip short windows (past end of sequence).
        if p + KMER_LEN > n {
            continue;
        }

        // N-run mask.
        if mask.mask_n_runs && n_in_window(n_positions, p) {
            continue;
        }

        // Homopolymer mask.
        if let Some(k) = mask.mask_homopolymers {
            if homopolymer_in_window(bases, p, k) {
                continue;
            }
        }

        // BED mask.
        if let Some(ref intervals) = mask.mask_bed {
            if covered_by_bed(intervals, sa_pos) {
                continue;
            }
        }

        let avail = n.saturating_sub(p).min(KMER_LEN);
        let key = tokenize_32mer(&bases[p..p + avail], avail);
        keys.push(key);
        sa_indices.push(sa_idx as u64);
        if use_weights {
            weights.push(crate::train::prior::weight_for_pair(prior, key, sa_pos));
        }
    }

    TrainingSet {
        keys: Keys::Materialized(Arc::new(keys)),
        sa_indices: SaIndices::Materialized(Arc::new(sa_indices)),
        sa_num,
        weights: if use_weights { Some(weights) } else { None },
    }
}

/// Produce a tiered (Design Z) training set whose SA-index targets are the
/// **compacted ranks** `0..N_kept` over only the keep-set entries — matching the
/// position-filtered `.sa` write order so model predictions index directly into
/// the shrunken on-disk array.
///
/// `sa` is the lex-sorted 2× suffix array (doubled coordinates over the
/// `text_bases` 2× text of length `2*l_pac+1`); `keep` is the forward-coordinate
/// keep-set; `l_pac` is the full forward genome length. The compacted rank is
/// advanced for EVERY retained entry — including short-window suffixes
/// (`p + 32 > text len`) that are written to the `.sa` but excluded as training
/// targets — so the targets stay aligned with the filtered `.sa` indices. Keys
/// remain non-decreasing (a kept subsequence of the lex-sorted SA), and
/// `sa_num == N_kept` is the prediction clamp bound for the tiered index.
///
/// The keep filter (which entries the `.sa` retains) is ORTHOGONAL to the
/// mask/prior options: `mask` (`mask_n_runs`/homopolymer/`mask_bed`) and `prior`
/// are applied to the RETAINED entries exactly as [`masked_training_set`] applies
/// them to the full SA, so `--keep-bed` composes with `--mask-*`/`--prior-*`
/// instead of silently dropping them. A masked entry is still WRITTEN to the
/// `.sa` (only `keep_doubled_pos` filters the write), so the compacted rank still
/// advances for it — it is merely excluded as a training target, like a
/// short-window suffix. `n_positions` is the doubled-text N bitmap (same
/// coordinates as `sa`/`text_bases`).
pub fn keep_masked_training_set(
    sa: &[u64],
    text_bases: &[u8],
    n_positions: &NBitmap,
    keep: &[BedInterval],
    l_pac: u64,
    mask: &MaskConfig,
    prior: &Prior,
) -> TrainingSet {
    let n = text_bases.len();
    let mut keys: Vec<u64> = Vec::new();
    let mut sa_indices: Vec<u64> = Vec::new();
    let use_weights = !matches!(prior, Prior::Uniform);
    let mut weights: Vec<f64> = Vec::new();
    // Compacted rank: the index this entry occupies in the filtered `.sa`.
    let mut compacted: u64 = 0;
    for &sa_pos in sa.iter() {
        // Not retained → absent from both the `.sa` and the training set.
        if !keep_doubled_pos(keep, sa_pos, l_pac) {
            continue;
        }
        // Retained: it occupies `.sa` index `compacted` whether or not it forms
        // a full 32-mer training key OR is masked out below.
        let rank = compacted;
        compacted += 1;
        let p = sa_pos as usize;
        // Short windows past the end of the doubled text are written to the
        // `.sa` but are not trainable targets (no full 32-mer), exactly as in
        // `masked_training_set`.
        if p + KMER_LEN > n {
            continue;
        }
        // Apply the same exclusion masks as `masked_training_set`. Each excluded
        // entry is dropped only as a training target — its `.sa` slot (and hence
        // the compacted rank already advanced above) is preserved.
        if mask.mask_n_runs && n_in_window(n_positions, p) {
            continue;
        }
        if let Some(k) = mask.mask_homopolymers {
            if homopolymer_in_window(text_bases, p, k) {
                continue;
            }
        }
        if let Some(ref intervals) = mask.mask_bed {
            if covered_by_bed(intervals, sa_pos) {
                continue;
            }
        }
        let key = tokenize_32mer(&text_bases[p..p + KMER_LEN], KMER_LEN);
        keys.push(key);
        sa_indices.push(rank);
        if use_weights {
            weights.push(crate::train::prior::weight_for_pair(prior, key, sa_pos));
        }
    }
    TrainingSet {
        keys: Keys::Materialized(Arc::new(keys)),
        sa_indices: SaIndices::Materialized(Arc::new(sa_indices)),
        sa_num: compacted,
        weights: if use_weights { Some(weights) } else { None },
    }
}

/// Build a training set whose keys are STREAMED from the suffix array and 2×
/// text rather than materialised, and whose SA-index targets are dense
/// `0..sa_num` minus the short-window skips. Yields byte-identical
/// `(key, sa_index)` pairs to [`masked_training_set`] on the uniform /
/// no-mask / no-N path, at a fraction of the memory: no ~51.5 GB key vector and
/// no `text_bases` array.
///
/// `text` is the `b+1`-alphabet 2× text (length `sa_num`); `sa` is its
/// generalized suffix array. Only valid for the byte-identical path — the
/// caller gates on a uniform prior with no homopolymer/BED mask and no N
/// positions (see `build_sidecar_core`).
pub fn streamed_training_set(sa: Arc<Vec<u64>>, text: Arc<Vec<u8>>) -> TrainingSet {
    let n = text.len();
    let sa_num = sa.len() as u64;
    let mut skips: Vec<u64> = Vec::new();
    for (sa_idx, &sa_pos) in sa.iter().enumerate() {
        if sa_pos as usize + KMER_LEN > n {
            skips.push(sa_idx as u64);
        }
    }
    let len = sa.len() - skips.len();
    TrainingSet {
        keys: Keys::Streamed {
            sa: Arc::clone(&sa),
            text,
        },
        sa_indices: SaIndices::Dense {
            len,
            skips: Arc::new(skips),
        },
        sa_num,
        weights: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::mask::MaskConfig;
    use std::collections::HashSet;

    /// Explicit kept ranks: `[0, sa_num)` with `skips` removed.
    fn explicit_kept(sa_num: u64, skips: &[u64]) -> Vec<u64> {
        let sk: HashSet<u64> = skips.iter().copied().collect();
        (0..sa_num).filter(|v| !sk.contains(v)).collect()
    }

    #[test]
    fn sa_indices_dense_get_and_iter_match_explicit() {
        let cases: &[(u64, &[u64])] = &[
            (10, &[]),
            (10, &[0]),
            (10, &[9]),
            (10, &[2, 3, 7]),
            (10, &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (1, &[]),
        ];
        for &(sa_num, skips) in cases {
            let kept = explicit_kept(sa_num, skips);
            let dense = SaIndices::Dense {
                len: kept.len(),
                skips: Arc::new(skips.to_vec()),
            };
            assert_eq!(dense.len(), kept.len(), "len skips={skips:?}");
            assert_eq!(
                dense.iter().collect::<Vec<_>>(),
                kept,
                "iter skips={skips:?}"
            );
            for (i, &k) in kept.iter().enumerate() {
                assert_eq!(dense.get(i), k, "get({i}) skips={skips:?}");
            }
        }
    }

    #[test]
    fn sa_indices_dense_random_matches_materialized() {
        let mut state = 0x00C0_FFEEu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for _ in 0..300 {
            let sa_num = 1 + next() % 500;
            let mut skips: Vec<u64> = (0..sa_num).filter(|_| next() % 5 == 0).collect();
            skips.sort_unstable();
            skips.dedup();
            let kept = explicit_kept(sa_num, &skips);
            if kept.is_empty() {
                continue;
            }
            let dense = SaIndices::Dense {
                len: kept.len(),
                skips: Arc::new(skips),
            };
            assert_eq!(dense.iter().collect::<Vec<_>>(), kept);
            for (i, &k) in kept.iter().enumerate() {
                assert_eq!(dense.get(i), k);
            }
        }
    }

    /// The virtualized (Dense) build must yield BYTE-IDENTICAL (key, sa_index)
    /// pairs to an explicit materialized computation over the same input. This
    /// is the coverage the chr17 end-to-end gate cannot provide (chr17.fa has N
    /// runs, so it takes the Materialized path).
    #[test]
    fn masked_uniform_virtualizes_byte_identically() {
        // Synthetic no-N base array (values 0..=3).
        let bases: Vec<u8> = (0..256u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let sa = crate::sa::build_suffix_array(&bases, 1).unwrap();
        let n_positions = NBitmap::zeros(bases.len());

        // Reference materialized (key, rank) pairs via the same short-window rule.
        let mut ref_keys: Vec<u64> = Vec::new();
        let mut ref_idx: Vec<u64> = Vec::new();
        for (sa_idx, &sa_pos) in sa.iter().enumerate() {
            let p = sa_pos as usize;
            if p + KMER_LEN > bases.len() {
                continue;
            }
            ref_keys.push(tokenize_32mer(&bases[p..p + KMER_LEN], KMER_LEN));
            ref_idx.push(sa_idx as u64);
        }
        assert!(
            !ref_idx.is_empty() && ref_idx.len() < sa.len(),
            "test needs some skips"
        );

        // Both mask_n_runs=false and =true (with all-false n_positions) must virtualize.
        for mask_n_runs in [false, true] {
            let mask = MaskConfig {
                mask_n_runs,
                ..MaskConfig::default()
            };
            let ts = masked_training_set(&sa, &bases, &n_positions, &mask, &Prior::Uniform);
            assert!(
                matches!(ts.sa_indices, SaIndices::Dense { .. }),
                "should virtualize (mask_n_runs={mask_n_runs})"
            );
            assert_eq!(
                ts.keys_iter().collect::<Vec<_>>(),
                ref_keys,
                "keys must match reference"
            );
            assert_eq!(
                ts.sa_indices.iter().collect::<Vec<_>>(),
                ref_idx,
                "sa_indices iter must match reference"
            );
            for (i, &r) in ref_idx.iter().enumerate() {
                assert_eq!(ts.sa_indices.get(i), r, "get({i})");
            }
            assert!(ts.weights.is_none());
        }
    }

    /// A homopolymer mask must NOT virtualize (it falls back to Materialized),
    /// preserving the existing masked behaviour.
    #[test]
    fn masked_with_homopolymer_stays_materialized() {
        let bases: Vec<u8> = (0..256u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        let sa = crate::sa::build_suffix_array(&bases, 1).unwrap();
        let n_positions = NBitmap::zeros(bases.len());
        let mask = MaskConfig {
            mask_n_runs: true,
            mask_homopolymers: Some(5),
            ..MaskConfig::default()
        };
        let ts = masked_training_set(&sa, &bases, &n_positions, &mask, &Prior::Uniform);
        assert!(matches!(ts.sa_indices, SaIndices::Materialized(_)));
        assert!(matches!(ts.keys, Keys::Materialized(_)));
        assert_eq!(ts.len(), ts.sa_indices.len());
    }

    /// The tiered (keep-mask) training set must COMPOSE with the mask/prior
    /// options, not silently ignore them: a mask drops training PAIRS while the
    /// compacted rank (and hence `sa_num`) still counts every retained `.sa`
    /// entry (masked entries are still written to the `.sa`), and a non-uniform
    /// prior produces weights aligned 1:1 with the emitted pairs.
    #[test]
    fn keep_masked_composes_with_mask_and_prior() {
        // A base array with an embedded poly-A run so a homopolymer mask has
        // something to drop; the rest cycles 1,0,3,2 with no long homopolymers.
        let mut bases: Vec<u8> = (0..200u32).map(|i| ((i * 7 + 1) % 4) as u8).collect();
        for b in bases.iter_mut().take(72).skip(60) {
            *b = 0; // 12-base poly-A run (>= the homopolymer threshold)
        }
        let l_pac = bases.len() as u64;
        let text = crate::sa::build_doubled_2x_text(&bases);
        let text_bases: Vec<u8> = text
            .iter()
            .map(|&v| crate::sa::text_value_to_base(v))
            .collect();
        let sa = crate::sa::build_gsa(&text, 1).unwrap();
        let n_positions = NBitmap::zeros(text.len()); // no N

        // Keep the whole genome so the keep filter retains every entry — this
        // isolates the mask/prior effect from the keep filter itself.
        let keep = vec![BedInterval {
            start: 0,
            end: l_pac,
        }];
        let no_mask = MaskConfig {
            mask_n_runs: false,
            ..MaskConfig::default()
        };

        let unmasked = keep_masked_training_set(
            &sa,
            &text_bases,
            &n_positions,
            &keep,
            l_pac,
            &no_mask,
            &Prior::Uniform,
        );
        let homo_mask = MaskConfig {
            mask_n_runs: false,
            mask_homopolymers: Some(6),
            ..MaskConfig::default()
        };
        let masked = keep_masked_training_set(
            &sa,
            &text_bases,
            &n_positions,
            &keep,
            l_pac,
            &homo_mask,
            &Prior::Uniform,
        );

        // `sa_num` is the count of RETAINED `.sa` entries and must NOT change
        // when a mask only drops training targets.
        assert_eq!(
            masked.sa_num, unmasked.sa_num,
            "masking must not change the .sa cardinality"
        );
        // The homopolymer mask actually dropped some — but not all — training pairs.
        assert!(
            masked.len() < unmasked.len(),
            "homopolymer mask should drop training pairs (masked={}, unmasked={})",
            masked.len(),
            unmasked.len()
        );
        assert!(!masked.is_empty(), "mask should not drop every pair");
        // Every retained index is a valid compacted rank and a subset of the
        // unmasked indices (masking only removes pairs, never adds/renumbers).
        let unmasked_idx: std::collections::HashSet<u64> = unmasked.sa_indices.iter().collect();
        for idx in masked.sa_indices.iter() {
            assert!(idx < masked.sa_num, "compacted rank {idx} out of range");
            assert!(
                unmasked_idx.contains(&idx),
                "masked index {idx} is not a subset of the unmasked indices"
            );
        }

        // A non-uniform prior must yield weights aligned 1:1 with the pairs.
        let prior = Prior::Bed {
            intervals: vec![BedInterval {
                start: 0,
                end: l_pac,
            }],
            weight: 3.5,
            path: None,
        };
        let weighted = keep_masked_training_set(
            &sa,
            &text_bases,
            &n_positions,
            &keep,
            l_pac,
            &no_mask,
            &prior,
        );
        let w = weighted
            .weights
            .as_ref()
            .expect("a non-uniform prior must produce weights");
        assert_eq!(
            w.len(),
            weighted.len(),
            "weights must align 1:1 with training pairs"
        );
        // Forward-half pairs fall in the BED (weight 3.5); RC-half pairs sit at
        // doubled coords >= l_pac, outside the forward BED, so weight 1.0.
        assert!(
            w.iter().all(|&x| x == 3.5 || x == 1.0),
            "every weight is either the BED weight or 1.0"
        );
        assert!(
            w.contains(&1.0),
            "some RC-half pair must receive the default weight"
        );
        assert!(
            w.contains(&3.5),
            "some forward-half pair must receive the BED weight"
        );
    }

    /// Streamed keys (recomputed from the 2× SA + text) must equal the keys a
    /// materialized build would produce from `text_bases`.
    #[test]
    fn streamed_keys_match_materialized() {
        let bases: Vec<u8> = (0..300u32).map(|i| ((i * 5 + 2) % 4) as u8).collect();
        let text = crate::sa::build_doubled_2x_text(&bases);
        let sa = crate::sa::build_gsa(&text, 1).unwrap();
        let n = text.len();

        // Reference keys via the materialized text_bases tokenisation.
        let text_bases: Vec<u8> = text
            .iter()
            .map(|&v| crate::sa::text_value_to_base(v))
            .collect();
        let mut ref_keys: Vec<u64> = Vec::new();
        let mut ref_idx: Vec<u64> = Vec::new();
        for (sa_idx, &sa_pos) in sa.iter().enumerate() {
            let p = sa_pos as usize;
            if p + KMER_LEN > n {
                continue;
            }
            ref_keys.push(tokenize_32mer(&text_bases[p..p + KMER_LEN], KMER_LEN));
            ref_idx.push(sa_idx as u64);
        }

        let ts = streamed_training_set(Arc::new(sa), Arc::new(text));
        assert!(matches!(ts.keys, Keys::Streamed { .. }));
        assert_eq!(
            ts.keys_iter().collect::<Vec<_>>(),
            ref_keys,
            "streamed keys must match materialized"
        );
        assert_eq!(ts.sa_indices.iter().collect::<Vec<_>>(), ref_idx);
    }
}
