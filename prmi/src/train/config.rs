// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer configuration knobs. Defaults are adopted from BWA-MEME's
//! published empirical tuning; see citation-chain in the doc comments
//! below. Override via `TrainerConfig` for non-human references or
//! different quality / training-time trade-offs.

/// Configuration for the P-RMI trainer. All fields have BWA-MEME-derived
/// defaults; override only if you know what you're doing.
///
/// `#[non_exhaustive]` so v0.2 can add fields without breaking external
/// constructors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrainerConfig {
    /// Minimum L2 leaf size to trigger L1 fallback partitioning.
    ///
    /// Adopted from BWA-MEME's empirical `make_partial_threshold = 1000`,
    /// validated on the human genome at `l2_leaf_count = 2^28`. Tuning
    /// for non-human references or different memory budgets may benefit
    /// from a different value.
    pub fallback_threshold: usize,

    /// Target keys per L1 sub-model when a leaf triggers fallback.
    ///
    /// Adopted from BWA-MEME's empirical `average_partial_model_num = 20`.
    /// Roughly one sub-model per cache-line-sized cluster of SA entries.
    pub partial_target_size: u64,

    /// Hard cap on L1 array size. Brief §4.4's `partial_start` field uses
    /// 31 bits, capping the L1 array at `2^31 - 1` entries. Exceeding the
    /// cap returns `Error::Internal`.
    pub max_l1_entries: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            fallback_threshold: 1000,
            partial_target_size: 20,
            max_l1_entries: (1 << 31) - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_match_spec() {
        let c = TrainerConfig::default();
        assert_eq!(c.fallback_threshold, 1000);
        assert_eq!(c.partial_target_size, 20);
        assert_eq!(c.max_l1_entries, (1 << 31) - 1);
    }

    #[test]
    fn config_is_clone_and_debug() {
        let c = TrainerConfig::default();
        let _clone: TrainerConfig = c.clone();
        let _debug = format!("{c:?}");
    }
}
