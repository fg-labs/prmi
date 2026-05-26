// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer configuration knobs. Defaults are adopted from BWA-MEME's
//! published empirical tuning; see citation-chain in the doc comments
//! below. Override via `TrainerConfig` for non-human references or
//! different quality / training-time trade-offs.

use crate::train::prior::Prior;

/// Memory-mode for the `.sa` sidecar file.
///
/// Each mode trades additional disk/RAM for faster lookup by storing extra
/// data alongside each SA position so that the `smem_range` inner loop can
/// skip per-candidate `read_unpacked_window + tokenize_32mer` calls.
///
/// | Mode | Bytes/entry | Extra data stored | ~Size for human genome |
/// |---|---|---|---|
/// | `Mode1` | 5 | nothing (position only) | ~15 GB |
/// | `Mode2` | 13 | position + 8-byte 32-mer key | ~39 GB |
/// | `Mode3` | 21 | position + 8-byte key + 8-byte ISA | ~63 GB |
/// | `SuffixKeyCache` | 5 + separate `.skc` | top-N keys in a companion sidecar | varies |
///
/// **Mode 1 is the default** and the only mode that existed in v0.1 before
/// this menu was introduced. Existing sidecars are always mode 1.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// 5 B/entry — position only. Default. No keys or ISA stored.
    #[default]
    Mode1,
    /// 13 B/entry — position + stored 32-mer key. Skips per-candidate
    /// pac tokenization in `smem_range`.
    Mode2,
    /// 21 B/entry — position + stored key + ISA entry. Adds forward-
    /// extension capability on top of mode 2.
    Mode3,
    /// 5 B/entry in `.sa` + separate `.skc` companion file containing
    /// keys for the top-N most-queried positions.
    SuffixKeyCache {
        /// Number of (sa_index, key) pairs to cache.
        cache_size: u64,
    },
}

impl MemoryMode {
    /// Bytes used per entry in the `.sa` file for this mode.
    ///
    /// For `SuffixKeyCache` the `.sa` layout is identical to mode 1 (5 B/entry);
    /// the keys live in the separate `.skc` file.
    pub fn bytes_per_entry(self) -> u8 {
        match self {
            MemoryMode::Mode1 | MemoryMode::SuffixKeyCache { .. } => 5,
            MemoryMode::Mode2 => 13,
            MemoryMode::Mode3 => 21,
        }
    }

    /// Human-readable encoding name stored in `.meta [sa] encoding`.
    pub fn encoding_name(self) -> &'static str {
        match self {
            MemoryMode::Mode1 | MemoryMode::SuffixKeyCache { .. } => "packed_lo8_hi32",
            MemoryMode::Mode2 => "packed_lo8_hi32_key64",
            MemoryMode::Mode3 => "packed_lo8_hi32_key64_isa64",
        }
    }

    /// Integer mode tag stored in `.meta [sa] mode`.
    ///
    /// Returns `None` for `SuffixKeyCache` (its mode string is `"suffix_key_cache"`).
    pub fn mode_tag(self) -> Option<u8> {
        match self {
            MemoryMode::Mode1 => Some(1),
            MemoryMode::Mode2 => Some(2),
            MemoryMode::Mode3 => Some(3),
            MemoryMode::SuffixKeyCache { .. } => None,
        }
    }
}

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

    /// Training-pair prior. Controls how (key, sa_index) pairs are weighted
    /// during model fitting. Defaults to [`Prior::Uniform`] (all pairs
    /// weighted equally). Set to [`Prior::Bed`] to bias the fit toward
    /// regions covered by a target-capture BED file.
    pub prior: Prior,

    /// Memory mode for the `.sa` sidecar file. Controls the bytes-per-entry
    /// layout and what extra data is stored alongside each SA position.
    /// Defaults to [`MemoryMode::Mode1`] (position only, 5 B/entry).
    pub memory_mode: MemoryMode,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            fallback_threshold: 1000,
            partial_target_size: 20,
            max_l1_entries: (1 << 31) - 1,
            prior: Prior::Uniform,
            memory_mode: MemoryMode::Mode1,
        }
    }
}

impl TrainerConfig {
    /// Return a copy of this config with the given [`MemoryMode`].
    ///
    /// Convenience method for external callers that cannot use struct
    /// initialisation syntax due to the `#[non_exhaustive]` attribute.
    pub fn with_memory_mode(self, mode: MemoryMode) -> Self {
        Self {
            memory_mode: mode,
            ..self
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
        assert_eq!(c.memory_mode, MemoryMode::Mode1);
    }

    #[test]
    fn config_is_clone_and_debug() {
        let c = TrainerConfig::default();
        let _clone: TrainerConfig = c.clone();
        let _debug = format!("{c:?}");
    }

    #[test]
    fn memory_mode_bytes_per_entry() {
        assert_eq!(MemoryMode::Mode1.bytes_per_entry(), 5);
        assert_eq!(MemoryMode::Mode2.bytes_per_entry(), 13);
        assert_eq!(MemoryMode::Mode3.bytes_per_entry(), 21);
        assert_eq!(
            MemoryMode::SuffixKeyCache { cache_size: 1000 }.bytes_per_entry(),
            5
        );
    }

    #[test]
    fn memory_mode_encoding_names() {
        assert_eq!(MemoryMode::Mode1.encoding_name(), "packed_lo8_hi32");
        assert_eq!(MemoryMode::Mode2.encoding_name(), "packed_lo8_hi32_key64");
        assert_eq!(
            MemoryMode::Mode3.encoding_name(),
            "packed_lo8_hi32_key64_isa64"
        );
        assert_eq!(
            MemoryMode::SuffixKeyCache { cache_size: 100 }.encoding_name(),
            "packed_lo8_hi32"
        );
    }

    #[test]
    fn memory_mode_tags() {
        assert_eq!(MemoryMode::Mode1.mode_tag(), Some(1));
        assert_eq!(MemoryMode::Mode2.mode_tag(), Some(2));
        assert_eq!(MemoryMode::Mode3.mode_tag(), Some(3));
        assert_eq!(
            MemoryMode::SuffixKeyCache { cache_size: 1 }.mode_tag(),
            None
        );
    }
}
