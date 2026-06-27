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
/// data alongside each SA position so that the query path's candidate-scan
/// loop can skip per-candidate pac reads and tokenization.
///
/// Sizes below assume the 2× sidecar layout (`2*l_pac + 1` SA entries — forward
/// + reverse-complement + sentinel), so they are roughly double a forward-only SA.
///
/// | Mode | Bytes/entry | Extra data stored | ~Size for human genome (2×) |
/// |---|---|---|---|
/// | `Mode1` | 5 | nothing (position only) | ~30 GB |
/// | `Mode2` | 13 | position + 8-byte 32-mer key | ~78 GB |
///
/// **Mode 1 is the default** and the only mode that existed in v0.1 before
/// this menu was introduced. Existing sidecars are always mode 1.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// 5 B/entry — position only. Default. No keys or ISA stored.
    #[default]
    Mode1,
    /// 13 B/entry — position + stored 32-mer key. Skips per-candidate
    /// pac tokenization in the query path.
    Mode2,
}

impl MemoryMode {
    /// Bytes used per entry in the `.sa` file for this mode.
    pub fn bytes_per_entry(self) -> u8 {
        match self {
            MemoryMode::Mode1 => 5,
            MemoryMode::Mode2 => 13,
        }
    }

    /// Human-readable encoding name stored in `.meta [sa] encoding`.
    pub fn encoding_name(self) -> &'static str {
        match self {
            MemoryMode::Mode1 => "packed_lo8_hi32",
            MemoryMode::Mode2 => "packed_lo8_hi32_key64",
        }
    }

    /// Integer mode tag stored in `.meta [sa] mode`.
    pub fn mode_tag(self) -> Option<u8> {
        match self {
            MemoryMode::Mode1 => Some(1),
            MemoryMode::Mode2 => Some(2),
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

    /// If `Some(k)`, build and persist a `.kmt` k-mer table of order `k` that
    /// accelerates `forward_spectrum`'s shallow prefix bands. `None` (default)
    /// builds no table. `k` is capped to the reference size at build time.
    pub kmer_table_k: Option<u32>,

    /// If `true`, also emit a `.isa` inverse-suffix-array sidecar (the ISA launch
    /// hint for `prmi_mem_search`'s no-search fast path). `false` (default) builds
    /// no `.isa`. Costs ~+5 B per SA entry on disk (~+32 GB at hg38) — opt-in via
    /// `prmi build --with-isa`.
    pub with_isa: bool,

    /// If `true`, also emit a `.blm` bloom-filter sidecar over the keep-set's
    /// 32-mer keys — the Design-Z any-window dispatch gate
    /// ([`crate::index::LearnedIndex::present_anchor_bloom`]). `false` (default)
    /// builds no `.blm`. Meaningful for a tiered `--keep-bed` build (the bloom is
    /// then small and cache-resident); opt-in via `prmi build --with-bloom`.
    pub with_bloom: bool,

    /// Target false-positive rate for the `.blm` bloom gate (used only when
    /// `with_bloom`). Controls bloom size and probe count via the optimal
    /// formulas; a false positive only costs an extra confirming `mem_search`
    /// (never a wrong verdict), so a modest default is fine. Defaults to `0.01`.
    pub bloom_fp_rate: f64,

    /// Design-Z Lever 3: pad the `.blm` routing key set by this many bases on each
    /// side of every keep-bed interval, while leaving the `.sa` keep-mask TIGHT.
    /// This DECOUPLES routing (the bloom) from seeding (the SA): the wider bloom
    /// routes capture-boundary / flank-starting reads to the on-target index (whose
    /// in-target windows still seed; the 5' flank soft-clips), recovering recall
    /// WITHOUT the 3.3× SA bloat that padding the keep-bed itself would cause.
    /// `0` (default) keeps the bloom coupled to the keep-set (prior behavior).
    /// Used only when `with_bloom` and a `keep_bed` is set. Padding is in the flat
    /// concatenated-genome coordinate space; intervals are clamped to
    /// `[0, l_pac)` and merged (a ±pad straddling a chromosome junction may add a
    /// few cross-boundary keys, harmless to the best-effort/confirmed gate).
    pub routing_pad: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            fallback_threshold: 1000,
            partial_target_size: 20,
            max_l1_entries: (1 << 31) - 1,
            prior: Prior::Uniform,
            memory_mode: MemoryMode::Mode1,
            kmer_table_k: None,
            with_isa: false,
            with_bloom: false,
            bloom_fp_rate: 0.01,
            routing_pad: 0,
        }
    }
}

impl TrainerConfig {
    /// Return a copy of this config that builds a `.kmt` k-mer table of order
    /// `k` (forward-spectrum shallow-band accelerator).
    ///
    /// Convenience method for external callers that cannot use struct
    /// initialisation syntax due to the `#[non_exhaustive]` attribute.
    pub fn with_kmer_table_k(self, k: u32) -> Self {
        Self {
            kmer_table_k: Some(k),
            ..self
        }
    }

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

    /// Return a copy of this config that also emits a `.isa` inverse-suffix-array
    /// sidecar (the ISA launch hint).
    pub fn with_isa(self, with_isa: bool) -> Self {
        Self { with_isa, ..self }
    }

    /// Return a copy of this config that also emits a `.blm` bloom-filter sidecar
    /// (the any-window dispatch gate) at the given target false-positive rate.
    pub fn with_bloom(self, bloom_fp_rate: f64) -> Self {
        Self {
            with_bloom: true,
            bloom_fp_rate,
            ..self
        }
    }

    /// Return a copy of this config that pads the `.blm` routing key set by `pad`
    /// bases on each side of the keep-bed intervals (Design-Z Lever 3), leaving the
    /// `.sa` keep-mask tight. See [`TrainerConfig::routing_pad`].
    pub fn with_routing_pad(self, pad: u64) -> Self {
        Self {
            routing_pad: pad,
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
    }

    #[test]
    fn memory_mode_encoding_names() {
        assert_eq!(MemoryMode::Mode1.encoding_name(), "packed_lo8_hi32");
        assert_eq!(MemoryMode::Mode2.encoding_name(), "packed_lo8_hi32_key64");
    }

    #[test]
    fn memory_mode_tags() {
        assert_eq!(MemoryMode::Mode1.mode_tag(), Some(1));
        assert_eq!(MemoryMode::Mode2.mode_tag(), Some(2));
    }
}
