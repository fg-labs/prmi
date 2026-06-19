// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Trainer: builds a prmi sidecar from a reference FASTA or bwa `.pac`.

pub mod config;
pub mod keys;
pub mod mask;
pub mod prior;
pub mod prmi;
pub mod trainer;
pub mod training_set;
pub mod verify;

use crate::encoding::{tokenize_32mer, BASE_T, KMER_LEN};
use crate::error::{Error, Result};
use crate::fasta::fasta_to_2bit_with_sha256;
use crate::index::smem::PacEncoding;
use crate::index::LearnedIndex;
use crate::sidecar::kmt_file::KmtFileWriter;
use crate::sidecar::magic::{FORMAT_VERSION, META_MAGIC};
use crate::sidecar::meta::{Meta, Priors, Prmi, Ref, RmiSpec, Sa};
use crate::sidecar::model_file::{ModelFileWriter, ModelLayer};
use crate::sidecar::sa_file::{SaFileWriter, BPE_MODE2};
use crate::sidecar::SidecarPaths;
use crate::train::config::{MemoryMode, TrainerConfig};
use crate::train::mask::{MaskConfig, NBitmap};
use crate::train::prior::Prior;
use crate::train::trainer::default_l2_leaf_count;
use crate::train::training_set::{
    keep_masked_training_set, masked_training_set, streamed_training_set,
};
use crate::train::verify::compute_error_distribution;
use std::path::Path;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Where the forward reference bases come from.
pub enum BuildSource<'a> {
    /// bwa's forward `.pac` — byte-identical to the FMI (N already substituted).
    Pac(&'a Path),
    /// A reference FASTA — NOT byte-identical to a bwa FMI (prmi maps N→A).
    Fasta(&'a Path),
}

/// Build a complete sidecar (`.meta`, `.sa`, `.l1`, `.l2`) from a reference FASTA.
///
/// `prefix` is the output prefix; e.g., for `/data/ref.fa.prmi` the four
/// files become `/data/ref.fa.prmi.{meta,sa,l1,l2}`.
///
/// If `l2_leaf_count` is `None`, it is auto-scaled from the SA size (targeting
/// ~12 entries per leaf, rounded down to a power of two, clamped to [16, 2^28]).
/// If `Some(n)`, the provided `n` must be a power of two ≥ 2.
///
/// The `mask` parameter controls which (key, SA-index) training pairs are
/// excluded from model fitting and error-bound verification. The SA on disk
/// always contains all genome-region entries regardless of masking; only the
/// training set is filtered. See [`MaskConfig`] for details.
///
/// `threads` is passed to [`crate::sa::build_gsa`]: `0` = auto (OpenMP default,
/// typically the CPU count), `1` = single-threaded, `N > 1` = exactly N threads.
///
/// ## 2× (forward + reverse-complement) suffix array
///
/// The SA is the generalized suffix array of the doubled text
/// `[Fwd || RC] + sentinel` in the `b+1` alphabet (see
/// [`crate::sa::build_doubled_2x_text`]), byte-identical in order to the FMI.
/// All `2 * l_pac + 1` entries are retained on disk — including the RC half and
/// the sentinel/empty-suffix row — so SA positions are *doubled coordinates* in
/// `[0, 2 * l_pac]`. The SHA-256 hash and `size_bytes` in `.meta` reflect the
/// original (forward-only) FASTA.
pub fn build_sidecar(
    ref_fa: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> Result<()> {
    build_sidecar_core(
        BuildSource::Fasta(ref_fa),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        None,
    )
}

/// Like [`build_sidecar`] but accepts an optional [`TrainerConfig`] for
/// non-default training parameters (e.g. a BED prior for target-aware fitting).
pub fn build_sidecar_with_config(
    ref_fa: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    build_sidecar_core(
        BuildSource::Fasta(ref_fa),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        trainer_config,
    )
}

/// Build a sidecar from bwa's forward `.pac` (byte-identical path).
///
/// The `.pac` carries bwa's N→random substitution, so the SA built here is
/// byte-identical in rank order to the bwa FM-index. `pac_sha256` in the
/// resulting `.meta` records the SHA-256 of the `.pac` file for provenance.
pub fn build_sidecar_from_pac(
    pac: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> Result<()> {
    build_sidecar_from_pac_with_config(pac, prefix, l2_leaf_count, mask, threads, None)
}

/// Build a sidecar from bwa's forward `.pac` (byte-identical path), with an
/// optional [`TrainerConfig`] (e.g. to select `--store-keys` / Mode2).
pub fn build_sidecar_from_pac_with_config(
    pac: &Path,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    build_sidecar_core(
        BuildSource::Pac(pac),
        prefix,
        l2_leaf_count,
        mask,
        threads,
        trainer_config,
    )
}

/// Core sidecar builder. All public entry points delegate here.
fn build_sidecar_core(
    source: BuildSource<'_>,
    prefix: &Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
    trainer_config: Option<TrainerConfig>,
) -> Result<()> {
    let config = trainer_config.unwrap_or_default();

    let (bases, n_positions, ref_path, ref_sha256, ref_size_bytes, pac_sha256) = match source {
        BuildSource::Fasta(p) => {
            log::warn!(
                "building sidecar from FASTA ({}): result is NOT byte-identical to a bwa \
                 FM-index (prmi maps N→A; bwa substitutes N→random at pack time). \
                 For byte-identical builds use --pac / BuildSource::Pac.",
                p.display()
            );
            let (bases, n_positions, _stats, sha, size) = fasta_to_2bit_with_sha256(p)?;
            (
                bases,
                Some(n_positions),
                p.display().to_string(),
                sha,
                size,
                None,
            )
        }
        BuildSource::Pac(p) => {
            let (bases, _l_pac) = crate::pac::read_bwa_pac_forward(p)?;
            let psha = crate::pac::pac_sha256(p)?;
            // bwa already substituted every N at pack time, so the `.pac` has no
            // N positions. Represent that with `None` instead of allocating (and
            // later O(N)-scanning) an all-false `vec![false; l_pac]` — a ~3.2 GB
            // allocation plus a full pass on a human-scale build.
            let ref_path = p.display().to_string();
            let ref_sha256 = psha.clone();
            let ref_size_bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            (
                bases,
                None,
                ref_path,
                ref_sha256,
                ref_size_bytes,
                Some(psha),
            )
        }
    };

    // Forward (unpadded) genome length = l_pac.
    let genome_len = bases.len() as u64;

    // Fail fast on a degenerate tiered keep-set BEFORE building the (expensive)
    // doubled GSA. A `--keep-bed` with no intervals, or whose every interval
    // starts at or after the forward genome length, can retain only the
    // sentinel row; it would otherwise build the entire 2× SA and then fail at
    // the empty-training-set guard far below. An interval `[s, e)` can retain a
    // forward coordinate iff `s < genome_len` (parse_bed guarantees `e > s`, so
    // a `0 < e` lower bound always holds). This is a user-input error.
    if let Some(keep) = mask.keep_bed.as_deref() {
        if !keep.iter().any(|iv| iv.start < genome_len) {
            return Err(Error::InvalidInput {
                detail: "--keep-bed contains no intervals overlapping the reference \
                         (every interval starts at or after the genome length); \
                         nothing to retain"
                    .to_string(),
            });
        }
    }

    // Build the 2× generalized suffix array over [Fwd||RC]+sentinel. No
    // T-padding and no filtering: every entry is retained, including the
    // sentinel/empty-suffix row, so SA order is byte-identical to the FMI.
    // `sa` and `text` are shared (Arc) so the streamed training set can borrow
    // them zero-copy; both are also needed later for the `.sa` write.
    let text = std::sync::Arc::new(crate::sa::build_doubled_2x_text(&bases));
    let sa = std::sync::Arc::new(crate::sa::build_gsa(&text, threads)?);

    // Tiered (Design Z) keep-mask. When set, the `.sa` retains only entries
    // whose forward reference coordinate lies in the keep-set, applied
    // RC-symmetrically over the doubled text; `genome_len` (the full forward
    // length) stays `l_pac` so doubled-coordinate decoding and native genome
    // positions are unchanged, while the entry COUNT shrinks. `keep_doubled_pos`
    // returns true for every entry when there is no keep-set, so the full build
    // is byte-identical.
    let keep = mask.keep_bed.as_deref();
    let keep_pos =
        |pos: u64| keep.is_none_or(|k| crate::train::mask::keep_doubled_pos(k, pos, genome_len));
    let num_sa_entries: u64 = match keep {
        None => sa.len() as u64,
        Some(_) => sa.iter().filter(|&&pos| keep_pos(pos)).count() as u64,
    };
    if keep.is_some() {
        log::info!(
            "tiered keep-mask: retaining {num_sa_entries} of {} SA entries ({:.3}%)",
            sa.len(),
            100.0 * num_sa_entries as f64 / sa.len() as f64
        );
    }

    // Resolve the optional l2_leaf_count: use provided value or auto-scale to
    // the number of RETAINED entries (so a tiered model is sized to its SA).
    let l2_leaf_count =
        l2_leaf_count.unwrap_or_else(|| default_l2_leaf_count(num_sa_entries as usize));

    // Dense/streamed fast path (the byte-identical `.pac` build): a uniform
    // prior with no homopolymer/BED mask and no effective N-mask. On this path
    // the keys are streamed from `sa`+`text` and the SA-index targets are
    // virtual, so neither the ~51.5 GB key vector, the ~51.5 GB target vector,
    // nor the `text_bases` / `n_positions_2x` arrays are materialised. The
    // resulting (key, sa_index) pairs — and the model — are byte-identical to
    // the materialized path.
    // `None` (the `.pac` path) carries no N positions, so the N-mask is a no-op
    // without any allocation or scan.
    let no_n_effect =
        !mask.mask_n_runs || n_positions.as_ref().is_none_or(|np| np.iter().all(|&b| !b));
    let virtualize = matches!(config.prior, Prior::Uniform)
        && mask.mask_homopolymers.is_none()
        && mask.mask_bed.is_none()
        && mask.keep_bed.is_none()
        && no_n_effect;
    let ts = if let Some(keep_intervals) = keep {
        // Tiered (Design Z): train the RMI to predict COMPACTED ranks
        // (`0..num_sa_entries`) over only the retained entries, matching the
        // filtered `.sa` write order. Keys are tokenised from the doubled text.
        // The keep filter (which entries the `.sa` retains) is ORTHOGONAL to the
        // mask/prior options (which retained entries are training targets and how
        // they are weighted): a tiered build composes with them exactly as the
        // materialized path does, so `--keep-bed` does not silently drop
        // `--mask-*`/`--prior-*`. Masked entries are still written to the `.sa`
        // (only `keep_pos` filters the write), so the compacted rank advances for
        // every retained entry regardless of masking.
        let text_bases: Vec<u8> = text
            .iter()
            .map(|&v| crate::sa::text_value_to_base(v))
            .collect();
        let n_positions_2x = doubled_n_bitmap(&n_positions, &mask, text.len());
        keep_masked_training_set(
            &sa,
            &text_bases,
            &n_positions_2x,
            keep_intervals,
            genome_len,
            &mask,
            &config.prior,
        )
    } else if virtualize {
        streamed_training_set(std::sync::Arc::clone(&sa), std::sync::Arc::clone(&text))
    } else {
        // Materialized path. The SA is in doubled coordinates over `text`
        // (length 2*l_pac+1), so the training set indexes bases/n_positions in
        // those same coordinates: the 0..=3 base array from `text` (1..=4,
        // sentinel 0 → T) and an N bitmap whose forward half carries the FASTA N
        // flags and whose RC half mirrors them (the RC of an ambiguous base is
        // itself ambiguous — those RC suffixes must see the same `mask_n_runs`
        // filtering as the forward half; the sentinel row stays non-N).
        let text_bases: Vec<u8> = text
            .iter()
            .map(|&v| crate::sa::text_value_to_base(v))
            .collect();
        // Doubled-text N bitmap (empty when no N-masking applies); see
        // `doubled_n_bitmap` for the materialization policy.
        let n_positions_2x = doubled_n_bitmap(&n_positions, &mask, text.len());
        masked_training_set(&sa, &text_bases, &n_positions_2x, &mask, &config.prior)
    };
    // Fail closed on an empty training set before model fitting. This is a
    // user-input failure — an out-of-reference `--keep-bed` (or an over-broad
    // `--mask-*`/empty reference) that selects no trainable 32-mer — not an
    // internal bug, so report it as `InvalidInput` rather than letting
    // `train_with_config` surface its `Internal` "empty training set" guard.
    if ts.is_empty() {
        return Err(Error::InvalidInput {
            detail: "training set is empty: --keep-bed/--mask-* selected no \
                     trainable position; relax them or use a reference with at \
                     least one 32-mer"
                .into(),
        });
    }
    let model = crate::train::trainer::train_with_config(&ts, l2_leaf_count, &config)?;
    // Streaming histogram: returns the percentile distribution and the max in
    // two passes, without materialising a ~51.5 GB per-key error vector. The
    // returned `max` equals the former `compute_max_error_bound`.
    let (p50, p90, p99, max_err) = compute_error_distribution(&model, &ts);
    log::info!("prmi model error bound: max={max_err} p50={p50} p90={p90} p99={p99}");

    let paths = SidecarPaths::from_prefix(prefix);

    // Remove any pre-existing `.kmt`/`.isa` so a rebuild (especially one WITHOUT
    // `--kmer-table-k` / `--with-isa`) can never inherit a stale sidecar for the
    // old reference. Fresh ones are written at the end if requested. Propagate
    // anything other than "already absent": a swallowed permission/I/O error
    // would silently leave the stale sidecar in place for the next open.
    for stale in [&paths.kmt, &paths.isa] {
        match std::fs::remove_file(stale) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::Io {
                    path: stale.clone(),
                    source: e,
                })
            }
        }
    }

    // .isa — optional inverse-suffix-array sidecar (the ISA launch hint). Built
    // directly from the SA permutation; independent of the `.sa` memory mode.
    if config.with_isa {
        match keep {
            // Full index: dense inverse SA, indexed by reference position.
            None => crate::sidecar::isa_file::write_isa_file(&paths.isa, &sa)?,
            // Tiered (Design Z): a dense refpos-indexed ISA would be genome-scale
            // even for a small keep-set. Instead emit a SPARSE ISA over exactly
            // the kept positions, mapping each to its COMPACTED `.sa` rank (the
            // rank space the tiered model predicts). The compacted rank is
            // assigned in the SAME lex-order filtered scan the `.sa` write below
            // uses, so the two agree; pairs are then sorted by refpos for the
            // reader's binary search. Present reads' seed positions are in the
            // keep-set by definition, so the launch-hint fast path applies; an
            // off-keep position simply misses and falls back to a cold launch.
            Some(_) => {
                let mut pairs: Vec<(u64, u64)> = Vec::with_capacity(num_sa_entries as usize);
                let mut compacted: u64 = 0;
                for &pos in sa.iter() {
                    if keep_pos(pos) {
                        pairs.push((pos, compacted));
                        compacted += 1;
                    }
                }
                pairs.sort_unstable_by_key(|&(refpos, _)| refpos);
                crate::sidecar::isa_file::write_tiered_isa_file(&paths.isa, &pairs)?;
            }
        }
    }

    // .sa — write the retained genome-region entries in the requested memory mode.
    let memory_mode = config.memory_mode;
    match memory_mode {
        MemoryMode::Mode1 => {
            // Position-only, 5 B/entry. Original layout; unchanged.
            let mut w = SaFileWriter::create(&paths.sa, num_sa_entries)?;
            for &pos in sa.iter().filter(|&&pos| keep_pos(pos)) {
                w.write_position(pos)?;
            }
            w.finish()?;
        }
        MemoryMode::Mode2 => {
            // Position + 32-mer key, 13 B/entry.
            let mut w = SaFileWriter::create_with_mode(&paths.sa, num_sa_entries, BPE_MODE2)?;
            for &pos in sa.iter().filter(|&&pos| keep_pos(pos)) {
                let key = key_for_position_2x(pos, &text);
                w.write_entry_with_key(pos, key)?;
            }
            w.finish()?;
        }
    }

    // .l1 / .l2
    ModelFileWriter::write(&paths.l1, ModelLayer::L1, &model.l1)?;
    ModelFileWriter::write(&paths.l2, ModelLayer::L2, &model.l2)?;

    // .meta
    let created_utc = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|e| Error::Internal {
            detail: format!("timestamp: {e}"),
        })?;
    let spec = format!("pwl{},linear,linear_spline", l2_leaf_count.trailing_zeros());
    let masked_bed_path: Option<String> = mask
        .mask_bed_path
        .as_deref()
        .map(|p| p.display().to_string());
    let keep_bed_path: Option<String> = mask
        .keep_bed_path
        .as_deref()
        .map(|p| p.display().to_string());

    // Determine the [priors] TOML section from the trainer config.
    let priors = priors_for_meta(&config.prior);

    // Mode-dependent meta fields.
    let (mode_str, skc_cache_size_meta) = match memory_mode {
        MemoryMode::Mode1 => ("1".to_string(), None),
        MemoryMode::Mode2 => ("2".to_string(), None),
    };

    let meta = Meta {
        prmi: Prmi {
            magic: META_MAGIC.into(),
            format_version: FORMAT_VERSION,
            trainer_version: format!("prmi={}", env!("CARGO_PKG_VERSION")),
            created_utc,
        },
        ref_: Ref {
            path: ref_path,
            sha256: ref_sha256,
            size_bytes: ref_size_bytes,
        },
        sa: Sa {
            num_entries: num_sa_entries,
            bytes_per_entry: memory_mode.bytes_per_entry(),
            encoding: memory_mode.encoding_name().to_string(),
            mode: mode_str,
            skc_cache_size: skc_cache_size_meta,
            strand: "forward_rc_2x".into(),
            masked_n_runs: mask.mask_n_runs,
            masked_homopolymers: mask.mask_homopolymers,
            masked_bed: masked_bed_path,
            l_pac: Some(genome_len),
            stored_keys: Some(memory_mode.bytes_per_entry() >= 13),
            tiered: keep.is_some().then_some(true),
            keep_bed: keep_bed_path,
            pac_sha256,
        },
        rmi: RmiSpec {
            spec,
            l2_leaf_count,
            bit_shift: model.bit_shift,
            max_error_bound: max_err,
            err_p50: Some(p50),
            err_p90: Some(p90),
            err_p99: Some(p99),
        },
        priors,
    };
    meta.write_file(&paths.meta)?;

    // ── optional: build and persist the forward k-mer table ───────────────
    if let Some(requested_k) = config.kmer_table_k {
        // Reject a zero order rather than coercing it to a 1-mer table — the
        // caller would silently get a different artifact than requested.
        if requested_k == 0 {
            return Err(Error::InvalidInput {
                detail: "kmer_table_k must be >= 1".into(),
            });
        }
        // Use the RETAINED entry count (matches the on-disk `.sa`/`.meta` under a
        // tiered keep-mask). The table is built by reopening the just-written,
        // position-filtered `.sa`, so binding `.kmt`'s `sa_num` to the full
        // `sa.len()` would leave `.kmt` and `.sa`/`.meta` with different SA
        // cardinalities — invalidating k-mer table bounds at read time. Equal to
        // `sa.len()` when there is no keep-set.
        let sa_num = num_sa_entries;
        // Cap k so 4^k does not dwarf the SA (avoids a huge mostly-empty table
        // on small references); the table is self-describing via its header.
        let k_max = ((sa_num as f64).log2() / 2.0).floor().clamp(1.0, 16.0) as u32;
        let k = requested_k.min(k_max);
        if k < requested_k {
            log::warn!(
                "k-mer table k capped to {k} (requested {requested_k}) for this reference size"
            );
        }

        // Reopen the just-written sidecar to drive the table search: it streams
        // the SA positions (and, in key-storing modes, the inline 32-mer keys)
        // from the `.sa` via the page cache, reusing the exact compare
        // `forward_spectrum` uses. We deliberately do NOT rebuild positions/keys
        // from the in-memory `sa` — that would either recompute every key per
        // probe (slower) or re-materialize the ~51 GB key vector the streamed
        // build avoids, defeating the memory budget.
        //
        // The reference bases, however, are taken from the in-memory `bases` we
        // already hold (decoded from the `.pac` for a Pac build, or the FASTA
        // 2-bit), instead of re-reading the entire `.pac` from disk. `bases` is
        // byte-identical to the decoded `.pac`, so the unpacked compare yields a
        // byte-identical table to a packed `.pac` compare — verified by
        // `kmer_table_packed_pac_equals_unpacked_bases`.
        let idx = LearnedIndex::open(prefix)?;
        let table = idx.build_kmer_table(k, &bases, PacEncoding::Unpacked);
        let (tk, tlo, thi) = table.parts();
        // Bind the table to its reference: prefer `pac_sha256` (the byte-identical
        // build), else the FASTA ref sha. A 0 digest (unparsable) just means the
        // open path falls back to the full search.
        let digest_hex = meta
            .sa
            .pac_sha256
            .as_deref()
            .unwrap_or(meta.ref_.sha256.as_str());
        let ref_digest = crate::sidecar::kmt_file::hex32(digest_hex).unwrap_or([0u8; 32]);
        KmtFileWriter::write(&paths.kmt, tk, sa_num, &ref_digest, tlo, thi)?;
        log::info!("prmi forward k-mer table: k={tk}, sa_num={sa_num}");
    }

    Ok(())
}

/// Build the doubled-text N bitmap (length `text_len`) for `mask_n_runs`: the
/// forward half carries the FASTA N flags and the RC half mirrors them (the RC
/// of an ambiguous base is itself ambiguous, so those RC suffixes see the same
/// `mask_n_runs` filtering; the sentinel row stays non-N).
///
/// Returns an EMPTY bitmap (which reads as "no N anywhere": `n_in_window` clamps
/// to its length and `any()` is false) when no N-masking will occur — i.e.
/// `mask_n_runs` is off or there are no FASTA N positions — avoiding the large
/// all-clear allocation (and its OOM risk on big references). Shared by the
/// materialized and tiered (keep-mask) training paths so both apply N-masking
/// identically.
fn doubled_n_bitmap(
    n_positions: &Option<Vec<bool>>,
    mask: &MaskConfig,
    text_len: usize,
) -> NBitmap {
    // Only materialize when at least one FASTA position is actually N: an
    // all-clear `Some(vec![false; ..])` would otherwise allocate (and scan) a
    // full doubled-length bitmap, contradicting the empty-bitmap fast path
    // documented above. Mirrors the `no_n_effect` all-clear check at L242.
    let materialise_n = mask.mask_n_runs
        && n_positions
            .as_ref()
            .is_some_and(|np| np.iter().any(|&is_n| is_n));
    let mut n_positions_2x = NBitmap::zeros(if materialise_n { text_len } else { 0 });
    if materialise_n {
        if let Some(np) = n_positions {
            let l_pac = np.len();
            for (i, &is_n) in np.iter().enumerate() {
                if is_n {
                    n_positions_2x.set(i);
                    // RC half: forward pos i → doubled coord l_pac+(l_pac-1-i).
                    n_positions_2x.set(l_pac + (l_pac - 1 - i));
                }
            }
        }
    }
    n_positions_2x
}

/// Construct the [`Priors`] meta struct from the trainer's [`Prior`].
fn priors_for_meta(prior: &Prior) -> Priors {
    match prior {
        Prior::Uniform => Priors {
            kind: "uniform".into(),
            bed: None,
            weight: None,
            histogram: None,
            base_weight: None,
            formula: None,
        },
        Prior::Bed { weight, path, .. } => Priors {
            kind: "bed".into(),
            bed: path.as_deref().map(|p| p.display().to_string()),
            weight: Some(*weight),
            histogram: None,
            base_weight: None,
            formula: None,
        },
        Prior::FastqHistogram {
            base_weight, path, ..
        } => Priors {
            kind: "fastq_histogram".into(),
            bed: None,
            weight: None,
            histogram: path.as_deref().map(|p| p.display().to_string()),
            base_weight: Some(*base_weight),
            formula: Some("1.0 + log2(1 + freq)".into()),
        },
    }
}

/// Resolve a `MaskConfig` from raw CLI values, parsing the BED file if
/// `bed_path` is `Some`. The source path is stored in `MaskConfig::mask_bed_path`
/// for provenance in the meta TOML.
///
/// `#[allow(dead_code)]` because the only caller is `cli.rs`, which lands in
/// PR #6 (`feat/v0.1-cli`).
#[allow(dead_code)]
pub(crate) fn mask_config_from_cli(
    no_mask_n_runs: bool,
    mask_homopolymers: Option<u32>,
    bed_path: Option<&Path>,
    keep_bed_path: Option<&Path>,
) -> Result<MaskConfig> {
    use crate::train::mask::parse_bed;
    let bed = bed_path.map(parse_bed).transpose()?;
    let keep = keep_bed_path.map(parse_bed).transpose()?;
    Ok(MaskConfig {
        mask_n_runs: !no_mask_n_runs,
        mask_homopolymers,
        mask_bed: bed,
        mask_bed_path: bed_path.map(std::path::PathBuf::from),
        keep_bed: keep,
        keep_bed_path: keep_bed_path.map(std::path::PathBuf::from),
    })
}

/// Resolve a [`Prior`] from raw CLI values.
///
/// Returns `Err` if `prior_bed_weight` is not positive.
pub fn prior_from_cli(prior_bed: Option<&Path>, prior_bed_weight: f64) -> Result<Prior> {
    use crate::train::mask::parse_bed;
    match prior_bed {
        None => Ok(Prior::Uniform),
        Some(path) => {
            if prior_bed_weight <= 0.0 {
                return Err(Error::InvalidInput {
                    detail: format!("--prior-bed-weight must be > 0, got {prior_bed_weight}"),
                });
            }
            let intervals = parse_bed(path)?;
            Ok(Prior::Bed {
                intervals,
                weight: prior_bed_weight,
                path: Some(path.to_path_buf()),
            })
        }
    }
}

/// Compute the 32-mer key for a doubled-coordinate SA position over the
/// `b+1`-alphabet 2× text. Text values are 1..=4 (+ 0 sentinel); shift back to
/// 0..=3 for tokenisation, mapping the sentinel `0` to a T (3) terminator (it
/// only appears once, at the very end, past any real 32-mer window).
#[inline]
fn key_for_position_2x(pos: u64, text: &[u8]) -> u64 {
    let start = pos as usize;
    let avail = text.len().saturating_sub(start).min(KMER_LEN);
    let mut window = [BASE_T; KMER_LEN];
    for (slot, &v) in window.iter_mut().zip(&text[start..start + avail]) {
        *slot = crate::sa::text_value_to_base(v);
    }
    tokenize_32mer(&window[..avail], avail)
}

/// Compute the 32-mer key for a forward-only SA position. Legacy helper from the v1 forward-only build path; retained pending removal in a later plan.
#[allow(dead_code)]
#[inline]
fn key_for_position(pos: u64, bases: &[u8]) -> u64 {
    let start = pos as usize;
    let n = bases.len();
    let avail = n.saturating_sub(start).min(KMER_LEN);
    tokenize_32mer(&bases[start..start + avail], avail)
}

/// Resolve a [`Prior::FastqHistogram`] from raw CLI values.
///
/// Returns `Err` if `base_weight` is not positive or if the histogram TSV
/// cannot be parsed.
pub fn prior_from_cli_fastq(histogram_path: &Path, base_weight: f64) -> Result<Prior> {
    if base_weight <= 0.0 {
        return Err(Error::InvalidInput {
            detail: format!("--prior-fastq-base-weight must be > 0, got {base_weight}"),
        });
    }
    let keys_to_freq = crate::train::prior::parse_histogram_tsv(histogram_path)?;
    Ok(Prior::FastqHistogram {
        keys_to_freq,
        base_weight,
        path: Some(histogram_path.to_path_buf()),
    })
}

/// Validate that the mutually-exclusive prior inputs are not both supplied.
///
/// `--prior-bed` and `--prior-fastq-histogram` select different weighting
/// schemes and cannot be combined. Returns `Err(Error::InvalidInput)` if both
/// paths are `Some`. This is the shared guard the CLI calls before resolving a
/// [`Prior`]; living here next to `prior_from_cli` / `prior_from_cli_fastq`
/// lets it be unit-tested without driving the binary.
pub fn validate_prior_paths(prior_bed: Option<&Path>, prior_fastq: Option<&Path>) -> Result<()> {
    if prior_bed.is_some() && prior_fastq.is_some() {
        return Err(Error::InvalidInput {
            detail: "--prior-bed and --prior-fastq-histogram are mutually exclusive; \
                     supply at most one"
                .to_string(),
        });
    }
    Ok(())
}
