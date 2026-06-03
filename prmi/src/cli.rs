// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Command-line interface for the `prmi` binary.

use anyhow::Context;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "prmi",
    version,
    about = "P-RMI trainer over genomic suffix arrays"
)]
/// Top-level CLI entry point; dispatches to a subcommand.
pub struct Cli {
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub cmd: Cmd,
}

/// Subcommands available in the `prmi` binary.
#[derive(Subcommand)]
pub enum Cmd {
    /// Build a prmi sidecar from a reference FASTA.
    Build {
        /// Path to the reference FASTA.
        ref_fa: PathBuf,
        /// Output prefix; produces `<prefix>.{meta,sa,l1,l2}`.
        /// Defaults to <ref_fa>.prmi.
        #[arg(short = 'o', long)]
        out: Option<PathBuf>,
        /// L2 leaf count (must be a power of two, ≥ 16).
        /// If omitted, auto-scaled to ~12 SA entries per leaf,
        /// clamped to [2^4, 2^28].
        #[arg(long)]
        l2_leaf_count: Option<u64>,
        /// Disable masking of N-runs. By default, training pairs whose
        /// 32-mer window covers any N base are excluded from the model fit.
        #[arg(long, default_value_t = false)]
        no_mask_n_runs: bool,
        /// Mask training pairs whose 32-mer window contains a run of the
        /// same base of length >= K. Typical value: 20.
        #[arg(long, value_name = "K")]
        mask_homopolymers: Option<u32>,
        /// Mask training pairs whose SA position falls in any interval of
        /// the given BED file (0-based, half-open).
        #[arg(long, value_name = "PATH")]
        mask_bed: Option<PathBuf>,
        /// Use a BED file as a training-weight prior. Training pairs whose SA
        /// position falls in any BED interval receive a weight of
        /// --prior-bed-weight (default 10.0); pairs outside receive weight 1.0.
        /// This is different from --mask-bed, which excludes pairs entirely.
        /// Mutually exclusive: --prior-bed and --mask-bed must not point at the
        /// same path. Also mutually exclusive with --prior-fastq-histogram.
        #[arg(long, value_name = "PATH")]
        prior_bed: Option<PathBuf>,
        /// Weight multiplier for in-BED pairs when --prior-bed is active.
        #[arg(long, value_name = "W", default_value_t = 10.0)]
        prior_bed_weight: f64,
        /// Use a pre-computed 32-mer frequency histogram (TSV: key_u64\tcount_u64)
        /// to weight training pairs by lookup likelihood. Training pairs whose
        /// 32-mer key appears in the histogram receive weight
        /// `--prior-fastq-base-weight + log2(1 + freq)`; pairs absent from the
        /// histogram receive --prior-fastq-base-weight. Mutually exclusive with
        /// --prior-bed.
        #[arg(long, value_name = "PATH")]
        prior_fastq_histogram: Option<PathBuf>,
        /// Base weight for k-mers absent from the FASTQ histogram
        /// (default 1.0). Must be > 0.
        #[arg(long, value_name = "W", default_value_t = 1.0)]
        prior_fastq_base_weight: f64,
        /// Number of threads for suffix-array construction.
        /// 0 = OpenMP default (honours `OMP_NUM_THREADS` if set, otherwise
        /// typically all available CPUs). Default: 0.
        #[arg(long, short = 't', default_value_t = 0)]
        threads: usize,
        /// Memory mode for the `.sa` sidecar file. Controls the per-entry layout
        /// and what extra data is stored alongside each SA position.
        ///
        /// - `1` (default): position only, 5 B/entry (~15 GB for human). Minimal memory.
        /// - `2`: position + stored 32-mer key, 13 B/entry (~39 GB). Skips per-candidate
        ///   pac reads in smem_range at the cost of 2.6x larger `.sa`.
        /// - `3`: mode 2 + stored ISA entry, 21 B/entry (~63 GB). Adds forward-extension
        ///   capability on top of mode 2.
        /// - `suffix-key-cache`: mode-1 `.sa` + companion `.skc` caching keys for the
        ///   top-N positions (see --suffix-key-cache-size). Lower memory than mode 2.
        #[arg(long, value_name = "MODE", default_value = "1")]
        memory_mode: String,
        /// For --memory-mode suffix-key-cache: number of SA positions to cache keys for.
        /// Caches the first N SA index entries (lexicographically smallest suffixes).
        /// Ignored for other memory modes. Default: 1000000.
        #[arg(long, value_name = "N", default_value_t = 1_000_000u64)]
        suffix_key_cache_size: u64,
    },
    /// Convert a KMC text-format dump to a prmi u64-key histogram TSV.
    ///
    /// KMC produces a text dump with lines like:
    ///
    /// ```text
    /// ACGTACGTACGTACGT...  count
    /// ```
    ///
    /// This subcommand reads that format (whitespace-separated 32-mer string +
    /// count) and writes a two-column TSV (key_u64\tcount_u64) to stdout,
    /// suitable for use with --prior-fastq-histogram.
    ///
    /// Only k-mers of exactly 32 bases are processed; lines with a different
    /// k-mer length are skipped (with a warning to stderr). Lines beginning
    /// with '#' and blank lines are skipped. N-containing k-mers are skipped.
    HistogramFromKmc {
        /// Path to the KMC text-format dump file (kmc_tools transform dump output).
        kmc_dump: PathBuf,
    },
    /// Print sidecar metadata.
    Info {
        /// Sidecar prefix (e.g. /data/hg38.fa.prmi).
        prefix: PathBuf,
    },
    /// Print diagnostic statistics about a built sidecar.
    Inspect {
        /// Sidecar prefix (without the .meta / .sa / .l1 / .l2 suffix).
        prefix: PathBuf,
    },
    /// Shared-memory operations: load a sidecar into a shm blob or remove it.
    Shm {
        #[command(subcommand)]
        #[allow(missing_docs)]
        cmd: ShmCmd,
    },
}

/// Subcommands for `prmi shm`.
#[derive(Subcommand)]
pub enum ShmCmd {
    /// Pack a sidecar into a shared-memory-backed blob file.
    ///
    /// Reads `<sidecar-prefix>.{meta,sa,l1,l2}` and writes a single combined
    /// blob to `<shm-path>`. An existing file at `<shm-path>` is overwritten.
    ///
    /// The typical destination on Linux is `/dev/shm/<name>` (tmpfs-backed
    /// shared memory). On macOS use `/tmp/<name>` instead (`/dev/shm` does
    /// not exist by default). Multiple aligner processes can then open the
    /// same blob via `prmi_open_shm` and share OS page-cache pages without
    /// each re-paying the I/O cost.
    ///
    /// After this command exits, `<shm-path>` persists until explicitly
    /// removed (see `prmi shm unload`) or the OS clears it on reboot.
    Load {
        /// Sidecar prefix (e.g. /data/hg38.fa.prmi).
        sidecar_prefix: PathBuf,
        /// Destination path for the shm blob (e.g. /dev/shm/hg38 on Linux,
        /// /tmp/hg38 on macOS).
        shm_path: PathBuf,
    },
    /// Remove a shm blob previously created by `prmi shm load`.
    ///
    /// Equivalent to `rm -f <shm-path>`. A missing file is not an error.
    Unload {
        /// Path to the shm blob to remove.
        shm_path: PathBuf,
    },
}

/// Parse the `--memory-mode` CLI string into a [`MemoryMode`].
fn parse_memory_mode(
    mode_str: &str,
    suffix_key_cache_size: u64,
) -> anyhow::Result<crate::train::config::MemoryMode> {
    use crate::train::config::MemoryMode;
    match mode_str {
        "1" => Ok(MemoryMode::Mode1),
        "2" => Ok(MemoryMode::Mode2),
        "3" => Ok(MemoryMode::Mode3),
        "suffix-key-cache" => Ok(MemoryMode::SuffixKeyCache {
            cache_size: suffix_key_cache_size,
        }),
        other => anyhow::bail!(
            "unrecognised --memory-mode {other:?}; \
             must be one of: 1, 2, 3, suffix-key-cache"
        ),
    }
}

impl Cli {
    /// Execute the selected subcommand and return its result.
    pub fn run(self) -> anyhow::Result<()> {
        match self.cmd {
            Cmd::Build {
                ref_fa,
                out,
                l2_leaf_count,
                no_mask_n_runs,
                mask_homopolymers,
                mask_bed,
                prior_bed,
                prior_bed_weight,
                prior_fastq_histogram,
                prior_fastq_base_weight,
                threads,
                memory_mode,
                suffix_key_cache_size,
            } => {
                let prefix = out.unwrap_or_else(|| {
                    let mut p = ref_fa.clone().into_os_string();
                    p.push(".prmi");
                    PathBuf::from(p)
                });

                // Guard: --prior-bed and --mask-bed must not point at the same path.
                if let (Some(ref pb), Some(ref mb)) = (&prior_bed, &mask_bed) {
                    if pb == mb {
                        anyhow::bail!(
                            "--prior-bed and --mask-bed must not point at the same BED file ({}); \
                             use one mechanism per BED: mask to exclude, prior to up-weight",
                            pb.display()
                        );
                    }
                }

                // Guard: --prior-bed and --prior-fastq-histogram are mutually
                // exclusive. The shared library helper is the single source of
                // the rule and now returns a clean `InvalidInput` error, so
                // surface it directly.
                crate::train::validate_prior_paths(
                    prior_bed.as_deref(),
                    prior_fastq_histogram.as_deref(),
                )?;

                let mask = crate::train::mask_config_from_cli(
                    no_mask_n_runs,
                    mask_homopolymers,
                    mask_bed.as_deref(),
                )
                .with_context(|| "parsing mask options")?;

                let prior = if let Some(ref hist_path) = prior_fastq_histogram {
                    crate::train::prior_from_cli_fastq(hist_path, prior_fastq_base_weight)
                        .with_context(|| "parsing --prior-fastq-histogram")?
                } else {
                    crate::train::prior_from_cli(prior_bed.as_deref(), prior_bed_weight)
                        .with_context(|| "parsing prior options")?
                };

                // Parse --memory-mode.
                let mem_mode = parse_memory_mode(&memory_mode, suffix_key_cache_size)
                    .with_context(|| format!("parsing --memory-mode {memory_mode:?}"))?;

                let trainer_config = crate::train::config::TrainerConfig {
                    prior,
                    memory_mode: mem_mode,
                    ..Default::default()
                };

                crate::train::build_sidecar_with_config(
                    &ref_fa,
                    &prefix,
                    l2_leaf_count,
                    mask,
                    threads,
                    Some(trainer_config),
                )
                .with_context(|| format!("building sidecar at {}", prefix.display()))?;
                println!("wrote sidecar prefix: {}", prefix.display());
                Ok(())
            }
            Cmd::HistogramFromKmc { kmc_dump } => {
                crate::histogram::kmc_dump_to_histogram_tsv(&kmc_dump)
                    .with_context(|| format!("converting KMC dump {}", kmc_dump.display()))?;
                Ok(())
            }
            Cmd::Info { prefix } => {
                let paths = crate::sidecar::SidecarPaths::from_prefix(&prefix);
                let meta = crate::sidecar::meta::Meta::read_file(&paths.meta)
                    .with_context(|| format!("reading {}", paths.meta.display()))?;
                println!("{}", meta.to_toml()?);
                Ok(())
            }
            Cmd::Inspect { prefix } => crate::inspect::inspect(&prefix)
                .with_context(|| format!("inspecting sidecar at {}", prefix.display())),
            Cmd::Shm { cmd } => match cmd {
                ShmCmd::Load {
                    sidecar_prefix,
                    shm_path,
                } => {
                    let t0 = Instant::now();
                    crate::index::shm::write_shm_blob(&sidecar_prefix, &shm_path).with_context(
                        || {
                            format!(
                                "writing shm blob for {} -> {}",
                                sidecar_prefix.display(),
                                shm_path.display()
                            )
                        },
                    )?;
                    println!(
                        "wrote shm blob: {} ({:.1} s)",
                        shm_path.display(),
                        t0.elapsed().as_secs_f64()
                    );
                    Ok(())
                }
                ShmCmd::Unload { shm_path } => {
                    crate::index::shm::unload_shm_blob(&shm_path)
                        .with_context(|| format!("removing shm blob {}", shm_path.display()))?;
                    println!("removed shm blob: {}", shm_path.display());
                    Ok(())
                }
            },
        }
    }
}
