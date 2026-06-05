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
    /// Build a prmi sidecar from a reference FASTA or a bwa `.pac`.
    ///
    /// Exactly one reference source must be given: either the positional
    /// reference FASTA (`--ref`, NOT byte-identical to a bwa FM-index) or
    /// `--pac` (byte-identical). The two are mutually exclusive.
    Build {
        /// Path to the reference FASTA (positional). NOT byte-identical to a
        /// bwa FM-index (prmi maps N→A). Mutually exclusive with `--pac`.
        #[arg(conflicts_with = "pac")]
        ref_fa: Option<PathBuf>,
        /// Path to bwa's forward `.pac`. Produces a sidecar byte-identical in
        /// rank order to the bwa FM-index. Mutually exclusive with the
        /// reference FASTA.
        #[arg(long)]
        pac: Option<PathBuf>,
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
        /// Store a 32-mer key alongside each SA position (13 B/entry vs 5 B).
        /// Speeds the compare loop by skipping re-tokenisation, at ~50 GB extra
        /// for hg38. A pure speed A/B knob; correctness is identical either way.
        ///
        /// Defaults to on. Pass `--store-keys` (bare) to keep it on, or
        /// `--store-keys=false` to select the position-only layout (mode 1). A
        /// bare boolean flag cannot be negated under clap, so the explicit
        /// `=<bool>` form is required to reach mode 1 from the CLI.
        #[arg(
            long,
            default_value_t = true,
            action = clap::ArgAction::Set,
            num_args = 0..=1,
            require_equals = true,
            default_missing_value = "true",
        )]
        store_keys: bool,
        /// Build and persist a `.kmt` k-mer table of order K (forward-spectrum
        /// shallow-band accelerator; K capped to the reference size). Recommended
        /// K=12 (~358 MB) for human-scale references. Omit to build no table.
        #[arg(long, value_name = "K")]
        kmer_table_k: Option<u32>,
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
    /// Certify that the 2× SA built from a reference equals the unique
    /// lexicographic suffix ordering (byte-identity foundation gate).
    SaVerify {
        /// Reference FASTA.
        #[arg(long)]
        r#ref: std::path::PathBuf,
        /// Run the exhaustive O(N²) oracle (small references only).
        #[arg(long, default_value_t = false)]
        exhaustive: bool,
        /// Threads for SA construction.
        #[arg(long, default_value_t = 1)]
        threads: usize,
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

impl Cli {
    /// Execute the selected subcommand and return its result.
    pub fn run(self) -> anyhow::Result<()> {
        match self.cmd {
            Cmd::Build {
                ref_fa,
                pac,
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
                store_keys,
                kmer_table_k,
            } => {
                // Resolve the reference source: exactly one of the positional
                // FASTA or `--pac`. clap's `conflicts_with` rejects both; here
                // we reject neither.
                let source_path = match (ref_fa.as_ref(), pac.as_ref()) {
                    (Some(_), Some(_)) => {
                        anyhow::bail!("--pac and the reference FASTA are mutually exclusive")
                    }
                    (None, None) => anyhow::bail!(
                        "no reference source given: provide a reference FASTA (positional) or --pac"
                    ),
                    (Some(fa), None) => fa.clone(),
                    (None, Some(p)) => p.clone(),
                };

                let prefix = out.unwrap_or_else(|| {
                    let mut p = source_path.clone().into_os_string();
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

                let mem_mode = if store_keys {
                    crate::train::config::MemoryMode::Mode2
                } else {
                    crate::train::config::MemoryMode::Mode1
                };
                let trainer_config = crate::train::config::TrainerConfig {
                    prior,
                    memory_mode: mem_mode,
                    kmer_table_k,
                    ..Default::default()
                };

                if let Some(pac_path) = pac.as_ref() {
                    // Byte-identical build straight from bwa's forward `.pac`.
                    crate::train::build_sidecar_from_pac_with_config(
                        pac_path,
                        &prefix,
                        l2_leaf_count,
                        mask,
                        threads,
                        Some(trainer_config),
                    )
                    .with_context(|| format!("building sidecar at {}", prefix.display()))?;
                } else {
                    // FASTA build (warns: NOT byte-identical to a bwa FM-index).
                    let fa = ref_fa
                        .as_ref()
                        .expect("source resolution guaranteed a FASTA");
                    crate::train::build_sidecar_with_config(
                        fa,
                        &prefix,
                        l2_leaf_count,
                        mask,
                        threads,
                        Some(trainer_config),
                    )
                    .with_context(|| format!("building sidecar at {}", prefix.display()))?;
                }
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
            Cmd::SaVerify {
                r#ref,
                exhaustive,
                threads,
            } => {
                anyhow::ensure!(
                    exhaustive,
                    "non-exhaustive sampling mode not yet implemented; pass --exhaustive for small references"
                );
                let n = crate::verify_sa::sa_verify_fasta(&r#ref, threads)?;
                println!("OK: {n} SA entries certified against the lexicographic oracle");
                Ok(())
            }
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
