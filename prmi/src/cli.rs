// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Command-line interface for the `prmi` binary.

use anyhow::Context;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
        #[arg(long, default_value_t = 262144)]
        l2_leaf_count: u64,
    },
    /// Print sidecar metadata.
    Info {
        /// Sidecar prefix (e.g. /data/hg38.fa.prmi).
        prefix: PathBuf,
    },
}

impl Cli {
    /// Execute the selected subcommand and return its result.
    pub fn run(self) -> anyhow::Result<()> {
        match self.cmd {
            Cmd::Build {
                ref_fa,
                out,
                l2_leaf_count,
            } => {
                let prefix = out.unwrap_or_else(|| {
                    let mut p = ref_fa.clone().into_os_string();
                    p.push(".prmi");
                    PathBuf::from(p)
                });
                crate::train::build_sidecar(&ref_fa, &prefix, l2_leaf_count)
                    .with_context(|| format!("building sidecar at {}", prefix.display()))?;
                println!("wrote sidecar prefix: {}", prefix.display());
                Ok(())
            }
            Cmd::Info { prefix } => {
                let paths = crate::sidecar::SidecarPaths::from_prefix(&prefix);
                let meta = crate::sidecar::meta::Meta::read_file(&paths.meta)
                    .with_context(|| format!("reading {}", paths.meta.display()))?;
                println!("{}", meta.to_toml()?);
                Ok(())
            }
        }
    }
}
