// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! On-disk sidecar format: TOML meta + binary `.sa` / `.l1` / `.l2`, plus an
//! optional `.kmt` k-mer-table accelerator sidecar.

pub mod isa_file;
pub mod kmt_file;
pub mod magic;
pub mod meta;
pub mod model_file;
pub mod sa_file;

use std::path::{Path, PathBuf};

/// Resolve the sidecar file paths from a common prefix.
///
/// `prefix = "/data/hg38.fa.prmi"` →
///   `meta = "/data/hg38.fa.prmi.meta"`, `.sa`, `.l1`, `.l2`, and the optional
///   `.kmt`.
#[derive(Debug, Clone)]
pub struct SidecarPaths {
    /// Path to the `.meta` TOML file.
    pub meta: PathBuf,
    /// Path to the `.sa` packed suffix-array file.
    pub sa: PathBuf,
    /// Path to the `.l1` model file (L1 leaves).
    pub l1: PathBuf,
    /// Path to the `.l2` model file (L2 routing layer).
    pub l2: PathBuf,
    /// Path to the optional `.kmt` k-mer table file (forward-shallow accelerator).
    pub kmt: PathBuf,
    /// Path to the optional `.isa` inverse-suffix-array file (ISA launch hint).
    pub isa: PathBuf,
}

impl SidecarPaths {
    /// Build the sidecar paths by literal concatenation of suffixes onto
    /// `prefix`. This intentionally uses raw `OsStr` concatenation rather than
    /// [`Path::with_extension`] so that an existing extension on `prefix` (e.g.
    /// `.prmi` in `hg38.fa.prmi`) is preserved intact.
    pub fn from_prefix(prefix: &Path) -> Self {
        let p = prefix.to_path_buf();
        let with = |suffix: &str| {
            let mut s = p.as_os_str().to_owned();
            s.push(suffix);
            PathBuf::from(s)
        };
        Self {
            meta: with(".meta"),
            sa: with(".sa"),
            l1: with(".l1"),
            l2: with(".l2"),
            kmt: with(".kmt"),
            isa: with(".isa"),
        }
    }
}
