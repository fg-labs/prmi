// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! On-disk sidecar format: TOML meta + binary `.sa` / `.l1` / `.l2`.

pub mod magic;
pub mod meta;

use std::path::{Path, PathBuf};

/// Resolve the four sidecar file paths from a common prefix.
///
/// `prefix = "/data/hg38.fa.prmi"` →
///   `meta = "/data/hg38.fa.prmi.meta"`, `.sa`, `.l1`, `.l2`.
#[derive(Debug, Clone)]
pub struct SidecarPaths {
    pub meta: PathBuf,
    pub sa: PathBuf,
    pub l1: PathBuf,
    pub l2: PathBuf,
}

impl SidecarPaths {
    /// Build the four sidecar paths by literal concatenation of suffixes onto
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
        }
    }
}
