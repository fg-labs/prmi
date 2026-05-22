// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Runtime lookup machinery — implements the §4.4 lookup math the v0.1
//! sidecar format encodes. The `LearnedIndex` type owns mmap-backed readers
//! for the four sidecar files and answers `lookup(key) -> (pos, err)` queries.

/// §4.4 lookup math: parameterized over mmap-backed readers or in-memory slices.
pub mod lookup;
pub mod smem;

use crate::error::{Error, Result};
use crate::index::lookup::lookup_core;
use crate::sidecar::magic::META_MAGIC;
use crate::sidecar::meta::Meta;
use crate::sidecar::model_file::{ModelFileReader, ModelLayer};
use crate::sidecar::sa_file::SaFileReader;
use crate::sidecar::SidecarPaths;
use std::path::Path;

/// A loaded P-RMI sidecar: mmap-backed handle for the four sidecar files.
/// Read-only after `open`; safe for concurrent lookups across threads.
#[derive(Debug)]
pub struct LearnedIndex {
    meta: Meta,
    sa: SaFileReader,
    l1: ModelFileReader,
    l2: ModelFileReader,
}

const _ASSERT_SEND_SYNC: fn() = || {
    fn assert<T: Send + Sync>() {}
    assert::<LearnedIndex>();
};

impl LearnedIndex {
    /// Open a sidecar by prefix. Expects `<prefix>.{meta,sa,l1,l2}` to exist.
    /// Cross-validates the headers against each other before returning.
    pub fn open(prefix: &Path) -> Result<Self> {
        let paths = SidecarPaths::from_prefix(prefix);
        let meta = Meta::read_file(&paths.meta)?;
        let sa = SaFileReader::open(&paths.sa)?;
        let l1 = ModelFileReader::open(&paths.l1, ModelLayer::L1)?;
        let l2 = ModelFileReader::open(&paths.l2, ModelLayer::L2)?;
        cross_validate(&paths, &meta, &sa, &l1, &l2)?;
        Ok(Self { meta, sa, l1, l2 })
    }

    /// Number of entries in the suffix array.
    pub fn sa_num(&self) -> u64 {
        self.sa.num_entries()
    }

    /// SA position stored at index `i`. `i` must be less than [`sa_num`](Self::sa_num).
    pub fn sa_position_for(&self, i: u64) -> u64 {
        self.sa.position(i)
    }

    /// Global maximum prediction error bound recorded in `.meta`.
    pub fn max_error_bound(&self) -> u64 {
        self.meta.rmi.max_error_bound
    }

    /// `bit_shift = 64 - log2(l2_leaf_count)`. Used to compute the L2 index.
    pub fn bit_shift(&self) -> u32 {
        self.meta.rmi.bit_shift
    }

    /// Format-version string (always `"PRMIv1"` for v0.1).
    pub fn format_version(&self) -> &str {
        META_MAGIC
    }

    /// Brief §4.4 lookup: predict the SA index for a 32-mer `key`. Returns
    /// `(predicted_sa_pos, err)`.
    #[inline]
    pub fn lookup(&self, key: u64) -> (u64, u64) {
        lookup_core(key, &self.l1, &self.l2, self.bit_shift(), self.sa_num())
    }

    // Accessors used by Task 22+ (smem_range) and the C FFI layer.
    pub(crate) fn sa(&self) -> &SaFileReader {
        &self.sa
    }
    #[allow(dead_code)]
    pub(crate) fn l1(&self) -> &ModelFileReader {
        &self.l1
    }
    #[allow(dead_code)]
    pub(crate) fn l2(&self) -> &ModelFileReader {
        &self.l2
    }
    #[allow(dead_code)]
    pub(crate) fn meta(&self) -> &Meta {
        &self.meta
    }
}

fn cross_validate(
    paths: &SidecarPaths,
    meta: &Meta,
    sa: &SaFileReader,
    _l1: &ModelFileReader,
    l2: &ModelFileReader,
) -> Result<()> {
    if meta.sa.num_entries != sa.num_entries() {
        return Err(Error::SidecarMismatch {
            file: paths.meta.clone(),
            detail: format!(
                ".meta sa.num_entries={} but .sa header says {}",
                meta.sa.num_entries,
                sa.num_entries()
            ),
        });
    }
    if meta.rmi.l2_leaf_count as usize != l2.len() {
        return Err(Error::SidecarMismatch {
            file: paths.meta.clone(),
            detail: format!(
                ".meta rmi.l2_leaf_count={} but .l2 has {} entries",
                meta.rmi.l2_leaf_count,
                l2.len()
            ),
        });
    }
    Ok(())
}
