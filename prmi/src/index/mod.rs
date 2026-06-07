// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Runtime lookup machinery — implements the §4.4 lookup math the v0.1
//! sidecar format encodes. The `LearnedIndex` type owns mmap-backed readers
//! for the four sidecar files and answers `lookup(key) -> (pos, err)` queries.

/// §4.4 lookup math: parameterized over mmap-backed readers or in-memory slices.
pub mod lookup;
pub mod shm;
pub mod smem;
pub mod smem_simd;
pub mod spectrum;

use crate::error::{Error, Result};
use crate::index::lookup::lookup_core;
use crate::index::shm::{read_shm_blob, write_shm_blob};
use crate::sidecar::isa_file::IsaFileReader;
use crate::sidecar::magic::META_MAGIC;
use crate::sidecar::meta::Meta;
use crate::sidecar::model_file::{ModelFileReader, ModelLayer};
use crate::sidecar::sa_file::SaFileReader;
use crate::sidecar::SidecarPaths;
use std::path::Path;

/// A loaded P-RMI sidecar: mmap-backed handle for the sidecar files.
/// Read-only after `open`; safe for concurrent lookups across threads.
#[derive(Debug)]
pub struct LearnedIndex {
    meta: Meta,
    sa: SaFileReader,
    l1: ModelFileReader,
    l2: ModelFileReader,
    /// Optional inverse SA (ref2sa), present when a `.isa` file exists beside the sidecar.
    isa: Option<IsaFileReader>,
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
        // An `.isa` is optional, but if one is present beside the sidecar it must
        // describe the SAME suffix array — a stale/mismatched `.isa` would make
        // `isa_for_refpos()` return garbage (or panic) during backward extension.
        // Validate the entry count against the `.sa` here, and only treat a true
        // `NotFound` as "no `.isa`" rather than swallowing every open failure.
        let isa = match IsaFileReader::open(&paths.isa) {
            Ok(reader) => {
                if reader.num_entries() != sa.num_entries() {
                    return Err(Error::SidecarMismatch {
                        file: paths.isa.clone(),
                        detail: format!(
                            ".isa num_entries={} but .sa has {}",
                            reader.num_entries(),
                            sa.num_entries()
                        ),
                    });
                }
                Some(reader)
            }
            Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        Ok(Self {
            meta,
            sa,
            l1,
            l2,
            isa,
        })
    }

    /// Open a sidecar previously loaded into a shm blob by `prmi shm load`.
    ///
    /// `shm_path` is the path to the shm blob file (typically
    /// `/dev/shm/<name>` on Linux or `/tmp/<name>` on macOS). The blob must
    /// have been written by [`write_shm_blob`] (or `prmi shm load`).
    ///
    /// Pages are mmap'd with `MAP_SHARED`; multiple processes that open the
    /// same `shm_path` share the same OS page-cache pages without re-paying
    /// I/O or page-fault costs after the first open. Cross-process sharing
    /// relies on the OS honouring `MAP_SHARED` for the backing store (tmpfs on
    /// Linux, APFS on macOS); this is standard behaviour for any regular file
    /// or `/dev/shm` entry on both platforms.
    ///
    /// Thread-safe after return: `LearnedIndex` is `Send + Sync`.
    ///
    /// # Errors
    ///
    /// Returns an error if the blob file is absent, truncated, has an
    /// unrecognised wrapper header, or contains a component that fails
    /// sidecar validation.
    ///
    /// # Limitations
    ///
    /// - Concurrent writers are not supported. If `prmi shm load` is still
    ///   running when `open_shm` is called, behavior is undefined.
    /// - Crash safety is not provided: a partially written blob produces an
    ///   error rather than silently corrupt data (the component headers are
    ///   validated before any lookup).
    /// - Backward extension ([`isa_for_refpos`](Self::isa_for_refpos)) requires
    ///   a file-backed [`open`](Self::open); the shm blob does not yet carry the
    ///   `.isa` (added in a later milestone).
    pub fn open_shm(shm_path: impl AsRef<Path>) -> Result<Self> {
        let shm_path = shm_path.as_ref();
        let blob = read_shm_blob(shm_path)?;

        // Parse the meta TOML from the blob's meta component.
        let meta_slice = &blob.mmap[blob.meta_offset..blob.meta_offset + blob.meta_len];
        let meta_str = std::str::from_utf8(meta_slice).map_err(|_| Error::Internal {
            detail: "shm blob meta component is not valid UTF-8".to_string(),
        })?;
        let meta = Meta::from_toml_str(meta_str)?;

        // SHM blobs pack only the four core components; the `.skc` companion is
        // not included. Fail fast rather than silently loading a
        // suffix_key_cache sidecar with `skc = None`, which would change
        // `key_at()` semantics relative to a file-backed `open`.
        if meta.sa.mode == "suffix_key_cache" {
            return Err(Error::SidecarMismatch {
                file: shm_path.to_path_buf(),
                detail: "suffix_key_cache sidecars are not supported in SHM blobs".to_string(),
            });
        }

        let sa = SaFileReader::from_shm_slice(blob.mmap.clone(), blob.sa_offset, blob.sa_len)?;
        let l1 = ModelFileReader::from_shm_slice(
            blob.mmap.clone(),
            blob.l1_offset,
            blob.l1_len,
            ModelLayer::L1,
        )?;
        let l2 = ModelFileReader::from_shm_slice(
            blob.mmap.clone(),
            blob.l2_offset,
            blob.l2_len,
            ModelLayer::L2,
        )?;

        // Re-use cross_validate with a synthetic path for error messages.
        let fake_paths = SidecarPaths::from_prefix(shm_path);
        cross_validate(&fake_paths, &meta, &sa, &l1, &l2)?;
        // The .isa is not packed into the shm blob yet; backward extension
        // requires a file-backed open.
        let isa = None;
        Ok(Self {
            meta,
            sa,
            l1,
            l2,
            isa,
        })
    }

    /// Pack this sidecar's four component files into a single shm blob at
    /// `shm_path`. Convenience wrapper around [`write_shm_blob`].
    ///
    /// Equivalent to running `prmi shm load <prefix> <shm_path>` from the CLI.
    pub fn write_shm(sidecar_prefix: &Path, shm_path: &Path) -> Result<()> {
        write_shm_blob(sidecar_prefix, shm_path)
    }

    /// Number of entries in the suffix array.
    pub fn sa_num(&self) -> u64 {
        self.sa.num_entries()
    }

    /// Number of forward reference bases (`l_pac`). The 2× SA has `2*l_pac + 1`
    /// entries, so `l_pac = (sa_num - 1) / 2`. (Equals `meta.sa.l_pac` when set.)
    pub fn l_pac(&self) -> u64 {
        self.meta
            .sa
            .l_pac
            .unwrap_or_else(|| (self.sa_num().saturating_sub(1)) / 2)
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

    /// Number of L2 leaf models. Equals `l2_leaf_count` from the `.meta` file.
    pub fn l2_leaf_count(&self) -> u64 {
        self.meta.rmi.l2_leaf_count
    }

    /// Returns the sidecar format magic string (the value of `META_MAGIC`,
    /// currently `"PRMIv2"`). This reflects the format version written by the
    /// trainer and is not guaranteed to be a fixed constant across releases.
    pub fn format_version(&self) -> &str {
        META_MAGIC
    }

    /// Read `out.len()` packed SA positions starting at SA index `k` from
    /// the mmap'd `.sa` file into the caller-provided `out` slice.
    ///
    /// Each output is a raw doubled-coordinate SA position in `[0, 2*l_pac]`.
    /// A position `p < l_pac` is a forward-strand hit at reference position
    /// `p`; a position `p >= l_pac` is a reverse-strand hit. The consumer
    /// (e.g. bwa-mem3) applies the `pos >= l_pac` strand demux via its own
    /// `bns` structure — prmi does not perform strand demultiplexing.
    ///
    /// Returns `Err` if `k + out.len() > sa_num`. Returns `Ok(())` with
    /// no writes if `out.len() == 0`.
    ///
    /// No allocation; the caller provides the buffer. Designed for hot
    /// per-pivot loops in downstream aligners.
    ///
    /// Thread-safe: handle is read-only after `LearnedIndex::open`.
    pub fn sa_positions(&self, k: u64, out: &mut [u64]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let sa_num = self.sa_num();
        let end = k
            .checked_add(out.len() as u64)
            .ok_or_else(|| Error::Internal {
                detail: format!("sa_positions: k={k} + count={} overflows u64", out.len()),
            })?;
        if end > sa_num {
            return Err(Error::Internal {
                detail: format!("sa_positions: range [{k}, {end}) exceeds sa_num={sa_num}"),
            });
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.sa_position_for(k + i as u64);
        }
        Ok(())
    }

    /// Read `out.len()` SA positions starting at index `k`, sampling every
    /// `step`-th entry: `out[j] = SA[k + j*step]`. With `step == 1` this is
    /// `sa_positions`. The stride/subsampling policy (choosing `step`) is the
    /// caller's. Returns `Err` if the last sampled index `k + (out.len()-1)*step`
    /// is `>= sa_num`.
    pub fn sa_positions_strided(&self, k: u64, step: u64, out: &mut [u64]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if step == 0 {
            // A zero stride would fill `out` with the same `SA[k]` repeatedly,
            // silently violating the documented "every `step`-th entry" contract
            // and hiding caller bugs.
            return Err(Error::Internal {
                detail: "sa_positions_strided: step must be > 0".into(),
            });
        }
        let last = k
            .checked_add((out.len() as u64 - 1).checked_mul(step).ok_or_else(|| {
                Error::Internal {
                    detail: "sa_positions_strided: index overflow".into(),
                }
            })?)
            .ok_or_else(|| Error::Internal {
                detail: "sa_positions_strided: index overflow".into(),
            })?;
        if last >= self.sa_num() {
            return Err(Error::Internal {
                detail: format!(
                    "sa_positions_strided: last index {last} >= sa_num {}",
                    self.sa_num()
                ),
            });
        }
        for (j, slot) in out.iter_mut().enumerate() {
            *slot = self.sa_position_for(k + j as u64 * step);
        }
        Ok(())
    }

    /// Brief §4.4 lookup: predict the SA index for a 32-mer `key`. Returns
    /// `(predicted_sa_pos, err)`.
    #[inline]
    pub fn lookup(&self, key: u64) -> (u64, u64) {
        lookup_core(key, &self.l1, &self.l2, self.bit_shift(), self.sa_num())
    }

    // Accessors used by Task 22+ (smem_range), the C FFI layer, and inspect.
    pub(crate) fn sa(&self) -> &SaFileReader {
        &self.sa
    }
    /// L1 fallback layer reader. Used by inspect and smem_range.
    pub fn l1(&self) -> &ModelFileReader {
        &self.l1
    }
    /// L2 routing layer reader. Used by inspect.
    pub fn l2(&self) -> &ModelFileReader {
        &self.l2
    }
    #[allow(dead_code)]
    pub(crate) fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Memory mode string from `.meta [sa] mode` (e.g. `"1"` or `"2"`).
    pub fn memory_mode(&self) -> &str {
        &self.meta.sa.mode
    }

    /// Return the stored 32-mer key at SA index `i`, if available.
    ///
    /// Returns `Some(key)` for mode 2 (where the key is stored in the `.sa`
    /// file alongside the position). Returns `None` for mode 1.
    #[inline]
    pub fn key_at(&self, i: u64) -> Option<u64> {
        self.sa.key_at(i)
    }

    /// Return the stored ISA value at SA index `i`, if available.
    ///
    /// Always `None` for the current memory modes (1 and 2), which do not
    /// store an inline ISA column in the `.sa` file.
    #[inline]
    pub fn isa_at(&self, i: u64) -> Option<u64> {
        self.sa.isa_at(i)
    }

    /// SA index whose suffix starts at reference (doubled-coordinate) position
    /// `p`. `None` if this sidecar has no `.isa` (e.g. shm-opened). Used by
    /// backward extension (Plan 3).
    pub fn isa_for_refpos(&self, p: u64) -> Option<u64> {
        self.isa.as_ref().and_then(|r| {
            // `sa_index_for_refpos` asserts on `p >= num_entries`; bound-check
            // here so a stray caller input yields `None` instead of panicking.
            (p < r.num_entries()).then(|| r.sa_index_for_refpos(p))
        })
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
    // Guard: l2_leaf_count must be a power of two >= 2 (the trainer enforces
    // this; a tampered .meta could violate it and cause trailing_zeros() to
    // return 0 for l2_leaf_count=1, making bit_shift=64 which the lookup path
    // handles, but l2_leaf_count=0 would underflow).
    if !meta.rmi.l2_leaf_count.is_power_of_two() || meta.rmi.l2_leaf_count < 2 {
        return Err(Error::SidecarMismatch {
            file: paths.meta.clone(),
            detail: format!(
                "l2_leaf_count={} must be a power of two ≥ 2",
                meta.rmi.l2_leaf_count
            ),
        });
    }
    // Guard: bit_shift must be consistent with l2_leaf_count to prevent OOB
    // indexing in the lookup path. If bit_shift is wrong, `key >> bit_shift`
    // can produce an l2_idx up to 2^(64 - bit_shift) - 1, which may exceed
    // the actual .l2 file length and cause an mmap slice panic.
    let expected_bit_shift = 64u32 - meta.rmi.l2_leaf_count.trailing_zeros();
    if meta.rmi.bit_shift != expected_bit_shift {
        return Err(Error::SidecarMismatch {
            file: paths.meta.clone(),
            detail: format!(
                "bit_shift={} inconsistent with l2_leaf_count={} (expected bit_shift={})",
                meta.rmi.bit_shift, meta.rmi.l2_leaf_count, expected_bit_shift,
            ),
        });
    }
    Ok(())
}
