// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.l1` / `.l2` file format. Both layers share the same 16-byte header +
//! `leaf_count × 24` entries layout; only the magic differs.

#![allow(unsafe_code)]

use crate::error::{Error, Result};
use crate::sidecar::magic::{FORMAT_VERSION, L1_MAGIC, L2_MAGIC};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Size of the binary header at the start of each model file, in bytes.
pub const MODEL_FILE_HEADER_BYTES: usize = 16;
/// Size of one serialised `ModelEntry` on disk, in bytes.
pub const BYTES_PER_MODEL_ENTRY: usize = 24;

/// One model leaf entry: a linear predictor `(alpha, beta)` plus an `err`
/// bound. The `err` interpretation differs between L1 and L2 — see brief
/// §4.3-4.4. For L2 leaves on the fallback path, `err`'s high bit is set
/// and the rest encodes (partial_start, partial_num) into the L1 array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelEntry {
    /// Intercept of the linear predictor.
    pub alpha: f64,
    /// Slope of the linear predictor.
    pub beta: f64,
    /// Error bound (or packed L1 pointer for L2 entries on the fallback path).
    pub err: u64,
}

/// Which sidecar layer a file represents — controls the magic bytes used
/// for I/O and reader-side validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLayer {
    /// L1 leaf layer (the large per-key linear predictors).
    L1,
    /// L2 routing layer (one entry per power-of-two bucket).
    L2,
}

impl ModelLayer {
    fn magic(self) -> u32 {
        match self {
            ModelLayer::L1 => L1_MAGIC,
            ModelLayer::L2 => L2_MAGIC,
        }
    }
}

/// Writes a model file (`.l1` or `.l2`) in one pass given a slice of
/// entries. The whole layer must fit in memory at write time — this is
/// fine for v0.1 since L1 caps at ~2^31 entries (~50 GB worst case).
pub struct ModelFileWriter;

impl ModelFileWriter {
    /// Write all `entries` for the given `layer` to `path`, overwriting any existing file.
    pub fn write(path: &Path, layer: ModelLayer, entries: &[ModelEntry]) -> Result<()> {
        let f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| Error::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
        let mut w = BufWriter::new(f);
        let mut header = [0u8; MODEL_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], layer.magic());
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], entries.len() as u64);
        w.write_all(&header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut buf = [0u8; BYTES_PER_MODEL_ENTRY];
        for e in entries {
            LittleEndian::write_f64(&mut buf[0..8], e.alpha);
            LittleEndian::write_f64(&mut buf[8..16], e.beta);
            LittleEndian::write_u64(&mut buf[16..24], e.err);
            w.write_all(&buf).map_err(|e| Error::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
        }
        w.flush().map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }
}

/// mmap-backed reader for a model file.
pub struct ModelFileReader {
    /// Keeps the file open for file-backed instances; `None` for shm-backed.
    _file: Option<File>,
    /// Owned mmap for file-backed instances; `None` for shm-backed.
    /// Prefixed with `_` because data is accessed via `data_ptr`; field exists
    /// solely to extend the Mmap's lifetime.
    _mmap: Option<Mmap>,
    /// Shared shm blob mmap for shm-backed instances; `None` for file-backed.
    _shm_mmap: Option<Arc<Mmap>>,
    /// Pointer to the model data bytes (from either `mmap` or a sub-slice of
    /// `_shm_mmap`). Valid for the lifetime of this struct.
    data_ptr: *const u8,
    /// Total length (in bytes) of the model data region.
    data_len: usize,
    len: usize,
}

// SAFETY: same rationale as SaFileReader — `data_ptr` is read-only and
// kept alive by a field in this struct.
unsafe impl Send for ModelFileReader {}
unsafe impl Sync for ModelFileReader {}

impl std::fmt::Debug for ModelFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelFileReader")
            .field("len", &self.len)
            .field("data_len", &self.data_len)
            .finish()
    }
}

impl ModelFileReader {
    /// Open and mmap a model file, validating its header against `expected_layer`.
    pub fn open(path: &Path, expected_layer: ModelLayer) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: read-only mmap; the backing File is kept alive in `_file`;
        // we don't support concurrent writers.
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let leaf_count = validate_model_header(&mmap, path, mmap.len(), expected_layer)?;
        let data_ptr = mmap.as_ptr();
        let data_len = mmap.len();
        Ok(Self {
            _file: Some(f),
            _mmap: Some(mmap),
            _shm_mmap: None,
            data_ptr,
            data_len,
            len: leaf_count,
        })
    }

    /// Construct a `ModelFileReader` backed by a sub-slice of a shm blob mmap.
    ///
    /// `shm_mmap` is the `Arc<Mmap>` of the parent shm blob. `offset` and `len`
    /// identify the component bytes within it. The component must start with a
    /// valid model file header for `expected_layer`.
    pub(crate) fn from_shm_slice(
        shm_mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
        expected_layer: ModelLayer,
    ) -> Result<Self> {
        let fake_path = PathBuf::from("<shm>");
        // Bounds-check the requested sub-range before slicing: a corrupted or
        // invalid component offset/len must return an error, not panic.
        let end = offset.checked_add(len).ok_or_else(|| Error::SizeMismatch {
            file: fake_path.clone(),
            detail: format!("shm range overflow: offset={offset}, len={len}"),
        })?;
        let slice = shm_mmap
            .get(offset..end)
            .ok_or_else(|| Error::SizeMismatch {
                file: fake_path.clone(),
                detail: format!(
                    "shm range out of bounds: offset={offset}, len={len}, shm_len={}",
                    shm_mmap.len()
                ),
            })?;
        let leaf_count = validate_model_header(slice, &fake_path, len, expected_layer)?;
        let data_ptr = slice.as_ptr();
        Ok(Self {
            _file: None,
            _mmap: None,
            _shm_mmap: Some(shm_mmap),
            data_ptr,
            data_len: len,
            len: leaf_count,
        })
    }

    /// Number of model entries in this file.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the file contains no model entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the model entry at index `i` via a zero-copy read.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`. The caller must ensure the index is in
    /// range; `cross_validate` in `index/mod.rs` ensures the sidecar is
    /// self-consistent (bit_shift ↔ l2_leaf_count) before any `entry` calls
    /// are made at runtime.
    #[inline]
    pub fn entry(&self, i: usize) -> ModelEntry {
        assert!(
            i < self.len,
            "ModelFileReader index {i} out of range (len={})",
            self.len
        );
        let off = MODEL_FILE_HEADER_BYTES + i * BYTES_PER_MODEL_ENTRY;
        // SAFETY: `data_ptr` points to valid bytes kept alive by `mmap` or
        // `_shm_mmap`. `off + 24 <= data_len` by header validation.
        unsafe {
            ModelEntry {
                alpha: LittleEndian::read_f64(std::slice::from_raw_parts(
                    self.data_ptr.add(off),
                    8,
                )),
                beta: LittleEndian::read_f64(std::slice::from_raw_parts(
                    self.data_ptr.add(off + 8),
                    8,
                )),
                err: LittleEndian::read_u64(std::slice::from_raw_parts(
                    self.data_ptr.add(off + 16),
                    8,
                )),
            }
        }
    }
}

/// Validate a model file header from a byte slice (used by both file-backed
/// and shm-backed constructors).
///
/// Returns `leaf_count` on success.
fn validate_model_header(
    data: &[u8],
    path: &Path,
    declared_len: usize,
    expected_layer: ModelLayer,
) -> Result<usize> {
    if data.len() < MODEL_FILE_HEADER_BYTES {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("too small ({} bytes) for model header", data.len()),
        });
    }
    let magic = LittleEndian::read_u32(&data[0..4]);
    if magic != expected_layer.magic() {
        return Err(Error::BadMagic {
            file: path.to_path_buf(),
            found: format!("{:#010x}", magic),
            expected: format!("{:#010x}", expected_layer.magic()),
        });
    }
    let version = LittleEndian::read_u32(&data[4..8]);
    if version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let leaf_count = LittleEndian::read_u64(&data[8..16]) as usize;
    // Checked arithmetic: a corrupted/malicious header declaring a huge
    // `leaf_count` must not wrap `expected_size` into a value that
    // coincidentally matches `declared_len` (which would let `entry()` read
    // out of bounds). Reject overflow explicitly.
    let expected_size = leaf_count
        .checked_mul(BYTES_PER_MODEL_ENTRY)
        .and_then(|body| body.checked_add(MODEL_FILE_HEADER_BYTES))
        .ok_or_else(|| Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("leaf_count {leaf_count} overflows the model-file size calculation"),
        })?;
    if declared_len != expected_size {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("component is {declared_len} bytes, expected {expected_size}"),
        });
    }
    Ok(leaf_count)
}
