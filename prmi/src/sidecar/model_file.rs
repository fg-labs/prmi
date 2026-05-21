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
use std::path::Path;

pub const MODEL_FILE_HEADER_BYTES: usize = 16;
pub const BYTES_PER_MODEL_ENTRY: usize = 24;

/// One model leaf entry: a linear predictor `(alpha, beta)` plus an `err`
/// bound. The `err` interpretation differs between L1 and L2 — see brief
/// §4.3-4.4. For L2 leaves on the fallback path, `err`'s high bit is set
/// and the rest encodes (partial_start, partial_num) into the L1 array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelEntry {
    pub alpha: f64,
    pub beta: f64,
    pub err: u64,
}

/// Which sidecar layer a file represents — controls the magic bytes used
/// for I/O and reader-side validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLayer {
    L1,
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
    _file: File,
    mmap: Mmap,
    len: usize,
}

impl std::fmt::Debug for ModelFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelFileReader")
            .field("len", &self.len)
            .finish()
    }
}

impl ModelFileReader {
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
        if mmap.len() < MODEL_FILE_HEADER_BYTES {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("file too small ({} bytes) for model header", mmap.len()),
            });
        }
        let magic = LittleEndian::read_u32(&mmap[0..4]);
        if magic != expected_layer.magic() {
            return Err(Error::BadMagic {
                file: path.to_path_buf(),
                found: format!("{:#010x}", magic),
                expected: format!("{:#010x}", expected_layer.magic()),
            });
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                expected: FORMAT_VERSION,
            });
        }
        let leaf_count = LittleEndian::read_u64(&mmap[8..16]) as usize;
        let expected_size = MODEL_FILE_HEADER_BYTES + leaf_count * BYTES_PER_MODEL_ENTRY;
        if mmap.len() != expected_size {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("file is {} bytes, expected {}", mmap.len(), expected_size),
            });
        }
        Ok(Self {
            _file: f,
            mmap,
            len: leaf_count,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn entry(&self, i: usize) -> ModelEntry {
        let off = MODEL_FILE_HEADER_BYTES + i * BYTES_PER_MODEL_ENTRY;
        ModelEntry {
            alpha: LittleEndian::read_f64(&self.mmap[off..off + 8]),
            beta: LittleEndian::read_f64(&self.mmap[off + 8..off + 16]),
            err: LittleEndian::read_u64(&self.mmap[off + 16..off + 24]),
        }
    }
}
