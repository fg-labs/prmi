// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.sa` file format: 24-byte header + `num_entries × 5` packed positions.
//!
//! # Safety
//!
//! `SaFileReader` uses a read-only mmap. The backing file is kept open for the
//! reader's lifetime via the `_file` field, satisfying the mmap safety
//! invariant. Concurrent writers to the file are not supported.

// This module contains the single unsafe island in the prmi library (mmap).
#![allow(unsafe_code)]

use crate::error::{Error, Result};
use crate::sa::{pack_position, unpack_position, BYTES_PER_PACKED_ENTRY};
use crate::sidecar::magic::{FORMAT_VERSION, SA_MAGIC};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const SA_FILE_HEADER_BYTES: usize = 24;
const BYTES_PER_ENTRY: usize = BYTES_PER_PACKED_ENTRY; // 5

/// Streaming writer for the `.sa` file. Writes the 24-byte header on
/// `create`, accepts one position at a time via `write_position`, and
/// validates the expected-vs-actual entry count on `finish`.
pub struct SaFileWriter {
    path: PathBuf,
    inner: BufWriter<File>,
    expected: u64,
    written: u64,
}

impl SaFileWriter {
    pub fn create(path: &Path, expected_entries: u64) -> Result<Self> {
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
        let mut header = [0u8; SA_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], SA_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], expected_entries);
        header[16] = BYTES_PER_ENTRY as u8;
        // bytes 17..24 reserved zero (already).
        w.write_all(&header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            inner: w,
            expected: expected_entries,
            written: 0,
        })
    }

    pub fn write_position(&mut self, pos: u64) -> Result<()> {
        let bytes = pack_position(pos);
        self.inner.write_all(&bytes).map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        self.written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.inner.flush().map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        if self.written != self.expected {
            return Err(Error::SizeMismatch {
                file: self.path.clone(),
                detail: format!("wrote {} entries, expected {}", self.written, self.expected),
            });
        }
        Ok(())
    }
}

/// mmap-backed reader. After `open` succeeds, `position(i)` is a cheap
/// indexed lookup over the OS page cache.
///
/// # Concurrency
///
/// Concurrent writers to the underlying file are not supported. The caller
/// must ensure the file is not modified while this reader is alive.
pub struct SaFileReader {
    _file: File,
    mmap: Mmap,
    num_entries: u64,
}

impl std::fmt::Debug for SaFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaFileReader")
            .field("num_entries", &self.num_entries)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl SaFileReader {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: The file is opened read-only and `_file` keeps it alive for
        // the full lifetime of this struct. No concurrent writers are supported
        // (documented on the struct).
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if mmap.len() < SA_FILE_HEADER_BYTES {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("file too small ({} bytes) for .sa header", mmap.len()),
            });
        }
        let magic = LittleEndian::read_u32(&mmap[0..4]);
        if magic != SA_MAGIC {
            return Err(Error::BadMagic {
                file: path.to_path_buf(),
                found: format!("{:#010x}", magic),
                expected: format!("{:#010x}", SA_MAGIC),
            });
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                expected: FORMAT_VERSION,
            });
        }
        let num_entries = LittleEndian::read_u64(&mmap[8..16]);
        let bpe = mmap[16];
        if bpe != BYTES_PER_ENTRY as u8 {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("bytes_per_entry={bpe} (v0.1 expects 5)"),
            });
        }
        let expected_len = SA_FILE_HEADER_BYTES + (num_entries as usize) * BYTES_PER_ENTRY;
        if mmap.len() != expected_len {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("file is {} bytes, expected {}", mmap.len(), expected_len),
            });
        }
        Ok(Self {
            _file: f,
            mmap,
            num_entries,
        })
    }

    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    #[inline]
    pub fn position(&self, i: u64) -> u64 {
        let off = SA_FILE_HEADER_BYTES + (i as usize) * BYTES_PER_ENTRY;
        let bytes: &[u8; BYTES_PER_ENTRY] = self.mmap[off..off + BYTES_PER_ENTRY]
            .try_into()
            .expect("slice length checked");
        unpack_position(bytes)
    }
}
