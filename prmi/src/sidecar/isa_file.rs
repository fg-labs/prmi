// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.isa` file: 24-byte header + `num_entries × 5` packed uint40 entries.
//! Stores the position-indexed inverse SA (`ref2sa`): `isa[p]` = SA index whose
//! suffix starts at reference position `p`. Powers O(≈1)-probe backward
//! extension (Plan 3).

// This module contains an unsafe island (mmap).
#![allow(unsafe_code)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::sa::{pack_position, unpack_position, BYTES_PER_PACKED_ENTRY};
use crate::sidecar::magic::ISA_MAGIC;

const ISA_HEADER_BYTES: usize = 24;

/// Streaming writer for the `.isa` file.
pub struct IsaFileWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    expected: u64,
    written: u64,
}

impl IsaFileWriter {
    /// Create an `.isa` file at `path` for exactly `expected_entries` entries.
    pub fn create(path: &Path, expected_entries: u64) -> Result<Self> {
        let file = File::create(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut writer = BufWriter::new(file);
        let mut header = [0u8; ISA_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], ISA_MAGIC);
        // bytes[4..8] reserved (0); [8..16] num_entries; [16] bytes/entry; rest 0.
        LittleEndian::write_u64(&mut header[8..16], expected_entries);
        header[16] = BYTES_PER_PACKED_ENTRY as u8;
        writer.write_all(&header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            expected: expected_entries,
            written: 0,
        })
    }

    /// Append one inverse-SA entry (an SA index).
    pub fn write_entry(&mut self, sa_index: u64) -> Result<()> {
        let bytes = pack_position(sa_index);
        self.writer.write_all(&bytes).map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        self.written += 1;
        Ok(())
    }

    /// Flush and validate the entry count.
    pub fn finish(mut self) -> Result<()> {
        self.writer.flush().map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        if self.written != self.expected {
            return Err(Error::Internal {
                detail: format!(
                    "isa wrote {} entries, expected {}",
                    self.written, self.expected
                ),
            });
        }
        Ok(())
    }
}

/// Memory-mapped reader for the `.isa` file.
#[derive(Debug)]
pub struct IsaFileReader {
    /// Keeps the file open for the lifetime of the mmap.
    _file: File,
    mmap: Mmap,
    num_entries: u64,
}

impl IsaFileReader {
    /// Open and validate the `.isa` header.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: The file is opened read-only and `_file` keeps it alive for
        // the full lifetime of this struct. No concurrent writers are supported.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if mmap.len() < ISA_HEADER_BYTES {
            return Err(Error::Internal {
                detail: format!("isa {} too short", path.display()),
            });
        }
        let magic = LittleEndian::read_u32(&mmap[0..4]);
        if magic != ISA_MAGIC {
            return Err(Error::Internal {
                detail: format!("isa {} bad magic {:#x}", path.display(), magic),
            });
        }
        let num_entries = LittleEndian::read_u64(&mmap[8..16]);
        // Compute the expected on-disk size with checked arithmetic: a crafted
        // large `num_entries` must fail validation rather than wrap `expect`
        // around and let a truncated file pass.
        let n = usize::try_from(num_entries).map_err(|_| Error::Internal {
            detail: format!(
                "isa {} num_entries too large: {num_entries}",
                path.display()
            ),
        })?;
        let body = n
            .checked_mul(BYTES_PER_PACKED_ENTRY)
            .ok_or_else(|| Error::Internal {
                detail: format!("isa {} body size overflow", path.display()),
            })?;
        let expect = ISA_HEADER_BYTES
            .checked_add(body)
            .ok_or_else(|| Error::Internal {
                detail: format!("isa {} total size overflow", path.display()),
            })?;
        if mmap.len() != expect {
            return Err(Error::Internal {
                detail: format!(
                    "isa {} size {} != expected {}",
                    path.display(),
                    mmap.len(),
                    expect
                ),
            });
        }
        Ok(Self {
            _file: file,
            mmap,
            num_entries,
        })
    }

    /// Number of entries (= `N+1`).
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// SA index whose suffix starts at reference position `p`. Panics if
    /// `p >= num_entries`.
    pub fn sa_index_for_refpos(&self, p: u64) -> u64 {
        assert!(
            p < self.num_entries,
            "isa index {p} out of range {}",
            self.num_entries
        );
        let off = ISA_HEADER_BYTES + p as usize * BYTES_PER_PACKED_ENTRY;
        let mut a = [0u8; BYTES_PER_PACKED_ENTRY];
        a.copy_from_slice(&self.mmap[off..off + BYTES_PER_PACKED_ENTRY]);
        unpack_position(&a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.isa");
        let vals = [4u64, 0, 2, 1, 3];
        let mut w = IsaFileWriter::create(&p, vals.len() as u64).unwrap();
        for &v in &vals {
            w.write_entry(v).unwrap();
        }
        w.finish().unwrap();
        let r = IsaFileReader::open(&p).unwrap();
        assert_eq!(r.num_entries(), vals.len() as u64);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(r.sa_index_for_refpos(i as u64), v);
        }
    }
}
