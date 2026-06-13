// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.sa` file format: 24-byte header + `num_entries × bytes_per_entry` packed entries.
//!
//! The `bytes_per_entry` field in the header determines the per-entry layout:
//!
//! | bytes_per_entry | Mode | Layout |
//! |---|---|---|
//! | 5  | 1 / suffix_key_cache | 5-byte packed position (`packed_lo8_hi32`) |
//! | 13 | 2 | 5-byte position + 8-byte LE u64 key |
//! | 21 | 3 | 5-byte position + 8-byte LE u64 key + 8-byte LE u64 ISA |
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
use std::sync::Arc;

/// Size of the binary header at the start of the `.sa` file, in bytes.
pub const SA_FILE_HEADER_BYTES: usize = 24;

/// Valid `bytes_per_entry` values, one per memory mode.
pub const VALID_BYTES_PER_ENTRY: &[u8] = &[5, 13, 21];

/// Mode 1 / suffix-key-cache: position only (5 bytes).
pub const BPE_MODE1: usize = BYTES_PER_PACKED_ENTRY; // 5
/// Mode 2: position + 8-byte key (13 bytes).
pub const BPE_MODE2: usize = BYTES_PER_PACKED_ENTRY + 8; // 13
/// Mode 3: position + 8-byte key + 8-byte ISA (21 bytes).
pub const BPE_MODE3: usize = BYTES_PER_PACKED_ENTRY + 16; // 21

/// Flush the per-entry accumulation buffer once it reaches this many bytes.
/// Entries are appended to a reusable `Vec` and written out one chunk at a
/// time, so a whole-genome build issues ~`total_bytes / SA_WRITE_CHUNK_BYTES`
/// `write_all` calls instead of one (or two, for mode 2) per SA entry.
const SA_WRITE_CHUNK_BYTES: usize = 64 * 1024;

/// Streaming writer for the `.sa` file. Writes the 24-byte header on
/// `create`, accepts one entry at a time via `write_entry`, and validates
/// the expected-vs-actual entry count on `finish`.
pub struct SaFileWriter {
    path: PathBuf,
    inner: BufWriter<File>,
    bytes_per_entry: usize,
    expected: u64,
    written: u64,
    /// Reusable accumulation buffer; flushed to `inner` per `SA_WRITE_CHUNK_BYTES`.
    chunk: Vec<u8>,
}

impl SaFileWriter {
    /// Create a new mode-1 `.sa` file (position only, 5 B/entry).
    ///
    /// This is the backward-compatible constructor used by code that predates
    /// the memory-mode menu. For other modes, use [`SaFileWriter::create_with_mode`].
    pub fn create(path: &Path, expected_entries: u64) -> Result<Self> {
        Self::create_with_mode(path, expected_entries, BPE_MODE1)
    }

    /// Create a `.sa` file with an explicit `bytes_per_entry`.
    ///
    /// `bytes_per_entry` must be one of `5`, `13`, or `21` (see module docs).
    /// Returns `Err(Error::Internal)` for any other value.
    pub fn create_with_mode(
        path: &Path,
        expected_entries: u64,
        bytes_per_entry: usize,
    ) -> Result<Self> {
        // Validate against the usize mode constants directly. Casting to `u8`
        // first would let large values wrap into a valid byte (e.g. 261 → 5)
        // and pass validation, producing an inconsistent header.
        if !matches!(bytes_per_entry, BPE_MODE1 | BPE_MODE2 | BPE_MODE3) {
            return Err(Error::Internal {
                detail: format!(
                    "SaFileWriter: invalid bytes_per_entry={bytes_per_entry}; \
                     must be one of {VALID_BYTES_PER_ENTRY:?}"
                ),
            });
        }
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
        // `bytes_per_entry` is one of {5, 13, 21} by the check above, so it
        // always fits in a u8; `expect` documents that invariant.
        header[16] = u8::try_from(bytes_per_entry).expect("validated bytes_per_entry fits in u8");
        // bytes 17..24 reserved zero (already).
        w.write_all(&header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            inner: w,
            bytes_per_entry,
            expected: expected_entries,
            written: 0,
            // One extra max-entry's worth of headroom so a final append never
            // forces a reallocation past the flush threshold.
            chunk: Vec::with_capacity(SA_WRITE_CHUNK_BYTES + BPE_MODE3),
        })
    }

    /// Flush the accumulation buffer to `inner` if it has reached the chunk
    /// threshold. Called after each entry is appended.
    #[inline]
    fn maybe_flush_chunk(&mut self) -> Result<()> {
        if self.chunk.len() >= SA_WRITE_CHUNK_BYTES {
            self.flush_chunk()?;
        }
        Ok(())
    }

    /// Write any buffered entry bytes to `inner` and clear the buffer.
    fn flush_chunk(&mut self) -> Result<()> {
        if !self.chunk.is_empty() {
            self.inner.write_all(&self.chunk).map_err(|e| Error::Io {
                path: self.path.clone(),
                source: e,
            })?;
            self.chunk.clear();
        }
        Ok(())
    }

    /// Append a single packed SA position to a mode-1 file.
    ///
    /// Returns `Err(Error::Internal)` if this writer was created with
    /// `bytes_per_entry != 5`. For modes 2/3 use [`Self::write_entry_with_key`]
    /// / [`Self::write_entry_with_key_isa`].
    pub fn write_position(&mut self, pos: u64) -> Result<()> {
        if self.bytes_per_entry != BPE_MODE1 {
            return Err(Error::Internal {
                detail: format!(
                    "write_position called on a writer with bytes_per_entry={}; expected {BPE_MODE1} (use write_entry_with_key or write_entry_with_key_isa)",
                    self.bytes_per_entry
                ),
            });
        }
        self.chunk.extend_from_slice(&pack_position(pos));
        self.written += 1;
        self.maybe_flush_chunk()
    }

    /// Append a mode-2 entry: `pos` (5 bytes) + `key` (8-byte LE u64).
    ///
    /// Returns `Err(Error::Internal)` if this writer was created with
    /// `bytes_per_entry != 13`.
    pub fn write_entry_with_key(&mut self, pos: u64, key: u64) -> Result<()> {
        if self.bytes_per_entry != BPE_MODE2 {
            return Err(Error::Internal {
                detail: format!(
                    "write_entry_with_key called on a writer with bytes_per_entry={}; expected {BPE_MODE2}",
                    self.bytes_per_entry
                ),
            });
        }
        self.chunk.extend_from_slice(&pack_position(pos));
        self.chunk.extend_from_slice(&key.to_le_bytes());
        self.written += 1;
        self.maybe_flush_chunk()
    }

    /// Append a mode-3 entry: `pos` (5 bytes) + `key` (8-byte LE u64) + `isa` (8-byte LE u64).
    ///
    /// Returns `Err(Error::Internal)` if this writer was created with
    /// `bytes_per_entry != 21`.
    pub fn write_entry_with_key_isa(&mut self, pos: u64, key: u64, isa: u64) -> Result<()> {
        if self.bytes_per_entry != BPE_MODE3 {
            return Err(Error::Internal {
                detail: format!(
                    "write_entry_with_key_isa called on a writer with bytes_per_entry={}; expected {BPE_MODE3}",
                    self.bytes_per_entry
                ),
            });
        }
        self.chunk.extend_from_slice(&pack_position(pos));
        self.chunk.extend_from_slice(&key.to_le_bytes());
        self.chunk.extend_from_slice(&isa.to_le_bytes());
        self.written += 1;
        self.maybe_flush_chunk()
    }

    /// Flush and close the writer, verifying that exactly `expected_entries` were written.
    pub fn finish(mut self) -> Result<()> {
        self.flush_chunk()?;
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
    /// Keeps the file open (and therefore the mmap valid) for file-backed instances.
    /// `None` for shm-backed instances (the shm blob Mmap is owned by `_shm_mmap`).
    _file: Option<File>,
    /// For file-backed instances, this is the owned mmap of the `.sa` file.
    /// For shm-backed instances, this is `None`; the shm backing is `_shm_mmap`.
    /// Prefixed with `_` because data is accessed via `data_ptr`; field exists
    /// solely to extend the Mmap's lifetime.
    _mmap: Option<Mmap>,
    /// For shm-backed instances, the shared mmap of the parent shm blob and the
    /// byte offset + length of the `.sa` component within it.
    _shm_mmap: Option<Arc<Mmap>>,
    /// Pointer to the SA data bytes, either from `mmap` or from a sub-slice of
    /// `_shm_mmap`. Valid for the lifetime of this struct.
    data_ptr: *const u8,
    /// Total length (in bytes) of the SA data region pointed to by `data_ptr`.
    data_len: usize,
    num_entries: u64,
    /// Validated bytes-per-entry value read from the header.
    bytes_per_entry: usize,
}

// SAFETY: `data_ptr` points into a `Mmap` (or Arc<Mmap>) that this struct
// keeps alive. The data is read-only; no mutation occurs after construction.
unsafe impl Send for SaFileReader {}
unsafe impl Sync for SaFileReader {}

impl std::fmt::Debug for SaFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaFileReader")
            .field("num_entries", &self.num_entries)
            .field("bytes_per_entry", &self.bytes_per_entry)
            .field("data_len", &self.data_len)
            .finish()
    }
}

impl SaFileReader {
    /// Open and mmap the `.sa` file at `path`, validating its header.
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
        // Hint transparent huge pages for the SA. Seeding does ~20 RANDOM reads
        // per call into this many-GB array (≈83 GB for hg38), so 4 KB pages force
        // a page-table walk on nearly every probe — the TLB (~1.5 K entries)
        // cannot cover the working set. 2 MB pages cut the TLB footprint ~512×,
        // removing the per-probe page walk (orthogonal to cache residency: it
        // helps even when the entry is in L3). Advisory and Linux-only — a no-op
        // elsewhere and harmless if THP is unavailable; never affects
        // correctness, only latency.
        #[cfg(target_os = "linux")]
        let _ = mmap.advise(memmap2::Advice::HugePage);
        let (num_entries, bpe_usize) = validate_sa_header(&mmap, path, mmap.len())?;
        let data_ptr = mmap.as_ptr();
        let data_len = mmap.len();
        Ok(Self {
            _file: Some(f),
            _mmap: Some(mmap),
            _shm_mmap: None,
            data_ptr,
            data_len,
            num_entries,
            bytes_per_entry: bpe_usize,
        })
    }

    /// Construct an `SaFileReader` backed by a sub-slice of a shm blob mmap.
    ///
    /// `shm_mmap` is the `Arc<Mmap>` of the parent shm blob. `offset` and `len`
    /// identify the `.sa` component within it (as returned by `ShmBlob`).
    /// The component bytes must start with a valid `.sa` header.
    pub(crate) fn from_shm_slice(shm_mmap: Arc<Mmap>, offset: usize, len: usize) -> Result<Self> {
        // Build a fake path for error messages.
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
        let (num_entries, bpe_usize) = validate_sa_header(slice, &fake_path, len)?;
        let data_ptr = slice.as_ptr();
        Ok(Self {
            _file: None,
            _mmap: None,
            _shm_mmap: Some(shm_mmap),
            data_ptr,
            data_len: len,
            num_entries,
            bytes_per_entry: bpe_usize,
        })
    }

    /// Number of SA entries stored in this file.
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// Return the unpacked SA position at index `i` via a zero-copy read.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_entries()`. The caller must ensure the index is
    /// in range; `cross_validate` in `index/mod.rs` ensures the sidecar
    /// is self-consistent before any `position` calls are made at runtime.
    #[inline]
    pub fn position(&self, i: u64) -> u64 {
        assert!(
            i < self.num_entries,
            "SaFileReader index {i} out of range (len={})",
            self.num_entries
        );
        let bpe = self.bytes_per_entry;
        let off = SA_FILE_HEADER_BYTES + (i as usize) * bpe;
        // SAFETY: `data_ptr` points to a valid byte region of length `data_len`
        // kept alive by either `mmap` or `_shm_mmap`. `off + bpe <= data_len`
        // because header validation checked the exact expected length.
        let bytes: &[u8; BPE_MODE1] =
            unsafe { &*(self.data_ptr.add(off) as *const [u8; BPE_MODE1]) };
        unpack_position(bytes)
    }

    /// Issue a software prefetch hint for SA entry `i` into L1. Advisory and
    /// side-effect-free: an out-of-range `i` is ignored, and on architectures
    /// without a prefetch intrinsic it compiles to nothing. The 5-byte position
    /// and (mode 2/3) 8-byte key share one entry — a single cache line — so one
    /// prefetch warms both. Used by the binary-search probe loops to overlap the
    /// next cold DRAM read with the current keyed compare; it never changes a
    /// search result, only its latency.
    #[inline(always)]
    pub fn prefetch(&self, i: u64) {
        if i >= self.num_entries {
            return;
        }
        let off = SA_FILE_HEADER_BYTES + (i as usize) * self.bytes_per_entry;
        // SAFETY: `i < num_entries` and the header validation checked the exact
        // mapped length, so `off` (the entry start) is within `[0, data_len)`.
        let addr = unsafe { self.data_ptr.add(off) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `_mm_prefetch` is a pure hint — it touches no memory
        // architecturally, never faults, and has no memory-safety effects.
        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                addr as *const i8,
            );
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `prfm` is a hint instruction; it never faults and has no
        // memory-safety effects.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) addr,
                options(nostack, preserves_flags),
            );
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = addr;
    }

    /// Return the stored 32-mer key at SA index `i`, if this file was built in
    /// mode 2 or mode 3 (both of which store the key). Returns `None` for mode 1
    /// and suffix-key-cache mode (where keys are absent or stored separately).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_entries()`.
    #[inline]
    pub fn key_at(&self, i: u64) -> Option<u64> {
        if self.bytes_per_entry < BPE_MODE2 {
            return None;
        }
        assert!(
            i < self.num_entries,
            "SaFileReader::key_at index {i} out of range (len={})",
            self.num_entries
        );
        // Key starts at byte 5 within each entry (after the 5-byte position).
        let off = SA_FILE_HEADER_BYTES + (i as usize) * self.bytes_per_entry + BPE_MODE1;
        // SAFETY: same invariant as `position`.
        let bytes: &[u8; 8] = unsafe { &*(self.data_ptr.add(off) as *const [u8; 8]) };
        Some(LittleEndian::read_u64(bytes))
    }

    /// Combined read of SA entry `i`: `(position, key)`. The 5-byte position and
    /// the (mode 2/3) 8-byte key live in one entry / cache line, so the hot
    /// keyed-compare loop fetches both with a SINGLE bounds check and a SINGLE
    /// offset computation instead of one each via `position` + `key_at`. `key` is
    /// `None` in mode 1 (no inline key). Equivalent to `(position(i), key_at(i))`.
    #[inline]
    pub fn entry(&self, i: u64) -> (u64, Option<u64>) {
        assert!(
            i < self.num_entries,
            "SaFileReader::entry index {i} out of range (len={})",
            self.num_entries
        );
        let base = SA_FILE_HEADER_BYTES + (i as usize) * self.bytes_per_entry;
        // SAFETY: `i < num_entries` and header validation checked the exact mapped
        // length, so `[base, base + bytes_per_entry)` is within `[0, data_len)`.
        let pos_bytes: &[u8; BPE_MODE1] =
            unsafe { &*(self.data_ptr.add(base) as *const [u8; BPE_MODE1]) };
        let position = unpack_position(pos_bytes);
        let key = if self.bytes_per_entry >= BPE_MODE2 {
            // SAFETY: key occupies bytes `[base+5, base+13)`, within the entry.
            let key_bytes: &[u8; 8] =
                unsafe { &*(self.data_ptr.add(base + BPE_MODE1) as *const [u8; 8]) };
            Some(LittleEndian::read_u64(key_bytes))
        } else {
            None
        };
        (position, key)
    }

    /// Return the stored ISA (inverse suffix array) value at SA index `i`, if
    /// this file was built in mode 3. Returns `None` for modes 1 and 2.
    ///
    /// For the genome offset `p` stored at SA index `i` (`sa[i] = p`),
    /// `isa_at(i)` returns `isa[p]`: the inverse suffix array, which maps a
    /// genome position back to its rank in the suffix array. By definition
    /// `isa[sa[i]] = i`, so this equals `i` for v0.1 sidecars (entries are
    /// written in SA order).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_entries()`.
    #[inline]
    pub fn isa_at(&self, i: u64) -> Option<u64> {
        if self.bytes_per_entry < BPE_MODE3 {
            return None;
        }
        assert!(
            i < self.num_entries,
            "SaFileReader::isa_at index {i} out of range (len={})",
            self.num_entries
        );
        // ISA starts at byte 13 within each entry (after 5-byte position + 8-byte key).
        let off = SA_FILE_HEADER_BYTES + (i as usize) * self.bytes_per_entry + BPE_MODE1 + 8;
        // SAFETY: same invariant as `position`.
        let bytes: &[u8; 8] = unsafe { &*(self.data_ptr.add(off) as *const [u8; 8]) };
        Some(LittleEndian::read_u64(bytes))
    }

    /// Bytes per entry stored in this file (5, 13, or 21).
    pub fn bytes_per_entry(&self) -> usize {
        self.bytes_per_entry
    }
}

/// Validate a `.sa` binary header from a byte slice (used by both file-backed
/// and shm-backed constructors).
///
/// Returns `(num_entries, bytes_per_entry)` on success.
fn validate_sa_header(data: &[u8], path: &Path, declared_len: usize) -> Result<(u64, usize)> {
    if data.len() < SA_FILE_HEADER_BYTES {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("too small ({} bytes) for .sa header", data.len()),
        });
    }
    let magic = LittleEndian::read_u32(&data[0..4]);
    if magic != SA_MAGIC {
        return Err(Error::BadMagic {
            file: path.to_path_buf(),
            found: format!("{:#010x}", magic),
            expected: format!("{:#010x}", SA_MAGIC),
        });
    }
    let version = LittleEndian::read_u32(&data[4..8]);
    if version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let num_entries = LittleEndian::read_u64(&data[8..16]);
    let bpe = data[16];
    if !VALID_BYTES_PER_ENTRY.contains(&bpe) {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("bytes_per_entry={bpe} (must be one of {VALID_BYTES_PER_ENTRY:?})"),
        });
    }
    let bpe_usize = bpe as usize;
    // Checked conversion + arithmetic: a crafted header with a huge
    // `num_entries` must not wrap `expected_len` into a value that matches
    // `declared_len`, since `position`/`key_at`/`isa_at` rely on that exact
    // length invariant for their unsafe offset computations.
    let num_entries_usize = usize::try_from(num_entries).map_err(|_| Error::SizeMismatch {
        file: path.to_path_buf(),
        detail: format!("num_entries={num_entries} does not fit in usize"),
    })?;
    let expected_len = num_entries_usize
        .checked_mul(bpe_usize)
        .and_then(|body| body.checked_add(SA_FILE_HEADER_BYTES))
        .ok_or_else(|| Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!(
                "num_entries={num_entries} (bpe={bpe_usize}) overflows the .sa size calculation"
            ),
        })?;
    if declared_len != expected_len {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("component is {declared_len} bytes, expected {expected_len}"),
        });
    }
    Ok((num_entries, bpe_usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// A `bytes_per_entry` that wraps to a valid mode byte when cast to `u8`
    /// (261 & 0xFF == 5) must still be rejected, not silently accepted.
    #[test]
    fn create_with_mode_rejects_wrapping_bytes_per_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wrap.sa");
        // `SaFileWriter` is not `Debug`, so use `.err().unwrap()` rather than
        // `.unwrap_err()`.
        let err = SaFileWriter::create_with_mode(&path, 0, 256 + BPE_MODE1)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("invalid bytes_per_entry"));
    }

    /// A crafted header with a `num_entries` so large that
    /// `num_entries * bytes_per_entry` overflows must return an error rather
    /// than panic on the unchecked multiply.
    #[test]
    fn validate_sa_header_rejects_overflowing_num_entries() {
        let mut header = [0u8; SA_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], SA_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], u64::MAX);
        header[16] = BPE_MODE3 as u8;
        let err =
            validate_sa_header(&header, Path::new("<test>"), SA_FILE_HEADER_BYTES).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("overflow"));
    }

    /// `from_shm_slice` must reject an offset/len that falls outside the
    /// backing mmap instead of panicking on the slice index.
    #[test]
    fn from_shm_slice_rejects_out_of_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&[0u8; 64]).unwrap();
        }
        let f = File::open(&path).unwrap();
        // SAFETY: read-only mmap of a file we just wrote; no concurrent writers.
        let mmap = Arc::new(unsafe { Mmap::map(&f) }.unwrap());
        // offset + len far exceeds the 64-byte blob.
        let err = SaFileReader::from_shm_slice(Arc::clone(&mmap), 32, 1024).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("out of bounds"));
        // offset + len that overflows usize is also rejected.
        let err = SaFileReader::from_shm_slice(mmap, usize::MAX, 1).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("overflow"));
    }
}
