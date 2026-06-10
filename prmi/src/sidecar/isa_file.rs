// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.isa` file format: the inverse suffix array, indexed by reference position.
//!
//! 24-byte header + `num_entries × 5` packed entries. Entry at reference
//! position `p` is the SA index `i` such that `sa[i] == p` — i.e. `prmi_isa_at`'s
//! `inv[p] = i` (BWA-MEME's `ref2sa`). The forward `mem_search` `est_hint`
//! (no-search launch) is `inv[refpos]`, so a refpos→SA-index lookup must be O(1);
//! the suffix array's per-row layout (ordered by suffix, not by refpos) cannot
//! serve that, which is why the inverse SA is a separate refpos-indexed file.
//!
//! SA indices share the `.sa` file's 5-byte (uint40) packing — both index the
//! same `[0, 2·l_pac]` doubled-coordinate space, well within 2^40.
//!
//! This sidecar is OPTIONAL (emitted only by `prmi build --with-isa`): it costs
//! `+5` bytes per SA entry (~+32 GB at hg38), so it is the operator's choice. The
//! same binary runs ISA-accelerated or model-launch-only depending on whether the
//! `.isa` is present (`LearnedIndex::has_isa`).
//!
//! # Safety
//!
//! `IsaFileReader` uses a read-only mmap kept alive for the reader's lifetime.

// mmap island.
#![allow(unsafe_code)]

use crate::error::{Error, Result};
use crate::sa::{pack_position, unpack_position, BYTES_PER_PACKED_ENTRY};
use crate::sidecar::magic::{FORMAT_VERSION, ISA_MAGIC};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Size of the `.isa` binary header, in bytes (mirrors the `.sa` header).
pub const ISA_FILE_HEADER_BYTES: usize = 24;

/// Build the inverse suffix array from the SA permutation and write it to `path`.
///
/// `sa` is the suffix array: a permutation of `[0, sa.len())` where `sa[i]` is the
/// reference position of the `i`-th suffix in sorted order. The inverse is
/// `inv[sa[i]] = i`; we materialise it directly into the packed (5-byte) on-disk
/// layout to avoid a second `u64` array (the packed buffer is ~5/8 the size of a
/// `Vec<u64>` inverse).
pub fn write_isa_file(path: &Path, sa: &[u64]) -> Result<()> {
    let n = sa.len();
    let io = |e: std::io::Error| Error::Io {
        path: path.to_path_buf(),
        source: e,
    };

    // Size the file up front (header + n packed 5-byte entries) and pack the
    // entries directly into a writable mmap. This avoids staging the whole
    // `n * 5`-byte body in a heap `Vec` first (~32 GB at hg38 scale, enough to
    // OOM a `--with-isa` build). `set_len` zero-fills, so reserved header bytes
    // and any never-written body bytes are already zero.
    let total = ISA_FILE_HEADER_BYTES + n * BYTES_PER_PACKED_ENTRY;
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(io)?;
    f.set_len(total as u64).map_err(io)?;

    // SAFETY: `f` is opened read+write and sized to `total`; this mmap is the
    // sole accessor for the file over its (function-scoped) lifetime — no
    // concurrent writers — and is flushed and unmapped before we return.
    let mut mmap = unsafe { MmapMut::map_mut(&f) }.map_err(io)?;

    LittleEndian::write_u32(&mut mmap[0..4], ISA_MAGIC);
    LittleEndian::write_u32(&mut mmap[4..8], FORMAT_VERSION);
    LittleEndian::write_u64(&mut mmap[8..16], n as u64);
    mmap[16] = BYTES_PER_PACKED_ENTRY as u8;
    // bytes 17..24 reserved zero (from `set_len`).

    // Pack inv[p] = i directly: for each sorted index i at reference position
    // p = sa[i], store the 5-byte SA index i at body offset p*5.
    let body = &mut mmap[ISA_FILE_HEADER_BYTES..];
    for (i, &p) in sa.iter().enumerate() {
        let off = (p as usize) * BYTES_PER_PACKED_ENTRY;
        body[off..off + BYTES_PER_PACKED_ENTRY].copy_from_slice(&pack_position(i as u64));
    }

    mmap.flush().map_err(io)?;
    Ok(())
}

/// mmap-backed reader for the `.isa` inverse suffix array. After `open`,
/// `sa_index_at(refpos)` is a cheap indexed lookup.
///
/// # Concurrency
///
/// Concurrent writers to the underlying file are not supported.
pub struct IsaFileReader {
    /// Keeps the file open (and the mmap valid) for the reader's lifetime.
    _file: File,
    /// Owned mmap of the `.isa` file. Prefixed `_` because data is read via
    /// `data_ptr`; the field exists to extend the mmap's lifetime.
    _mmap: Mmap,
    /// Pointer to the entry bytes (after the header).
    data_ptr: *const u8,
    num_entries: u64,
}

// SAFETY: `data_ptr` points into a `Mmap` this struct keeps alive; the data is
// read-only and never mutated after construction.
unsafe impl Send for IsaFileReader {}
unsafe impl Sync for IsaFileReader {}

impl std::fmt::Debug for IsaFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IsaFileReader")
            .field("num_entries", &self.num_entries)
            .finish()
    }
}

impl IsaFileReader {
    /// Open and mmap the `.isa` file at `path`, validating its header.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: opened read-only; `_file` keeps it alive for the struct's
        // lifetime; no concurrent writers (documented).
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let num_entries = validate_isa_header(&mmap, path)?;
        let data_ptr = unsafe { mmap.as_ptr().add(ISA_FILE_HEADER_BYTES) };
        Ok(Self {
            _file: f,
            _mmap: mmap,
            data_ptr,
            num_entries,
        })
    }

    /// Number of reference positions covered (= the SA length, `2·l_pac + 1`).
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// Return the SA index of reference position `refpos` (the inverse SA).
    ///
    /// # Panics
    ///
    /// Panics if `refpos >= self.num_entries()`. Callers (the FFI) bounds-check
    /// and return an error code instead.
    #[inline]
    pub fn sa_index_at(&self, refpos: u64) -> u64 {
        assert!(
            refpos < self.num_entries,
            "IsaFileReader index {refpos} out of range (len={})",
            self.num_entries
        );
        let off = (refpos as usize) * BYTES_PER_PACKED_ENTRY;
        // SAFETY: `data_ptr` points to `num_entries * 5` valid bytes (checked by
        // `validate_isa_header`); `off + 5 <= num_entries * 5` since
        // `refpos < num_entries`.
        let bytes: &[u8; BYTES_PER_PACKED_ENTRY] =
            unsafe { &*(self.data_ptr.add(off) as *const [u8; BYTES_PER_PACKED_ENTRY]) };
        unpack_position(bytes)
    }
}

/// Validate an `.isa` header. Returns `num_entries` on success.
fn validate_isa_header(data: &[u8], path: &Path) -> Result<u64> {
    if data.len() < ISA_FILE_HEADER_BYTES {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("too small ({} bytes) for .isa header", data.len()),
        });
    }
    let magic = LittleEndian::read_u32(&data[0..4]);
    if magic != ISA_MAGIC {
        return Err(Error::BadMagic {
            file: path.to_path_buf(),
            found: format!("{magic:#010x}"),
            expected: format!("{ISA_MAGIC:#010x}"),
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
    if bpe as usize != BYTES_PER_PACKED_ENTRY {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("bytes_per_index={bpe} (must be {BYTES_PER_PACKED_ENTRY})"),
        });
    }
    // Compute the expected size with checked arithmetic: a crafted large
    // num_entries must fail validation rather than wrap expected_len around and
    // let a truncated file pass.
    let n = usize::try_from(num_entries).map_err(|_| Error::SizeMismatch {
        file: path.to_path_buf(),
        detail: format!("num_entries too large: {num_entries}"),
    })?;
    let expected_len = n
        .checked_mul(BYTES_PER_PACKED_ENTRY)
        .and_then(|body| body.checked_add(ISA_FILE_HEADER_BYTES))
        .ok_or_else(|| Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!(".isa size overflow for num_entries={num_entries}"),
        })?;
    if data.len() != expected_len {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("file is {} bytes, expected {expected_len}", data.len()),
        });
    }
    Ok(num_entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Round-trip: `write_isa_file(sa)` then `sa_index_at(p)` reproduces the
    /// inverse permutation `inv[sa[i]] = i` for every position.
    #[test]
    fn isa_round_trips_inverse_permutation() {
        // A small but non-trivial permutation of [0, 8).
        let sa: Vec<u64> = vec![3, 0, 7, 1, 6, 2, 5, 4];
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.isa");
        write_isa_file(&path, &sa).unwrap();

        let r = IsaFileReader::open(&path).unwrap();
        assert_eq!(r.num_entries(), sa.len() as u64);
        for (i, &p) in sa.iter().enumerate() {
            assert_eq!(r.sa_index_at(p), i as u64, "inv[sa[{i}]={p}] should be {i}");
        }
        // Exact file size.
        let size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(
            size,
            (ISA_FILE_HEADER_BYTES + sa.len() * BYTES_PER_PACKED_ENTRY) as u64
        );
    }

    /// A large permutation crosses the chunk-flush boundary and exercises the
    /// 5-byte packing on big SA indices.
    #[test]
    fn isa_round_trips_large_and_crosses_chunk() {
        // Reverse permutation of [0, 20000): sa[i] = n-1-i, inv[p] = n-1-p.
        let n = 20_000u64;
        let sa: Vec<u64> = (0..n).rev().collect();
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.isa");
        write_isa_file(&path, &sa).unwrap();
        let r = IsaFileReader::open(&path).unwrap();
        for p in 0..n {
            assert_eq!(r.sa_index_at(p), n - 1 - p, "mismatch at refpos {p}");
        }
    }

    #[test]
    fn isa_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("garbage.isa");
        std::fs::write(&path, vec![0xffu8; 100]).unwrap();
        let err = IsaFileReader::open(&path).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("magic"));
    }

    #[test]
    fn isa_rejects_overflowing_num_entries() {
        // A valid header claiming `u64::MAX` entries: `num_entries * 5` overflows
        // `usize`, so the size check must fail (SizeMismatch) rather than wrap the
        // expected length around and accept a truncated file (or panic).
        let dir = tempdir().unwrap();
        let path = dir.path().join("overflow.isa");
        let mut header = vec![0u8; ISA_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], ISA_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], u64::MAX);
        header[16] = BYTES_PER_PACKED_ENTRY as u8;
        std::fs::write(&path, &header).unwrap();
        let err = IsaFileReader::open(&path).unwrap_err();
        assert!(
            matches!(err, Error::SizeMismatch { .. }),
            "expected SizeMismatch on overflowing num_entries, got: {err:?}"
        );
    }
}
