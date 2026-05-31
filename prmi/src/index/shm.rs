// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Shared-memory-backed sidecar loader: `prmi shm load` + `LearnedIndex::open_shm`.
//!
//! # Overview
//!
//! `write_shm_blob` packs the four sidecar files (`.meta`, `.sa`, `.l1`, `.l2`)
//! into a single blob file. The blob header records the byte offset and length of
//! each component; each component starts on a 4 KiB-aligned boundary.
//!
//! `read_shm_blob` opens and mmaps a previously-written blob, validates the
//! wrapper header, and returns an [`ShmBlob`] describing the layout.
//! `LearnedIndex::open_shm` calls `read_shm_blob` then passes the appropriate
//! sub-slices to the internal slice-backed reader constructors.
//!
//! # Blob wrapper format
//!
//! ```text
//! [0..16)   : magic "PRMI_SHM_v1\0\0\0\0\0"  (16 bytes, NUL-padded)
//! [16..24)  : u64 wrapper_format_version = 1   (little-endian)
//! [24..32)  : u64 meta_offset
//! [32..40)  : u64 meta_len
//! [40..48)  : u64 sa_offset
//! [48..56)  : u64 sa_len
//! [56..64)  : u64 l1_offset
//! [64..72)  : u64 l1_len
//! [72..80)  : u64 l2_offset
//! [80..88)  : u64 l2_len
//! [88..4096): zero padding (reserved)
//! [4096..)  : concatenated components, each starting on a 4 KiB boundary
//! ```
//!
//! All multi-byte integers are little-endian. Component alignment to 4 KiB
//! ensures that each component starts on a page boundary, which is necessary for
//! callers that want to mmap individual components independently.
//!
//! # Cross-process sharing
//!
//! Multiple processes that `mmap(MAP_SHARED)` the same blob path share physical
//! pages via the OS page cache. Verified empirically on Linux (`/dev/shm/…`)
//! and macOS (`/tmp/…`). True cross-process isolation testing will be added in
//! v0.2. See also: [`LearnedIndex::open_shm`] docs.
//!
//! # Limitations
//!
//! - **Concurrent writers not supported.** If `prmi shm load` is interrupted
//!   mid-write, the blob is corrupt and `open_shm` will return an error.
//! - **Crash safety not provided.** A process killed while reading an in-progress
//!   write may observe a partially written blob.

// This module contains a read-only mmap (unsafe island).
#![allow(unsafe_code)]

use crate::error::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

// ── constants ─────────────────────────────────────────────────────────────────

/// Wrapper header magic string (16 bytes, NUL-padded ASCII).
pub const SHM_MAGIC: &[u8; 16] = b"PRMI_SHM_v1\x00\x00\x00\x00\x00";

/// Wrapper format version stored at offset 16.
pub const SHM_WRAPPER_VERSION: u64 = 1;

/// Total wrapper header size; components are aligned to this boundary.
const HEADER_SIZE: usize = 4096;

/// Alignment boundary for each component within the blob (4 KiB = one page).
const COMPONENT_ALIGN: usize = 4096;

// ── blob layout descriptor ────────────────────────────────────────────────────

/// Offsets and lengths of the four sidecar components within the blob.
///
/// Returned by [`read_shm_blob`]; consumed by [`LearnedIndex::open_shm`].
#[derive(Debug, Clone)]
pub struct ShmBlob {
    /// The memory-mapped blob file. Shared across all component slice views.
    pub mmap: Arc<Mmap>,
    /// Byte offset of the `.meta` component within `mmap`.
    pub meta_offset: usize,
    /// Byte length of the `.meta` component.
    pub meta_len: usize,
    /// Byte offset of the `.sa` component.
    pub sa_offset: usize,
    /// Byte length of the `.sa` component.
    pub sa_len: usize,
    /// Byte offset of the `.l1` component.
    pub l1_offset: usize,
    /// Byte length of the `.l1` component.
    pub l1_len: usize,
    /// Byte offset of the `.l2` component.
    pub l2_offset: usize,
    /// Byte length of the `.l2` component.
    pub l2_len: usize,
}

// ── internal helpers ──────────────────────────────────────────────────────────

/// Round `n` up to the next multiple of `align` (which must be a power of two).
#[inline]
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

/// Read the entire contents of `path` into a `Vec<u8>`.
fn read_all(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(buf)
}

/// Write zero bytes to pad `writer` to an `align`-byte boundary.
/// `current_offset` is the number of bytes already written.
fn write_padding(
    writer: &mut BufWriter<File>,
    current_offset: usize,
    align: usize,
    path: &Path,
) -> Result<usize> {
    let target = align_up(current_offset, align);
    let pad = target - current_offset;
    if pad > 0 {
        let zeros = vec![0u8; pad];
        writer.write_all(&zeros).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
    }
    Ok(target)
}

// ── public writer ─────────────────────────────────────────────────────────────

/// Pack a sidecar into a single shm blob at `shm_path`.
///
/// Reads the four sidecar files from `<sidecar_prefix>.{meta,sa,l1,l2}` and
/// concatenates them into a blob at `shm_path`. An existing file at `shm_path`
/// is overwritten. The blob can be opened by [`read_shm_blob`] or
/// [`LearnedIndex::open_shm`].
///
/// The typical destination on Linux is `/dev/shm/<name>` (tmpfs-backed shared
/// memory). On macOS `/tmp/<name>` is the equivalent. Multiple processes that
/// subsequently `mmap` the same `shm_path` share the OS page-cache pages.
pub fn write_shm_blob(sidecar_prefix: &Path, shm_path: &Path) -> Result<()> {
    use crate::sidecar::SidecarPaths;
    let paths = SidecarPaths::from_prefix(sidecar_prefix);

    // Read all four component files.
    let meta_bytes = read_all(&paths.meta)?;
    let sa_bytes = read_all(&paths.sa)?;
    let l1_bytes = read_all(&paths.l1)?;
    let l2_bytes = read_all(&paths.l2)?;

    // Compute component offsets (each 4 KiB-aligned, starting after the header).
    let meta_offset = HEADER_SIZE; // first component starts right after the header
    let sa_offset = align_up(meta_offset + meta_bytes.len(), COMPONENT_ALIGN);
    let l1_offset = align_up(sa_offset + sa_bytes.len(), COMPONENT_ALIGN);
    let l2_offset = align_up(l1_offset + l1_bytes.len(), COMPONENT_ALIGN);

    // Build the 4 KiB wrapper header.
    let mut header = [0u8; HEADER_SIZE];
    header[0..16].copy_from_slice(SHM_MAGIC);
    LittleEndian::write_u64(&mut header[16..24], SHM_WRAPPER_VERSION);
    LittleEndian::write_u64(&mut header[24..32], meta_offset as u64);
    LittleEndian::write_u64(&mut header[32..40], meta_bytes.len() as u64);
    LittleEndian::write_u64(&mut header[40..48], sa_offset as u64);
    LittleEndian::write_u64(&mut header[48..56], sa_bytes.len() as u64);
    LittleEndian::write_u64(&mut header[56..64], l1_offset as u64);
    LittleEndian::write_u64(&mut header[64..72], l1_bytes.len() as u64);
    LittleEndian::write_u64(&mut header[72..80], l2_offset as u64);
    LittleEndian::write_u64(&mut header[80..88], l2_bytes.len() as u64);
    // bytes [88..4096) remain zero (reserved).

    // Write the blob file.
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(shm_path)
        .map_err(|e| Error::Io {
            path: shm_path.to_path_buf(),
            source: e,
        })?;
    let mut w = BufWriter::new(f);

    // Header.
    w.write_all(&header).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    // Meta component (already at header_size == meta_offset, no leading padding needed).
    debug_assert_eq!(HEADER_SIZE, meta_offset);
    w.write_all(&meta_bytes).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    // Pad to sa_offset.
    let off = write_padding(
        &mut w,
        meta_offset + meta_bytes.len(),
        COMPONENT_ALIGN,
        shm_path,
    )?;
    debug_assert_eq!(off, sa_offset);
    w.write_all(&sa_bytes).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    // Pad to l1_offset.
    let off = write_padding(
        &mut w,
        sa_offset + sa_bytes.len(),
        COMPONENT_ALIGN,
        shm_path,
    )?;
    debug_assert_eq!(off, l1_offset);
    w.write_all(&l1_bytes).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    // Pad to l2_offset.
    let off = write_padding(
        &mut w,
        l1_offset + l1_bytes.len(),
        COMPONENT_ALIGN,
        shm_path,
    )?;
    debug_assert_eq!(off, l2_offset);
    w.write_all(&l2_bytes).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;

    w.flush().map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

// ── public reader ─────────────────────────────────────────────────────────────

/// Open and mmap a previously-written shm blob, validate the wrapper header,
/// and return the component layout.
///
/// The returned [`ShmBlob`] holds an `Arc<Mmap>` shared across all component
/// slice views. Multiple processes mapping the same `shm_path` with
/// `MAP_SHARED` share the underlying OS pages.
pub fn read_shm_blob(shm_path: &Path) -> Result<ShmBlob> {
    let f = File::open(shm_path).map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?;
    // SAFETY: The file is opened read-only; `mmap` is wrapped in Arc so its
    // lifetime covers all derived slice views. No concurrent writers are
    // supported (documented on the module).
    let mmap = Arc::new(unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
        path: shm_path.to_path_buf(),
        source: e,
    })?);

    let blob = mmap.as_ref();
    if blob.len() < HEADER_SIZE {
        return Err(Error::SizeMismatch {
            file: shm_path.to_path_buf(),
            detail: format!(
                "shm blob too small ({} bytes) for the {} byte wrapper header",
                blob.len(),
                HEADER_SIZE
            ),
        });
    }

    // Validate magic.
    if &blob[0..16] != SHM_MAGIC.as_slice() {
        return Err(Error::BadMagic {
            file: shm_path.to_path_buf(),
            found: format!("{:?}", &blob[0..16]),
            expected: format!("{:?}", SHM_MAGIC.as_slice()),
        });
    }

    // Validate wrapper version.
    let version = LittleEndian::read_u64(&blob[16..24]);
    if version != SHM_WRAPPER_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version as u32,
            expected: SHM_WRAPPER_VERSION as u32,
        });
    }

    // Read component offsets + lengths.
    let meta_offset = LittleEndian::read_u64(&blob[24..32]) as usize;
    let meta_len = LittleEndian::read_u64(&blob[32..40]) as usize;
    let sa_offset = LittleEndian::read_u64(&blob[40..48]) as usize;
    let sa_len = LittleEndian::read_u64(&blob[48..56]) as usize;
    let l1_offset = LittleEndian::read_u64(&blob[56..64]) as usize;
    let l1_len = LittleEndian::read_u64(&blob[64..72]) as usize;
    let l2_offset = LittleEndian::read_u64(&blob[72..80]) as usize;
    let l2_len = LittleEndian::read_u64(&blob[80..88]) as usize;

    // Enforce the documented PRMI_SHM_v1 wrapper layout: every component must
    // start after the reserved header, be 4 KiB-aligned, sit within the blob,
    // and follow the previous component without overlap (components are written
    // in meta → sa → l1 → l2 order).
    let mut prev_end = HEADER_SIZE;
    for (name, offset, len) in [
        ("meta", meta_offset, meta_len),
        ("sa", sa_offset, sa_len),
        ("l1", l1_offset, l1_len),
        ("l2", l2_offset, l2_len),
    ] {
        if offset < HEADER_SIZE {
            return Err(Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: format!("{name} offset {offset} is inside the reserved header"),
            });
        }
        if offset % COMPONENT_ALIGN != 0 {
            return Err(Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: format!("{name} offset {offset} is not {COMPONENT_ALIGN}-byte aligned"),
            });
        }
        if offset < prev_end {
            return Err(Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: format!(
                    "{name} offset {offset} overlaps or precedes the previous component (ends at {prev_end})"
                ),
            });
        }
        let end = offset.checked_add(len).ok_or_else(|| Error::SizeMismatch {
            file: shm_path.to_path_buf(),
            detail: format!("{name} offset+len overflows usize"),
        })?;
        if end > blob.len() {
            return Err(Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: format!(
                    "{name} range [{offset}, {end}) exceeds blob size {}",
                    blob.len()
                ),
            });
        }
        prev_end = end;
    }

    Ok(ShmBlob {
        mmap,
        meta_offset,
        meta_len,
        sa_offset,
        sa_len,
        l1_offset,
        l1_len,
        l2_offset,
        l2_len,
    })
}

/// Remove the shm blob at `shm_path`, if it exists.
///
/// Equivalent to `rm -f <shm_path>`. This is a convenience wrapper; callers
/// may also remove the file directly.
pub fn unload_shm_blob(shm_path: &Path) -> Result<()> {
    match std::fs::remove_file(shm_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::Io {
            path: shm_path.to_path_buf(),
            source: e,
        }),
    }
}
