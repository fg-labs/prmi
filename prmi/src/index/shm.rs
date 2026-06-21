// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Shared-memory-backed sidecar loader: `prmi shm load` + `LearnedIndex::open_shm`.
//!
//! # Overview
//!
//! `write_shm_blob` packs the four sidecar files (`.meta`, `.sa`, `.l1`, `.l2`)
//! into a single blob file. The blob header records the byte offset and
//! length of each component; each component starts on a 4 KiB-aligned boundary.
//!
//! `read_shm_blob` opens and mmaps a previously-written blob, validates the
//! wrapper header, and returns an [`ShmBlob`] describing the layout.
//! `LearnedIndex::open_shm` calls `read_shm_blob` then passes the appropriate
//! sub-slices to the internal slice-backed reader constructors.
//!
//! # Blob wrapper format
//!
//! ```text
//! [0..16)    : magic "PRMI_SHM_v3\0\0\0\0\0"  (16 bytes, NUL-padded)
//! [16..24)   : u64 wrapper_format_version = 3   (little-endian)
//! [24..32)   : u64 meta_offset
//! [32..40)   : u64 meta_len
//! [40..48)   : u64 sa_offset
//! [48..56)   : u64 sa_len
//! [56..64)   : u64 l1_offset
//! [64..72)   : u64 l1_len
//! [72..80)   : u64 l2_offset
//! [80..88)   : u64 l2_len
//! [88..96)   : u64 kmt_offset  (0 when no `.kmt` is carried)
//! [96..104)  : u64 kmt_len     (0 when no `.kmt` is carried)
//! [104..112) : u64 blm_offset  (0 when no `.blm` is carried)
//! [112..120) : u64 blm_len     (0 when no `.blm` is carried)
//! [120..4096): zero padding (reserved)
//! [4096..)   : concatenated components, each starting on a 4 KiB boundary
//! ```
//!
//! All multi-byte integers are little-endian. Component alignment to 4 KiB
//! ensures that each component starts on a page boundary, which is necessary for
//! callers that want to mmap individual components independently.
//!
//! The optional `.kmt` (a 5th component, a forward k-mer table) and `.blm` (a 6th
//! component, the Design-Z bloom dispatch gate) were added without bumping the
//! wrapper version: their header slots were specified-zero from the start, so a
//! `*_len == 0` reads as "not carried" on both old and new readers. The version is
//! therefore *intentionally* retained at 3 — do not bump it for these optional
//! components. An old reader simply ignores a carried table/bloom; a new reader
//! treats an old (zero) blob as lacking them. Both are pure accelerators (the
//! `.kmt` for forward search; the `.blm` gate confirms every hit with a
//! `mem_search`), so either way the result is correct (see
//! [`LearnedIndex::open_shm`] best-effort loading).
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
pub const SHM_MAGIC: &[u8; 16] = b"PRMI_SHM_v3\x00\x00\x00\x00\x00";

/// Wrapper format version stored at offset 16.
///
/// Bumped to 3 when the `.isa` component was removed (the format is unreleased,
/// so older blobs are rejected outright rather than supported).
pub const SHM_WRAPPER_VERSION: u64 = 3;

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
    /// Byte offset of the optional `.kmt` component (`0` when absent).
    pub kmt_offset: usize,
    /// Byte length of the optional `.kmt` component (`0` when absent).
    pub kmt_len: usize,
    /// Byte offset of the optional `.blm` bloom component (`0` when absent).
    pub blm_offset: usize,
    /// Byte length of the optional `.blm` bloom component (`0` when absent).
    pub blm_len: usize,
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
/// Reads the four sidecar files from `<sidecar_prefix>.{meta,sa,l1,l2}` (plus
/// the optional `.kmt` forward k-mer table, when present) and concatenates them
/// into a blob at `shm_path`. An existing file at `shm_path` is overwritten. The
/// blob can be opened by [`read_shm_blob`] or [`LearnedIndex::open_shm`].
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
    // Optional `.kmt` forward k-mer table (a 5th component). Absent for sidecars
    // built without `--kmer-table-k`; carried so shm-loaded indexes get the
    // table acceleration too.
    // Read the `.kmt` directly rather than gating on `Path::exists()` (which also
    // reports `false` when metadata is inaccessible, silently dropping a real
    // failure): treat only a genuine `NotFound` as "no table", surface any other
    // I/O error.
    let kmt_bytes: Option<Vec<u8>> = match std::fs::read(&paths.kmt) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            return Err(Error::Io {
                path: paths.kmt.clone(),
                source: e,
            })
        }
    };
    // Optional `.blm` bloom dispatch gate (a 6th component). Absent for sidecars
    // built without `--with-bloom`; carried so shm-loaded indexes get the cheap
    // first-window bloom gate (Lever 2, A1) too — otherwise a shared index would
    // silently fall back to the `mem_search` first-window gate. Same NotFound-vs-
    // real-error handling as `.kmt`.
    let blm_bytes: Option<Vec<u8>> = match std::fs::read(&paths.bloom) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            return Err(Error::Io {
                path: paths.bloom.clone(),
                source: e,
            })
        }
    };

    // Compute component offsets (each 4 KiB-aligned, starting after the header).
    let meta_offset = HEADER_SIZE; // first component starts right after the header
    let sa_offset = align_up(meta_offset + meta_bytes.len(), COMPONENT_ALIGN);
    let l1_offset = align_up(sa_offset + sa_bytes.len(), COMPONENT_ALIGN);
    let l2_offset = align_up(l1_offset + l1_bytes.len(), COMPONENT_ALIGN);
    let kmt_offset = align_up(l2_offset + l2_bytes.len(), COMPONENT_ALIGN);
    let kmt_len = kmt_bytes.as_ref().map_or(0, |b| b.len());
    // The `.blm` follows the `.kmt` (or `.l2` when no table is carried). Anchor it
    // on the actual end of the previous component so a `kmt_len == 0` build packs
    // the bloom immediately after `.l2`. Sum with `checked_add` so a (pathologically
    // large) component layout fails closed rather than wrapping `blm_offset` into a
    // malformed blob. (The pre-existing sa/l1/l2/kmt offset sums above keep their
    // unchecked form; only the new `.blm` slot is hardened here.)
    let blm_offset = {
        let end = kmt_offset
            .checked_add(kmt_len)
            .ok_or_else(|| Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: "shm component offsets overflow usize before the .blm component".into(),
            })?;
        align_up(end, COMPONENT_ALIGN)
    };
    let blm_len = blm_bytes.as_ref().map_or(0, |b| b.len());

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
    // `.kmt` component: offset+len, both 0 when absent (backward-compatible —
    // old blobs leave these reserved bytes zero, which reads as "no table").
    if kmt_len > 0 {
        LittleEndian::write_u64(&mut header[88..96], kmt_offset as u64);
        LittleEndian::write_u64(&mut header[96..104], kmt_len as u64);
    }
    // `.blm` component: offset+len, both 0 when absent (backward-compatible —
    // old blobs leave these reserved bytes zero, which reads as "no bloom").
    if blm_len > 0 {
        LittleEndian::write_u64(&mut header[104..112], blm_offset as u64);
        LittleEndian::write_u64(&mut header[112..120], blm_len as u64);
    }
    // bytes [120..4096) remain zero (reserved).

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
    // Optional `.kmt` component.
    if let Some(kmt) = &kmt_bytes {
        let off = write_padding(
            &mut w,
            l2_offset + l2_bytes.len(),
            COMPONENT_ALIGN,
            shm_path,
        )?;
        debug_assert_eq!(off, kmt_offset);
        w.write_all(kmt).map_err(|e| Error::Io {
            path: shm_path.to_path_buf(),
            source: e,
        })?;
    }
    // Optional `.blm` component (follows `.kmt`, or `.l2` when no table carried).
    if let Some(blm) = &blm_bytes {
        // Pad from the actual end of the last written component: `.kmt` if one was
        // carried, otherwise `.l2`.
        let prev_end = match &kmt_bytes {
            Some(kmt) => kmt_offset + kmt.len(),
            None => l2_offset + l2_bytes.len(),
        };
        let off = write_padding(&mut w, prev_end, COMPONENT_ALIGN, shm_path)?;
        debug_assert_eq!(off, blm_offset);
        w.write_all(blm).map_err(|e| Error::Io {
            path: shm_path.to_path_buf(),
            source: e,
        })?;
    }

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
    // Optional `.kmt` (both 0 when absent — old blobs leave these zero).
    let kmt_offset = LittleEndian::read_u64(&blob[88..96]) as usize;
    let kmt_len = LittleEndian::read_u64(&blob[96..104]) as usize;
    // Optional `.blm` (both 0 when absent — old blobs leave these zero).
    // The `.blm` slots are new in this layout; parse them with `try_from` so a
    // crafted u64 that does not fit `usize` is rejected before the layout checks
    // below (an `as usize` cast would truncate it into a spuriously in-range
    // value on a 32-bit target). The meta/sa/l1/l2/kmt slots above predate this
    // and keep their `as usize` form.
    let blm_offset = usize::try_from(LittleEndian::read_u64(&blob[104..112])).map_err(|_| {
        Error::SizeMismatch {
            file: shm_path.to_path_buf(),
            detail: "blm_offset exceeds usize on this platform".into(),
        }
    })?;
    let blm_len = usize::try_from(LittleEndian::read_u64(&blob[112..120])).map_err(|_| {
        Error::SizeMismatch {
            file: shm_path.to_path_buf(),
            detail: "blm_len exceeds usize on this platform".into(),
        }
    })?;

    // Enforce the documented PRMI_SHM_v1 wrapper layout: every component must
    // start after the reserved header, be 4 KiB-aligned, sit within the blob,
    // and follow the previous component without overlap (components are written
    // in meta → sa → l1 → l2 → kmt order; the optional `.kmt` is len 0 when
    // absent and so trivially in range).
    let mut prev_end = HEADER_SIZE;
    for (name, offset, len) in [
        ("meta", meta_offset, meta_len),
        ("sa", sa_offset, sa_len),
        ("l1", l1_offset, l1_len),
        ("l2", l2_offset, l2_len),
        ("kmt", kmt_offset, kmt_len),
        ("blm", blm_offset, blm_len),
    ] {
        // `.kmt` and `.blm` are optional: an absent component is written as offset
        // 0 / len 0 and occupies no bytes, so the strict layout checks (which assume
        // a real, header-following, page-aligned range) do not apply — skip it, but
        // still enforce its offset is 0. A zero-length CORE component (meta/sa/l1/l2)
        // is a malformed wrapper and must be rejected, not silently passed through
        // as an empty slice.
        if (name == "kmt" || name == "blm") && len == 0 {
            if offset != 0 {
                return Err(Error::SizeMismatch {
                    file: shm_path.to_path_buf(),
                    detail: format!("{name} offset {offset} must be 0 when {name} len is 0"),
                });
            }
            continue;
        }
        if len == 0 {
            return Err(Error::SizeMismatch {
                file: shm_path.to_path_buf(),
                detail: format!("{name} component has zero length (malformed blob)"),
            });
        }
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
        kmt_offset,
        kmt_len,
        blm_offset,
        blm_len,
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
