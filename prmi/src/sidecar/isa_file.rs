// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.isa` file format: the inverse suffix array, indexed by reference position.
//!
//! Two layouts share a 24-byte header (byte 17 is the layout `mode`; it was
//! reserved-zero in the original format, so pre-existing files read as dense):
//!
//! - **Dense** (`mode = 0`, full index): `num_entries × 5` packed entries.
//!   Entry at reference position `p` is the SA index `i` such that `sa[i] == p`
//!   — `inv[p] = i` (BWA-MEME's `ref2sa`). The forward `mem_search` `est_hint`
//!   (no-search launch) is `inv[refpos]`, an O(1) refpos→SA-index lookup; the
//!   suffix array's per-row layout (ordered by suffix, not by refpos) cannot
//!   serve that, which is why the inverse SA is a separate refpos-indexed file.
//!   `num_entries == 2·l_pac + 1` (every doubled-coordinate position).
//!
//! - **Sparse** (`mode = 1`, tiered/Design-Z index): `num_entries × 10` entries,
//!   each a `(refpos, rank)` pair (two uint40s) sorted by `refpos`. A dense
//!   refpos-indexed ISA would be genome-scale (~31 GB on hg38) even for a small
//!   on-target keep-set, defeating the tiered index's footprint; the sparse form
//!   stores only the KEPT positions and maps each to its COMPACTED `.sa` rank
//!   (the same rank space the tiered model predicts). Lookup is a binary search
//!   on `refpos` (O(log k)); a `refpos` not in the keep-set returns `None`, and
//!   the consumer falls back to a cold/model launch (byte-identical — the hint
//!   only seeds the search). `num_entries` == the tiered `.sa` entry count.
//!
//! SA indices/ranks share the `.sa` file's 5-byte (uint40) packing — both index
//! the same `[0, 2·l_pac]` doubled-coordinate space, well within 2^40.
//!
//! This sidecar is OPTIONAL (emitted only by `prmi build --with-isa`): for a full
//! index it costs `+5` bytes per SA entry (~+32 GB at hg38), so it is the
//! operator's choice. The same binary runs ISA-accelerated or model-launch-only
//! depending on whether the `.isa` is present (`LearnedIndex::has_isa`).
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
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Size of the `.isa` binary header, in bytes (mirrors the `.sa` header).
pub const ISA_FILE_HEADER_BYTES: usize = 24;

/// Header byte holding the layout mode. Byte 17 was reserved-zero originally, so
/// pre-existing dense `.isa` files (written with all-zero reserved bytes) read
/// back as [`ISA_MODE_DENSE`].
const ISA_MODE_OFFSET: usize = 17;
/// Dense full ISA: `num_entries × 5`, indexed directly by reference position.
const ISA_MODE_DENSE: u8 = 0;
/// Sparse tiered ISA: `num_entries × 10`, `(refpos, rank)` pairs sorted by refpos.
const ISA_MODE_SPARSE: u8 = 1;
/// Width of one sparse entry: a `(refpos, rank)` pair, each uint40.
const SPARSE_ENTRY_BYTES: usize = 2 * BYTES_PER_PACKED_ENTRY;
/// Largest value representable in the 5-byte (uint40) packing shared by the `.sa`
/// and `.isa` files; refpos/rank/entry-count guards reject anything above it.
const UINT40_MAX: u64 = (1u64 << 40) - 1;

/// Validate that `sa` is a permutation of `[0, n)` — distinctness AND range.
///
/// The dense scatter in [`write_isa_file`] writes through raw pointers with no
/// bounds or overlap check, so a duplicate or out-of-range value would be a data
/// race / out-of-bounds write (UB). A 1-bit-per-entry bitset keeps the guard to
/// ~n/8 bytes (~775 MB at hg38) and one O(n) pass — negligible against the
/// multi-GB `.isa` it protects. A violation is an internal bug (`build_gsa`
/// guarantees a permutation), hence [`Error::Internal`].
fn validate_dense_sa_permutation(sa: &[u64], n: usize) -> Result<()> {
    let mut seen = vec![0u64; n.div_ceil(64)];
    for &p in sa {
        let p = usize::try_from(p).map_err(|_| Error::Internal {
            detail: format!("SA value {p} does not fit in usize; SA must be a permutation"),
        })?;
        if p >= n {
            return Err(Error::Internal {
                detail: format!("SA value {p} out of range (n={n}); SA must be a permutation"),
            });
        }
        let (word, bit) = (p / 64, 1u64 << (p % 64));
        if seen[word] & bit != 0 {
            return Err(Error::Internal {
                detail: format!(
                    "SA value {p} appears twice; SA must be a permutation (distinct values)"
                ),
            });
        }
        seen[word] |= bit;
    }
    Ok(())
}

/// Build the inverse suffix array from the SA permutation and write it to `path`.
///
/// `sa` is the suffix array: a permutation of `[0, sa.len())` where `sa[i]` is the
/// reference position of the `i`-th suffix in sorted order. The inverse is
/// `inv[sa[i]] = i`; we materialise it directly into the packed (5-byte) on-disk
/// layout to avoid a second `u64` array (the packed buffer is ~5/8 the size of a
/// `Vec<u64>` inverse).
pub fn write_isa_file(path: &Path, sa: &[u64]) -> Result<()> {
    let n = sa.len();
    // Ranks (the stored SA indices `0..n`) are packed as uint40, so the largest
    // rank `n - 1` must fit in 5 bytes; `n > 2^40` would silently truncate. This
    // is a writer/SA-integrity invariant on internally generated input, hence
    // `Error::Internal` (mirrors the tiered writer's uint40 guard).
    if n as u64 > UINT40_MAX + 1 {
        return Err(Error::Internal {
            detail: format!(
                "dense ISA rank count {n} exceeds uint40 capacity {}",
                UINT40_MAX + 1
            ),
        });
    }
    // Validate the permutation BEFORE any file I/O so a malformed `sa` fails
    // without truncating/creating the output (the sparse writer validates up
    // front too); the raw-pointer scatter below then assumes a valid permutation.
    validate_dense_sa_permutation(sa, n)?;
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
    mmap[ISA_MODE_OFFSET] = ISA_MODE_DENSE;
    // remaining reserved header bytes stay zero (from `set_len`).

    // Pack inv[p] = i directly: for each sorted index i at reference position
    // p = sa[i], store the 5-byte SA index i at body offset p*5.
    //
    // Parallel scatter. `sa` is a permutation of [0, n) (validated above), which
    // is the safety-critical invariant here: distinctness (not just the range
    // bound) is what makes the concurrent raw-pointer writes race-free, since
    // every `p = sa[i]` targets a disjoint, non-overlapping 5-byte range. A big
    // win at genome scale, where the serial random-write over a multi-GB mmap is
    // the dominant `--with-isa` build cost.
    let body = &mut mmap[ISA_FILE_HEADER_BYTES..];
    let body_addr = body.as_mut_ptr() as usize;
    sa.par_iter().enumerate().for_each(|(i, &p)| {
        let p = p as usize;
        // Runtime (not debug-only) bound: the unsafe write below has no slice
        // bounds check, so a malformed (non-permutation) SA must fault loudly
        // here rather than write out of bounds in release. The serial version
        // this replaced got that panic for free from slice indexing; one compare
        // per entry is negligible against the write it guards.
        assert!(
            p < n,
            "SA value {p} out of range (n={n}); SA must be a permutation"
        );
        let off = p * BYTES_PER_PACKED_ENTRY;
        let packed = pack_position(i as u64);
        // SAFETY: `p < n` (SA permutation) so `[off, off + 5)` lies within the
        // `n * 5`-byte body; offsets are distinct across `i`, so no two threads
        // write overlapping bytes. `body_addr` aliases the live `body` mmap for
        // the duration of this parallel region (joined before `body` is used
        // again or flushed).
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                (body_addr as *mut u8).add(off),
                BYTES_PER_PACKED_ENTRY,
            );
        }
    });

    mmap.flush().map_err(io)?;
    Ok(())
}

/// Write a SPARSE tiered inverse suffix array: a list of `(refpos, rank)` pairs
/// sorted by `refpos`, mapping each KEPT doubled-coordinate position to its
/// COMPACTED rank in the tiered `.sa`. Used by `prmi build --keep-bed
/// --with-isa`; non-kept positions are absent (lookup returns `None`).
///
/// `pairs` MUST be sorted ascending by `refpos` (the reader binary-searches it)
/// and `rank` values MUST be the compacted `.sa` indices (0..pairs.len()) the
/// tiered model predicts. Both fields are uint40, sharing the `.sa` packing.
pub fn write_tiered_isa_file(path: &Path, pairs: &[(u64, u64)]) -> Result<()> {
    // Enforce the reader's invariants in ALL builds (a debug-only assert would let
    // a release build silently produce corrupt lookups): pairs sorted ascending by
    // refpos (the reader binary-searches), and both fields within the 5-byte
    // (uint40) packing — `pack_position` would otherwise truncate silently.
    let n = pairs.len();
    let mut prev: Option<u64> = None;
    for &(refpos, rank) in pairs {
        // These are writer-API / SA-integrity invariants on internally generated
        // pairs (compacted SA -> (refpos, rank)), not user input, so a violation
        // is an internal bug -> `Error::Internal` (matching the dense writer, which
        // treats its permutation invariant as a hard internal contract).
        if refpos > UINT40_MAX || rank > UINT40_MAX {
            return Err(Error::Internal {
                detail: format!(
                    "tiered ISA entry (refpos={refpos}, rank={rank}) exceeds uint40 max {UINT40_MAX}"
                ),
            });
        }
        // `rank` is handed back as the launch hint into the compacted `.sa`, so a
        // rank past the compacted length would serialize a loadable `.isa` that
        // points outside the `.sa` — reject it (uint40 alone is not enough).
        if rank >= n as u64 {
            return Err(Error::Internal {
                detail: format!("tiered ISA rank {rank} is outside compacted SA range [0, {n})"),
            });
        }
        if prev.is_some_and(|p| refpos <= p) {
            return Err(Error::Internal {
                detail: "write_tiered_isa_file requires pairs STRICTLY ascending by refpos \
                         (duplicate refpos would corrupt the binary-search lookup)"
                    .into(),
            });
        }
        prev = Some(refpos);
    }
    let io = |e: std::io::Error| Error::Io {
        path: path.to_path_buf(),
        source: e,
    };
    let total = ISA_FILE_HEADER_BYTES + n * SPARSE_ENTRY_BYTES;
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(io)?;
    f.set_len(total as u64).map_err(io)?;
    // SAFETY: as in `write_isa_file` — sole accessor, sized to `total`, flushed
    // and unmapped before return.
    let mut mmap = unsafe { MmapMut::map_mut(&f) }.map_err(io)?;
    LittleEndian::write_u32(&mut mmap[0..4], ISA_MAGIC);
    LittleEndian::write_u32(&mut mmap[4..8], FORMAT_VERSION);
    LittleEndian::write_u64(&mut mmap[8..16], n as u64);
    mmap[16] = BYTES_PER_PACKED_ENTRY as u8;
    mmap[ISA_MODE_OFFSET] = ISA_MODE_SPARSE;
    let body = &mut mmap[ISA_FILE_HEADER_BYTES..];
    for (i, &(refpos, rank)) in pairs.iter().enumerate() {
        let off = i * SPARSE_ENTRY_BYTES;
        body[off..off + BYTES_PER_PACKED_ENTRY].copy_from_slice(&pack_position(refpos));
        body[off + BYTES_PER_PACKED_ENTRY..off + SPARSE_ENTRY_BYTES]
            .copy_from_slice(&pack_position(rank));
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
    /// Layout: [`ISA_MODE_DENSE`] (direct refpos index) or [`ISA_MODE_SPARSE`]
    /// (binary-searched `(refpos, rank)` pairs).
    mode: u8,
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
        let (num_entries, mode) = validate_isa_header(&mmap, path)?;
        let data_ptr = unsafe { mmap.as_ptr().add(ISA_FILE_HEADER_BYTES) };
        Ok(Self {
            _file: f,
            _mmap: mmap,
            data_ptr,
            num_entries,
            mode,
        })
    }

    /// Number of entries: for a dense ISA the reference-position count
    /// (`2·l_pac + 1`); for a sparse tiered ISA the kept-position count (== the
    /// tiered `.sa` entry count). The loader checks this against `sa_num`.
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// Look up the SA index (launch hint) for reference position `refpos`.
    ///
    /// Dense: `refpos < num_entries` ? the directly-indexed inverse SA : `None`.
    /// Sparse (tiered): binary-search the `(refpos, rank)` pairs; `Some(rank)` if
    /// `refpos` is in the keep-set, else `None`. A `None` is safe — the consumer
    /// falls back to a cold/model launch, byte-identical to a hinted one.
    #[inline]
    pub fn lookup(&self, refpos: u64) -> Option<u64> {
        match self.mode {
            ISA_MODE_SPARSE => self.sparse_lookup(refpos),
            _ => (refpos < self.num_entries).then(|| self.dense_at(refpos)),
        }
    }

    /// Dense direct index: the SA index stored at reference position `refpos`.
    /// `refpos` must be `< num_entries` and the ISA must be dense.
    #[inline]
    fn dense_at(&self, refpos: u64) -> u64 {
        debug_assert_eq!(self.mode, ISA_MODE_DENSE);
        assert!(
            refpos < self.num_entries,
            "dense_at refpos {refpos} out of range (len={})",
            self.num_entries
        );
        let off = (refpos as usize) * BYTES_PER_PACKED_ENTRY;
        // SAFETY: dense body is `num_entries * 5` valid bytes; the assert above
        // guarantees `refpos < num_entries`, so `off + 5 <= num_entries * 5`.
        let bytes: &[u8; BYTES_PER_PACKED_ENTRY] =
            unsafe { &*(self.data_ptr.add(off) as *const [u8; BYTES_PER_PACKED_ENTRY]) };
        unpack_position(bytes)
    }

    /// Read the `refpos` field of sparse entry `i` (entries are sorted by it).
    #[inline]
    fn sparse_refpos(&self, i: u64) -> u64 {
        assert!(
            i < self.num_entries,
            "sparse_refpos index {i} out of range (len={})",
            self.num_entries
        );
        let off = (i as usize) * SPARSE_ENTRY_BYTES;
        // SAFETY: sparse body is `num_entries * 10` valid bytes; the assert above
        // guarantees `i < num_entries`, so `off + 5 <= num_entries * 10`.
        let bytes: &[u8; BYTES_PER_PACKED_ENTRY] =
            unsafe { &*(self.data_ptr.add(off) as *const [u8; BYTES_PER_PACKED_ENTRY]) };
        unpack_position(bytes)
    }

    /// Read the `rank` field of sparse entry `i`.
    #[inline]
    fn sparse_rank(&self, i: u64) -> u64 {
        assert!(
            i < self.num_entries,
            "sparse_rank index {i} out of range (len={})",
            self.num_entries
        );
        let off = (i as usize) * SPARSE_ENTRY_BYTES + BYTES_PER_PACKED_ENTRY;
        // SAFETY: sparse body is `num_entries * 10` valid bytes; the assert above
        // guarantees `i < num_entries`, so `off + 5 <= num_entries * 10`.
        let bytes: &[u8; BYTES_PER_PACKED_ENTRY] =
            unsafe { &*(self.data_ptr.add(off) as *const [u8; BYTES_PER_PACKED_ENTRY]) };
        unpack_position(bytes)
    }

    /// Binary-search the sorted `(refpos, rank)` pairs for `refpos`.
    fn sparse_lookup(&self, refpos: u64) -> Option<u64> {
        let (mut lo, mut hi) = (0u64, self.num_entries);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_ref = self.sparse_refpos(mid);
            match mid_ref.cmp(&refpos) {
                std::cmp::Ordering::Equal => return Some(self.sparse_rank(mid)),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }
}

/// Validate an `.isa` header. Returns `(num_entries, mode)` on success.
fn validate_isa_header(data: &[u8], path: &Path) -> Result<(u64, u8)> {
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
    let mode = data[ISA_MODE_OFFSET];
    let entry_bytes = match mode {
        ISA_MODE_DENSE => BYTES_PER_PACKED_ENTRY,
        ISA_MODE_SPARSE => SPARSE_ENTRY_BYTES,
        other => {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!(".isa layout mode {other} is not supported"),
            })
        }
    };
    // Compute the expected size with checked arithmetic: a crafted large
    // num_entries must fail validation rather than wrap expected_len around and
    // let a truncated file pass.
    let n = usize::try_from(num_entries).map_err(|_| Error::SizeMismatch {
        file: path.to_path_buf(),
        detail: format!("num_entries too large: {num_entries}"),
    })?;
    let expected_len = n
        .checked_mul(entry_bytes)
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
    Ok((num_entries, mode))
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
            assert_eq!(
                r.lookup(p),
                Some(i as u64),
                "inv[sa[{i}]={p}] should be {i}"
            );
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
            assert_eq!(r.lookup(p), Some(n - 1 - p), "mismatch at refpos {p}");
        }
    }

    /// Sparse tiered round-trip: kept refpos map to their compacted ranks; a
    /// refpos absent from the keep-set returns `None`.
    #[test]
    fn tiered_isa_sparse_round_trips_and_misses_return_none() {
        // (refpos, compacted_rank) pairs, sorted by refpos. Gaps (1,4,6,...)
        // model dropped (non-kept) positions.
        let pairs: Vec<(u64, u64)> = vec![(0, 3), (2, 0), (5, 1), (9, 2), (12, 4)];
        let mut sorted = pairs.clone();
        sorted.sort_by_key(|&(r, _)| r);
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.tiered.isa");
        write_tiered_isa_file(&path, &sorted).unwrap();

        let r = IsaFileReader::open(&path).unwrap();
        assert_eq!(r.num_entries(), sorted.len() as u64);
        for &(refpos, rank) in &sorted {
            assert_eq!(
                r.lookup(refpos),
                Some(rank),
                "kept refpos {refpos} -> {rank}"
            );
        }
        // Non-kept positions: None (consumer falls back to a cold launch).
        for miss in [1u64, 3, 4, 6, 7, 8, 10, 11, 13, 100] {
            assert_eq!(r.lookup(miss), None, "non-kept refpos {miss} must miss");
        }
        // Exact file size: header + n * 10.
        let size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(
            size,
            (ISA_FILE_HEADER_BYTES + sorted.len() * SPARSE_ENTRY_BYTES) as u64
        );
    }

    #[test]
    fn tiered_isa_writer_contract_violations_are_internal() {
        // The pairs are generated internally by the tiered build, so a violated
        // writer invariant is an internal bug, not user input -> Error::Internal.
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.tiered.isa");
        // Non-strictly-ascending refpos (duplicate) corrupts the binary search.
        assert!(matches!(
            write_tiered_isa_file(&path, &[(0, 0), (0, 1)]),
            Err(Error::Internal { .. })
        ));
        // refpos beyond the uint40 packing width.
        let over = (1u64 << 40) + 1;
        assert!(matches!(
            write_tiered_isa_file(&path, &[(over, 0)]),
            Err(Error::Internal { .. })
        ));
        // rank within uint40 but past the compacted SA length (n=1 here) would
        // serialize a hint pointing outside the `.sa`.
        assert!(matches!(
            write_tiered_isa_file(&path, &[(0, 5)]),
            Err(Error::Internal { .. })
        ));
    }

    #[test]
    fn write_isa_file_rejects_duplicate_sa() {
        // The dense writer's parallel scatter is race-free only if `sa` is a
        // permutation; the all-builds distinctness guard must reject a duplicate
        // (which would otherwise let two threads write the same offset — UB)
        // before any write. `[0, 1, 1]` is in-range (every value < 3) but not a
        // permutation.
        let dir = tempdir().unwrap();
        let path = dir.path().join("dup.isa");
        assert!(matches!(
            write_isa_file(&path, &[0, 1, 1]),
            Err(Error::Internal { .. })
        ));
        // An out-of-range value (>= n) is likewise rejected, not written OOB.
        assert!(matches!(
            write_isa_file(&path, &[0, 5, 1]),
            Err(Error::Internal { .. })
        ));
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
