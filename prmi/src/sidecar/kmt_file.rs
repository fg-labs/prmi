// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.kmt` k-mer table file format: a forward-spectrum shallow-band accelerator.
//!
//! Layout: a 56-byte header (`magic`, `version`, `k`, reserved, `sa_num`,
//! `ref_digest[32]`) followed, for each prefix length `m = 1..=k`, by the
//! length-`m` lower-bound array (`4^m` little-endian `u64`) then the matching
//! upper-bound array (`4^m` `u64`). `lo[m-1][w]` / `hi[m-1][w]` are the SA
//! bounds of the length-`m` mer with lex index `w`.
//!
//! The stored bounds are SA indices fed into the SA reader, so a `.kmt` paired
//! with the wrong `.sa` would drive out-of-range probes or silently-wrong
//! SMEMs. Two fields bind a table to its reference: `sa_num` (the suffix-array
//! size) and `ref_digest` (the reference content hash — `pac_sha256`, or the
//! FASTA sha). `sa_num` alone is too weak (two references of equal length share
//! it), so `ref_digest` is the authoritative binding the open path checks. The
//! table is a pure accelerator, so open treats ANY `.kmt` problem (corrupt, or
//! either binding mismatched) as "no table" and falls back to the full search.
//! The reader is a read-only mmap (the same single `unsafe` island as
//! `sa_file` / `model_file`).

#![allow(unsafe_code)]

use crate::error::{Error, Result};
use crate::sidecar::magic::{FORMAT_VERSION, KMT_MAGIC};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Size of the binary header at the start of a `.kmt` file, in bytes.
/// `magic(4) | version(4) | k(4) | reserved(4) | sa_num(8) | ref_digest(32)`.
pub const KMT_FILE_HEADER_BYTES: usize = 56;

/// Parse a 64-char hex string (e.g. a sha256) into 32 bytes; `None` if the
/// input is not exactly 64 hex digits. Used to bind a `.kmt` to its reference.
pub(crate) fn hex32(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (nib(b[2 * i])? << 4) | nib(b[2 * i + 1])?;
    }
    Some(out)
}

/// Number of entries (`4^m`) in the length-`m` arrays.
#[inline]
fn level_len(m: u32) -> usize {
    1usize << (2 * m)
}

/// Total file size (header + per-length lo/hi arrays) for a table of order `k`.
fn kmt_file_size(k: u32) -> usize {
    let mut bytes = KMT_FILE_HEADER_BYTES;
    for m in 1..=k {
        // 4^m entries × 8 bytes × 2 arrays (lo, hi).
        bytes += level_len(m) * 8 * 2;
    }
    bytes
}

/// Byte offsets (within the file) of `lo[m-1]` and `hi[m-1]` for `m = 1..=k`.
fn level_offsets(k: u32) -> (Vec<usize>, Vec<usize>) {
    let mut lo_off = Vec::with_capacity(k as usize);
    let mut hi_off = Vec::with_capacity(k as usize);
    let mut off = KMT_FILE_HEADER_BYTES;
    for m in 1..=k {
        let count_bytes = level_len(m) * 8;
        lo_off.push(off);
        off += count_bytes;
        hi_off.push(off);
        off += count_bytes;
    }
    (lo_off, hi_off)
}

/// Writes a `.kmt` file in one pass from the per-length lower/upper-bound
/// arrays. `lo`/`hi` must each have exactly `k` entries, with `lo[m-1]` and
/// `hi[m-1]` of length `4^m` for `m = 1..=k`.
pub struct KmtFileWriter;

impl KmtFileWriter {
    /// Write the table to `path`, overwriting any existing file. `sa_num` and
    /// `ref_digest` bind the table to its `.sa`/reference for open-time
    /// cross-validation.
    pub fn write(
        path: &Path,
        k: u32,
        sa_num: u64,
        ref_digest: &[u8; 32],
        lo: &[Vec<u64>],
        hi: &[Vec<u64>],
    ) -> Result<()> {
        let bad = |detail: String| Error::Internal { detail };
        if !(1..=16).contains(&k) {
            return Err(bad(format!("kmt k={k} out of range 1..=16")));
        }
        if lo.len() != k as usize || hi.len() != k as usize {
            return Err(bad(format!(
                "kmt expects {k} levels, got lo={} hi={}",
                lo.len(),
                hi.len()
            )));
        }
        for m in 1..=k as usize {
            let expect = level_len(m as u32);
            if lo[m - 1].len() != expect || hi[m - 1].len() != expect {
                return Err(bad(format!(
                    "kmt level m={m} must be 4^{m}={expect}, got lo={} hi={}",
                    lo[m - 1].len(),
                    hi[m - 1].len()
                )));
            }
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
        let io = |e: std::io::Error| Error::Io {
            path: path.to_path_buf(),
            source: e,
        };

        let mut header = [0u8; KMT_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], KMT_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u32(&mut header[8..12], k);
        // header[12..16] reserved (zero).
        LittleEndian::write_u64(&mut header[16..24], sa_num);
        header[24..56].copy_from_slice(ref_digest);
        w.write_all(&header).map_err(io)?;

        // Serialise each array in 8 KiB-bounded chunks (avoids both a giant temp
        // buffer and millions of one-u64 `write_all` calls).
        let mut buf: Vec<u8> = Vec::with_capacity(8192 * 8);
        for m in 1..=k as usize {
            for arr in [&lo[m - 1], &hi[m - 1]] {
                for chunk in arr.chunks(8192) {
                    buf.clear();
                    for &v in chunk {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    w.write_all(&buf).map_err(io)?;
                }
            }
        }
        w.flush().map_err(io)?;
        Ok(())
    }
}

/// Read-only access to per-length k-mer SA bounds, abstracting over the
/// in-memory `KmerTable` (build/test) and the mmap-backed [`KmtFileReader`]
/// (production). `forward_spectrum_tabled` is generic over this so the same
/// shallow-band logic serves both.
pub trait KmerBounds {
    /// Max prefix length covered.
    fn k(&self) -> u32;
    /// SA lower bound of the length-`m` mer with lex index `w` (`1<=m<=k`).
    fn lo(&self, m: usize, w: u64) -> u64;
    /// SA upper bound of the length-`m` mer with lex index `w`.
    fn hi(&self, m: usize, w: u64) -> u64;
}

impl KmerBounds for KmtFileReader {
    #[inline]
    fn k(&self) -> u32 {
        KmtFileReader::k(self)
    }
    #[inline]
    fn lo(&self, m: usize, w: u64) -> u64 {
        KmtFileReader::lo(self, m, w)
    }
    #[inline]
    fn hi(&self, m: usize, w: u64) -> u64 {
        KmtFileReader::hi(self, m, w)
    }
}

/// mmap-backed reader for a `.kmt` file. Exposes the per-length lower/upper SA
/// bounds via zero-copy reads.
pub struct KmtFileReader {
    /// Keeps the file open for file-backed instances; `None` for shm-backed.
    _file: Option<File>,
    /// Owned mmap for file-backed instances; `None` for shm-backed. Data is read
    /// via `data_ptr`; this field exists solely to extend the mmap's lifetime.
    _mmap: Option<Mmap>,
    /// Shared shm blob mmap for shm-backed instances; `None` for file-backed.
    _shm_mmap: Option<Arc<Mmap>>,
    /// Pointer to the file bytes (file mmap or shm sub-slice). Valid for `self`.
    data_ptr: *const u8,
    /// Total length of the mapped region in bytes (backstop for accessor reads).
    data_len: usize,
    /// Max prefix length covered.
    k: u32,
    /// Suffix-array size the bounds index into (cross-validated at open).
    sa_num: u64,
    /// Reference content hash the table was built against (cross-validated).
    ref_digest: [u8; 32],
    /// Byte offset of `lo[m-1]` / `hi[m-1]` within the file (`m = 1..=k`).
    lo_off: Vec<usize>,
    hi_off: Vec<usize>,
}

// SAFETY: same rationale as SaFileReader / ModelFileReader — `data_ptr` is
// read-only and kept alive by an owned field in this struct.
unsafe impl Send for KmtFileReader {}
unsafe impl Sync for KmtFileReader {}

impl std::fmt::Debug for KmtFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KmtFileReader")
            .field("k", &self.k)
            .field("sa_num", &self.sa_num)
            .finish()
    }
}

impl KmtFileReader {
    /// Open and mmap a `.kmt` file, validating its header and size.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: read-only mmap; the backing File is kept alive in `_file`;
        // concurrent writers are not supported.
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let data_len = mmap.len();
        let (k, sa_num, ref_digest, lo_off, hi_off) = validate_kmt_header(&mmap, path, data_len)?;
        let data_ptr = mmap.as_ptr();
        Ok(Self {
            _file: Some(f),
            _mmap: Some(mmap),
            _shm_mmap: None,
            data_ptr,
            data_len,
            k,
            sa_num,
            ref_digest,
            lo_off,
            hi_off,
        })
    }

    /// Construct a reader backed by a sub-slice of a shm blob mmap. Used by
    /// [`LearnedIndex::open_shm`](crate::index::LearnedIndex::open_shm) to load
    /// the `.kmt` component carried inside an shm blob.
    pub(crate) fn from_shm_slice(shm_mmap: Arc<Mmap>, offset: usize, len: usize) -> Result<Self> {
        let fake_path = PathBuf::from("<shm>");
        let end = offset.checked_add(len).ok_or_else(|| Error::SizeMismatch {
            file: fake_path.clone(),
            detail: "kmt shm slice offset+len overflow".into(),
        })?;
        if end > shm_mmap.len() {
            return Err(Error::SizeMismatch {
                file: fake_path,
                detail: format!(
                    "kmt shm slice [{offset}, {end}) exceeds blob {}",
                    shm_mmap.len()
                ),
            });
        }
        let slice = &shm_mmap[offset..end];
        let (k, sa_num, ref_digest, lo_off, hi_off) = validate_kmt_header(slice, &fake_path, len)?;
        let data_ptr = slice.as_ptr();
        Ok(Self {
            _file: None,
            _mmap: None,
            _shm_mmap: Some(shm_mmap),
            data_ptr,
            data_len: len,
            k,
            sa_num,
            ref_digest,
            lo_off,
            hi_off,
        })
    }

    /// Max prefix length covered by the table.
    #[inline]
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Suffix-array size the bounds index into (for open-time cross-validation).
    #[inline]
    pub fn sa_num(&self) -> u64 {
        self.sa_num
    }

    /// Reference content hash the table was built against (for cross-validation
    /// against `.meta`'s `pac_sha256` / ref sha).
    #[inline]
    pub fn ref_digest(&self) -> &[u8; 32] {
        &self.ref_digest
    }

    /// Read a `u64` at byte offset `off`, asserting it is fully in bounds.
    #[inline]
    fn read_u64(&self, off: usize) -> u64 {
        assert!(
            off + 8 <= self.data_len,
            "kmt read at {off}+8 exceeds {} bytes",
            self.data_len
        );
        // SAFETY: bounds asserted above; `data_ptr` is kept alive by an owned
        // field for the lifetime of `self`.
        unsafe { LittleEndian::read_u64(std::slice::from_raw_parts(self.data_ptr.add(off), 8)) }
    }

    /// SA lower bound of the length-`m` mer with lex index `w`
    /// (`1 <= m <= k`, `w < 4^m`).
    #[inline]
    pub fn lo(&self, m: usize, w: u64) -> u64 {
        debug_assert!(m >= 1 && m as u32 <= self.k && (w as usize) < level_len(m as u32));
        self.read_u64(self.lo_off[m - 1] + (w as usize) * 8)
    }

    /// SA upper bound of the length-`m` mer with lex index `w`.
    #[inline]
    pub fn hi(&self, m: usize, w: u64) -> u64 {
        debug_assert!(m >= 1 && m as u32 <= self.k && (w as usize) < level_len(m as u32));
        self.read_u64(self.hi_off[m - 1] + (w as usize) * 8)
    }
}

/// Validate a `.kmt` header from a byte slice. Returns
/// `(k, sa_num, ref_digest, lo_off, hi_off)`.
#[allow(clippy::type_complexity)]
fn validate_kmt_header(
    data: &[u8],
    path: &Path,
    declared_len: usize,
) -> Result<(u32, u64, [u8; 32], Vec<usize>, Vec<usize>)> {
    if data.len() < KMT_FILE_HEADER_BYTES {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("too small ({} bytes) for kmt header", data.len()),
        });
    }
    let magic = LittleEndian::read_u32(&data[0..4]);
    if magic != KMT_MAGIC {
        return Err(Error::BadMagic {
            file: path.to_path_buf(),
            found: format!("{:#010x}", magic),
            expected: format!("{:#010x}", KMT_MAGIC),
        });
    }
    let version = LittleEndian::read_u32(&data[4..8]);
    if version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let k = LittleEndian::read_u32(&data[8..12]);
    if !(1..=16).contains(&k) {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("kmt k={k} out of range 1..=16"),
        });
    }
    let sa_num = LittleEndian::read_u64(&data[16..24]);
    let mut ref_digest = [0u8; 32];
    ref_digest.copy_from_slice(&data[24..56]);
    let expected_size = kmt_file_size(k);
    if declared_len != expected_size {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("kmt is {declared_len} bytes, expected {expected_size} for k={k}"),
        });
    }
    let (lo_off, hi_off) = level_offsets(k);
    Ok((k, sa_num, ref_digest, lo_off, hi_off))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIGEST: [u8; 32] = [0x5a; 32];

    #[test]
    fn hex32_parses_and_rejects() {
        let s: String = (0..32).map(|i| format!("{:02x}", i)).collect();
        let d = hex32(&s).unwrap();
        assert_eq!(d[0], 0x00);
        assert_eq!(d[31], 0x1f);
        assert!(hex32("short").is_none());
        assert!(hex32(&"zz".repeat(32)).is_none());
    }

    /// Synthetic per-length lo/hi arrays for `k`; deterministic, distinct values
    /// so a wrong offset/byte-order is caught.
    fn synth(k: u32) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for m in 1..=k {
            let n = level_len(m) as u64;
            let lom: Vec<u64> = (0..n).map(|w| w * 1000 + m as u64).collect();
            let him: Vec<u64> = (0..n).map(|w| w * 1000 + m as u64 + 7).collect();
            lo.push(lom);
            hi.push(him);
        }
        (lo, hi)
    }

    fn write_synth(dir: &std::path::Path, name: &str, k: u32, sa_num: u64) -> std::path::PathBuf {
        let (lo, hi) = synth(k);
        let path = dir.join(name);
        KmtFileWriter::write(&path, k, sa_num, &TEST_DIGEST, &lo, &hi).unwrap();
        path
    }

    #[test]
    fn kmt_size_layout_is_locked() {
        // Header(24) + sum_{m=1}^{12} 2·4^m·8.
        assert_eq!(kmt_file_size(1), 56 + 2 * 4 * 8);
        assert_eq!(kmt_file_size(12), 357_913_976);
    }

    #[test]
    fn kmt_round_trips_k1_and_k5() {
        let dir = tempfile::tempdir().unwrap();
        for &k in &[1u32, 5] {
            let (lo, hi) = synth(k);
            let path = dir.path().join(format!("t{k}.kmt"));
            KmtFileWriter::write(&path, k, 123_456_789, &TEST_DIGEST, &lo, &hi).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len() as usize,
                kmt_file_size(k)
            );
            let r = KmtFileReader::open(&path).unwrap();
            assert_eq!(r.k(), k);
            assert_eq!(r.sa_num(), 123_456_789);
            assert_eq!(r.ref_digest(), &TEST_DIGEST);
            for m in 1..=k as usize {
                for w in 0..level_len(m as u32) as u64 {
                    assert_eq!(r.lo(m, w), lo[m - 1][w as usize], "lo m={m} w={w}");
                    assert_eq!(r.hi(m, w), hi[m - 1][w as usize], "hi m={m} w={w}");
                }
            }
        }
    }

    #[test]
    fn kmt_shm_slice_round_trips() {
        // Embed the file bytes at a non-zero offset in a larger blob, mmap it,
        // and read via from_shm_slice (the shm path).
        let dir = tempfile::tempdir().unwrap();
        let path = write_synth(dir.path(), "s.kmt", 4, 999);
        let body = std::fs::read(&path).unwrap();
        let pad = 17usize;
        let mut blob = vec![0xABu8; pad];
        blob.extend_from_slice(&body);
        let blob_path = dir.path().join("blob.bin");
        std::fs::write(&blob_path, &blob).unwrap();
        let f = File::open(&blob_path).unwrap();
        let mmap = unsafe { Mmap::map(&f) }.unwrap();
        let r = KmtFileReader::from_shm_slice(Arc::new(mmap), pad, body.len()).unwrap();
        let (lo, _hi) = synth(4);
        assert_eq!(r.k(), 4);
        assert_eq!(r.sa_num(), 999);
        assert_eq!(r.lo(2, 5), lo[1][5]);
        // Out-of-range slice is rejected, not a panic.
        let f2 = File::open(&blob_path).unwrap();
        let mmap2 = unsafe { Mmap::map(&f2) }.unwrap();
        assert!(KmtFileReader::from_shm_slice(Arc::new(mmap2), pad, body.len() + 1).is_err());
    }

    #[test]
    fn kmt_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.kmt");
        std::fs::write(&path, vec![0u8; KMT_FILE_HEADER_BYTES + 64]).unwrap();
        assert!(KmtFileReader::open(&path).is_err());
    }

    #[test]
    fn kmt_rejects_wrong_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synth(dir.path(), "v.kmt", 3, 1);
        // Corrupt the version field (bytes 4..8).
        let mut bytes = std::fs::read(&path).unwrap();
        LittleEndian::write_u32(&mut bytes[4..8], FORMAT_VERSION + 99);
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            KmtFileReader::open(&path),
            Err(Error::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn kmt_rejects_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synth(dir.path(), "trunc.kmt", 3, 1);
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(kmt_file_size(3) as u64 - 1).unwrap();
        assert!(KmtFileReader::open(&path).is_err());
    }

    #[test]
    fn kmt_writer_rejects_bad_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.kmt");
        // Wrong number of levels.
        let (lo, hi) = synth(2);
        assert!(KmtFileWriter::write(&path, 3, 1, &TEST_DIGEST, &lo, &hi).is_err());
        // Wrong array length within a level.
        let (mut lo2, hi2) = synth(2);
        lo2[0].pop();
        assert!(KmtFileWriter::write(&path, 2, 1, &TEST_DIGEST, &lo2, &hi2).is_err());
    }
}
