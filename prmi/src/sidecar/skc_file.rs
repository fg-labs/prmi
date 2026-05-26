// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.skc` file format: suffix-key-cache companion file for `suffix_key_cache`
//! memory mode.
//!
//! The `.skc` file caches 32-mer keys for a subset of SA positions — typically
//! the top-N most-queried ones — allowing `smem_range` to skip per-candidate
//! `read_unpacked_window + tokenize_32mer` calls for cache-hit entries.
//!
//! # Binary layout
//!
//! ```text
//! Header (16 bytes):
//!   bytes  0..4  : u32  magic = SKC_MAGIC (0x5043_4B53, "SKCP" in LE)
//!   bytes  4..8  : u32  format_version = 1
//!   bytes  8..16 : u64  cache_size (number of (sa_index, key) pairs)
//!
//! Body (cache_size × 16 bytes):
//!   bytes  0..8  : u64  sa_index (LE)
//!   bytes  8..16 : u64  key      (LE u64, same 32-mer encoding as the SA keys)
//! ```
//!
//! Entries are stored in **ascending `sa_index` order**. The reader builds an
//! in-memory `HashMap<u64, u64>` (sa_index → key) at open time for O(1) lookup.
//!
//! # Safety
//!
//! The reader reads the full body into a `HashMap` at open time — no mmap needed
//! for a file that is at most a few hundred megabytes for typical cache sizes.

use crate::error::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Binary magic for the `.skc` file header: `"SKCP"` = `0x50434B53` on disk
/// (little-endian).
pub const SKC_MAGIC: u32 = 0x5043_4B53; // "SKCP" in LE
/// Format version for the `.skc` binary layout.
pub const SKC_FORMAT_VERSION: u32 = 1;
/// Size of the `.skc` binary header in bytes.
pub const SKC_HEADER_BYTES: usize = 16;
/// Bytes per entry in the body: 8-byte sa_index + 8-byte key.
pub const SKC_BYTES_PER_ENTRY: usize = 16;

/// Streaming writer for the `.skc` suffix-key-cache file.
pub struct SkcFileWriter {
    path: PathBuf,
    inner: BufWriter<File>,
    expected: u64,
    written: u64,
    /// Last `sa_index` written, used to enforce the ascending-order invariant
    /// documented on `write_entry`. `None` until the first entry is written.
    last_sa_index: Option<u64>,
}

impl SkcFileWriter {
    /// Create a new `.skc` file at `path` with room for `cache_size` entries.
    pub fn create(path: &Path, cache_size: u64) -> Result<Self> {
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
        let mut header = [0u8; SKC_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], SKC_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], SKC_FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], cache_size);
        w.write_all(&header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            inner: w,
            expected: cache_size,
            written: 0,
            last_sa_index: None,
        })
    }

    /// Append one `(sa_index, key)` pair. Pairs must be written in **strictly
    /// ascending `sa_index` order**; the writer rejects out-of-order or
    /// duplicate keys with `Err(Error::Internal)`. The reader only requires
    /// a valid header and body count, but the on-disk ordering invariant is
    /// part of the format spec (`docs/sidecar-format.md §2a`).
    pub fn write_entry(&mut self, sa_index: u64, key: u64) -> Result<()> {
        // Fail fast on overrun: never persist more body bytes than the header
        // declared. Without this, an extra write is only caught in `finish`,
        // after the bytes are already on disk.
        if self.written >= self.expected {
            return Err(Error::SizeMismatch {
                file: self.path.clone(),
                detail: format!(
                    "attempted to write entry {} but header cache_size is {}",
                    self.written + 1,
                    self.expected
                ),
            });
        }
        if let Some(prev) = self.last_sa_index {
            if sa_index <= prev {
                return Err(Error::Internal {
                    detail: format!(
                        "SkcFileWriter: sa_index={sa_index} must be strictly greater than the previous sa_index={prev} (entries must be written in ascending order)"
                    ),
                });
            }
        }
        let mut buf = [0u8; SKC_BYTES_PER_ENTRY];
        LittleEndian::write_u64(&mut buf[0..8], sa_index);
        LittleEndian::write_u64(&mut buf[8..16], key);
        self.inner.write_all(&buf).map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        self.last_sa_index = Some(sa_index);
        self.written += 1;
        Ok(())
    }

    /// Flush and close, verifying the expected entry count was written.
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

/// Reader for the `.skc` suffix-key-cache file.
///
/// Reads the entire file into a `HashMap<sa_index, key>` at open time so
/// that `lookup_key` is O(1) and the hot path in `smem_range` never touches
/// disk after startup.
pub struct SkcFileReader {
    /// Number of entries as declared in the file header.
    cache_size: u64,
    /// In-memory map: SA index → stored 32-mer key.
    cache: HashMap<u64, u64>,
}

impl std::fmt::Debug for SkcFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkcFileReader")
            .field("cache_size", &self.cache_size)
            .finish()
    }
}

impl SkcFileReader {
    /// Open and read the `.skc` file at `path` into memory.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut r = BufReader::new(f);

        // Read and validate header.
        let mut header = [0u8; SKC_HEADER_BYTES];
        r.read_exact(&mut header).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let magic = LittleEndian::read_u32(&header[0..4]);
        if magic != SKC_MAGIC {
            return Err(Error::BadMagic {
                file: path.to_path_buf(),
                found: format!("{:#010x}", magic),
                expected: format!("{:#010x}", SKC_MAGIC),
            });
        }
        let version = LittleEndian::read_u32(&header[4..8]);
        if version != SKC_FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                expected: SKC_FORMAT_VERSION,
            });
        }
        let cache_size = LittleEndian::read_u64(&header[8..16]);

        // Guard `cache_size` before allocating: a corrupted/malicious header
        // declaring a huge count must not drive a giant `with_capacity`
        // allocation. Convert with checked arithmetic and require the file to
        // actually hold the declared body before trusting the count.
        let cache_size_usize = usize::try_from(cache_size).map_err(|_| Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("cache_size={cache_size} does not fit in usize"),
        })?;
        let expected_len = cache_size_usize
            .checked_mul(SKC_BYTES_PER_ENTRY)
            .and_then(|body| body.checked_add(SKC_HEADER_BYTES))
            .ok_or_else(|| Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!("cache_size={cache_size} overflows the .skc size calculation"),
            })?;
        let file_len = r
            .get_ref()
            .metadata()
            .map_err(|e| Error::Io {
                path: path.to_path_buf(),
                source: e,
            })?
            .len();
        if file_len < expected_len as u64 {
            return Err(Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!(
                    "file is {file_len} bytes, header declares {cache_size} entries \
                     (expected at least {expected_len} bytes)"
                ),
            });
        }

        // Read body, enforcing the strictly-ascending `sa_index` invariant the
        // writer guarantees. Inserting blindly would let duplicate or
        // out-of-order entries silently overwrite earlier ones.
        let mut cache = HashMap::with_capacity(cache_size_usize);
        let mut entry = [0u8; SKC_BYTES_PER_ENTRY];
        let mut prev_sa_index: Option<u64> = None;
        for i in 0..cache_size {
            r.read_exact(&mut entry).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    Error::SizeMismatch {
                        file: path.to_path_buf(),
                        detail: format!("truncated at entry {i}: expected {cache_size} entries"),
                    }
                } else {
                    Error::Io {
                        path: path.to_path_buf(),
                        source: e,
                    }
                }
            })?;
            let sa_index = LittleEndian::read_u64(&entry[0..8]);
            let key = LittleEndian::read_u64(&entry[8..16]);
            if let Some(prev) = prev_sa_index {
                if sa_index <= prev {
                    return Err(Error::SizeMismatch {
                        file: path.to_path_buf(),
                        detail: format!(
                            "entry {i}: sa_index={sa_index} is not strictly greater than the \
                             previous sa_index={prev} (entries must be in ascending order)"
                        ),
                    });
                }
            }
            cache.insert(sa_index, key);
            prev_sa_index = Some(sa_index);
        }

        Ok(Self { cache_size, cache })
    }

    /// Look up the stored key for SA index `sa_idx`. Returns `Some(key)` on a
    /// cache hit, `None` on a miss.
    #[inline]
    pub fn lookup_key(&self, sa_idx: u64) -> Option<u64> {
        self.cache.get(&sa_idx).copied()
    }

    /// Number of `(sa_index, key)` pairs in this cache.
    pub fn cache_size(&self) -> u64 {
        self.cache_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_skc_three_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.skc");

        {
            let mut w = SkcFileWriter::create(&path, 3).unwrap();
            w.write_entry(0, 0xAAAA_BBBB_CCCC_DDDD).unwrap();
            w.write_entry(5, 0x1111_2222_3333_4444).unwrap();
            w.write_entry(100, 0).unwrap();
            w.finish().unwrap();
        }

        let r = SkcFileReader::open(&path).unwrap();
        assert_eq!(r.cache_size(), 3);
        assert_eq!(r.lookup_key(0), Some(0xAAAA_BBBB_CCCC_DDDD));
        assert_eq!(r.lookup_key(5), Some(0x1111_2222_3333_4444));
        assert_eq!(r.lookup_key(100), Some(0));
        assert_eq!(r.lookup_key(1), None);
        assert_eq!(r.lookup_key(99), None);
    }

    #[test]
    fn skc_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.skc");
        std::fs::write(&path, vec![0xffu8; 64]).unwrap();
        let err = SkcFileReader::open(&path).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("magic"));
    }

    #[test]
    fn skc_writer_rejects_wrong_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("short.skc");
        let mut w = SkcFileWriter::create(&path, 5).unwrap();
        w.write_entry(0, 42).unwrap(); // only write 1 of 5
        assert!(w.finish().is_err());
    }

    /// Writing more entries than the declared `cache_size` must fail
    /// immediately, before the extra bytes are persisted — not only at
    /// `finish`.
    #[test]
    fn skc_writer_rejects_overrun() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("overrun.skc");
        let mut w = SkcFileWriter::create(&path, 1).unwrap();
        w.write_entry(0, 42).unwrap();
        let err = w.write_entry(1, 43).unwrap_err();
        assert!(format!("{err}").contains("cache_size is 1"));
    }

    /// Build a `.skc` file from raw bytes: valid header + caller-supplied body.
    fn write_raw_skc(path: &Path, cache_size: u64, body: &[u8]) {
        let mut bytes = Vec::new();
        let mut header = [0u8; SKC_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], SKC_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], SKC_FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], cache_size);
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(body);
        std::fs::write(path, &bytes).unwrap();
    }

    /// A header declaring a huge `cache_size` for a tiny file must be rejected
    /// before allocating, rather than driving a giant `with_capacity`.
    #[test]
    fn skc_reader_rejects_oversized_cache_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("huge.skc");
        // Header claims u64::MAX entries but the body is empty.
        write_raw_skc(&path, u64::MAX, &[]);
        let err = SkcFileReader::open(&path).unwrap_err();
        let msg = format!("{err}").to_lowercase();
        assert!(msg.contains("overflow") || msg.contains("expected at least"));
    }

    /// The reader must reject duplicate or out-of-order `sa_index` values
    /// instead of silently overwriting earlier entries in the `HashMap`.
    #[test]
    fn skc_reader_rejects_out_of_order_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unordered.skc");
        let mut body = [0u8; 2 * SKC_BYTES_PER_ENTRY];
        // entry 0: sa_index = 5
        LittleEndian::write_u64(&mut body[0..8], 5);
        LittleEndian::write_u64(&mut body[8..16], 100);
        // entry 1: sa_index = 5 again (not strictly greater) → must error
        LittleEndian::write_u64(&mut body[16..24], 5);
        LittleEndian::write_u64(&mut body[24..32], 200);
        write_raw_skc(&path, 2, &body);
        let err = SkcFileReader::open(&path).unwrap_err();
        assert!(format!("{err}").contains("ascending order"));
    }
}
