// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! FASTA → 2-bit PAC conversion. Reads using `noodles-fasta`, concatenates
//! every contig in order, and emits one byte per base where ACGT → 0/1/2/3
//! and N (or any non-IUPAC base) → 0 (A). Contigs are concatenated without
//! sentinels; 32-mer queries spanning contig boundaries can therefore
//! spuriously match — documented as a v0.1 limitation in the README and
//! the trainer's user-visible docs.

use crate::encoding::{base_to_2bit, BASE_A};
use crate::error::{Error, Result};
use noodles_fasta::io::Reader;
use sha2::{Digest, Sha256};
use std::io::BufRead;
use std::path::Path;

// ── HashingFileReader ─────────────────────────────────────────────────────────

/// A thin `Read` wrapper around `std::fs::File` that hashes every byte as it
/// passes through.
///
/// Single-pass design: the file is opened once and read once. As bytes stream
/// from `inner` through `Read::read` to the caller (typically a `BufReader`
/// wrapping a `noodles-fasta` reader), each chunk is immediately fed to the
/// SHA-256 hasher. Because every `BufRead` and `noodles-fasta` operation
/// ultimately calls `Read::read` on the underlying source, all bytes the file
/// contains are hashed exactly once regardless of how noodles buffers
/// internally.
///
/// No TOCTOU window: file content is hashed as it is parsed, so the hash
/// always corresponds to the exact bytes noodles operated on.
struct HashingFileReader {
    inner: std::fs::File,
    hasher: Sha256,
    bytes_hashed: u64,
}

impl HashingFileReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: std::fs::File::open(path).map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?,
            hasher: Sha256::new(),
            bytes_hashed: 0,
        })
    }

    /// Consume the reader and return `(sha256_hex, bytes_hashed)`.
    fn finish(self) -> (String, u64) {
        (format!("{:x}", self.hasher.finalize()), self.bytes_hashed)
    }
}

impl std::io::Read for HashingFileReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.hasher.update(&buf[..n]);
            self.bytes_hashed += n as u64;
        }
        Ok(n)
    }
}

/// Return type of [`fasta_to_2bit_with_sha256`]:
/// `(bases, n_positions, stats, sha256_hex, file_size_bytes)`.
pub type FastaWithSha256 = (Vec<u8>, Vec<bool>, FastaStats, String, u64);

/// Counts gathered while reading a FASTA file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FastaStats {
    /// Total number of bases seen across all contigs (including N).
    pub total_bases: u64,
    /// Number of N (or non-IUPAC) bases substituted with A.
    pub n_bases: u64,
    /// Number of records (contigs) in the FASTA.
    pub contigs: u64,
}

/// Read all records from a FASTA stream, return the concatenated 2-bit
/// sequence (one byte per base; values 0..=3) and counts.
pub fn fasta_to_2bit<R: BufRead>(reader: &mut R) -> Result<(Vec<u8>, FastaStats)> {
    let (bases, _, stats) = fasta_to_2bit_with_n_positions(reader)?;
    Ok((bases, stats))
}

/// Like [`fasta_to_2bit`], but also returns a parallel `Vec<bool>` of the same
/// length as `bases` where `true` means the original FASTA had an N (or any
/// non-IUPAC base) at that position. The bitmap is produced during the single
/// encode pass — no second scan is required.
///
/// The returned tuple is `(bases, n_positions, stats)`.
pub fn fasta_to_2bit_with_n_positions<R: BufRead>(
    reader: &mut R,
) -> Result<(Vec<u8>, Vec<bool>, FastaStats)> {
    let mut reader = Reader::new(reader);
    let mut bases: Vec<u8> = Vec::new();
    let mut n_positions: Vec<bool> = Vec::new();
    let mut stats = FastaStats::default();

    for record in reader.records() {
        let record = record.map_err(|e| Error::Fasta {
            file: std::path::PathBuf::new(),
            detail: e.to_string(),
        })?;
        stats.contigs += 1;
        for &b in record.sequence().as_ref() {
            stats.total_bases += 1;
            match base_to_2bit(b) {
                Some(c) => {
                    bases.push(c);
                    n_positions.push(false);
                }
                None => {
                    stats.n_bases += 1;
                    bases.push(BASE_A);
                    n_positions.push(true);
                }
            }
        }
    }
    Ok((bases, n_positions, stats))
}

/// Convenience wrapper: open the file by path.
///
/// On error, the file path is included in the returned `Error`.
pub fn fasta_file_to_2bit(path: &Path) -> Result<(Vec<u8>, FastaStats)> {
    let f = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut r = std::io::BufReader::new(f);
    fasta_to_2bit(&mut r).map_err(|e| match e {
        Error::Fasta { detail, .. } => Error::Fasta {
            file: path.to_path_buf(),
            detail,
        },
        other => other,
    })
}

/// Convenience wrapper: open the file by path and return bases + N-position bitmap.
///
/// On error, the file path is included in the returned `Error`.
pub fn fasta_file_to_2bit_with_n_positions(
    path: &Path,
) -> Result<(Vec<u8>, Vec<bool>, FastaStats)> {
    let f = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut r = std::io::BufReader::new(f);
    fasta_to_2bit_with_n_positions(&mut r).map_err(|e| match e {
        Error::Fasta { detail, .. } => Error::Fasta {
            file: path.to_path_buf(),
            detail,
        },
        other => other,
    })
}

/// Parse a FASTA file to 2-bit PAC and simultaneously compute a SHA-256 hash
/// of the raw file bytes. Also returns an N-position bitmap (same length as
/// `bases`; `true` at positions originally N).
///
/// Single-pass: the file is opened once and read exactly once. [`HashingFileReader`]
/// hashes every byte as it streams through to the `BufReader` / noodles-fasta
/// layer. Because all noodles read operations ultimately call `Read::read` on
/// the underlying source, every byte in the file is hashed exactly once
/// regardless of how noodles buffers internally — including any lookahead that
/// `BufReader::fill_buf` performs.
///
/// No TOCTOU window: the hash and the parsed bases are derived from the same
/// single file read, so the `.meta` SHA-256 always corresponds to the content
/// the trainer operated on.
///
/// Returns `(bases, n_positions, stats, sha256_hex, file_size_bytes)`.
pub fn fasta_to_2bit_with_sha256(path: &Path) -> Result<FastaWithSha256> {
    let hfr = HashingFileReader::open(path)?;
    let mut r = std::io::BufReader::new(hfr);
    let (bases, n_positions, stats) =
        fasta_to_2bit_with_n_positions(&mut r).map_err(|e| match e {
            Error::Fasta { detail, .. } => Error::Fasta {
                file: path.to_path_buf(),
                detail,
            },
            other => other,
        })?;
    // Recover the hasher from inside the BufReader → HashingFileReader.
    // BufReader wraps the HashingFileReader; `into_inner()` returns it.
    let hfr = r.into_inner();
    let (hex, size_bytes) = hfr.finish();
    Ok((bases, n_positions, stats, hex, size_bytes))
}
