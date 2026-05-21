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

/// Counts gathered while reading a FASTA file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FastaStats {
    pub total_bases: u64,
    pub n_bases: u64,
    pub contigs: u64,
}

/// Read all records from a FASTA stream, return the concatenated 2-bit
/// sequence (one byte per base; values 0..=3) and counts.
pub fn fasta_to_2bit<R: BufRead>(reader: &mut R) -> Result<(Vec<u8>, FastaStats)> {
    let mut reader = Reader::new(reader);
    let mut bases: Vec<u8> = Vec::new();
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
                Some(c) => bases.push(c),
                None => {
                    stats.n_bases += 1;
                    bases.push(BASE_A);
                }
            }
        }
    }
    Ok((bases, stats))
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

/// `BufRead` adapter: every byte passed through `consume()` is also hashed
/// and counted, so a single read pass produces both the parsed 2-bit data
/// and the source-file SHA-256 + byte count.
///
/// We hash in `consume()` (the `fill_buf`/`consume` path used by noodles-fasta)
/// rather than in `read()` so that we do not double-count bytes when both paths
/// are exercised. The `Read::read` implementation delegates to the inner reader
/// without hashing. This is correct because noodles-fasta is a line-oriented
/// parser that exclusively uses the `fill_buf` + `consume` interface.
struct HashingTee<R: BufRead> {
    inner: R,
    hasher: Sha256,
    size_bytes: u64,
}

impl<R: BufRead> HashingTee<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            size_bytes: 0,
        }
    }

    fn finalize(self) -> (String, u64) {
        let hex = hex::encode(self.hasher.finalize());
        (hex, self.size_bytes)
    }
}

impl<R: BufRead> std::io::Read for HashingTee<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Delegate; hashing happens in consume() for BufRead consumers.
        self.inner.read(buf)
    }
}

impl<R: BufRead> BufRead for HashingTee<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        // Peek at the inner buffer before releasing it, then hash the prefix
        // that the caller is consuming.
        if amt > 0 {
            if let Ok(buf) = self.inner.fill_buf() {
                let to_hash = &buf[..amt.min(buf.len())];
                self.hasher.update(to_hash);
                self.size_bytes += to_hash.len() as u64;
            }
        }
        self.inner.consume(amt);
    }
}

/// Stream a FASTA file through the 2-bit parser AND a SHA-256 hash
/// simultaneously, avoiding a second read pass.
///
/// Returns `(bases, stats, sha256_hex, file_size_bytes)`.
pub fn fasta_to_2bit_with_sha256(path: &Path) -> Result<(Vec<u8>, FastaStats, String, u64)> {
    let f = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut tee = HashingTee::new(std::io::BufReader::new(f));
    let (bases, stats) = fasta_to_2bit(&mut tee).map_err(|e| match e {
        Error::Fasta { detail, .. } => Error::Fasta {
            file: path.to_path_buf(),
            detail,
        },
        other => other,
    })?;
    let (hex, size_bytes) = tee.finalize();
    Ok((bases, stats, hex, size_bytes))
}
