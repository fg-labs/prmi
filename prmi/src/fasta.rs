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
pub fn fasta_to_2bit<R: BufRead>(reader: R) -> Result<(Vec<u8>, FastaStats)> {
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
    fasta_to_2bit(std::io::BufReader::new(f)).map_err(|e| match e {
        Error::Fasta { detail, .. } => Error::Fasta {
            file: path.to_path_buf(),
            detail,
        },
        other => other,
    })
}
