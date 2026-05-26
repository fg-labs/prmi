// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Histogram utilities: converting k-mer frequency dumps from external tools
//! into prmi's u64-key histogram TSV format.

use crate::encoding::tokenize_32mer;
use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Convert a KMC text-format dump file to a prmi u64-key histogram TSV.
///
/// KMC's `kmc_tools transform <db> dump` produces lines like:
///
/// ```text
/// ACGTACGT...  count
/// ```
///
/// This function reads that format (whitespace-separated 32-mer string +
/// count) and writes a two-column TSV (`key_u64\tcount_u64`) to stdout.
/// Lines with a k-mer other than 32 bases are skipped with a warning to
/// stderr. K-mers containing non-ACGT characters are skipped, as are lines
/// whose count field is not a valid `u64`. Lines beginning with `#` and blank
/// lines are skipped. Each skip category is summarised on stderr at the end.
///
/// The output is suitable for use with `--prior-fastq-histogram`.
pub fn kmc_dump_to_histogram_tsv(kmc_dump: &Path) -> Result<()> {
    let file = std::fs::File::open(kmc_dump).map_err(|e| Error::Io {
        path: kmc_dump.to_path_buf(),
        source: e,
    })?;
    let reader = BufReader::new(file);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut skipped_len = 0usize;
    let mut skipped_n = 0usize;
    let mut skipped_bad_count = 0usize;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| Error::Io {
            path: kmc_dump.to_path_buf(),
            source: e,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Split on whitespace (KMC uses spaces or tabs between kmer and count).
        let mut parts = trimmed.split_whitespace();
        let kmer_str = match parts.next() {
            Some(s) => s,
            None => continue,
        };
        let count_str = match parts.next() {
            Some(s) => s,
            None => {
                eprintln!(
                    "warning: {}:{}: skipping line with no count field",
                    kmc_dump.display(),
                    line_num + 1
                );
                continue;
            }
        };

        if kmer_str.len() != 32 {
            skipped_len += 1;
            continue;
        }

        // Convert ASCII bases to 2-bit. Skip k-mers with non-ACGT characters.
        let mut bases = [0u8; 32];
        let mut has_n = false;
        for (i, b) in kmer_str.bytes().enumerate() {
            bases[i] = match b {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => {
                    has_n = true;
                    break;
                }
            };
        }
        if has_n {
            skipped_n += 1;
            continue;
        }

        let key = tokenize_32mer(&bases, 32);
        // A non-numeric count is treated like the other malformed records in a
        // KMC dump: skip the line with a warning rather than aborting the whole
        // conversion (the dump is foreign tool output, handled best-effort).
        let count: u64 = match count_str.parse() {
            Ok(c) => c,
            Err(_) => {
                skipped_bad_count += 1;
                continue;
            }
        };

        writeln!(out, "{key}\t{count}").map_err(|e| Error::Io {
            path: kmc_dump.to_path_buf(),
            source: e,
        })?;
    }

    if skipped_len > 0 {
        eprintln!(
            "warning: skipped {skipped_len} lines with k-mer length != 32 in {}",
            kmc_dump.display()
        );
    }
    if skipped_n > 0 {
        eprintln!(
            "warning: skipped {skipped_n} lines with non-ACGT characters in {}",
            kmc_dump.display()
        );
    }
    if skipped_bad_count > 0 {
        eprintln!(
            "warning: skipped {skipped_bad_count} lines with a non-numeric count in {}",
            kmc_dump.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn kmc_dump_acgt_32mer_produces_valid_key() {
        // Verified output: all-A 32-mer maps to key 0 (all bits 0b00).
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t5").unwrap();
        // Redirect stdout is not trivial in unit tests; just verify the function
        // does not return an error.
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }

    #[test]
    fn kmc_dump_skips_short_kmers() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "ACGT\t10").unwrap(); // 4-mer, skip
        writeln!(f, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t5").unwrap(); // 32-mer, keep
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }

    #[test]
    fn kmc_dump_skips_n_containing_kmers() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t10").unwrap(); // N, skip
        writeln!(f, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t5").unwrap(); // clean, keep
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }

    #[test]
    fn kmc_dump_empty_file_is_ok() {
        let f = NamedTempFile::new().unwrap();
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }

    #[test]
    fn kmc_dump_skips_non_numeric_count() {
        // A well-formed 32-mer with a non-numeric count is skipped (like other
        // malformed KMC-dump records), not treated as a fatal error.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\tNOTANUM").unwrap(); // bad count, skip
        writeln!(f, "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\t5").unwrap(); // valid, keep
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }

    #[test]
    fn kmc_dump_skips_comments_and_blank_lines() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "# comment").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t7").unwrap();
        kmc_dump_to_histogram_tsv(f.path()).unwrap();
    }
}
