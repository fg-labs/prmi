// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::fasta::{
    fasta_to_2bit, fasta_to_2bit_with_n_positions, fasta_to_2bit_with_sha256, FastaStats,
};
use std::io::{Cursor, Write};

#[test]
fn single_contig_acgt() {
    let fasta = b">chr1\nACGTACGT\n";
    let (bases, stats) = fasta_to_2bit(&mut Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(
        stats,
        FastaStats {
            total_bases: 8,
            n_bases: 0,
            contigs: 1
        }
    );
}

#[test]
fn n_bases_become_a() {
    let fasta = b">chr1\nACNT\n";
    let (bases, stats) = fasta_to_2bit(&mut Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 1, 0, 3]);
    assert_eq!(stats.n_bases, 1);
    assert_eq!(stats.total_bases, 4);
}

#[test]
fn multiple_contigs_concatenate() {
    let fasta = b">a\nAA\n>b\nCC\n>c\nGG\n";
    let (bases, stats) = fasta_to_2bit(&mut Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(stats.contigs, 3);
}

#[test]
fn n_positions_bitmap_marks_n_and_aligns_with_bases() {
    let fasta = b">chr1\nACNTA\n";
    let (bases, n_positions, stats) =
        fasta_to_2bit_with_n_positions(&mut Cursor::new(&fasta[..])).unwrap();
    // The 2-bit bases are identical to the convenience wrapper (N → A).
    let (bases_plain, stats_plain) = fasta_to_2bit(&mut Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, bases_plain);
    assert_eq!(stats, stats_plain);
    assert_eq!(bases, vec![0, 1, 0, 3, 0]);
    // The bitmap is parallel to `bases` and carries a single `true` at the N.
    assert_eq!(n_positions.len(), bases.len());
    assert_eq!(n_positions, vec![false, false, true, false, false]);
    assert_eq!(
        stats,
        FastaStats {
            total_bases: 5,
            n_bases: 1,
            contigs: 1
        }
    );
}

#[test]
fn sha256_digest_is_stable_and_sized() {
    // `fasta_to_2bit_with_sha256` takes a path, so stage a known FASTA on disk.
    // The expected digest is the SHA-256 of the exact 9 raw file bytes
    // (`>s\nACGTN\n`), computed independently with `shasum -a 256`.
    let content: &[u8] = b">s\nACGTN\n";
    let expected_hex = "fe0b460a3e4e95f19766f5babad3275e5040ad63b9877365930d1078a026e3d4";
    let path = std::env::temp_dir().join(format!("prmi_sha256_test_{}.fa", std::process::id()));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(content)
        .unwrap();

    let (bases, n_positions, stats, hex, size_bytes) = fasta_to_2bit_with_sha256(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(bases, vec![0, 1, 2, 3, 0]); // N → A
    assert_eq!(n_positions, vec![false, false, false, false, true]);
    assert_eq!(
        stats,
        FastaStats {
            total_bases: 5,
            n_bases: 1,
            contigs: 1
        }
    );
    assert_eq!(hex, expected_hex);
    // Byte count is the raw file size (including FASTA framing), not the base count.
    assert_eq!(size_bytes, content.len() as u64);
}
