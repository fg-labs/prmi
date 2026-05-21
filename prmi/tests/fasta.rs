// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::fasta::{fasta_to_2bit, FastaStats};
use std::io::Cursor;

#[test]
fn single_contig_acgt() {
    let fasta = b">chr1\nACGTACGT\n";
    let (bases, stats) = fasta_to_2bit(Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(stats, FastaStats { total_bases: 8, n_bases: 0, contigs: 1 });
}

#[test]
fn n_bases_become_a() {
    let fasta = b">chr1\nACNT\n";
    let (bases, stats) = fasta_to_2bit(Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 1, 0, 3]);
    assert_eq!(stats.n_bases, 1);
}

#[test]
fn multiple_contigs_concatenate() {
    let fasta = b">a\nAA\n>b\nCC\n>c\nGG\n";
    let (bases, stats) = fasta_to_2bit(Cursor::new(&fasta[..])).unwrap();
    assert_eq!(bases, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(stats.contigs, 3);
}
