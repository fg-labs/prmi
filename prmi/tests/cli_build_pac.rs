// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT
use clap::Parser;
use prmi::cli::Cli;

#[test]
fn build_accepts_pac_and_conflicts_with_ref() {
    assert!(Cli::try_parse_from(["prmi", "build", "--pac", "r.pac", "--out", "r.prmi"]).is_ok());
    // FASTA source still works (positional or --ref, whichever the CLI uses):
    assert!(
        Cli::try_parse_from(["prmi", "build", "r.fa", "--out", "r.prmi"]).is_ok()
            || Cli::try_parse_from(["prmi", "build", "--ref", "r.fa", "--out", "r.prmi"]).is_ok()
    );
    // both sources => error. Assert against the ref-source syntax the CLI
    // actually accepts, so an OR can't mask a parser regression where one
    // accepted form wrongly allows `--pac` alongside the FASTA.
    let positional_ok = Cli::try_parse_from(["prmi", "build", "r.fa", "--out", "r.prmi"]).is_ok();
    if positional_ok {
        assert!(Cli::try_parse_from([
            "prmi", "build", "r.fa", "--pac", "r.pac", "--out", "r.prmi"
        ])
        .is_err());
    } else {
        assert!(Cli::try_parse_from([
            "prmi", "build", "--ref", "r.fa", "--pac", "r.pac", "--out", "r.prmi"
        ])
        .is_err());
    }
}
