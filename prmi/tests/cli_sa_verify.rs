// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use clap::Parser;
use prmi::cli::Cli;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn sa_verify_passes_on_small_reference() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("r.fa");
    writeln!(
        std::fs::File::create(&fa).unwrap(),
        ">c\nACGTACGTACGTACGTAA"
    )
    .unwrap();

    let cli = Cli::try_parse_from([
        "prmi",
        "sa-verify",
        "--ref",
        fa.to_str().unwrap(),
        "--exhaustive",
    ])
    .unwrap();
    assert!(cli.run().is_ok());
}
