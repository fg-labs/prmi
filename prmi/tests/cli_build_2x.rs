// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use clap::Parser;
use prmi::cli::Cli;

#[test]
fn build_accepts_store_keys_flag() {
    let cli = Cli::try_parse_from(["prmi", "build", "r.fa", "--store-keys"]);
    assert!(cli.is_ok(), "{:?}", cli.err());

    let bad = Cli::try_parse_from(["prmi", "build", "r.fa", "--memory-mode", "2"]);
    assert!(bad.is_err(), "--memory-mode should no longer be accepted");
}
