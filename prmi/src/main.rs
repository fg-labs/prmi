// Copyright Ryan Marcus 2020
// Modified by Fulcrum Genomics 2026
// SPDX-License-Identifier: MIT

use clap::Parser;
use prmi::cli::Cli;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    Cli::parse().run()
}
