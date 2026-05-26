// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    let config_path = PathBuf::from(&crate_dir).join("cbindgen.toml");
    let config = cbindgen::Config::from_file(&config_path).expect("load cbindgen.toml");

    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen generate");

    // Authoritative location: OUT_DIR, which is always writable even when the
    // source tree is read-only or vendored. INSTALL.md documents locating the
    // header here for packaging.
    bindings.write_to_file(PathBuf::from(&out_dir).join("prmi.h"));

    // Convenience mirror into the (gitignored) source `include/` for local C
    // consumers, but only when that tree is writable — skip silently on
    // read-only / vendored checkouts rather than failing the build.
    let src_include = PathBuf::from(&crate_dir).join("include");
    if std::fs::create_dir_all(&src_include).is_ok() {
        bindings.write_to_file(src_include.join("prmi.h"));
    }
}
