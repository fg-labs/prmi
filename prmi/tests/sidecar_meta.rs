// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::meta::{Meta, Prmi, Priors, Ref, RmiSpec, Sa};
use prmi::Error;

fn sample() -> Meta {
    Meta {
        prmi: Prmi {
            magic: "PRMIv1".into(),
            format_version: 1,
            trainer_version: format!("prmi={}", env!("CARGO_PKG_VERSION")),
            created_utc: "2026-05-20T15:00:00Z".into(),
        },
        ref_: Ref {
            path: "/tmp/test.fa".into(),
            sha256: "00".repeat(32),
            size_bytes: 4096,
        },
        sa: Sa {
            num_entries: 4096,
            bytes_per_entry: 5,
            encoding: "packed_lo8_hi32".into(),
        },
        rmi: RmiSpec {
            spec: "pwl,linear,linear_spline".into(),
            l2_leaf_count: 256,
            bit_shift: 56,
            max_error_bound: 12345,
        },
        priors: Priors { kind: "uniform".into() },
    }
}

#[test]
fn roundtrip_toml() {
    let m = sample();
    let s = m.to_toml().unwrap();
    let parsed: Meta = Meta::from_toml_str(&s).unwrap();
    assert_eq!(parsed, m);
}

#[test]
fn reject_bad_magic() {
    let mut m = sample();
    m.prmi.magic = "RMIv2".into();
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    assert!(format!("{err}").contains("PRMIv1"));
}

#[test]
fn reject_future_version() {
    let mut m = sample();
    m.prmi.format_version = 2;
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    assert!(format!("{err}").contains("version"));
}

#[test]
fn unknown_priors_type_is_format_too_new() {
    let mut s = sample().to_toml().unwrap();
    s = s.replace("type = \"uniform\"", "type = \"bed\"");
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("too new"));
    assert!(msg.contains("bed"));
}

#[test]
fn reject_wrong_bytes_per_entry() {
    let mut m = sample();
    m.sa.bytes_per_entry = 4;
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("bytes_per_entry"), "expected 'bytes_per_entry' in: {msg}");
    assert!(matches!(err, Error::SizeMismatch { .. }));
}

#[test]
fn file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.meta");
    let m = sample();
    m.write_file(&path).unwrap();
    let parsed = Meta::read_file(&path).unwrap();
    assert_eq!(parsed, m);
}

#[test]
fn read_file_error_includes_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("garbage.meta");
    std::fs::write(&path, b"this is not valid toml [[[[").unwrap();
    let err = Meta::read_file(&path).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("garbage.meta"),
        "expected file path in error message: {msg}"
    );
}
