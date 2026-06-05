// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::meta::{Meta, Priors, Prmi, Ref, RmiSpec, Sa};
use prmi::Error;

fn sample() -> Meta {
    Meta {
        prmi: Prmi {
            magic: "PRMIv2".into(),
            format_version: 2,
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
            mode: "1".into(),
            skc_cache_size: None,
            strand: "forward_rc_2x".into(),
            masked_n_runs: false,
            masked_homopolymers: None,
            masked_bed: None,
            l_pac: None,
            stored_keys: None,
            pac_sha256: None,
        },
        rmi: RmiSpec {
            spec: "pwl,linear,linear_spline".into(),
            l2_leaf_count: 256,
            bit_shift: 56,
            max_error_bound: 12345,
            err_p50: None,
            err_p90: None,
            err_p99: None,
        },
        priors: Priors {
            kind: "uniform".into(),
            bed: None,
            weight: None,
            histogram: None,
            base_weight: None,
            formula: None,
        },
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
    m.prmi.magic = "RMIv3".into();
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    assert!(format!("{err}").contains("PRMIv2"));
}

#[test]
fn reject_future_version() {
    let mut m = sample();
    m.prmi.format_version = 3;
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    assert!(format!("{err}").contains("version"));
}

#[test]
fn unknown_priors_type_is_format_too_new() {
    // "bed" and "fastq_histogram" are known types; use a truly unknown value.
    let mut s = sample().to_toml().unwrap();
    s = s.replace("type = \"uniform\"", "type = \"future_prior_type\"");
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("too new"));
    assert!(msg.contains("future_prior_type"));
}

#[test]
fn bed_prior_type_roundtrips() {
    // Verify that a sidecar with [priors] type = "bed" parses and round-trips.
    let mut m = sample();
    m.priors = Priors {
        kind: "bed".into(),
        bed: Some("/tmp/targets.bed".into()),
        weight: Some(10.0),
        histogram: None,
        base_weight: None,
        formula: None,
    };
    let s = m.to_toml().unwrap();
    let parsed = Meta::from_toml_str(&s).unwrap();
    assert_eq!(parsed.priors.kind, "bed");
    assert_eq!(parsed.priors.bed.as_deref(), Some("/tmp/targets.bed"));
    assert!((parsed.priors.weight.unwrap() - 10.0).abs() < 1e-9);
}

#[test]
fn reject_wrong_bytes_per_entry() {
    // bytes_per_entry=4 is not valid for any mode; should be rejected at validation.
    let mut m = sample();
    m.sa.bytes_per_entry = 4;
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("bytes_per_entry"),
        "expected 'bytes_per_entry' in: {msg}"
    );
    assert!(matches!(err, Error::SizeMismatch { .. }));
}

#[test]
fn reject_bytes_per_entry_mode_mismatch() {
    // Mode 2 expects bytes_per_entry=13, not 5.
    let mut m = sample();
    m.sa.mode = "2".into();
    m.sa.encoding = "packed_lo8_hi32_key64".into();
    // bytes_per_entry remains 5 — inconsistent with mode 2
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("bytes_per_entry") || msg.contains("inconsistent"),
        "expected mismatch error, got: {msg}"
    );
}

#[test]
fn mode2_roundtrips() {
    let mut m = sample();
    m.sa.mode = "2".into();
    m.sa.bytes_per_entry = 13;
    m.sa.encoding = "packed_lo8_hi32_key64".into();
    let s = m.to_toml().unwrap();
    let parsed = Meta::from_toml_str(&s).unwrap();
    assert_eq!(parsed.sa.mode, "2");
    assert_eq!(parsed.sa.bytes_per_entry, 13);
}

#[test]
fn unknown_mode_is_format_too_new() {
    let mut m = sample();
    m.sa.mode = "future_mode".into();
    let s = m.to_toml().unwrap();
    let err = Meta::from_toml_str(&s).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("too new"),
        "expected format-too-new: {msg}"
    );
}

#[test]
fn old_sidecar_without_mode_field_defaults_to_mode1() {
    // Simulate a sidecar written before the mode field was introduced.
    // The `mode` field should default to "1" (backward compat).
    let toml = sample().to_toml().unwrap();
    // Remove the mode field by replacing it with nothing.
    let toml_no_mode = toml.replace("mode = \"1\"\n", "");
    let parsed = Meta::from_toml_str(&toml_no_mode).unwrap();
    assert_eq!(parsed.sa.mode, "1");
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

#[test]
fn reject_unknown_field_in_rmi_section() {
    // A v0.2 sidecar might add a field to [rmi]; with deny_unknown_fields this
    // causes a parse error instead of silently ignoring the new data.
    let toml = sample().to_toml().unwrap();
    // Inject an unknown field into the [rmi] section.
    let toml_with_extra = toml.replace(
        "max_error_bound = 12345",
        "max_error_bound = 12345\nsome_future_field = \"v0.2\"",
    );
    let err = Meta::from_toml_str(&toml_with_extra).unwrap_err();
    // Should fail at deserialization, not pass through silently.
    assert!(
        matches!(err, Error::TomlParse { .. }),
        "expected TomlParse for unknown field, got: {err:?}"
    );
}

#[test]
fn reject_unknown_field_in_prmi_section() {
    let toml = sample().to_toml().unwrap();
    let toml_with_extra = toml.replace(
        "magic = \"PRMIv2\"",
        "magic = \"PRMIv2\"\nsome_future_field = \"v0.2\"",
    );
    let err = Meta::from_toml_str(&toml_with_extra).unwrap_err();
    assert!(
        matches!(err, Error::TomlParse { .. }),
        "expected TomlParse for unknown field, got: {err:?}"
    );
}
