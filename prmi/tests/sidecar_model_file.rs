// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::model_file::{ModelEntry, ModelFileReader, ModelFileWriter, ModelLayer};
use tempfile::tempdir;

fn fixtures() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            alpha: 0.0,
            beta: 1.0,
            err: 0,
        },
        ModelEntry {
            alpha: -1.5,
            beta: 2.5,
            err: 0x8000_0000_0000_0001,
        },
        ModelEntry {
            alpha: 3.15,
            beta: -2.71,
            err: 42,
        },
    ]
}

#[test]
fn roundtrip_l1() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.l1");
    let entries = fixtures();
    ModelFileWriter::write(&path, ModelLayer::L1, &entries).unwrap();
    let reader = ModelFileReader::open(&path, ModelLayer::L1).unwrap();
    assert_eq!(reader.len(), entries.len());
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(reader.entry(i), *e);
    }
}

#[test]
fn roundtrip_l2() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.l2");
    let entries = fixtures();
    ModelFileWriter::write(&path, ModelLayer::L2, &entries).unwrap();
    let reader = ModelFileReader::open(&path, ModelLayer::L2).unwrap();
    assert_eq!(reader.len(), entries.len());
}

#[test]
fn wrong_magic_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mismatch.l2");
    ModelFileWriter::write(&path, ModelLayer::L1, &fixtures()).unwrap();
    // Opening an L1 file as L2 should fail on magic.
    let err = ModelFileReader::open(&path, ModelLayer::L2).unwrap_err();
    assert!(format!("{err}").to_lowercase().contains("magic"));
}
