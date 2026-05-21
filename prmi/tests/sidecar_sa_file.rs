// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::sa_file::{SaFileReader, SaFileWriter, SA_FILE_HEADER_BYTES};
use tempfile::tempdir;

#[test]
fn roundtrip_three_entries() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.sa");

    {
        let mut w = SaFileWriter::create(&path, 3).unwrap();
        w.write_position(0x00).unwrap();
        w.write_position(0x123456).unwrap();
        w.write_position(0xffffffffff).unwrap();
        w.finish().unwrap();
    }

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert_eq!(file_size, (SA_FILE_HEADER_BYTES + 3 * 5) as u64);

    let r = SaFileReader::open(&path).unwrap();
    assert_eq!(r.num_entries(), 3);
    assert_eq!(r.position(0), 0x00);
    assert_eq!(r.position(1), 0x123456);
    assert_eq!(r.position(2), 0xffffffffff);
}

#[test]
fn reject_size_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.sa");
    let mut w = SaFileWriter::create(&path, 10).unwrap();
    w.write_position(0).unwrap();
    // Drop without writing remaining 9 → finish() will error.
    assert!(w.finish().is_err());
}

#[test]
fn reader_rejects_bad_magic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("garbage.sa");
    std::fs::write(&path, vec![0xffu8; 100]).unwrap();
    let err = SaFileReader::open(&path).unwrap_err();
    assert!(format!("{err}").to_lowercase().contains("magic"));
}
