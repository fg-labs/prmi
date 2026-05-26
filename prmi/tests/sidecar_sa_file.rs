// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::sa_file::{
    SaFileReader, SaFileWriter, BPE_MODE2, BPE_MODE3, SA_FILE_HEADER_BYTES,
};
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
    // Mode 1 stores no key or ISA.
    assert_eq!(r.key_at(0), None);
    assert_eq!(r.isa_at(0), None);
}

#[test]
fn roundtrip_mode2_position_and_key() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mode2.sa");

    let entries: [(u64, u64); 3] = [
        (0x00, 0xAAAA_BBBB_CCCC_DDDD),
        (0x123456, 0x1111_2222_3333_4444),
        (0xffffffffff, 0),
    ];
    {
        let mut w = SaFileWriter::create_with_mode(&path, 3, BPE_MODE2).unwrap();
        for &(pos, key) in &entries {
            w.write_entry_with_key(pos, key).unwrap();
        }
        w.finish().unwrap();
    }

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert_eq!(file_size, (SA_FILE_HEADER_BYTES + 3 * BPE_MODE2) as u64);

    let r = SaFileReader::open(&path).unwrap();
    assert_eq!(r.num_entries(), 3);
    assert_eq!(r.bytes_per_entry(), BPE_MODE2);
    for (i, &(pos, key)) in entries.iter().enumerate() {
        assert_eq!(r.position(i as u64), pos);
        assert_eq!(r.key_at(i as u64), Some(key));
        // Mode 2 has no ISA column.
        assert_eq!(r.isa_at(i as u64), None);
    }
}

#[test]
fn roundtrip_mode3_position_key_and_isa() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mode3.sa");

    let entries: [(u64, u64, u64); 3] = [
        (0x00, 0xAAAA_BBBB_CCCC_DDDD, 2),
        (0x123456, 0x1111_2222_3333_4444, 0),
        (0xffffffffff, 0, 1),
    ];
    {
        let mut w = SaFileWriter::create_with_mode(&path, 3, BPE_MODE3).unwrap();
        for &(pos, key, isa) in &entries {
            w.write_entry_with_key_isa(pos, key, isa).unwrap();
        }
        w.finish().unwrap();
    }

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert_eq!(file_size, (SA_FILE_HEADER_BYTES + 3 * BPE_MODE3) as u64);

    let r = SaFileReader::open(&path).unwrap();
    assert_eq!(r.num_entries(), 3);
    assert_eq!(r.bytes_per_entry(), BPE_MODE3);
    for (i, &(pos, key, isa)) in entries.iter().enumerate() {
        assert_eq!(r.position(i as u64), pos);
        assert_eq!(r.key_at(i as u64), Some(key));
        assert_eq!(r.isa_at(i as u64), Some(isa));
    }
}

#[test]
fn reject_size_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.sa");
    let mut w = SaFileWriter::create(&path, 10).unwrap();
    w.write_position(0).unwrap();
    // Calling finish() after writing only 1 of 10 declared entries should error.
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
