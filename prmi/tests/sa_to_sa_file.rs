// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sa::{build_suffix_array, BYTES_PER_PACKED_ENTRY};
use prmi::sidecar::sa_file::{SaFileReader, SaFileWriter, SA_FILE_HEADER_BYTES};
use tempfile::tempdir;

#[test]
fn build_and_persist_roundtrip() {
    let bases: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
    let sa = build_suffix_array(&bases).unwrap();
    assert_eq!(sa.len(), bases.len());

    let dir = tempdir().unwrap();
    let path = dir.path().join("rt.sa");
    {
        let mut w = SaFileWriter::create(&path, sa.len() as u64).unwrap();
        for &pos in &sa {
            w.write_position(pos).unwrap();
        }
        w.finish().unwrap();
    }
    let expected_size = SA_FILE_HEADER_BYTES + sa.len() * BYTES_PER_PACKED_ENTRY;
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        expected_size
    );

    let r = SaFileReader::open(&path).unwrap();
    for (i, &pos) in sa.iter().enumerate() {
        assert_eq!(r.position(i as u64), pos);
    }
}
