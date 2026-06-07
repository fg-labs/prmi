// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_from_pac;
use prmi::train::mask::MaskConfig;
use std::io::Write;
use tempfile::tempdir;

fn write_pac(path: &std::path::Path, bases: &[u8]) {
    let l = bases.len();
    let mut buf = vec![0u8; l / 4 + 1];
    for (i, &b) in bases.iter().enumerate() {
        buf[i >> 2] |= b << ((3 - (i & 3)) * 2);
    }
    buf.push((l % 4) as u8);
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&buf).unwrap();
}

#[test]
fn open_loads_isa_and_inverts_sa() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = (0..36).map(|i| (i % 4) as u8).collect();
    let pac = dir.path().join("r.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("r.prmi");
    build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();

    let idx = LearnedIndex::open(&prefix).unwrap();
    let n = 2 * bases.len() as u64 + 1;
    for i in [0u64, 2, 9, n - 1] {
        let pos = idx.sa_position_for(i);
        assert_eq!(
            idx.isa_for_refpos(pos),
            Some(i),
            "isa_for_refpos inverse failed at i={i}"
        );
    }
}
