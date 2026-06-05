// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT
use prmi::index::LearnedIndex;
use prmi::train::{build_sidecar_from_pac, mask::MaskConfig};
use std::io::Write;
use tempfile::tempdir;

fn write_pac(path: &std::path::Path, bases: &[u8]) {
    let l = bases.len();
    let mut buf = vec![0u8; l / 4 + 1];
    for (i, &b) in bases.iter().enumerate() {
        buf[i >> 2] |= b << ((3 - (i & 3)) * 2);
    }
    buf.push((l % 4) as u8);
    std::fs::File::create(path)
        .unwrap()
        .write_all(&buf)
        .unwrap();
}

#[test]
fn strided_matches_manual_stride() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = (0..40).map(|i| (i % 4) as u8).collect();
    let pac = dir.path().join("r.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("r.prmi");
    build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    let (k, step, n_out) = (3u64, 4u64, 5u64);
    let mut out = vec![0u64; n_out as usize];
    idx.sa_positions_strided(k, step, &mut out).unwrap();
    for j in 0..n_out {
        assert_eq!(out[j as usize], idx.sa_position_for(k + j * step));
    }
}
