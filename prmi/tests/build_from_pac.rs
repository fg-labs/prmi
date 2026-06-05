// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use std::io::Write;
use tempfile::tempdir;

use prmi::train::config::{MemoryMode, TrainerConfig};

use prmi::sidecar::meta::Meta;
use prmi::sidecar::SidecarPaths;
use prmi::train::mask::MaskConfig;

// Mirror bwa's .pac writer (4 bases/byte MSB-first + l_pac%4 tail byte).
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
fn pac_built_sidecar_is_2x_and_records_pac_sha() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = (0..40).map(|i| (i % 4) as u8).collect(); // 40 fwd bases
    let pac = dir.path().join("ref.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("ref.prmi");

    prmi::train::build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();

    let meta = Meta::read_file(&SidecarPaths::from_prefix(&prefix).meta).unwrap();
    assert_eq!(meta.sa.strand, "forward_rc_2x");
    assert_eq!(meta.sa.num_entries, 2 * bases.len() as u64 + 1);
    assert_eq!(meta.sa.pac_sha256.as_ref().unwrap().len(), 64);
}

#[test]
fn pac_build_with_store_keys_is_mode2() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = (0..40).map(|i| (i % 4) as u8).collect();
    let pac = dir.path().join("ref.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("ref.prmi");

    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    prmi::train::build_sidecar_from_pac_with_config(
        &pac,
        &prefix,
        None,
        prmi::train::mask::MaskConfig::default(),
        1,
        Some(cfg),
    )
    .unwrap();

    let meta = Meta::read_file(&SidecarPaths::from_prefix(&prefix).meta).unwrap();
    assert_eq!(meta.sa.bytes_per_entry, 13);
    assert_eq!(meta.sa.stored_keys, Some(true));
}

#[test]
fn pac_and_fasta_agree_on_sa_for_n_free_reference() {
    use prmi::index::LearnedIndex;
    // N-free sequence; FASTA(N->A) and .pac(no N to substitute) carry identical bases.
    let seq: Vec<u8> = (0..50).map(|i| (i % 4) as u8).collect();
    let letters = [b'A', b'C', b'G', b'T'];
    let fasta_seq: String = seq.iter().map(|&b| letters[b as usize] as char).collect();

    let dir = tempdir().unwrap();
    // .pac sidecar
    let pac = dir.path().join("r.pac");
    write_pac(&pac, &seq);
    let pac_prefix = dir.path().join("r_pac.prmi");
    prmi::train::build_sidecar_from_pac(&pac, &pac_prefix, None, MaskConfig::default(), 1).unwrap();
    // FASTA sidecar
    let fa = dir.path().join("r.fa");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&fa).unwrap();
        writeln!(f, ">c\n{fasta_seq}").unwrap();
    }
    let fa_prefix = dir.path().join("r_fa.prmi");
    prmi::train::build_sidecar(&fa, &fa_prefix, None, MaskConfig::default(), 1).unwrap();

    let a = LearnedIndex::open(&pac_prefix).unwrap();
    let b = LearnedIndex::open(&fa_prefix).unwrap();
    assert_eq!(a.sa_num(), b.sa_num());
    for i in 0..a.sa_num() {
        assert_eq!(
            a.sa_position_for(i),
            b.sa_position_for(i),
            "SA divergence at index {i}"
        );
    }
}
