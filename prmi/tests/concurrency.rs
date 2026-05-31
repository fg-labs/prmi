// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::LearnedIndex;
use prmi::train::build_sidecar;
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn concurrent_lookups_are_safe() {
    let dir = tempdir().unwrap();
    let fa = dir.path().join("c.fa");
    let mut fa_bytes = b">c\n".to_vec();
    for _ in 0..128 {
        fa_bytes.extend_from_slice(b"ACGT");
    }
    fa_bytes.push(b'\n');
    std::fs::write(&fa, &fa_bytes).unwrap();
    let prefix = dir.path().join("c.fa.prmi");
    build_sidecar(&fa, &prefix, Some(16), Default::default(), 1).unwrap();
    let idx = Arc::new(LearnedIndex::open(&prefix).unwrap());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let idx = idx.clone();
            std::thread::spawn(move || {
                for k in 0..1000u64 {
                    let _ = idx.lookup(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
