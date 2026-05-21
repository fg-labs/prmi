// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::encoding::tokenize_32mer;
use prmi::train::keys::sa_to_keys;

#[test]
fn keys_track_suffix_prefix() {
    // 10 bases, all A — every SA suffix tokenizes to the same all-T-padded
    // pattern that starts with the available A's.
    let bases = vec![0u8; 10];
    // Brute SA (identity since all bases equal): every suffix lex-equal,
    // libsais will return some stable order — we don't assert the order,
    // just that each emitted key matches what tokenize_32mer would produce
    // when fed the suffix.
    let sa: Vec<u64> = (0..10).collect();
    let keys = sa_to_keys(&sa, &bases);
    assert_eq!(keys.len(), sa.len());
    for (i, &pos) in sa.iter().enumerate() {
        let suffix_len = bases.len() - pos as usize;
        let expected = tokenize_32mer(&bases[pos as usize..], suffix_len);
        assert_eq!(keys[i], expected);
    }
}

#[test]
fn full_length_kmer_keys_are_non_decreasing_in_sa_order() {
    // The T-padding (0b11) used for sub-32-mer suffixes does NOT preserve
    // lex order: a short suffix S that is a prefix of a longer suffix S' sorts
    // before S' in the SA (short-prefix-first lex order) but its T-padded
    // key sorts ABOVE S's key. To exercise the property that matters for
    // P-RMI training — keys are sorted along SA order — restrict the check
    // to SA entries whose suffix has the full 32 bases (no padding).
    let pattern = b"\x00\x01\x02\x03"; // ACGT repeated (A=0, C=1, G=2, T=3)
    let bases: Vec<u8> = pattern.iter().copied().cycle().take(128).collect();
    let sa = prmi::sa::build_suffix_array(&bases).unwrap();
    let keys = sa_to_keys(&sa, &bases);
    let full_kmer_keys: Vec<u64> = sa
        .iter()
        .zip(keys.iter())
        .filter(|(&pos, _)| bases.len() - pos as usize >= 32)
        .map(|(_, &k)| k)
        .collect();
    assert!(
        full_kmer_keys.len() >= 64,
        "expected at least 64 full-length 32-mer SA entries, got {}",
        full_kmer_keys.len()
    );
    for w in full_kmer_keys.windows(2) {
        assert!(w[0] <= w[1], "full-kmer keys must be sorted along SA order");
    }
}
