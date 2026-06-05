// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::spectrum::SmemStep;
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

/// Independent oracle: doubled text base at q (mirror of doubled_base_at), then
/// the SA interval for query[..m] by brute force over all suffixes.
fn doubled(bases: &[u8], q: usize) -> Option<u8> {
    let l = bases.len();
    if q >= 2 * l {
        None
    } else if q < l {
        Some(bases[q])
    } else {
        Some(3 - bases[2 * l - 1 - q])
    }
}

/// Brute-force: count suffixes of the doubled text whose first `m` bases equal
/// `query[..m]` (sentinel never matches a real base). Returns occ_count.
fn oracle_occ(bases: &[u8], query: &[u8], m: usize) -> u64 {
    let n = 2 * bases.len() + 1;
    let mut c = 0u64;
    for start in 0..n {
        let mut ok = true;
        for (j, &qb) in query.iter().enumerate().take(m) {
            match doubled(bases, start + j) {
                Some(rb) if rb == qb => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            c += 1;
        }
    }
    c
}

#[test]
fn forward_spectrum_matches_oracle() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = (0..60).map(|i| ((i * 7 + 3) % 4) as u8).collect();
    let pac = dir.path().join("r.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("r.prmi");
    build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    // Query = the forward text starting at position 10 (guaranteed to occur).
    let query: Vec<u8> = bases[10..10 + 32].to_vec();
    let steps: Vec<SmemStep> =
        idx.forward_spectrum(&query, &bases, prmi::index::smem::PacEncoding::Unpacked);

    assert!(!steps.is_empty(), "expected a match");
    // Every step's occ_count must equal the brute-force occ for that match_len,
    // and the positions [sa_start, sa_start+occ_count) must all share the prefix.
    for s in &steps {
        assert_eq!(
            s.occ_count,
            oracle_occ(&bases, &query, s.match_len as usize),
            "occ mismatch at match_len={}",
            s.match_len
        );
        for i in s.sa_start..s.sa_start + s.occ_count {
            let pos = idx.sa_position_for(i);
            let (_, lcp) = prmi_compare(&bases, &query, pos);
            assert!(
                u64::from(lcp) >= s.match_len,
                "pos {pos} shares < match_len with query"
            );
        }
    }
    // Monotonicity: strictly increasing match_len, non-increasing occ_count.
    for w in steps.windows(2) {
        assert!(w[1].match_len > w[0].match_len);
        assert!(w[1].occ_count <= w[0].occ_count);
    }
}

/// Deterministic 64-bit LCG (Knuth/MMIX constants) for reproducible test data.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) & 0x7fff_ffff
}

#[test]
fn forward_spectrum_occ_correct_for_wide_shallow_intervals() {
    let dir = tempdir().unwrap();
    // An A-biased pseudo-random reference (~70% 'A'=0): the local 32-mer context is
    // diverse enough that the learned model fits TIGHTLY (small error window), yet
    // the single-base 'A' interval spans a huge fraction of the SA. This is exactly
    // the case the old model-window clamp undercounted — a shallow interval far
    // WIDER than the model's error window. (A tandem repeat does NOT exercise the
    // bug: its near-identical 32-mers blow the model window up to cover the whole
    // block, so the clamp never bites.)
    let mut state = 0x1234_5678_9abc_def0u64;
    let bases: Vec<u8> = (0..4000)
        .map(|_| {
            let r = lcg_next(&mut state) % 10;
            if r < 7 {
                0
            } else {
                (r % 3 + 1) as u8
            }
        })
        .collect();
    let pac = dir.path().join("abias.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("abias.prmi");
    build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let enc = prmi::index::smem::PacEncoding::Unpacked;

    // Query starts on an 'A' so its shallowest (m=1) interval is the huge 'A' block.
    let off = (200..260).find(|&o| bases[o] == 0).expect("an 'A' offset");
    let query: Vec<u8> = bases[off..off + 24].to_vec();
    let steps = idx.forward_spectrum(&query, &bases, enc);
    assert!(!steps.is_empty());
    // The shallowest step must have a LARGE occ (proves we did NOT clamp to the
    // model error window — that window is only a few entries wide here).
    assert!(
        steps[0].occ_count >= 500,
        "shallow 'A' interval should be wide, got {}",
        steps[0].occ_count
    );
    // Every step's occ_count must equal the brute-force occ at that match_len.
    for s in &steps {
        assert_eq!(
            s.occ_count,
            oracle_occ(&bases, &query, s.match_len as usize),
            "occ mismatch at match_len={}",
            s.match_len
        );
    }
    // And monotonicity holds.
    for w in steps.windows(2) {
        assert!(w[1].match_len > w[0].match_len);
        assert!(w[1].occ_count <= w[0].occ_count);
    }
}

#[test]
fn backward_spectrum_matches_oracle_including_nonrepresentative_prefix() {
    let dir = tempdir().unwrap();
    // Construct bases so that a right-anchored interval has members with DIFFERENT
    // preceding bases (so SA[sa_start]'s predecessor != some others').
    let bases: Vec<u8> = b"ACGTACGTTACGTAACGTACGTGACGTACGT"
        .iter()
        .map(|&c| match c {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            _ => 3,
        })
        .collect();
    let pac = dir.path().join("r.pac");
    write_pac(&pac, &bases);
    let prefix = dir.path().join("r.prmi");
    build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let enc = prmi::index::smem::PacEncoding::Unpacked;

    // read = full forward text; pivot chosen so the anchor interval (a) is COMPLETE
    // (its occ_count equals the true brute-force occ — the model error window did not
    // clamp it), (b) has members with mixed preceding bases (all four, here), and
    // (c) has a representative SA[sa_start] whose predecessor differs from the
    // prepended base read[pivot-1] — the non-representative-predecessor case that the
    // iterate-and-filter logic must handle.
    let read = bases.clone();
    let pivot = 23usize;
    let q: Vec<u8> = read[pivot..].to_vec();
    let fwd = idx.forward_spectrum(&q, &bases, enc);
    let anchor = *fwd.first().unwrap(); // shortest forward step (largest occ)

    // Sanity: the anchor interval must have members with mixed preceding bases,
    // so the iterate-and-filter logic (not just SA[sa_start]'s own predecessor)
    // is actually exercised.
    let prepend = read[pivot - 1];
    let mut preds = std::collections::BTreeSet::new();
    for i in anchor.sa_start..anchor.sa_start + anchor.occ_count {
        let pos = idx.sa_position_for(i) as usize;
        if pos > 0 {
            if let Some(b) = doubled(&bases, pos - 1) {
                preds.insert(b);
            }
        }
    }
    // The anchor must be COMPLETE (the model error window did not clamp it), so the
    // full-text brute-force oracle is the correct baseline for the extended queries.
    let anchor_q: Vec<u8> = read[pivot..pivot + anchor.match_len as usize].to_vec();
    assert_eq!(
        anchor.occ_count,
        oracle_occ(&bases, &anchor_q, anchor_q.len()),
        "anchor interval must be complete (unclamped by the model window)"
    );
    assert!(
        anchor.occ_count >= 2,
        "anchor interval must have >=2 members"
    );
    assert!(
        preds.len() >= 2,
        "anchor members must have mixed preceding bases: {preds:?}"
    );
    assert!(
        preds.contains(&prepend),
        "the prepended base must be among the predecessors"
    );
    let rep_pred = doubled(&bases, idx.sa_position_for(anchor.sa_start) as usize - 1);
    assert_ne!(
        rep_pred,
        Some(prepend),
        "representative SA[sa_start]'s predecessor must DIFFER from the prepended base"
    );

    let steps = idx.backward_spectrum(
        anchor.sa_start,
        anchor.occ_count,
        anchor.match_len,
        &read,
        pivot,
        &bases,
        enc,
    );
    assert!(!steps.is_empty(), "expected at least one backward step");
    // For each backward step, brute-force the interval for the fully extended query.
    for s in &steps {
        let total = s.match_len as usize;
        let left_ext = total - anchor.match_len as usize;
        let qstart = pivot - left_ext;
        let extq: Vec<u8> = read[qstart..pivot + anchor.match_len as usize].to_vec();
        assert_eq!(
            s.occ_count,
            oracle_occ(&bases, &extq, extq.len()),
            "backward occ mismatch at total match_len={total}"
        );
        // contiguity: positions are exactly the brute-force matching set.
        for i in s.sa_start..s.sa_start + s.occ_count {
            let pos = idx.sa_position_for(i);
            let (_, lcp) = prmi_compare(&bases, &extq, pos);
            assert!(lcp as usize >= extq.len());
        }
    }
}

/// Backward extension across the forward/RC junction.
///
/// Sequence: `[(i*3+1)%4 for i in 0..39] + [0]` (40 bases, last base = A=0).
/// Doubled text: Fwd(40) || RC(40). The junction is the boundary between
/// position 39 (last forward base, A=0) and position 40 (first RC base,
/// T=3 = complement of A).
///
/// Anchor: `forward_spectrum` over the RC-side query starting at `l_pac`
/// yields a step (m≥4) that contains only the suffix at position `l_pac`.
/// Backward extension prepends `doubled(l_pac-1)=A=0`, landing at `l_pac-1`
/// — crossing the Fwd/RC junction. prmi does NOT special-case the junction
/// (the FMI text adjacency is exact; bns filters junction-spanning seeds
/// downstream), so `backward_spectrum` must handle this transparently.
#[test]
fn backward_spectrum_crosses_fwd_rc_junction() {
    let dir = tempdir().unwrap();
    // 40-base sequence ending in A(=0); last forward base = 0.
    // doubled(39) = 0 (A), doubled(40) = 3 (T = 3-fwd[39] = 3-0).
    let bases: Vec<u8> = (0u8..39)
        .map(|i| (i * 3 + 1) % 4)
        .chain(std::iter::once(0))
        .collect();
    let l_pac = bases.len() as u64; // = 40
    let pac_path = dir.path().join("junc.pac");
    write_pac(&pac_path, &bases);
    let prefix = dir.path().join("junc.prmi");
    build_sidecar_from_pac(&pac_path, &prefix, None, MaskConfig::default(), 1).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();
    let enc = prmi::index::smem::PacEncoding::Unpacked;

    // The doubled text at l_pac: T=3, A=0, T=3, G=2, C=1, A=0, T=3, G=2 (8 bases).
    // This is unique at m=4: only the suffix at l_pac matches.
    let jpos = l_pac as usize;
    let anchor_query: Vec<u8> = (0..8).map(|j| doubled(&bases, jpos + j).unwrap()).collect();

    // Run forward_spectrum on the anchor query; find the step that uniquely pins to l_pac.
    let fwd_steps = idx.forward_spectrum(&anchor_query, &bases, enc);
    assert!(
        !fwd_steps.is_empty(),
        "forward_spectrum returned no steps for junction query"
    );

    // The anchor is the deepest forward step — at m≥4 it uniquely contains l_pac.
    let anchor = *fwd_steps.last().unwrap();

    // Assert: l_pac is actually in the anchor's SA interval.
    let lp_in_anchor = (anchor.sa_start..anchor.sa_start + anchor.occ_count)
        .any(|i| idx.sa_position_for(i) == l_pac);
    assert!(
        lp_in_anchor,
        "anchor SA interval does not contain the suffix at l_pac={l_pac}"
    );

    // Anchor occ must equal the brute-force oracle (interval is complete).
    assert_eq!(
        anchor.occ_count,
        oracle_occ(&bases, &anchor_query, anchor.match_len as usize),
        "anchor occ_count must match oracle at match_len={}",
        anchor.match_len
    );

    // Build the read: prepend doubled(jpos-1) = bases[l_pac-1] = 0 (A) then the anchor query.
    // pivot=1: anchor starts at read[1..], so read[0] = the junction-left base.
    let prepend = doubled(&bases, jpos - 1).expect("jpos-1 is a valid doubled-text position");
    let read: Vec<u8> = std::iter::once(prepend)
        .chain(anchor_query.iter().copied())
        .collect();
    let pivot = 1usize;

    let steps = idx.backward_spectrum(
        anchor.sa_start,
        anchor.occ_count,
        anchor.match_len,
        &read,
        pivot,
        &bases,
        enc,
    );
    assert!(
        !steps.is_empty(),
        "expected at least one backward step across the Fwd/RC junction"
    );

    // Validate each backward step against the brute-force oracle (same pattern as
    // backward_spectrum_matches_oracle_including_nonrepresentative_prefix).
    for s in &steps {
        let total = s.match_len as usize;
        let left_ext = total - anchor.match_len as usize;
        let qstart = pivot - left_ext;
        let extq: Vec<u8> = read[qstart..pivot + anchor.match_len as usize].to_vec();
        assert_eq!(
            s.occ_count,
            oracle_occ(&bases, &extq, extq.len()),
            "junction backward occ mismatch at total match_len={total}"
        );
        for i in s.sa_start..s.sa_start + s.occ_count {
            let pos = idx.sa_position_for(i);
            let (_, lcp) = prmi_compare(&bases, &extq, pos);
            assert!(
                lcp as usize >= extq.len(),
                "SA position {pos} shares < match_len with extended query (lcp={lcp}, needed {})",
                extq.len()
            );
        }
    }
}

// helper mirroring compare for the assert above
fn prmi_compare(bases: &[u8], query: &[u8], sa_pos: u64) -> (bool, u32) {
    let mut lcp = 0u32;
    for (j, &qb) in query.iter().enumerate() {
        match doubled(bases, sa_pos as usize + j) {
            None => return (true, lcp),
            Some(rb) if rb == qb => lcp += 1,
            Some(rb) => return (rb < qb, lcp),
        }
    }
    (false, lcp)
}
