// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::index::spectrum::SmemStep;
use prmi::index::LearnedIndex;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi::train::{build_sidecar_from_pac_with_config, mask::MaskConfig};
use proptest::prelude::*;
use std::io::Write;
use tempfile::tempdir;

/// Build a sidecar in **mode 2** (position + stored 32-mer key), so the spectrum
/// query path exercises the stored-key compare fast path end-to-end. Keys are
/// asserted scalar-identical elsewhere, so the oracle results are unchanged; this
/// just ensures the key path runs. Mirrors the arg shape of the upstream
/// `build_sidecar_from_pac` so call sites are unchanged.
fn build_sidecar_from_pac(
    pac: &std::path::Path,
    prefix: &std::path::Path,
    l2_leaf_count: Option<u64>,
    mask: MaskConfig,
    threads: usize,
) -> prmi::error::Result<()> {
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_from_pac_with_config(pac, prefix, l2_leaf_count, mask, threads, Some(cfg))
}

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

/// Wide-interval stress: a SHORT anchor with LARGE occ (an A-biased reference, the
/// same construction `forward_spectrum_occ_correct_for_wide_shallow_intervals` uses).
/// The old member-enumeration backward extension was O(occ) per left base; here occ is
/// in the hundreds–thousands, the regime the small-ref backward tests never exercised.
/// Every backward step's `(sa_start, occ_count)` must still equal the brute-force oracle.
#[test]
fn backward_spectrum_wide_interval_matches_oracle() {
    let dir = tempdir().unwrap();
    // A-biased pseudo-random reference (~70% 'A'=0): a short anchor spanning the 'A'
    // block has a very wide SA interval (occ in the hundreds–thousands).
    let mut state = 0x0bad_f00d_dead_beefu64;
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

    // Pick a right-anchor whose interval is WIDE (occ in the hundreds–thousands) yet
    // still extends leftward. Search forward steps for one with occ in that band; use
    // its match_len as the anchor span, pivoting so there is room to extend left.
    let read = bases.clone();
    let pivot = 1000usize;
    let q: Vec<u8> = read[pivot..].to_vec();
    let fwd = idx.forward_spectrum(&q, &bases, enc);
    let anchor = *fwd
        .iter()
        .find(|s| (100..=5000).contains(&s.occ_count))
        .or_else(|| fwd.first())
        .expect("at least one forward step");

    // The anchor must be COMPLETE (unclamped) so the full-text oracle is the baseline.
    let anchor_q: Vec<u8> = read[pivot..pivot + anchor.match_len as usize].to_vec();
    assert_eq!(
        anchor.occ_count,
        oracle_occ(&bases, &anchor_q, anchor_q.len()),
        "anchor interval must be complete (unclamped by the model window)"
    );
    assert!(
        anchor.occ_count >= 100,
        "wide-interval test needs a WIDE anchor; got occ={}",
        anchor.occ_count
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
    for s in &steps {
        let total = s.match_len as usize;
        let left_ext = total - anchor.match_len as usize;
        let qstart = pivot - left_ext;
        let extq: Vec<u8> = read[qstart..pivot + anchor.match_len as usize].to_vec();
        assert_eq!(
            (s.sa_start, s.occ_count),
            (
                oracle_lower(&bases, &extq, extq.len()),
                oracle_occ(&bases, &extq, extq.len())
            ),
            "wide-interval backward step mismatch at total match_len={total}"
        );
    }
}

/// Brute-force lower bound: the SA index of the first suffix that is >= `query[..m]`
/// over the doubled text. Computed by sorting suffix start positions by their (sentinel-
/// terminated) doubled-text content, then finding the first whose first `m` bases match.
/// Returns the count of strictly-smaller suffixes (i.e. the SA interval start). This is
/// the oracle for `SmemStep::sa_start`.
fn oracle_lower(bases: &[u8], query: &[u8], m: usize) -> u64 {
    let n = 2 * bases.len() + 1;
    // Number of suffixes strictly less than query[..m] under the GSA (sentinel-smallest)
    // ordering. A suffix is < query[..m] iff at the first differing position its base is
    // smaller, OR it runs out (hits the sentinel) before matching all m bases.
    let mut less = 0u64;
    for start in 0..n {
        if suffix_lt_query(bases, start, query, m) {
            less += 1;
        }
    }
    less
}

/// True iff the doubled-text suffix at `start` is lexicographically < `query[..m]`
/// under the sentinel-smallest ordering.
fn suffix_lt_query(bases: &[u8], start: usize, query: &[u8], m: usize) -> bool {
    for (j, &qb) in query.iter().enumerate().take(m) {
        match doubled(bases, start + j) {
            None => return true, // ref exhausted -> ref < query
            Some(rb) if rb == qb => {}
            Some(rb) => return rb < qb,
        }
    }
    false // matched all m bases -> ref >= query
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    /// Random refs + random anchors: the new boundary-binary-search backward trace must
    /// be byte-identical to the brute-force oracle trace (sa_start, occ_count, match_len).
    #[test]
    fn backward_spectrum_matches_oracle_proptest(
        bases in prop::collection::vec(0u8..=3, 8..120),
        pivot_frac in 0u64..1000,
    ) {
        let dir = tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, &bases);
        let prefix = dir.path().join("r.prmi");
        // Degenerate refs (too few distinct keys) can't train a model; skip those —
        // they're not the regime this test targets (wide/diverse SA intervals).
        prop_assume!(build_sidecar_from_pac(&pac, &prefix, None, MaskConfig::default(), 1).is_ok());
        let idx = LearnedIndex::open(&prefix).unwrap();
        let enc = prmi::index::smem::PacEncoding::Unpacked;

        let read = bases.clone();
        // Choose a pivot in [1, len) so there is at least one base to extend left into.
        let pivot = 1 + (pivot_frac as usize % (read.len() - 1));
        let q: Vec<u8> = read[pivot..].to_vec();
        let fwd = idx.forward_spectrum(&q, &bases, enc);
        // Use the shortest forward step (largest occ) as the right anchor, when present.
        let Some(anchor) = fwd.first().copied() else { return Ok(()); };

        let steps = idx.backward_spectrum(
            anchor.sa_start,
            anchor.occ_count,
            anchor.match_len,
            &read,
            pivot,
            &bases,
            enc,
        );

        // (1) The model-launched trace must equal the model-free full-SA reference.
        let reference = idx.backward_spectrum_reference(
            anchor.sa_start,
            anchor.occ_count,
            anchor.match_len,
            &read,
            pivot,
            &bases,
            enc,
        );
        prop_assert_eq!(&steps, &reference, "model-launch != full-SA reference");

        // (2) Wrong-seed / err=0 recovery: forcing a deliberately-wrong, zero-width
        // window at each SA extreme must STILL reproduce the oracle trace, proving the
        // expand-on-miss recovery (the window is a hint, never a clamp).
        let sa_num = idx.sa_num();
        for seed in [
            (|_k: u64| (0u64, 0u64)) as fn(u64) -> (u64, u64),
            (|_k: u64| (u64::MAX, 0u64)) as fn(u64) -> (u64, u64),
            (|_k: u64| (1u64, 0u64)) as fn(u64) -> (u64, u64),
        ] {
            let forced = idx.backward_spectrum_with_seed(
                anchor.sa_start,
                anchor.occ_count,
                anchor.match_len,
                &read,
                pivot,
                &bases,
                enc,
                seed,
            );
            prop_assert_eq!(&forced, &steps, "wrong-seed (sa_num={}) diverged", sa_num);
        }

        // Independently compute the oracle backward trace: prepend bases left from the
        // pivot, recomputing the full doubled-text interval each step until it empties or
        // the read boundary is reached (mirrors backward_spectrum's loop/break contract).
        let mut oracle_steps: Vec<SmemStep> = Vec::new();
        let mut left_ext = 0usize;
        while left_ext < pivot {
            let c = read[pivot - 1 - left_ext];
            if c >= 4 { break; }
            let qstart = pivot - 1 - left_ext;
            let qend = pivot + anchor.match_len as usize;
            let extq: Vec<u8> = read[qstart..qend].to_vec();
            let occ = oracle_occ(&bases, &extq, extq.len());
            if occ == 0 { break; }
            let lower = oracle_lower(&bases, &extq, extq.len());
            left_ext += 1;
            oracle_steps.push(SmemStep {
                sa_start: lower,
                occ_count: occ,
                match_len: anchor.match_len + left_ext as u64,
            });
        }
        prop_assert_eq!(steps, oracle_steps);
    }
}

/// Wrong-seed / `err = 0` recovery (deterministic): a poorly-fit model window must
/// not change the result. Force every left step to launch from a single wrong SA index
/// with zero width and assert the trace stays identical to both the real model launch
/// and the full-SA reference — proving expand-on-miss recovers the TRUE interval.
#[test]
fn backward_spectrum_wrong_seed_recovers_true_interval() {
    let dir = tempdir().unwrap();
    let bases: Vec<u8> = b"ACGTACGTTACGTAACGTACGTGACGTACGTACGTACGTT"
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
    let sa_num = idx.sa_num();

    let read = bases.clone();
    let pivot = 30usize;
    let q: Vec<u8> = read[pivot..].to_vec();
    let anchor = *idx
        .forward_spectrum(&q, &bases, enc)
        .first()
        .expect("a forward anchor");

    let truth = idx.backward_spectrum_reference(
        anchor.sa_start,
        anchor.occ_count,
        anchor.match_len,
        &read,
        pivot,
        &bases,
        enc,
    );
    let model = idx.backward_spectrum(
        anchor.sa_start,
        anchor.occ_count,
        anchor.match_len,
        &read,
        pivot,
        &bases,
        enc,
    );
    assert_eq!(model, truth, "real model launch != reference");

    // Deliberately-wrong, zero-width windows at every SA extreme + an interior point.
    let seeds: [fn(u64) -> (u64, u64); 4] =
        [|_k| (0, 0), |_k| (u64::MAX, 0), |_k| (1, 0), |_k| (7, 0)];
    for seed in seeds {
        let forced = idx.backward_spectrum_with_seed(
            anchor.sa_start,
            anchor.occ_count,
            anchor.match_len,
            &read,
            pivot,
            &bases,
            enc,
            seed,
        );
        let (p0, _) = seed(0);
        assert_eq!(
            forced, truth,
            "wrong seed pred={} (err=0, sa_num={}) did not recover true interval",
            p0, sa_num
        );
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
