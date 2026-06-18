// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Design Z (tiered / position-filtered `.sa`) feasibility proof.
//!
//! A keep-masked index retains ONLY suffix-array entries whose forward
//! reference coordinate lies in the keep-set, over the UNCHANGED full-genome
//! text. The load-bearing claims under test:
//!
//!   1. **Byte-identity for served reads.** A read whose maximal matches all
//!      fall inside the keep-set must yield the SAME `(m, n, s)` SMEMs from the
//!      keep-masked index as from the full index — because `match_len` is
//!      computed against the real (full) genome bases and every retained suffix
//!      is a real genome suffix, so search over the position-sparse SA returns
//!      the identical answer.
//!   2. **Divergence (reject signal) for off-keep reads.** A read whose match
//!      lies OUTSIDE the keep-set must NOT be reproduced by the keep-masked
//!      index (its suffixes were dropped), so the dispatcher would reject it to
//!      the genome fallback.

use prmi::index::collect::{CollectOpts, Smem};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use prmi::train::build_sidecar_from_pac_with_config;
use prmi::train::config::{MemoryMode, TrainerConfig};
use prmi::train::mask::{BedInterval, MaskConfig};
use std::io::Write;
use tempfile::tempdir;

/// Pack a 2-bit base array into bwa-style `.pac` (2 bits/base, big-endian within
/// a byte) with the trailing partial-byte count, matching `spectrum_oracle`.
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

/// Deterministic pseudo-random 0..=3 base sequence (xorshift). Long enough that
/// 32-mers are effectively unique, so a read's maximal match has occ == 1.
fn random_bases(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x & 3) as u8
        })
        .collect()
}

/// Build a mode-2 sidecar from `bases`, optionally with a tiered keep-mask.
fn build(bases: &[u8], keep: Option<Vec<BedInterval>>, prefix: &std::path::Path) {
    let dir = prefix.parent().unwrap();
    let pac = dir.join(format!(
        "{}.pac",
        prefix.file_name().unwrap().to_string_lossy()
    ));
    write_pac(&pac, bases);
    let mask = MaskConfig {
        keep_bed: keep,
        ..Default::default()
    };
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_from_pac_with_config(&pac, prefix, None, mask, 1, Some(cfg)).unwrap();
}

/// Collect SMEMs for `read` and return the canonicalized `(m, n, s)` set (sorted;
/// `k`/SA-index excluded — it is index-specific by design).
fn smem_mns(
    idx: &LearnedIndex,
    read: &[u8],
    bases: &[u8],
    opts: &CollectOpts,
) -> Vec<(u32, u32, i64)> {
    let mut buf = vec![
        Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: 0,
            l: 0,
            s: 0
        };
        2 * read.len() + 8
    ];
    let n = idx
        .collect_smems(read, 0, opts, bases, PacEncoding::Unpacked, &mut buf)
        .unwrap();
    let mut v: Vec<(u32, u32, i64)> = buf[..n].iter().map(|s| (s.m, s.n, s.s)).collect();
    v.sort_unstable();
    v
}

/// The cheap tiered-dispatch pre-reject (`present_anchor`) must discriminate a
/// served read (anchor in the keep-set → present) from an off-keep read (anchor's
/// suffixes dropped → absent), which is the ~1-probe reject signal.
#[test]
fn present_anchor_discriminates_served_from_off_keep() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0x5EED);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let full_p = dir.path().join("full.prmi");
    let z_p = dir.path().join("z.prmi");
    build(&bases, None, &full_p);
    build(&bases, Some(keep), &z_p);
    let full = LearnedIndex::open(&full_p).unwrap();
    let z = LearnedIndex::open(&z_p).unwrap();

    // A read from the kept region: anchor present in Z (and full).
    let served = &bases[3000..3060];
    assert!(
        z.present_anchor(served, &bases, PacEncoding::Unpacked),
        "served anchor must be present in Z"
    );
    assert!(full.present_anchor(served, &bases, PacEncoding::Unpacked));

    // A read from an off-keep region: anchor absent from Z (its suffixes were
    // dropped), present in the full index. This is the cheap-reject signal.
    let off = &bases[256..316];
    assert!(
        !z.present_anchor(off, &bases, PacEncoding::Unpacked),
        "off-keep anchor must be absent from Z"
    );
    assert!(full.present_anchor(off, &bases, PacEncoding::Unpacked));

    // Leading-N contract: `present_anchor` skips N-containing windows and keys
    // off the FIRST N-free 32-mer. An N at position 0 of a served read moves the
    // anchor to `[1..33]`, which is still inside the keep-set, so the read stays
    // present in Z. (An implementation that inspected only `read[0..32]` would
    // wrongly reject it.)
    let mut served_after_n = served.to_vec();
    served_after_n[0] = 4; // N
    assert!(
        z.present_anchor(&served_after_n, &bases, PacEncoding::Unpacked),
        "present_anchor must skip the N window and use the first N-free 32-mer"
    );
    assert!(full.present_anchor(&served_after_n, &bases, PacEncoding::Unpacked));

    // First-N-free-window contract: only the FIRST N-free 32-mer decides. An
    // off-keep anchor followed by a served one must be rejected by Z — a later
    // served 32-mer must NOT override the off-keep first anchor.
    let mut off_then_served = Vec::with_capacity(64);
    off_then_served.extend_from_slice(&off[..32]);
    off_then_served.extend_from_slice(&served[..32]);
    assert!(
        !z.present_anchor(&off_then_served, &bases, PacEncoding::Unpacked),
        "present_anchor must inspect only the first N-free 32-mer, not later windows"
    );
    assert!(full.present_anchor(&off_then_served, &bases, PacEncoding::Unpacked));
}

/// Per-SMEM genome positions: decode each occurrence in `[k, k+occ)` from its
/// doubled-coordinate SA position to a forward genome coordinate
/// (`p < l_pac ? p : 2*l_pac-1-p`). Native coords means these must match the
/// whole-genome index for served reads — the end-to-end check the `(m,n,s)` diff
/// (which excludes the index-specific `k`) cannot make.
fn smem_genome_positions(
    idx: &LearnedIndex,
    read: &[u8],
    bases: &[u8],
    opts: &CollectOpts,
) -> Vec<(u32, u32, Vec<u64>)> {
    let mut buf = vec![
        Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: 0,
            l: 0,
            s: 0
        };
        2 * read.len() + 8
    ];
    let n = idx
        .collect_smems(read, 0, opts, bases, PacEncoding::Unpacked, &mut buf)
        .unwrap();
    let lp = idx.l_pac();
    let mut out: Vec<(u32, u32, Vec<u64>)> = buf[..n]
        .iter()
        .map(|s| {
            let mut pos: Vec<u64> = (s.k..s.k + s.s)
                .map(|i| {
                    let p = idx.sa_position_for(i as u64);
                    if p < lp {
                        p
                    } else {
                        2 * lp - 1 - p
                    }
                })
                .collect();
            pos.sort_unstable();
            (s.m, s.n, pos)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn served_smem_genome_positions_match_full() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0xD0E2_7A11);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let full_p = dir.path().join("full.prmi");
    let z_p = dir.path().join("z.prmi");
    build(&bases, None, &full_p);
    build(&bases, Some(keep), &z_p);
    let full = LearnedIndex::open(&full_p).unwrap();
    let z = LearnedIndex::open(&z_p).unwrap();

    let opts = CollectOpts {
        min_seed_len: 19,
        split_len: 24,
        split_width: 10,
        max_mem_intv: 20,
    };
    let read_len = 60usize;
    let mut checked = 0usize;
    for start in (2048 + 64..6144 - 64).step_by(289) {
        let read = &bases[start..start + read_len];
        let f = smem_genome_positions(&full, read, &bases, &opts);
        let zz = smem_genome_positions(&z, read, &bases, &opts);
        // Served interior read: identical SMEMs AND identical genome positions.
        assert_eq!(f, zz, "served read at {start}: genome positions diverged");
        assert!(!f.is_empty());
        checked += 1;
    }
    assert!(checked >= 8);
}

/// Tiered correctness on REPETITIVE sequence — the case that exercises the
/// forward-progress guard's *firing* path (random sequence has unique k-mers and
/// can never stall the reseed walk; low-copy homology with a partial keep-set
/// makes the left RC span and right extension disagree, as in the i31 hang).
///
/// Asserts: (1) the tiered walk TERMINATES (the test completing is the proof — a
/// regressed guard would hang here), and (2) the SUBSET property holds — for any
/// `(m,n)` present in both indexes, `occ_Z <= occ_full` (the tiered SA can only
/// *lose* occurrences; it must never inflate `occ`).
#[test]
fn keep_masked_repetitive_terminates_and_is_occ_subset() {
    let dir = tempdir().unwrap();
    // LOW-COPY homology: a 150 bp "gene" duplicated at 4 scattered loci within
    // unique filler. occ=4 <= split_width, so its seeds ARE reseeded (a high-copy
    // tandem array would have occ >> split_width and never reseed). A keep-set
    // that retains only SOME copies makes the reseed's left RC span and right
    // extension disagree over the partially-present copies — the stall the
    // forward-progress guard terminates.
    let gene = random_bases(150, 0x6E11E);
    let mut bases = random_bases(600, 0xA11CE);
    let mut loci = Vec::new();
    for seed in [0xB0Bu64, 0xCAFE, 0xD00D, 0xFEED] {
        loci.push(bases.len());
        bases.extend_from_slice(&gene);
        bases.extend(random_bases(600, seed)); // unique filler between copies
    }
    // Keep copy 0 (+flanks) and only the FRONT HALF of copy 1 — so a seed
    // spanning copy 1 has some copies present, some absent, partially.
    let keep = vec![
        BedInterval {
            start: (loci[0] as u64).saturating_sub(80),
            end: loci[0] as u64 + 230,
        },
        BedInterval {
            start: loci[1] as u64,
            end: loci[1] as u64 + 75,
        },
    ];

    let full_p = dir.path().join("full.prmi");
    let z_p = dir.path().join("z.prmi");
    build(&bases, None, &full_p);
    build(&bases, Some(keep), &z_p);
    let full = LearnedIndex::open(&full_p).unwrap();
    let z = LearnedIndex::open(&z_p).unwrap();

    let opts = CollectOpts {
        min_seed_len: 12,
        split_len: 16,
        split_width: 8,
        max_mem_intv: 20,
    };

    // Walk reads across the whole repetitive region (these are the reads that drive
    // reseed over the partially-kept tandem copies). If the guard regressed, one of
    // these collect_smems calls would never return and the test would hang.
    for start in (520..bases.len() - 90).step_by(11) {
        let read = &bases[start..start + 80];
        let f: std::collections::HashMap<(u32, u32), i64> = smem_mns(&full, read, &bases, &opts)
            .into_iter()
            .map(|(m, n, s)| ((m, n), s))
            .collect();
        let z_set = smem_mns(&z, read, &bases, &opts);
        for (m, n, s_z) in z_set {
            // Independent oracle: the full index's occurrence count of the EXACT
            // Z span substring `read[m..=n]`, computed directly via `mem_search`
            // rather than read off the full collector's SMEM set. The full
            // collector only emits spans that are maximal in the full index, so a
            // span emitted ONLY by the tiered index is absent from `f` and would
            // silently bypass the "must never inflate occ" contract. The tiered
            // SA is a position-filtered subset of the SAME doubled text, so every
            // Z span must occur in the full index at full length with occ >= the
            // tiered occ (the tiered SA can only LOSE occurrences, never add any).
            let span = &read[m as usize..=n as usize];
            let full_match = full.mem_search(span, &bases, PacEncoding::Unpacked);
            assert_eq!(
                full_match.match_len as usize,
                span.len(),
                "Z span ({m},{n}) at read {start} does not fully occur in the full index"
            );
            let full_occ = full_match.occ as i64;
            assert!(
                s_z <= full_occ,
                "occ inflation at read {start} span ({m},{n}): z occ {s_z} > full occ {full_occ}"
            );
            // Cross-check the oracle itself: where the full collector also emitted
            // this span, its occ must equal the direct `mem_search` occ.
            if let Some(&s_full) = f.get(&(m, n)) {
                assert_eq!(
                    s_full, full_occ,
                    "oracle mismatch at read {start} span ({m},{n}): \
                     collector occ {s_full} != mem_search occ {full_occ}"
                );
            }
        }
    }

    // Independent coverage: the occ-subset + no-hang checks above prove the call
    // RETURNS, but not that the forward-progress stall guard actually FIRED for
    // this synthetic case. With the `spectrum-probe-count` instrumentation, count
    // guard firings directly: the tiered index must fire it (the partially-kept
    // tandem copies are exactly the stall it terminates), and the full index must
    // NOT — the guard is a documented strict no-op on the whole-genome SA. The
    // counter is thread-local and these `smem_mns` calls run on the current
    // thread, so the counts are exact and free of cross-test interference.
    #[cfg(feature = "spectrum-probe-count")]
    {
        use prmi::index::collect::stall_guard;

        stall_guard::reset();
        for start in (520..bases.len() - 90).step_by(11) {
            let read = &bases[start..start + 80];
            let _ = smem_mns(&z, read, &bases, &opts);
        }
        assert!(
            stall_guard::count() > 0,
            "stall guard never fired on the tiered SA (test would have hung without it, \
             but the guard branch must be exercised, not just present)"
        );

        stall_guard::reset();
        for start in (520..bases.len() - 90).step_by(11) {
            let read = &bases[start..start + 80];
            let _ = smem_mns(&full, read, &bases, &opts);
        }
        assert_eq!(
            stall_guard::count(),
            0,
            "stall guard fired on a full SA, but it must be a strict no-op there \
             (the walk always advances on the whole-genome SA)"
        );
    }
}

#[test]
fn keep_masked_sa_is_byte_identical_for_served_reads_and_rejects_off_keep() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0xD0E2_7A11);

    // Keep-set: a single forward interval well inside the genome.
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];

    let full_prefix = dir.path().join("full.prmi");
    let z_prefix = dir.path().join("z.prmi");
    build(&bases, None, &full_prefix);
    build(&bases, Some(keep.clone()), &z_prefix);

    let full = LearnedIndex::open(&full_prefix).unwrap();
    let z = LearnedIndex::open(&z_prefix).unwrap();

    // l_pac must be preserved as the FULL forward length (positions stay native
    // genome coordinates), even though the keep-masked SA has far fewer entries.
    assert_eq!(
        z.l_pac(),
        bases.len() as u64,
        "keep-mask must preserve l_pac"
    );
    assert_eq!(full.l_pac(), bases.len() as u64);
    assert!(
        z.sa_num() < full.sa_num(),
        "keep-masked SA must be smaller: z={} full={}",
        z.sa_num(),
        full.sa_num()
    );

    // The compacted-rank model must be WELL-FIT to the shrunken SA — not a
    // full-SA model whose predictions land out of range and only converge via
    // `find_boundary` expand-on-miss (which would be byte-identical but slow,
    // hiding the working-set win). A well-fit model's error bound is a small
    // fraction of its SA; a clamped full-SA model's would be ~sa_num.
    assert!(
        z.max_error_bound() <= z.sa_num() / 8 + 64,
        "tiered model looks mis-fit: max_error_bound={} vs sa_num={} (expected a well-fit \
         compacted-rank model, not a clamped full-SA model)",
        z.max_error_bound(),
        z.sa_num()
    );

    // A small opts grid spanning passes 1+2 and pass-3.
    let grids = [
        CollectOpts {
            min_seed_len: 19,
            split_len: 24,
            split_width: 10,
            max_mem_intv: 0,
        },
        CollectOpts {
            min_seed_len: 12,
            split_len: 16,
            split_width: 6,
            max_mem_intv: 20,
        },
    ];

    // --- Claim 1: served (fully-interior) reads are byte-identical. ---
    let read_len = 60usize;
    let mut served = 0usize;
    for start in (2048 + 64..6144 - 64).step_by(317) {
        let read = &bases[start..start + read_len];
        for opts in &grids {
            let f = smem_mns(&full, read, &bases, opts);
            let zz = smem_mns(&z, read, &bases, opts);
            assert_eq!(
                f, zz,
                "served read at {start} (opts min_seed_len={}) diverged:\n full={f:?}\n z={zz:?}",
                opts.min_seed_len
            );
            // Sanity: the read really does match (occ==1 full-span) in the full index.
            assert!(
                !f.is_empty(),
                "expected a match for interior read at {start}"
            );
        }
        served += 1;
    }
    assert!(
        served >= 8,
        "test should exercise several served reads, got {served}"
    );

    // --- Claim 2: off-keep reads diverge (would be rejected to fallback). ---
    let mut diverged = 0usize;
    for start in (128..1900).step_by(311) {
        let read = &bases[start..start + read_len];
        let opts = &grids[0];
        let f = smem_mns(&full, read, &bases, opts);
        let zz = smem_mns(&z, read, &bases, opts);
        // The full index finds the off-keep match; the keep-masked index dropped
        // those suffixes, so it must NOT reproduce the full-span SMEM.
        assert!(
            !f.is_empty(),
            "full index should match off-keep read at {start}"
        );
        // A plain `assert_ne!(f, zz)` would pass even if `zz` still reproduced the
        // off-keep full-span SMEM (just with extra/different entries alongside).
        // The contract is stronger: the keep-masked index dropped those suffixes,
        // so it must NOT reproduce the full-span `(m=0, n=read_len-1)` SMEM(s).
        let full_span_end = (read_len - 1) as u32;
        let full_span: Vec<_> = f
            .iter()
            .copied()
            .filter(|(m, n, _)| *m == 0 && *n == full_span_end)
            .collect();
        assert!(
            !full_span.is_empty(),
            "full index should have a full-span SMEM for off-keep read at {start}"
        );
        for smem in full_span {
            assert!(
                !zz.contains(&smem),
                "off-keep read at {start} reproduced full-span SMEM {smem:?}"
            );
        }
        diverged += 1;
    }
    assert!(
        diverged >= 4,
        "test should exercise several off-keep reads, got {diverged}"
    );
}

/// A `--keep-bed` that selects nothing in this reference (here, an interval
/// entirely beyond the genome) can retain only the always-kept sentinel row.
/// That is a user-input failure (the BED matches no reference position), not an
/// internal bug, and the build must fail closed with `Error::InvalidInput`. A
/// keep-set whose every interval starts at/after the genome length is rejected
/// up front by the overlap guard — before the (expensive) doubled GSA is built;
/// the empty-training-set guard further below still backstops subtler keep-sets
/// that overlap the reference but select no trainable 32-mer.
#[test]
fn empty_keep_set_is_rejected_as_invalid_input() {
    let dir = tempdir().unwrap();
    let prefix = dir.path().join("idx");
    let pac = dir.path().join("idx.pac");
    let bases = random_bases(512, 7);
    write_pac(&pac, &bases);
    // Interval entirely beyond the 512-base reference: covers no forward position.
    let keep = vec![BedInterval {
        start: 10_000,
        end: 10_100,
    }];
    let mask = MaskConfig {
        keep_bed: Some(keep),
        ..Default::default()
    };
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    let err = build_sidecar_from_pac_with_config(&pac, &prefix, None, mask, 1, Some(cfg))
        .expect_err("out-of-reference keep-bed selects nothing and must fail");
    assert!(
        matches!(err, prmi::Error::InvalidInput { .. }),
        "expected InvalidInput for an empty keep-set, got {err:?}"
    );
}
