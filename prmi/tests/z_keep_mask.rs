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

/// Build a mode-2 tiered sidecar from `bases` that ALSO emits a `.blm` bloom gate.
fn build_with_bloom(bases: &[u8], keep: Vec<BedInterval>, prefix: &std::path::Path) {
    let dir = prefix.parent().unwrap();
    let pac = dir.join(format!(
        "{}.pac",
        prefix.file_name().unwrap().to_string_lossy()
    ));
    write_pac(&pac, bases);
    let mask = MaskConfig {
        keep_bed: Some(keep),
        ..Default::default()
    };
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_bloom(0.01);
    build_sidecar_from_pac_with_config(&pac, prefix, None, mask, 1, Some(cfg)).unwrap();
}

/// Build a mode-2 tiered sidecar with a Lever 3 `routing_pad`-padded `.blm` (the
/// routing bloom covers keep ± pad; the `.sa` keep-mask stays tight).
fn build_with_routing_pad(bases: &[u8], keep: Vec<BedInterval>, pad: u64, prefix: &std::path::Path) {
    let dir = prefix.parent().unwrap();
    let pac = dir.join(format!(
        "{}.pac",
        prefix.file_name().unwrap().to_string_lossy()
    ));
    write_pac(&pac, bases);
    let mask = MaskConfig {
        keep_bed: Some(keep),
        ..Default::default()
    };
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_bloom(0.01)
        .with_routing_pad(pad);
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

/// Independent ANY-window oracle: scan EVERY N-free 32-mer window via `mem_search`
/// — a different code path than the gates' internal loop — and report whether any
/// matches at `>= 32` on this (tiered) index. The any-window gates
/// (`present_anchor_any`, `present_anchor_bloom`) must agree with it. Validating
/// against this rather than against another production gate keeps the check from
/// being a self-consistency test of shared first-window/N-skip code.
fn any_window_present_oracle(
    idx: &LearnedIndex,
    read: &[u8],
    bases: &[u8],
    e: PacEncoding,
) -> bool {
    const K: usize = 32;
    // `windows(K)` is a deliberately different traversal than the gate's manual
    // `start`/N-skip loop, so an off-by-one in that loop can't pass both impl and
    // oracle. The N-free windows it visits are exactly the gate's anchor windows.
    read.windows(K)
        .filter(|w| !w.iter().any(|&b| b >= 4)) // skip N-containing windows
        .any(|w| idx.mem_search(w, bases, e).match_len as usize >= K)
}

/// Independent FIRST-window oracle: the verdict of the FIRST N-free 32-mer window
/// only (`mem_search(...).match_len >= 32`). The first-window gates
/// (`present_anchor`, `present_anchor_exact`) must agree with it.
fn first_window_present_oracle(
    idx: &LearnedIndex,
    read: &[u8],
    bases: &[u8],
    e: PacEncoding,
) -> bool {
    const K: usize = 32;
    // First N-free window via `windows(K)` (a different traversal than the gates).
    read.windows(K)
        .find(|w| !w.iter().any(|&b| b >= 4))
        .is_some_and(|w| idx.mem_search(w, bases, e).match_len as usize >= K)
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

/// The any-window gate (`present_anchor_any`, Design-Z item 6) recovers a
/// boundary read that the first-window `present_anchor` mis-routes: a read whose
/// 5' 32-mer starts just BEFORE the keep interval (so its suffix-start is dropped
/// from the tiered SA → first-window miss) but which extends INTO the keep-set
/// (a later window's suffix-start is retained → any-window hit). A fully-off-keep
/// read still misses under both.
#[test]
fn present_anchor_any_recovers_boundary_read() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0x5EED);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let z_p = dir.path().join("z.prmi");
    build(&bases, Some(keep), &z_p);
    let z = LearnedIndex::open(&z_p).unwrap();
    let e = PacEncoding::Unpacked;

    // Independent oracle: the module-level `any_window_present_oracle` scans every
    // N-free 32-mer window via mem_search (a different code path than
    // present_anchor_any's internal loop). present_anchor_any must agree with it.

    // Boundary read [2020, 2080): the first 32-mer starts at 2020 (< 2048, not
    // kept → first-window miss), but the window at read-offset 28 starts at 2048
    // (kept → any-window hit).
    let boundary = &bases[2020..2080];
    assert!(
        !z.present_anchor(boundary, &bases, e),
        "first-window gate mis-routes the boundary read"
    );
    assert!(
        z.present_anchor_any(boundary, &bases, e),
        "any-window gate recovers the boundary read"
    );

    // A served read (interior to the keep-set) is present under both gates.
    let served = &bases[3000..3060];
    assert!(z.present_anchor(served, &bases, e));
    assert!(z.present_anchor_any(served, &bases, e));

    // A fully off-keep read (every window's suffix-start dropped) misses both.
    let off = &bases[256..316];
    assert!(!z.present_anchor(off, &bases, e));
    assert!(
        !z.present_anchor_any(off, &bases, e),
        "a fully off-keep read is not recoverable by scanning windows"
    );

    // present_anchor_any agrees with the independent any-window oracle on all
    // three reads (cross-checks the gate against a separate mem_search scan).
    for r in [boundary, served, off] {
        assert_eq!(
            z.present_anchor_any(r, &bases, e),
            any_window_present_oracle(&z, r, &bases, e),
            "present_anchor_any must match the independent any-window oracle"
        );
    }
}

/// The bloom gate (`present_anchor_bloom`, the production `prmi_present_bloom`
/// path) must reproduce the EXACT any-window verdict when a `.blm` is loaded: the
/// bloom has no false negatives over the keep-set's 32-mer keys (so it serves
/// every read `present_anchor_any` serves), and the confirming `mem_search`
/// removes every bloom false positive (so it serves NO read `present_anchor_any`
/// rejects). And with NO `.blm` loaded it must degrade to the cheap first-window
/// `present_anchor`, never the expensive any-window scan.
#[test]
fn present_anchor_bloom_matches_any_window_with_blm_and_first_window_without() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0x5EED);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let e = PacEncoding::Unpacked;

    // (a) Tiered index WITH a `.blm`: bloom gate == any-window gate.
    let blm_p = dir.path().join("zblm.prmi");
    build_with_bloom(&bases, keep.clone(), &blm_p);
    let zb = LearnedIndex::open(&blm_p).unwrap();
    assert!(zb.has_bloom(), "build_with_bloom must produce a loadable .blm");

    // (b) Same tiered index WITHOUT a `.blm`: bloom gate == first-window gate.
    let noblm_p = dir.path().join("znoblm.prmi");
    build(&bases, Some(keep), &noblm_p);
    let zn = LearnedIndex::open(&noblm_p).unwrap();
    assert!(!zn.has_bloom(), "plain build must not produce a .blm");

    // Reads spanning the served / boundary / off-keep cases plus a dense sweep,
    // so the equivalence is exercised on hits, recoverable boundary reads, and
    // misses alike.
    let boundary = &bases[2020..2080]; // first window off-keep, later window kept
    let served = &bases[3000..3060]; // interior, present under every gate
    let off = &bases[256..316]; // fully off-keep, absent under every gate
    let mut checked = 0usize;
    for read in [boundary, served, off] {
        // (a) With the bloom: matches the INDEPENDENT any-window oracle (a separate
        // mem_search scan), not just the sibling `present_anchor_any` gate.
        assert_eq!(
            zb.present_anchor_bloom(read, &bases, e),
            any_window_present_oracle(&zb, read, &bases, e),
            "bloom gate must match the independent any-window oracle"
        );
        // (b) Without the bloom: matches the INDEPENDENT first-window oracle (the
        // gate degrades to the cheap first-window verdict).
        assert_eq!(
            zn.present_anchor_bloom(read, &bases, e),
            first_window_present_oracle(&zn, read, &bases, e),
            "no-.blm gate must match the independent first-window oracle"
        );
        checked += 1;
    }
    // Spot the three intended verdicts so the equivalences above aren't vacuous.
    assert!(zb.present_anchor_bloom(boundary, &bases, e), "boundary read should be served by the bloom gate");
    assert!(!zn.present_anchor_bloom(boundary, &bases, e), "boundary read is mis-routed by the first-window gate");
    assert!(zb.present_anchor_bloom(served, &bases, e));
    assert!(!zb.present_anchor_bloom(off, &bases, e), "fully off-keep read must miss the bloom gate");

    // Dense sweep across the boundary so any false negative/positive in the
    // bloom+confirm path (vs the exact any-window gate) surfaces.
    for start in (1980..2140).step_by(4) {
        let read = &bases[start..start + 60];
        assert_eq!(
            zb.present_anchor_bloom(read, &bases, e),
            any_window_present_oracle(&zb, read, &bases, e),
            "bloom gate diverged from the independent any-window oracle at read start {start}"
        );
        checked += 1;
    }
    assert!(checked >= 40);
}

/// Lever 2, A1 — the cheap first-window BLOOM gate (`present_anchor_bloom_first`,
/// the `prmi_present_bloom_first` path) has NO false negatives over the keep-set:
/// every read the exact first-window `present_anchor` serves, the bloom gate also
/// serves. It may add bloom false positives (admitting an off-keep read), which the
/// Design-Z consumer's present-read fallback absorbs — so we assert the
/// no-false-negative direction, that off-keep reads are MOSTLY rejected (low FP),
/// and that with NO `.blm` it degrades to the exact first-window verdict.
#[test]
fn present_anchor_bloom_first_has_no_false_negatives_vs_first_window() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0x5EED);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let e = PacEncoding::Unpacked;

    // (a) Tiered index WITH a `.blm`.
    let blm_p = dir.path().join("zblm.prmi");
    build_with_bloom(&bases, keep.clone(), &blm_p);
    let zb = LearnedIndex::open(&blm_p).unwrap();
    assert!(zb.has_bloom(), "build_with_bloom must produce a loadable .blm");

    // (b) Same tiered index WITHOUT a `.blm`.
    let noblm_p = dir.path().join("znoblm.prmi");
    build(&bases, Some(keep), &noblm_p);
    let zn = LearnedIndex::open(&noblm_p).unwrap();
    assert!(!zn.has_bloom(), "plain build must not produce a .blm");

    // No false negatives: every read served by the exact first-window gate is also
    // served by the bloom first-window gate. Count off-keep false positives to
    // confirm the gate is selective (bloom built at fp=0.01).
    let mut served = 0usize;
    let mut off_keep = 0usize;
    let mut off_keep_fp = 0usize;
    for start in (64..bases.len() - 60).step_by(7) {
        let read = &bases[start..start + 60];
        let exact = zb.present_anchor(read, &bases, e);
        let bloom = zb.present_anchor_bloom_first(read, &bases, e);
        if exact {
            assert!(
                bloom,
                "bloom first-window gate must have no false negatives (read start {start})"
            );
            served += 1;
        } else {
            off_keep += 1;
            if bloom {
                off_keep_fp += 1;
            }
        }
        // Without a `.blm`, the gate is byte-identical to the exact first-window.
        assert_eq!(
            zn.present_anchor_bloom_first(read, &bases, e),
            zn.present_anchor(read, &bases, e),
            "no-.blm bloom_first gate must equal the first-window verdict (read start {start})"
        );
    }
    assert!(served > 0 && off_keep > 0, "sweep must cover both cases");
    // Selectivity sanity: false positives are a small fraction of off-keep reads
    // (generous bound; the gate is fp=0.01 but the sweep is small).
    assert!(
        (off_keep_fp as f64) < 0.10 * (off_keep as f64),
        "bloom first-window FP rate too high: {off_keep_fp}/{off_keep}"
    );
}

/// Lever 2, A2 — the cheap EXACT first-window gate (`present_anchor_exact`, the
/// `prmi_present_exact` path) produces a verdict BYTE-IDENTICAL to the exact
/// first-window `present_anchor` (`mem_search(window).match_len >= 32`) on every
/// read, because `kmer_exists` computes the same `match_len` while skipping the
/// interval recovery the gate never reads. Also checks `kmer_exists` directly
/// against `mem_search(...).match_len >= 32` over present and absent 32-mers, and
/// the N-containing-first-window skip behaviour.
#[test]
fn present_anchor_exact_equals_first_window_and_kmer_exists_is_exact() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0xE7AC_7000_u64);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let e = PacEncoding::Unpacked;
    let z_p = dir.path().join("zexact.prmi");
    build(&bases, Some(keep), &z_p);
    let z = LearnedIndex::open(&z_p).unwrap();

    // Dense sweep: present_anchor_exact must agree with the INDEPENDENT first-window
    // oracle (a separate mem_search scan), not just the sibling `present_anchor`.
    let mut checked = 0usize;
    for start in (0..bases.len() - 60).step_by(5) {
        let read = &bases[start..start + 60];
        assert_eq!(
            z.present_anchor_exact(read, &bases, e),
            first_window_present_oracle(&z, read, &bases, e),
            "exact gate diverged from the independent first-window oracle at {start}"
        );
        checked += 1;
    }
    assert!(checked >= 100);

    // kmer_exists agrees with mem_search(...).match_len >= 32 directly. A 32-mer
    // lifted from a KEPT region occurs; a synthetic 32-mer (alternating bases not
    // present in this random reference window) is checked for agreement either way.
    for start in [2100usize, 3000, 5000] {
        let kmer = &bases[start..start + 32];
        let via_search = z.mem_search(kmer, &bases, e).match_len as usize >= 32;
        assert_eq!(
            z.kmer_exists(kmer, &bases, e),
            via_search,
            "kmer_exists must agree with mem_search>=32 on a kept 32-mer at {start}"
        );
    }
    // A 32-mer from an OFF-keep region: its suffix-start was dropped from the tiered
    // SA, so both report the same (absent) verdict.
    let off_kmer = &bases[300..332];
    assert_eq!(
        z.kmer_exists(off_kmer, &bases, e),
        z.mem_search(off_kmer, &bases, e).match_len as usize >= 32,
        "kmer_exists must agree with mem_search>=32 on an off-keep 32-mer"
    );

    // N-containing first window: present_anchor skips to the next N-free window;
    // present_anchor_exact must make the same skip and return the same verdict.
    let mut nread = bases[3000..3060].to_vec();
    nread[0] = 4; // N in the first window forces the skip
    assert_eq!(
        z.present_anchor_exact(&nread, &bases, e),
        first_window_present_oracle(&z, &nread, &bases, e),
        "exact gate must skip the N window and match the independent first-window oracle"
    );
}

/// Lever 3 — the routing bloom DECOUPLED from the seeding SA. With `routing_pad`,
/// the `.blm` covers keep ± pad so a flank-starting read whose first 32-mer window
/// sits just before the keep interval (and is therefore mis-routed by both the
/// tight first-window gate AND the unpadded bloom-first gate) is now served by the
/// padded bloom — while the `.sa` keep-mask, and hence `sa_num()`, is IDENTICAL to
/// the tight build (no index bloat). A position beyond the pad stays excluded.
#[test]
fn routing_pad_serves_flank_read_without_growing_the_sa() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0x1EE3);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let e = PacEncoding::Unpacked;

    // Tight bloom (pad 0) vs Lever-3 padded bloom (pad 64), SAME tight keep-bed.
    let tight_p = dir.path().join("ztight.prmi");
    let pad_p = dir.path().join("zpad.prmi");
    build_with_bloom(&bases, keep.clone(), &tight_p);
    build_with_routing_pad(&bases, keep.clone(), 64, &pad_p);
    let zt = LearnedIndex::open(&tight_p).unwrap();
    let zp = LearnedIndex::open(&pad_p).unwrap();
    assert!(zt.has_bloom() && zp.has_bloom());

    // The `.sa` is untouched by routing_pad: identical entry count (no bloat).
    assert_eq!(
        zt.sa_num(),
        zp.sa_num(),
        "routing_pad must not change the seeding SA size"
    );

    // A flank read: first 32-mer window starts at 2020, inside the keep±64 pad
    // [1984, 6208) but OUTSIDE the tight keep [2048, 6144).
    let flank = &bases[2020..2080];
    // Use the EXACT first-window gate for the miss: `bloom_first` admits ~fp_rate
    // false positives by design, so asserting it MISSES would be a too-strong
    // (flaky) claim. The exact gate has no false positives, so its miss soundly
    // proves the flank's first window is off-keep in the tight index.
    assert!(
        !zt.present_anchor_exact(flank, &bases, e),
        "tight exact first-window gate must MISS the flank read (first window not kept)"
    );
    assert!(
        zp.present_anchor_bloom_first(flank, &bases, e),
        "padded bloom-first must SERVE the flank read (first window in the pad)"
    );

    // An interior read is served by both (bloom has no false negatives over the
    // keep-set). (The exact pad BOUNDARY — that a position beyond ±pad is excluded
    // from the padded key set — is asserted in mask::tests via covered_by_bed;
    // bloom_first itself admits ~fp_rate false positives by design, so it is not a
    // sound exclusion oracle here.)
    let interior = &bases[3000..3060];
    assert!(zt.present_anchor_bloom_first(interior, &bases, e));
    assert!(zp.present_anchor_bloom_first(interior, &bases, e));
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

/// `build_sidecar_core` enforces the bloom-gate invariants for direct library
/// callers (who bypass `Cli::run`): a bloom without a keep-set, a routing-pad
/// without a bloom, and an out-of-range fp-rate must all fail fast with
/// `Error::InvalidInput` rather than build a whole-genome `.blm` or a no-op.
#[test]
fn bloom_config_is_validated_for_library_callers() {
    let dir = tempdir().unwrap();
    let pac = dir.path().join("idx.pac");
    let bases = random_bases(512, 11);
    write_pac(&pac, &bases);
    let keep = vec![BedInterval {
        start: 64,
        end: 256,
    }];

    let attempt = |mask: MaskConfig, cfg: TrainerConfig, label: &str| {
        let prefix = dir.path().join(label);
        let err = build_sidecar_from_pac_with_config(&pac, &prefix, None, mask, 1, Some(cfg))
            .expect_err(label);
        assert!(
            matches!(err, prmi::Error::InvalidInput { .. }),
            "{label}: expected InvalidInput, got {err:?}"
        );
    };

    // with_bloom but no keep-set -> whole-genome bloom, rejected.
    attempt(
        MaskConfig::default(),
        TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_bloom(0.01),
        "no_keep",
    );
    // routing_pad > 0 without with_bloom -> no-op padding, rejected.
    attempt(
        MaskConfig {
            keep_bed: Some(keep.clone()),
            ..Default::default()
        },
        TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_routing_pad(8),
        "pad_no_bloom",
    );
    // out-of-range fp-rate, rejected.
    attempt(
        MaskConfig {
            keep_bed: Some(keep.clone()),
            ..Default::default()
        },
        TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_bloom(1.5),
        "bad_fp_high",
    );
    // fp-rate of exactly 0.0 is also out of the open interval (0, 1) — the old
    // `(0.0..1.0)` range admitted it; the explicit open-interval check rejects it.
    attempt(
        MaskConfig {
            keep_bed: Some(keep.clone()),
            ..Default::default()
        },
        TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_bloom(0.0),
        "bad_fp_zero",
    );
}

/// The `.blm` is bound to its index by `ref_digest` (like `.kmt`): a bloom whose
/// digest does not match the loaded sidecar must be IGNORED, not trusted. A stale
/// bloom can omit current keep-set keys (a false negative that would drop servable
/// reads on the unconfirmed `bloom_first` path), so binding is a correctness guard.
#[test]
fn stale_blm_is_rejected_by_digest_binding() {
    let dir = tempdir().unwrap();
    let bases = random_bases(8192, 0xB10);
    let keep = vec![BedInterval {
        start: 2048,
        end: 6144,
    }];
    let prefix = dir.path().join("z.prmi");
    build_with_bloom(&bases, keep, &prefix);

    // Sanity: the freshly built, matching `.blm` loads.
    assert!(
        LearnedIndex::open(&prefix).unwrap().has_bloom(),
        "matching .blm must load"
    );

    let blm = std::path::PathBuf::from(format!("{}.blm", prefix.display()));
    let pristine = std::fs::read(&blm).unwrap();

    // (a) Tamper the on-disk `ref_digest` (bytes [40, 72) of the .blm header) so it
    // no longer matches the index; the loader must drop it (has_bloom() == false).
    let mut bytes = pristine.clone();
    bytes[40] ^= 0xFF;
    std::fs::write(&blm, &bytes).unwrap();
    assert!(
        !LearnedIndex::open(&prefix).unwrap().has_bloom(),
        "a .blm whose ref_digest mismatches the index must be ignored"
    );

    // (b) Tamper the `keyset_digest` (bytes [72, 80)) — this stands in for a bloom
    // built over a DIFFERENT keep-set/routing-pad of the same reference (same
    // sa_num + ref_digest, different key set), which the keyset binding rejects.
    let mut bytes = pristine.clone();
    bytes[72] ^= 0xFF;
    std::fs::write(&blm, &bytes).unwrap();
    assert!(
        !LearnedIndex::open(&prefix).unwrap().has_bloom(),
        "a .blm whose keyset_digest mismatches the index must be ignored"
    );

    // Restoring the pristine bytes makes it load again (proves the tamper, not a
    // build-side failure, is what the loader rejected).
    std::fs::write(&blm, &pristine).unwrap();
    assert!(
        LearnedIndex::open(&prefix).unwrap().has_bloom(),
        "the pristine .blm must load"
    );
}
