// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Design-Z accept-condition de-risk harness.
//!
//! Opens BOTH a tiered fast-path index (`PRMI_FAST`, e.g. chr22.zh) and the
//! whole-genome truth index (`PRMI_FULL`). For each read it runs the cheap
//! `present_anchor` pre-reject on the fast index, then — for present reads —
//! compares the fast-path `collect_smems` to the whole-genome result
//! (`(m,n,s)` + genome positions) to establish ground-truth `identical`, and
//! evaluates several candidate accept predicates, tallying **false-accepts**
//! (a predicate accepts a read whose served SMEMs are NOT byte-identical to
//! whole-genome — a correctness violation) and **served fraction** for each.
//!
//! The goal: find a predicate with ZERO false-accepts at the highest served
//! fraction. Correctness here is the gate; bwa-mem3 then just plumbs the rule.
//!
//! ```text
//! PRMI_FULL=chr22.full.prmi PRMI_FAST=chr22.zh.prmi PRMI_PAC=chr22.fa.pac \
//!   PRMI_CAP=10 cargo run --release --example z_accept_gate -- on.fq off.fq ...
//! ```

#[cfg(not(feature = "spectrum-probe-count"))]
fn main() {
    eprintln!("z_accept_gate requires --features spectrum-probe-count");
    std::process::exit(2);
}

#[cfg(feature = "spectrum-probe-count")]
fn main() {
    use prmi::index::collect::{CollectOpts, CollectScratch, Smem};
    use prmi::index::smem::PacEncoding;
    use prmi::index::LearnedIndex;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    let require = |k: &str| -> String {
        std::env::var(k).unwrap_or_else(|_| {
            eprintln!("z_accept_gate: required env var {k} is not set");
            std::process::exit(2);
        })
    };
    let full = LearnedIndex::open(Path::new(&require("PRMI_FULL"))).expect("open full");
    let fast = LearnedIndex::open(Path::new(&require("PRMI_FAST"))).expect("open fast");
    let pac = std::fs::read(require("PRMI_PAC")).expect("read pac");
    // Default to 10 only when PRMI_CAP is absent; a present-but-unparsable value is
    // misconfiguration and must fail fast rather than silently run as cap 10.
    let cap: i64 = match std::env::var("PRMI_CAP") {
        Ok(s) => s
            .parse()
            .unwrap_or_else(|e| panic!("invalid PRMI_CAP {s:?}: {e}")),
        Err(std::env::VarError::NotPresent) => 10,
        Err(e) => panic!("read PRMI_CAP: {e}"),
    };
    // A non-positive cap makes `max_s < cap` unsatisfiable for every read, so the
    // P2/P3 predicates silently collapse to zero served and the harness reports
    // nonsense. Fail fast on bad configuration instead.
    assert!(cap > 0, "PRMI_CAP must be > 0 (got {cap})");
    let l_pac = full.l_pac();
    assert_eq!(
        l_pac,
        fast.l_pac(),
        "full and fast must share the genome length"
    );
    // A truncated or wrong-reference PRMI_PAC would let the harness print convincing
    // but meaningless concordance numbers; reject it up front against the genome
    // length both indices agree on (bntpac is 2 bits/base → `l_pac.div_ceil(4)` bytes).
    let expected_pac_bytes =
        usize::try_from(l_pac.div_ceil(4)).expect("l_pac does not fit in usize");
    assert_eq!(
        pac.len(),
        expected_pac_bytes,
        "PRMI_PAC length does not match index l_pac"
    );
    let enc = PacEncoding::Packed { num_bases: l_pac };
    let opts = CollectOpts {
        min_seed_len: 19,
        split_len: 28,
        split_width: 10,
        max_mem_intv: 20,
    };

    // A read's SMEM signature: sorted (m, n, s, sorted genome positions) — the
    // byte-identity-relevant content. Positions are fetched only when (m,n,s)
    // already agree (an (m,n,s) mismatch is divergence regardless of positions).
    // Reuse one `CollectScratch` AND a pair of output buffers across every per-read
    // `collect_smems_into` call (the scratch is cleared on entry, so this stays
    // byte-identical to `collect_smems`) to avoid allocator churn: both the two
    // internal scratch buffers and the 4096-`Smem` output Vec would otherwise be
    // reallocated and zero-filled on each of the two calls per present read.
    let zero = Smem {
        rid: 0,
        m: 0,
        n: 0,
        k: 0,
        l: 0,
        s: 0,
    };
    let mut fast_smems = vec![zero; 4096];
    let mut full_smems = vec![zero; 4096];
    // Collect into the caller-held `buf` (grown on overflow) and return the count;
    // the caller then slices `buf[..n]`. Reused across reads — no per-read alloc.
    let smem_mns = |idx: &LearnedIndex,
                    read: &[u8],
                    buf: &mut Vec<Smem>,
                    scratch: &mut CollectScratch|
     -> usize {
        loop {
            match idx.collect_smems_into(read, 0, &opts, &pac, enc, buf.as_mut_slice(), scratch) {
                Ok(n) => return n,
                Err(need) => buf.resize(need, zero),
            }
        }
    };
    let mut scratch = CollectScratch::new();
    let positions = |idx: &LearnedIndex, k: u64, s: i64| -> Vec<u64> {
        let mut out = vec![0u64; s as usize];
        idx.sa_positions(k, &mut out).expect("sa_positions");
        out.sort_unstable();
        out
    };
    // True iff fast's SMEM set is byte-identical to full's: same (m,n,s) multiset
    // AND, for each, the same genome-position set.
    let identical = |fa: &[Smem], fu: &[Smem]| -> bool {
        let key = |v: &[Smem]| {
            let mut t: Vec<(u32, u32, i64)> = v.iter().map(|s| (s.m, s.n, s.s)).collect();
            t.sort_unstable();
            t
        };
        if key(fa) != key(fu) {
            return false;
        }
        // (m,n,s) agree; confirm positions per matching seed (sort both by (m,n,k)).
        let mut fa_s: Vec<&Smem> = fa.iter().collect();
        let mut fu_s: Vec<&Smem> = fu.iter().collect();
        let ord = |a: &&Smem, b: &&Smem| (a.m, a.n, a.k).cmp(&(b.m, b.n, b.k));
        fa_s.sort_unstable_by(ord);
        fu_s.sort_unstable_by(ord);
        fa_s.iter().zip(&fu_s).all(|(a, b)| {
            a.m == b.m
                && a.n == b.n
                && a.s == b.s
                && positions(&fast, a.k as u64, a.s) == positions(&full, b.k as u64, b.s)
        })
    };

    let files: Vec<String> = std::env::args().skip(1).collect();
    // No input paths → every tally stays zero and the harness would print convincing
    // but meaningless all-zero metrics; surface the bad invocation instead.
    assert!(
        !files.is_empty(),
        "usage: z_accept_gate <reads.fq> [reads2.fq ...]"
    );
    // Tallies. P1 = present; P2 = present & max_s<cap; P3 = present & max_s<cap & full-cover.
    let (mut total, mut absent) = (0u64, 0u64);
    let (mut p_acc_ok, mut p_acc_bad) = ([0u64; 3], [0u64; 3]); // served-correct / FALSE-ACCEPT
    let mut present_total = 0u64;
    let mut present_identical = 0u64;
    let mut diag = [0u64; 4]; // present reads: [identical&lo, identical&hi, divergent&lo, divergent&hi] by max_s_full vs cap
                              // Divergent-span classification across all divergent reads, keyed by (m,n).
    let mut miss_class = [0u64; 3]; // [reseed-like(contained), read-boundary, primary-interior]
    let mut miss_occ = [0u64; 5]; // full-occ of the divergent span: 1,2,3,4,5+
    let (mut div_occ_reduced, mut div_wholly_missing) = (0u64, 0u64);
    let (mut div_occ_inflated, mut div_over_emit) = (0u64, 0u64);

    for path in &files {
        let f = std::fs::File::open(path).unwrap_or_else(|_| panic!("open {path}"));
        let mut lines = BufReader::new(f).lines();
        while let Some(h) = lines.next() {
            // A truncated tail (a record missing its `+` or quality line) must fail
            // fast: counting it would skew present_total, the predicate tallies, and
            // the divergence diagnostics. Require all four FASTQ lines per record.
            let header = h.unwrap_or_else(|e| panic!("read {path}: {e}"));
            assert!(
                header.starts_with('@'),
                "invalid FASTQ in {path}: header line expected, got {header:?}"
            );
            let Some(seq) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing sequence after header");
            };
            let seq = seq.unwrap_or_else(|e| panic!("read {path}: {e}"));
            let Some(plus) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing '+' line after sequence");
            };
            let plus = plus.unwrap_or_else(|e| panic!("read {path}: {e}"));
            let Some(qual) = lines.next() else {
                panic!("truncated FASTQ in {path}: missing quality line after '+'");
            };
            let qual = qual.unwrap_or_else(|e| panic!("read {path}: {e}"));
            assert!(
                plus.starts_with('+'),
                "invalid FASTQ in {path}: '+' line expected, got {plus:?}"
            );
            assert_eq!(
                qual.len(),
                seq.len(),
                "invalid FASTQ in {path}: seq/qual length mismatch"
            );
            let read: Vec<u8> = seq
                .bytes()
                .map(|b| match b {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 4,
                })
                .collect();
            total += 1;
            if !fast.present_anchor(&read, &pac, enc) {
                absent += 1; // → whole-genome fallback (correct by construction)
                continue;
            }
            present_total += 1;
            let fa_n = smem_mns(&fast, &read, &mut fast_smems, &mut scratch);
            let fu_n = smem_mns(&full, &read, &mut full_smems, &mut scratch);
            let fa = &fast_smems[..fa_n];
            let fu = &full_smems[..fu_n];
            let id = identical(fa, fu);
            if id {
                present_identical += 1;
            }
            // Diagnostic: does max occ in the *full* (truth) result exceed the cap?
            // Hypothesis: divergence happens iff the read touches a >C-count k-mer.
            let max_s_full = fu.iter().map(|s| s.s).max().unwrap_or(0);
            let hi_full = max_s_full > cap;
            diag[(!id as usize) * 2 + hi_full as usize] += 1; // [id&lo, id&hi, div&lo, div&hi]
            if !id {
                // Classify divergence by (m,n) SPAN (SMEMs are unique per (m,n) in a
                // read): for each full span, is fast's occ equal / lower / higher /
                // span wholly absent? And which fast spans does full lack (over-emit)?
                let rend = read.len() as u32 - 1;
                let fast_at = |m: u32, n: u32| -> Option<i64> {
                    fa.iter().find(|a| a.m == m && a.n == n).map(|a| a.s)
                };
                let full_has = |m: u32, n: u32| -> bool { fu.iter().any(|g| g.m == m && g.n == n) };
                // A span is reseed-like if strictly contained in another full span.
                let classify = |f: &Smem| -> usize {
                    let contained = fu
                        .iter()
                        .any(|g| (g.m, g.n) != (f.m, f.n) && g.m <= f.m && f.n <= g.n);
                    let boundary = f.m == 0 || f.n == rend;
                    if contained {
                        0
                    } else if boundary {
                        1
                    } else {
                        2
                    } // reseed/boundary/primary
                };
                for f in fu {
                    let divergent_span = match fast_at(f.m, f.n) {
                        Some(s) if s == f.s => false, // matched exactly
                        Some(s) if s < f.s => {
                            // occ-reduced (missing copies)
                            div_occ_reduced += 1;
                            true
                        }
                        Some(_) => {
                            div_occ_inflated += 1; // fast occ > full occ — ALARMING
                            true
                        }
                        None => {
                            // span wholly absent in fast
                            div_wholly_missing += 1;
                            true
                        }
                    };
                    if divergent_span {
                        miss_class[classify(f)] += 1;
                        miss_occ[(f.s.clamp(1, 5) - 1) as usize] += 1;
                    }
                }
                // Over-emit: fast spans full entirely lacks (true subset violation).
                for a in fa {
                    if !full_has(a.m, a.n) {
                        div_over_emit += 1;
                    }
                }
            }
            let max_s = fa.iter().map(|s| s.s).max().unwrap_or(0);
            // full read coverage by served SMEM spans [m, n] (n inclusive).
            let mut covered = vec![false; read.len()];
            for s in fa {
                for p in s.m..=s.n.min(read.len() as u32 - 1) {
                    covered[p as usize] = true;
                }
            }
            let full_cover = covered.iter().all(|&c| c);

            let preds = [true, max_s < cap, max_s < cap && full_cover];
            for (i, &accept) in preds.iter().enumerate() {
                if accept {
                    if id {
                        p_acc_ok[i] += 1;
                    } else {
                        p_acc_bad[i] += 1;
                    }
                }
            }
        }
    }

    let pct = |x: u64| 100.0 * x as f64 / total.max(1) as f64;
    eprintln!("=== z_accept_gate (cap={cap}) ===");
    eprintln!("reads={total}  absent(→fallback)={absent} ({:.1}%)  present={present_total}  of which identical-to-full={present_identical}", pct(absent));
    let names = [
        "P1 present-only",
        "P2 present & max_s<cap",
        "P3 present & max_s<cap & full-cover",
    ];
    for i in 0..3 {
        // A predicate "serves" every read it accepts — both the correct ones and the
        // false-accepts — so the served fraction is `ok + false_accept`; report that
        // total alongside the correct count and the false-accept count.
        let served = p_acc_ok[i] + p_acc_bad[i];
        eprintln!(
            "  {:<38} served={:<6} ({:.1}%)  served-correct={:<6}  FALSE-ACCEPT={}",
            names[i],
            served,
            pct(served),
            p_acc_ok[i],
            p_acc_bad[i]
        );
    }
    eprintln!(
        "present-read divergence vs max_s_full>cap:  identical[lo={} hi={}]  divergent[lo={} hi={}]",
        diag[0], diag[1], diag[2], diag[3]
    );
    let miss_total: u64 = miss_class.iter().sum();
    let mpct = |x: u64| 100.0 * x as f64 / miss_total.max(1) as f64;
    eprintln!("--- divergent-span classification (by (m,n)) over divergent reads ---");
    eprintln!(
        "  divergence kind: occ-reduced(fast<full)={div_occ_reduced}  wholly-missing={div_wholly_missing}  occ-INFLATED(fast>full)={div_occ_inflated}  over-emit(span full lacks)={div_over_emit}",
    );
    eprintln!(
        "  divergent spans={miss_total}:  contained/reseed-like={} ({:.0}%)  read-boundary={} ({:.0}%)  primary-interior={} ({:.0}%)",
        miss_class[0], mpct(miss_class[0]), miss_class[1], mpct(miss_class[1]), miss_class[2], mpct(miss_class[2])
    );
    eprintln!(
        "  divergent-span full-occ:  occ1={} occ2={} occ3={} occ4={} occ5+={}",
        miss_occ[0], miss_occ[1], miss_occ[2], miss_occ[3], miss_occ[4]
    );
}
