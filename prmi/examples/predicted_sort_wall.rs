// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Experiment: does sorting a query batch by its MODEL-PREDICTED SA position make
//! the subsequent serial `mem_search` faster, by turning random cold SA misses into
//! clustered/streaming access? Compares three strategies, paired-interleaved on one
//! binary (same thermal state):
//!
//!   A = serial baseline      (mem_search in read order)
//!   B = predicted-sort serial(compute lookup() pred per query, sort by pred, then
//!                             mem_search in that order; result scatter is trivial)
//!   C = lockstep batch=64    (the good MLP operating point, for reference)
//!
//! B's timing INCLUDES the pred-compute + sort (the real-world overhead). All arms
//! are byte-identical in output (B/C only reorder/interleave independent queries).
//! Also dumps the model `err` (per-leaf error) distribution, which sizes the
//! "materialize the [pred±err] window" idea.
//!
//! ```text
//! PRMI_PREFIX=/path/ref.prmi PRMI_FQ=/path/reads.fq PRMI_PAC=/path/ref.fa.pac \
//! PRMI_REPEAT=14 PRMI_LOCKSTEP_BATCH=64 \
//!   cargo run --release --example predicted_sort_wall
//! ```

use prmi::encoding::tokenize_32mer;
use prmi::index::smem::PacEncoding;
use prmi::index::spectrum::MsTask;
use prmi::index::LearnedIndex;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn require(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| {
        eprintln!("predicted_sort_wall: required env var {k} is not set");
        std::process::exit(2);
    })
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

const KMER_LEN: usize = 32;

fn main() {
    let prefix = require("PRMI_PREFIX");
    let fq = require("PRMI_FQ");
    let pac_path = require("PRMI_PAC");
    let repeat = env_usize("PRMI_REPEAT", 14).max(1);
    let ls_batch = env_usize("PRMI_LOCKSTEP_BATCH", 64).max(1);
    let max_reads = env_usize("PRMI_MAX_READS", usize::MAX);

    let idx = LearnedIndex::open(Path::new(&prefix)).expect("open index");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(&pac_path).expect("read pac");
    let enc = PacEncoding::Packed { num_bases: l_pac };

    let f = std::fs::File::open(&fq).expect("open fq");
    let mut lines = BufReader::new(f).lines();
    let mut queries: Vec<Vec<u8>> = Vec::new();
    // Fail fast on I/O errors or truncated records so a corrupt FASTQ can never
    // silently shrink the benchmark input set (which would skew ns/query).
    // Enforce the cap before consuming each record so `PRMI_MAX_READS=0` reads
    // nothing (checking after a parse would still consume one record).
    while queries.len() < max_reads {
        let Some(_hdr) = lines.next().transpose().expect("read FASTQ header line") else {
            break;
        };
        let seq = lines
            .next()
            .transpose()
            .expect("read FASTQ sequence line")
            .expect("truncated FASTQ: missing sequence line");
        let plus = lines
            .next()
            .transpose()
            .expect("read FASTQ separator line")
            .expect("truncated FASTQ: missing '+' line");
        let qual = lines
            .next()
            .transpose()
            .expect("read FASTQ quality line")
            .expect("truncated FASTQ: missing quality line");
        assert!(
            plus.starts_with('+'),
            "invalid FASTQ separator line: {plus:?}"
        );
        assert_eq!(
            qual.len(),
            seq.len(),
            "FASTQ quality length != sequence length"
        );
        let mut q = Vec::with_capacity(seq.len());
        for b in seq.bytes() {
            let code = match b {
                b'A' | b'a' => 0u8,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => break,
            };
            q.push(code);
        }
        if !q.is_empty() {
            queries.push(q);
        }
    }
    if queries.is_empty() {
        eprintln!("predicted_sort_wall: no usable reads");
        std::process::exit(1);
    }
    let nq = queries.len();

    // err distribution (sizes the window-materialization idea).
    let mut errs: Vec<u64> = queries
        .iter()
        .map(|q| {
            let key = tokenize_32mer(q, q.len().min(KMER_LEN));
            idx.lookup(key).1
        })
        .collect();
    errs.sort_unstable();
    let pct = |p: f64| errs[((errs.len() as f64 * p) as usize).min(errs.len() - 1)];
    eprintln!(
        "[psort] queries={nq} sa_num={} max_error_bound={}",
        idx.sa_num(),
        idx.max_error_bound()
    );
    eprintln!(
        "[psort] per-query err: min={} median={} p90={} p99={} max={} (window=2*err entries)",
        errs[0],
        pct(0.5),
        pct(0.9),
        pct(0.99),
        errs[errs.len() - 1]
    );

    // A: serial baseline.
    let serial = || {
        let mut sink = 0u64;
        for q in &queries {
            let m = idx.mem_search(black_box(q), &pac, enc);
            sink = sink.wrapping_add(m.match_len).wrapping_add(m.occ);
        }
        black_box(sink)
    };
    // B: predicted-sort serial — pred-compute + sort INCLUDED in the timed cost.
    let sorted = || {
        let mut order: Vec<u32> = (0..nq as u32).collect();
        let preds: Vec<u64> = queries
            .iter()
            .map(|q| idx.lookup(tokenize_32mer(q, q.len().min(KMER_LEN))).0)
            .collect();
        order.sort_unstable_by_key(|&i| preds[i as usize]);
        let mut sink = 0u64;
        for &i in &order {
            let m = idx.mem_search(black_box(&queries[i as usize]), &pac, enc);
            sink = sink.wrapping_add(m.match_len).wrapping_add(m.occ);
        }
        black_box(sink)
    };
    // C: small-batch lockstep.
    let lockstep = || {
        let mut sink = 0u64;
        for chunk in queries.chunks(ls_batch) {
            let tasks: Vec<MsTask> = chunk
                .iter()
                .map(|q| MsTask {
                    query: q,
                    seed_hint: None,
                })
                .collect();
            let all = idx.mem_search_lockstep(black_box(&tasks), &pac, enc);
            // Same sink algebra as `serial`/`sorted` (sum of match_len + occ) so the
            // three aggregates are directly comparable in the warm equivalence check.
            sink = sink.wrapping_add(
                all.iter()
                    .map(|m| m.match_len.wrapping_add(m.occ))
                    .fold(0, u64::wrapping_add),
            );
        }
        black_box(sink)
    };

    // Warm + smoke-check that all three strategies agree before timing. They are
    // byte-identical mem_search results, so the aggregates must match exactly.
    let warm_a = serial();
    let warm_b = sorted();
    let warm_c = lockstep();
    assert_eq!(warm_a, warm_b, "serial/sorted aggregate differ");
    assert_eq!(warm_a, warm_c, "serial/lockstep aggregate differ");
    black_box(warm_a);
    black_box(warm_b);
    black_box(warm_c);

    let (mut a_ns, mut b_ns, mut c_ns) = (Vec::new(), Vec::new(), Vec::new());
    let (mut b_pct, mut c_pct) = (Vec::new(), Vec::new());
    for _ in 0..repeat {
        let t = Instant::now();
        black_box(serial());
        let a = t.elapsed().as_nanos() as f64 / nq as f64;
        let t = Instant::now();
        black_box(sorted());
        let b = t.elapsed().as_nanos() as f64 / nq as f64;
        let t = Instant::now();
        black_box(lockstep());
        let c = t.elapsed().as_nanos() as f64 / nq as f64;
        a_ns.push(a);
        b_ns.push(b);
        c_ns.push(c);
        b_pct.push((a - b) / a * 100.0);
        c_pct.push((a - c) / a * 100.0);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        v[v.len() / 2]
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!("[psort] serial   ns/query median={:.1}", med(&mut a_ns));
    eprintln!(
        "[psort] sorted   ns/query median={:.1}  paired-mean speedup vs serial = {:+.2}%",
        med(&mut b_ns),
        mean(&b_pct)
    );
    eprintln!(
        "[psort] lockstep ns/query median={:.1}  paired-mean speedup vs serial = {:+.2}%  (batch={ls_batch})",
        med(&mut c_ns),
        mean(&c_pct)
    );
    println!("sorted={:.4} lockstep={:.4}", mean(&b_pct), mean(&c_pct));
}
