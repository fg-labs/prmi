// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Wall-clock A/B harness for the maximal-match lockstep (`mem_search_lockstep`).
//! Same paired-interleaved structure as `batch_wall.rs` (which benches the full
//! forward-spectrum trace), but here the serial arm loops `mem_search` and the
//! lockstep arm batches `MsTask`s through `mem_search_lockstep`. The maximal-match
//! path issues independent O(log n) cold probes over the full SA with NO warming
//! narrow, so it is expected to benefit more from memory-level parallelism than the
//! forward-spectrum trace did (which warmed the SA region via nested narrowing).
//!
//! Latency-hiding → measure on high-DRAM-latency hardware (Graviton); on Apple
//! Silicon lockstep is expected to lose. Both arms are table-free (no-kmt path).
//!
//! ```text
//! PRMI_PREFIX=/path/ref.prmi PRMI_FQ=/path/reads.fq PRMI_PAC=/path/ref.fa.pac \
//! PRMI_REPEAT=14 PRMI_BATCH=4096 \
//!   cargo run --release --example mem_search_batch_wall
//! ```
//!
//! Env mirrors `batch_wall`: `PRMI_PREFIX`, `PRMI_FQ`, `PRMI_PAC` (required);
//! `PRMI_REPEAT` (default 14), `PRMI_BATCH` (default 4096), `PRMI_MAX_READS`.

use prmi::index::smem::PacEncoding;
use prmi::index::spectrum::MsTask;
use prmi::index::LearnedIndex;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn require(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| {
        eprintln!("mem_search_batch_wall: required env var {k} is not set");
        std::process::exit(2);
    })
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let prefix = require("PRMI_PREFIX");
    let fq = require("PRMI_FQ");
    let pac_path = require("PRMI_PAC");
    let repeat = env_usize("PRMI_REPEAT", 14).max(1); // >=1 so the stats indexing can't panic
    let batch = env_usize("PRMI_BATCH", 4096).max(1);
    let max_reads = env_usize("PRMI_MAX_READS", usize::MAX);

    let idx = LearnedIndex::open(Path::new(&prefix)).expect("open index");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(&pac_path).expect("read pac");
    let enc = PacEncoding::Packed { num_bases: l_pac };

    // Load reads as 2-bit queries, truncating at the first non-ACGT base.
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
        eprintln!("mem_search_batch_wall: no usable reads in {fq}");
        std::process::exit(1);
    }
    let nq = queries.len();
    eprintln!(
        "[ms_batch_wall] queries={nq} batch={batch} repeat={repeat} sa_num={} l_pac={l_pac}",
        idx.sa_num()
    );

    let serial_pass = || {
        let mut sink = 0u64;
        for q in &queries {
            let m = idx.mem_search(black_box(q), &pac, enc);
            sink = sink.wrapping_add(m.match_len).wrapping_add(m.occ);
        }
        black_box(sink)
    };
    let lockstep_pass = || {
        let mut sink = 0u64;
        for chunk in queries.chunks(batch) {
            let tasks: Vec<MsTask> = chunk
                .iter()
                .map(|q| MsTask {
                    query: q,
                    seed_hint: None,
                })
                .collect();
            let all = idx.mem_search_lockstep(black_box(&tasks), &pac, enc);
            sink = sink.wrapping_add(
                all.iter()
                    .map(|m| m.match_len.wrapping_add(m.occ))
                    .fold(0u64, u64::wrapping_add),
            );
        }
        black_box(sink)
    };

    // Warm + cheap end-to-end byte-identity smoke check (serial total == lockstep).
    let warm_s = serial_pass();
    let warm_l = lockstep_pass();
    assert_eq!(warm_s, warm_l, "serial/lockstep aggregate differ");

    let mut serial_ns = Vec::with_capacity(repeat);
    let mut lockstep_ns = Vec::with_capacity(repeat);
    let mut paired_pct = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        let _ = serial_pass();
        let s = t0.elapsed().as_nanos() as f64 / nq as f64;
        let t1 = Instant::now();
        let _ = lockstep_pass();
        let l = t1.elapsed().as_nanos() as f64 / nq as f64;
        serial_ns.push(s);
        lockstep_ns.push(l);
        paired_pct.push((s - l) / s * 100.0);
    }

    let stats = |v: &mut Vec<f64>| -> (f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v.iter().sum::<f64>() / v.len() as f64)
    };
    let (s_min, s_med, s_mean) = stats(&mut serial_ns);
    let (l_min, l_med, l_mean) = stats(&mut lockstep_ns);
    let paired_mean = paired_pct.iter().sum::<f64>() / paired_pct.len() as f64;

    eprintln!(
        "[ms_batch_wall] serial   ns/query: min={s_min:.1} median={s_med:.1} mean={s_mean:.1}"
    );
    eprintln!(
        "[ms_batch_wall] lockstep ns/query: min={l_min:.1} median={l_med:.1} mean={l_mean:.1}"
    );
    eprintln!(
        "[ms_batch_wall] paired mean (serial-lockstep)/serial = {paired_mean:+.2}%  \
         (positive = lockstep faster)"
    );
    println!("{paired_mean:.4}");
}
