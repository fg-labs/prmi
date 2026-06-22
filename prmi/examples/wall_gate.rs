// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Minimal seeding-wall harness (NO probe counting / NO feature deps): load index,
//! preload+decode reads, loop the per-read SMEM collector `PRMI_REPEAT` times, and
//! report the load-excluded seeding wall via `Instant`. Built WITHOUT
//! `spectrum-probe-count` so the wall is unbiased by the probe counter.
//!
//! `PRMI_SCRATCH` selects the entry point so the SAME binary/index/reads can measure
//! both: `1` (default) holds one `CollectScratch` across all reads and calls
//! `collect_smems_into` (amortized allocations); `0` calls the plain `collect_smems`
//! (a fresh per-read allocation), reproducing the pre-change path.
//!
//! ```text
//! PRMI_PREFIX=/path/ref.prmi PRMI_FQ=/path/reads.fq PRMI_PAC=/path/ref.fa.pac \
//! PRMI_REPEAT=20 PRMI_SCRATCH=1 \
//!   cargo run --release --example wall_gate
//! ```
use prmi::index::collect::{CollectOpts, CollectScratch, Smem};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn env_u(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}
fn env_i(k: &str, d: i64) -> i64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let req = |k: &str| {
        std::env::var(k).unwrap_or_else(|_| {
            eprintln!("missing env {k}");
            std::process::exit(2);
        })
    };
    let prefix = req("PRMI_PREFIX");
    let fq = req("PRMI_FQ");
    let pac_path = req("PRMI_PAC");
    let opts = CollectOpts {
        min_seed_len: env_u("PRMI_MIN_SEED_LEN", 19),
        split_len: env_u("PRMI_SPLIT_LEN", 28),
        split_width: env_i("PRMI_SPLIT_WIDTH", 10),
        max_mem_intv: env_i("PRMI_MAX_MEM_INTV", 20),
    };
    let repeat: u32 = std::env::var("PRMI_REPEAT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // PRMI_SCRATCH=1 (default): reuse one scratch via collect_smems_into.
    // PRMI_SCRATCH=0: fresh per-read alloc via collect_smems.
    let use_scratch = env_u("PRMI_SCRATCH", 1) != 0;

    let idx = LearnedIndex::open(Path::new(&prefix)).expect("open index");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(&pac_path).expect("read pac");
    // Fail fast if the PAC file can't encode all `l_pac` bases (4 per byte),
    // rather than deferring to an out-of-range byte access deep in the seeding
    // walk where the cause would be far harder to diagnose.
    let needed_pac_bytes =
        usize::try_from(l_pac.div_ceil(4)).expect("l_pac too large for this platform");
    assert!(
        pac.len() >= needed_pac_bytes,
        "PAC too short: need >= {needed_pac_bytes} bytes for l_pac={l_pac}, got {}",
        pac.len()
    );
    let enc = PacEncoding::Packed { num_bases: l_pac };
    let f = std::fs::File::open(&fq).expect("open fq");
    let mut lines = BufReader::new(f).lines();
    let mut reads: Vec<Vec<u8>> = Vec::new();
    // Parse strict 4-line FASTQ blocks, failing closed on I/O errors or a
    // truncated final record so a malformed input cannot silently undercount
    // reads and skew the measured throughput.
    while let Some(h) = lines.next() {
        let h = h.expect("read FASTQ header line");
        let seq = lines
            .next()
            .transpose()
            .expect("read FASTQ sequence line")
            .expect("truncated FASTQ: missing sequence line");
        let plus = lines
            .next()
            .transpose()
            .expect("read FASTQ '+' line")
            .expect("truncated FASTQ: missing '+' line");
        let qual = lines
            .next()
            .transpose()
            .expect("read FASTQ quality line")
            .expect("truncated FASTQ: missing quality line");
        assert!(h.starts_with('@'), "invalid FASTQ header: {h}");
        assert!(plus.starts_with('+'), "invalid FASTQ '+' line: {plus}");
        assert_eq!(seq.len(), qual.len(), "FASTQ seq/qual length mismatch");
        reads.push(
            seq.bytes()
                .map(|b| match b {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 4,
                })
                .collect(),
        );
    }
    let zero = Smem {
        rid: 0,
        m: 0,
        n: 0,
        k: 0,
        l: 0,
        s: 0,
    };
    let mut buf = vec![zero; 4096];
    let mut scratch = CollectScratch::new();
    let mut nsmems: u64 = 0;
    let mut n: u64 = 0;
    let t = Instant::now();
    for _ in 0..repeat {
        for (i, read) in reads.iter().enumerate() {
            loop {
                let res = if use_scratch {
                    idx.collect_smems_into(read, i as u32, &opts, &pac, enc, &mut buf, &mut scratch)
                } else {
                    idx.collect_smems(read, i as u32, &opts, &pac, enc, &mut buf)
                };
                match res {
                    Ok(c) => {
                        nsmems += c as u64;
                        break;
                    }
                    Err(need) => buf.resize(need, zero),
                }
            }
            n += 1;
        }
    }
    let el = t.elapsed();
    // Guard against `n == 0` (empty input or `PRMI_REPEAT=0`) before computing
    // per-read metrics, which would otherwise divide by zero.
    if n == 0 {
        eprintln!("no reads processed; check PRMI_REPEAT and FASTQ input");
        std::process::exit(2);
    }
    let path = if use_scratch { "scratch" } else { "alloc" };
    println!(
        "wall_gate path={path} reads={n} repeat={repeat} smems={nsmems} \
         seed_wall_ms={:.3} ns/read={:.1} reads/sec={:.0}",
        el.as_nanos() as f64 / 1e6,
        el.as_nanos() as f64 / n as f64,
        n as f64 / el.as_secs_f64()
    );
}
