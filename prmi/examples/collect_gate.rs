// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Profiling harness for `LearnedIndex::collect_smems`: run the fused per-read
//! SMEM collector over a FASTQ against a built prmi sidecar, write the emitted
//! SMEMs to a TSV (`rid m n k s`, unsorted), and report total SA-probes/read with
//! a per-phase breakdown (pass-1 left/right, reseed left/forward, pass 3) from the
//! `attrib` buckets — so the per-read probe budget can be attributed to each search
//! phase without an end-to-end consumer round-trip.
//!
//! Requires the `spectrum-probe-count` feature (the probe counter + `attrib`
//! buckets are compiled out without it):
//!
//! ```text
//! PRMI_PREFIX=/path/to/ref.prmi \
//! PRMI_FQ=/path/to/reads.fq \
//! PRMI_PAC=/path/to/ref.fa.pac \
//!   cargo run --release --features spectrum-probe-count --example collect_gate
//! ```
//!
//! Required env: `PRMI_PREFIX` (sidecar prefix), `PRMI_FQ` (reads), `PRMI_PAC`
//! (forward bntpac, packed). Optional: `PRMI_OUT` (TSV path, default
//! `/tmp/collect_smems.tsv`) and the `CollectOpts` knobs `PRMI_MIN_SEED_LEN` /
//! `PRMI_SPLIT_LEN` / `PRMI_SPLIT_WIDTH` / `PRMI_MAX_MEM_INTV` (bwa-mem defaults).
//! stdout is just the probes/read number, for scripting.

#[cfg(not(feature = "spectrum-probe-count"))]
fn main() {
    eprintln!(
        "collect_gate requires the `spectrum-probe-count` feature:\n  \
         cargo run --release --features spectrum-probe-count --example collect_gate"
    );
    std::process::exit(2);
}

#[cfg(feature = "spectrum-probe-count")]
fn main() {
    use prmi::index::collect::{CollectOpts, Smem};
    use prmi::index::smem::PacEncoding;
    use prmi::index::spectrum::probe_count;
    use prmi::index::LearnedIndex;
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use std::path::Path;

    // Required paths — no machine-specific defaults; fail with a clear message.
    let require = |k: &str| -> String {
        std::env::var(k).unwrap_or_else(|_| {
            eprintln!("collect_gate: required env var {k} is not set (see the module docs)");
            std::process::exit(2);
        })
    };
    let prefix = require("PRMI_PREFIX");
    let fq = require("PRMI_FQ");
    let pac_path = require("PRMI_PAC");
    let out_path = std::env::var("PRMI_OUT").unwrap_or_else(|_| "/tmp/collect_smems.tsv".into());

    let env_u = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let env_i = |k: &str, d: i64| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let opts = CollectOpts {
        min_seed_len: env_u("PRMI_MIN_SEED_LEN", 19),
        split_len: env_u("PRMI_SPLIT_LEN", 28),
        split_width: env_i("PRMI_SPLIT_WIDTH", 10),
        max_mem_intv: env_i("PRMI_MAX_MEM_INTV", 20),
    };
    eprintln!("[collect_gate] opts={opts:?} prefix={prefix}");

    let idx = LearnedIndex::open(Path::new(&prefix)).expect("open index");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(&pac_path).expect("read pac");
    let enc = PacEncoding::Packed { num_bases: l_pac };

    let f = std::fs::File::open(&fq).expect("open fq");
    let mut lines = BufReader::new(f).lines();
    let mut out = BufWriter::new(std::fs::File::create(&out_path).expect("create out"));
    let zero = Smem {
        rid: 0,
        m: 0,
        n: 0,
        k: 0,
        l: 0,
        s: 0,
    };
    let mut buf: Vec<Smem> = vec![zero; 4096];

    probe_count::reset();
    probe_count::reset_depth_probes();
    prmi::index::collect::attrib::reset();
    let (mut nreads, mut nsmems) = (0u32, 0u64);
    while let Some(Ok(_hdr)) = lines.next() {
        let Some(Ok(seq)) = lines.next() else { break };
        let _ = lines.next(); // '+'
        let _ = lines.next(); // qual
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
        let rid = nreads;
        loop {
            match idx.collect_smems(&read, rid, &opts, &pac, enc, &mut buf) {
                Ok(n) => {
                    for s in &buf[..n] {
                        writeln!(out, "{}\t{}\t{}\t{}\t{}", s.rid, s.m, s.n, s.k, s.s).unwrap();
                    }
                    nsmems += n as u64;
                    break;
                }
                Err(need) => buf.resize(need, zero),
            }
        }
        nreads += 1;
    }
    out.flush().unwrap();
    if nreads == 0 {
        eprintln!("[collect_gate] no reads in {fq}");
        std::process::exit(1);
    }
    let total = probe_count::get();
    let buckets = probe_count::depth_probes();
    let per_read = total as f64 / nreads as f64;
    eprintln!(
        "[collect_gate] reads={nreads} smems={nsmems} total_probes={total} probes/read={per_read:.1}"
    );
    eprintln!("[collect_gate] depth buckets (probes by sub-search) = {buckets:?}");
    let attrib = prmi::index::collect::attrib::snapshot();
    let labels = prmi::index::collect::attrib::LABELS;
    eprintln!("[collect_gate] per-phase probes/read (disjoint buckets):");
    for (label, bucket_total) in labels.iter().zip(attrib.iter()) {
        eprintln!(
            "    {label:9} = {:.1}",
            *bucket_total as f64 / nreads as f64
        );
    }
    eprintln!("[collect_gate] wrote {out_path}");
    // stdout: just the probes/read number, for scripting.
    println!("{per_read:.4}");
}
