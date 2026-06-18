// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Wall-clock harness for the per-read `collect_smems` driver — used to measure the
//! reseed-neighbor-scan (#48) win on a large ref by building this at two commits
//! (pre/post #48) and comparing `ns/read`. Loads a sidecar + forward pac + FASTQ,
//! runs `collect_smems` over every read for `PRMI_REPEAT` passes, prints ns/read.
//!
//! ```text
//! PRMI_PREFIX=ref.prmi PRMI_FQ=reads.fq PRMI_PAC=ref.fa.pac PRMI_REPEAT=10 \
//!   cargo run --release --example collect_wall
//! ```
//! Opts via env (bwa-mem defaults): PRMI_MIN_SEED_LEN=19, PRMI_SPLIT_LEN=28,
//! PRMI_SPLIT_WIDTH=10, PRMI_MAX_MEM_INTV=20.

use prmi::index::collect::{CollectOpts, Smem};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn req(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| {
        eprintln!("collect_wall: required env var {k} not set");
        std::process::exit(2);
    })
}
fn envu<T: std::str::FromStr>(k: &str, d: T) -> T {
    match std::env::var(k) {
        Ok(s) => s.parse().unwrap_or_else(|_| {
            eprintln!("collect_wall: invalid value for {k}: {s}");
            std::process::exit(2);
        }),
        Err(std::env::VarError::NotPresent) => d,
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("collect_wall: env var {k} is not valid unicode");
            std::process::exit(2);
        }
    }
}

fn main() {
    let idx = LearnedIndex::open(Path::new(&req("PRMI_PREFIX"))).expect("open index");
    let l_pac = idx.l_pac();
    let pac = std::fs::read(req("PRMI_PAC")).expect("read pac");
    // `Packed` requires at least `ceil(l_pac / 4)` bytes; a real bwa `.pac` carries a
    // trailing length byte and is slightly larger, so the contract is `>=`, not `==`.
    // Catch a truncated/mismatched pac here rather than letting the SMEM walk read past
    // the end (or silently benchmark a wrong reference).
    let min_pac_len =
        usize::try_from(l_pac.div_ceil(4)).expect("l_pac too large for this platform");
    assert!(
        pac.len() >= min_pac_len,
        "pac file too small for l_pac={l_pac}: got {} bytes, need at least {min_pac_len}",
        pac.len(),
    );
    let enc = PacEncoding::Packed { num_bases: l_pac };
    let repeat = envu::<usize>("PRMI_REPEAT", 10).max(1);
    let opts = CollectOpts {
        min_seed_len: envu("PRMI_MIN_SEED_LEN", 19u32),
        split_len: envu("PRMI_SPLIT_LEN", 28u32),
        split_width: envu("PRMI_SPLIT_WIDTH", 10i64),
        max_mem_intv: envu("PRMI_MAX_MEM_INTV", 20i64),
    };

    let f = std::fs::File::open(req("PRMI_FQ")).expect("open fq");
    let mut lines = BufReader::new(f).lines();
    let mut reads: Vec<Vec<u8>> = Vec::new();
    while let Some(h) = lines.next() {
        let _h = h.expect("read FASTQ header");
        let seq = lines
            .next()
            .expect("truncated FASTQ: missing sequence")
            .expect("read FASTQ sequence");
        let plus = lines
            .next()
            .expect("truncated FASTQ: missing '+' line")
            .expect("read FASTQ '+' line");
        let _qual = lines
            .next()
            .expect("truncated FASTQ: missing quality")
            .expect("read FASTQ quality");
        assert!(
            plus.starts_with('+'),
            "invalid FASTQ record: '+' line missing"
        );
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
    let nr = reads.len();
    assert!(nr > 0, "no reads");
    eprintln!("[collect_wall] reads={nr} repeat={repeat} opts={opts:?}");

    let mut buf: Vec<Smem> = vec![
        Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: 0,
            l: 0,
            s: 0
        };
        4096
    ];
    let pass = |buf: &mut Vec<Smem>| {
        let mut sink = 0u64;
        for (rid, read) in reads.iter().enumerate() {
            loop {
                match idx.collect_smems(black_box(read), rid as u32, &opts, &pac, enc, buf) {
                    Ok(n) => {
                        sink = sink.wrapping_add(n as u64);
                        break;
                    }
                    Err(need) => buf.resize(
                        need,
                        Smem {
                            rid: 0,
                            m: 0,
                            n: 0,
                            k: 0,
                            l: 0,
                            s: 0,
                        },
                    ),
                }
            }
        }
        black_box(sink)
    };

    black_box(pass(&mut buf)); // warm
    let mut ns: Vec<f64> = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t = Instant::now();
        let _ = pass(&mut buf);
        ns.push(t.elapsed().as_nanos() as f64 / nr as f64);
    }
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ns.iter().sum::<f64>() / ns.len() as f64;
    eprintln!(
        "[collect_wall] ns/read: min={:.1} median={:.1} mean={:.1}",
        ns[0],
        ns[ns.len() / 2],
        mean
    );
    println!("{:.2}", ns[ns.len() / 2]); // stdout: median ns/read
}
