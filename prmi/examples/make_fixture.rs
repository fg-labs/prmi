// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Deterministic synthetic-fixture generator for the forward-lockstep wall bench
//! (`batch_wall`). Emits a bwa-format forward `.pac` (so `prmi build --pac` yields
//! a byte-matched sidecar) and a FASTQ of reads sampled from the same sequence.
//! Self-contained — an LCG, no `rand`/`Date` deps — so a given seed reproduces the
//! same fixture on any host (e.g. a Graviton instance).
//!
//! ```text
//! FIX_BASES=134217728 FIX_READS=20000 FIX_READLEN=150 FIX_SEED=1 \
//! FIX_OUT=/var/tmp/synth \
//!   cargo run --release --example make_fixture
//! # writes /var/tmp/synth.pac and /var/tmp/synth.reads.fq, then:
//! #   prmi build --pac /var/tmp/synth.pac -o /var/tmp/synth.prmi
//! ```
//!
//! Env: `FIX_BASES` (genome length, default 64 Mbp), `FIX_READS` (default 20000),
//! `FIX_READLEN` (default 150), `FIX_SEED` (default 1), `FIX_SUBS_PCT` (per-base
//! substitution %, default 1), `FIX_OUT` (output prefix, default `/var/tmp/synth`).

use std::io::{BufWriter, Write};

/// Tiny SplitMix64-style PRNG: deterministic, no external deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn base(&mut self) -> u8 {
        (self.next_u64() & 3) as u8
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    // `try_from` rather than `as usize` so an oversized env value fails fast on a
    // 32-bit target instead of silently truncating to a wrong fixture size.
    let nbases = usize::try_from(env_u64("FIX_BASES", 64 * 1024 * 1024))
        .expect("FIX_BASES exceeds usize on this platform");
    let nreads = usize::try_from(env_u64("FIX_READS", 20_000))
        .expect("FIX_READS exceeds usize on this platform");
    let readlen = usize::try_from(env_u64("FIX_READLEN", 150))
        .expect("FIX_READLEN exceeds usize on this platform");
    let seed = env_u64("FIX_SEED", 1);
    let subs_pct = env_u64("FIX_SUBS_PCT", 1);
    assert!(
        subs_pct <= 100,
        "FIX_SUBS_PCT must be a percentage in 0..=100, got {subs_pct}"
    );
    let out = std::env::var("FIX_OUT").unwrap_or_else(|_| "/var/tmp/synth".into());
    assert!(nbases >= readlen, "FIX_BASES must be at least FIX_READLEN");

    // Generate the forward base sequence (0..=3) deterministically.
    let mut rng = Rng(seed.wrapping_add(0xABCD_1234));
    let seq: Vec<u8> = (0..nbases).map(|_| rng.base()).collect();

    // Write the bwa-format forward .pac: ceil-ish data region (floor(l/4)+1 bytes,
    // bwa's convention) packed MSB-first 2 bits/base, then a trailing byte = l%4.
    let data_bytes = nbases / 4 + 1;
    let mut pac = vec![0u8; data_bytes + 1];
    for (i, &b) in seq.iter().enumerate() {
        let shift = (3 - (i & 3)) * 2;
        pac[i >> 2] |= b << shift;
    }
    pac[data_bytes] = (nbases % 4) as u8;
    let pac_path = format!("{out}.pac");
    std::fs::write(&pac_path, &pac).expect("write pac");

    // Sample reads as substrings (+ optional substitutions), distinct RNG stream.
    let bases_char = [b'A', b'C', b'G', b'T'];
    let mut rng2 = Rng(seed.wrapping_add(0x5555_AAAA));
    let fq_path = format!("{out}.reads.fq");
    let mut fq = BufWriter::new(std::fs::File::create(&fq_path).expect("create fq"));
    let mut line = Vec::with_capacity(readlen);
    for r in 0..nreads {
        // Inclusive upper bound: `nbases - readlen` is a valid start (the read then
        // ends exactly at the sequence end), so sample over `0..=(nbases - readlen)`.
        let p = rng2.below((nbases - readlen + 1) as u64) as usize;
        line.clear();
        for j in 0..readlen {
            let mut b = seq[p + j];
            if subs_pct > 0 && rng2.below(100) < subs_pct {
                // Force a real substitution: pick one of the OTHER 3 bases so the
                // realized rate matches `subs_pct` (a free `rng2.base()` could land
                // on the original base, silently understating it).
                let delta = 1 + rng2.below(3) as u8; // 1..=3
                b = (b + delta) & 3;
            }
            line.push(bases_char[b as usize]);
        }
        writeln!(fq, "@r{r}").unwrap();
        fq.write_all(&line).unwrap();
        fq.write_all(b"\n+\n").unwrap();
        fq.write_all(&vec![b'I'; readlen]).unwrap();
        fq.write_all(b"\n").unwrap();
    }
    fq.flush().unwrap();

    eprintln!(
        "[make_fixture] bases={nbases} reads={nreads} readlen={readlen} seed={seed} \
         subs%={subs_pct}\n  pac  = {pac_path} ({} bytes)\n  reads= {fq_path}\n  \
         next: prmi build --pac {pac_path} -o {out}.prmi",
        pac.len()
    );
}
