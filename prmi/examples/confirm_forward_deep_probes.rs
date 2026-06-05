// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Confirmation harness for the "is a forward deep-launch (item #2) worth it?"
//! question.
//!
//! It measures, per prefix depth `m`, how many cold SA probes the forward
//! spectrum issues (a) WITHOUT the k-mer table and (b) WITH it, over a corpus of
//! reference-lifted queries (which extend to full depth, so the deep bands are
//! exercised). The headline is: once the table resolves the shallow bands
//! (`m <= k`) with zero probes, how many probes remain in the DEEP bands
//! (`m > k`) — that residue is the absolute ceiling on what any forward
//! model-launch could save.
//!
//! Build + run (the probe counters require the feature):
//!   cargo run --release --features spectrum-probe-count \
//!     --example confirm_forward_deep_probes -- <fasta> [k] [query_len] [corpus]
//!
//! Defaults: k=12, query_len=100, corpus=20000.

#[cfg(not(feature = "spectrum-probe-count"))]
fn main() {
    eprintln!(
        "this example needs the probe counters; rebuild with \
         --features spectrum-probe-count"
    );
}

/// Load a FASTA into one-base-per-byte (`0..=3`), mapping any non-ACGT to A
/// exactly as the prmi trainer does (so the in-memory bases match the sidecar).
#[cfg(feature = "spectrum-probe-count")]
fn load_unpacked(fasta: &std::path::Path) -> Vec<u8> {
    let raw = std::fs::read(fasta).expect("read fasta");
    let mut bases = Vec::with_capacity(raw.len());
    let mut in_header = false;
    for &b in &raw {
        match b {
            b'>' => in_header = true,
            b'\n' => in_header = false,
            _ if !in_header => bases.push(match b {
                b'C' | b'c' => 1u8,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => 0,
            }),
            _ => {}
        }
    }
    bases
}

#[cfg(feature = "spectrum-probe-count")]
fn main() {
    use prmi::index::smem::PacEncoding;
    use prmi::index::spectrum::probe_count;
    use prmi::index::LearnedIndex;
    use prmi::train::build_sidecar_with_config;
    use prmi::train::mask::MaskConfig;

    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!(
            "usage: confirm_forward_deep_probes <fasta> [k=12] [query_len=100] [corpus=20000]"
        );
        std::process::exit(2);
    }
    let fasta = std::path::PathBuf::from(&argv[1]);
    let k: u32 = argv.get(2).map_or(12, |s| s.parse().expect("k"));
    let query_len: usize = argv.get(3).map_or(100, |s| s.parse().expect("query_len"));
    let corpus_size: usize = argv.get(4).map_or(20_000, |s| s.parse().expect("corpus"));

    let bases = load_unpacked(&fasta);
    let l_pac = bases.len() as u64;
    // Guard the corpus math below: `l_pac - query_len` underflows for an
    // oversized query, and `max_start / corpus_size` divides by zero for an
    // empty corpus. Both are reachable from CLI args.
    if query_len == 0 || query_len > bases.len() {
        eprintln!("query_len must be in 1..={}", bases.len());
        std::process::exit(2);
    }
    if corpus_size == 0 {
        eprintln!("corpus must be > 0");
        std::process::exit(2);
    }
    eprintln!(
        "[confirm] fasta={fasta:?} l_pac={l_pac} k={k} query_len={query_len} corpus={corpus_size}"
    );

    // Build a sidecar (no on-disk table needed; we build the table in memory).
    let tmp = std::env::temp_dir().join(format!("prmi_confirm_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prefix = tmp.join("ref.prmi");
    let t0 = std::time::Instant::now();
    build_sidecar_with_config(&fasta, &prefix, None, MaskConfig::default(), 0, None)
        .expect("build sidecar");
    let idx = LearnedIndex::open(&prefix).expect("open sidecar");
    let enc = PacEncoding::Unpacked;
    let table = idx.build_kmer_table(k, &bases, enc);
    eprintln!(
        "[confirm] sidecar+table built in {:.1}s (sa_num={}, log2={:.1})",
        t0.elapsed().as_secs_f64(),
        idx.sa_num(),
        (idx.sa_num() as f64).log2()
    );

    // Corpus: `corpus_size` evenly-strided query_len windows lifted from the
    // forward reference (these match the reference, so they extend to full
    // depth `query_len` — exercising every deep band up to query_len).
    let max_start = l_pac as usize - query_len;
    let stride = (max_start / corpus_size).max(1);
    let corpus: Vec<&[u8]> = (0..corpus_size)
        .map(|i| (i * stride) % (max_start + 1))
        .map(|s| &bases[s..s + query_len])
        .collect();
    eprintln!(
        "[confirm] corpus = {} windows of length {query_len}",
        corpus.len()
    );

    // ── Pass A: NO table (forward_spectrum). Per-depth probe histogram. ──
    probe_count::reset_depth_probes();
    for q in &corpus {
        let _ = idx.forward_spectrum(q, &bases, enc);
    }
    let no_table = probe_count::depth_probes();

    // ── Pass B: WITH table (forward_spectrum_tabled). ──
    probe_count::reset_depth_probes();
    for q in &corpus {
        let _ = idx.forward_spectrum_tabled(q, &bases, enc, &table);
    }
    let with_table = probe_count::depth_probes();

    // ── Aggregate: shallow (m<=k) vs deep (m>k). ──
    let ku = k as usize;
    let sum = |h: &[u64], lo: usize, hi: usize| -> u64 { h[lo..hi.min(h.len())].iter().sum() };
    let nt_total: u64 = no_table.iter().sum();
    let nt_shallow = sum(&no_table, 0, ku + 1);
    let nt_deep = sum(&no_table, ku + 1, no_table.len());
    let wt_total: u64 = with_table.iter().sum();
    let wt_shallow = sum(&with_table, 0, ku + 1);
    let wt_deep = sum(&with_table, ku + 1, with_table.len());
    let n = corpus.len() as f64;

    println!(
        "\n=== forward probe profile (per query, corpus={}) ===",
        corpus.len()
    );
    println!(
        "{:<22} {:>14} {:>14} {:>14}",
        "path", "total/q", "shallow m<=k", "deep m>k"
    );
    println!(
        "{:<22} {:>14.2} {:>14.2} {:>14.2}",
        "no table",
        nt_total as f64 / n,
        nt_shallow as f64 / n,
        nt_deep as f64 / n
    );
    println!(
        "{:<22} {:>14.2} {:>14.2} {:>14.2}",
        format!("with table (k={k})"),
        wt_total as f64 / n,
        wt_shallow as f64 / n,
        wt_deep as f64 / n
    );
    println!(
        "\ntable cuts total forward probes by {:.1}%  (shallow m<=k -> 0)",
        100.0 * (nt_total - wt_total) as f64 / nt_total.max(1) as f64
    );
    println!(
        "AFTER the table, deep-band (m>k) probes = {:.2}/query  ({:.1}% of the no-table total).",
        wt_deep as f64 / n,
        100.0 * wt_deep as f64 / nt_total.max(1) as f64
    );
    println!("That residue is the entire ceiling a forward model-launch could touch.\n");

    // Per-depth deep-band detail (only where there are probes), to show the
    // deep interval collapsing toward occ=1.
    println!("per-depth probes/query (deep bands m>k, with table):");
    for (m, probes) in with_table.iter().enumerate().skip(ku + 1) {
        if *probes > 0 {
            println!("  m={m:<3} {:>10.3}/q", *probes as f64 / n);
        }
    }
}
