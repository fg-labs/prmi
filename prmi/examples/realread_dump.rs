// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Per-query last-mile measurement harness (measurement-only; not shipped).
//!
//! For every 32-mer window of every read in a FASTQ, queries an on-disk prmi
//! index and emits one TSV row capturing the model prediction, the per-leaf
//! error field, the ACTUAL last-mile probe count (requires the
//! `spectrum-probe-count` feature), and the converged SA interval. This lets us
//! join, on real sequencing reads, last-mile cost against genome multiplicity
//! (`occ`) and key position — to test whether last-mile cost tracks multiplicity
//! or extrapolation risk (the open question from the ε-bounded-RMI hybrid note).
//!
//! Usage:
//!   realread_dump <index_prefix> <pac_path> <fastq> <max_reads> [stride]
//!
//! `index_prefix` is the sidecar prefix (e.g. `.../large.prmi`); `pac_path` is
//! the matching `.pac` (2-bit packed). `stride` subsamples windows per read
//! (default 1 = every window). Output TSV goes to stdout.

use prmi::encoding::{base_to_2bit, tokenize_32mer};
use prmi::index::smem::PacEncoding;
use prmi::index::LearnedIndex;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Sentinel 2-bit code for a non-ACGT base (N); any window containing it is skipped.
const NON_ACGT: u8 = 4;

/// Stream every 32-mer window of the input FASTQ through the on-disk index and
/// emit one TSV measurement row per valid window to stdout.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: realread_dump <index_prefix> <pac_path> <fastq> <max_reads> [stride]");
        std::process::exit(2);
    }
    let prefix = &args[1];
    let pac_path = &args[2];
    let fastq = &args[3];
    let max_reads: usize = args[4].parse().expect("max_reads");
    let stride: usize = args.get(5).map(|s| s.parse().expect("stride")).unwrap_or(1);
    // `stride == 0` would never advance `i`, hanging the window loop forever on
    // any read with `n >= 32`; reject it before processing any reads.
    if stride == 0 {
        eprintln!("stride must be > 0");
        std::process::exit(2);
    }

    let idx = LearnedIndex::open(Path::new(prefix)).expect("open index sidecar");
    let pac = std::fs::read(pac_path).expect("read pac");
    let num_bases = idx.l_pac();
    // Fail fast on a truncated pac before any `mem_search`: a 2-bit packed
    // `.pac` needs at least `ceil(num_bases / 4)` bytes. This mirrors the
    // crate-internal `validate_packed_pac` lower-bound invariant — the decoder
    // bounds reads by `l_pac`, so a longer pac is harmless but a shorter one
    // corrupts every measurement or panics deep inside the search.
    let required_pac_bytes = usize::try_from(num_bases.div_ceil(4))
        .expect("index too large for this platform's pointer width");
    if pac.len() < required_pac_bytes {
        eprintln!(
            "pac too short: need >= {required_pac_bytes} bytes for num_bases={num_bases}, got {}",
            pac.len()
        );
        std::process::exit(2);
    }
    let enc = PacEncoding::Packed { num_bases };
    // Report whether a `.kmt` shallow-band accelerator was auto-loaded, so the
    // with-/without-kmt runs are unambiguous.
    eprintln!("kmt_loaded={} sa_num={}", idx.has_kmt(), idx.sa_num());

    let reader = BufReader::new(File::open(fastq).expect("open fastq"));
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(
        out,
        "rid\tqpos\tkey\tpred\terr\tprobes\tmatch_len\tsa_start\tocc\tpresent"
    )
    .unwrap();

    let mut lines = reader.lines();
    let mut rid = 0usize;
    while rid < max_reads {
        // FASTQ record: header / sequence / '+' / quality. Only a clean `None`
        // here is EOF; a read `Err` is a hard I/O failure and must not be
        // mistaken for end-of-input (silent output truncation).
        let hdr = match lines.next() {
            None => break,
            Some(Ok(l)) => l,
            Some(Err(e)) => {
                eprintln!("failed reading FASTQ header at record {rid}: {e}");
                std::process::exit(1);
            }
        };
        // Fail closed on a header lacking the '@' marker, for parity with the
        // '+'/sequence/quality checks below — a misframed stream must not
        // silently feed rows into the measurement output.
        if !hdr.starts_with('@') {
            eprintln!("malformed FASTQ: expected '@' header, got {hdr:?} at record {rid}");
            std::process::exit(1);
        }
        // A consumed header with no following sequence is a truncated record,
        // not clean EOF — fail closed for parity with the '+'/quality lines.
        let seq = match lines.next() {
            Some(Ok(l)) => l,
            None => {
                eprintln!("malformed FASTQ: missing sequence line after record {rid}");
                std::process::exit(1);
            }
            Some(Err(e)) => {
                eprintln!("failed reading FASTQ sequence at record {rid}: {e}");
                std::process::exit(1);
            }
        };
        // Fail closed on truncated/malformed records: require the '+' separator
        // (starting with '+') and the quality line, rather than emitting rows
        // from a partial record.
        let plus = match lines.next() {
            Some(Ok(l)) => l,
            None => {
                eprintln!("malformed FASTQ: missing '+' separator after record {rid}");
                std::process::exit(1);
            }
            Some(Err(e)) => {
                eprintln!("failed reading FASTQ '+' separator at record {rid}: {e}");
                std::process::exit(1);
            }
        };
        if !plus.starts_with('+') {
            eprintln!("malformed FASTQ: expected '+' separator, got {plus:?} at record {rid}");
            std::process::exit(1);
        }
        let qual = match lines.next() {
            Some(Ok(l)) => l,
            None => {
                eprintln!("malformed FASTQ: missing quality line for record {rid}");
                std::process::exit(1);
            }
            Some(Err(e)) => {
                eprintln!("failed reading FASTQ quality at record {rid}: {e}");
                std::process::exit(1);
            }
        };
        // A quality string of a different length than the sequence is a
        // malformed record; reject it rather than feed it to window metrics.
        if qual.len() != seq.len() {
            eprintln!(
                "malformed FASTQ: sequence/quality length mismatch at record {rid} (seq={}, qual={})",
                seq.len(),
                qual.len()
            );
            std::process::exit(1);
        }

        // Encode ASCII bases to 2-bit; N (and any non-ACGT) becomes NON_ACGT so
        // windows spanning it are dropped (the trainer masks N-runs).
        let codes: Vec<u8> = seq
            .as_bytes()
            .iter()
            .map(|&b| base_to_2bit(b).unwrap_or(NON_ACGT))
            .collect();

        let n = codes.len();
        let mut i = 0;
        // `n >= 32` skips reads shorter than a window (adapter-trimmed/short reads)
        // without underflowing `n - 32`; `i <= n - 32` keeps `codes[i..i + 32]`
        // in-bounds, and the saturating increment avoids any usize overflow.
        while n >= 32 && i <= n - 32 {
            let win = &codes[i..i + 32];
            if !win.iter().any(|&c| c > 3) {
                let key = tokenize_32mer(win, 32);
                let (pred, err) = idx.lookup(key);

                #[cfg(feature = "spectrum-probe-count")]
                prmi::index::spectrum::probe_count::reset();
                let mm = idx.mem_search(win, &pac, enc);
                #[cfg(feature = "spectrum-probe-count")]
                let probes = prmi::index::spectrum::probe_count::get();
                #[cfg(not(feature = "spectrum-probe-count"))]
                let probes = 0u64;

                let present = u8::from(mm.match_len == 32);
                writeln!(
                    out,
                    "{rid}\t{i}\t{key}\t{pred}\t{err}\t{probes}\t{}\t{}\t{}\t{present}",
                    mm.match_len, mm.sa_start, mm.occ
                )
                .unwrap();
            }
            i = i.saturating_add(stride);
        }
        rid += 1;
    }
}
