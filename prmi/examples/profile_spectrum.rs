// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Profiling driver for `forward_spectrum` and `backward_spectrum`.
//!
//! Runs a heavy, realistic workload suitable for CPU profiling with `samply`
//! or `cargo flamegraph`. Also emits coarse per-phase wall-clock attribution
//! via `--phase-time` so hot sections can be identified even without a
//! sampling profiler.
//!
//! # Usage
//!
//!   cargo build --release --example profile_spectrum
//!
//!   # Sampling profiler (function-level breakdown):
//!   samply record --save-only -o /tmp/prof.json -- \
//!       ./target/release/examples/profile_spectrum \
//!       --sidecar /tmp/chr21.prmi \
//!       --fasta /tmp/chr21.fa
//!
//!   # Coarse per-phase timing (no profiler needed):
//!   ./target/release/examples/profile_spectrum \
//!       --sidecar /tmp/chr21.prmi \
//!       --fasta /tmp/chr21.fa \
//!       --phase-time
//!
//! # Flags
//!
//!   --sidecar <prefix>   prmi sidecar prefix (produces <prefix>.{meta,sa,l1,l2})
//!   --fasta   <path>     reference FASTA used to derive the unpacked pac
//!   --n-fwd   <N>        forward_spectrum calls in main workload (default: 500000)
//!   --n-bwd   <N>        backward_spectrum calls in main workload (default: 200000)
//!   --query-len <K>      query length in bases (default: 75)
//!   --corpus-size <C>    number of distinct queries in corpus (default: 1024)
//!   --phase-time         emit coarse per-phase timing and exit

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use prmi::encoding::tokenize_32mer;
use prmi::index::smem::{pac_base_at, PacEncoding};
use prmi::index::LearnedIndex;
#[cfg(feature = "spectrum-probe-count")]
use prmi::train::build_sidecar_from_pac_with_config;
#[cfg(feature = "spectrum-probe-count")]
use prmi::train::config::{MemoryMode, TrainerConfig};
#[cfg(feature = "spectrum-probe-count")]
use prmi::train::mask::MaskConfig;

// ── CLI ────────────────────────────────────────────────────────────────────────

struct Args {
    sidecar: PathBuf,
    fasta: PathBuf,
    n_fwd: usize,
    n_bwd: usize,
    query_len: usize,
    corpus_size: usize,
    phase_time: bool,
    /// `--pac <path>`: build a mode-2 sidecar from this bwa `.pac` and run the
    /// chr17 cold-probe backward measurement (reference vs model launch). Implies
    /// the probe-bench mode; the `.pac` also supplies the reference bases (packed),
    /// so neither `--sidecar` nor `--fasta` is required when this is set.
    pac: Option<PathBuf>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut sidecar = PathBuf::new();
    let mut fasta = PathBuf::new();
    let mut n_fwd: usize = 500_000;
    let mut n_bwd: usize = 200_000;
    let mut query_len: usize = 75;
    let mut corpus_size: usize = 1024;
    let mut phase_time = false;
    let mut pac: Option<PathBuf> = None;

    let mut i = 1;
    // Consume the value following a value-taking flag, exiting with a clean usage
    // error (not an out-of-bounds panic) when a trailing flag has no value.
    macro_rules! value_for {
        () => {{
            let flag = &argv[i];
            i += 1;
            if i >= argv.len() {
                eprintln!("Missing value for {flag}");
                std::process::exit(1);
            }
            &argv[i]
        }};
    }
    while i < argv.len() {
        match argv[i].as_str() {
            "--sidecar" => {
                sidecar = PathBuf::from(value_for!());
            }
            "--fasta" => {
                fasta = PathBuf::from(value_for!());
            }
            "--n-fwd" => {
                n_fwd = value_for!().parse().expect("n-fwd must be integer");
            }
            "--n-bwd" => {
                n_bwd = value_for!().parse().expect("n-bwd must be integer");
            }
            "--query-len" => {
                query_len = value_for!().parse().expect("query-len must be integer");
            }
            "--corpus-size" => {
                corpus_size = value_for!().parse().expect("corpus-size must be integer");
            }
            "--phase-time" => {
                phase_time = true;
            }
            "--pac" => {
                pac = Some(PathBuf::from(value_for!()));
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    if pac.is_none() && (sidecar.as_os_str().is_empty() || fasta.as_os_str().is_empty()) {
        eprintln!("Usage: profile_spectrum --sidecar <prefix> --fasta <path> [options]");
        eprintln!("       profile_spectrum --pac <chr17.fa.pac> [--corpus-size C --query-len K --n-bwd N]");
        eprintln!("       Options: --n-fwd N --n-bwd N --query-len K --corpus-size C --phase-time");
        std::process::exit(1);
    }
    Args {
        sidecar,
        fasta,
        n_fwd,
        n_bwd,
        query_len,
        corpus_size,
        phase_time,
        pac,
    }
}

// ── PAC loading ────────────────────────────────────────────────────────────────

/// Load FASTA and return `(pac_bases, l_pac)` in unpacked (0..=3) encoding.
/// N/ambiguous → 0 (matching the prmi N→A build-time substitution).
fn load_unpacked_pac(fasta_path: &std::path::Path) -> (Vec<u8>, u64) {
    let raw = std::fs::read(fasta_path).expect("read fasta");
    let mut bases = Vec::with_capacity(raw.len());
    let mut in_header = false;
    for &b in &raw {
        match b {
            b'>' => {
                in_header = true;
            }
            b'\n' => {
                in_header = false;
            }
            _ if !in_header => {
                bases.push(match b {
                    b'A' | b'a' => 0u8,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 0, // N/ambiguous → A (same as prmi trainer)
                });
            }
            _ => {}
        }
    }
    let l_pac = bases.len() as u64;
    (bases, l_pac)
}

// ── chr17 cold-probe backward measurement ───────────────────────────────────────

/// Percentile (nearest-rank) of a sorted slice. `p` in 0.0..=1.0.
#[cfg(feature = "spectrum-probe-count")]
fn percentile_sorted(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank]
}

/// Build a mode-2 sidecar from `pac_path` into a temp prefix, then measure
/// `backward_spectrum` (model launch) vs `backward_spectrum_reference` (full-SA)
/// on a chr17-derived anchor corpus: per-left-step SA probes (median + p99) and
/// wall-time per backward call. Requires the `spectrum-probe-count` feature for
/// the probe counts (wall-time is always reported).
#[cfg(feature = "spectrum-probe-count")]
fn run_probe_bench(args: &Args) {
    use prmi::index::spectrum::probe_count;

    let pac_path = args.pac.as_ref().unwrap();
    // Build the sidecar next to a temp prefix (mode-2 = stored keys).
    let tmp = std::env::temp_dir().join(format!("prmi_probe_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let prefix = tmp.join("chr17.prmi");
    eprintln!("[probe-bench] building mode-2 sidecar from {pac_path:?} ...");
    let t0 = Instant::now();
    let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
    build_sidecar_from_pac_with_config(
        pac_path,
        &prefix,
        None,
        MaskConfig::default(),
        0,
        Some(cfg),
    )
    .expect("build sidecar from pac");
    eprintln!(
        "[probe-bench] sidecar built in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let idx = LearnedIndex::open(&prefix).expect("open sidecar");
    let l_pac = idx.l_pac();
    let sa_num = idx.sa_num();
    eprintln!(
        "[probe-bench] sa_num={sa_num} l_pac={l_pac} max_err={} log2(sa_num)={:.1}",
        idx.max_error_bound(),
        (sa_num as f64).log2()
    );

    // Reference bases come from the SAME .pac (packed, MSB-first bntpac), so the
    // compare is consistent with the SA the sidecar was built from.
    let packed = std::fs::read(pac_path).expect("read pac");
    let enc = PacEncoding::Packed { num_bases: l_pac };

    // Build a chr17 read corpus: lift query_len-base windows from the forward pac.
    let corpus = build_corpus_packed(&packed, l_pac, args.query_len, args.corpus_size, enc);
    // Derive backward anchors (forward run at mid-query pivot), as the real driver does.
    let anchors: Vec<(prmi::index::spectrum::SmemStep, Vec<u8>, usize)> = corpus
        .iter()
        .filter_map(|q| {
            let pivot = q.len() / 2;
            let steps = idx.forward_spectrum(&q[pivot..], &packed, enc);
            steps.last().copied().map(|s| (s, q.clone(), pivot))
        })
        .collect();
    eprintln!(
        "[probe-bench] {}/{} queries produced anchors",
        anchors.len(),
        corpus.len()
    );
    assert!(!anchors.is_empty(), "no anchors derived from chr17 corpus");

    // Sanity: the two implementations must produce IDENTICAL traces on chr17.
    let mut mismatches = 0usize;
    for (step, q, pivot) in &anchors {
        let model = idx.backward_spectrum(
            step.sa_start,
            step.occ_count,
            step.match_len,
            q,
            *pivot,
            &packed,
            enc,
        );
        let reference = idx.backward_spectrum_reference(
            step.sa_start,
            step.occ_count,
            step.match_len,
            q,
            *pivot,
            &packed,
            enc,
        );
        if model != reference {
            mismatches += 1;
        }
    }
    eprintln!(
        "[probe-bench] equality on chr17 anchors: {} / {} identical{}",
        anchors.len() - mismatches,
        anchors.len(),
        if mismatches == 0 {
            " (PASS)"
        } else {
            " (MISMATCH!)"
        }
    );
    assert_eq!(
        mismatches, 0,
        "model-launch diverged from reference on chr17"
    );

    // ── Per-left-step probe counts (median + p99) ────────────────────────────────
    let mut ref_probes: Vec<u64> = Vec::new();
    let mut model_probes: Vec<u64> = Vec::new();
    let mut total_steps = 0usize;
    for (step, q, pivot) in &anchors {
        // Reference: count probes per left step.
        probe_count::reset();
        let ref_steps = idx.backward_spectrum_reference(
            step.sa_start,
            step.occ_count,
            step.match_len,
            q,
            *pivot,
            &packed,
            enc,
        );
        let ref_total = probe_count::get();
        // Model: count probes per left step.
        probe_count::reset();
        let model_steps = idx.backward_spectrum(
            step.sa_start,
            step.occ_count,
            step.match_len,
            q,
            *pivot,
            &packed,
            enc,
        );
        let model_total = probe_count::get();
        let n = ref_steps.len().max(1) as u64;
        debug_assert_eq!(ref_steps.len(), model_steps.len());
        total_steps += ref_steps.len();
        // Average probes per left step for this call (both bounds counted).
        ref_probes.push(ref_total / n);
        model_probes.push(model_total / n);
    }
    ref_probes.sort_unstable();
    model_probes.sort_unstable();
    eprintln!("\n=== chr17 backward SA probes per left step ===");
    eprintln!("  total left steps across corpus: {total_steps}");
    eprintln!(
        "  reference (full-SA) : median={:>4}  p99={:>4}",
        percentile_sorted(&ref_probes, 0.5),
        percentile_sorted(&ref_probes, 0.99)
    );
    eprintln!(
        "  model launch        : median={:>4}  p99={:>4}",
        percentile_sorted(&model_probes, 0.5),
        percentile_sorted(&model_probes, 0.99)
    );

    // ── Wall-time per backward call (warm pages: each variant runs the full corpus
    //    a few times so the comparison is steady-state). ──────────────────────────
    let reps = 5usize;
    let mut t_ref = std::time::Duration::ZERO;
    let mut t_model = std::time::Duration::ZERO;
    let mut sink = 0usize;
    for _ in 0..reps {
        let t0 = Instant::now();
        for (step, q, pivot) in &anchors {
            let s = idx.backward_spectrum_reference(
                step.sa_start,
                step.occ_count,
                step.match_len,
                q,
                *pivot,
                &packed,
                enc,
            );
            sink += black_box(&s).len();
        }
        t_ref += t0.elapsed();
        let t0 = Instant::now();
        for (step, q, pivot) in &anchors {
            let s = idx.backward_spectrum(
                step.sa_start,
                step.occ_count,
                step.match_len,
                q,
                *pivot,
                &packed,
                enc,
            );
            sink += black_box(&s).len();
        }
        t_model += t0.elapsed();
    }
    let _ = black_box(sink);
    let calls = (anchors.len() * reps) as f64;
    eprintln!(
        "\n=== chr17 backward wall-time per call ({reps} reps over {} anchors) ===",
        anchors.len()
    );
    eprintln!(
        "  reference (full-SA) : {:>8.2} us/call",
        t_ref.as_secs_f64() * 1e6 / calls
    );
    eprintln!(
        "  model launch        : {:>8.2} us/call",
        t_model.as_secs_f64() * 1e6 / calls
    );

    // ── Forward SA probes by prefix depth (Step-1: WHERE is the forward cold
    //    cost?). The current forward_spectrum binary-searches within the
    //    previous (nested) interval at each prefix length. This bins the cold
    //    probes by depth m, so we can see whether the cost is the shallow bands
    //    (m small, huge intervals → a precomputed k-mer table is the lever) or
    //    the deep/mid bands (which an anchor-at-`p` model launch can collapse).
    let fwd_query_len = corpus.first().map(|q| q.len() / 2).unwrap_or(0);
    probe_count::reset_depth_probes();
    let mut fwd_calls = 0usize;
    for q in &corpus {
        let pivot = q.len() / 2;
        let _ = idx.forward_spectrum(&q[pivot..], &packed, enc);
        fwd_calls += 1;
    }
    let depth = probe_count::depth_probes();
    let grand: u64 = depth.iter().sum();
    let per_call = |x: u64| x as f64 / fwd_calls.max(1) as f64;
    eprintln!(
        "\n=== chr17 forward SA probes by prefix depth ({fwd_calls} calls, forward query_len={fwd_query_len}) ==="
    );
    eprintln!("  total forward probes/call: {:.1}", per_call(grand));
    eprintln!("  depth-bin    probes/call    %of-total");
    let bins: &[(usize, usize)] = &[
        (1, 4),
        (5, 8),
        (9, 12),
        (13, 18),
        (19, 24),
        (25, 31),
        (32, 49),
        (50, 99),
        (100, probe_count::MAX_DEPTH - 1),
    ];
    for &(lo, hi) in bins {
        let s: u64 = (lo..=hi.min(depth.len() - 1)).map(|m| depth[m]).sum();
        if s == 0 {
            continue;
        }
        eprintln!(
            "  m={:>3}-{:<3}   {:>10.2}    {:>5.1}%",
            lo,
            hi,
            per_call(s),
            100.0 * s as f64 / grand.max(1) as f64
        );
    }
    // k-mer table projection: an exact table over the first K bases makes the
    // m<=K bands O(1) — these probes vanish. Residual = what the model launch
    // must collapse on the deep/mid bands.
    for k in [8usize, 12, 16] {
        let saved: u64 = (1..=k.min(depth.len() - 1)).map(|m| depth[m]).sum();
        eprintln!(
            "  table m<={:>2}: removes {:>5.1}% of forward probes  ({:.1} -> {:.1} probes/call residual)",
            k,
            100.0 * saved as f64 / grand.max(1) as f64,
            per_call(grand),
            per_call(grand - saved)
        );
    }

    // ── K=12 table: byte-identity oracle + measured post-table profile ───────────
    eprintln!("\n[probe-bench] building K=12 forward table ...");
    let t0 = Instant::now();
    let table = idx.build_kmer_table(12, &packed, enc);
    eprintln!(
        "[probe-bench] table built in {:.1}s ({} entries)",
        t0.elapsed().as_secs_f64(),
        1usize << 24
    );
    let mut tbl_mismatch = 0usize;
    for q in &corpus {
        let fq = &q[q.len() / 2..];
        if idx.forward_spectrum_tabled(fq, &packed, enc, &table)
            != idx.forward_spectrum(fq, &packed, enc)
        {
            tbl_mismatch += 1;
        }
    }
    eprintln!(
        "[probe-bench] tabled forward equality: {} / {} identical{}",
        corpus.len() - tbl_mismatch,
        corpus.len(),
        if tbl_mismatch == 0 {
            " (PASS)"
        } else {
            " (MISMATCH!)"
        }
    );
    assert_eq!(tbl_mismatch, 0, "tabled forward diverged from reference");

    probe_count::reset_depth_probes();
    for q in &corpus {
        let _ = idx.forward_spectrum_tabled(&q[q.len() / 2..], &packed, enc, &table);
    }
    let tdepth = probe_count::depth_probes();
    let tgrand: u64 = tdepth.iter().sum();
    let shallow_after: u64 = (1..=12).map(|m| tdepth[m]).sum();
    eprintln!("\n=== chr17 forward SA probes WITH K=12 table ===");
    eprintln!(
        "  total: {:.1} probes/call (was {:.1})  ->  {:.0}% reduction",
        per_call(tgrand),
        per_call(grand),
        100.0 * (1.0 - tgrand as f64 / grand.max(1) as f64)
    );
    eprintln!(
        "  shallow (m<=12): {:.2} probes/call (should be ~0)   residual (m>12): {:.1} probes/call",
        per_call(shallow_after),
        per_call(tgrand - shallow_after)
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Probe-bench stub when the counter feature is disabled: explain how to enable it.
#[cfg(not(feature = "spectrum-probe-count"))]
fn run_probe_bench(_args: &Args) {
    eprintln!(
        "[probe-bench] requires the `spectrum-probe-count` feature; rebuild with:\n  \
         cargo build --release --features spectrum-probe-count --example profile_spectrum"
    );
    std::process::exit(1);
}

/// Build a corpus of `query_len`-base queries lifted from the PACKED forward pac.
#[cfg(feature = "spectrum-probe-count")]
fn build_corpus_packed(
    packed: &[u8],
    l_pac: u64,
    query_len: usize,
    corpus_size: usize,
    enc: PacEncoding,
) -> Vec<Vec<u8>> {
    let max_start = l_pac.saturating_sub(query_len as u64);
    assert!(
        max_start > 0,
        "reference too short for query_len={query_len}"
    );
    let mut queries = Vec::with_capacity(corpus_size);
    let mut x = 0xDEAD_BEEF_1234_5678u64;
    for _ in 0..corpus_size {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = x % max_start;
        let q: Vec<u8> = (0..query_len as u64)
            .map(|j| pac_base_at(packed, start + j, enc).unwrap_or(0))
            .collect();
        queries.push(q);
    }
    queries
}

// ── Corpus ─────────────────────────────────────────────────────────────────────

/// Build `corpus_size` queries of length `query_len` lifted from pseudo-random
/// positions in the forward `pac`. Queries are 0..=3 encoded.
fn build_corpus(pac: &[u8], l_pac: u64, query_len: usize, corpus_size: usize) -> Vec<Vec<u8>> {
    let max_start = l_pac.saturating_sub(query_len as u64) as usize;
    assert!(
        max_start > 0,
        "reference too short for query_len={query_len}"
    );
    let mut queries = Vec::with_capacity(corpus_size);
    let mut x = 0xDEAD_BEEF_1234_5678u64;
    for _ in 0..corpus_size {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = (x as usize) % max_start;
        queries.push(pac[start..start + query_len].to_vec());
    }
    queries
}

/// Build a high-occurrence (tandem-repeat) query: cycle through ACGT for `query_len` bases.
fn build_repeat_query(query_len: usize) -> Vec<u8> {
    [0u8, 1, 2, 3]
        .iter()
        .cycle()
        .take(query_len)
        .copied()
        .collect()
}

// ── Per-phase timing harness ───────────────────────────────────────────────────

/// Measure each logical cost centre independently and print ns/call breakdowns.
///
/// Run with --phase-time to get this instead of the full profiler workload.
fn run_phase_timing(idx: &LearnedIndex, pac: &[u8], l_pac: u64, enc: PacEncoding) {
    let corpus = build_corpus(pac, l_pac, 75, 512);
    let sa_num = idx.sa_num();

    eprintln!("\n=== Phase timing breakdown ===");
    eprintln!("  sa_num={sa_num}  l_pac={l_pac}");

    // ── (1) Model lookup: L2 routing + L1 leaf ────────────────────────────────
    const N_LOOKUP: usize = 100_000;
    let t0 = Instant::now();
    let mut dummy = 0u64;
    for q in corpus.iter().cycle().take(N_LOOKUP) {
        // Build a 32-mer key from the query's first 32 bases (0..=3 encoded).
        // tokenize_32mer expects a 0..=3 slice and the number of bases to encode.
        let key = if q.len() >= 32 {
            tokenize_32mer(&q[..32], 32)
        } else {
            0
        };
        let (pos, err) = idx.lookup(key);
        dummy = dummy.wrapping_add(pos).wrapping_add(err);
    }
    let t_lookup = t0.elapsed();
    let _ = black_box(dummy);
    eprintln!(
        "  (1) model lookup       : {:>7} ns/call  ({N_LOOKUP} calls)",
        t_lookup.as_nanos() / N_LOOKUP as u128
    );

    // ── (2) sa_position_for: sequential access (warm cache) ──────────────────
    const N_SA_SEQ: usize = 100_000;
    let step = (sa_num / N_SA_SEQ as u64).max(1);
    let t0 = Instant::now();
    let mut dummy = 0u64;
    for i in 0..N_SA_SEQ as u64 {
        let pos = idx.sa_position_for(i * step % sa_num);
        dummy = dummy.wrapping_add(pos);
    }
    let t_sa_seq = t0.elapsed();
    let _ = black_box(dummy);
    eprintln!(
        "  (2) sa_position_for seq: {:>7} ns/call  ({N_SA_SEQ} sequential, stride={step})",
        t_sa_seq.as_nanos() / N_SA_SEQ as u128
    );

    // ── (3) sa_position_for: random access (cold mmap = true query cost) ─────
    const N_SA_RND: usize = 50_000;
    let t0 = Instant::now();
    let mut x = 0xCAFE_BABE_DEAD_BEEF_u64;
    let mut dummy = 0u64;
    for _ in 0..N_SA_RND {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i = x % sa_num;
        let pos = idx.sa_position_for(i);
        dummy = dummy.wrapping_add(pos);
    }
    let t_sa_rnd = t0.elapsed();
    let _ = black_box(dummy);
    eprintln!(
        "  (3) sa_position_for rnd: {:>7} ns/call  ({N_SA_RND} random = cold mmap)",
        t_sa_rnd.as_nanos() / N_SA_RND as u128
    );

    // ── (4) pac_base_at: random reads into the forward pac ────────────────────
    const N_PAC: usize = 1_000_000;
    let t0 = Instant::now();
    let mut x = 0xFEED_FACE_u64;
    let mut dummy = 0u64;
    for _ in 0..N_PAC {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pos = x % l_pac;
        if let Some(b) = pac_base_at(pac, pos, enc) {
            dummy = dummy.wrapping_add(b as u64);
        }
    }
    let t_pac = t0.elapsed();
    let _ = black_box(dummy);
    eprintln!(
        "  (4) pac_base_at (rnd)  : {:>7} ns/call  ({N_PAC} random pac reads)",
        t_pac.as_nanos() / N_PAC as u128
    );

    // ── (5) full forward_spectrum ─────────────────────────────────────────────
    const N_FWD: usize = 20_000;
    let t0 = Instant::now();
    let mut total_steps = 0usize;
    for q in corpus.iter().cycle().take(N_FWD) {
        let steps = idx.forward_spectrum(q, pac, enc);
        total_steps += steps.len();
        let _ = black_box(&steps);
    }
    let t_fwd = t0.elapsed();
    let avg_steps_fwd = total_steps as f64 / N_FWD as f64;
    eprintln!(
        "  (5) forward_spectrum   : {:>7} ns/call  ({N_FWD} calls, avg_steps={avg_steps_fwd:.1})",
        t_fwd.as_nanos() / N_FWD as u128
    );

    // ── (6) forward_spectrum: high-occ repeat query ───────────────────────────
    let repeat_q = build_repeat_query(75);
    const N_HOC: usize = 2_000;
    let t0 = Instant::now();
    let mut total_steps = 0usize;
    for _ in 0..N_HOC {
        let steps = idx.forward_spectrum(&repeat_q, pac, enc);
        total_steps += steps.len();
        let _ = black_box(&steps);
    }
    let t_hoc = t0.elapsed();
    let avg_steps_hoc = total_steps as f64 / N_HOC as f64;
    eprintln!(
        "  (6) fwd_spectrum(rep)  : {:>7} ns/call  ({N_HOC} calls, avg_steps={avg_steps_hoc:.1})",
        t_hoc.as_nanos() / N_HOC as u128
    );

    // ── (7) backward_spectrum ─────────────────────────────────────────────────
    let anchors: Vec<_> = corpus
        .iter()
        .filter_map(|q| {
            let pivot = q.len() / 2;
            let steps = idx.forward_spectrum(&q[pivot..], pac, enc);
            steps.last().copied().map(|s| (s, q.clone(), pivot))
        })
        .collect();
    const N_BWD: usize = 10_000;
    let t0 = Instant::now();
    let mut total_steps = 0usize;
    for (step, q, pivot) in anchors.iter().cycle().take(N_BWD) {
        let steps = idx.backward_spectrum(
            step.sa_start,
            step.occ_count,
            step.match_len,
            q,
            *pivot,
            pac,
            enc,
        );
        total_steps += steps.len();
        let _ = black_box(&steps);
    }
    let t_bwd = t0.elapsed();
    let avg_steps_bwd = total_steps as f64 / N_BWD as f64;
    eprintln!(
        "  (7) backward_spectrum  : {:>7} ns/call  ({N_BWD} calls, avg_steps={avg_steps_bwd:.1})",
        t_bwd.as_nanos() / N_BWD as u128
    );

    // ── (8) high-occ backward_spectrum ────────────────────────────────────────
    let repeat_pivot = repeat_q.len() / 2;
    let repeat_fwd = idx.forward_spectrum(&repeat_q[repeat_pivot..], pac, enc);
    if let Some(ra) = repeat_fwd.iter().max_by_key(|s| s.occ_count).copied() {
        eprintln!(
            "        repeat anchor: sa_start={} occ_count={} match_len={}",
            ra.sa_start, ra.occ_count, ra.match_len
        );
        const N_HOC_BWD: usize = 500;
        let t0 = Instant::now();
        let mut total_steps = 0usize;
        for _ in 0..N_HOC_BWD {
            let steps = idx.backward_spectrum(
                ra.sa_start,
                ra.occ_count,
                ra.match_len,
                &repeat_q,
                repeat_pivot,
                pac,
                enc,
            );
            total_steps += steps.len();
            let _ = black_box(&steps);
        }
        let t_hoc_bwd = t0.elapsed();
        let avg_steps_hoc_bwd = total_steps as f64 / N_HOC_BWD as f64;
        eprintln!(
            "  (8) bwd_spectrum(rep)  : {:>7} ns/call  ({N_HOC_BWD} calls, avg_steps={avg_steps_hoc_bwd:.1})",
            t_hoc_bwd.as_nanos() / N_HOC_BWD as u128
        );
    }

    // ── Derived analysis ──────────────────────────────────────────────────────
    eprintln!("\n=== Derived attribution ===");
    let log2_sa = (sa_num as f64).log2();

    // forward_spectrum: estimate probes per call and cost per probe
    let fwd_ns_per_call = t_fwd.as_nanos() / N_FWD as u128;
    let probes_per_fwd = avg_steps_fwd * 2.0 * log2_sa;
    let ns_per_fwd_probe = fwd_ns_per_call as f64 / probes_per_fwd;
    eprintln!(
        "  fwd probes/call      : ~{probes_per_fwd:.0}  ({avg_steps_fwd:.1} steps × 2 × log2(sa_num)={log2_sa:.1})"
    );
    eprintln!("  fwd ns/probe (total) : ~{ns_per_fwd_probe:.0} ns");
    let sa_cold_ns = t_sa_rnd.as_nanos() / N_SA_RND as u128;
    let sa_seq_ns = t_sa_seq.as_nanos() / N_SA_SEQ as u128;
    eprintln!("  sa_pos cost (seq)    : ~{sa_seq_ns} ns/call  (warm, ~L3 cache latency)");
    eprintln!("  sa_pos cost (rnd)    : ~{sa_cold_ns} ns/call  (cold mmap page faults / TLB)");
    eprintln!(
        "  compare overhead     : ~{} ns/probe  (= fwd_probe_total - sa_cold_pos, lower bound)",
        ns_per_fwd_probe as u128 - sa_cold_ns.min(ns_per_fwd_probe as u128)
    );

    // backward_spectrum: each step is 2 full-SA binary searches
    let bwd_ns_per_call = t_bwd.as_nanos() / N_BWD as u128;
    let probes_per_bwd = avg_steps_bwd * 2.0 * log2_sa;
    let ns_per_bwd_probe = bwd_ns_per_call as f64 / probes_per_bwd;
    eprintln!(
        "  bwd probes/call      : ~{probes_per_bwd:.0}  ({avg_steps_bwd:.1} steps × 2 × log2(sa_num)={log2_sa:.1})"
    );
    eprintln!("  bwd ns/probe (total) : ~{ns_per_bwd_probe:.0} ns");

    eprintln!("\n=== Vec<SmemStep> allocation estimate ===");
    // A Vec<SmemStep> alloc: SmemStep is 3×u64 = 24 bytes.
    // Typical forward spectrum yields ~3–10 steps → small heap alloc each call.
    // This is hard to isolate with wall timing; the sampling profiler will show
    // alloc::vec or jemalloc contributions if they're significant.
    eprintln!(
        "  SmemStep size        : {} bytes/step  (3 × u64)",
        std::mem::size_of::<prmi::index::spectrum::SmemStep>()
    );
    eprintln!(
        "  fwd avg_steps        : {avg_steps_fwd:.1} → ~{:.0} bytes heap/call",
        avg_steps_fwd * std::mem::size_of::<prmi::index::spectrum::SmemStep>() as f64
    );
    eprintln!("  Significance: use samply to see if alloc/dealloc appear in hot frames.");
}

// ── Main workload (profiler target) ───────────────────────────────────────────

fn main() {
    let args = parse_args();

    // chr17 cold-probe backward measurement: build a mode-2 sidecar from the .pac
    // and compare the model launch vs the full-SA reference (probes/step + wall).
    if args.pac.is_some() {
        run_probe_bench(&args);
        return;
    }

    eprintln!("[profile_spectrum] opening sidecar {:?}", args.sidecar);
    let idx = LearnedIndex::open(&args.sidecar).expect("open sidecar");
    eprintln!(
        "[profile_spectrum] sa_num={}  l_pac={}  max_err={}",
        idx.sa_num(),
        idx.l_pac(),
        idx.max_error_bound()
    );

    eprintln!("[profile_spectrum] loading pac from {:?}", args.fasta);
    let (pac, l_pac) = load_unpacked_pac(&args.fasta);
    assert_eq!(
        l_pac,
        idx.l_pac(),
        "pac l_pac={l_pac} != sidecar l_pac={}",
        idx.l_pac()
    );
    eprintln!("[profile_spectrum] l_pac={l_pac}");
    let enc = PacEncoding::Unpacked;

    if args.phase_time {
        run_phase_timing(&idx, &pac, l_pac, enc);
        return;
    }

    eprintln!(
        "[profile_spectrum] building corpus: {} queries × {} bp",
        args.corpus_size, args.query_len
    );
    let corpus = build_corpus(&pac, l_pac, args.query_len, args.corpus_size);
    let repeat_q = build_repeat_query(args.query_len);

    // Pre-compute backward anchors (forward run, not in the timed section).
    let anchors: Vec<_> = corpus
        .iter()
        .filter_map(|q| {
            let pivot = q.len() / 2;
            let steps = idx.forward_spectrum(&q[pivot..], &pac, enc);
            steps.last().copied().map(|s| (s, q.clone(), pivot))
        })
        .collect();
    eprintln!(
        "[profile_spectrum] {}/{} queries produced backward anchors",
        anchors.len(),
        corpus.len()
    );

    // Pre-compute high-occ anchor (not in the timed section).
    let repeat_pivot = repeat_q.len() / 2;
    let repeat_fwd = idx.forward_spectrum(&repeat_q[repeat_pivot..], &pac, enc);
    let repeat_anchor = repeat_fwd.iter().max_by_key(|s| s.occ_count).copied();
    if let Some(a) = repeat_anchor {
        eprintln!(
            "[profile_spectrum] repeat anchor: sa_start={} occ_count={} match_len={}",
            a.sa_start, a.occ_count, a.match_len
        );
    }

    // ── Phase A: forward_spectrum ─────────────────────────────────────────────
    eprintln!(
        "[profile_spectrum] Phase A: {} forward_spectrum calls ...",
        args.n_fwd
    );
    let t0 = Instant::now();
    let mut fwd_steps = 0usize;
    for q in corpus.iter().cycle().take(args.n_fwd) {
        let steps = idx.forward_spectrum(black_box(q), black_box(&pac), enc);
        fwd_steps += steps.len();
        let _ = black_box(steps);
    }
    let t_fwd = t0.elapsed();
    eprintln!(
        "[profile_spectrum] Phase A done: {:.3}s  {:.0} calls/s  avg_steps={:.1}",
        t_fwd.as_secs_f64(),
        args.n_fwd as f64 / t_fwd.as_secs_f64(),
        fwd_steps as f64 / args.n_fwd as f64,
    );

    // ── Phase B: backward_spectrum ────────────────────────────────────────────
    eprintln!(
        "[profile_spectrum] Phase B: {} backward_spectrum calls ...",
        args.n_bwd
    );
    let t0 = Instant::now();
    let mut bwd_steps = 0usize;
    for (step, q, pivot) in anchors.iter().cycle().take(args.n_bwd) {
        let steps = idx.backward_spectrum(
            black_box(step.sa_start),
            black_box(step.occ_count),
            black_box(step.match_len),
            black_box(q),
            black_box(*pivot),
            black_box(&pac),
            enc,
        );
        bwd_steps += steps.len();
        let _ = black_box(steps);
    }
    let t_bwd = t0.elapsed();
    eprintln!(
        "[profile_spectrum] Phase B done: {:.3}s  {:.0} calls/s  avg_steps={:.1}",
        t_bwd.as_secs_f64(),
        args.n_bwd as f64 / t_bwd.as_secs_f64(),
        bwd_steps as f64 / args.n_bwd as f64,
    );

    // ── Phase C: high-occ forward_spectrum (repeat query) ────────────────────
    let n_hoc = (args.n_fwd / 50).max(1);
    eprintln!("[profile_spectrum] Phase C: {n_hoc} high-occ forward_spectrum calls ...");
    let t0 = Instant::now();
    let mut hoc_fwd_steps = 0usize;
    for _ in 0..n_hoc {
        let steps = idx.forward_spectrum(black_box(&repeat_q), black_box(&pac), enc);
        hoc_fwd_steps += steps.len();
        let _ = black_box(steps);
    }
    let t_hoc = t0.elapsed();
    eprintln!(
        "[profile_spectrum] Phase C done: {:.3}s  {:.0} calls/s  avg_steps={:.1}",
        t_hoc.as_secs_f64(),
        n_hoc as f64 / t_hoc.as_secs_f64(),
        hoc_fwd_steps as f64 / n_hoc as f64,
    );

    // ── Phase D: high-occ backward_spectrum ───────────────────────────────────
    if let Some(ra) = repeat_anchor {
        let n_hoc_bwd = (args.n_bwd / 20).max(1);
        eprintln!("[profile_spectrum] Phase D: {n_hoc_bwd} high-occ backward_spectrum calls ...");
        let t0 = Instant::now();
        let mut hoc_bwd_steps = 0usize;
        for _ in 0..n_hoc_bwd {
            let steps = idx.backward_spectrum(
                black_box(ra.sa_start),
                black_box(ra.occ_count),
                black_box(ra.match_len),
                black_box(&repeat_q),
                black_box(repeat_pivot),
                black_box(&pac),
                enc,
            );
            hoc_bwd_steps += steps.len();
            let _ = black_box(steps);
        }
        let t_hoc_bwd = t0.elapsed();
        eprintln!(
            "[profile_spectrum] Phase D done: {:.3}s  {:.0} calls/s  avg_steps={:.1}",
            t_hoc_bwd.as_secs_f64(),
            n_hoc_bwd as f64 / t_hoc_bwd.as_secs_f64(),
            hoc_bwd_steps as f64 / n_hoc_bwd as f64,
        );
    }

    eprintln!("[profile_spectrum] all phases complete.");
}
