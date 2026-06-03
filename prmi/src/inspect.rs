// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `prmi inspect` — per-layer error-distribution diagnostics for a built sidecar.

use crate::error::Result;
use crate::index::LearnedIndex;
use std::path::Path;

// ── statistics helpers ────────────────────────────────────────────────────────

/// Compute min, max, mean, and percentiles from a **sorted** slice of u64.
/// Returns `(min, max, mean, p50, p90, p99, p99_9, p100)`.
/// Panics if the slice is empty.
fn stats_sorted(v: &[u64]) -> (u64, u64, f64, u64, u64, u64, u64, u64) {
    let n = v.len();
    debug_assert!(n > 0);
    let min = v[0];
    let max = v[n - 1];
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let pct = |frac: f64| v[((n as f64 * frac) as usize).min(n - 1)];
    let p50 = pct(0.50);
    let p90 = pct(0.90);
    let p99 = pct(0.99);
    let p999 = pct(0.999);
    let p100 = v[n - 1];
    (min, max, mean, p50, p90, p99, p999, p100)
}

/// Helper to print a stats line in a consistent format.
macro_rules! print_stats {
    ($label:expr, $v:expr) => {{
        let (mn, mx, mean, p50, p90, p99, p999, p100) = stats_sorted($v);
        println!(
            "{}: min={} max={} mean={:.1} p50={} p90={} p99={} p99.9={} p100={}",
            $label, mn, mx, mean, p50, p90, p99, p999, p100
        );
    }};
}

// ── L2 entry classification ───────────────────────────────────────────────────

/// Decode a fallback-path L2 err field.
/// Returns `(partial_start, partial_num)`.
#[inline]
fn decode_fallback(err: u64) -> (usize, usize) {
    let partial_start = ((err >> 32) & 0x7fff_ffff) as usize;
    let partial_num = (err & 0xffff_ffff) as usize;
    (partial_start, partial_num)
}

// ── public entry point ────────────────────────────────────────────────────────

/// Run the `inspect` diagnostic pass on the sidecar at `prefix` and print
/// results to stdout.
pub fn inspect(prefix: &Path) -> Result<()> {
    let idx = LearnedIndex::open(prefix)?;
    let l2_reader = idx.l2();
    let l1_reader = idx.l1();

    let sa_num = idx.sa_num();
    let l2_leaf_count = idx.l2_leaf_count();
    let bit_shift = idx.bit_shift();
    let max_error_bound = idx.max_error_bound();
    let l1_len = l1_reader.len();

    println!("=== prmi sidecar inspection: {} ===", prefix.display());
    println!("sa_num            = {}", sa_num);
    println!("l2_leaf_count     = {}", l2_leaf_count);
    println!("bit_shift         = {}", bit_shift);
    println!("max_error_bound   = {}", max_error_bound);
    println!("l1_entries        = {}", l1_len);

    // ── read all L2 entries ───────────────────────────────────────────────────
    let l2_n = l2_reader.len();

    let mut direct_errs: Vec<u64> = Vec::new();
    let mut fallback_partial_nums: Vec<u64> = Vec::new();
    // Leaves with beta==0 and err==0 are either:
    //   a) True empty leaves (zero training pairs, constant prediction to next
    //      non-empty leaf; the trainer sets alpha = lbc.next_index), or
    //   b) Single-key direct leaves (one training pair, slr returns beta=0;
    //      if the prediction is exact, err=0 too).
    // Both produce identical on-disk bit patterns — we can only count the
    // aggregate. Report both a combined "zero-beta/zero-err" count and, of
    // those, how many have an integer-valued alpha (not dispositive, but
    // informative: single-key direct leaves have alpha == sa_index as f64).
    let mut zero_beta_zero_err_count: usize = 0;
    let mut zero_beta_zero_err_integer_alpha: usize = 0;

    // For top-10 worst: (err, idx, alpha, beta)
    let mut worst_direct: Vec<(u64, usize, f64, f64)> = Vec::new();

    for i in 0..l2_n {
        let e = l2_reader.entry(i);
        if (e.err >> 63) != 0 {
            // fallback
            let (_, partial_num) = decode_fallback(e.err);
            fallback_partial_nums.push(partial_num as u64);
        } else if e.beta == 0.0 && e.err == 0 {
            // Zero-beta / zero-err: could be an empty leaf (constant prediction
            // to next leaf) or a single-key direct leaf with a perfect fit.
            // These are indistinguishable from the on-disk representation alone.
            zero_beta_zero_err_count += 1;
            if e.alpha == e.alpha.trunc() {
                zero_beta_zero_err_integer_alpha += 1;
            }
        } else {
            // direct
            direct_errs.push(e.err);
            // track for top-10
            worst_direct.push((e.err, i, e.alpha, e.beta));
        }
    }

    let direct_count = direct_errs.len();
    let fallback_count = fallback_partial_nums.len();
    let total_l2 = l2_n;

    println!();
    println!("== L2 routing layer ==");
    println!(
        "direct leaves     = {} ({:.1}%)",
        direct_count,
        100.0 * direct_count as f64 / total_l2 as f64
    );
    println!(
        "fallback leaves   = {} ({:.1}%)",
        fallback_count,
        100.0 * fallback_count as f64 / total_l2 as f64
    );
    println!(
        "empty/single-key leaves (beta=0 err=0) = {} ({:.1}%); of those, integer alpha = {}",
        zero_beta_zero_err_count,
        100.0 * zero_beta_zero_err_count as f64 / total_l2 as f64,
        zero_beta_zero_err_integer_alpha
    );

    if !direct_errs.is_empty() {
        direct_errs.sort_unstable();
        print_stats!("direct err", &direct_errs);
    } else {
        println!("direct err: (no direct leaves)");
    }

    if !fallback_partial_nums.is_empty() {
        fallback_partial_nums.sort_unstable();
        print_stats!("fallback partial_num", &fallback_partial_nums);
    } else {
        println!("fallback partial_num: (no fallback leaves)");
    }

    // Top 10 worst direct leaves
    worst_direct.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
    worst_direct.truncate(10);
    println!("top 10 worst direct leaves (by err):");
    if worst_direct.is_empty() {
        println!("  (none)");
    }
    for (err, idx, alpha, beta) in &worst_direct {
        println!(
            "  leaf_idx={} alpha={:.6e} beta={:.6e} err={}",
            idx, alpha, beta, err
        );
    }

    // ── read all L1 entries ───────────────────────────────────────────────────
    let mut l1_errs: Vec<u64> = Vec::new();
    let mut l1_err_zero: usize = 0;
    let mut l1_constant_models: usize = 0;
    let mut worst_l1: Vec<(u64, usize, f64, f64)> = Vec::new();

    for i in 0..l1_len {
        let e = l1_reader.entry(i);
        l1_errs.push(e.err);
        if e.err == 0 {
            l1_err_zero += 1;
        }
        if e.beta == 0.0 {
            l1_constant_models += 1;
        }
        worst_l1.push((e.err, i, e.alpha, e.beta));
    }

    println!();
    println!("== L1 fallback layer ==");
    println!("total entries     = {}", l1_len);

    if !l1_errs.is_empty() {
        l1_errs.sort_unstable();
        print_stats!("err", &l1_errs);
        println!(
            "err == 0 entries  = {} ({:.1}%)",
            l1_err_zero,
            100.0 * l1_err_zero as f64 / l1_len as f64
        );
        println!(
            "constant models (beta == 0) = {} ({:.1}%)",
            l1_constant_models,
            100.0 * l1_constant_models as f64 / l1_len as f64
        );
    } else {
        println!("(no L1 entries)");
    }

    // Top 10 worst L1 entries
    worst_l1.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
    worst_l1.truncate(10);
    println!("top 10 worst L1 entries:");
    if worst_l1.is_empty() {
        println!("  (none)");
    }
    for (err, idx, alpha, beta) in &worst_l1 {
        println!(
            "  l1_idx={} alpha={:.6e} beta={:.6e} err={}",
            idx, alpha, beta, err
        );
    }

    Ok(())
}
