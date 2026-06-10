// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Probe-count audit for the model-launch (`est_hint=0`) search paths.
//!
//! The criterion benches (`mem_search_bench`) show the model-launch fall-backs
//! cost ~1.2 µs (forward unique), ~4.6 µs (backward unique), and ~17 µs (forward
//! high-occ repeat) per call, vs 28–165 ns for the `est_hint>0` fast paths. The
//! open question for "is the model launch fully optimized" is whether those
//! microseconds are spent on irreducible COLD SA PROBES (random suffix-array
//! reads, each a cache/DRAM miss) or on avoidable CPU work.
//!
//! This audit answers it directly: with the `spectrum-probe-count` feature it
//! counts SA probes per call and reports the mean, so the per-call time can be
//! attributed (ns/probe ≈ memory latency ⇒ probe-bound ⇒ no CPU headroom). It
//! also confirms the `est_hint` paths collapse to the theoretical minimum
//! (1 confirm probe + 2 boundary gallops).
//!
//! Run: `cargo run --release --features spectrum-probe-count --example probe_audit`
//! Optional env: `PRMI_BENCH_REFLEN` (default 2_000_000).

#[cfg(not(feature = "spectrum-probe-count"))]
fn main() {
    eprintln!(
        "probe_audit requires the `spectrum-probe-count` feature:\n  \
         cargo run --release --features spectrum-probe-count --example probe_audit"
    );
}

#[cfg(feature = "spectrum-probe-count")]
fn main() {
    use prmi::index::smem::PacEncoding;
    use prmi::index::spectrum::probe_count;
    use prmi::index::LearnedIndex;
    use prmi::train::build_sidecar_with_config;
    use prmi::train::config::{MemoryMode, TrainerConfig};
    use std::io::Write;

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn synth_bases(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 61) & 3) as u8
            })
            .collect()
    }

    let ref_len = env_usize("PRMI_BENCH_REFLEN", 2_000_000);
    let enc = PacEncoding::Unpacked;
    let qlen = 80usize;
    let corpus = 256usize;
    let repeat_count = 1024usize;

    // Reference = backbone[..half] + (ACGT × repeat_count) + backbone[half..].
    let backbone = synth_bases(ref_len, 0x2545_F491_4F6C_DD1D);
    let half = ref_len / 2;
    let mut pac = Vec::with_capacity(ref_len + 4 * repeat_count);
    pac.extend_from_slice(&backbone[..half]);
    let repeat_start = pac.len();
    for _ in 0..repeat_count {
        pac.extend_from_slice(&[0u8, 1, 2, 3]);
    }
    let repeat_len = 4 * repeat_count;
    pac.extend_from_slice(&backbone[half..]);
    let backbone_end = repeat_start;

    let tmp = std::env::temp_dir().join(format!("prmi_probe_audit_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let fa = tmp.join("ref.fa");
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&fa).unwrap());
        writeln!(w, ">audit").unwrap();
        let alpha = [b'A', b'C', b'G', b'T'];
        for c in pac.chunks(60) {
            let l: Vec<u8> = c.iter().map(|&b| alpha[b as usize]).collect();
            w.write_all(&l).unwrap();
            w.write_all(b"\n").unwrap();
        }
    }
    let prefix = tmp.join("ref.prmi");
    let cfg = TrainerConfig::default()
        .with_memory_mode(MemoryMode::Mode2)
        .with_isa(true);
    build_sidecar_with_config(&fa, &prefix, None, Default::default(), 0, Some(cfg)).unwrap();
    let idx = LearnedIndex::open(&prefix).unwrap();

    let sa_num = idx.sa_num();
    let max_err = idx.max_error_bound();
    println!(
        "sa_num={sa_num} log2(sa_num)={:.1} max_error_bound={max_err} log2(2*err)={:.1} \
         repeat=[{repeat_start}..{}) ({repeat_len}bp)\n",
        (sa_num as f64).log2(),
        ((2 * max_err.max(1)) as f64).log2(),
        repeat_start + repeat_len
    );

    // Mean probes over a closure that performs one call per corpus item.
    fn mean<F: FnMut()>(label: &str, calls: usize, mut run: F) {
        probe_count::reset();
        run();
        let total = probe_count::get();
        println!(
            "  {label:<34} {:7.1} probes/call   ({total} over {calls} calls)",
            total as f64 / calls as f64
        );
    }

    // ── Forward corpora ──────────────────────────────────────────────────────
    let fwd_unique: Vec<(Vec<u8>, u64)> = {
        let max = backbone_end - qlen;
        let stride = (max / corpus).max(1);
        (0..corpus)
            .filter_map(|k| {
                let s = (k * stride) % max;
                let q = pac[s..s + qlen].to_vec();
                let m = idx.mem_search(&q, &pac, enc);
                (m.match_len > 0 && m.sa_start > 0).then_some((q, m.sa_start))
            })
            .collect()
    };
    let fwd_repeat: Vec<(Vec<u8>, u64)> = {
        let max = repeat_len - qlen;
        (0..corpus)
            .filter_map(|k| {
                let s = repeat_start + (k % max);
                let q = pac[s..s + qlen].to_vec();
                let m = idx.mem_search(&q, &pac, enc);
                (m.match_len > 0 && m.sa_start > 0).then_some((q, m.sa_start))
            })
            .collect()
    };

    println!("forward (per call):");
    mean("model_launch / unique", fwd_unique.len(), || {
        for (q, _) in &fwd_unique {
            std::hint::black_box(idx.mem_search(q, &pac, enc));
        }
    });
    mean("model_launch / repeat (high-occ)", fwd_repeat.len(), || {
        for (q, _) in &fwd_repeat {
            std::hint::black_box(idx.mem_search(q, &pac, enc));
        }
    });
    mean("est_hint_interval / unique", fwd_unique.len(), || {
        for (q, h) in &fwd_unique {
            std::hint::black_box(idx.mem_search_from_hint(q, *h, true, &pac, enc));
        }
    });

    // ── Forward SPECTRUM (full breakpoint trace) ─────────────────────────────
    // The consumer's `prmi_forward_spectrum` returns the full trace; this is the
    // path the hinted spectrum (parent-interval walk) replaces. Compare the cold
    // plain spectrum, the .kmt-tabled cold spectrum (what the consumer runs when
    // a k-mer table is loaded — it zeroes the shallow-band probes), and the
    // hinted spectrum. Byte-identity is proven by the proptest; here we only
    // count probes.
    let kmt_k = env_usize("PRMI_BENCH_KMT_K", 10) as u32; // 4^10 = ~1M entries
    eprintln!("[probe_audit] building k={kmt_k} kmer table for the tabled-cold comparison ...");
    let table = idx.build_kmer_table(kmt_k, &pac, enc);

    println!("\nforward spectrum / full trace (per call):");
    mean("cold plain / unique", fwd_unique.len(), || {
        for (q, _) in &fwd_unique {
            std::hint::black_box(idx.forward_spectrum(q, &pac, enc));
        }
    });
    mean(
        &format!("cold tabled k={kmt_k} / unique"),
        fwd_unique.len(),
        || {
            for (q, _) in &fwd_unique {
                std::hint::black_box(idx.forward_spectrum_tabled(q, &pac, enc, &table));
            }
        },
    );
    mean("hinted parent-walk / unique", fwd_unique.len(), || {
        for (q, h) in &fwd_unique {
            std::hint::black_box(idx.forward_spectrum_from_hint(q, *h, &pac, enc));
        }
    });
    mean("cold plain / repeat (high-occ)", fwd_repeat.len(), || {
        for (q, _) in &fwd_repeat {
            std::hint::black_box(idx.forward_spectrum(q, &pac, enc));
        }
    });
    mean(
        &format!("cold tabled k={kmt_k} / repeat"),
        fwd_repeat.len(),
        || {
            for (q, _) in &fwd_repeat {
                std::hint::black_box(idx.forward_spectrum_tabled(q, &pac, enc, &table));
            }
        },
    );
    mean("hinted parent-walk / repeat", fwd_repeat.len(), || {
        for (q, h) in &fwd_repeat {
            std::hint::black_box(idx.forward_spectrum_from_hint(q, *h, &pac, enc));
        }
    });

    // ── Backward corpus ──────────────────────────────────────────────────────
    let bwd_len = qlen.max(60);
    let pivot = bwd_len / 2;
    struct B {
        read: Vec<u8>,
        anchor_len: u64,
        sa_start: u64,
        occ: u64,
        hint: u64,
    }
    // Backward corpus from a genomic region. `region_start..region_start+region_len`
    // bounds where reads are lifted from (the unique backbone or the repeat block).
    let make_bwork = |region_start: usize, region_len: usize| -> Vec<B> {
        let max = region_len.saturating_sub(bwd_len).max(1);
        let stride = (max / corpus).max(1);
        (0..corpus)
            .filter_map(|k| {
                let s = region_start + (k * stride) % max;
                let read = pac[s..s + bwd_len].to_vec();
                let fwd = idx.mem_search(&read[pivot..], &pac, enc);
                if fwd.match_len == 0 {
                    return None;
                }
                let hint = idx.isa_at((s + pivot) as u64).filter(|&h| h != 0)?;
                // Byte-identity insurance on THIS corpus (esp. the occ~2000 repeat
                // case the small-ref proptests don't reach): the hinted one-shot
                // must equal cold before we report its probe count as a speedup.
                let cold = idx.mem_search_backward(
                    fwd.sa_start,
                    fwd.occ,
                    fwd.match_len,
                    &read,
                    pivot,
                    &pac,
                    enc,
                );
                let hinted = idx.mem_search_backward_from_hint(
                    &read,
                    pivot,
                    fwd.match_len,
                    hint,
                    true,
                    &pac,
                    enc,
                );
                assert_eq!(hinted, cold, "hinted one-shot != cold during corpus prep");
                Some(B {
                    read,
                    anchor_len: fwd.match_len,
                    sa_start: fwd.sa_start,
                    occ: fwd.occ,
                    hint,
                })
            })
            .collect()
    };
    let bwork = make_bwork(0, backbone_end);
    let bwork_rep = make_bwork(repeat_start, repeat_len);

    // One-shot backward (the BWA-MEME reseed primitive: emit ONE maximal SMEM,
    // gated by min_intv — no per-length trace). The question: does the hinted
    // one-shot's win hold in the high-occ reseed regime, or does the wide maximal
    // interval hurt it? Unlike the hinted TRACE, the one-shot recovers the maximal
    // interval ONCE, not per left step.
    let bwd_maximal = |label: &str, work: &[B]| {
        if work.is_empty() {
            println!("  {label:<26} (no corpus)");
            return;
        }
        mean(&format!("{label} cold one-shot"), work.len(), || {
            for it in work {
                std::hint::black_box(idx.mem_search_backward(
                    it.sa_start,
                    it.occ,
                    it.anchor_len,
                    &it.read,
                    pivot,
                    &pac,
                    enc,
                ));
            }
        });
        mean(
            &format!("{label} hinted one-shot (interval)"),
            work.len(),
            || {
                for it in work {
                    std::hint::black_box(idx.mem_search_backward_from_hint(
                        &it.read,
                        pivot,
                        it.anchor_len,
                        it.hint,
                        true,
                        &pac,
                        enc,
                    ));
                }
            },
        );
        mean(
            &format!("{label} hinted one-shot (match_len)"),
            work.len(),
            || {
                for it in work {
                    std::hint::black_box(idx.mem_search_backward_from_hint(
                        &it.read,
                        pivot,
                        it.anchor_len,
                        it.hint,
                        false,
                        &pac,
                        enc,
                    ));
                }
            },
        );
    };
    let mean_occ = |work: &[B]| -> f64 {
        if work.is_empty() {
            return 0.0;
        }
        work.iter().map(|b| b.occ as f64).sum::<f64>() / work.len() as f64
    };
    println!(
        "\nbackward corpora: unique n={} mean_anchor_occ={:.1} | repeat n={} mean_anchor_occ={:.1}",
        bwork.len(),
        mean_occ(&bwork),
        bwork_rep.len(),
        mean_occ(&bwork_rep),
    );
    println!("\nbackward / maximal one-shot (per call):");
    bwd_maximal("unique:", &bwork);
    bwd_maximal("repeat:", &bwork_rep);

    // Full backward TRACE — the per-anchor reseed cost. Compare cold (model-seeded
    // per left step), .kmt-seeded cold (no hint — the reseed case), and hinted
    // (.isa-seeded; available because reseed carries the parent SMEM's refpos).
    let bwd_trace = |label: &str, work: &[B]| {
        if work.is_empty() {
            println!("  {label:<34} (no corpus)");
            return;
        }
        mean(&format!("{label} cold model-seeded"), work.len(), || {
            for it in work {
                std::hint::black_box(idx.backward_spectrum(
                    it.sa_start,
                    it.occ,
                    it.anchor_len,
                    &it.read,
                    pivot,
                    &pac,
                    enc,
                ));
            }
        });
        mean(&format!("{label} cold .kmt-seeded"), work.len(), || {
            for it in work {
                std::hint::black_box(idx.backward_spectrum_tabled(
                    it.sa_start,
                    it.occ,
                    it.anchor_len,
                    &it.read,
                    pivot,
                    &pac,
                    enc,
                    &table,
                ));
            }
        });
        mean(&format!("{label} hinted .isa-seeded"), work.len(), || {
            for it in work {
                std::hint::black_box(idx.backward_spectrum_from_hint(
                    &it.read,
                    pivot,
                    it.anchor_len,
                    it.hint,
                    &pac,
                    enc,
                ));
            }
        });
    };
    println!("\nbackward / full trace (per call):");
    bwd_trace("unique:", &bwork);
    bwd_trace("repeat:", &bwork_rep);

    let _ = std::fs::remove_dir_all(&tmp);
}
