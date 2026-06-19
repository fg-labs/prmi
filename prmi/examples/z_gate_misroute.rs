// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Design-Z item 6: measure the dispatch-gate mis-route rate.
//!
//! The production gate `present_anchor` checks only the read's FIRST N-free
//! 32-mer window. A read that IS servable by the on-target index — some window
//! occurs in the keep-set — but whose 5' window falls outside it (a read that
//! starts just before a target, or one with a variant/error in its first 32 bp)
//! is mis-routed to the whole-genome fallback. This tool quantifies that loss by
//! comparing the first-window gate against the exact any-window gate
//! (`present_anchor_any`), which is the upper bound on what a richer
//! any-window/bloom gate could serve.
//!
//! "Servable" is approximated by any-window-present: if any 32-mer of the read
//! occurs in the on-target index, the read can be seeded there. The mis-route
//! rate is then (any && !first) — servable reads the first-window gate rejects.
//!
//! ```text
//! PRMI_FAST=chr22.zh.prmi PRMI_PAC=chr22.fa.pac \
//!   cargo run --release --example z_gate_misroute -- on.fq
//! ```

fn main() {
    use prmi::index::smem::PacEncoding;
    use prmi::index::LearnedIndex;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    let require = |k: &str| -> String {
        std::env::var(k).unwrap_or_else(|_| {
            eprintln!("z_gate_misroute: required env var {k} is not set");
            std::process::exit(2);
        })
    };
    let fast = LearnedIndex::open(Path::new(&require("PRMI_FAST"))).expect("open fast");
    let pac = std::fs::read(require("PRMI_PAC")).expect("read pac");
    let enc = PacEncoding::Packed {
        num_bases: fast.l_pac(),
    };

    const K: usize = 32;
    // Index of the first N-free 32-mer window that occurs in the index, scanning
    // every window (the exact any-window gate, with the hit offset). `None` = no
    // window occurs (truly off-target for this index). Uses the public
    // `mem_search` so the example can also report WHERE the first hit lands.
    let first_hit_window = |read: &[u8]| -> Option<usize> {
        let mut start = 0usize;
        'w: while start + K <= read.len() {
            for j in 0..K {
                if read[start + j] >= 4 {
                    start += j + 1;
                    continue 'w;
                }
            }
            if fast.mem_search(&read[start..start + K], &pac, enc).match_len as usize >= K {
                return Some(start);
            }
            start += 1;
        }
        None
    };

    let files: Vec<String> = std::env::args().skip(1).collect();
    let (mut total, mut first_present, mut any_present, mut misroute) = (0u64, 0u64, 0u64, 0u64);
    // Histogram of the first hitting window offset among MIS-ROUTED reads — how
    // far in the recoverable anchor sits. A cheap "check the first few windows"
    // gate recovers the early buckets; a long tail argues for a full bloom.
    let mut hit_at: [u64; 5] = [0; 5]; // [1, 2, 3-5, 6-10, >10] windows past the start
    for path in &files {
        let f = std::fs::File::open(path).unwrap_or_else(|_| panic!("open {path}"));
        let mut lines = BufReader::new(f).lines();
        while let Some(Ok(_h)) = lines.next() {
            let Some(Ok(seq)) = lines.next() else { break };
            let _ = lines.next();
            let _ = lines.next();
            let read: Vec<u8> = seq
                .bytes()
                .map(|b| prmi::encoding::base_to_2bit(b).unwrap_or(4))
                .collect();
            total += 1;
            let first = fast.present_anchor(&read, &pac, enc);
            if first {
                first_present += 1;
                any_present += 1;
                continue;
            }
            // First window missed: scan for any later hit (the mis-route case).
            if let Some(off) = first_hit_window(&read) {
                any_present += 1;
                misroute += 1;
                // A mis-route's leading N-free window missed, so the first hit is
                // always at offset >= 1 (offset 0 is unreachable here).
                debug_assert!(off >= 1, "mis-route first hit must be past the leading window");
                let bucket = match off {
                    1 => 0,
                    2 => 1,
                    3..=5 => 2,
                    6..=10 => 3,
                    _ => 4,
                };
                hit_at[bucket] += 1;
            }
        }
    }

    let pct = |num: u64, den: u64| 100.0 * num as f64 / den.max(1) as f64;
    eprintln!("z_gate_misroute: reads={total}");
    eprintln!(
        "  first-window present (production gate): {first_present} ({:.2}%)",
        pct(first_present, total)
    );
    eprintln!(
        "  any-window present   (upper bound)    : {any_present} ({:.2}%)",
        pct(any_present, total)
    );
    eprintln!(
        "  mis-routed (servable but first-window rejects): {misroute} \
         ({:.2}% of all reads, {:.2}% of servable)",
        pct(misroute, total),
        pct(misroute, any_present)
    );
    // Offset = the 0-based window index of the first hit. For a mis-route the
    // leading window missed, so the offset is always >= 1.
    let labels = ["offset 1", "offset 2", "offset 3-5", "offset 6-10", "offset >10"];
    eprintln!("  mis-route first-hit window offset (windows past read start; >=1):");
    for (i, lbl) in labels.iter().enumerate() {
        eprintln!("    {lbl:>12}: {} ({:.1}% of mis-routes)", hit_at[i], pct(hit_at[i], misroute));
    }
}
