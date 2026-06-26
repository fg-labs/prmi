# prmi bench baselines

Criterion harness lives under `prmi/benches/`. Run with:

```sh
cargo bench -p prmi --bench <bench_name>
```

---

## Baseline (Plan 5 T5)

**Machine:** `Darwin arm64`
**Reference:** synthetic, 502 048 bp (500 000 bp random backbone + 512 × 4 bp tandem-repeat block = 2 048 bp repeat region for the high-occ case)
**Encoding:** `PacEncoding::Unpacked` (one base per byte, 0..=3)
**Corpus:** 256 queries × 75 bp lifted from the forward reference (all match)

Run: `cargo bench -p prmi --bench spectrum_bench`

| Group | Benchmark | Median time | Throughput |
|---|---|---|---|
| `forward_spectrum` | `corpus_256x75bp` | 24.09 ms / batch | 10.63 Kelem/s |
| `forward_spectrum_batch` | `batch/64` | 6.07 ms / batch | 10.55 Kelem/s |
| `backward_spectrum` | `corpus_256x75bp` | 84.66 ms / batch | 3.02 Kelem/s |
| `sa_positions_strided` | `contiguous_64` | 34.21 ns / call | 1.87 Gelem/s |
| `high_occ_backward` | `repeat_anchor` | 303.17 µs / call | — |

**Per-query derived costs (single query, not batch):**

| Primitive | Per-query cost |
|---|---|
| `forward_spectrum` (random queries) | ~94 µs |
| `backward_spectrum` (random anchors) | ~331 µs |
| `high_occ_backward` (occ_count = 251 024) | ~303 µs |

**High-occ anchor (D15 / T7 target):**
The repeat-query anchor has `occ_count = 251 024` (the tandem-repeat region yields a very wide SA interval). The `high_occ_backward` group benches `backward_spectrum` from this anchor; T7's single-representative launch is expected to reduce this from O(occ) to O(1).

These numbers are the accept/roll-back gate for T6 (model-accelerated forward launch), T7 (single-representative backward launch), and T8 (lockstep batch). T6/T7/T8 must quote their deltas against these medians.

---

## Plan 5 T7 (O(log) backward)

`backward_spectrum` was reimplemented to find each left-extended interval by two
binary searches over the full SA (boundary search), instead of enumerating every member
of the current interval and mapping it through the ISA. Each step is now
`O(log N · |pattern|)` regardless of occupancy; the prior implementation was `O(occ)` per
prepended base. Results are byte-identical to the brute-force oracle (verified by the
existing backward tests plus a new wide-interval test and a random-ref/random-anchor
proptest in `prmi/tests/spectrum_oracle.rs`). The ISA is no longer consulted by this path.

**Machine/reference/encoding:** identical to the T5 baseline above.

| Benchmark | Before (T5) | After (T7) |
|---|---|---|
| `high_occ_backward` / `repeat_anchor` (occ_count = 251 024) | 303.17 µs / call | ~22.9 µs / call |

The after-cost is independent of `occ_count` (the 251 024-wide anchor now costs the same
order as a low-occ backward step), confirming the O(occ) → O(log N) change. The
`backward_spectrum` corpus bench was also re-shaped: its pivot is now placed mid-query so
the right-anchored span lies within the read (the boundary search re-derives the interval
from that span and so requires the anchor query to be present at `read[pivot..]`), which
also shifts that group's absolute number.

---

## Plan 5 VEC (word-at-a-time compare)

`compare_query_vs_suffix_2x` — the inner-loop compare under both `forward_spectrum` and `backward_spectrum` — was reimplemented to read the doubled `[Fwd || RC]` reference in 32-base chunks (a `fill_doubled_chunk` helper that `copy_from_slice`s forward `Unpacked` runs and mirror+XORs RC runs) and compare each chunk against the query 8 bases (one `u64`) at a time. A within-word mismatch is located via `(xor.trailing_zeros() / 8)` on the little-endian XOR; full words advance the LCP by 8. The previous implementation walked one base at a time through `doubled_base_at` (per-base forward/RC/sentinel branch + `pac_base_at`). The scalar version is retained (test-only) as `compare_query_vs_suffix_2x_scalar`; a 2000-case proptest asserts the vectorized result is byte-identical to the scalar one over random references, random `sa_pos` spanning forward/RC/sentinel, and random queries, for BOTH `Unpacked` and `Packed` (bntpac) encodings. Safe Rust only (`u64::from_le_bytes` over bounds-checked slices; no `unsafe`).

**Machine/reference/encoding:** identical to the T5 baseline above (synthetic 500 kbp random ACGT reference, `Unpacked` pac, `corpus_256x75bp` group). Before/after measured against a freshly re-recorded scalar baseline on the same machine in this session.

| Benchmark | Before (scalar) | After (VEC) | Speedup |
|---|---|---|---|
| `forward_spectrum` / `corpus_256x75bp` | 23.92 ms | 10.80 ms | ~2.2× |
| `backward_spectrum` / `corpus_256x75bp` | 16.49 ms | 7.86 ms | ~2.1× |

Both primitives improve ~2×. The forward speedup is below the 3–4× ceiling profiled on real chr17 because this synthetic bench's per-query cost is dominated less by the compare and more by the surrounding binary-search / `sa_position` lookups (random short-LCP suffixes mismatch within the first word, so the word-at-a-time fast path fires once before falling to the per-base tail). On real references with longer common prefixes the chunked path amortizes further. The compare is verified byte-identical to the scalar reference (proptest, both encodings) and the forward/backward oracles are unchanged.

---

## Plan 5 C (stored-key skip)

The mode-2 sidecar stores a 32-mer key beside each SA position (13-byte entry, same cache line as the 5-byte position). `compare_query_vs_suffix_2x_keyed` uses that stored key plus a precomputed query-key to resolve the first ≤32 bases of each compare from two `u64` XORs — no per-base pac reads or forward/RC demux for that prefix (BWA-MEME's `suffixarray_uint64` trick). The first differing base comes from `(query_key ^ stored_key).leading_zeros() / 2`; if the keys agree over the full 32 and the query is longer, the compare continues from base 32 via the existing word-at-a-time pac compare on `query[32..]` vs `sa_pos + 32`. The query-key is precomputed ONCE per `forward_spectrum` call (the query is fixed across prefix steps; the keyed compare masks it to the active prefix length) and ONCE PER STEP in `backward_spectrum` (the pattern's first 32 bases change as bases are prepended).

**Sentinel guard (correctness-critical):** the stored key T-pads a suffix shorter than 32 bases, but the compare treats the sentinel/end-of-reference as SMALLEST — the OPPOSITE order. The key path is therefore taken ONLY when `stored_key.is_some()` AND `sa_pos + 32 <= 2*l_pac` (≥32 real doubled bases, so the stored key is a true 32-mer with no T-pad). Otherwise it falls back to the safe vectorized pac compare. A proptest over a real mode-2 sidecar asserts the keyed result is byte-identical to `compare_query_vs_suffix_2x_scalar` for random queries and EVERY SA index (full sweep, so the near-sentinel fallback positions are covered); the forward/backward `spectrum_oracle.rs` oracles and the FFI `spectrum_ffi.rs` tests now build mode-2, exercising the key path end-to-end.

**Machine/reference/encoding:** identical to the T5 baseline (synthetic 500 kbp random ACGT reference, `Unpacked` pac, `corpus_256x75bp` group). Both columns build the sidecar in **mode 2** (identical 13-byte `.sa` layout); the only variable is whether the stored key is used vs forced to fall back to the pac compare — so the delta isolates the key-skip logic, not the mode-1→mode-2 entry-size change.

| Benchmark | Before (key fallback) | After (key skip) | Speedup |
|---|---|---|---|
| `forward_spectrum` / `corpus_256x75bp` | 10.32 ms | 5.89 ms | ~1.75× |
| `backward_spectrum` / `corpus_256x75bp` | 7.49 ms | 5.06 ms | ~1.48× |

The stored-key skip DOES help forward (~1.75×) and backward (~1.48×) on this synthetic corpus: it removes the first-32-base pac traversal and forward/RC demux from every binary-search probe, which is the bulk of the per-probe compare cost when most suffixes already share a multi-base prefix with the query. **Keep-or-drop decision:** the data supports KEEPING the stored keys for the spectrum search — the ~50 GB they cost buys a measurable speedup on the hottest primitive, and the key sits on the same cache line as the position so reading it is effectively free once the position is read. The fallback path keeps correctness intact for the near-sentinel entries where the T-padded key is unsafe.

---

## Plan 5 T6 (forward model-launch)

**Negative result — NOT shipped; the nested-narrowing baseline was kept.**

The idea: instead of the model-free nested narrowing (m=1 binary-searches the full SA, each deeper prefix narrows within the previous, wider interval), SEED the search for the deepest key-resolvable prefix — `query[..min(len, 32)]` — from the learned model's `lookup(query_key).pred`, then expand OUTWARD by exponential galloping (the model window is a HINT, never a clamp: the gallop always recovers the TRUE boundary even with a wrong `pred` or `err = 0`). The shallower prefixes were then to be found by bounded binary search via the nesting property, and any prefixes past base 32 by narrowing inward. The deep launch was the only place the model can help, since the model reliably brackets only the narrow deepest interval — the shallow/wide prefixes lie outside any window and must be searched unbounded anyway.

The implementation was completed and proven oracle-identical: a three-way proptest (`forward_spectrum_model_launch_equals_reference_and_oracle`) asserting model-launch == nested-narrowing reference == brute-force oracle over random refs + queries (incl. long matches running past the 32-mer key and the wide-shallow case), plus a wrong-seed test (`forward_spectrum_wrong_seed_still_correct`) sweeping deliberately-wrong seeds (every SA extreme, each effectively `err = 0`) and asserting the trace stays oracle-identical — confirming the window is a hint, not a clamp. All passed; `forward_spectrum_occ_correct_for_wide_shallow_intervals` and the FFI tests stayed green.

**Machine/reference/encoding:** identical to the T5 baseline (synthetic 500 kbp random ACGT reference, `Unpacked` pac, `corpus_256x75bp` group; the current post-VEC, post-stored-key code). Before = nested-narrowing reference (`forward_spectrum_reference`), after = model-launch (`forward_spectrum`), measured back-to-back on the same machine in the same session.

| Benchmark | Nested-narrowing (reference) | Model-launch | Result |
|---|---|---|---|
| `forward_spectrum` / `corpus_256x75bp` | 6.60 ms, 6.21 ms | 6.80 ms, 6.43 ms, 6.34 ms | no win (within run-to-run noise; if anything the reference is marginally faster) |

The model-launch did NOT beat the nested-narrowing baseline — the medians overlap within run-to-run noise. On this synthetic corpus the per-query cost is dominated by the `sa_position` lookups and the keyed compare, not by the count of binary-search probes; the model-launch trades the (already cheap, already-narrow) deep nested searches for a galloping bracket that costs extra probes when the seed is not tight, while still paying for the same shallow searches — a net wash. Per the perf gate (accept only on a measurable win), this was ROLLED BACK to the nested-narrowing reference. The deep-anchor + outward-expansion approach is correct and may yet pay off on real references with longer LCPs or in a batched/latency-hidden form, but it is not a win as a per-query drop-in on this corpus.

---

## Plan 5 A2 (backward model-launch)

**ACCEPTED — shipped; measured on real chr17.**

`backward_spectrum` previously binary-searched the FULL SA `[0, sa_num)` for both interval boundaries on every left step (each probe a cold `sa_position_for` read into the ~2 GB mmapped 2× SA). Profiling on real chr17 attributed ~72% of the backward cost to these cold SA probes. This change SEEDS each step's two boundary searches from the learned model: `key = tokenize_32mer(P, min(32, |P|))`, `(pred, err) = lookup(key)`, and the search window is `[pred - err, pred + err + 1)`. The window is a HINT only — a unified `find_boundary` helper binary-searches within the window and, on a window-edge miss, exponentially gallops the bracket outward (left or right) until the true boundary is strictly interior or the SA end is reached. NEVER clamps to `[pred±err]`; a wrong `pred` or `err = 0` still yields the TRUE interval via expansion.

The model-free full-SA implementation is retained as `backward_spectrum_reference` (`#[doc(hidden)]`). Correctness is proven byte-identical: the `backward_spectrum_matches_oracle_proptest` now asserts model-launch == `backward_spectrum_reference` == brute-force oracle element-for-element over random refs + anchors, AND drives a deliberately-wrong / `err = 0` seed (`backward_spectrum_with_seed`, a `#[doc(hidden)]` test hook) asserting the trace is unchanged — proving expand-on-miss recovery. A dedicated `backward_spectrum_wrong_seed_recovers_true_interval` test sweeps wrong seeds at every SA extreme. The existing wide-interval, non-representative-predecessor, and Fwd/RC-junction oracle cases plus all FFI/shm tests stay green.

**Perf gate — measured on real chr17, NOT the synthetic** (the 500 kbp synthetic is cache-resident and cannot show a cold-memory win). Sidecar: mode-2 (stored keys) built from `chr17.fa.pac` via `build_sidecar_from_pac_with_config` (`sa_num = 166 514 883`, `l_pac = 83 257 441`, `max_err = 31 735`, `log2(sa_num) ≈ 27.3`). Corpus: chr17-derived read windows (forward `forward_spectrum` at a mid-query pivot supplies each backward anchor). Probe counts gated behind the `spectrum-probe-count` feature (thread-local counter in `ref_less`/`shares_prefix` — test/profiling only, no counter in the default build). Measured via `examples/profile_spectrum.rs --pac`:

```sh
cargo build --release --features spectrum-probe-count --example profile_spectrum
./target/release/examples/profile_spectrum --pac <chr17.fa.pac> --corpus-size 2048 --query-len 100
```

| Metric (chr17) | Reference (full-SA) | Model launch | Win |
|---|---|---|---|
| SA probes / left step — median | 53 | 13 | 4.1× fewer |
| SA probes / left step — p99 | 54 | 36 | 1.5× fewer |
| Wall-time / backward call (corpus 2048×100 bp) | 179.9 µs | 37.4 µs | ~4.8× faster |
| Wall-time / backward call (corpus 1024×75 bp) | 126.5 µs | 24.2 µs | ~5.2× faster |

The 53 probes/step ≈ 2 boundaries × log2(166 M) ≈ 2 × 26.5 confirms the reference searched the full SA; the model launch drops the median to 13 (≈ 2 × log2(2·err) for the windowed search) and shrinks the cold-probe count materially. Wall-time drops further than the probe ratio because the windowed probes are cache- and TLB-local rather than scattered across the 2 GB SA. Both probes/step AND wall-time drop on the real cold-SA case with all oracle tests green and a byte-identical trace on every chr17 anchor (2048/2048), so per the perf gate this is ACCEPTED. This is the gating learned-index win the consumers (bwa-mem3 hot-path item A / minibwa P1) asked for.

---

## Benchmark build profile and target-cpu

`cargo bench` builds under Cargo's `bench` profile, which inherits its settings
from `[profile.release]` — so the `lto = "thin"` and `codegen-units = 1` added
there apply to the benches as well as the shipped library (verified: the bench
rustc invocation carries `-C lto=thin -C codegen-units=1`). `target-cpu` is
intentionally NOT pinned, so distributed artifacts stay portable (baseline
x86-64 / aarch64).

For host-tuned measurement, use `scripts/bench-native.sh`, which builds
hermetically with exactly `RUSTFLAGS="-C target-cpu=native"` (it ignores any
ambient `RUSTFLAGS` so a stale value can't change what is measured). Note: on
Apple Silicon `native` selects `apple-mN` codegen the shipped baseline never
receives — treat aarch64 numbers for scalar micro-changes as indicative only;
the x86 floor (`x86-64-v3`) is the cost model of record. To bench a specific
floor instead of native, run the bench directly with your own flags, e.g.
`RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench -p prmi` (documented floor:
`x86-64-v3` on x86, `neoverse-n1`/`v2` on Graviton).
