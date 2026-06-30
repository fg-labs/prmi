# Environment variables

This page documents every environment variable the crate reads. It is exhaustive
as of the audit date below — if you add a new `env::var`/`env::var_os`/`option_env!`
read, document it here in the same release.

There is exactly **one** runtime knob that affects the production library
(`PRMI_ISA`). Everything else is read only by the benchmark/example/test harnesses
under `prmi/examples/`, `prmi/benches/`, and `prmi-sys/examples/`, or is a standard
Cargo build-script variable. Consumers linking `libprmi_sys` only ever need to care
about `PRMI_ISA`.

## Runtime (production library)

### `PRMI_ISA`

- **Read at:** `prmi/src/index/collect.rs` (`isa_reseed_enabled()`, cached once per
  process via `OnceLock`).
- **Values:** presence-only. Set to any non-empty value to enable; unset/absent to
  disable. The *value* is not inspected — `PRMI_ISA=1`, `PRMI_ISA=on`, and
  `PRMI_ISA=anything` are equivalent.
- **Default:** disabled (unset).
- **Effect:** enables the **ISA reseed fast-path** inside the fused per-read SMEM
  collector (`collect_smems` / `collect_smems_into`). When enabled *and* a `.isa`
  sidecar is loaded *and* no k-mer table is present, reseeded pass-1 SMEMs resolve
  their forward search with `mem_search_warmstart`, seeding the search from an
  SA-index hint computed by `isa_at(refpos + offset)`. This skips the learned-model
  launch and boundary gallop, cutting per-reseed SA probes from ~10–20 (cold model
  launch) down to ~1–2 (a confirm-only insertion search). If the `.isa` sidecar is
  absent the flag has no effect (the fast-path requires it).
- **Correctness:** **byte-identity-safe.** The hint is only a *starting point* for a
  confirming search, so it produces the same `(k, l, s)` SA interval as a cold launch
  for any occurrence; it cannot change which seeds are emitted.
- **Performance:** read **once per process** (cached in a `OnceLock`), so it is not on
  any per-read or per-call hot path. Build the sidecar with `--with-isa` (or the
  equivalent `isa` build flag) to produce the `.isa` file this path consumes.
- **Note on the consumer-side name collision:** the bwa-mem3 consumer also has a
  variable called `PRMI_ZIGZAG_ISA`. That is a **different** variable that lives in
  the consumer's own C++ code and only affects the consumer's (non-fused) "zigzag"
  finder. To turn on this library's ISA reseed on the consumer's default *fused*
  path, set **`PRMI_ISA`** in the process environment — the consumer's FFI calls run
  in the same process, so this library reads it directly.

## Build-script / compile-time (set automatically by Cargo)

These are not user knobs; Cargo populates them. Listed for completeness.

| Variable | Read at | Purpose |
|---|---|---|
| `CARGO_MANIFEST_DIR` | `prmi-sys/build.rs` | Locate `cbindgen.toml` for FFI header generation. |
| `OUT_DIR` | `prmi-sys/build.rs` | Output directory for the generated `prmi.h`. |
| `CARGO_PKG_VERSION` | `prmi/src/train/mod.rs` (via `env!`) | Embedded as the sidecar `trainer_version` (`prmi=<version>`). |

## Development harnesses only (`examples/`, `benches/`, `tests/`)

None of these are read by the library or the FFI surface. They configure the
standalone benchmark/fixture binaries and have no effect on a linked consumer. Unless
noted, integers parse with `.parse()` and fall back to the default on a missing or
unparsable value.

### Corpus / input selection

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `PRMI_PREFIX` | `collect_gate`, `wall_gate`, `batch_wall`, `mem_search_batch_wall`, `predicted_sort_wall` | *(required — exits if unset)* | Sidecar prefix (`.meta`/`.sa`/optional `.isa`/`.kmt`). |
| `PRMI_FQ` | same harnesses as above | *(required)* | FASTQ of reads to process. |
| `PRMI_PAC` | `collect_gate`, `wall_gate`, `batch_wall` | *(required)* | Forward-strand `.pac` (2-bit packed reference). |
| `PRMI_OUT` | `collect_gate` | `/tmp/collect_smems.tsv` | Output TSV of emitted SMEMs (`rid m n k s`). |
| `PRMI_MAX_READS` | `batch_wall`, `mem_search_batch_wall`, `predicted_sort_wall` | `usize::MAX` (no cap) | Cap reads loaded from the FASTQ (`0` reads nothing). |
| `PRMI_HANDOFF` | `prmi-sys/examples/ffi_overhead.rs` | *(required)* | Directory holding `chr22_A.fa.pac` for the FFI-overhead microbench. |
| `PRMI_FULL` / `PRMI_FAST` | Design-Z gates (`z_accept_gate`, `z_present_dump`, `z_gate_misroute`) | *(required)* | Whole-genome truth index / tiered fast-path index prefixes. |

### Seeding parameters (mirror bwa-mem defaults)

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `PRMI_MIN_SEED_LEN` | collect/wall harnesses | `19` | Minimum SMEM length to emit. |
| `PRMI_SPLIT_LEN` | collect/wall harnesses | `28` | Span threshold that triggers pass-2 reseed. |
| `PRMI_SPLIT_WIDTH` | collect/wall harnesses | `10` | Occurrence threshold for reseed triggering. |
| `PRMI_MAX_MEM_INTV` | collect/wall harnesses | `20` | Threshold for pass-3 (long-MEM) FMI reseed. |

### Timing / batching / strategy

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `PRMI_REPEAT` | `wall_gate` | `1` | Times to repeat the per-read loop (reduce timing variance). |
| `PRMI_SCRATCH` | `wall_gate` | `1` (on) | `1` reuses a held `CollectScratch` (`collect_smems_into`); `0` allocates per read. |
| `PRMI_BATCH` | `batch_wall` (`256`), `mem_search_batch_wall` (`4096`) | see used-by | Lockstep batch size (clamped to ≥ 1). |
| `PRMI_LOCKSTEP_BATCH` | `predicted_sort_wall` | `64` | Lockstep batch size for the latency-hiding A/B. |
| `PRMI_CAP` | `z_accept_gate` | `10` | Occurrence cap for Design-Z acceptance predicate (must be > 0). |
| `PRMI_ITERS` | `ffi_overhead` | `2_000_000` | Iterations in the FFI-overhead timing loop. |
| `PRMI_LEAVES` | `ffi_overhead` | `8_388_608` | Model leaf count for the benchmark sidecar build. |
| `PRMI_FALLBACK` | `ffi_overhead` | `def` | Tag for the fallback cache directory. |

### Criterion / `bench_est_hint` corpus knobs

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `PRMI_BENCH_REFLEN` | `mem_search_bench`, `primitives_bench`, `probe_audit`, `bench_est_hint` | `2_000_000` | Synthetic reference length (raise toward chr22 ~50M for realistic locality). |
| `PRMI_BENCH_QUERIES` | `bench_est_hint` | `5_000` | Number of synthetic queries. |
| `PRMI_BENCH_QLEN` | `bench_est_hint` | `80` | Per-read window length. |
| `PRMI_BENCH_REPS` | `bench_est_hint` | `20` | Timed passes over the query set. |

### Synthetic fixture generator (`make_fixture`)

| Variable | Default | Purpose |
|---|---|---|
| `FIX_BASES` | `67_108_864` (64 MiB) | Genome length in bases (must be ≥ `FIX_READLEN`). |
| `FIX_READS` | `20_000` | Number of reads to generate. |
| `FIX_READLEN` | `150` | Read length in bases. |
| `FIX_SEED` | `1` | PRNG seed (same seed → identical fixture on any host). |
| `FIX_SUBS_PCT` | `1` | Per-base substitution rate, 0–100. |
| `FIX_OUT` | `/var/tmp/synth` | Output prefix for the generated `.pac` and `.reads.fq`. |
