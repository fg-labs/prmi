# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Long-read seeding helpers**: `prmi_smem_range_long_read[_packed]` (multi-pivot batch on a single read buffer) + `prmi_minimizer_32mer` (lex-min 32-mer extraction for minimizer-window seeding). Rust API: `LearnedIndex::smem_range_long_read[_packed]` + `prmi::encoding::minimizer_32mer`. The 32-mer key constraint is unchanged — long reads use 32-base prefixes at each pivot or minimizer window. See the "Long-read integration" section of README for usage patterns.

- **Memory-mode menu** (`--memory-mode 1|2|3|suffix-key-cache`): four modes for the `.sa` sidecar file trading disk/RAM for lookup speed.

  - **Mode 1** (default, unchanged): position only, 5 B/entry (`~15 GB` for human). On-disk format is byte-for-byte identical to all previous sidecars; no migration required.
  - **Mode 2**: position + stored 32-mer key, 13 B/entry (`~39 GB`). The `smem_range` inner loop uses stored keys instead of re-tokenizing from pac, skipping per-candidate `read_unpacked_window + tokenize_32mer` calls. `.sa` body ratio vs mode 1: `13/5 = 2.6×`.
  - **Mode 3**: mode 2 + stored ISA (inverse suffix array), 21 B/entry (`~63 GB`). `isa_at(i)` returns `isa[sa[i]] = i` — the inverse-SA mapping from a genome position `p = sa[i]` back to its rank in the suffix array — enabling forward-extension lookups without re-scanning the pac.
  - **suffix-key-cache**: mode-1 `.sa` (5 B/entry) + a companion `.skc` file caching 32-mer keys for the top-N SA positions (configurable via `--suffix-key-cache-size`, default 1 000 000). Cache hits skip pac tokenization; misses fall back to the standard pac path. Lower memory overhead than mode 2 when only a fraction of positions are hot.

  The active mode is recorded in `.meta [sa] mode` and in `.meta [sa] bytes_per_entry` / `encoding`. Old sidecars (no `mode` field) default to `"1"`. The reader validates `bytes_per_entry` consistency with `mode` at open time.

  New public API: `LearnedIndex::key_at(i) -> Option<u64>` and `LearnedIndex::isa_at(i) -> Option<u64>` return stored data when available. `LearnedIndex::memory_mode() -> &str` reports the active mode string.

  New trainer config API: `MemoryMode` enum + `TrainerConfig::with_memory_mode(mode)` builder method.

  New sidecar file: `<prefix>.skc` (suffix-key-cache binary, 16-byte header + `cache_size × 16` body). See `docs/sidecar-format.md §2a` for the layout. New reader: `sidecar::skc_file::SkcFileReader`.

  Sidecar format `§2` expanded in `docs/sidecar-format.md` with per-mode `.sa` body layouts and a new `§2a` describing the `.skc` format.

- **`--prior-fastq-histogram` trainer flag** for workload-aware training: weight training pairs by `base_weight + log2(1 + freq(key))` where `freq(key)` is the key's count in a pre-computed 32-mer frequency histogram TSV. Hot k-mers (high frequency) receive higher weight during the weighted SLR / linear-spline fit, biasing the learned index toward the observed query distribution. The companion `--prior-fastq-base-weight W` flag (default `1.0`, must be > 0) controls the weight for k-mers absent from the histogram. The active prior is stored in `.meta` as `[priors] type = "fastq_histogram"` with `histogram = <path>`, `base_weight = <W>`, and `formula = "1.0 + log2(1 + freq)"`. Mutually exclusive with `--prior-bed`. A new `prmi histogram-from-kmc <kmc_dump>` subcommand converts KMC's text-format dump to prmi's u64-key TSV.

- **`--prior-bed` trainer flag** for target-aware training: weight training pairs inside a BED region higher during model fitting (weighted SLR / linear-spline fit), biasing the learned index toward hotspot intervals at the cost of potentially looser error outside them. The companion `--prior-bed-weight W` flag (default `10.0`, must be > 0) controls the multiplier. The active prior is stored in `.meta` as `[priors] type = "bed"` with `bed = <path>` and `weight = <W>`. Passing no `--prior-bed` flag produces the existing `type = "uniform"` behaviour.

- **Shared-memory loader** (`prmi shm`): pack a four-file sidecar into a single mmap-friendly blob and open it without re-paying I/O or page-fault costs across processes.

  - `prmi shm load <sidecar-prefix> <shm-path>` — writes all four sidecar components into a combined blob at `<shm-path>` (typically `/dev/shm/<name>` on Linux, `/tmp/<name>` on macOS). Existing files are overwritten.
  - `prmi shm unload <shm-path>` — removes the blob (`rm -f`; missing file is not an error).
  - `LearnedIndex::open_shm(shm_path)` — Rust API; mmaps the blob, validates all component headers, and returns a `LearnedIndex` identical in behaviour to one produced by `LearnedIndex::open`. Pages are shared between processes via the OS page cache.
  - `LearnedIndex::write_shm(sidecar_prefix, shm_path)` — convenience Rust wrapper around the CLI writer.
  - `prmi_open_shm(shm_path, out_handle)` — C FFI; identical post-open semantics to `prmi_open`; handle is closed with `prmi_close`.

  **Blob format** (`PRMI_SHM_v1`): a 4 KiB wrapper header followed by the four sidecar components (`.meta`, `.sa`, `.l1`, `.l2`), each starting on a 4 KiB boundary. Header records magic (16 bytes), wrapper version (`u64`), and the offset + length of each component (`u64` pairs). All fields little-endian.

  **Limitations** (documented inline and in README):
  - Concurrent writers not supported; a partially written blob yields a validation error on open.
  - Crash safety not addressed; a blob written by a killed process may be corrupt.
  - True cross-process sharing is asserted (we use `MAP_SHARED` and empirically confirm page sharing); dedicated multi-process tests will be added in v0.2.

- C batch FFI: `prmi_smem_range_batch` + `prmi_smem_range_batch_packed` for downstream consumers (e.g. aligners) that process many seeds per read. Both functions take a flat `count * 32` byte buffer of 32-base queries and parallel `out_k`/`out_l`/`out_s` arrays. The unpacked variant takes a 1-base-per-byte pac; the packed variant takes a 2-bit BWA/BWA-MEME `bntpac`-encoded pac with an explicit base count. Both delegate to the existing Rust `LearnedIndex::smem_range_batch` / `smem_range_enc` internals. The single-key `prmi_smem_range` and `prmi_smem_range_packed` functions are unchanged.

- SIMD-accelerated smem_range local-search: AVX2 (x86_64) + NEON (aarch64) + scalar fallback. The `resolve_one` inner loop now processes SA candidates in chunks of 4 using `tokenize_4_at_once` (dispatched at compile time on aarch64, at runtime on x86_64). Results are bit-identical to the prior scalar path. New `resolve_one_scalar` and `tokenize_4_scalar` test helpers exposed as `#[doc(hidden)] pub` for integration-test use. New integration-test file `prmi/tests/smem_simd_equivalence.rs` asserts SIMD ↔ scalar equivalence over small / medium / large / packed-pac corpus runs. New Criterion `smem_range_simd` vs `smem_range_scalar` bench group added to `lookup_bench.rs`.

- Parallel suffix-array construction via libsais OpenMP. `--threads <N>` CLI flag (short: `-t`); `0` = auto (CPU count), `1` = single-threaded, `N > 1` = exactly N threads. Default is auto. Single-threaded fallback is preserved when `--threads 1` is passed.
- `LearnedIndex::sa_positions` + C `prmi_sa_positions` for resolving SA index ranges to genome positions (unblocks bwa-mem3 seeding integration).

### Fixed

- **LBC sentinel detection (pass-3 #1):** `fit_direct_leaf` previously detected
  the LBC "no next leaf" sentinel by comparing the next key against `u64::MAX`.
  This has a false positive when a real all-T (TTTT…T) 32-mer is the first
  training key of the next non-empty leaf (`u64::MAX` is a valid tokenisation
  output). Added a dedicated `next_is_real: Vec<bool>` field to
  `LowerBoundCorrection` (populated at construction time in `compute_next_for_leaf`)
  and an `is_next_real(leaf_idx)` accessor. Both call sites in `trainer.rs` now
  use `lbc.is_next_real(leaf_idx)` instead of the ambiguous key comparison.

- **Trailing-empty-leaf search window (pass-3 #2):** when masking causes 32-mer
  keys to lex-sort above every training key, they route to a trailing empty L2
  leaf whose LBC `next` returns the sentinel (no next non-empty leaf). Previously
  the empty-leaf branch emitted `err = 0` for this sentinel case, producing a
  1-slot search window that misses SA positions in `[prev_last_sa + 1, sa_num - 1]`.
  The branch now emits a centred constant model covering the full valid SA tail:
  `mid = (lo + hi) / 2`, `err = (hi - lo + 1).div_ceil(2)`, where `lo` is
  `prev_last_sa + 1` when a previous non-empty leaf exists, else 0. Added a
  dedicated `prev_is_real: Vec<bool>` field and `is_prev_real` / `prev_sa_index`
  accessors to `LowerBoundCorrection`. Regression test:
  `smem_range_resolves_trailing_empty_leaf_query` in `prmi/tests/mask_trailing_empty.rs`.

### Changed

- **Crate manifests (pass-3 #4):** `prmi/Cargo.toml` and `prmi-sys/Cargo.toml`
  now inherit `rust-version` from the workspace (`rust-version.workspace = true`).
  Previously the `rust-version = "1.83"` setting in `[workspace.package]` was
  not propagated to either member crate, making the MSRV declaration a no-op in
  `cargo publish` and `cargo metadata` output.

### Documentation

- **README (pass-3 #5, #6):** softened the research-preview disclaimer. The
  previous wording ("extensively tested… on real reference genomes (E. coli K-12,
  chr22)" and "validated correctness against BWA-MEME's RMI as an oracle") was not
  reproducible from `cargo test`. Updated to distinguish in-repo CI coverage
  (synthetic references) from out-of-tree ad-hoc validation.

### Tests

- **Sub-bug A regression (pass-3 #3):** added `smem_range_resolves_empty_leaf_query`
  in `prmi/tests/mask_lookup.rs`. This test builds a fixture where the masked
  region's 32-mer keys route to an L2 leaf that has no other training pairs,
  exercising the empty-leaf `err = next_sa_idx` path directly. The existing
  `smem_range_resolves_masked_region_query` test is annotated to clarify that
  it exercises sub-bugs B and C (sentinel check, last-SA arithmetic) but not
  sub-bug A (empty-leaf `err = 0`).

## [0.1.0] - 2026-05-22

First release of prmi as a fork of [`learnedsystems/RMI`](https://github.com/learnedsystems/RMI)
(Ryan Marcus, MIT) at commit `8e147da`, extended with a P-RMI trainer
targeting genomic suffix-array seeding and a stable C FFI.

### Added

**Crate structure**

- Two-crate Cargo workspace: `prmi` (library + CLI) and `prmi-sys` (C FFI shim).
- `prmi-sys/include/prmi.h` generated at build time by `cbindgen`.
- `LICENSE-FORK-NOTICE.md` documenting the fork lineage and attribution convention.

**Trainer (`prmi build`)**

- Three training-pair masking flags to exclude degenerate SA positions from model fitting:
  - `--no-mask-n-runs` (opt-out): by default, any SA position whose 32-mer window covers an N base is excluded; pass this flag to disable. On chr22, N-run masking alone reduces `max_error_bound` from ~27% to ~0.019% of SA size.
  - `--mask-homopolymers K` (opt-in): exclude positions whose 32-mer window contains a run of the same base of length ≥ K.
  - `--mask-bed PATH` (opt-in): exclude SA positions falling in any interval of a BED file (0-based, half-open). Overlapping intervals in the BED file are merged at parse time.
  Masking only filters the (key, SA-index) training set; the SA on disk is always complete. The `.meta` file records active mask config in new `[sa]` fields (`masked_n_runs`, `masked_homopolymers`, `masked_bed`).
- `--l2-leaf-count` auto-scales to `ceil(sa_num / 12)` when omitted (heuristic targeting ~12 SA entries per leaf).
- Clap-based CLI with `build`, `info`, and `inspect` subcommands.
- `prmi inspect <prefix>` — prints per-layer error distribution diagnostics (min / p50 / p95 / p99 / max err, leaf count, fallback rate).
- FASTA → 2-bit PAC conversion via `noodles-fasta`.
- Suffix array construction via `libsais` (the `feldroop/libsais` safe Rust wrapper).
- 32-mer MSB-first tokenization producing `u64` keys for training and lookup.
- `TrainingSet` abstraction for prior-agnostic (32-mer key, SA index) pair generation.
- Fulcrum-authored P-RMI trainer: `train_prmi` integrating Marcus's
  `radix,linear,linear_spline` machinery with L1-fallback support.
- `TrainerConfig` with BWA-MEME-derived empirical defaults.
- Full-SA `max_error_bound` verification pass after training.
- `build_sidecar`: writes all four sidecar files from a FASTA in one call.

**On-disk sidecar format (`PRMIv1`)**

- `.meta`: UTF-8 TOML header with `[prmi]`, `[ref]`, `[sa]`, `[rmi]`, and
  `[priors]` sections. Magic string `"PRMIv1"`, `format_version = 1`.
  Records `strand = "forward_only"` in `[sa]`.
- `.sa`: 24-byte binary header (`u32` magic `0x50524D53`, `u32` version,
  `u64` sa_num, `u8` bytes_per_entry, 7 reserved bytes) followed by
  `sa_num × 5` bytes of uint40-packed SA positions (`packed_lo8_hi32` encoding).
- `.l1` / `.l2`: 16-byte binary header (`u32` magic `0x504D4C31`/`0x504D4C32`,
  `u32` version, `u64` leaf_count) followed by `leaf_count × 24`-byte model
  entries (`f64 alpha`, `f64 beta`, `u64 err`).
- Public format spec at `docs/sidecar-format.md`.

**Rust reader (`prmi` library)**

- `LearnedIndex::open(prefix)` — mmap-backed loader with cross-file validation.
- `LearnedIndex::lookup(key) -> (u64, u64)` — L2-primary / L1-fallback lookup
  returning `(predicted_sa_pos, err)`.
- `LearnedIndex::smem_range` and `smem_range_batch` — bounded local SA search
  returning `(k, l, s)` SA intervals. Internal batch shape enables a future
  `prmi_smem_range_batch` C wrapper without API changes. Single-key path avoids
  a Vec allocation.
- `LearnedIndex: Send + Sync` — concurrent lookup from multiple threads after open.
- `thiserror`-based `Error` enum with `FormatTooNew { kind }` for unknown
  enumeration values and `UnsupportedEncoding` for unrecognized SA packing modes.

**C FFI (`prmi-sys`)**

- `prmi_open(sidecar_prefix, out_handle)` — opens a sidecar; opaque handle.
- `prmi_close(handle)` — frees all resources; safe to call with NULL.
- `prmi_lookup(handle, key, out_predicted_sa_pos, out_err)`.
- `prmi_smem_range(handle, query, query_len, pac, pac_len, out_k, out_l, out_s)` — 1-base-per-byte pac.
- `prmi_smem_range_packed(handle, query, query_len, pac, pac_num_bases, out_k, out_l, out_s)` — 2-bit packed pac (BWA / BWA-MEME `bntpac` convention).
- `prmi_tokenize_32mer(bases, bases_len, len, out_key)` — mirrors internal tokenization; `bases_len` bounds the read and is clamped together with `len` to 32.
- `prmi_reverse_complement_key(key, len, out_key)` — reverse-complement a tokenized u64 32-mer key for two-strand lookup against a forward-only SA.
- `prmi_reverse_complement_2bit(in_, len, out)` — reverse-complement a raw 2-bit unpacked base array; handles aliasing (in_ and out may overlap).
- Introspection: `prmi_sa_num`, `prmi_max_error_bound`, `prmi_format_version`.
- `prmi_last_error_message()` — thread-local error string; never returns NULL.
- All `int`-returning functions return 0 on success or a negative integer on error; the introspection accessors (`prmi_sa_num`, `prmi_max_error_bound`, `prmi_format_version`) and `prmi_last_error_message` return their values directly, and `prmi_close` returns `void`.

**Build configuration**

- `prmi-sys/prmi-sys.pc.in` — pkg-config template for downstream C/C++ consumers.
- `prmi-sys/cmake/PrmiSysConfig.cmake.in` — CMake `find_package(PrmiSys)` template.
- `prmi-sys/INSTALL.md` — substitution recipe for both pkg-config and CMake.

**CI**

- GitHub Actions workflow: `cargo build`, `cargo test`, `cargo clippy`,
  `cargo +nightly fmt --check`, and `cpp-example` job (builds and runs
  `examples/cpp_caller.cc` against `libprmi_sys.a` with a synthetic sidecar).
- Rust toolchain pinned to stable via `rust-toolchain.toml`; the fmt check
  runs on nightly so `rustfmt.toml`'s `ignore` of the vendored
  `prmi/src/upstream/` tree is honored (a nightly-only rustfmt feature).
- `rustfmt.toml` with `max_width = 100` and `ignore = ["prmi/src/upstream"]`.

**Tests and benchmarks**

- 98 unit + integration tests (round-trip trainer → reader → lookup; SA
  construction; brute-force every-suffix bound verification; concurrent
  lookups; sidecar meta validation; error-model rejection paths).
- 3 property-based tests (proptest).
- 2 Criterion benchmarks (`lookup_bench`, `trainer_bench`).
- PhiX-scale (5,386 bp) round-trip test exercising the L1 fallback path.
- `examples/cpp_caller.cc` — C++ smoke harness covering the core C API surface.

**Documentation**

- README rewritten for prmi v0.1: provenance, quick-start, sidecar layout,
  C API reference, scope, roadmap, and integrator section.
- Rustdoc on all public items in `prmi` and `prmi-sys`.
- `docs/sidecar-format.md` — third-party-readable format specification.

### Changed

- Workspace renamed: `learnedsystems/RMI` → `fg-labs/prmi`; package names
  updated in all `Cargo.toml` files.
- Default branch: `main`.
- GitHub Actions replaces Drone CI.
- `rmi_lib` upstream module is now a private dependency of the `prmi` library
  crate; the legacy `main.rs` binary entry point is retained with updated headers.
- Unused Marcus RMI model types removed: `PiecewiselinearModel`, partial
  three-layer BWA-MEME extensions, SOSD benchmark suite.
- All files touched by Fulcrum annotated with
  `// Modified by Fulcrum Genomics 2026`.
- **Breaking (behavior):** `prmi_smem_range` and `prmi_smem_range_packed` now
  reject queries shorter than 32 bases with a non-zero return code and an error
  message via `prmi_last_error_message`. Sub-32-mer queries were previously
  silently accepted but produced incorrect results because T-padding does not
  preserve lexicographic order relative to longer SA suffixes. Callers must
  pad queries to exactly 32 bases before calling.
- **API:** `prmi_tokenize_32mer` gained an explicit `bases_len: size_t`
  parameter that bounds the read before `len` and 32 are applied. The previous
  signature `(bases, len, out_key)` becomes `(bases, bases_len, len, out_key)`.
  This change is required to express the memory contract without relying on
  `len` as a combined length-and-bound parameter.
- `upstream::train::train` is now `pub(crate)` (was `pub`). This gates a latent
  `minus_epsilon` underflow path that could be reached through the public surface.
- `[sa] strand` meta field: unknown values now propagate through
  `Error::FormatTooNew { kind }` instead of being silently ignored.
- `[sa]` meta struct now uses `#[serde(deny_unknown_fields)]`; unknown fields
  from a newer writer trigger `Error::FormatTooNew` on read.
- `prmi_index_t` C type is now a forward-declared opaque struct rather than
  a zero-length array typedef, making the header C99 `-pedantic -Werror` clean.

### Removed

- SOSD benchmark suite (upstream benchmarking infrastructure not needed for v0.1).
- Drone CI configuration.
- `train_pwl_smoke.rs` upstream test (covered by prmi-specific tests).

### Fixed

- Trainer prediction semantics: use truncation (not rounding) to convert
  `f64` predictions to `u64` SA indices, matching BWA-MEME's integer cast.
- Decode BWA-MEME-packed `err` field at sidecar-write time rather than lookup time.
- `LowerBoundCorrection`: reverted BWA-MEME fields and methods that had been
  partially merged into the upstream struct.
- Trainer LBC neighbor correction now uses the last SA index of the leaf (was
  using a stale intermediate index), fixing under-masking that inflated the
  reported `max_error_bound`.
- `linear_splines` model now guards against f64 key collision: u64-distinct keys
  that compare equal as f64 no longer cause incorrect slope computation.
- Cross-validation checks `bit_shift ↔ l2_leaf_count` consistency at open time;
  a mismatch now returns an error rather than producing silently wrong predictions.
- Runtime checks at open time for `partial_start ≥ 2^31` and `partial_num == 0`
  (corrupt-sidecar attack surface).
- OOB-safe accessors on SA and model file readers reject out-of-range indices
  rather than panicking.
- `set_last_error` no longer panics on hypothetical allocator OOM; uses a
  static fallback message instead.
- BED interval parser now merges overlapping and adjacent intervals; previously
  overlapping ranges could cause covered-by checks to miss positions between
  merged intervals.

### Crate metadata

- `prmi 0.1.0` (library + CLI)
- `prmi-sys 0.1.0` (C FFI shim, depends on `prmi 0.1.0`)

### Compatibility

- Sidecar format: `PRMIv1` (`format_version = 1`). Stable across the `0.1.x` series.
- C ABI: stable across the `0.1.x` series. Exception: `prmi_tokenize_32mer`
  gained a `bases_len` parameter (see Changed above); callers must be recompiled.

[0.1.0]: https://github.com/fg-labs/prmi/releases/tag/v0.1.0
