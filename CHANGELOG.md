# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- Clap-based CLI with `build` and `info` subcommands.
- FASTA → 2-bit PAC conversion via `noodles-fasta`.
- Suffix array construction via `libsais` (the `feldroop/libsais` safe Rust wrapper).
- 32-mer MSB-first tokenization producing `u64` keys for training and lookup.
- `TrainingSet` abstraction for prior-agnostic (32-mer key, SA index) pair generation.
- Fulcrum-authored P-RMI trainer: `train_prmi` integrating Marcus's
  `radix,linear,linear_spline` machinery with L1-fallback support.
- `TrainerConfig` with BWA-MEME-derived empirical defaults; `--l2-leaf-count`
  auto-scales based on SA size.
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
  `prmi_smem_range_batch` C wrapper without API changes.
- `LearnedIndex: Send + Sync` — concurrent lookup from multiple threads after open.
- `thiserror`-based `Error` enum with `FormatTooNew { kind }` for unknown
  enumeration values and `UnsupportedEncoding` for unrecognized SA packing modes.

**C FFI (`prmi-sys`)**

- `prmi_open(sidecar_prefix, out_handle)` — opens a sidecar; opaque handle.
- `prmi_close(handle)` — frees all resources; safe to call with NULL.
- `prmi_lookup(handle, key, out_predicted_sa_pos, out_err)`.
- `prmi_smem_range(handle, query, query_len, pac, pac_len, out_k, out_l, out_s)` — 1-base-per-byte pac.
- `prmi_smem_range_packed(handle, query, query_len, pac, pac_num_bases, out_k, out_l, out_s)` — 2-bit packed pac (BWA / BWA-MEME `bntpac` convention).
- `prmi_tokenize_32mer(bases, len, out_key)` — mirrors internal tokenization.
- Introspection: `prmi_sa_num`, `prmi_max_error_bound`, `prmi_format_version`.
- `prmi_last_error_message()` — thread-local error string; never returns NULL.
- All functions return 0 on success or a negative integer on error.

**Build configuration**

- `prmi-sys/prmi-sys.pc.in` — pkg-config template for downstream C/C++ consumers.
- `prmi-sys/cmake/PrmiSysConfig.cmake.in` — CMake `find_package(PrmiSys)` template.
- `prmi-sys/INSTALL.md` — substitution recipe for both pkg-config and CMake.

**CI**

- GitHub Actions workflow: `cargo build`, `cargo test`, `cargo clippy`,
  `cargo fmt --check`, and `cpp-example` job (builds and runs
  `examples/cpp_caller.cc` against `libprmi_sys.a` with a synthetic sidecar).
- Rust toolchain pinned to stable via `rust-toolchain.toml`.
- `rustfmt.toml` with `max_width = 100`.

**Tests and benchmarks**

- 98 unit + integration tests (round-trip trainer → reader → lookup; SA
  construction; brute-force every-suffix bound verification; concurrent
  lookups; sidecar meta validation; error-model rejection paths).
- 3 property-based tests.
- 2 Criterion benchmarks (`lookup_bench`, `trainer_bench`).
- PhiX-scale (5,386 bp) round-trip test exercising the L1 fallback path.
- `examples/cpp_caller.cc` — C++ smoke harness covering the full C API surface.

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

### Crate metadata

- `prmi 0.1.0` (library + CLI)
- `prmi-sys 0.1.0` (C FFI shim, depends on `prmi 0.1.0`)

### Compatibility

- Sidecar format: `PRMIv1` (`format_version = 1`). Stable across the `0.1.x` series.
- C ABI: stable across the `0.1.x` series.

[0.1.0]: https://github.com/fg-labs/prmi/releases/tag/v0.1.0
