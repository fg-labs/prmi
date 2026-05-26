# prmi — Piecewise Recursive Model Index (genomics)

> **Work in progress (v0.1).** This branch is being landed across a stack of 11
> PRs. Each PR adds one submodule. The full prmi feature set (trainer, sidecar,
> CLI, FFI, SIMD, shm loader, priors, memory modes, long-read helpers) lands by
> the end of the stack.

prmi is a Rust workspace housing a P-RMI learned-index over the suffix array of
a reference genome, plus a stable C ABI for downstream aligners. It is a
genomics-oriented fork of Ryan Marcus's
[`learnedsystems/RMI`](https://github.com/learnedsystems/RMI) reference
implementation.

## What this PR adds

This PR (`feat/v0.1-skeleton-and-upstream`) establishes the workspace shape and
relocates Marcus's RMI primitives into `prmi/src/upstream/` so that
Fulcrum-authored code can compose them in subsequent PRs without polluting the
upstream-licensed source surface.

- `Cargo.toml` (workspace) with `prmi` + `prmi-sys` members.
- `prmi/Cargo.toml`, `prmi/src/lib.rs`, `prmi/src/upstream/` (Marcus's `models`
  + `train`, kept verbatim with Fulcrum extensions for the cleanroom trainer).
- `prmi-sys/Cargo.toml`, `prmi-sys/src/lib.rs` (placeholder; FFI in PR #7).
- `LICENSE-FORK-NOTICE.md` documenting the lineage.

## What's still missing

These submodules land later in the v0.1 stack:

| PR | Branch | Adds |
|---|---|---|
| 2 | `feat/v0.1-prmi-utilities` | `encoding`, `fasta`, `sa`, `error`, `histogram`, `inspect`, `sidecar::magic` |
| 3 | `feat/v0.1-sidecar` | `sidecar::{meta, sa_file, model_file, skc_file}` |
| 4a | `feat/v0.1-trainer-core` | `train::{trainer, training_set, mask, config, prmi, verify, keys}` |
| 4b | `feat/v0.1-trainer-priors-and-modes` | `train::prior`, BED + FASTQ-histogram priors, memory-mode menu |
| 5a | `feat/v0.1-index-base` | `LearnedIndex::open`, `lookup`, `smem_range` (scalar) |
| 5b | `feat/v0.1-index-simd-and-longread` | SIMD smem inner loop, long-read seeding helpers |
| 5c | `feat/v0.1-index-shm` | Shared-memory blob format + `open_shm` |
| 6 | `feat/v0.1-cli` | `prmi build` / `info` / `inspect` / `shm` / `histogram-from-kmc` CLI |
| 7 | `feat/v0.1-prmi-sys` | C FFI exports + cbindgen + C header |
| 8 | `feat/v0.1-polish` | Examples, docs, CHANGELOG, CI |

## Provenance

The `prmi/src/upstream/` directory contains Marcus's MIT-licensed code, moved
verbatim from the repo root. Per-file copyright headers identify Fulcrum
modifications (the LBC `Option`-returning accessors and `weighted_slr` /
`new_weighted` SLR variants). See `LICENSE-FORK-NOTICE.md`.

## License

MIT — see `LICENSE` (Marcus's upstream license, preserved).
