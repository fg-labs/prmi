# prmi

> **Status: v0.1 — pre-release.** Not yet published to crates.io. Build from source.

## What it is

**prmi** is a P-RMI (Piecewise Recursive Model Index) over a genomic suffix array. It is a two-crate Cargo workspace — `prmi` (Rust library + CLI trainer) and `prmi-sys` (thin C FFI shim) — designed as a drop-in learned-index accelerator for short-read aligners such as bwa-mem3. Given a reference FASTA, `prmi build` trains a two-layer radix-routed RMI over the suffix array, writes a compact sidecar on disk, and exposes a stable C and Rust API for fast bounded-search seeding at alignment time.

## Provenance

`fg-labs/prmi` is a GitHub fork of [`learnedsystems/RMI`](https://github.com/learnedsystems/RMI) — Ryan Marcus's MIT-licensed reference implementation of recursive model indexes. The P-RMI trainer additionally embeds verbatim training-pipeline code from the [`kaist-ina/BWA-MEME`](https://github.com/kaist-ina/BWA-MEME) RMI fork, which is also MIT-licensed. See [`LICENSE-FORK-NOTICE.md`](LICENSE-FORK-NOTICE.md) for the full fork lineage and tri-attribution convention.

## Quick start (build from source)

```bash
git clone https://github.com/fg-labs/prmi
cd prmi
cargo build --release -p prmi
./target/release/prmi build path/to/ref.fa
# Produces ref.fa.prmi.{meta,sa,l1,l2}
```

To use the Rust reader from another crate (path dependency — prmi is not yet on crates.io):

```rust
// In your Cargo.toml: prmi = { path = "../prmi/prmi" }
use prmi::index::LearnedIndex;

let idx = LearnedIndex::open("ref.fa.prmi")?;
let (pos, err) = idx.lookup(my_32mer_key);
```

## Sidecar layout

The trainer emits four files sharing a common prefix (typically `<ref>.fa.prmi`):

```
<prefix>.meta    TOML header (small)
<prefix>.sa      Packed suffix array (large; ~15 GB for human genome)
<prefix>.l1      L1 model parameters (fallback layer)
<prefix>.l2      L2 model parameters (primary radix-routed layer)
```

All multi-byte fields are little-endian. The reader mmaps the files at open time and validates magic bytes, counts, and sizes across all four files before returning a handle.

## C API

The stable C API is declared in `prmi-sys/include/prmi.h` (generated at build time by `cbindgen`):

```c
int  prmi_open (const char* sidecar_prefix, prmi_index_t** out_handle);
void prmi_close(prmi_index_t* handle);
int  prmi_lookup(const prmi_index_t* handle, uint64_t key,
                 uint64_t* out_predicted_sa_pos, uint64_t* out_err);
int  prmi_smem_range(const prmi_index_t* handle,
                     const uint8_t* query, int query_len,
                     const uint8_t* pac, size_t pac_len,
                     uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
size_t      prmi_sa_num         (const prmi_index_t*);
uint64_t    prmi_max_error_bound(const prmi_index_t*);
const char* prmi_format_version (const prmi_index_t*);
const char* prmi_last_error_message(void);
```

All functions return 0 on success or a negative integer on error. `prmi_last_error_message()` returns a thread-local human-readable string valid until the next `prmi_*` call on the same thread. Handles are safe for concurrent lookup calls after `prmi_open` returns. See `examples/cpp_caller.cc` for a working C++ consumer.

Note: `prmi_smem_range` takes an explicit `pac_len` parameter beyond the bare C API sketch in the brief; see `examples/cpp_caller.cc` for the authoritative signature.

## v0.1 scope

What is in v0.1:

- Reference-only training (no BED or FASTQ priors yet).
- Single curated memory mode: 5-byte packed SA entries (uint40 positions).
- 1-base-per-byte pac at the C ABI — not the 4-bpb BWA-MEME format.
- Single-key C API (`prmi_smem_range` and `prmi_lookup`); no batch API — `prmi_smem_range_batch` is v0.2.
- Contigs concatenated without sentinels — 32-mer queries can spuriously match across contig boundaries; callers are responsible for filtering these hits.
- N bases encoded as A (0) in the 2-bit pac.

## Roadmap (v0.2+)

- BED priors: target-aware training (capture priors from a target-capture BED).
- FASTQ histogram priors: workload-aware training from a read-set histogram.
- Batch FFI: `prmi_smem_range_batch` for aligned-SIMD seeding pipelines.
- Optional 4-bpb pac mode: drop-in compatibility with BWA-MEME's packed reference.
- Parallel SA construction: faster `prmi build` on many-core machines.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
cargo build -p prmi-sys --release   # generates prmi-sys/include/prmi.h
make -C examples                     # builds cpp_caller against libprmi_sys.a
./examples/run_smoke.sh              # end-to-end smoke test
```

## Citation

If you use prmi in published work, please cite [forthcoming].

## License

MIT throughout. See [`LICENSE`](LICENSE) (Marcus's MIT license text, which applies to the whole repository including new files) and [`LICENSE-FORK-NOTICE.md`](LICENSE-FORK-NOTICE.md) (fork lineage and tri-attribution convention).
