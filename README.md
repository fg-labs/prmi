![Build](https://github.com/fg-labs/prmi/actions/workflows/ci.yml/badge.svg)
[![Version at crates.io](https://img.shields.io/crates/v/prmi)](https://crates.io/crates/prmi)
[![Documentation at docs.rs](https://img.shields.io/docsrs/prmi)](https://docs.rs/prmi)
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fg-labs/prmi/blob/main/LICENSE)

# prmi

**⚠️ RESEARCH PREVIEW — USE AT YOUR OWN RISK**

This software is in a **research preview** state. The trainer and reader have
been tested against synthetic genomic references (PhiX-scale and smaller,
exercised in CI via `cargo test`). Ad-hoc out-of-tree validation against real
reference genomes (E. coli K-12, human chr22) and against BWA-MEME's RMI as an
oracle was performed during v0.1 development; see release notes for results.
**No guarantees** are made regarding correctness or stability for production
aligner integration.

<p>
<a href="https://fulcrumgenomics.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fulcrumgenomics/ferro-hgvs/main/.github/logos/fulcrumgenomics-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/fulcrumgenomics/ferro-hgvs/main/.github/logos/fulcrumgenomics-light.svg">
    <img alt="Fulcrum Genomics" src="https://raw.githubusercontent.com/fulcrumgenomics/ferro-hgvs/main/.github/logos/fulcrumgenomics-light.svg" height="100">
  </picture>
</a>
</p>

[Visit us at Fulcrum Genomics](https://www.fulcrumgenomics.com) to learn more.

<a href="mailto:contact@fulcrumgenomics.com?subject=[GitHub inquiry]"><img alt="Email us" src="https://img.shields.io/badge/Email_us-%2338b44a.svg?&style=for-the-badge&logo=gmail&logoColor=white"/></a>
<a href="https://www.fulcrumgenomics.com"><img alt="Visit us at fulcrumgenomics.com" src="https://img.shields.io/badge/Visit_Us-%2326a8e0.svg?&style=for-the-badge&logo=wordpress&logoColor=white"/></a>

## What it is

**prmi** is a P-RMI (Piecewise Recursive Model Index) over a genomic suffix array. It is a two-crate Cargo workspace — `prmi` (Rust library + CLI trainer) and `prmi-sys` (thin C FFI shim) — designed as a drop-in learned-index accelerator for short-read aligners such as bwa-mem3. Given a reference FASTA, `prmi build` trains a two-layer radix-routed RMI over the suffix array, writes a compact sidecar on disk, and exposes a stable C and Rust API for fast bounded-search seeding at alignment time.

## Provenance

`fg-labs/prmi` is a GitHub fork of [`learnedsystems/RMI`](https://github.com/learnedsystems/RMI) — Ryan Marcus's MIT-licensed reference implementation of recursive model indexes. The P-RMI trainer is Fulcrum-authored on top of Marcus's RMI primitives; BWA-MEME (Jung & Han 2022) demonstrated the variant on suffix-array seeding and informed the algorithmic shape, but no code from BWA-MEME is used. See [`LICENSE-FORK-NOTICE.md`](LICENSE-FORK-NOTICE.md) for the fork lineage.

## Quick start (build from source)

```bash
git clone https://github.com/fg-labs/prmi
cd prmi
cargo build --release -p prmi
./target/release/prmi build path/to/ref.fa
# Produces ref.fa.prmi.{meta,sa,l1,l2}

# Use all CPUs for suffix-array construction (default):
./target/release/prmi build path/to/ref.fa --threads 0

# Pin to a specific thread count:
./target/release/prmi build path/to/ref.fa --threads 4

# Single-threaded (reproducible, baseline):
./target/release/prmi build path/to/ref.fa --threads 1
```

On macOS, `brew install libomp` is required before building (see `prmi-sys/INSTALL.md` for details). Linux uses GCC's `libgomp` automatically.

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
int  prmi_open    (const char* sidecar_prefix, prmi_index_t** out_handle);
int  prmi_open_shm(const char* shm_path,       prmi_index_t** out_handle);
void prmi_close   (prmi_index_t* handle);
int  prmi_lookup(const prmi_index_t* handle, uint64_t key,
                 uint64_t* out_predicted_sa_pos, uint64_t* out_err);
int  prmi_smem_range(const prmi_index_t* handle,
                     const uint8_t* query, int query_len,
                     const uint8_t* pac, size_t pac_len,
                     uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
int  prmi_smem_range_packed(const prmi_index_t* handle,
                            const uint8_t* query, int query_len,
                            const uint8_t* pac, uint64_t pac_num_bases,
                            uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
int  prmi_smem_range_batch(const prmi_index_t* handle,
                           const uint8_t* queries, uint64_t count,
                           const uint8_t* pac, size_t pac_len,
                           uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
int  prmi_smem_range_batch_packed(const prmi_index_t* handle,
                                  const uint8_t* queries, uint64_t count,
                                  const uint8_t* pac, uint64_t pac_num_bases,
                                  uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);

// SA-range readout, long-read (multi-pivot) SMEM, and sequence/seed helpers:
int  prmi_sa_positions(const prmi_index_t* handle, uint64_t k, uint64_t count,
                       uint64_t* out_positions);
int  prmi_smem_range_long_read(const prmi_index_t* handle,
                               const uint8_t* read_bases, uint64_t read_len,
                               const uint64_t* pivot_offsets, uint64_t npivots,
                               const uint8_t* pac, size_t pac_len,
                               uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
int  prmi_smem_range_long_read_packed(const prmi_index_t* handle,
                                      const uint8_t* read_bases, uint64_t read_len,
                                      const uint64_t* pivot_offsets, uint64_t npivots,
                                      const uint8_t* pac, uint64_t pac_num_bases,
                                      uint64_t* out_k, uint64_t* out_l, uint64_t* out_s);
int  prmi_tokenize_32mer(const uint8_t* bases, size_t bases_len, int len,
                         uint64_t* out_key);
int  prmi_minimizer_32mer(const uint8_t* bases, uint64_t len,
                          uint64_t* out_key, uint64_t* out_offset);
int  prmi_reverse_complement_key(uint64_t key, int len, uint64_t* out_key);
int  prmi_reverse_complement_2bit(const uint8_t* in, int len, uint8_t* out);

// Accessors — return their value directly (NOT the 0/negative convention):
size_t      prmi_sa_num         (const prmi_index_t*);
uint64_t    prmi_max_error_bound(const prmi_index_t*);
const char* prmi_format_version (const prmi_index_t*);
const char* prmi_last_error_message(void);
```

The `int`-returning functions (everything above except the four accessors and `prmi_close`) follow a uniform contract: `0` on success, a negative integer on error, with a human-readable description retrievable from `prmi_last_error_message()`. The accessors do **not** use that convention — they return their value directly: `prmi_sa_num` → `size_t`, `prmi_max_error_bound` → `uint64_t`, `prmi_format_version` → a static NUL-terminated version string (`const char*`, always `"PRMIv1"` in v0.1, valid for the process lifetime, non-NULL even for a NULL handle), and `prmi_last_error_message` → a thread-local human-readable string valid until the next `prmi_*` call on the same thread. `prmi_close` returns `void`. Handles are safe for concurrent lookup calls after `prmi_open` or `prmi_open_shm` returns; both must be closed with `prmi_close`. See `examples/cpp_caller.cc` for a working C++ consumer.

`prmi_smem_range` takes a 1-base-per-byte pac and an explicit `pac_len` parameter. `prmi_smem_range_packed` accepts a 2-bit packed pac in BWA / BWA-MEME's `bntpac` convention (4 bases per byte, MSB-first within each byte) and takes `pac_num_bases` — the exact base count — instead of a byte length; this allows bwa-mem3 to pass its packed reference directly without unpacking. Both functions return identical `(k, l, s)` results for the same reference and query.

## Integrating with a C/C++ project

The intended consumer is a short-read aligner (the project was extracted
from a bwa-mem3 design exercise). The integration pattern looks like:

```text
read → tokenize first 32 bases → prmi_lookup → predicted SA position + err bound
                                       ↓
                              search SA[pred-err .. pred+err]
                                       ↓
                              prmi_smem_range_packed( query, pac )
                                       ↓
                                  (k, l, s) SA interval
                                       ↓
                              prmi_sa_positions( k, l, positions )
                                       ↓
                           positions[0..l): forward-strand genome offsets
```

Once you have `(k, l, s)` from `prmi_smem_range_packed`, resolve the SA interval to genome positions with `prmi_sa_positions`:

```c
uint64_t k, l, s;
prmi_smem_range_packed(idx, query, 32, pac, pac_num_bases, &k, &l, &s);
if (l > 0) {
    std::vector<uint64_t> positions(l);
    prmi_sa_positions(idx, k, l, positions.data());
    // positions[0..l) are forward-strand genome offsets of the matches.
}
```

`prmi_sa_positions` writes `l` uint64 genome positions into the caller-provided buffer starting at SA index `k`. Returns 0 on success, negative on error (see `prmi_last_error_message`). No allocation; the caller owns the buffer. Thread-safe — concurrent calls on the same handle are safe.

### Link line

```bash
# Build:
cargo build --release -p prmi-sys
# Produces: target/release/libprmi_sys.a, target/release/libprmi_sys.so|dylib
#           prmi-sys/include/prmi.h  (generated by cbindgen)

# Link against the static archive. libsais's OpenMP object code is embedded in
# the archive, so a static consumer must also supply an OpenMP runtime:
# -fopenmp (libgomp) on Linux, Homebrew libomp on macOS.

# Linux:
c++ -std=c++17 -I prmi-sys/include my_app.cc \
    target/release/libprmi_sys.a \
    -fopenmp -lpthread -ldl -lm

# macOS:
c++ -std=c++17 -I prmi-sys/include my_app.cc \
    target/release/libprmi_sys.a \
    -L"$(brew --prefix libomp)/lib" -lomp \
    -framework Security -framework CoreFoundation
```

`libsais` is statically embedded into `libprmi_sys.a`; no system libsais is required (but the OpenMP runtime above is, because the embedded libsais uses it).

### pkg-config / CMake

`prmi-sys/prmi-sys.pc.in` and `prmi-sys/cmake/PrmiSysConfig.cmake.in` are
templates downstream consumers can substitute and install. See
`prmi-sys/INSTALL.md` for the substitution recipe.

### Query length

`prmi_smem_range` and `prmi_smem_range_packed` require `query_len == 32`. Queries shorter than 32 bases are rejected with a non-zero return code and a message via `prmi_last_error_message` (Rust: `Error::Internal`). The T-padding used for sub-32-mer keys does not preserve lexicographic sort order relative to longer SA suffixes, so the `err` bound in the learned index does not cover the discrepancy. Pad your query to 32 bases before calling. For short reads at the end of a contig, the missing trailing bases are simply absent from the SA — they will not match — so padding with any constant value (e.g. `T`) is safe.

### PAC encoding

`prmi_smem_range` expects the reference as **1 base per byte** (values
`0..=3`). For BWA / BWA-MEME-style 2-bit packed pac (4 bases per byte,
MSB-first within byte), use `prmi_smem_range_packed` instead and pass
the base count separately. The two functions produce identical results
on the same logical reference.

### SA coordinate convention

The v0.1 trainer builds the suffix array over the **forward strand
only**. The `.meta` file records this as `[sa] strand = "forward_only"`.
Consumers needing reverse-strand matches must either (a) tokenize the
reverse complement of each query and call lookup/smem_range a second
time, or (b) wait for v0.2 which will expose `--strand forward_reverse`
trainer support. Readers must reject unknown `strand` values via
`Error::FormatTooNew`.

For reverse-strand lookups against a forward-only SA, use `prmi_reverse_complement_key` (u64 key form) or `prmi_reverse_complement_2bit` (raw 2-bit base array) to flip the query, then call `prmi_lookup` / `prmi_smem_range_packed` a second time.

### Memory ownership

All memory allocated by prmi is owned and freed by prmi (typically via
`prmi_close`). Callers never free any pointer returned by a `prmi_*`
function. The `const char*` returned by `prmi_last_error_message` is
thread-local and lives until the next `prmi_*` call on the same thread.

### Thread safety

The opaque handle is read-only after `prmi_open`. Concurrent calls to
`prmi_lookup` / `prmi_smem_range` / `prmi_smem_range_packed` on the
same handle from multiple threads are safe and intended.
`prmi_open` and `prmi_close` must NOT be called concurrently with each
other or with lookups on the same handle.

### Allocator

prmi uses Rust's default global allocator. If your application uses a
custom allocator (jemalloc, mimalloc, ...), prmi's allocations won't be
observed in that allocator's stats. Because the API is opaque-handle
(prmi never returns a pointer the caller is expected to free), this is
not a correctness issue.

### Shared-memory loading

A compute node running N aligner processes on the same genome normally maps the `.sa` file once per process, paying the I/O and page-fault cost N times. With the shm loader, a single pre-load step copies the sidecar into tmpfs-backed shared memory; subsequent processes open the blob without re-paying those costs.

```bash
# Pre-load (once, before aligners start)
# On Linux:  use /dev/shm for true tmpfs.
# On macOS:  use /tmp (macOS has no /dev/shm by default).
prmi shm load /data/hg38.fa.prmi /dev/shm/hg38

# Each aligner process opens the blob instead of the original sidecar:
prmi_open_shm("/dev/shm/hg38", &handle);  // C
LearnedIndex::open_shm("/dev/shm/hg38");  // Rust

# Remove when done (optional; the OS clears /dev/shm on reboot)
prmi shm unload /dev/shm/hg38
```

The blob format (`PRMI_SHM_v1`) stores the four sidecar components (`.meta`, `.sa`, `.l1`, `.l2`) in a single file with a 4 KiB wrapper header. Each component starts on a 4 KiB-aligned boundary. Multiple processes mmap'ing the same file with `MAP_SHARED` share the OS page-cache pages. Lookups, smem_range, and all other operations are identical to the file-backed path.

**Limitations:** concurrent writes are not supported (if `prmi shm load` is killed mid-write the blob is corrupt; `open_shm` will fail with a validation error). Crash safety is not addressed. Multi-process correctness relies on the OS honouring `MAP_SHARED` page sharing for the backing store, which is standard on Linux tmpfs (`/dev/shm`) and on macOS (`/tmp`).

### Performance

`prmi_smem_range` and `prmi_smem_range_packed` use a SIMD-accelerated inner loop: NEON on aarch64, AVX2 on x86_64 (runtime-detected), scalar fallback elsewhere. All paths produce bit-identical results.

### Long-read integration

For reads up to ~100 kb, seed at N pivot offsets in one batch call rather than invoking the single-key API in a loop:

```c
uint64_t pivots[] = {0, 100, 200, 300, /* ... every 100 bases */};
uint64_t out_k[N], out_l[N], out_s[N];
prmi_smem_range_long_read_packed(idx, read, read_len, pivots, N,
                                 pac, pac_num_bases, out_k, out_l, out_s);
```

Pivots whose 32-mer window would extend past the end of the read are skipped automatically (the corresponding output slot is `k=0, l=0, s=0`). All other pivots are resolved with the same SIMD-accelerated `smem_range` path used by the single-key API.

For minimizer-window seeding (reduces seed count by ~W relative to dense tiling):

```c
for (uint64_t offset = 0; offset + window_size <= read_len; offset += window_step) {
    uint64_t mkey, moff;
    if (prmi_minimizer_32mer(&read[offset], window_size, &mkey, &moff) == 0) {
        uint64_t abs_off = offset + moff;   /* absolute pivot in read */
        uint64_t k, l, s;
        prmi_smem_range_packed(idx, &read[abs_off], 32, pac, pac_num_bases, &k, &l, &s);
        /* ... chain (k, l, s) hit at position abs_off ... */
    }
}
```

`prmi_minimizer_32mer` extracts the lexicographically-smallest 32-mer from `bases[0..len]` and writes its tokenized key and starting offset to the out-pointers. Returns 0 on success; -1 if `len < 32`.

The 32-mer key constraint is unchanged — long reads use 32-base prefixes at each pivot or minimizer window. The Rust equivalents are `LearnedIndex::smem_range_long_read[_packed]` and `prmi::encoding::minimizer_32mer`.

### Batching

v0.1 exposes both single-key and batch C APIs. `prmi_smem_range_batch` (unpacked pac) and `prmi_smem_range_batch_packed` (2-bit packed pac) accept a flat `count * 32` byte buffer of queries and three parallel `out_k`/`out_l`/`out_s` arrays of `count` u64s each. The single-key `prmi_smem_range` and `prmi_smem_range_packed` functions are preserved unchanged. Both single-key and batch paths delegate to the same SIMD-accelerated `resolve_one` inner loop and produce bit-identical results.

## Repeat masking

Real-genome references contain degenerate regions (N-runs, homopolymers, tandem repeats) that collapse many SA positions to identical 32-mer keys. Training on those positions inflates `max_error_bound` without improving lookup quality for meaningful queries. Three masking flags let the trainer skip degenerate (key, SA-index) pairs:

| Flag | Default | What it drops |
|---|---|---|
| `--no-mask-n-runs` | off (N-run masking is **on** by default) | 32-mer windows that cover any N base |
| `--mask-homopolymers K` | off | windows containing a run of the same base of length ≥ K |
| `--mask-bed PATH` | off | SA positions falling in BED intervals (0-based, half-open) |

```bash
# Default build: N-runs masked automatically
prmi build ref.fa

# Opt out of N-run masking (e.g. to measure the raw effect)
prmi build ref.fa --no-mask-n-runs

# Mask N-runs + poly-A/T runs of ≥ 20 bp
prmi build ref.fa --mask-homopolymers 20

# Mask N-runs + a RepeatMasker/TRF BED file
prmi build ref.fa --mask-bed repeats.bed
```

The SA on disk is **never affected** — it always contains all genome-region entries. Masking only narrows the (key, SA-index) pairs used to fit the model and compute `max_error_bound`. Queries over masked positions still produce predictions; the error bound applies to unmasked queries only. The `.meta` file records which masks were active (`[sa] masked_n_runs`, `masked_homopolymers`, `masked_bed`).

When using `--mask-bed`, overlapping intervals in the BED file are merged at parse time, so a BED with redundant or overlapping regions (e.g. RepeatMasker output) is handled correctly without preprocessing.

### Priors (target-aware training)

Masking removes training pairs entirely. A *prior* keeps all pairs but biases the model fit toward a region of interest, trading tighter in-region error for potentially looser out-of-region error. Use `--prior-bed` when you have a target-capture BED and want the learned index to be most accurate over those intervals:

| Flag | Default | Effect |
|---|---|---|
| `--prior-bed PATH` | off | Weight training pairs inside the BED intervals by `--prior-bed-weight` during model fitting |
| `--prior-bed-weight W` | `10.0` | Multiplier (must be > 0) applied to in-BED pairs |

```bash
# Build with BED prior (target-capture scenario)
prmi build ref.fa --prior-bed targets.bed

# Adjust the weight (higher = tighter in-BED error, at the cost of out-of-BED)
prmi build ref.fa --prior-bed targets.bed --prior-bed-weight 20.0

# Combine masking and prior: mask repeats, bias toward target regions
prmi build ref.fa --mask-bed repeats.bed --prior-bed targets.bed
```

The `--prior-bed` and `--mask-bed` flags must reference different files. The `.meta` file records the active prior under `[priors] type = "bed"`, along with the BED path and weight.

### FASTQ histogram prior (workload-aware training)

Use `--prior-fastq-histogram` when you have a 32-mer frequency histogram derived from a read set and want to bias the learned index toward the k-mers that queries look up most often:

| Flag | Default | Effect |
|---|---|---|
| `--prior-fastq-histogram PATH` | off | Weight training pairs by lookup likelihood; hot k-mers get higher weight |
| `--prior-fastq-base-weight W` | `1.0` | Weight for k-mers absent from the histogram (must be > 0) |

Weight formula: `base_weight + log2(1 + freq)` where `freq` is the k-mer's count in the histogram. Keys absent from the histogram get `base_weight` (they are not suppressed — they still need to be findable). The `log2` transformation compresses heavy-tailed frequency distributions without completely swamping low-frequency keys.

`--prior-bed` and `--prior-fastq-histogram` are mutually exclusive. Either may be combined with any `--mask-*` flag.

```bash
# Build with FASTQ histogram prior
prmi build ref.fa --prior-fastq-histogram workload.tsv

# Adjust the base weight for k-mers not in the histogram
prmi build ref.fa --prior-fastq-histogram workload.tsv --prior-fastq-base-weight 0.5

# Combine masking and histogram prior
prmi build ref.fa --mask-bed repeats.bed --prior-fastq-histogram workload.tsv
```

The histogram TSV has two tab-separated columns: `key_u64\tcount_u64`. Comment lines (starting with `#`) and blank lines are ignored. Duplicate keys are rejected. The `.meta` file records the active prior under `[priors] type = "fastq_histogram"`, along with the histogram path, base weight, and the formula string `"1.0 + log2(1 + freq)"`.

#### Producing a histogram

prmi does not count k-mers itself. Use an external tool to produce the histogram TSV, then pass it to `--prior-fastq-histogram`. Only keys with frequency ≥ 2 contribute lift above `base_weight`; filtering to `freq ≥ 2` keeps the file manageable for large sequencing runs.

**Using KMC** (recommended; handles human-scale FASTQs efficiently):

```bash
# Count 32-mers in a FASTQ, keep only freq >= 2
kmc -k32 -ci2 -fm sample.fastq.gz kmc_db tmp/
kmc_tools transform kmc_db dump -ci2 kmc_dump.txt

# Convert KMC's string-key dump to prmi's u64-key TSV
prmi histogram-from-kmc kmc_dump.txt > workload.tsv
```

`prmi histogram-from-kmc` reads KMC's text format (`ACGT...  count`) and converts each 32-mer string to prmi's MSB-first 2-bit u64 tokenization.

**Using a simple streaming counter (small datasets):**

```bash
# awk-based 32-mer counter over a FASTQ — suitable only for small test sets
zcat sample.fastq.gz | mawk 'NR%4==2' | \
  mawk '{for(i=1;i+31<=length($0);i++) print substr($0,i,32)}' | \
  sort | uniq -c | mawk '$1>=2{print $2"\t"$1}' > raw_kmers.txt
# Then convert to u64 keys with prmi histogram-from-kmc or a custom script.
```

## v0.1 scope

What is in v0.1:

- BED-prior target-aware training via `--prior-bed` (weights in-BED pairs higher during model fit).
- FASTQ histogram prior via `--prior-fastq-histogram` (weights pairs by observed query frequency; workload-aware training).
- **Memory-mode menu** (`--memory-mode 1|2|3|suffix-key-cache`): four modes trading disk/RAM for lookup speed.

  | Mode | B/entry | Extra data | ~Human .sa size | Notes |
  |---|---|---|---|---|
  | `1` (default) | 5 | none | ~15 GB | Unchanged from previous versions |
  | `2` | 13 | 32-mer key | ~39 GB | Skips per-candidate pac tokenization |
  | `3` | 21 | key + ISA | ~63 GB | Adds forward-extension capability |
  | `suffix-key-cache` | 5 + `.skc` | top-N keys | varies | Lower overhead than mode 2 |

  Modes 2/3 store keys (and ISA for mode 3) inline; `suffix-key-cache` uses a companion `.skc` file. All modes produce identical `smem_range` results. See `docs/sidecar-format.md §2` for the per-mode binary layout.

- 1-base-per-byte pac via `prmi_smem_range` / `prmi_smem_range_batch`; 2-bit packed (BWA-MEME `bntpac`) via `prmi_smem_range_packed` / `prmi_smem_range_batch_packed`.
- Single-key and batch C APIs for `smem_range`; single-key `prmi_lookup`.
- Contigs concatenated without sentinels — 32-mer queries can spuriously match across contig boundaries; callers are responsible for filtering these hits.
- N bases encoded as A (0) in the 2-bit pac.
- N-run masking on by default; homopolymer and BED masking available via opt-in flags.
- Parallel suffix-array construction via libsais OpenMP. `--threads <N>` CLI flag (default: `0` = auto, uses all available CPUs). Pass `--threads 1` for single-threaded behaviour.

## Roadmap (v0.2+)

- Forward/reverse strand SA with `--strand forward_reverse` trainer support.

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

## Resources

- [Documentation](https://docs.rs/prmi) (once published)
- [Sidecar format spec](docs/sidecar-format.md)
- [Releases](https://github.com/fg-labs/prmi/releases)
- [Issues](https://github.com/fg-labs/prmi/issues)
- [Pull requests](https://github.com/fg-labs/prmi/pulls)
- [License](LICENSE)
