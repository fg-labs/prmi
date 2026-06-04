# prmi sidecar format (v0.1, magic `PRMIv1`)

This document is the authoritative reference for the prmi sidecar format.
Third-party reader implementations should follow this spec; any divergence
between this document and the trainer's emitted format is a bug.

This document specifies the on-disk layout of a prmi sidecar. The format
is the binding contract between the trainer (`prmi build`) and any reader
(`prmi`, `prmi-sys`, or a third-party implementation). All multi-byte
fields are little-endian.

A sidecar is a set of four files sharing a common prefix (typically
`<ref>.fa.prmi`):

| Suffix | Contents |
|---|---|
| `.meta` | TOML header (small) |
| `.sa` | Packed suffix array (large; ~15 GB for a human-scale genome) |
| `.l1` | L1 model parameters (fallback layer) |
| `.l2` | L2 model parameters (primary radix-routed layer) |

---

## 1. `.meta` (TOML)

UTF-8 TOML. The reader uses this file to validate the rest of the sidecar
before mapping any large files.

```toml
[prmi]
magic = "PRMIv1"
format_version = 1
trainer_version = "<crate-name>=<semver>"    # e.g. "prmi=0.1.0"
created_utc = "2026-05-20T15:00:00Z"         # RFC 3339, UTC

[ref]
path = "<path-to-source-FASTA>"              # absolute or relative
sha256 = "<hex sha256 of FASTA bytes>"
size_bytes = <integer>

[sa]
num_entries = <integer>          # N — total number of suffixes (= genome length)
bytes_per_entry = <integer>      # 5 (mode 1 / skc), 13 (mode 2), or 21 (mode 3)
encoding = "<string>"            # see §2 / §4.2; mode-dependent encoding name
mode = "1"                       # "1" (default), "2", "3", or "suffix_key_cache"
skc_cache_size = <integer>       # optional; present only when mode = "suffix_key_cache"
strand = "forward_only"          # v0.1 builds the SA over the forward strand only
masked_n_runs = <bool>           # true if N-run positions were excluded from training
masked_homopolymers = <integer>  # optional; k value if homopolymer masking was applied
masked_bed = "<path>"            # optional; BED file path if BED masking was applied

[rmi]
spec = "radix,linear,linear_spline"
l2_leaf_count = <integer>        # power of two; determines the radix routing shift
bit_shift = <integer>            # = 64 - log2(l2_leaf_count); stored explicitly
max_error_bound = <integer>      # global worst-case prediction error across all keys

[priors]
type = "uniform"                 # "uniform", "bed", or "fastq_histogram"; see §6
# The following fields are present only when type = "bed":
# bed = "<path-to-BED-file>"    # path recorded at build time
# weight = <float>              # multiplier applied to in-BED training pairs
# The following fields are present only when type = "fastq_histogram":
# histogram = "<path-to-histogram-tsv>"   # path recorded at build time
# base_weight = <float>                   # weight for keys absent from histogram
# formula = "1.0 + log2(1 + freq)"        # weight formula string (informational)
```

### Memory mode (`mode` / `skc_cache_size`)

`[sa] mode` selects the per-entry layout (see §4.2): `"1"`, `"2"`, `"3"`, or `"suffix_key_cache"`.

- **Missing `mode` defaults to `"1"`.** The key was introduced with the memory-mode menu; sidecars built before it omit `mode` entirely. A reader **must** treat an absent `mode` as `"1"` (position-only, 5 bytes/entry) rather than rejecting the sidecar. The in-tree reader does this via a serde default.
- **`skc_cache_size` is valid only when `mode = "suffix_key_cache"`.** It records the number of `(sa_index, key)` pairs in the companion `.skc` file, and must be present and non-zero for that mode. For every other mode it must be **absent** — a reader rejects a sidecar that carries `skc_cache_size` outside `suffix_key_cache` (see the Rejection rules below), matching the in-tree reader.

### Mask fields (informational)

`masked_n_runs`, `masked_homopolymers`, and `masked_bed` are optional informational fields that record which training-pair masks were active when the sidecar was built. They have no effect on lookup semantics — the SA is always complete regardless of masking — but consumers may use them to understand the model's error-bound scope:

- `masked_n_runs = true` → the error bound applies only to queries whose 32-mer window does not cover an N base.
- `masked_homopolymers = <k>` → the error bound excludes positions in homopolymer runs of length ≥ k.
- `masked_bed = "<path>"` → the error bound excludes positions in the listed BED intervals.

All three fields default to `false` / absent when missing, for backward compatibility with sidecars built before this spec revision.

### Rejection rules

The reader **must** reject the sidecar if any of the following hold:

- `magic != "PRMIv1"` or `format_version != 1`
- Any companion file (`.sa`, `.l1`, `.l2`) is absent
- The companion file binary headers (§2 and §3) disagree with `.meta` on
  `num_entries`, `bytes_per_entry`, or `leaf_count`
- `[priors] type` is not in the v0.1-known set (`"uniform"`, `"bed"`, `"fastq_histogram"`); report
  `Error::FormatTooNew { kind: <value> }` — do not attempt partial support
- `[sa] strand` is not in the v0.1-known set (`"forward_only"`); report
  `Error::FormatTooNew { kind: <value> }` for the same reason
- `[sa] mode`, **when present**, is not in the v0.1-known set (`"1"`, `"2"`, `"3"`,
  `"suffix_key_cache"`); report `Error::FormatTooNew { kind: <value> }`. A *missing*
  `mode` is **not** a rejection cause — it defaults to `"1"` (see above)
- `skc_cache_size` is present for a mode other than `"suffix_key_cache"`, or is absent
  or zero when `mode = "suffix_key_cache"`
- `bytes_per_entry` or `encoding` disagrees with `mode` (per §4.2: `5`/`13`/`21` bytes
  and `packed_lo8_hi32` / `…_key64` / `…_key64_isa64` respectively)

---

## 2. `.sa` (suffix array)

Binary file. A 24-byte header followed by packed SA entries. The
`bytes_per_entry` field in the header determines the per-entry layout
(see §4.2 for the memory-mode menu).

### Header (24 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | `u32` | Magic: bytes-on-disk `[0x50, 0x52, 0x4D, 0x53]` = ASCII `"PRMS"`; read as little-endian `uint32_t` the value is `0x534D5250` |
| 4 | 4 | `u32` | `format_version = 1` |
| 8 | 8 | `u64` | `sa_num` (number of entries) |
| 16 | 1 | `u8` | `bytes_per_entry` — must be `5`, `13`, or `21` |
| 17 | 7 | — | Reserved, must be zero |

### Body (`sa_num × bytes_per_entry` bytes)

The per-entry layout depends on `bytes_per_entry` (i.e. memory mode):

#### Mode 1 / suffix\_key\_cache — `bytes_per_entry = 5`

Each entry encodes a single SA position as a 40-bit unsigned integer
(`uint40`), stored as two fields in little-endian order:

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 4 | `u32` | High 32 bits of position (`position_hi`) |
| 4 | 1 | `u8` | Low 8 bits of position (`position_lo`) |

Reconstruction: `position = (u64(position_hi) << 8) | u64(position_lo)`

This `packed_lo8_hi32` layout matches the `encoding` field in `.meta`.
The uint40 representation supports positions up to approximately
1.1 trillion bases, which is well past a human-scale genome (~3.1 Gbp).

#### Mode 2 — `bytes_per_entry = 13`

Each entry stores the position plus the 32-mer key at that suffix:

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 4 | `u32` | `position_hi` (high 32 bits) |
| 4 | 1 | `u8` | `position_lo` (low 8 bits) |
| 5 | 8 | `u64` LE | Stored 32-mer key (same encoding as §5) |

The stored key lets `smem_range` skip per-candidate pac reads when
scanning the local search window. Encoding name: `packed_lo8_hi32_key64`.

#### Mode 3 — `bytes_per_entry = 21`

Each entry stores the position, the key, and the ISA (inverse suffix
array) value:

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 4 | `u32` | `position_hi` (high 32 bits) |
| 4 | 1 | `u8` | `position_lo` (low 8 bits) |
| 5 | 8 | `u64` LE | Stored 32-mer key |
| 13 | 8 | `u64` LE | ISA value: `isa[sa[i]] = i` |

The stored value is the inverse suffix array: for the genome offset `p`
recorded at entry `i` (`sa[i] = p`), the field holds `isa[p]`, the rank of
the suffix starting at `p` in the suffix array. By definition `isa[sa[i]] = i`,
so — because v0.1 writes entries in SA order — the value stored at entry `i`
is `i` itself. The mapping lets a consumer recover a suffix's SA rank directly
from its genome position without re-scanning the pac. Encoding name:
`packed_lo8_hi32_key64_isa64`.

### Size invariant

The reader must reject the file if
`sa_num * bytes_per_entry + 24 != file_size_in_bytes`.

---

## 2a. `.skc` (suffix-key-cache companion file)

Present only when `[sa] mode = "suffix_key_cache"` in `.meta`. Caches
32-mer keys for a subset of SA positions to speed up `smem_range` for
high-frequency query positions without the full 13 B/entry overhead of
mode 2.

### Header (16 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | `u32` | Magic: bytes-on-disk `[0x53, 0x4B, 0x43, 0x50]` = ASCII `"SKCP"` LE `0x50434B53` |
| 4 | 4 | `u32` | `format_version = 1` |
| 8 | 8 | `u64` | `cache_size` (number of entries) |

### Body (`cache_size × 16` bytes)

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 8 | `u64` LE | SA index |
| 8 | 8 | `u64` LE | Stored 32-mer key |

Entries are written in ascending SA-index order. The reader builds an
in-memory hash map (sa\_index → key) at open time for O(1) lookup. A
miss falls back to on-the-fly pac tokenization, so correctness is
unaffected by cache size.

---

## 3. `.l1` and `.l2` (model layers)

Both files share the same binary layout: a 16-byte header followed by a
flat array of model entries.

### Header (16 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | `u32` | Magic: bytes-on-disk `[0x50, 0x4D, 0x4C, 0x31]` = ASCII `"PML1"` for `.l1` (LE `uint32_t` value `0x314C4D50`); bytes `[0x50, 0x4D, 0x4C, 0x32]` = `"PML2"` for `.l2` (LE value `0x324C4D50`) |
| 4 | 4 | `u32` | `format_version = 1` |
| 8 | 8 | `u64` | `leaf_count` |

### Body (`leaf_count × 24` bytes)

Each entry is a linear model segment with an associated error or routing field:

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 8 | `f64` | `alpha` (intercept) |
| 8 | 8 | `f64` | `beta` (slope) |
| 16 | 8 | `u64` | `err` (interpretation differs; see §4) |

---

## 4. Lookup semantics

Lookup takes a `u64` key (produced by the tokenization rule in §5) and
returns a predicted SA position plus a search error bound.

L2 leaves are radix-routed: `l2_idx = key >> bit_shift`. If the L2
entry's `err` field has bit 63 set, the prediction falls back through the
L1 layer; otherwise the L2 entry is authoritative.

```
fn lookup(key: u64) -> (predicted_sa_pos: u64, err: u64):
    l2_idx = key >> bit_shift
    l2     = L2[l2_idx]
    fpred  = l2.alpha + l2.beta * f64(key)
    err    = l2.err

    if err >> 63 != 0:
        # L1 fallback: l2.err encodes the slice of L1 to search
        partial_start = ((err >> 32) & 0x7FFF_FFFF) as usize
        partial_num   = (err & 0xFFFF_FFFF) as usize
        local = clamp(fpred, 0.0, f64(partial_num - 1)) as usize
        l1    = L1[partial_start + local]
        fpred = l1.alpha + l1.beta * f64(key)
        err   = l1.err

    pos = clamp(fpred, 0.0, f64(sa_num - 1)) as u64
    return (pos, err)
```

`clamp(v, lo, hi)` returns `lo` if `v < lo`, `hi` if `v > hi`, else `v`.
`bit_shift` is read from `.meta`; it equals `64 - log2(l2_leaf_count)`.

After `lookup`, the caller must scan the SA interval
`[pos - err, pos + err]` (clamped to `[0, sa_num - 1]`) to find the
exact SA range that matches the query key. The global `max_error_bound`
from `.meta` is an upper bound on `err` across all possible keys; callers
may use it to pre-allocate comparison buffers.

---

## 5. 32-mer tokenization (key derivation)

Both the trainer and any reader must derive keys identically. The rule:

Given a query position with up to 32 valid (non-N) bases extending
rightward, build a `u64` key with bases placed MSB-first:
- Base 0 occupies bits 63–62
- Base 1 occupies bits 61–60
- ...
- Base 31 occupies bits 1–0

Each base is encoded as a 2-bit value: A=0, C=1, G=2, T=3.

If fewer than 32 valid bases are available, the remaining low-bit slots
are padded with `0b11` (the encoding of `T`). T-padding preserves
lexicographic sort order under unsigned integer comparison, which is a
correctness requirement for the index.

N bases (value 4 in unpacked form) are not valid 2-bit values. Callers
must not tokenize queries containing N at positions 0–31 of the key
window; behavior is unspecified for such inputs.

---

## 6. Version compatibility

v0.1 defines:
- `format_version = 1`
- `[sa] mode = "1"` — 5 B/entry, position only (default; backward-compatible)
- `[sa] mode = "2"` — 13 B/entry, position + stored 32-mer key
- `[sa] mode = "3"` — 21 B/entry, position + stored key + ISA
- `[sa] mode = "suffix_key_cache"` — 5 B/entry `.sa` + companion `.skc` for top-N keys
- `[priors] type = "uniform"` — uniform weighting; no additional `[priors]` fields
- `[priors] type = "bed"` — BED-prior weighting; additionally records `bed = "<path>"` and `weight = <float>`
- `[priors] type = "fastq_histogram"` — FASTQ-histogram workload-aware weighting; additionally records `histogram = "<path>"`, `base_weight = <float>`, and `formula = "<string>"`. Weight formula: `base_weight + log2(1 + freq(key))`.
- `[sa] strand = "forward_only"` (SA built over forward strand only)

Future versions will extend the set of known `[priors] type` values and
`[sa] strand` values (e.g. `"forward_reverse"`), and may raise
`format_version`. Readers must treat any unknown value in a versioned
enumeration field as `Error::FormatTooNew { kind: <value> }` rather than
silently ignoring or partially supporting it. Implementations that do
partial support risk correctness bugs; the format is designed for clean
rejection instead.

The C ABI is stable across the `0.1.x` patch series. The on-disk format
is stable across the `0.1.x` patch series.

---

## 7. Shared-memory blob format (`PRMI_SHM_v1`)

`prmi shm load` packs the four sidecar files into a single blob. The blob
is designed so that multiple processes can `mmap(MAP_SHARED)` it and share
OS page-cache pages, avoiding per-process I/O and page-fault costs.

### Wrapper header (4096 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 16 | `u8[16]` | Magic: `"PRMI_SHM_v1\x00\x00\x00\x00\x00"` (NUL-padded) |
| 16 | 8 | `u64` | `wrapper_format_version = 1` |
| 24 | 8 | `u64` | `meta_offset` (bytes from start of file) |
| 32 | 8 | `u64` | `meta_len` (bytes) |
| 40 | 8 | `u64` | `sa_offset` |
| 48 | 8 | `u64` | `sa_len` |
| 56 | 8 | `u64` | `l1_offset` |
| 64 | 8 | `u64` | `l1_len` |
| 72 | 8 | `u64` | `l2_offset` |
| 80 | 8 | `u64` | `l2_len` |
| 88 | 4008 | — | Reserved, zero |

All multi-byte integers are little-endian.

### Component layout

Each component starts at its declared `*_offset`, which is aligned to a
4096-byte boundary. The component bytes are identical to the corresponding
standalone sidecar file (including its binary header). Readers must validate
each component's magic, version, and size using the rules in §2 and §3.

```
[0..4096)      : wrapper header
[4096..*)      : .meta component  (meta_offset = 4096, always)
[sa_offset..*) : .sa component
[l1_offset..*) : .l1 component
[l2_offset..*) : .l2 component
```

### Rejection rules

Readers must reject the blob if:

- The first 16 bytes are not the magic string
- `wrapper_format_version != 1`
- Any `*_offset` or `*_offset + *_len` exceeds the file size
- Any component fails the sidecar validation rules (§2 / §3)

---

*The layout described here is a Fulcrum Genomics original specification,
inspired by the P-RMI variant introduced in BWA-MEME (Jung & Han,
Bioinformatics 2022). No BWA-MEME source code was used.*
