# prmi sidecar format (v0.1, magic `PRMIv1`)

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
bytes_per_entry = 5              # v0.1 ships only 5-byte (uint40) packing
encoding = "packed_lo8_hi32"     # see §2 for byte layout
strand = "forward_only"          # v0.1 builds the SA over the forward strand only

[rmi]
spec = "radix,linear,linear_spline"
l2_leaf_count = <integer>        # power of two; determines the radix routing shift
bit_shift = <integer>            # = 64 - log2(l2_leaf_count); stored explicitly
max_error_bound = <integer>      # global worst-case prediction error across all keys

[priors]
type = "uniform"                 # v0.1 only; see §6 for extension rules
```

### Rejection rules

The reader **must** reject the sidecar if any of the following hold:

- `magic != "PRMIv1"` or `format_version != 1`
- Any companion file (`.sa`, `.l1`, `.l2`) is absent
- The companion file binary headers (§2 and §3) disagree with `.meta` on
  `num_entries`, `bytes_per_entry`, or `leaf_count`
- `[priors] type` is not in the v0.1-known set (`"uniform"`); report
  `Error::FormatTooNew { kind: <value> }` — do not attempt partial support
- `[sa] strand` is not in the v0.1-known set (`"forward_only"`); report
  `Error::FormatTooNew { kind: <value> }` for the same reason

---

## 2. `.sa` (suffix array)

Binary file. A 24-byte header followed by packed SA entries.

### Header (24 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | `u32` | Magic `0x50524D53` (`"PRMS"`) |
| 4 | 4 | `u32` | `format_version = 1` |
| 8 | 8 | `u64` | `sa_num` (number of entries) |
| 16 | 1 | `u8` | `bytes_per_entry = 5` |
| 17 | 7 | — | Reserved, must be zero |

### Body (`sa_num × 5` bytes)

Each entry encodes a single SA position as a 40-bit unsigned integer
(`uint40`), stored as two fields in little-endian order:

| Field offset within entry | Size | Type | Meaning |
|---|---|---|---|
| 0 | 4 | `u32` | High 32 bits of position |
| 4 | 1 | `u8` | Low 8 bits of position |

Reconstruction: `position = (u64(position_hi) << 8) | u64(position_lo)`

This `packed_lo8_hi32` layout matches the `encoding` field in `.meta`.
The uint40 representation supports positions up to approximately
1.1 trillion bases, which is well past a human-scale genome (~3.1 Gbp).

### Size invariant

The reader must reject the file if
`sa_num * 5 + 24 != file_size_in_bytes`.

---

## 3. `.l1` and `.l2` (model layers)

Both files share the same binary layout: a 16-byte header followed by a
flat array of model entries.

### Header (16 bytes)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | `u32` | Magic: `0x504D4C31` (`"PML1"`) for `.l1`; `0x504D4C32` (`"PML2"`) for `.l2` |
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
- `bytes_per_entry = 5` (the only SA packing mode)
- `[priors] type = "uniform"` (the only prior type)
- `[sa] strand = "forward_only"` (SA built over forward strand only)

Future versions will extend the set of known `[priors] type` values
(e.g. `"bed"`, `"fastq_histogram"`) and `[sa] strand` values (e.g.
`"forward_reverse"`), and may raise `format_version`. Readers must treat
any unknown value in a versioned enumeration field as
`Error::FormatTooNew { kind: <value> }` rather than silently ignoring or
partially supporting it. Implementations that do partial support risk
correctness bugs; the format is designed for clean rejection instead.

The C ABI is stable across the `0.1.x` patch series. The on-disk format
is stable across the `0.1.x` patch series.

---

*The layout described here is a Fulcrum Genomics original specification,
inspired by the P-RMI variant introduced in BWA-MEME (Jung & Han,
Bioinformatics 2022). No BWA-MEME source code was used.*
