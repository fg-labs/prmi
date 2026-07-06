// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! `.blm` file format: a classic bloom filter over the keep-set's 32-mer keys,
//! used as the Design-Z any-window dispatch gate.
//!
//! # Why
//!
//! The first-window gate ([`crate::index::LearnedIndex::present_anchor`])
//! inspects only the read's leading N-free 32-mer window, mis-routing ~46–48% of
//! *servable* reads (some later window IS in the keep-set) to the whole-genome
//! fallback. The exact any-window gate ([`present_anchor_any`]) recovers them but
//! pays up to `read_len - 31` `mem_search` locates on every truly-absent read.
//!
//! This bloom holds exactly the set of 32-mers `mem_search` can match at
//! `match_len >= 32` on the tiered `.sa` — i.e. `key_for_position_2x(p)` for
//! every KEPT doubled-coordinate position `p` whose 32-base window is full. The
//! gate then probes the bloom for every window (an O(read_len) sequence of cheap
//! in-memory probes) and, on a probe HIT, runs ONE confirming `mem_search` to
//! reject bloom false positives. Net effect:
//!
//! - **No false negatives** — every key `mem_search` matches is in the bloom, so
//!   the gate serves exactly the reads `present_anchor_any` serves (full
//!   served-fraction recovery).
//! - **No false positives in the gate's verdict** — a bloom false positive is
//!   filtered by the confirming `mem_search`, so the routed-to-on-target set is
//!   identical to `present_anchor_any`. This matters because the Design-Z
//!   consumer seeds present reads via the on-target index ONLY (no fallback), so
//!   an unconfirmed false positive would be a correctness regression, not just
//!   wasted work. (This "verdict identical to any-window" guarantee is the
//!   CONFIRMED path's — `present_anchor_bloom`/`anywin`. The unconfirmed Lever 2
//!   `present_anchor_bloom_first`/`bloomfw` path admits the bloom's false positives
//!   — and, under a Lever 3 `--routing-pad` index, the padded-only flank members —
//!   which the consumer's Lever 0 present-fallback re-seeds rather than dropping.)
//! - **Lever 3 padded member set** — the bloom may be built (`--routing-pad`) over
//!   the keep-set PADDED by ±N bp while the `.sa` stays tight, so it is a SUPERSET
//!   of the tight matchable set. No-false-negatives still holds; the extra members
//!   are flank/capture-boundary reads `bloomfw` routes on-target (seeded on the
//!   tight `.sa`, flank soft-clipped; Lever 0 rescues any the tight SA can't seed).
//! - **Cheap on absent reads** — a truly-absent read's windows all miss the
//!   bloom (modulo the false-positive rate), so it pays ~0 locates instead of
//!   `present_anchor_any`'s O(read_len).
//!
//! [`present_anchor_any`]: crate::index::LearnedIndex::present_anchor_any
//!
//! # Bit-selection: Lemire multiply-shift reduction
//!
//! [`bit_at`] maps a 64-bit combined hash into `[0, num_bits)` using Lemire's
//! multiply-shift reduction:
//!
//! ```text
//! bit = (combined as u128 * num_bits as u128) >> 64
//! ```
//!
//! This is division-free (no `%` instruction), uniform over any `num_bits`, and
//! deterministic across writer and reader — both call the same `bit_at`, so a
//! self-consistent `.blm` has no false negatives.
//!
//! **`.blm` rebuild requirement.** The Lemire reduction selects *different* bits
//! than the prior `combined % num_bits` form for the same key, so the on-disk
//! `.blm` body bytes differ between builds using different prmi binaries. The
//! shared header layout and `FORMAT_VERSION` are unchanged — bumping the shared
//! `FORMAT_VERSION` (used by `.sa`, `.l1`, `.l2`, `.blm`, …) to guard a bloom-only
//! change would invalidate all sidecars and force a full index rebuild. Instead a
//! `.blm` written by a binary using a different reduction is **rejected on open**
//! via the [`BLOOM_BODY_VERSION`] field (header byte 20): its bits would otherwise
//! be stale, producing bloom false negatives — harmless on the `mem_search`-confirmed
//! any-window gate, but the unconfirmed `bloom_first` gate would surface them as
//! silently mis-routed reads. Rejecting drops the bloom so the gate degrades to the
//! exact first-window path (`present_anchor`); a `.blm` is a build artifact
//! regenerated per index build. The bloom is an accelerator, not a correctness oracle.
//!
//! # Layout
//!
//! An 80-byte header followed by the bit array (`ceil(num_bits / 8)` bytes):
//!
//! | offset | field | type | meaning |
//! |---|---|---|---|
//! | 0  | magic | u32 | [`BLM_MAGIC`] ("PMBL") |
//! | 4  | version | u32 | [`FORMAT_VERSION`] |
//! | 8  | num_bits | u64 | bit-array length (always a multiple of 64) |
//! | 16 | num_hashes | u32 | k (double-hashing probe count) |
//! | 20 | body_version | u32 | [`BLOOM_BODY_VERSION`] — bit_at reduction scheme (rejected on mismatch) |
//! | 24 | num_keys | u64 | inserted key count (provenance / sizing only) |
//! | 32 | sa_num | u64 | tiered `.sa` entry count this bloom was built for (binding) |
//! | 40 | ref_digest | [u8; 32] | reference content hash (binding; same digest as `.kmt`) |
//! | 72 | keyset_digest | u64 | order-independent fingerprint of the routing key set (binding) |
//!
//! This sidecar is OPTIONAL (emitted only by `prmi build --with-bloom`, which is
//! meaningful for a tiered `--keep-bed` build). Loading is best-effort AND BOUND:
//! the loader keeps `.blm` only when its `sa_num`, `ref_digest`, AND
//! `keyset_digest` match the loaded index (the first two as `.kmt` binds via
//! [`crate::index::kmt_matches`]; the `keyset_digest` is also recorded in `.meta`
//! and compared there). A corrupt, absent, OR mismatched `.blm` simply leaves
//! `has_bloom() == false` and the gate falls back to the cheap first-window
//! `present_anchor`. Binding matters because a STALE bloom — built over a
//! different reference, OR the same reference with a different keep-set /
//! routing-pad (which `ref_digest` + `sa_num` alone cannot distinguish) — can OMIT
//! current keys, a false negative that would silently drop servable reads on the
//! unconfirmed `bloom_first` path; the `keyset_digest` closes that gap.
//!
//! # Safety
//!
//! `BloomFileReader` uses a read-only mmap kept alive for the reader's lifetime.

// mmap island.
#![allow(unsafe_code)]

use crate::error::{Error, Result};
use crate::sidecar::magic::{BLM_MAGIC, FORMAT_VERSION};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;

/// Size of the `.blm` binary header, in bytes (includes the `sa_num`,
/// `ref_digest`, and `keyset_digest` fields that bind the bloom to its index).
pub const BLM_FILE_HEADER_BYTES: usize = 80;

/// `.blm` body-version (header byte 20): the bit-selection reduction scheme used
/// to build the bit array. Distinct from the shared `FORMAT_VERSION` (header
/// layout) — it discriminates the `bit_at` reduction so a `.blm` whose body bits
/// were written by a binary using a different reduction is rejected on open
/// rather than served with stale bits. `1` = Lemire multiply-shift. A `.blm`
/// written before this field existed has `0` here (the zeroed reserved field) and
/// is therefore rejected. Bump this whenever `bit_at`'s mapping changes.
pub const BLOOM_BODY_VERSION: u32 = 1;

/// Bloom-filter sizing: bit-array length and probe count, derived from the key
/// count and a target false-positive rate via the textbook optimal formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BloomParams {
    /// Bit-array length in bits (always rounded up to a multiple of 64).
    pub num_bits: u64,
    /// Number of hash probes per key (`k`), clamped to `[1, 32]`.
    pub num_hashes: u32,
}

impl BloomParams {
    /// Optimal `(num_bits, num_hashes)` for `num_keys` elements at false-positive
    /// rate `fp_rate` (clamped to `(0, 1)`):
    ///
    /// - `m = ceil(-n · ln(fp) / (ln 2)^2)`, rounded up to a multiple of 64
    /// - `k = round((m / n) · ln 2)`, clamped to `[1, 32]`
    ///
    /// `num_keys == 0` yields a minimal 64-bit, 1-hash filter (every query
    /// misses, which is correct: an empty keep-set serves nothing).
    pub fn for_keys(num_keys: u64, fp_rate: f64) -> Self {
        const LN2: f64 = std::f64::consts::LN_2;
        let fp = fp_rate.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        if num_keys == 0 {
            return Self {
                num_bits: 64,
                num_hashes: 1,
            };
        }
        let n = num_keys as f64;
        let m = (-n * fp.ln() / (LN2 * LN2)).ceil().max(64.0);
        // Round up to a whole number of 64-bit words.
        let num_bits = (m as u64).div_ceil(64) * 64;
        let k = ((num_bits as f64 / n) * LN2).round() as i64;
        let num_hashes = k.clamp(1, 32) as u32;
        Self {
            num_bits,
            num_hashes,
        }
    }

    /// Byte length of the bit array (`ceil(num_bits / 8)`), or `None` if
    /// `num_bits / 8` does not fit `usize` on this platform.
    fn body_bytes(&self) -> Option<usize> {
        // num_bits is a multiple of 64, so this is exact when it fits usize.
        usize::try_from(self.num_bits / 8).ok()
    }
}

/// SplitMix64 finaliser — a strong integer mixer. Used to derive two independent
/// hashes from one 32-mer key for double-hashing.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The two base hashes for `key` (Kirsch–Mitzenmacher double-hashing). `h2` is
/// forced odd so the probe sequence visits distinct residues. Computed ONCE per
/// key; [`bit_at`] then derives every probe from this pair, so the two
/// `splitmix64` calls do not repeat across the `num_hashes` inner loop.
#[inline]
fn key_hashes(key: u64) -> (u64, u64) {
    let h1 = splitmix64(key);
    let h2 = splitmix64(h1) | 1;
    (h1, h2)
}

/// The `i`-th bit position from a key's `(h1, h2)` pair: `(h1 + i · h2)`
/// reduced uniformly into `[0, num_bits)` via Lemire multiply-shift.
/// Caller guarantees `num_bits > 0`.
#[inline]
fn bit_at(h1: u64, h2: u64, i: u32, num_bits: u64) -> u64 {
    let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
    // Lemire reduction: (combined * num_bits) >> 64, computed in u128. Maps
    // uniformly into [0, num_bits) with no division. num_bits is a multiple of
    // 64 (NOT a power of two in general), so `& (num_bits - 1)` is not safe;
    // this form works for any num_bits. Replaces `combined % num_bits`.
    //
    // Before: combined % num_bits  (~20-40 cycles, one true 64-bit div per probe)
    // After:  (combined as u128 * num_bits as u128) >> 64  (mul + shift, ~3-5 cycles)
    (((combined as u128) * (num_bits as u128)) >> 64) as u64
}

/// Build a bloom filter over `keys` sized per `params` and write it to `path`,
/// binding it to its index via `sa_num` (the tiered `.sa` entry count) and
/// `ref_digest` (the reference content hash — `pac_sha256`, else `ref.sha256`,
/// the same digest `.kmt` uses). The loader checks both before using the bloom,
/// so a stale `.blm` over a different reference/keep-set is ignored rather than
/// risking false negatives.
///
/// `params` should come from [`BloomParams::for_keys`] with `num_keys` keys;
/// `keys` yields the 32-mer keys (`tokenize_32mer` packing) — consumed as an
/// ITERATOR so the caller never materializes the full key set (a large keep-set
/// could otherwise be a multi-GB `Vec`). `num_keys` is the (already-counted) key
/// count, written to the header and used for sizing. Duplicate keys are harmless
/// (idempotent bit sets).
///
/// Returns the order-independent `keyset_digest` (a set fingerprint over the keys)
/// that was written to the header; the caller persists the same value in `.meta`
/// so the loader can reject a `.blm` built over a DIFFERENT keep-set/routing-pad
/// even when the reference and `sa_num` coincide.
pub fn write_bloom_file(
    path: &Path,
    num_keys: u64,
    keys: impl IntoIterator<Item = u64>,
    params: BloomParams,
    sa_num: u64,
    ref_digest: &[u8; 32],
) -> Result<u64> {
    // `BloomParams` fields are public, so a caller can bypass `for_keys` and pass
    // `num_bits == 0` (a degenerate zero-size filter — `bit_at`'s multiply-shift
    // maps everything to bit 0 rather than panicking, but it is still invalid) or a
    // `num_hashes` outside the writer's `[1, 32]` contract (which `open()` would
    // then reject). Fail closed here rather than emit an unloadable `.blm`. These
    // are writer-contract invariants on internally generated params, hence
    // `Error::Internal`; they mirror the reader's `validate_blm_header` checks.
    if params.num_bits == 0 || !params.num_bits.is_multiple_of(64) {
        return Err(Error::Internal {
            detail: format!(
                "bloom num_bits={} must be a positive multiple of 64",
                params.num_bits
            ),
        });
    }
    if !(1..=32).contains(&params.num_hashes) {
        return Err(Error::Internal {
            detail: format!("bloom num_hashes={} must be in 1..=32", params.num_hashes),
        });
    }
    let io = |e: std::io::Error| Error::Io {
        path: path.to_path_buf(),
        source: e,
    };
    // Size the file with the same overflow discipline as `validate_blm_header`:
    // a crafted huge `num_bits` must fail closed before `set_len`, never wrap
    // `total` into a header/body-size mismatch.
    let body_bytes = params.body_bytes().ok_or_else(|| Error::Internal {
        detail: format!(
            "bloom num_bits too large for this platform: {}",
            params.num_bits
        ),
    })?;
    let total = BLM_FILE_HEADER_BYTES
        .checked_add(body_bytes)
        .ok_or_else(|| Error::Internal {
            detail: format!("bloom file size overflow for num_bits={}", params.num_bits),
        })?;
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(io)?;
    f.set_len(total as u64).map_err(io)?;
    // SAFETY: `f` is opened read+write and sized to `total`; this mmap is the sole
    // accessor over its function-scoped lifetime (no concurrent writers) and is
    // flushed and unmapped before return.
    let mut mmap = unsafe { MmapMut::map_mut(&f) }.map_err(io)?;

    LittleEndian::write_u32(&mut mmap[0..4], BLM_MAGIC);
    LittleEndian::write_u32(&mut mmap[4..8], FORMAT_VERSION);
    LittleEndian::write_u64(&mut mmap[8..16], params.num_bits);
    LittleEndian::write_u32(&mut mmap[16..20], params.num_hashes);
    // bytes 20..24: bloom body-version (bit_at reduction scheme); see BLOOM_BODY_VERSION.
    LittleEndian::write_u32(&mut mmap[20..24], BLOOM_BODY_VERSION);
    LittleEndian::write_u64(&mut mmap[24..32], num_keys);
    LittleEndian::write_u64(&mut mmap[32..40], sa_num);
    mmap[40..72].copy_from_slice(ref_digest);

    // Stream keys in: set their bits AND fold them into an order-independent set
    // fingerprint (`wrapping_add` of a per-key mix is commutative, so the digest
    // is the same regardless of the parallel/serial iteration order, while two
    // different key multisets differ w.h.p.). No `Vec<u64>` is materialized.
    let mut keyset_digest = 0u64;
    {
        let body = &mut mmap[BLM_FILE_HEADER_BYTES..];
        for key in keys {
            keyset_digest = keyset_digest.wrapping_add(splitmix64(key));
            // Hash the key once; derive all `num_hashes` probes from the pair.
            let (h1, h2) = key_hashes(key);
            for i in 0..params.num_hashes {
                let bit = bit_at(h1, h2, i, params.num_bits);
                body[(bit / 8) as usize] |= 1u8 << (bit % 8);
            }
        }
    }
    LittleEndian::write_u64(&mut mmap[72..80], keyset_digest);

    mmap.flush().map_err(io)?;
    Ok(keyset_digest)
}

/// mmap-backed reader for a `.blm` bloom filter. After `open`, `contains(key)` is
/// a cheap in-memory membership probe (no I/O after the pages are resident).
///
/// # Concurrency
///
/// Concurrent writers to the underlying file are not supported.
pub struct BloomFileReader {
    /// Keeps the file open (and the mmap valid) for the reader's lifetime.
    /// `None` for shm-backed instances (the shm blob Mmap is owned by `_shm_mmap`).
    _file: Option<File>,
    /// Owned mmap of the `.blm` file. Prefixed `_` because data is read via
    /// `data_ptr`; the field exists to extend the mmap's lifetime. `None` for
    /// shm-backed instances; the shm backing is `_shm_mmap`.
    _mmap: Option<Mmap>,
    /// For shm-backed instances, the `Arc<Mmap>` of the parent shm blob, keeping
    /// the mapped pages alive for this reader's lifetime. `None` for file-backed.
    _shm_mmap: Option<Arc<Mmap>>,
    /// Pointer to the bit-array bytes (after the header).
    data_ptr: *const u8,
    num_bits: u64,
    num_hashes: u32,
    num_keys: u64,
    /// Tiered `.sa` entry count this bloom was built for (index binding).
    sa_num: u64,
    /// Reference content hash this bloom was built for (index binding).
    ref_digest: [u8; 32],
    /// Order-independent fingerprint of the routing key set (index binding;
    /// distinguishes a different keep-set/routing-pad over the same reference).
    keyset_digest: u64,
}

// SAFETY: `data_ptr` points into a `Mmap` this struct keeps alive; the data is
// read-only and never mutated after construction.
unsafe impl Send for BloomFileReader {}
unsafe impl Sync for BloomFileReader {}

impl std::fmt::Debug for BloomFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFileReader")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .field("num_keys", &self.num_keys)
            .finish()
    }
}

impl BloomFileReader {
    /// Open and mmap the `.blm` file at `path`, validating its header.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: opened read-only; `_file` keeps it alive for the struct's
        // lifetime; no concurrent writers (documented).
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let (num_bits, num_hashes, num_keys, sa_num, ref_digest, keyset_digest) =
            validate_blm_header(&mmap, path)?;
        let data_ptr = unsafe { mmap.as_ptr().add(BLM_FILE_HEADER_BYTES) };
        Ok(Self {
            _file: Some(f),
            _mmap: Some(mmap),
            _shm_mmap: None,
            data_ptr,
            num_bits,
            num_hashes,
            num_keys,
            sa_num,
            ref_digest,
            keyset_digest,
        })
    }

    /// Construct a reader over a `.blm` carried inside a shm blob.
    ///
    /// `shm_mmap` is the `Arc<Mmap>` of the parent shm blob; `offset`/`len`
    /// delimit the `.blm` component within it (validated to lie in range). The
    /// header is validated exactly as in [`open`](Self::open); the returned reader
    /// keeps the blob mmap alive via `_shm_mmap`.
    pub(crate) fn from_shm_slice(shm_mmap: Arc<Mmap>, offset: usize, len: usize) -> Result<Self> {
        let synthetic = Path::new("<shm:.blm>");
        let end = offset.checked_add(len).ok_or_else(|| Error::SizeMismatch {
            file: synthetic.to_path_buf(),
            detail: format!(".blm shm range offset {offset} + len {len} overflows usize"),
        })?;
        let slice = shm_mmap
            .get(offset..end)
            .ok_or_else(|| Error::SizeMismatch {
                file: synthetic.to_path_buf(),
                detail: format!(
                    ".blm shm range [{offset}, {end}) exceeds blob size {}",
                    shm_mmap.len()
                ),
            })?;
        let (num_bits, num_hashes, num_keys, sa_num, ref_digest, keyset_digest) =
            validate_blm_header(slice, synthetic)?;
        let data_ptr = unsafe { slice.as_ptr().add(BLM_FILE_HEADER_BYTES) };
        Ok(Self {
            _file: None,
            _mmap: None,
            _shm_mmap: Some(shm_mmap),
            data_ptr,
            num_bits,
            num_hashes,
            num_keys,
            sa_num,
            ref_digest,
            keyset_digest,
        })
    }

    /// Number of keys inserted at build time (provenance/sizing).
    pub fn num_keys(&self) -> u64 {
        self.num_keys
    }

    /// Tiered `.sa` entry count this bloom was built for (index binding).
    pub fn sa_num(&self) -> u64 {
        self.sa_num
    }

    /// Reference content hash this bloom was built for (index binding); the
    /// loader compares it to the index's [`crate::index::LearnedIndex::ref_digest_hex`].
    pub fn ref_digest(&self) -> &[u8; 32] {
        &self.ref_digest
    }

    /// Order-independent routing-key-set fingerprint (index binding); the loader
    /// compares it to the `keyset_digest` recorded in `.meta` so a `.blm` built
    /// over a different keep-set/routing-pad of the same reference is rejected.
    pub fn keyset_digest(&self) -> u64 {
        self.keyset_digest
    }

    /// `true` if `key` is POSSIBLY in the set (a true member, or a false
    /// positive); `false` if `key` is DEFINITELY absent. The dispatch gate
    /// confirms a `true` with a `mem_search` before routing.
    #[inline]
    pub fn contains(&self, key: u64) -> bool {
        // Hash the key once; derive all `num_hashes` probes from the pair.
        let (h1, h2) = key_hashes(key);
        for i in 0..self.num_hashes {
            let bit = bit_at(h1, h2, i, self.num_bits);
            let byte = (bit / 8) as usize;
            debug_assert!(
                byte < (self.num_bits / 8) as usize,
                "bloom byte index {byte} out of range (body {} bytes)",
                self.num_bits / 8
            );
            // SAFETY: `bit < num_bits` (Lemire multiply-shift maps into `[0, num_bits)`) and the body is
            // `num_bits / 8` valid bytes (validated at open), so `byte` is in
            // range — the debug_assert above checks it in debug builds (this is a
            // hot per-probe read, so the bound stays debug-only in release).
            let b = unsafe { *self.data_ptr.add(byte) };
            if b & (1u8 << (bit % 8)) == 0 {
                return false;
            }
        }
        true
    }
}

/// Validate a `.blm` header. Returns `(num_bits, num_hashes, num_keys, sa_num,
/// ref_digest, keyset_digest)` — the last three are the index-binding fields the
/// loader checks.
fn validate_blm_header(data: &[u8], path: &Path) -> Result<(u64, u32, u64, u64, [u8; 32], u64)> {
    if data.len() < BLM_FILE_HEADER_BYTES {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("too small ({} bytes) for .blm header", data.len()),
        });
    }
    let magic = LittleEndian::read_u32(&data[0..4]);
    if magic != BLM_MAGIC {
        return Err(Error::BadMagic {
            file: path.to_path_buf(),
            found: format!("{magic:#010x}"),
            expected: format!("{BLM_MAGIC:#010x}"),
        });
    }
    let version = LittleEndian::read_u32(&data[4..8]);
    if version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    // Body-version (byte 20): the bit_at reduction scheme. A `.blm` whose body
    // bits were written by a binary with a different reduction (or before this
    // field existed, i.e. `0`) must be rejected — otherwise its stale bits would
    // produce bloom false negatives, which the unconfirmed `bloom_first` gate
    // would surface as silently mis-routed reads. Reject here so the loader drops
    // the bloom and the gate degrades to the exact first-window path.
    let body_version = LittleEndian::read_u32(&data[20..24]);
    if body_version != BLOOM_BODY_VERSION {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!(
                ".blm body-version {body_version} != expected {BLOOM_BODY_VERSION} \
                 (built by an incompatible prmi binary; rebuild the index's .blm)"
            ),
        });
    }
    let num_bits = LittleEndian::read_u64(&data[8..16]);
    let num_hashes = LittleEndian::read_u32(&data[16..20]);
    let num_keys = LittleEndian::read_u64(&data[24..32]);
    let sa_num = LittleEndian::read_u64(&data[32..40]);
    let mut ref_digest = [0u8; 32];
    ref_digest.copy_from_slice(&data[40..72]);
    let keyset_digest = LittleEndian::read_u64(&data[72..80]);
    if num_bits == 0 || num_bits % 64 != 0 {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("num_bits={num_bits} must be a positive multiple of 64"),
        });
    }
    // The writer clamps k to [1, 32]; reject anything outside that so a corrupt
    // file with a huge num_hashes cannot make every `contains()` do an unbounded
    // number of probes (it should be dropped by the best-effort loader instead).
    if !(1..=32).contains(&num_hashes) {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("num_hashes={num_hashes} must be in 1..=32"),
        });
    }
    // Body is `num_bits / 8` bytes. Checked arithmetic so a crafted huge num_bits
    // fails validation rather than wrapping the expected length.
    let body = usize::try_from(num_bits / 8).map_err(|_| Error::SizeMismatch {
        file: path.to_path_buf(),
        detail: format!("num_bits too large: {num_bits}"),
    })?;
    let expected_len =
        body.checked_add(BLM_FILE_HEADER_BYTES)
            .ok_or_else(|| Error::SizeMismatch {
                file: path.to_path_buf(),
                detail: format!(".blm size overflow for num_bits={num_bits}"),
            })?;
    if data.len() != expected_len {
        return Err(Error::SizeMismatch {
            file: path.to_path_buf(),
            detail: format!("file is {} bytes, expected {expected_len}", data.len()),
        });
    }
    Ok((
        num_bits,
        num_hashes,
        num_keys,
        sa_num,
        ref_digest,
        keyset_digest,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Every inserted key reports present (no false negatives by construction).
    #[test]
    fn bloom_has_no_false_negatives() {
        let keys: Vec<u64> = (0..10_000u64)
            .map(|i| splitmix64(i.wrapping_mul(7)))
            .collect();
        let params = BloomParams::for_keys(keys.len() as u64, 0.01);
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.blm");
        write_bloom_file(
            &path,
            keys.len() as u64,
            keys.iter().copied(),
            params,
            keys.len() as u64,
            &[0u8; 32],
        )
        .unwrap();

        let r = BloomFileReader::open(&path).unwrap();
        assert_eq!(r.num_keys(), keys.len() as u64);
        for &k in &keys {
            assert!(r.contains(k), "inserted key {k} must report present");
        }
    }

    /// The empirical false-positive rate over never-inserted keys is near the
    /// target (generous tolerance — this is a sanity bound, not an exact test).
    #[test]
    fn bloom_false_positive_rate_is_near_target() {
        let inserted: Vec<u64> = (0..50_000u64).map(splitmix64).collect();
        let params = BloomParams::for_keys(inserted.len() as u64, 0.01);
        let dir = tempdir().unwrap();
        let path = dir.path().join("fp.blm");
        write_bloom_file(
            &path,
            inserted.len() as u64,
            inserted.iter().copied(),
            params,
            inserted.len() as u64,
            &[0u8; 32],
        )
        .unwrap();
        let r = BloomFileReader::open(&path).unwrap();

        // Probe keys disjoint from the inserted set (different generator domain).
        let mut fp = 0u64;
        let trials = 100_000u64;
        for i in 0..trials {
            let k = splitmix64(i.wrapping_add(1_000_000_000));
            if r.contains(k) {
                fp += 1;
            }
        }
        let rate = fp as f64 / trials as f64;
        eprintln!("bloom FP rate (Lemire): {rate:.4} ({fp}/{trials}, target ~0.01)");
        assert!(
            rate < 0.05,
            "empirical fp rate {rate} unexpectedly high (target 1%)"
        );
    }

    /// Empty key set: a minimal filter where every query misses.
    #[test]
    fn bloom_empty_set_serves_nothing() {
        let params = BloomParams::for_keys(0, 0.01);
        assert_eq!(params.num_bits, 64);
        assert_eq!(params.num_hashes, 1);
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.blm");
        write_bloom_file(&path, 0, std::iter::empty(), params, 0, &[0u8; 32]).unwrap();
        let r = BloomFileReader::open(&path).unwrap();
        for i in 0..1000u64 {
            assert!(!r.contains(splitmix64(i)), "empty bloom must report absent");
        }
    }

    /// Sizing follows the optimal formulas: m grows with -ln(fp), k ~ (m/n)ln2.
    #[test]
    fn bloom_params_track_optimal_formulas() {
        let p1 = BloomParams::for_keys(1_000_000, 0.01);
        // ~9.585 bits/key at 1%; rounded to 64-bit words.
        let bits_per_key = p1.num_bits as f64 / 1_000_000.0;
        assert!(
            (9.0..11.0).contains(&bits_per_key),
            "1% bits/key = {bits_per_key}"
        );
        assert!(p1.num_bits.is_multiple_of(64));
        // A tighter fp needs a bigger filter and more hashes.
        let p2 = BloomParams::for_keys(1_000_000, 0.001);
        assert!(p2.num_bits > p1.num_bits);
        assert!(p2.num_hashes >= p1.num_hashes);
    }

    #[test]
    fn bloom_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("garbage.blm");
        std::fs::write(&path, vec![0xffu8; 100]).unwrap();
        let err = BloomFileReader::open(&path).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("magic"));
    }

    #[test]
    fn bloom_rejects_truncated_body() {
        // Valid header claiming more bits than the body provides.
        let dir = tempdir().unwrap();
        let path = dir.path().join("trunc.blm");
        let mut header = vec![0u8; BLM_FILE_HEADER_BYTES];
        LittleEndian::write_u32(&mut header[0..4], BLM_MAGIC);
        LittleEndian::write_u32(&mut header[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut header[8..16], 1024); // claims 128 body bytes
        LittleEndian::write_u32(&mut header[16..20], 4);
        LittleEndian::write_u32(&mut header[20..24], BLOOM_BODY_VERSION); // pass the body-version check
        std::fs::write(&path, &header).unwrap(); // body missing
        let err = BloomFileReader::open(&path).unwrap_err();
        assert!(matches!(err, Error::SizeMismatch { .. }), "got: {err:?}");
    }

    #[test]
    fn bloom_rejects_stale_body_version() {
        // A `.blm` whose body bits were written by a binary with a different
        // `bit_at` reduction (or before the body-version field existed, i.e. `0`)
        // must be rejected on open — otherwise its stale bits cause bloom false
        // negatives on the unconfirmed `bloom_first` gate.
        let dir = tempdir().unwrap();
        let path = dir.path().join("stale.blm");
        write_bloom_file(
            &path,
            3,
            [1u64, 2, 3],
            BloomParams::for_keys(3, 0.01),
            3,
            &[0u8; 32],
        )
        .unwrap();
        // Sanity: as written it opens fine.
        assert!(BloomFileReader::open(&path).is_ok());
        // Corrupt the body-version (byte 20) to a stale value and confirm rejection.
        for stale in [0u32, 2, 99] {
            let mut bytes = std::fs::read(&path).unwrap();
            LittleEndian::write_u32(&mut bytes[20..24], stale);
            std::fs::write(&path, &bytes).unwrap();
            let err = BloomFileReader::open(&path).unwrap_err();
            assert!(
                matches!(err, Error::SizeMismatch { .. })
                    && format!("{err}").contains("body-version"),
                "stale body-version {stale} must be rejected, got: {err:?}"
            );
        }
    }

    #[test]
    fn write_bloom_file_rejects_invalid_params() {
        // `BloomParams` fields are public, so guard against out-of-contract values
        // that are invalid (num_bits==0 — a degenerate zero-size filter) or would
        // emit an unloadable file.
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.blm");
        for bad in [
            BloomParams {
                num_bits: 0,
                num_hashes: 4,
            },
            BloomParams {
                num_bits: 100,
                num_hashes: 4,
            }, // not a multiple of 64
            BloomParams {
                num_bits: 64,
                num_hashes: 0,
            },
            BloomParams {
                num_bits: 64,
                num_hashes: 33,
            },
        ] {
            assert!(
                matches!(
                    write_bloom_file(&path, 3, [1u64, 2, 3], bad, 3, &[0u8; 32]),
                    Err(Error::Internal { .. })
                ),
                "expected Internal for {bad:?}"
            );
        }
        // A valid params set still writes.
        let ok = BloomParams {
            num_bits: 64,
            num_hashes: 4,
        };
        write_bloom_file(&path, 3, [1u64, 2, 3], ok, 3, &[0u8; 32]).unwrap();
    }

    #[test]
    fn bloom_rejects_num_hashes_over_32() {
        // Correctly sized file (num_bits=64 -> 8 body bytes) but num_hashes=33,
        // which the writer never emits; the reader must drop it.
        let dir = tempdir().unwrap();
        let path = dir.path().join("k33.blm");
        let mut buf = vec![0u8; BLM_FILE_HEADER_BYTES + 8];
        LittleEndian::write_u32(&mut buf[0..4], BLM_MAGIC);
        LittleEndian::write_u32(&mut buf[4..8], FORMAT_VERSION);
        LittleEndian::write_u64(&mut buf[8..16], 64);
        LittleEndian::write_u32(&mut buf[16..20], 33); // > 32
        LittleEndian::write_u32(&mut buf[20..24], BLOOM_BODY_VERSION); // pass the body-version check
        std::fs::write(&path, &buf).unwrap();
        let err = BloomFileReader::open(&path).unwrap_err();
        assert!(matches!(err, Error::SizeMismatch { .. }), "got: {err:?}");
    }
}
