// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Fused per-read SMEM collection — `LearnedIndex::collect_smems`.
//!
//! Collapses the consumer's per-read zigzag SMEM walk (currently 91–155 stateless
//! C→Rust FFI calls/read) into ONE native Rust entrypoint, byte-identical to
//! bwa-mem3's FMI seeding. The per-call search cost is already at parity with
//! bwa-meme (MODE2: prmi 7.0 vs 7.76 probes/call); the entire residual is **call
//! count** (bwa-meme ~53). Running the walk in one process lets the SA-interval
//! state cross `zz_left_span` → `zz_right_emit` → next-step in-register — the thing
//! the FFI boundary forces cold — collapsing the call count toward bwa-meme's.
//!
//! This is a port of the consumer's zigzag (B) driver `mem_collect_smem_zigzag`
//! (`mem_collect_smem_learned.cpp`) with `PRMI_ZIGZAG_RESEED` semantics, no-ISA
//! (MODE_NONE): every search runs at `est_hint = 0`, so the `ZzHintState`
//! machinery is omitted. Built on the existing, byte-identity-tested per-call
//! primitives (`mem_search`, `mem_search_backward`,
//! `mem_search_backward_truncated_span_rc`, `forward_truncate_below_maximal`,
//! `forward_narrow_first_below`) — no new search math.
//!
//! Pass 3 (`max_mem_intv > 0`, the FMI long-MEM reseed round) is model-seeded: one
//! `mem_search` locate at `Lstart`, then a forward narrowing
//! (`forward_narrow_first_below`) from that interval to the first depth whose occ
//! drops below `max_mem_intv` — byte-identical to the cold `forward_spectrum`
//! reference walk (the `pass3_seed_one_pivot_spectrum` oracle).

use crate::index::smem::PacEncoding;
use crate::index::LearnedIndex;

/// `true` if the ISA reseed fast-path (`PRMI_ISA`) is enabled. Read once from the
/// environment and cached for the process lifetime (the index/sidecar are immutable
/// per run). When off, the reseed runs the cold model-launch path unchanged.
fn isa_reseed_enabled() -> bool {
    // Test-only override: the env `OnceLock` below caches the first read for the whole
    // process, so tests can't toggle `PRMI_ISA` per case. This thread-local lets the
    // byte-id proptest force the ISA path on/off. Compiled out of production builds.
    #[cfg(test)]
    if let Some(forced) = tests::isa_force_get() {
        return forced;
    }
    use std::sync::OnceLock;
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var_os("PRMI_ISA").is_some())
}

/// The ISA launch hint for a reseed of a pass-1 SMEM: the reference position
/// `refpos` of the SMEM's first base, plus the SMEM's read span `[m_start, n_end]`.
/// Within that span the read matches the reference contiguously, so a sub-pivot `p`
/// matches at `refpos + (p - m_start)` and the SA index there (`isa_at`) seeds the
/// reseed forward search directly, skipping the model launch + boundary gallop.
///
/// BYTE-IDENTITY INVARIANT: the hint is fed to a seed-independent WARM START
/// (`mem_search_warmstart`), whose result is byte-identical to the cold `mem_search`
/// for ANY hint — the hint only seeds the insertion search, and the boundary search
/// expands on a miss to the true interval. So a hint that is NOT a maximal occurrence
/// (the common reseed case, where `read[pivot..]` diverges from the cached occurrence
/// past `n_end`) is still safe; it just costs a few extra probes. This is why every
/// reseeded SMEM is hinted, not only full-match ones. Proven by
/// `mem_search_warmstart_equals_cold` plus the `collect_smems_isa_*_equals_cold`
/// driver gates.
#[derive(Debug, Clone, Copy)]
struct ReseedHint {
    /// Reference position of the SMEM's first base (`sa_position_for(k)`).
    refpos: u64,
    /// Inclusive read span of the cached SMEM — the window in which `reseed_isa_hint`
    /// projects a hint for a pivot. Not a validity constraint: warm-start is
    /// byte-identical for any hint; this just bounds where a hint is worth projecting.
    m_start: usize,
    n_end: usize,
}

/// Per-primitive SA-probe attribution for the `collect_gate` harness (profiling
/// only; compiled out without `spectrum-probe-count`). Each [`Guard`] brackets one
/// helper's probe delta into a disjoint bucket so the per-read probe budget can be
/// split across pass-1 left/right, reseed left/forward, and pass 3 (slot 4).
#[cfg(feature = "spectrum-probe-count")]
pub mod attrib {
    use crate::index::spectrum::probe_count;
    use std::cell::Cell;

    /// Bucket labels, indexed by slot (see [`Guard::new`]).
    pub const LABELS: [&str; 5] = ["p1_left", "p1_right", "rs_left", "rs_fwd", "pass3"];
    thread_local! {
        static COUNTS: [Cell<u64>; 5] = const { [const { Cell::new(0) }; 5] };
    }

    /// Zero all buckets.
    pub fn reset() {
        COUNTS.with(|c| c.iter().for_each(|x| x.set(0)));
    }

    /// Snapshot of the buckets (parallel to [`LABELS`]).
    pub fn snapshot() -> [u64; 5] {
        COUNTS.with(|c| std::array::from_fn(|i| c[i].get()))
    }

    /// RAII guard: on drop, adds the probes issued during its lifetime to bucket
    /// `slot`. Place at the top of a helper; nesting two guards would double-count,
    /// so guard only one disjoint level (the `collect` helpers don't nest).
    pub struct Guard {
        slot: usize,
        start: u64,
    }
    impl Guard {
        /// Start attributing probes to bucket `slot` until this guard drops.
        pub fn new(slot: usize) -> Self {
            Self {
                slot,
                start: probe_count::get(),
            }
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            let delta = probe_count::get() - self.start;
            COUNTS.with(|c| c[self.slot].set(c[self.slot].get() + delta));
        }
    }
}

/// Test/coverage-only counter for the zigzag forward-progress ("stall") guard
/// (`spectrum-probe-count` only; compiled out otherwise). Bumped exactly when a
/// guard `break`s on no net `search_pivot` advance — which never happens on a
/// full SA (the walk always advances there), only on a tiered (position-filtered)
/// SA stall. Lets a tiered-SA test prove the guard FIRED, not merely that the
/// call returned. Thread-local, so it carries no cross-test interference under
/// parallel `cargo test`.
#[cfg(feature = "spectrum-probe-count")]
pub mod stall_guard {
    use std::cell::Cell;

    thread_local! {
        static FIRES: Cell<u64> = const { Cell::new(0) };
    }

    /// Zero the fire count for the current thread.
    pub fn reset() {
        FIRES.with(|c| c.set(0));
    }

    /// Number of times the forward-progress guard fired on the current thread.
    pub fn count() -> u64 {
        FIRES.with(|c| c.get())
    }

    /// Record one guard firing (called from the search hot path).
    pub(crate) fn bump() {
        FIRES.with(|c| c.set(c.get() + 1));
    }
}

/// Note that the zigzag forward-progress guard fired. A no-op in production (no
/// `spectrum-probe-count`), so the search hot path carries no counter.
#[cfg(not(feature = "spectrum-probe-count"))]
#[inline(always)]
fn note_stall_guard_fired() {}
#[cfg(feature = "spectrum-probe-count")]
#[inline]
fn note_stall_guard_fired() {
    stall_guard::bump();
}

/// One SMEM, mirroring the consumer's `SMEM` (FMI_search.h:86-89) field-for-field,
/// INCLUDING the signed `i64` interval fields. `l` (the FMI reverse-complement
/// bi-interval) is always 0 on the learned path (cpp:760); kept for struct/layout
/// compatibility with the consumer's `memcpy` into `SMEM[]`. `k` (SA index) and
/// `s` (occurrence count) are guaranteed non-negative on a real index; typed `i64`
/// to match the consumer's comparator and `--dump-smems` byte-for-byte.
///
/// `#[repr(C)]` with this exact field order matches `prmi_smem_t` (prmi-sys); the
/// `i64` triple forces 8-byte alignment, so a 4-byte pad follows the `u32` triple
/// (`k` at offset 16, not 12). A layout test pins this.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Smem {
    /// Read index (stamped into every emitted SMEM; the consumer groups by it).
    pub rid: u32,
    /// Match start in the read (inclusive).
    pub m: u32,
    /// Match end in the read (inclusive).
    pub n: u32,
    /// SA-interval start index (raw 2× SA). Non-negative; `i64` for ABI parity.
    pub k: i64,
    /// Reverse-complement bi-interval start; always 0 on the learned path.
    pub l: i64,
    /// Occurrence count = SA-interval size. Non-negative; `i64` for ABI parity.
    pub s: i64,
}

/// SMEM-collection parameters, mirroring the bwa-mem3 `mem_opt_t` fields the
/// seeding walk reads. `split_len` is caller-precomputed
/// `round(min_seed_len * split_factor)` (cpp:1700). `max_mem_intv == 0` skips
/// pass 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectOpts {
    /// Minimum seed length; emitted SMEMs span `>= min_seed_len`.
    pub min_seed_len: u32,
    /// Reseed length threshold: pass-1 SMEMs with span `>= split_len` are reseeded.
    pub split_len: u32,
    /// Reseed occurrence threshold: only pass-1 SMEMs with `s <= split_width`.
    pub split_width: i64,
    /// Pass-3 `max_mem_intv` strategy; `0` disables pass 3.
    pub max_mem_intv: i64,
}

/// Reusable per-read scratch for [`LearnedIndex::collect_smems_into`], letting a
/// caller amortize the two per-read heap allocations the walk needs — the emitted
/// SMEM buffer and the reseed work-list — across many reads. bwa-mem3 invokes the
/// collector once per read over millions of reads; constructing one `CollectScratch`
/// per worker thread and passing it by `&mut` to every call replaces those two
/// allocate/free pairs per read with a `clear()` (capacity is retained, not freed).
///
/// The buffers are an internal detail with no observable effect on output: each call
/// clears them before use, so a fresh `CollectScratch` and a reused one produce
/// byte-identical SMEMs. The plain [`LearnedIndex::collect_smems`] entry point keeps
/// its old behavior by allocating a throwaway `CollectScratch` per call.
#[derive(Debug, Default)]
pub struct CollectScratch {
    /// Emitted SMEMs in walk order, before the within-read sort (grows to the
    /// per-read SMEM count, then is cleared for the next read).
    smems: Vec<Smem>,
    /// Reseed work-list (pivot, `min_intv`, optional ISA hint) selected from pass 1
    /// and consumed by pass 2.
    reseeds: Vec<(usize, i64, Option<ReseedHint>)>,
}

impl CollectScratch {
    /// A scratch with no preallocated capacity. The first read grows the buffers;
    /// subsequent reads reuse that capacity.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LearnedIndex {
    /// Collect all SMEMs for ONE read (passes 1+2[+3], sorted within-read per the
    /// design §4.7), byte-identical to FMI seeding.
    ///
    /// `read` is 2-bit encoded (`0..=3`, `4` = N). `rid` is stamped into every
    /// emitted [`Smem::rid`]. `pac`/`enc` are the reference encoding (as for
    /// [`LearnedIndex::mem_search`]). Writes up to `out.len()` SMEMs.
    ///
    /// Returns `Ok(count)` with the SMEMs written to `out[..count]`, or
    /// `Err(needed)` if `out` is too small (the caller grows `out` to `>= needed`
    /// and retries). A safe per-read capacity is `2 * read.len()` (passes 1+2 emit
    /// `<= read.len()`; pass 3 adds `<= read.len()`).
    ///
    /// `opts.max_mem_intv > 0` enables the pass-3 `max_mem_intv` strategy (the FMI
    /// long-MEM reseed round); `0` disables it.
    ///
    /// Concurrency-safe: `&self` is read-only (SA/model/kmt + the init-once
    /// `base_intervals`); all scratch is stack/local, so many worker threads may
    /// call this on a shared index with no shared mutable state.
    pub fn collect_smems(
        &self,
        read: &[u8],
        rid: u32,
        opts: &CollectOpts,
        pac: &[u8],
        enc: PacEncoding,
        out: &mut [Smem],
    ) -> Result<usize, usize> {
        // Thin wrapper: allocate a throwaway scratch and delegate. Callers that run
        // the collector per read over many reads should hold a `CollectScratch` and
        // call `collect_smems_into` to amortize these allocations.
        let mut scratch = CollectScratch::new();
        self.collect_smems_into(read, rid, opts, pac, enc, out, &mut scratch)
    }

    /// Like [`collect_smems`](Self::collect_smems), but reuses the caller-held
    /// `scratch` for the two per-read buffers (emitted SMEMs and reseed work-list)
    /// instead of allocating fresh ones. Byte-identical to `collect_smems` for any
    /// `scratch` state — the buffers are cleared on entry, so prior contents do not
    /// leak into the result; the only effect is amortizing the allocations across
    /// reads. Intended for the hot per-read loop (e.g. bwa-mem3 over millions of
    /// reads): construct one `CollectScratch` per worker thread and pass it to every
    /// call.
    ///
    /// `out` and the return value behave exactly as in `collect_smems`: `Ok(count)`
    /// with the sorted SMEMs in `out[..count]`, or `Err(needed)` if `out` is too
    /// small.
    #[allow(clippy::too_many_arguments)]
    pub fn collect_smems_into(
        &self,
        read: &[u8],
        rid: u32,
        opts: &CollectOpts,
        pac: &[u8],
        enc: PacEncoding,
        out: &mut [Smem],
        scratch: &mut CollectScratch,
    ) -> Result<usize, usize> {
        self.collect_smems_unsorted_into(read, rid, opts, pac, enc, scratch);
        let smems = &mut scratch.smems;
        Self::sort_within_read(smems);
        if smems.len() > out.len() {
            return Err(smems.len());
        }
        out[..smems.len()].copy_from_slice(smems);
        Ok(smems.len())
    }

    /// Cheap tiered-dispatch pre-reject: does the read's first full 32-mer window
    /// occur in this index? One `mem_search` locate (≈1–2 cold probes) — far
    /// cheaper than a full `collect_smems`, whose failed reseeds on an absent read
    /// cost hundreds of probes. Used by a tiered (Design Z) consumer to send
    /// off-target reads to the whole-genome fallback without paying the full
    /// fast-path search. `read` is 2-bit encoded (`0..=3`, `4` = N); windows
    /// containing an N are skipped. Inspects only the FIRST N-free 32-mer window
    /// from the read start: returns `true` iff that window fully occurs
    /// (`match_len == 32`), and `false` if it does not, or if the read has no
    /// N-free 32-mer window.
    pub fn present_anchor(&self, read: &[u8], pac: &[u8], enc: PacEncoding) -> bool {
        const K: usize = 32;
        let mut start = 0usize;
        'windows: loop {
            // Fail closed before slicing a caller-provided read: bound-check the
            // window end with checked arithmetic rather than `start + K`.
            let Some(end) = start.checked_add(K) else {
                return false;
            };
            if end > read.len() {
                return false;
            }
            // Reject a window with any N by jumping past the offending base.
            for (j, &base) in read[start..end].iter().enumerate() {
                if base >= 4 {
                    start += j + 1;
                    continue 'windows;
                }
            }
            return self.mem_search(&read[start..end], pac, enc).match_len >= K as u64;
        }
    }

    /// Within-read two-stage sort (port of cpp:1833-1850, restricted to one rid).
    /// Stage 1 is the per-rid restriction of the global `compare_smem`
    /// (`m ASC, n DESC`); stage 2 is the per-rid `ks_introsort(mem_intv1_learned)`
    /// by `(m<<32)|n` (`m ASC, n ASC`). Both C++ sorts are unstable, so for SMEMs
    /// that share `(m, n)` the `(k, s)` order is the unstable-sort result and is NOT
    /// derivable from the spec — this composition is deterministic but its tie order
    /// is verified at the consumer box-gate.
    // KNOWN OPEN: (m,n)-tie order vs the C++ unstable introsort (deferred to box-gate).
    fn sort_within_read(smems: &mut [Smem]) {
        // Stage 1: m ASC, n DESC.
        smems.sort_unstable_by(|a, b| a.m.cmp(&b.m).then(b.n.cmp(&a.n)));
        // Stage 2: m ASC, n ASC (dominates stage 1 for distinct (m,n)).
        smems.sort_unstable_by(|a, b| a.m.cmp(&b.m).then(a.n.cmp(&b.n)));
    }

    /// Allocating convenience used by the byte-identity oracles/proptests: runs the
    /// walk into a fresh scratch and returns the emitted SMEMs. Production code goes
    /// through [`collect_smems_into`](Self::collect_smems_into) with a reused scratch.
    #[cfg(test)]
    fn collect_smems_unsorted(
        &self,
        read: &[u8],
        rid: u32,
        opts: &CollectOpts,
        pac: &[u8],
        enc: PacEncoding,
    ) -> Vec<Smem> {
        let mut scratch = CollectScratch::new();
        self.collect_smems_unsorted_into(read, rid, opts, pac, enc, &mut scratch);
        scratch.smems
    }

    /// The per-read walk (passes 1+2[+3]) in EMISSION order, before the within-read
    /// sort (Task 7). Port of the `mem_collect_smem_zigzag` driver (cpp:1644-1834),
    /// no-ISA. Fills the caller-held `scratch.smems` (cleared on entry) and reuses
    /// `scratch.reseeds` as the pass-2 work-list, allocating nothing per read once
    /// the buffers have grown; the emitted SMEMs are left in `scratch.smems`.
    fn collect_smems_unsorted_into(
        &self,
        read: &[u8],
        rid: u32,
        opts: &CollectOpts,
        pac: &[u8],
        enc: PacEncoding,
        scratch: &mut CollectScratch,
    ) {
        let rlen = read.len();
        // Split the borrow so the emit buffer and the reseed work-list can be held as
        // disjoint `&mut Vec` (pass 2 drains `reseeds` while pushing into `smems`).
        let CollectScratch { smems, reseeds } = scratch;
        smems.clear();
        reseeds.clear();

        // ---- Pass 1: zigzag walk over all positions (min_intv = 1) ----
        let mut x = 0;
        while x < rlen {
            self.zz_step1(read, rid, &mut x, 1, opts.min_seed_len, smems, pac, enc);
        }
        let num1 = smems.len();

        // ---- Reseed selection (filter, preserves pass-1 order; cpp:1702-1719) ----
        // For each pass-1 SMEM with span >= split_len && s <= split_width, reseed at
        // the midpoint with min_intv = s + 1. (The no-ISA port omits the refpos cache.)
        let split_len = opts.split_len;
        // ISA reseed cache (gap A+E): when enabled and a `.isa` is loaded, each
        // reseeded pass-1 SMEM carries its `refpos` (a single SA read) so the reseed's
        // forward searches warm-start from `isa_at` instead of a model lookup + gallop.
        // Gated on `kmt.is_none()`: with a k-mer table loaded `mem_search_warmstart`
        // ignores the hint (the tabled trace is the launch), so building the hint
        // (`sa_position_for` + `isa_at`) would be dead per-reseed work.
        let use_isa = isa_reseed_enabled() && self.has_isa() && self.kmt.is_none();
        for p in &smems[..num1] {
            let span = p.n + 1 - p.m;
            if span < split_len || p.s > opts.split_width {
                continue;
            }
            let mid = ((p.m + p.n + 1) >> 1) as usize;
            // Hint EVERY reseeded SMEM (occ ∈ [1, split_width] by the loop filter
            // above). The reseed's forward search warm-starts from
            // `isa_at(refpos + offset)` — byte-identical for any hint (the search
            // expands on a miss to the true boundary), skipping the model launch when
            // the hint is good.
            let hint = if use_isa {
                Some(ReseedHint {
                    refpos: self.sa_position_for(p.k as u64),
                    m_start: p.m as usize,
                    n_end: p.n as usize,
                })
            } else {
                None
            };
            reseeds.push((mid, p.s + 1, hint));
        }

        // ---- Pass 2: one reseed walk per selected pivot ----
        // Drain (not consume-by-value) so `reseeds` keeps its capacity for the next read.
        for (pivot, min_intv, hint) in reseeds.drain(..) {
            self.zz_step1_reseed(
                read,
                rid,
                pivot,
                min_intv,
                opts.min_seed_len,
                smems,
                hint,
                pac,
                enc,
            );
        }

        // ---- Pass 3: max_mem_intv strategy (gated; cpp:1793-1829) ----
        if opts.max_mem_intv > 0 {
            let msl1 = opts.min_seed_len as usize + 1;
            let mut x = 0;
            while x < rlen {
                x = self.pass3_seed_one_pivot(
                    read,
                    rid,
                    x,
                    opts.max_mem_intv,
                    msl1,
                    smems,
                    pac,
                    enc,
                );
            }
        }
    }

    /// Maximal LEFT exact-match span INCLUDING the pivot base — the analogue of
    /// BWA-MEME's `ss_exact_match_len` (port of `zz_left_span`, cpp:834-878). The
    /// driver repositions `pivot = pivot - span + 1`. Returns `>= 1` (the pivot
    /// base itself), even on a degenerate/no-extension anchor.
    ///
    /// One backward extension with the SA-interval carried in-register (the
    /// in-process win vs the consumer's two FFI crossings): a length-1 forward
    /// anchor at `read[pivot]` is extended left, and `mem_search_backward` returns
    /// `match_len` = the TOTAL span (1 + left_ext). `zz_left_span` uses ONLY the
    /// span (the interval is discarded).
    ///
    /// The forward anchor `mem_search` is elided: on the span-only (ambiguous) path
    /// the anchor `sa_start` is unused, and its `occ` only gated an `occ==0 ->
    /// return 1` redundant with the RC search's `match_len <= anchor_len ->
    /// anchor_len` floor (a non-occurring base yields `match_len=0`). The unique-
    /// anchor fast path (`occ==1 && sa_start!=0`) cannot fire for the `sa_start=0`
    /// unit sentinel, so it reproduces the genome path exactly — one fewer
    /// `mem_search` per left-extension step.
    fn zz_left_span(&self, read: &[u8], pivot: usize, pac: &[u8], enc: PacEncoding) -> usize {
        #[cfg(feature = "spectrum-probe-count")]
        let _g = attrib::Guard::new(0); // p1_left
        let m = self.mem_search_backward(0, 1, 1, read, pivot, pac, enc);
        if m.match_len == 0 {
            return 1; // no extension (cpp:870-876)
        }
        m.match_len as usize
    }

    /// Forward maximal match from `pivot` over the N-clamped window of length
    /// `qlen`, emitting ONE SMEM if it clears the seed gate (port of `zz_right_emit`,
    /// cpp:694-816; the `ZzHintState` block cpp:713-812 is omitted in the no-ISA
    /// port). Returns `match_len` (returned ALWAYS — the driver advances
    /// `search_pivot = pivot + match_len`).
    ///
    /// Emit gate (cpp:753): `match_len >= min_seed_len && occ >= min_intv`. The
    /// emitted SMEM is `{rid, m=pivot, n=pivot+match_len-1, k=sa_start, l=0, s=occ}`.
    #[allow(clippy::too_many_arguments)]
    fn zz_right_emit(
        &self,
        read: &[u8],
        rid: u32,
        pivot: usize,
        qlen: usize,
        min_intv: i64,
        min_seed_len: u32,
        out: &mut Vec<Smem>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> usize {
        #[cfg(feature = "spectrum-probe-count")]
        let _g = attrib::Guard::new(1); // p1_right
        if qlen == 0 {
            return 0;
        }
        let m = self.mem_search(&read[pivot..pivot + qlen], pac, enc);
        let match_len = m.match_len as usize;
        if m.match_len >= min_seed_len as u64 && m.occ as i64 >= min_intv {
            out.push(Smem {
                rid,
                m: pivot as u32,
                n: (pivot + match_len - 1) as u32,
                k: m.sa_start as i64,
                l: 0,
                s: m.occ as i64,
            });
        }
        match_len
    }

    /// One pass-1 zigzag invocation at `*io_pivot` (port of `zz_step1`, cpp:898-975,
    /// non-tradeoff / no-ISA). Emits SMEMs onto `out` and advances `*io_pivot` to
    /// BWA-MEME's `next_pivot`. The driver's pass-1 loop is
    /// `x=0; while x<rlen { zz_step1(&mut x) }`.
    #[allow(clippy::too_many_arguments)]
    fn zz_step1(
        &self,
        read: &[u8],
        rid: u32,
        io_pivot: &mut usize,
        min_intv: i64,
        min_seed_len: u32,
        out: &mut Vec<Smem>,
        pac: &[u8],
        enc: PacEncoding,
    ) {
        let rlen = read.len();
        let msl = min_seed_len as usize;
        let mut pivot = *io_pivot;

        // ---- read[pivot] ambiguous (cpp:916-921) ----
        if read[pivot] >= 4 {
            pivot = if rlen - pivot < msl { rlen } else { pivot + 1 };
            *io_pivot = pivot;
            return;
        }

        if pivot != 0 && read[pivot - 1] < 4 {
            // ---- zigzag loop (cpp:923-959) ----
            let next_pivot = rlen;
            let mut search_pivot = pivot;
            while search_pivot < next_pivot {
                // Forward-progress guard (see `zz_step1_reseed` for the full rationale):
                // on the full-genome SA the walk always advances, so this is a no-op and
                // byte-identical; a tiered (position-filtered) SA can stall it with
                // `right > 0` but no net progress — break in that case.
                let entry_search_pivot = search_pivot;
                // Ambiguous guard at the (re)entry position (cpp:930-939).
                if read[search_pivot] >= 4 {
                    if rlen - search_pivot < msl {
                        pivot = rlen;
                        search_pivot = rlen;
                    } else {
                        search_pivot += 1;
                        pivot += 1;
                    }
                    continue;
                }
                // Left extension (non-emitting), reposition pivot. `pivot + 1 - left`
                // avoids usize underflow (left <= pivot+1 by construction).
                let left = self.zz_left_span(read, pivot, pac, enc);
                pivot = pivot + 1 - left;
                if next_pivot - pivot < msl {
                    break; // cpp:945
                }
                // Right extension (EMITS), advance search_pivot.
                let qlen = self.fwd_qlen(read, pivot);
                let right = self.zz_right_emit(
                    read,
                    rid,
                    pivot,
                    qlen,
                    min_intv,
                    min_seed_len,
                    out,
                    pac,
                    enc,
                );
                search_pivot = pivot + right; // cpp:956
                if search_pivot <= entry_search_pivot {
                    note_stall_guard_fired();
                    break; // stall guard: no forward progress (tiered-SA safety; no-op on full SA)
                }
                pivot = search_pivot; // cpp:957
            }
            pivot = next_pivot; // cpp:959
        } else {
            // ---- left boundary: single right emit (cpp:960-971) ----
            let qlen = self.fwd_qlen(read, pivot);
            let right = self.zz_right_emit(
                read,
                rid,
                pivot,
                qlen,
                min_intv,
                min_seed_len,
                out,
                pac,
                enc,
            );
            pivot += right;
            if right == 0 {
                pivot += 1; // infinite-loop guard (cpp:970): read[pivot] doesn't occur
            }
        }

        *io_pivot = pivot;
    }

    /// N-clamped forward window length from `p` (stop at the first `read[i] >= 4`),
    /// mirroring the consumer's `fwd_qlen` (cpp:907-913).
    #[inline]
    fn fwd_qlen(&self, read: &[u8], p: usize) -> usize {
        let rlen = read.len();
        for (i, &b) in read.iter().enumerate().skip(p) {
            if b >= 4 {
                return i - p;
            }
        }
        rlen - p
    }

    /// The `min_intv`-bounded forward extent from `pivot`: the longest `L` in
    /// `[1, Lmax]` whose length-`L` forward match has `occ >= min_intv` (`occ` is
    /// monotone non-increasing in `L`). Returns `(emit_len, sa, occ, lmax)` where
    /// `emit_len` is that `L` (or 0 if even length-1 is too frequent), `(sa, occ)`
    /// is its interval, and `lmax` is the maximal-match length. Shared by the
    /// next_pivot computation and `zz_right_emit_reseed` (emit) so they agree by
    /// construction. Port of cpp:1037-1118 / 1043-1057 (no-cap, no-hint).
    ///
    /// `want_interval == false` (the next_pivot caller, which uses only `emit_len`)
    /// returns `(emit_len, 0, 0, lmax)` and skips the truncation interval recovery.
    /// The truncation itself is `forward_truncate_below_maximal` (forward intervals
    /// nest, so it expands the maximal interval outward — no per-length re-locate).
    /// Resolve a [`ReseedHint`] to an SA launch index for `pivot`, or `None` to fall
    /// back to the cold search. Projected ONLY inside the cached SMEM span `[m_start,
    /// n_end]`: outside it the read no longer matches the reference at `refpos`, so a
    /// projected hint would be far from the maximal interval (still byte-id-safe via
    /// the warm start, but not worth projecting). `pivot < m_start` also guards the
    /// `pivot - m_start` subtraction.
    #[inline]
    fn reseed_isa_hint(&self, pivot: usize, hint: Option<ReseedHint>) -> Option<u64> {
        let ReseedHint {
            refpos,
            m_start,
            n_end,
        } = hint?;
        if pivot < m_start || pivot > n_end {
            return None;
        }
        self.isa_at(refpos + (pivot - m_start) as u64)
    }

    /// RC-strand launch hint for the LEFT reseed — the reverse-complement twin of
    /// [`reseed_isa_hint`](Self::reseed_isa_hint). Same `[m_start, n_end]` span guard;
    /// the formula is `2*l_pac − refpos − (pivot − m_start) − 1` (the doubled-text
    /// complement-mirror of the forward position `refpos + off`, matching BWA-MEME
    /// `LearnedIndex_seeding.cpp:1535`). Base is `2*l_pac` (BWA-MEME's
    /// `suffix_array_num`), NOT prmi's `sa_num() == 2*l_pac + 1` — the `+1` is the
    /// sentinel, and using it here is off by one (it points one past the RC
    /// occurrence, so the warm-start would gallop back; caught by
    /// `reseed_rc_hint_reduces_probes`). Returns `None` outside the span or on
    /// coordinate over/underflow (the `2*l_pac` base and each mirror subtraction
    /// are checked). Fed as the `seed_hint` of
    /// `mem_search_backward_span_rc_warmstart` (a WARM-START, so a wrong projection is
    /// byte-id-safe — it only costs probes).
    #[inline]
    fn reseed_rc_hint(&self, pivot: usize, hint: Option<ReseedHint>) -> Option<u64> {
        let ReseedHint {
            refpos,
            m_start,
            n_end,
        } = hint?;
        if pivot < m_start || pivot > n_end {
            return None;
        }
        let off = (pivot - m_start) as u64;
        let rc = self
            .l_pac()
            .checked_mul(2)?
            .checked_sub(refpos)?
            .checked_sub(off)?
            .checked_sub(1)?;
        self.isa_at(rc)
    }

    #[allow(clippy::too_many_arguments)]
    fn reseed_bounded_fwd(
        &self,
        read: &[u8],
        pivot: usize,
        qlen: usize,
        min_intv: i64,
        want_interval: bool,
        hint: Option<ReseedHint>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> (usize, u64, u64, usize) {
        #[cfg(feature = "spectrum-probe-count")]
        let _g = attrib::Guard::new(3); // rs_fwd
        if qlen == 0 {
            return (0, 0, 0, 0);
        }
        // Live ISA reseed (PRMI_ISA): warm-start the forward search from the projected
        // parent-SMEM hint. `mem_search_warmstart` is byte-identical to cold
        // `mem_search` for ANY hint (find_boundary expands on a miss to the true
        // boundary), so a non-maximal hint is safe — it just costs extra probes. When
        // the hint is good the insertion search collapses to ~1-2 probes, skipping the
        // model launch. No trust path, no maximality confirm, no fallback decision.
        let q = &read[pivot..pivot + qlen];
        let mm = match self.reseed_isa_hint(pivot, hint) {
            Some(h) => self.mem_search_warmstart(q, h, pac, enc),
            None => self.mem_search(q, pac, enc),
        };
        let lmax = mm.match_len as usize;
        if lmax == 0 {
            return (0, 0, 0, 0);
        }
        if mm.occ as i64 >= min_intv {
            return (lmax, mm.sa_start, mm.occ, lmax); // maximal already bounded
        }
        // Largest L < Lmax with occ(L) >= min_intv. The forward analogue of TRUNC_IV:
        // walk the forward interval outward from the maximal (one model launch + a
        // capped linear scan), then recover the crossing interval only when the
        // caller needs it. Byte-identical (same L*, exact interval).
        let t = self.forward_truncate_below_maximal(
            &read[pivot..pivot + qlen],
            mm,
            min_intv as u64,
            want_interval,
            pac,
            enc,
        );
        (t.match_len as usize, t.sa_start, t.occ, lmax)
    }

    /// Reseed (pass-2) non-emitting left extension (port of `zz_left_span_reseed`,
    /// TRUNC_IV branch cpp:1199-1209). One `mem_search_backward_truncated_span_rc`
    /// that returns the `min_intv`-bounded left span `L*` (the maximal RC search +
    /// truncation, interval recovery elided since the span is all the driver uses).
    /// Returns the total span `>= 1` (driver repositions `pivot = pivot - span + 1`).
    ///
    /// The len-1 forward anchor BWA-MEME's reseed launches from is elided: the
    /// span-only RC path never reads the anchor `sa_start`, and its `occ` only gated
    /// an `occ==0 -> return 1` that is redundant with the RC search's own
    /// `span_max <= anchor_len -> 1` floor (a non-occurring base yields
    /// `span_max=0`). So pass a unit sentinel interval (`occ_count = 1`,
    /// `anchor_len = 1`) and let the RC search do the occurrence test — one fewer
    /// `mem_search` per reseed-left step.
    fn zz_left_span_reseed(
        &self,
        read: &[u8],
        pivot: usize,
        min_intv: i64,
        hint: Option<ReseedHint>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> usize {
        #[cfg(feature = "spectrum-probe-count")]
        let _g = attrib::Guard::new(2); // rs_left

        // Live ISA reseed (PRMI_ISA): warm-start the RC walk from the projected
        // RC-strand hint. Byte-identical to the cold walk for ANY hint (the warm
        // start only seeds the internal insertion search), so a None/garbage
        // projection is safe — it just costs probes. Mirrors the forward reseed in
        // `reseed_bounded_fwd`.
        let seed = self.reseed_rc_hint(pivot, hint);
        let span = self.mem_search_backward_span_rc_warmstart(
            1, // occ_count: unit sentinel (nonzero); the RC floor handles non-occurrence
            1, // anchor_len
            read,
            pivot,
            min_intv as u64,
            seed,
            pac,
            enc,
        );
        if span == 0 {
            return 1;
        }
        span as usize
    }

    /// Reseed (pass-2) right extension that emits (port of `zz_right_emit_reseed`,
    /// cpp:1060-1138, no-ISA). Emits the `min_intv`-bounded SMEM gated on
    /// `emit_len >= min_seed_len` ALONE (the interval already satisfies `min_intv`
    /// by construction, cpp:1120-1122). Returns the length to advance by
    /// (`emit_len`, or `Lmax` when nothing emits — cpp:1137).
    #[allow(clippy::too_many_arguments)]
    fn zz_right_emit_reseed(
        &self,
        read: &[u8],
        rid: u32,
        pivot: usize,
        qlen: usize,
        min_intv: i64,
        min_seed_len: u32,
        out: &mut Vec<Smem>,
        hint: Option<ReseedHint>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> usize {
        if qlen == 0 {
            return 0;
        }
        let (emit_len, emit_sa, emit_occ, lmax) =
            self.reseed_bounded_fwd(read, pivot, qlen, min_intv, true, hint, pac, enc);
        if emit_len >= min_seed_len as usize && emit_occ > 0 {
            out.push(Smem {
                rid,
                m: pivot as u32,
                n: (pivot + emit_len - 1) as u32,
                k: emit_sa as i64,
                l: 0,
                s: emit_occ as i64,
            });
        }
        if emit_len > 0 {
            emit_len
        } else {
            lmax
        }
    }

    /// One reseed (pass-2) zigzag invocation at a single reseed `pivot` (port of
    /// `zz_step1_reseed`, cpp:1273-1350, no-ISA). Single-shot (does NOT walk
    /// pivots); `next_pivot` is bounded by the `min_intv`-bounded forward extent
    /// from the ORIGINAL pivot (cpp:1303-1306), NOT `rlen`. Emits onto `out`.
    #[allow(clippy::too_many_arguments)]
    fn zz_step1_reseed(
        &self,
        read: &[u8],
        rid: u32,
        pivot: usize,
        min_intv: i64,
        min_seed_len: u32,
        out: &mut Vec<Smem>,
        hint: Option<ReseedHint>,
        pac: &[u8],
        enc: PacEncoding,
    ) {
        let rlen = read.len();
        let msl = min_seed_len as usize;
        if pivot >= rlen || read[pivot] >= 4 {
            return; // ambiguous / OOB: no emit, single-shot (cpp:1288-1292)
        }
        let mut pivot = pivot;
        if pivot != 0 && read[pivot - 1] < 4 {
            // next_pivot from the ORIGINAL reseed pivot (cpp:1302-1306).
            let qlen0 = self.fwd_qlen(read, pivot);
            // next_pivot needs only the length -> span-only (skip interval recovery).
            let fwd_ext = self
                .reseed_bounded_fwd(read, pivot, qlen0, min_intv, false, hint, pac, enc)
                .0;
            let mut next_pivot = pivot + if fwd_ext > 0 { fwd_ext } else { 1 };
            if next_pivot > rlen {
                next_pivot = rlen;
            }
            let mut search_pivot = pivot;
            while search_pivot < next_pivot {
                // Forward-progress guard. The reseed walk must push the right end
                // (`search_pivot`) past where it was at the top of the iteration, or it has
                // stalled. On the full-genome SA the walk always advances, so this never
                // fires and the output is byte-identical; a tiered (position-filtered) SA
                // can leave `search_pivot` stationary — `pivot` is pulled left by
                // `zz_left_span_reseed` and pushed right by `zz_right_emit_reseed` by the
                // same amount (the left RC span and right extension disagree because some
                // copies are absent), so `right > 0` yet there is no net progress. This
                // supersedes the cpp:1337 `right == 0` guard, which only caught the
                // no-match special case (a subset of "no progress").
                let entry_search_pivot = search_pivot;
                if read[search_pivot] >= 4 {
                    if rlen - search_pivot < msl {
                        pivot = rlen;
                        search_pivot = rlen;
                    } else {
                        search_pivot += 1;
                        pivot += 1;
                    }
                    continue;
                }
                let left = self.zz_left_span_reseed(read, pivot, min_intv, hint, pac, enc);
                pivot = pivot + 1 - left;
                if next_pivot - pivot < msl {
                    break; // cpp:1324
                }
                let qlen = self.fwd_qlen(read, pivot);
                let right = self.zz_right_emit_reseed(
                    read,
                    rid,
                    pivot,
                    qlen,
                    min_intv,
                    min_seed_len,
                    out,
                    hint,
                    pac,
                    enc,
                );
                search_pivot = pivot + right; // cpp:1334
                if search_pivot <= entry_search_pivot {
                    note_stall_guard_fired();
                    break; // stall guard (supersedes cpp:1337 right==0): no forward progress
                }
                pivot = search_pivot; // cpp:1338
            }
        } else {
            // Left boundary: single right emit (cpp:1340-1347).
            let qlen = self.fwd_qlen(read, pivot);
            self.zz_right_emit_reseed(
                read,
                rid,
                pivot,
                qlen,
                min_intv,
                min_seed_len,
                out,
                hint,
                pac,
                enc,
            );
        }
    }

    /// Pass-3 (`max_mem_intv` strategy) seeding at one pivot (port of
    /// `pass3_seed_one_pivot_learned`, cpp:457-530). Forward-only: emits <=1 SMEM at
    /// the first length `L >= max(min_seed_len + 1, 2)` whose occ drops below
    /// `max_mem_intv` (and `> 0`). Returns `next_pivot` (the driver sets `x = next`).
    ///
    /// Perf: the common case (the length-`Lstart` window already occurs `<
    /// max_mem_intv` times) is served by ONE **model-seeded** `mem_search` locate,
    /// skipping the cold `forward_spectrum` trace that binary-searches the whole SA
    /// at shallow depths. Only the rare repetitive case (`occ(Lstart) >=
    /// max_mem_intv`, needing a forward extension to the crossing) falls back to the
    /// model-seeded forward narrow ([`forward_narrow_first_below`], seeded at the
    /// `Lstart` interval). Byte-identical to [`pass3_seed_one_pivot_spectrum`]:
    /// `mem_search`'s `occ` at length `L` equals `forward_spectrum`'s `occ_at(L)`,
    /// and the forward narrow reproduces each deeper break point exactly.
    ///
    /// [`forward_narrow_first_below`]: LearnedIndex::forward_narrow_first_below
    #[allow(clippy::too_many_arguments)]
    fn pass3_seed_one_pivot(
        &self,
        read: &[u8],
        rid: u32,
        pivot: usize,
        max_mem_intv: i64,
        min_seed_len_plus1: usize,
        out: &mut Vec<Smem>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> usize {
        #[cfg(feature = "spectrum-probe-count")]
        let _g = attrib::Guard::new(4); // pass3
        let rlen = read.len();
        if pivot >= rlen || read[pivot] >= 4 {
            return pivot + 1; // N / OOB (cpp:463-466)
        }
        let qlen = self.fwd_qlen(read, pivot);
        let lstart = min_seed_len_plus1.max(2);
        let boundary = || {
            if pivot + qlen < rlen {
                pivot + qlen + 1 // an N stopped the window (cpp:528)
            } else {
                pivot + qlen
            }
        };
        // The spectrum scan's loop `for L in lstart..=qlen` doesn't run when
        // lstart > qlen → window boundary.
        if lstart > qlen {
            return boundary();
        }
        // Fast path: one model-seeded locate at Lstart. occ_at(Lstart) = occ iff the
        // length-Lstart window fully occurs (match_len == Lstart), else 0.
        let m = self.mem_search(&read[pivot..pivot + lstart], pac, enc);
        let occ_lstart = if m.match_len as usize >= lstart {
            m.occ as i64
        } else {
            0
        };
        if occ_lstart < max_mem_intv {
            // Crossing is at Lstart (the spectrum loop's first iteration).
            if occ_lstart > 0 {
                out.push(Smem {
                    rid,
                    m: pivot as u32,
                    n: (pivot + lstart - 1) as u32,
                    k: m.sa_start as i64,
                    l: 0,
                    s: occ_lstart,
                });
            }
            return pivot + lstart;
        }
        // occ(Lstart) >= max_mem_intv: narrow FORWARD from the model-seeded Lstart
        // interval to the first deeper length with occ < max_mem_intv — instead of
        // the cold forward_spectrum. (occ_lstart >= max_mem_intv > 0 ⇒ match_len ==
        // Lstart, so [sa_start, sa_start+occ) is the exact Lstart interval.)
        match self.forward_narrow_first_below(
            &read[pivot..pivot + qlen],
            lstart,
            m.sa_start,
            m.sa_start + m.occ,
            max_mem_intv as u64,
            pac,
            enc,
        ) {
            Some((l, lo, occ)) => {
                if occ > 0 {
                    out.push(Smem {
                        rid,
                        m: pivot as u32,
                        n: (pivot + l - 1) as u32,
                        k: lo as i64,
                        l: 0,
                        s: occ as i64,
                    });
                }
                pivot + l
            }
            // occ stayed >= max_mem_intv through the window: boundary, no emit.
            None => boundary(),
        }
    }

    /// Reference pass-3 via the full `forward_spectrum` trace (cold; binary-searches
    /// the whole SA at shallow depths). Kept as the byte-identity oracle for the
    /// model-seeded [`pass3_seed_one_pivot`] (fast path + forward-narrow fallback).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn pass3_seed_one_pivot_spectrum(
        &self,
        read: &[u8],
        rid: u32,
        pivot: usize,
        max_mem_intv: i64,
        min_seed_len_plus1: usize,
        out: &mut Vec<Smem>,
        pac: &[u8],
        enc: PacEncoding,
    ) -> usize {
        let rlen = read.len();
        if pivot >= rlen || read[pivot] >= 4 {
            return pivot + 1; // N / OOB: SMEM block skipped, advance by 1 (cpp:463-466)
        }
        let qlen = self.fwd_qlen(read, pivot);
        let boundary_is_n = pivot + qlen < rlen;
        let steps = self.forward_spectrum(&read[pivot..pivot + qlen], pac, enc);

        // occ at forward length L (first breakpoint covering L); monotone non-increasing.
        let occ_at = |len: usize| -> (i64, i64) {
            for s in &steps {
                if s.match_len as usize >= len {
                    return (s.occ_count as i64, s.sa_start as i64);
                }
            }
            (0, 0) // L > maximal match: interval emptied
        };

        let lstart = min_seed_len_plus1.max(2);
        for len in lstart..=qlen {
            let (s, sa) = occ_at(len);
            if s < max_mem_intv {
                // FMI break point (cpp:509-523).
                if s > 0 {
                    out.push(Smem {
                        rid,
                        m: pivot as u32,
                        n: (pivot + len - 1) as u32,
                        k: sa,
                        l: 0,
                        s,
                    });
                }
                return pivot + len;
            }
        }
        // Window boundary reached without the condition firing (cpp:528).
        if boundary_is_n {
            pivot + qlen + 1
        } else {
            pivot + qlen
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::spectrum::MemMatch;
    use crate::train::config::{MemoryMode, TrainerConfig};
    use crate::train::{build_sidecar_from_pac_with_config, mask::MaskConfig};
    use proptest::prelude::*;
    use std::io::Write;
    use std::mem::{align_of, offset_of, size_of};

    thread_local! {
        /// Per-thread override of [`super::isa_reseed_enabled`] for the byte-id
        /// proptest (`None` = use the env gate). See `isa_force_set` / `isa_force_get`.
        static ISA_FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    /// Read the test-only ISA-enable override (called from `isa_reseed_enabled`).
    pub(super) fn isa_force_get() -> Option<bool> {
        ISA_FORCE.with(|c| c.get())
    }
    /// Force the ISA reseed path on/off for the current thread (test-only).
    fn isa_force_set(v: Option<bool>) {
        ISA_FORCE.with(|c| c.set(v));
    }

    fn write_pac(path: &std::path::Path, bases: &[u8]) {
        let l = bases.len();
        let mut buf = vec![0u8; l / 4 + 1];
        for (i, &b) in bases.iter().enumerate() {
            buf[i >> 2] |= b << ((3 - (i & 3)) * 2);
        }
        buf.push((l % 4) as u8);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    /// Build a mode-2 sidecar from forward bases `fwd`, returning the opened index
    /// and the tempdir (kept alive for the mmap lifetime). At runtime, pass `&fwd`
    /// as the `pac` arg with [`PacEncoding::Unpacked`] (mirrors the spectrum.rs tests).
    fn build_mode2(fwd: &[u8]) -> (tempfile::TempDir, LearnedIndex) {
        let dir = tempfile::tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, fwd);
        let prefix = dir.path().join("r.prmi");
        let cfg = TrainerConfig::default().with_memory_mode(MemoryMode::Mode2);
        build_sidecar_from_pac_with_config(
            &pac,
            &prefix,
            None,
            MaskConfig::default(),
            1,
            Some(cfg),
        )
        .unwrap();
        let idx = LearnedIndex::open(&prefix).unwrap();
        (dir, idx)
    }

    /// Like [`build_mode2`] but emits the `.isa` inverse-suffix-array sidecar, so the
    /// ISA reseed fast-path is available (`has_isa() == true`).
    fn build_mode2_with_isa(fwd: &[u8]) -> (tempfile::TempDir, LearnedIndex) {
        let dir = tempfile::tempdir().unwrap();
        let pac = dir.path().join("r.pac");
        write_pac(&pac, fwd);
        let prefix = dir.path().join("r.prmi");
        let cfg = TrainerConfig::default()
            .with_memory_mode(MemoryMode::Mode2)
            .with_isa(true);
        build_sidecar_from_pac_with_config(
            &pac,
            &prefix,
            None,
            MaskConfig::default(),
            1,
            Some(cfg),
        )
        .unwrap();
        let idx = LearnedIndex::open(&prefix).unwrap();
        assert!(idx.has_isa(), "build_mode2_with_isa produced no .isa");
        (dir, idx)
    }

    /// Independent ground-truth for [`LearnedIndex::zz_left_span`]: the largest
    /// left window `read[pivot-L+1..=pivot]` that occurs in the reference, found
    /// via FORWARD `mem_search` per window (a different code path than
    /// `zz_left_span`'s RC backward extension). Left-window occurrence is monotone
    /// in `L` (a shorter window is a suffix of a longer occurring one), so the
    /// first non-occurrence breaks. Stops at an N to the left or index 0.
    fn left_span_oracle(idx: &LearnedIndex, read: &[u8], pivot: usize, fwd: &[u8]) -> usize {
        let enc = PacEncoding::Unpacked;
        // Leftmost reachable index without crossing an N.
        let mut lo = pivot;
        while lo > 0 && read[lo - 1] < 4 {
            lo -= 1;
        }
        let max_window = pivot - lo + 1;
        let mut best = 1;
        for span in 1..=max_window {
            let start = pivot + 1 - span;
            let m = idx.mem_search(&read[start..=pivot], fwd, enc);
            if m.match_len as usize >= span && m.occ > 0 {
                best = span;
            } else {
                break;
            }
        }
        best
    }

    /// N-clamped forward window length from `p` (test mirror of `fwd_qlen`).
    fn fwd_qlen(read: &[u8], p: usize) -> usize {
        let rlen = read.len();
        for (i, &b) in read.iter().enumerate().skip(p) {
            if b >= 4 {
                return i - p;
            }
        }
        rlen - p
    }

    /// `true` iff `win` (no N) occurs in the 2× reference — via a full `mem_search`
    /// match (`match_len == win.len() && occ > 0`). Used by the definitional MEM
    /// oracle below.
    fn occurs(idx: &LearnedIndex, win: &[u8], fwd: &[u8]) -> bool {
        if win.iter().any(|&b| b >= 4) {
            return false;
        }
        let m = idx.mem_search(win, fwd, PacEncoding::Unpacked);
        m.match_len as usize == win.len() && m.occ > 0
    }

    /// Definitional set of maximal exact matches (MEMs) of `read` vs the reference,
    /// length `>= min_seed_len`, no N. A window `read[m..=n]` is a MEM iff it occurs
    /// and is both left-maximal (`m==0` || `read[m-1]` is N || `read[m-1..=n]` does
    /// not occur) and right-maximal (symmetric). For `min_intv == 1` this is exactly
    /// the pass-1 zigzag SMEM set — an oracle independent of the zigzag control flow.
    fn mem_set_oracle(
        idx: &LearnedIndex,
        read: &[u8],
        fwd: &[u8],
        min_seed_len: u32,
    ) -> std::collections::BTreeSet<(u32, u32)> {
        let rlen = read.len();
        let msl = min_seed_len as usize;
        let mut set = std::collections::BTreeSet::new();
        for m in 0..rlen {
            for n in m..rlen {
                if n - m + 1 < msl {
                    continue;
                }
                if !occurs(idx, &read[m..=n], fwd) {
                    continue;
                }
                let left_max = m == 0 || read[m - 1] >= 4 || !occurs(idx, &read[m - 1..=n], fwd);
                let right_max =
                    n + 1 == rlen || read[n + 1] >= 4 || !occurs(idx, &read[m..=n + 1], fwd);
                if left_max && right_max {
                    set.insert((m as u32, n as u32));
                }
            }
        }
        set
    }

    /// Drive the pass-1 zigzag (`min_intv = 1`) to completion over `read`,
    /// returning the emitted SMEMs.
    fn pass1_walk(idx: &LearnedIndex, read: &[u8], fwd: &[u8], min_seed_len: u32) -> Vec<Smem> {
        let enc = PacEncoding::Unpacked;
        let mut out = Vec::new();
        let mut pivot = 0;
        while pivot < read.len() {
            idx.zz_step1(read, 0, &mut pivot, 1, min_seed_len, &mut out, fwd, enc);
        }
        out
    }

    /// Drive the standalone pass-3 walk (model-seeded) over `read`.
    fn pass3_walk(
        idx: &LearnedIndex,
        read: &[u8],
        fwd: &[u8],
        max_mem_intv: i64,
        min_seed_len: u32,
    ) -> Vec<Smem> {
        let enc = PacEncoding::Unpacked;
        let msl1 = min_seed_len as usize + 1;
        let mut out = Vec::new();
        let mut x = 0;
        while x < read.len() {
            x = idx.pass3_seed_one_pivot(read, 0, x, max_mem_intv, msl1, &mut out, fwd, enc);
        }
        out
    }

    /// Drive pass 3 via the reference `forward_spectrum` path (the byte-identity
    /// oracle for the model-seeded fast path).
    fn pass3_walk_spectrum(
        idx: &LearnedIndex,
        read: &[u8],
        fwd: &[u8],
        max_mem_intv: i64,
        min_seed_len: u32,
    ) -> Vec<Smem> {
        let enc = PacEncoding::Unpacked;
        let msl1 = min_seed_len as usize + 1;
        let mut out = Vec::new();
        let mut x = 0;
        while x < read.len() {
            x = idx.pass3_seed_one_pivot_spectrum(
                read,
                0,
                x,
                max_mem_intv,
                msl1,
                &mut out,
                fwd,
                enc,
            );
        }
        out
    }

    /// Independent pass-3 oracle: same walk structure but `occ_at(L)` via a fresh
    /// FORWARD `mem_search` (a different code path than pass-3's `forward_spectrum`).
    /// Returns `(m, n, k, s)` tuples in emission order.
    fn pass3_oracle(
        idx: &LearnedIndex,
        read: &[u8],
        fwd: &[u8],
        max_mem_intv: i64,
        min_seed_len: u32,
    ) -> Vec<(u32, u32, i64, i64)> {
        let enc = PacEncoding::Unpacked;
        let rlen = read.len();
        let msl1 = min_seed_len as usize + 1;
        let occ_at = |pivot: usize, len: usize| -> (i64, i64) {
            let m = idx.mem_search(&read[pivot..pivot + len], fwd, enc);
            if m.match_len as usize >= len {
                (m.occ as i64, m.sa_start as i64)
            } else {
                (0, 0)
            }
        };
        let mut out = Vec::new();
        let mut x = 0;
        while x < rlen {
            let mut next_x = x + 1;
            if read[x] < 4 {
                let qlen = fwd_qlen(read, x);
                let boundary_is_n = x + qlen < rlen;
                let lstart = msl1.max(2);
                let mut broke = false;
                for len in lstart..=qlen {
                    let (s, sa) = occ_at(x, len);
                    if s < max_mem_intv {
                        next_x = x + len;
                        if s > 0 {
                            out.push((x as u32, (x + len - 1) as u32, sa, s));
                        }
                        broke = true;
                        break;
                    }
                }
                if !broke {
                    next_x = if boundary_is_n {
                        x + qlen + 1
                    } else {
                        x + qlen
                    };
                }
            }
            x = next_x;
        }
        out
    }

    /// `Smem` matches the C `prmi_smem_t` layout: `u32 rid,m,n` then (after the
    /// 4-byte pad the `i64` triple's 8-byte alignment forces) `i64 k,l,s`. This
    /// is the layout the consumer `memcpy`s into its `SMEM[]`; a drift here
    /// silently corrupts the consumer's seed array.
    #[test]
    fn smem_layout_matches_c_abi() {
        assert_eq!(size_of::<Smem>(), 40, "sizeof(Smem)");
        assert_eq!(align_of::<Smem>(), 8, "alignof(Smem)");
        assert_eq!(offset_of!(Smem, rid), 0);
        assert_eq!(offset_of!(Smem, m), 4);
        assert_eq!(offset_of!(Smem, n), 8);
        // 4-byte pad at offset 12 here.
        assert_eq!(
            offset_of!(Smem, k),
            16,
            "k must follow the u32 triple's pad"
        );
        assert_eq!(offset_of!(Smem, l), 24);
        assert_eq!(offset_of!(Smem, s), 32);
    }

    /// `Smem` is `Copy` (used in stack buffers / direct `out[]` fills).
    #[test]
    fn smem_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Smem>();
        assert_copy::<CollectOpts>();
    }

    proptest! {
        // A sidecar build is costly; keep case count modest, sweep all pivots per case.
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// `zz_left_span` == the independent forward-window oracle, over a real
        /// mode-2 sidecar, for every non-N pivot of random reads. Repetitive small
        /// refs naturally exercise occ=1 (unique) and occ>1 (ambiguous) anchors.
        #[test]
        fn zz_left_span_equals_oracle(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..60),
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            for pivot in 0..read.len() {
                let got = idx.zz_left_span(&read, pivot, &fwd, enc);
                let want = left_span_oracle(&idx, &read, pivot, &fwd);
                prop_assert_eq!(
                    got, want,
                    "zz_left_span({:?}, pivot={}) = {} but oracle = {}",
                    read, pivot, got, want
                );
            }
        }

        /// `zz_right_emit`: returns the maximal-match length always, emits exactly
        /// when `match_len >= min_seed_len && occ >= min_intv`, stamps the SMEM
        /// fields correctly, and the emitted `(k,s)` re-recovers from `read[m..=n]`
        /// (the design §8.2 cross-check — an independent confirmation the emitted
        /// interval is a real maximal match).
        #[test]
        fn zz_right_emit_gate_and_fields(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..60),
            min_seed_len in 1u32..20,
            min_intv in 1i64..4,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            for pivot in 0..read.len() {
                let qlen = fwd_qlen(&read, pivot);
                let mut out: Vec<Smem> = Vec::new();
                let ret_ml =
                    idx.zz_right_emit(&read, 7, pivot, qlen, min_intv, min_seed_len, &mut out, &fwd, enc);

                // Reference: a direct mem_search + gate (independent of zz_right_emit's branch).
                let m = if qlen == 0 {
                    MemMatch { match_len: 0, sa_start: 0, occ: 0 }
                } else {
                    idx.mem_search(&read[pivot..pivot + qlen], &fwd, enc)
                };
                prop_assert_eq!(ret_ml as u64, m.match_len, "match_len mismatch at pivot={}", pivot);

                let want_emit = qlen > 0 && m.match_len >= min_seed_len as u64 && m.occ as i64 >= min_intv;
                prop_assert_eq!(out.len(), want_emit as usize, "emit decision at pivot={}", pivot);

                if want_emit {
                    let s = out[0];
                    prop_assert_eq!(s.rid, 7);
                    prop_assert_eq!(s.m, pivot as u32);
                    prop_assert_eq!(s.n, (pivot + m.match_len as usize - 1) as u32);
                    prop_assert_eq!(s.k, m.sa_start as i64);
                    prop_assert_eq!(s.l, 0);
                    prop_assert_eq!(s.s, m.occ as i64);
                    // §8.2: re-recover (k,s) from the emitted span.
                    let recov = idx.mem_search(&read[s.m as usize..=s.n as usize], &fwd, enc);
                    prop_assert_eq!(recov.sa_start as i64, s.k, "k re-recovery at pivot={}", pivot);
                    prop_assert_eq!(recov.occ as i64, s.s, "s re-recovery at pivot={}", pivot);
                }
            }
        }

        /// Pass-1 zigzag (`min_intv=1`) emits exactly the definitional MEM set (the
        /// maximal exact matches of length `>= min_seed_len`), as a SET. This is
        /// independent of the zigzag control flow (the oracle enumerates windows and
        /// tests left/right maximality directly). Every emitted SMEM must also be a
        /// valid exact-match interval (the `(k,s)` re-recovers).
        #[test]
        fn zz_step1_pass1_equals_mem_set(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=4, 1..50), // 4 = N, exercises N boundaries
            min_seed_len in 1u32..12,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let emitted = pass1_walk(&idx, &read, &fwd, min_seed_len);

            // Set of emitted (m,n) == definitional MEM set.
            let got: std::collections::BTreeSet<(u32, u32)> =
                emitted.iter().map(|s| (s.m, s.n)).collect();
            let want = mem_set_oracle(&idx, &read, &fwd, min_seed_len);
            prop_assert_eq!(&got, &want, "pass-1 SMEM set != MEM set for read {:?}", read);

            // Every emitted SMEM is a real exact-match interval with span >= min_seed_len.
            for s in &emitted {
                prop_assert!(s.n >= s.m);
                prop_assert!(s.n - s.m + 1 >= min_seed_len);
                prop_assert_eq!(s.l, 0);
                let recov = idx.mem_search(&read[s.m as usize..=s.n as usize], &fwd, enc);
                prop_assert_eq!(recov.match_len as usize, (s.n - s.m + 1) as usize);
                prop_assert_eq!(recov.sa_start as i64, s.k);
                prop_assert_eq!(recov.occ as i64, s.s);
            }
        }

        /// The pre-sort driver (passes 1+2, `max_mem_intv=0`): pass-1 prefix equals
        /// the validated pass-1 walk; the pass-2 tail equals an independent re-drive
        /// of the reseed selection (filter transcribed in-test) + `zz_step1_reseed`;
        /// and every pass-2 SMEM is a valid exact-match interval with `occ >= 2`
        /// (reseed `min_intv = parent.s + 1 >= 2`).
        #[test]
        fn driver_pass1_pass2_pre_sort(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..50),
            min_seed_len in 2u32..8,
            split_len in 2u32..12,
            split_width in 1i64..6,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len, split_width, max_mem_intv: 0 };

            let unsorted = idx.collect_smems_unsorted(&read, 0, &opts, &fwd, enc);

            // Pass-1 prefix == the validated pass-1 walk.
            let pass1 = pass1_walk(&idx, &read, &fwd, min_seed_len);
            let num1 = pass1.len();
            prop_assert_eq!(&unsorted[..num1], pass1.as_slice(), "pass-1 prefix mismatch");

            // Independent reseed selection + re-drive == the pass-2 tail.
            let mut expected_pass2: Vec<Smem> = Vec::new();
            for p in &pass1 {
                let span = p.n + 1 - p.m;
                if span < split_len || p.s > split_width {
                    continue;
                }
                let mid = ((p.m + p.n + 1) >> 1) as usize;
                idx.zz_step1_reseed(&read, 0, mid, p.s + 1, min_seed_len, &mut expected_pass2, None, &fwd, enc);
            }
            prop_assert_eq!(&unsorted[num1..], expected_pass2.as_slice(), "pass-2 tail mismatch");

            // Every pass-2 SMEM is a real exact-match interval, span >= msl, occ >= 2.
            for s in &unsorted[num1..] {
                prop_assert!(s.n - s.m + 1 >= min_seed_len);
                prop_assert!(s.s >= 2, "reseed SMEM occ must be >= 2");
                prop_assert_eq!(s.l, 0);
                let recov = idx.mem_search(&read[s.m as usize..=s.n as usize], &fwd, enc);
                prop_assert_eq!(recov.match_len as usize, (s.n - s.m + 1) as usize);
                prop_assert_eq!(recov.sa_start as i64, s.k);
                prop_assert_eq!(recov.occ as i64, s.s);
            }
        }

        /// `collect_smems` output is the within-read two-stage sort of the unsorted
        /// set: a permutation of it, non-decreasing in `(m, n)`, and — when all
        /// `(m, n)` are distinct — positionally equal to the `(m ASC, n ASC)` order.
        /// Also deterministic across runs. (`max_mem_intv == 0`: passes 1+2.)
        #[test]
        fn collect_smems_sorted_within_read(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..50),
            min_seed_len in 2u32..8,
            split_len in 2u32..12,
            split_width in 1i64..6,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len, split_width, max_mem_intv: 0 };

            let unsorted = idx.collect_smems_unsorted(&read, 0, &opts, &fwd, enc);
            let cap = unsorted.len() + 8;
            let mut buf = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; cap];
            let n = idx.collect_smems(&read, 0, &opts, &fwd, enc, &mut buf).unwrap();
            let sorted = &buf[..n];

            // 1. Same multiset as the unsorted set.
            let mut a: Vec<Smem> = unsorted.clone();
            let mut b: Vec<Smem> = sorted.to_vec();
            let key = |s: &Smem| (s.m, s.n, s.k, s.s);
            a.sort_by_key(key);
            b.sort_by_key(key);
            prop_assert_eq!(&a, &b, "sorted output is not a permutation of the unsorted set");

            // 2. Non-decreasing (m, n).
            for w in sorted.windows(2) {
                prop_assert!((w[0].m, w[0].n) <= (w[1].m, w[1].n), "not (m,n)-sorted");
            }

            // 3. Distinct (m,n) => positionally equal to (m ASC, n ASC).
            let distinct = {
                let mut mn: Vec<(u32, u32)> = unsorted.iter().map(|s| (s.m, s.n)).collect();
                mn.sort_unstable();
                let len = mn.len();
                mn.dedup();
                mn.len() == len
            };
            if distinct {
                let mut expect = unsorted.clone();
                expect.sort_unstable_by(|x, y| x.m.cmp(&y.m).then(x.n.cmp(&y.n)));
                prop_assert_eq!(sorted, expect.as_slice(), "distinct (m,n) order mismatch");
            }

            // 4. Determinism.
            let mut buf2 = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; cap];
            let n2 = idx.collect_smems(&read, 0, &opts, &fwd, enc, &mut buf2).unwrap();
            prop_assert_eq!(&buf[..n], &buf2[..n2], "non-deterministic output");
        }

        /// Overflow contract: a too-small `out` returns `Err(needed)` with the exact
        /// count; a correctly-sized `out` succeeds.
        #[test]
        fn collect_smems_overflow(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..50),
            min_seed_len in 2u32..8,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len: 3, split_width: 4, max_mem_intv: 0 };
            let needed = idx.collect_smems_unsorted(&read, 0, &opts, &fwd, enc).len();
            // Too small (when there is at least one SMEM): Err(needed).
            if needed > 0 {
                let mut tiny = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; needed - 1];
                prop_assert_eq!(idx.collect_smems(&read, 0, &opts, &fwd, enc, &mut tiny), Err(needed));
            }
            // Exactly sized: Ok(needed).
            let mut ok = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; needed];
            prop_assert_eq!(idx.collect_smems(&read, 0, &opts, &fwd, enc, &mut ok), Ok(needed));
        }

        /// Pass-3 walk == the independent `mem_search`-based oracle (positional), and
        /// the model-seeded fast path == the reference `forward_spectrum` path; every
        /// pass-3 SMEM is a valid exact-match interval with `0 < occ < max_mem_intv`
        /// and span `> min_seed_len`.
        #[test]
        fn pass3_equals_oracle(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..50),
            min_seed_len in 1u32..8,
            max_mem_intv in 1i64..8,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let got = pass3_walk(&idx, &read, &fwd, max_mem_intv, min_seed_len);
            let want = pass3_oracle(&idx, &read, &fwd, max_mem_intv, min_seed_len);

            let got_tuples: Vec<(u32, u32, i64, i64)> =
                got.iter().map(|s| (s.m, s.n, s.k, s.s)).collect();
            prop_assert_eq!(&got_tuples, &want, "pass-3 walk != oracle for read {:?}", read);

            // Model-seeded fast path == the reference forward_spectrum path, positionally.
            let want_spectrum = pass3_walk_spectrum(&idx, &read, &fwd, max_mem_intv, min_seed_len);
            prop_assert_eq!(&got, &want_spectrum, "model-seeded pass3 != spectrum pass3");

            for s in &got {
                prop_assert!(s.n - s.m + 1 > min_seed_len);
                prop_assert!(s.s > 0 && s.s < max_mem_intv, "pass-3 occ must be 0 < s < max_mem_intv");
                prop_assert_eq!(s.l, 0);
                let recov = idx.mem_search(&read[s.m as usize..=s.n as usize], &fwd, enc);
                prop_assert_eq!(recov.match_len as usize, (s.n - s.m + 1) as usize);
                prop_assert_eq!(recov.sa_start as i64, s.k);
                prop_assert_eq!(recov.occ as i64, s.s);
            }
        }

        /// `max_mem_intv == 0` disables pass 3: the driver output equals passes 1+2
        /// only (no extra SMEMs appended). `split_width == 0` selects no reseeds, so
        /// this isolates the pass-3 gate against a pass-1-only baseline.
        #[test]
        fn pass3_disabled_when_zero(
            fwd in prop::collection::vec(0u8..=3, 40..200),
            read in prop::collection::vec(0u8..=3, 1..50),
            min_seed_len in 2u32..8,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts0 = CollectOpts { min_seed_len, split_len: 1000, split_width: 0, max_mem_intv: 0 };
            let out = idx.collect_smems_unsorted(&read, 0, &opts0, &fwd, enc);
            let pass1 = pass1_walk(&idx, &read, &fwd, min_seed_len);
            prop_assert_eq!(out, pass1);
        }
    }

    /// End-to-end: `collect_smems` equals the composition of the independently
    /// oracle-validated walks — pass-1 (== MEM set), pass-2 (reseed selection
    /// transcribed in-test + `zz_step1_reseed`), and pass-3 (gated on
    /// `max_mem_intv > 0`, via the `pass3_walk`) — concatenated in that order and
    /// run through the two-stage sort. Sweeps the reseed knobs and `max_mem_intv`.
    /// This confirms the driver interleaves the passes in the right order with the
    /// right args; the components themselves are validated by the tests above.
    /// Positional (order included). The `(m,n)`-tie order vs the C++ unstable
    /// introsort remains a KNOWN OPEN deferred to the consumer box-gate.
    fn full_pipeline_reference(
        idx: &LearnedIndex,
        read: &[u8],
        fwd: &[u8],
        opts: &CollectOpts,
    ) -> Vec<Smem> {
        let enc = PacEncoding::Unpacked;
        let mut expected = pass1_walk(idx, read, fwd, opts.min_seed_len);
        // Pass 2: reseed selection (transcribed) + zz_step1_reseed.
        let p1 = expected.clone();
        for p in &p1 {
            let span = p.n + 1 - p.m;
            if span < opts.split_len || p.s > opts.split_width {
                continue;
            }
            let mid = ((p.m + p.n + 1) >> 1) as usize;
            idx.zz_step1_reseed(
                read,
                0,
                mid,
                p.s + 1,
                opts.min_seed_len,
                &mut expected,
                None,
                fwd,
                enc,
            );
        }
        // Pass 3 (gated): the model-seeded walk, appended before the sort.
        if opts.max_mem_intv > 0 {
            let mut p3 = pass3_walk(idx, read, fwd, opts.max_mem_intv, opts.min_seed_len);
            expected.append(&mut p3);
        }
        LearnedIndex::sort_within_read(&mut expected);
        expected
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// `collect_smems` == the full-pipeline reference, across the whole opts
        /// sweep (including pass 3), positionally.
        #[test]
        fn collect_smems_full_pipeline(
            fwd in prop::collection::vec(0u8..=3, 40..220),
            read in prop::collection::vec(0u8..=4, 1..60),
            min_seed_len in 1u32..10,
            split_len in 2u32..14,
            split_width in 1i64..8,
            max_mem_intv in 0i64..6,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len, split_width, max_mem_intv };

            let expected = full_pipeline_reference(&idx, &read, &fwd, &opts);
            let mut buf = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; expected.len() + 8];
            let n = idx.collect_smems(&read, 0, &opts, &fwd, enc, &mut buf).unwrap();
            prop_assert_eq!(&buf[..n], expected.as_slice(), "collect_smems != composed reference");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Scratch-reuse byte-identity: `collect_smems_into` with a REUSED scratch
        /// (already dirtied by a longer prior read) produces output identical to the
        /// fresh-allocating `collect_smems`, for every read in a batch and across the
        /// full opts sweep. Running a long read first then shorter ones is what would
        /// expose a stale-tail bug if the `_into` path forgot to clear the buffers.
        #[test]
        fn collect_smems_into_equals_collect_smems(
            fwd in prop::collection::vec(0u8..=3, 40..220),
            reads in prop::collection::vec(prop::collection::vec(0u8..=4, 1..60), 1..6),
            min_seed_len in 1u32..10,
            split_len in 2u32..14,
            split_width in 1i64..8,
            max_mem_intv in 0i64..6,
        ) {
            let (_dir, idx) = build_mode2(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len, split_width, max_mem_intv };

            // One scratch reused across the whole batch (the production usage).
            let mut scratch = CollectScratch::new();
            for (rid, read) in reads.iter().enumerate() {
                let cap = 2 * read.len() + 8;
                let zero = Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 };
                let mut buf_fresh = vec![zero; cap];
                let mut buf_reuse = vec![zero; cap];
                let n_fresh = idx
                    .collect_smems(read, rid as u32, &opts, &fwd, enc, &mut buf_fresh)
                    .unwrap();
                let n_reuse = idx
                    .collect_smems_into(read, rid as u32, &opts, &fwd, enc, &mut buf_reuse, &mut scratch)
                    .unwrap();
                prop_assert_eq!(n_fresh, n_reuse, "scratch reuse changed SMEM count (read {})", rid);
                prop_assert_eq!(
                    &buf_fresh[..n_fresh],
                    &buf_reuse[..n_reuse],
                    "scratch reuse != fresh alloc (read {})",
                    rid
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// ISA byte-identity: `collect_smems` with the live ISA reseed warm-start
        /// ENABLED == with it disabled (the cold model-launch path), positionally,
        /// across the full opts sweep. This is the automated gate the `seed_bench`
        /// `PRMI_ISA` diff checks manually. Reads are REFERENCE SUBSTRINGS so they
        /// produce long pass-1 SMEMs whose reseeds are hinted (every reseeded SMEM is
        /// hinted, so the warm-start path is actually exercised here).
        #[test]
        fn collect_smems_isa_equals_cold(
            fwd in prop::collection::vec(0u8..=3, 60..220),
            cuts in prop::collection::vec((0usize..220, 8usize..80), 1..8),
            min_seed_len in 1u32..10,
            split_len in 2u32..14,
            split_width in 1i64..8,
            max_mem_intv in 0i64..6,
        ) {
            let (_dir, idx) = build_mode2_with_isa(&fwd);
            let enc = PacEncoding::Unpacked;
            let opts = CollectOpts { min_seed_len, split_len, split_width, max_mem_intv };

            // Reads = substrings of the reference (guaranteed full matches).
            let reads: Vec<Vec<u8>> = cuts
                .iter()
                .filter_map(|&(s, l)| {
                    let start = s % fwd.len();
                    let end = (start + l).min(fwd.len());
                    (end > start).then(|| fwd[start..end].to_vec())
                })
                .collect();

            for (i, r) in reads.iter().enumerate() {
                let mut buf_cold = vec![Smem { rid: 0, m: 0, n: 0, k: 0, l: 0, s: 0 }; 2 * r.len() + 8];
                let mut buf_isa = buf_cold.clone();

                isa_force_set(Some(false));
                let nc = idx.collect_smems(r, i as u32, &opts, &fwd, enc, &mut buf_cold).unwrap();
                isa_force_set(Some(true));
                let ni = idx.collect_smems(r, i as u32, &opts, &fwd, enc, &mut buf_isa).unwrap();
                isa_force_set(None);

                prop_assert_eq!(nc, ni, "ISA path changed SMEM count for read {}", i);
                prop_assert_eq!(&buf_cold[..nc], &buf_isa[..ni], "ISA path != cold for read {}", i);
            }
        }
    }

    /// Partial-SMEM byte-id: a reference substring with a flipped last base is NOT a
    /// full-read match, so its long pass-1 SMEM is a PARTIAL SMEM (span < rlen). That
    /// SMEM is still hinted and its reseed WARM-STARTS from a hint that is NOT a
    /// maximal occurrence (the read diverges past the cached span). Byte-identity must
    /// hold — this is the case the seed-independence of the warm start exists for.
    #[test]
    fn collect_smems_isa_warmstart_equals_cold_partial() {
        let fwd: Vec<u8> = (0..400).map(|i| ((i * 7 + 3) % 4) as u8).collect();
        let (_d, idx) = build_mode2_with_isa(&fwd);
        let enc = PacEncoding::Unpacked;
        let opts = CollectOpts {
            min_seed_len: 5,
            split_len: 6,
            split_width: 7,
            max_mem_intv: 0,
        };
        let mut r = fwd[20..180].to_vec();
        let last = r.len() - 1;
        r[last] = (r[last] + 1) % 4; // flip so the read does not fully match anywhere
        let mut a = vec![
            Smem {
                rid: 0,
                m: 0,
                n: 0,
                k: 0,
                l: 0,
                s: 0
            };
            2 * r.len() + 8
        ];
        let mut b = a.clone();
        isa_force_set(Some(false));
        let na = idx.collect_smems(&r, 0, &opts, &fwd, enc, &mut a).unwrap();
        isa_force_set(Some(true));
        let nb = idx.collect_smems(&r, 0, &opts, &fwd, enc, &mut b).unwrap();
        isa_force_set(None);
        assert_eq!(
            na, nb,
            "ISA warm-start changed SMEM count on a partial-match read"
        );
        assert_eq!(
            &a[..na],
            &b[..nb],
            "ISA warm-start != cold on a partial-match read"
        );
    }

    /// Left/RC byte-id: a partial-match read (reference substring with an INTERIOR
    /// flip) forces reseed steps with left context, so the reseed-left RC warm-start
    /// fires from a stale/non-maximal RC hint. ISA-on (left+fwd warm-start) must equal
    /// ISA-off (cold). Guards the RC wiring's byte-identity.
    #[test]
    fn collect_smems_isa_left_warmstart_equals_cold_partial() {
        // Low-occurrence reference (same generator as `reseed_rc_hint_reduces_probes`):
        // a 4-periodic `(i * a + b) % 4` reference repeats every exact block ~len/4
        // times, so the long blocks around the interior flip exceed `split_width` and
        // get filtered out of pass-2 — the test would then pass without ever reaching
        // the RC left-warmstart path. The hashed generator keeps occurrences low so a
        // reseed candidate survives the filter (asserted below under the probe-count
        // feature).
        let fwd: Vec<u8> = (0..400)
            .map(|i| (((i as u64 * 2654435761) >> 11) % 4) as u8)
            .collect();
        let (_d, idx) = build_mode2_with_isa(&fwd);
        let enc = PacEncoding::Unpacked;
        let opts = CollectOpts {
            min_seed_len: 5,
            split_len: 6,
            split_width: 7,
            max_mem_intv: 0,
        };
        let mut r = fwd[30..210].to_vec(); // long internal substring → reseeds with left context
        let mid = r.len() / 2;
        r[mid] = (r[mid] + 2) % 4; // interior flip → partial; reseed fires both directions
        let mut a = vec![
            Smem {
                rid: 0,
                m: 0,
                n: 0,
                k: 0,
                l: 0,
                s: 0
            };
            2 * r.len() + 8
        ];
        let mut b = a.clone();
        isa_force_set(Some(false));
        let na = idx.collect_smems(&r, 0, &opts, &fwd, enc, &mut a).unwrap();
        isa_force_set(Some(true));
        let nb = idx.collect_smems(&r, 0, &opts, &fwd, enc, &mut b).unwrap();
        isa_force_set(None);
        assert_eq!(na, nb, "left/RC warm-start changed SMEM count");
        assert_eq!(&a[..na], &b[..nb], "left/RC warm-start != cold");

        // Coverage guard: prove the fixture actually reaches the RC left-reseed path
        // (pass-2). `rs_left` (bucket 2) counts probes issued inside
        // `zz_left_span_reseed`; a non-periodic reference must leave a reseed candidate
        // under `split_width`, so this byte-id test cannot silently pass without
        // exercising the new warm-start.
        #[cfg(feature = "spectrum-probe-count")]
        {
            attrib::reset();
            isa_force_set(Some(true));
            let mut c = a.clone();
            let _ = idx.collect_smems(&r, 0, &opts, &fwd, enc, &mut c).unwrap();
            isa_force_set(None);
            assert!(
                attrib::snapshot()[2] > 0,
                "fixture never reached the RC left-reseed path (rs_left=0)"
            );
        }
    }

    /// PROJECTION-CORRECTNESS gate (the load-bearing one): the RC reseed-left
    /// warm-start with the CORRECT RC projection must touch STRICTLY FEWER SA probes
    /// than the cold RC search on a deep left extension. A wrong RC formula yields a
    /// seed far from the interval → no probe reduction, failing this — which neither
    /// the byte-id proptest nor the on==off gate can catch (a garbage projection
    /// passes both). `reseed_rc_hint`'s formula is exercised inline here.
    #[cfg(feature = "spectrum-probe-count")]
    #[test]
    fn reseed_rc_hint_reduces_probes() {
        use crate::index::spectrum::probe_count as pc;
        // Pseudo-random reference (low-occ) long enough for a deep left extension.
        let fwd: Vec<u8> = (0..4000)
            .map(|i| (((i * 2654435761u64) >> 11) % 4) as u8)
            .collect();
        let (_d, idx) = build_mode2_with_isa(&fwd);
        let enc = PacEncoding::Unpacked;
        // A reference-substring SMEM [m_start, ...]; a left reseed at a deep interior pivot.
        let m_start = 1000usize;
        let pivot = 1300usize; // 300 bp of left context → deep maximal RC extension
        let refpos = m_start as u64; // first-base ref pos of the full-substring SMEM

        // cold span + probes:
        pc::reset();
        let cold = idx.mem_search_backward_truncated_span_rc(1, 1, &fwd, pivot, 2, &fwd, enc);
        let cold_probes = pc::get();
        // CORRECT RC projection (same formula as `reseed_rc_hint`): base `2*l_pac`.
        let off = (pivot - m_start) as u64;
        let rc = 2 * idx.l_pac() - refpos - off - 1;
        let seed = idx.isa_at(rc);
        pc::reset();
        let warm = idx.mem_search_backward_span_rc_warmstart(1, 1, &fwd, pivot, 2, seed, &fwd, enc);
        let warm_probes = pc::get();
        assert!(seed.is_some(), "RC projection produced no SA index");
        assert_eq!(warm, cold, "RC warm-start changed the span");
        assert!(
            warm_probes < cold_probes,
            "RC projection did not reduce probes ({} vs cold {}) — projection wrong / no-op",
            warm_probes,
            cold_probes
        );
    }

    /// Concurrency: many threads calling `collect_smems(&self, ...)` on a shared
    /// index produce per-read results identical to single-threaded — locking the
    /// "no per-read mutable shared state" design claim (the `&self` is read-only).
    #[test]
    fn collect_smems_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        // A repetitive reference so reseed (pass 2) actually fires.
        let mut fwd = Vec::new();
        for _ in 0..40 {
            fwd.extend_from_slice(&[0, 1, 2, 3, 0, 1]);
        }
        let (_dir, idx) = build_mode2(&fwd);
        let idx = Arc::new(idx);
        let opts = CollectOpts {
            min_seed_len: 4,
            split_len: 6,
            split_width: 8,
            max_mem_intv: 0,
        };

        // A few distinct reads (slices of the reference + a shifted copy).
        let reads: Vec<Vec<u8>> = vec![
            fwd[0..30].to_vec(),
            fwd[5..50].to_vec(),
            fwd[10..60].to_vec(),
            fwd[2..40].to_vec(),
        ];

        // Single-threaded baseline.
        let baseline: Vec<Vec<Smem>> = reads
            .iter()
            .map(|r| {
                let mut out = vec![
                    Smem {
                        rid: 0,
                        m: 0,
                        n: 0,
                        k: 0,
                        l: 0,
                        s: 0
                    };
                    2 * r.len() + 8
                ];
                let n = idx
                    .collect_smems(r, 0, &opts, &fwd, PacEncoding::Unpacked, &mut out)
                    .unwrap();
                out[..n].to_vec()
            })
            .collect();

        // 4 threads each recompute all reads many times; every result must match.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let idx = Arc::clone(&idx);
            let reads = reads.clone();
            let fwd = fwd.clone();
            let baseline = baseline.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    for (i, r) in reads.iter().enumerate() {
                        let mut out = vec![
                            Smem {
                                rid: 0,
                                m: 0,
                                n: 0,
                                k: 0,
                                l: 0,
                                s: 0
                            };
                            2 * r.len() + 8
                        ];
                        let n = idx
                            .collect_smems(r, 0, &opts, &fwd, PacEncoding::Unpacked, &mut out)
                            .unwrap();
                        assert_eq!(&out[..n], baseline[i].as_slice(), "thread result diverged");
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}
