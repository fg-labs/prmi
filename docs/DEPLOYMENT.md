# Deployment notes

## Transparent Huge Pages (THP) — enable on production hosts

prmi's suffix array (`.sa`, ≈83 GB for hg38) and model (`.l2`, ≈6.4 GB) are
random-access, multi-GB mmap'd arrays. Seeding does ~20 random SA reads per call;
with 4 KB pages nearly every probe is a TLB miss + page-table walk (the TLB's
~1.5 K entries cannot cover the working set). prmi already hints 2 MB huge pages
on both mmaps via `madvise(MADV_HUGEPAGE)` (`SaFileReader::open`,
`ModelFileReader::open`; Linux-only, advisory — never affects correctness).

**That hint is a no-op unless the host enables THP.** Check:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# want: "always [madvise] never"  (madvise honors prmi's hint) or "[always] ..."
```

- `madvise` (recommended) — the kernel backs only `MADV_HUGEPAGE`-advised regions
  (prmi's SA + model) with 2 MB pages. Targeted; no system-wide THP overhead.
- `always` — all eligible regions get THP. Also fine for prmi.
- `never` — **THP disabled; prmi silently forfeits the hugepage win.**

To set (root, non-persistent):

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Persist via the kernel cmdline (`transparent_hugepage=madvise`), a tuned profile,
or your init system, per your distro.

### Measured impact (hg38, 2 M real WGS reads, 16 threads, cold real-fused seeding)

| SA pages | reads/sec | dTLB-load-misses | IPC |
|---|---:|---:|---:|
| 4 KB (THP=never)  | 150 886 | 457 M | 2.47 |
| 2 MB (THP=always) | 168 039 | 198 M (−57 %) | 2.71 |

**≈ +11 %** seeding throughput from huge pages alone; a host left on `THP=never`
loses ~7 % versus `madvise` at zero code cost. (A file-backed mmap under
`THP=madvise` already captures most of this via the existing advice; the full
benefit needs the pages actually backed by 2 MB, i.e. an anonymous/`always`
arrangement — but `madvise` is the recommended, lowest-overhead default.)
