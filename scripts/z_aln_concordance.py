#!/usr/bin/env python3
# Copyright (C) 2026 Fulcrum Genomics LLC
# SPDX-License-Identifier: MIT
"""Design-Z Stage-1 alignment-equivalence diff.

Compares two SAM files produced by running the *same* bwa-mem3 chain+extend on
SMEMs from a fast (tiered) prmi index vs a full whole-genome prmi index. The two
arms differ only in the seed set, so the diff isolates the keep-set effect.

Correctness bar (decided with the operator): the oracle is *alignment-identity*,
not seed-identity. A read's primary alignment is **concordant** iff (RNAME, POS,
CIGAR) match; MAPQ deltas and secondary/supplementary differences are reported
separately, not counted as discordance (MAPQ shifts on multi-mappers are the
expected, benign effect of dropping low-occ supplementary reseeds). The headline
metric is concordance over **present** reads — the only reads Design Z's fast
path actually serves; absent reads route to the whole-genome fallback (= full),
so they are concordant by construction in production.

Usage:
    z_aln_concordance.py --fast zh.sam --full full.sam --present present.on.tsv
    z_aln_concordance.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Primary:
    """The primary alignment record for a read (placement-relevant fields)."""

    rname: str
    pos: int
    cigar: str
    mapq: int
    unmapped: bool


def parse_sam(path: str) -> tuple[dict[str, Primary], dict[str, int]]:
    """Parse a SAM file into {qname: Primary} plus {qname: count of non-primary lines}.

    Primary = the line with neither the secondary (0x100) nor supplementary (0x800)
    flag set. Non-primary lines are tallied so set-differences can be reported.
    """
    primaries: dict[str, Primary] = {}
    secondary_supp: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("@") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            qname, flag, rname, pos, mapq, cigar = f[0], int(f[1]), f[2], int(f[3]), int(f[4]), f[5]
            if flag & 0x1:
                # Paired-end SAMs carry two primaries under one QNAME; keying by
                # qname alone would make the second mate overwrite the first
                # (last-record-wins). This harness is single-end only.
                raise ValueError("paired-end SAM is not supported by z_aln_concordance.py")
            if flag & 0x100 or flag & 0x800:
                secondary_supp[qname] = secondary_supp.get(qname, 0) + 1
                continue
            if qname in primaries:
                # A well-formed single-end SAM has exactly one primary per qname;
                # a second would silently last-row-wins and corrupt concordance.
                raise ValueError(f"duplicate primary alignment for qname: {qname!r}")
            primaries[qname] = Primary(
                rname=rname,
                pos=pos,
                cigar=cigar,
                mapq=mapq,
                unmapped=bool(flag & 0x4),
            )
    return primaries, secondary_supp


def parse_present(path: str) -> dict[str, bool]:
    """Parse the present/absent TSV (header `qname<TAB>present`) into {qname: bool}."""
    present: dict[str, bool] = {}
    with open(path) as handle:
        for i, line in enumerate(handle):
            if i == 0 and line.startswith("qname"):
                continue
            qname, flag = line.rstrip("\n").split("\t")
            # Surface a bad emitter or hand-edited TSV instead of silently coercing:
            # any token other than "0"/"1" would otherwise map to "absent", and a
            # repeated qname would be last-row-wins.
            if flag not in {"0", "1"}:
                raise ValueError(f"invalid present flag for {qname!r}: {flag!r}")
            if qname in present:
                raise ValueError(f"duplicate qname in present TSV: {qname!r}")
            present[qname] = flag == "1"
    return present


@dataclass
class Report:
    """Concordance tallies for one read partition (e.g. present-only)."""

    n: int = 0
    placement_concordant: int = 0  # (rname, pos, cigar) identical
    pos_concordant: int = 0  # RNAME+POS identical (CIGAR may differ)
    both_unmapped: int = 0
    mapped_status_diff: int = 0  # one mapped, one unmapped
    mapq_equal: int = 0  # among placement-concordant
    mapq_delta_hist: dict[int, int] | None = None
    secsupp_diff: int = 0  # differing count of secondary/supplementary lines

    def __post_init__(self) -> None:
        if self.mapq_delta_hist is None:
            self.mapq_delta_hist = {}


def compare(
    fast: dict[str, Primary],
    full: dict[str, Primary],
    fast_ss: dict[str, int],
    full_ss: dict[str, int],
    qnames: list[str],
) -> Report:
    """Compare fast vs full primaries over the given qnames."""
    rep = Report()
    for q in qnames:
        a, b = fast.get(q), full.get(q)
        if a is None or b is None:
            continue  # read missing from one SAM (shouldn't happen with paired runs)
        rep.n += 1
        # Tally the secondary/supplementary-count delta for every read BEFORE the
        # unmapped/status branches return early — otherwise both-unmapped and
        # mapped-status-diff reads are silently omitted from `secsupp_diff` even
        # when their counts differ.
        if fast_ss.get(q, 0) != full_ss.get(q, 0):
            rep.secsupp_diff += 1
        if a.unmapped and b.unmapped:
            rep.both_unmapped += 1
            rep.placement_concordant += 1  # both unmapped == concordant placement
            rep.pos_concordant += 1
            continue
        if a.unmapped != b.unmapped:
            rep.mapped_status_diff += 1
            continue
        # both mapped
        if a.rname == b.rname and a.pos == b.pos:
            rep.pos_concordant += 1
            if a.cigar == b.cigar:
                rep.placement_concordant += 1
                if a.mapq == b.mapq:
                    rep.mapq_equal += 1
                else:
                    d = a.mapq - b.mapq  # fast - full
                    rep.mapq_delta_hist[d] = rep.mapq_delta_hist.get(d, 0) + 1
    return rep


def fmt_report(title: str, rep: Report) -> str:
    """Render a Report as a human-readable block."""
    if rep.n == 0:
        return f"--- {title} ---\n  (no reads)\n"

    def pct(x: int) -> float:
        return 100.0 * x / rep.n

    lines = [
        f"--- {title} (n={rep.n}) ---",
        f"  primary POS+CIGAR concordant : {rep.placement_concordant:>6} ({pct(rep.placement_concordant):.2f}%)",
        f"  primary POS concordant       : {rep.pos_concordant:>6} ({pct(rep.pos_concordant):.2f}%)",
        f"  both unmapped                : {rep.both_unmapped:>6}",
        f"  mapped/unmapped disagreement : {rep.mapped_status_diff:>6}",
        f"  among POS+CIGAR concordant: MAPQ equal={rep.mapq_equal}  MAPQ differs={sum(rep.mapq_delta_hist.values())}",
    ]
    if rep.mapq_delta_hist:
        hist = ", ".join(f"{d:+d}:{c}" for d, c in sorted(rep.mapq_delta_hist.items()))
        lines.append(f"    MAPQ delta (fast-full) histogram: {hist}")
    lines.append(f"  reads with differing secondary/supplementary count: {rep.secsupp_diff}")
    return "\n".join(lines) + "\n"


def run(fast_path: str, full_path: str, present_path: str) -> Report:
    """Main analysis: returns the present-read Report (also prints all partitions)."""
    fast, fast_ss = parse_sam(fast_path)
    full, full_ss = parse_sam(full_path)
    present = parse_present(present_path)

    # `compare()` drops qnames missing from one SAM, so a primary missing from fast
    # or full would shrink `rep.n` and inflate concordance. The two arms run the same
    # reads through the same chain+extend, so their primary cohorts must match 1:1.
    fast_missing = set(full) - set(fast)
    fast_extra = set(fast) - set(full)
    if fast_missing or fast_extra:
        raise ValueError(
            f"fast/full SAM qname mismatch: missing={len(fast_missing)} extra={len(fast_extra)}"
        )

    # The emitter writes exactly one `qname<TAB>present` row per read, so the
    # present TSV must cover the SAM cohort 1:1. A key-set mismatch is contract
    # drift: defaulting a missing qname to "absent" would silently drop reads out
    # of the headline partition and inflate concordance — abort instead.
    missing = set(full) - set(present)
    extra = set(present) - set(full)
    if missing or extra:
        raise ValueError(
            f"present TSV/SAM qname mismatch: missing={len(missing)} extra={len(extra)}"
        )

    present_q = [q for q in full if present.get(q, False)]
    absent_q = [q for q in full if not present.get(q, False)]
    all_q = list(full.keys())

    print(f"fast={fast_path}  full={full_path}")
    print(f"reads: full={len(full)} fast={len(fast)} present={len(present_q)} absent={len(absent_q)}\n")

    present_rep = compare(fast, full, fast_ss, full_ss, present_q)
    print(fmt_report("PRESENT reads (Design-Z served by fast path) — HEADLINE", present_rep))
    print(fmt_report("ABSENT reads (production routes to fallback=full; shown for context)",
                     compare(fast, full, fast_ss, full_ss, absent_q)))
    print(fmt_report("ALL reads", compare(fast, full, fast_ss, full_ss, all_q)))
    return present_rep


# --------------------------------------------------------------------------- #
# Self-test: synthetic SAM strings exercise each concordance branch.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import os
    import tempfile

    full_sam = "\n".join([
        "@HD\tVN:1.5",
        "@SQ\tSN:chr22\tLN:50818468",
        # r1: identical placement, identical MAPQ
        "r1\t0\tchr22\t1000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r2: identical POS+CIGAR, MAPQ differs (60 vs 40) -> benign MAPQ shift
        "r2\t0\tchr22\t2000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r3: same POS, different CIGAR -> POS-concordant but NOT POS+CIGAR
        "r3\t0\tchr22\t3000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r4: different POS -> discordant
        "r4\t0\tchr22\t4000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r5: both unmapped -> concordant; full also carries a supplementary line
        # (fast does not) so secsupp_diff must be counted despite the early continue
        "r5\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t****",
        "r5\t2048\tchr22\t5000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r6: full mapped, fast unmapped -> status disagreement
        "r6\t0\tchr22\t6000\t60\t76M\t*\t0\t0\tACGT\t****",
        # r7: absent read (present=0) -> excluded from headline
        "r7\t0\tchr22\t7000\t60\t76M\t*\t0\t0\tACGT\t****",
    ]) + "\n"
    fast_sam = "\n".join([
        "@HD\tVN:1.5",
        "@SQ\tSN:chr22\tLN:50818468",
        "r1\t0\tchr22\t1000\t60\t76M\t*\t0\t0\tACGT\t****",
        "r2\t0\tchr22\t2000\t40\t76M\t*\t0\t0\tACGT\t****",
        "r3\t0\tchr22\t3000\t60\t40M2I34M\t*\t0\t0\tACGT\t****",
        "r4\t0\tchr22\t4500\t60\t76M\t*\t0\t0\tACGT\t****",
        "r5\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t****",
        "r6\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\t****",
        "r7\t0\tchr22\t7000\t60\t76M\t*\t0\t0\tACGT\t****",
    ]) + "\n"
    present_tsv = "qname\tpresent\n" + "".join(
        f"{q}\t{p}\n" for q, p in
        [("r1", 1), ("r2", 1), ("r3", 1), ("r4", 1), ("r5", 1), ("r6", 1), ("r7", 0)]
    )

    d = tempfile.mkdtemp()
    fp_full, fp_fast, fp_pres = (os.path.join(d, n) for n in ("full.sam", "fast.sam", "p.tsv"))
    for p, c in [(fp_full, full_sam), (fp_fast, fast_sam), (fp_pres, present_tsv)]:
        with open(p, "w") as h:
            h.write(c)

    fast, fast_ss = parse_sam(fp_fast)
    full, full_ss = parse_sam(fp_full)
    present = parse_present(fp_pres)
    present_q = [q for q in full if present.get(q, False)]
    rep = compare(fast, full, fast_ss, full_ss, present_q)

    checks = {
        "present n == 6 (r7 excluded)": rep.n == 6,
        "POS+CIGAR concordant == 3 (r1,r2,r5)": rep.placement_concordant == 3,
        "POS concordant == 4 (r1,r2,r3,r5)": rep.pos_concordant == 4,
        "both_unmapped == 1 (r5)": rep.both_unmapped == 1,
        "mapped_status_diff == 1 (r6)": rep.mapped_status_diff == 1,
        "mapq_equal == 1 (r1 only; r5 unmapped skips MAPQ)": rep.mapq_equal == 1,
        "mapq differs == 1 (r2)": sum(rep.mapq_delta_hist.values()) == 1,
        "mapq delta is -20 (40-60)": rep.mapq_delta_hist.get(-20) == 1,
        # r5 is both-unmapped yet its secondary/supplementary count differs (full=1,
        # fast=0); secsupp_diff must be tallied before the both-unmapped continue.
        "secsupp_diff == 1 (r5, counted before early continue)": rep.secsupp_diff == 1,
    }

    # Paired-end SAMs must be rejected, not silently last-record-wins.
    paired_sam = "r1\t1\tchr22\t1000\t60\t76M\t*\t0\t0\tACGT\t****\n"
    fp_paired = os.path.join(d, "paired.sam")
    with open(fp_paired, "w") as h:
        h.write(paired_sam)
    try:
        parse_sam(fp_paired)
        checks["paired-end SAM raises ValueError"] = False
    except ValueError:
        checks["paired-end SAM raises ValueError"] = True

    # A present TSV that does not cover the SAM cohort 1:1 must abort.
    fp_pres_bad = os.path.join(d, "p_bad.tsv")
    with open(fp_pres_bad, "w") as h:
        h.write("qname\tpresent\nr1\t1\n")  # missing r2..r7
    try:
        run(fp_fast, fp_full, fp_pres_bad)
        checks["present/SAM cohort mismatch raises ValueError"] = False
    except ValueError:
        checks["present/SAM cohort mismatch raises ValueError"] = True

    # An invalid present flag (not 0/1) must be rejected, not coerced to "absent".
    fp_pres_badflag = os.path.join(d, "p_badflag.tsv")
    with open(fp_pres_badflag, "w") as h:
        h.write("qname\tpresent\nr1\t2\n")
    try:
        parse_present(fp_pres_badflag)
        checks["invalid present flag raises ValueError"] = False
    except ValueError:
        checks["invalid present flag raises ValueError"] = True

    # A duplicate qname in the present TSV must fail fast, not last-row-wins.
    fp_pres_dup = os.path.join(d, "p_dup.tsv")
    with open(fp_pres_dup, "w") as h:
        h.write("qname\tpresent\nr1\t1\nr1\t0\n")
    try:
        parse_present(fp_pres_dup)
        checks["duplicate present qname raises ValueError"] = False
    except ValueError:
        checks["duplicate present qname raises ValueError"] = True

    # A duplicate single-end primary in one SAM must fail fast, not last-row-wins.
    dup_primary_sam = (
        "@HD\tVN:1.5\n"
        "r1\t0\tchr22\t1000\t60\t76M\t*\t0\t0\tACGT\t****\n"
        "r1\t0\tchr22\t2000\t60\t76M\t*\t0\t0\tACGT\t****\n"
    )
    fp_dup_primary = os.path.join(d, "dup_primary.sam")
    with open(fp_dup_primary, "w") as h:
        h.write(dup_primary_sam)
    try:
        parse_sam(fp_dup_primary)
        checks["duplicate primary alignment raises ValueError"] = False
    except ValueError:
        checks["duplicate primary alignment raises ValueError"] = True

    # fast/full SAM cohort drift (a primary missing from one arm) must abort.
    fp_fast_short = os.path.join(d, "fast_short.sam")
    with open(fp_fast_short, "w") as h:
        h.write("\n".join(fast_sam.splitlines()[:-1]) + "\n")  # drop r7's primary
    try:
        run(fp_fast_short, fp_full, fp_pres)
        checks["fast/full SAM cohort drift raises ValueError"] = False
    except ValueError:
        checks["fast/full SAM cohort drift raises ValueError"] = True

    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Design-Z Stage-1 alignment-equivalence diff")
    ap.add_argument("--fast", help="SAM from the fast (tiered) prmi index")
    ap.add_argument("--full", help="SAM from the full whole-genome prmi index")
    ap.add_argument("--present", help="present/absent TSV (qname<TAB>present)")
    ap.add_argument("--selftest", action="store_true", help="run the built-in self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not (args.fast and args.full and args.present):
        ap.error("--fast, --full and --present are required (or use --selftest)")
    run(args.fast, args.full, args.present)
    return 0


if __name__ == "__main__":
    sys.exit(main())
