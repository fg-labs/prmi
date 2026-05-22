#!/usr/bin/env bash
# Copyright (C) 2026 Fulcrum Genomics LLC
# SPDX-License-Identifier: MIT
#
# End-to-end smoke harness for the cpp_caller example.
#   1. Writes a synthetic 4-kb FASTA.
#   2. Runs `prmi build` to produce the sidecar.
#   3. Writes a binary PAC file (1 byte/base, values 0–3).
#   4. Writes a binary 32-base query file.
#   5. Computes the expected 32-mer hex key.
#   6. Runs cpp_caller and checks the exit code.
#
# Run from anywhere; all paths are relative to this script's directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRMI="${WORKSPACE}/target/release/prmi"
CPP_CALLER="${SCRIPT_DIR}/cpp_caller"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "=== smoke: tmpdir=${TMPDIR}"

# ---- 1. Synthetic FASTA (ACGT × 1024 = 4096 bases) -------------------------
FA="${TMPDIR}/smoke.fa"
python3 - "${FA}" <<'PYEOF'
import sys
path = sys.argv[1]
seq = "ACGT" * 1024     # 4096 bases
with open(path, "w") as f:
    f.write(">smoke\n")
    # wrap at 60 chars
    for i in range(0, len(seq), 60):
        f.write(seq[i:i+60] + "\n")
PYEOF
echo "=== wrote FASTA: ${FA}"

# ---- 2. Build sidecar -------------------------------------------------------
# Use l2_leaf_count=16 (2^4 = pwl4) — appropriate for a 4-kb synthetic reference.
PREFIX="${TMPDIR}/smoke.fa.prmi"
"${PRMI}" build "${FA}" -o "${PREFIX}" --l2-leaf-count 16
echo "=== sidecar built: ${PREFIX}.{meta,sa,l1,l2}"

# ---- 3. Binary PAC file (1 byte per base: A=0 C=1 G=2 T=3) -----------------
PAC="${TMPDIR}/smoke.pac"
python3 - "${PAC}" <<'PYEOF'
import sys
path = sys.argv[1]
# ACGT × 1024
enc = [0, 1, 2, 3] * 1024
with open(path, "wb") as f:
    f.write(bytes(enc))
PYEOF
echo "=== wrote PAC: ${PAC} ($(wc -c < "${PAC}") bytes)"

# ---- 4. Query file: first 32 bases starting at offset 10 -------------------
# offset 10 in ACGT×1024: bases at pos 10,11,...,41 → G T A C G T A C G T ...
#   (10 % 4 = 2 → G, 11%4=3 → T, 12%4=0 → A, 13%4=1 → C, ...)
QUERY="${TMPDIR}/query.bin"
python3 - "${PAC}" "${QUERY}" <<'PYEOF'
import sys
pac_path, q_path = sys.argv[1], sys.argv[2]
with open(pac_path, "rb") as f:
    pac = f.read()
query = pac[10:42]   # 32 bases
with open(q_path, "wb") as f:
    f.write(query)
PYEOF
echo "=== wrote query (32 bases at offset 10): ${QUERY}"

# ---- 5. Compute 32-mer hex key for the query --------------------------------
HEX_KEY="$(python3 - "${QUERY}" <<'PYEOF'
import sys
with open(sys.argv[1], "rb") as f:
    bases = list(f.read())   # list of int 0-3
key = 0
for i, b in enumerate(bases[:32]):
    shift = 2 * (31 - i)
    key |= (b & 3) << shift
print(f"{key:016x}")
PYEOF
)"
echo "=== 32-mer hex key: ${HEX_KEY}"

# ---- 6. Run cpp_caller ------------------------------------------------------
echo "=== running cpp_caller..."
"${CPP_CALLER}" "${PREFIX}" "${HEX_KEY}" "${QUERY}" "${PAC}"
echo "=== cpp_caller exited 0 — smoke PASSED"
