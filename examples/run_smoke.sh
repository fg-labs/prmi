#!/usr/bin/env bash
# Copyright (C) 2026 Fulcrum Genomics LLC
# SPDX-License-Identifier: MIT
#
# End-to-end smoke harness for the cpp_caller example.
#   1. Writes a synthetic 4-kb FASTA.
#   2. Runs `prmi build` to produce the sidecar.
#   3. Writes a binary PAC file (1 byte/base, values 0–3) and a 2-bit packed
#      PAC file (BWA-MEME bntpac convention: 4 bases/byte, MSB-first).
#   4. Writes a binary 32-base query file.
#   5. Computes the expected 32-mer hex key.
#   6. Runs cpp_caller and checks the exit code; verifies the spectrum,
#      sa_positions, and batch FFI smoke output.
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

# ---- 3a. Unpacked PAC file (1 byte per base: A=0 C=1 G=2 T=3) --------------
PAC_UNPACKED="${TMPDIR}/smoke.pac"
python3 - "${PAC_UNPACKED}" <<'PYEOF'
import sys
path = sys.argv[1]
# ACGT × 1024
enc = [0, 1, 2, 3] * 1024
with open(path, "wb") as f:
    f.write(bytes(enc))
PYEOF
echo "=== wrote unpacked PAC: ${PAC_UNPACKED} ($(wc -c < "${PAC_UNPACKED}") bytes)"

# ---- 3b. Packed PAC file (2 bits per base: BWA-MEME bntpac convention) ------
# Base 0 in bits 6-7, base 1 in bits 4-5, base 2 in bits 2-3, base 3 in bits 0-1.
PAC_PACKED="${TMPDIR}/smoke_packed.pac"
PAC_NUM_BASES=4096
python3 - "${PAC_UNPACKED}" "${PAC_PACKED}" <<'PYEOF'
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path, "rb") as f:
    bases = list(f.read())   # list of int 0-3
n = len(bases)
out = bytearray((n + 3) // 4)
for i, b in enumerate(bases):
    shift = 6 - 2 * (i % 4)
    out[i // 4] |= (b & 0x3) << shift
with open(dst_path, "wb") as f:
    f.write(out)
PYEOF
echo "=== wrote packed PAC: ${PAC_PACKED} ($(wc -c < "${PAC_PACKED}") bytes, ${PAC_NUM_BASES} bases)"

# ---- 4. Query file: first 32 bases starting at offset 10 -------------------
# offset 10 in ACGT×1024: bases at pos 10,11,...,41 → G T A C G T A C G T ...
#   (10 % 4 = 2 → G, 11%4=3 → T, 12%4=0 → A, 13%4=1 → C, ...)
QUERY="${TMPDIR}/query.bin"
python3 - "${PAC_UNPACKED}" "${QUERY}" <<'PYEOF'
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
OUTPUT="$("${CPP_CALLER}" "${PREFIX}" "${HEX_KEY}" "${QUERY}" \
    "${PAC_UNPACKED}" "${PAC_PACKED}" "${PAC_NUM_BASES}")"
echo "${OUTPUT}"

# ---- 7. Verify prmi_sa_positions appeared in output -------------------------
if ! echo "${OUTPUT}" | grep -q "prmi_sa_positions:"; then
    echo "FAIL: prmi_sa_positions line missing from cpp_caller output" >&2
    exit 1
fi
# Verify prmi_sa_positions returned rc=0
if ! echo "${OUTPUT}" | grep "prmi_sa_positions:" | grep -q "rc=0"; then
    echo "FAIL: prmi_sa_positions did not return rc=0" >&2
    exit 1
fi
echo "=== prmi_sa_positions rc=0 — OK"

# Verify spectrum smoke section: forward_spectrum appeared and returned rc=0
if ! echo "${OUTPUT}" | grep -q "forward_spectrum nsteps="; then
    echo "FAIL: forward_spectrum line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "forward_spectrum nsteps=" | grep -q "rc=0"; then
    echo "FAIL: forward_spectrum did not return rc=0" >&2
    exit 1
fi
echo "=== forward_spectrum rc=0 — OK"

# Verify backward_spectrum appeared and returned rc=0
if ! echo "${OUTPUT}" | grep -q "backward_spectrum nsteps="; then
    echo "FAIL: backward_spectrum line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "backward_spectrum nsteps=" | grep -q "rc=0"; then
    echo "FAIL: backward_spectrum did not return rc=0" >&2
    exit 1
fi
echo "=== backward_spectrum rc=0 — OK"

# Verify sa_positions_strided appeared and returned rc=0
if ! echo "${OUTPUT}" | grep -q "sa_positions_strided"; then
    echo "FAIL: sa_positions_strided line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "sa_positions_strided" | grep -q "rc=0"; then
    echo "FAIL: sa_positions_strided did not return rc=0" >&2
    exit 1
fi
echo "=== sa_positions_strided rc=0 — OK"

# Verify forward_spectrum_batch appeared, returned rc=0, and tasks matched
if ! echo "${OUTPUT}" | grep -q "forward_spectrum_batch rc="; then
    echo "FAIL: forward_spectrum_batch line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "forward_spectrum_batch rc=" | grep -q "rc=0"; then
    echo "FAIL: forward_spectrum_batch did not return rc=0" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "forward_spectrum_batch:" | grep -q "both tasks match single"; then
    echo "FAIL: forward_spectrum_batch tasks did not match single-call result" >&2
    exit 1
fi
echo "=== forward_spectrum_batch rc=0, tasks match single — OK"

echo "=== cpp_caller exited 0 — smoke PASSED"
