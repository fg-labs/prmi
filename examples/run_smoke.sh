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
#   6. Runs cpp_caller and checks the exit code; verifies packed == unpacked.
#
# Run from anywhere; all paths are relative to this script's directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Locate the `prmi` CLI. Honor an explicit $PRMI override, else prefer a release
# build, else fall back to a debug build so a plain `cargo build --workspace`
# (which produces target/debug/prmi) is enough to run the smoke test.
PRMI="${PRMI:-${WORKSPACE}/target/release/prmi}"
if [[ ! -x "${PRMI}" && -x "${WORKSPACE}/target/debug/prmi" ]]; then
  PRMI="${WORKSPACE}/target/debug/prmi"
fi
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
# Extract the position count from the smem_range packed result line.
# Line is: smem_range(packed):   rc=... k=... l=<count> s=...
# `|| true` guards against a grep no-match (exit 1) aborting the script under
# `set -o pipefail`; the emptiness check below handles a missing value.
L_VAL="$(echo "${OUTPUT}" | grep "smem_range(packed):" | grep -o 'l=[0-9]*' | cut -d= -f2 || true)"
# And the count from the sa_positions line: count=<N>
SA_COUNT="$(echo "${OUTPUT}" | grep "prmi_sa_positions:" | grep -o 'count=[0-9]*' | head -1 | cut -d= -f2 || true)"
if [ -n "${L_VAL}" ] && [ -n "${SA_COUNT}" ]; then
    if [ "${L_VAL}" != "${SA_COUNT}" ]; then
        echo "FAIL: smem_range l=${L_VAL} does not match prmi_sa_positions count=${SA_COUNT}" >&2
        exit 1
    fi
    echo "=== prmi_sa_positions count=${SA_COUNT} matches smem_range l=${L_VAL} — OK"
fi
# Verify prmi_smem_range_batch_packed appeared in output and returned rc=0
if ! echo "${OUTPUT}" | grep -q "prmi_smem_range_batch_packed:"; then
    echo "FAIL: prmi_smem_range_batch_packed line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "prmi_smem_range_batch_packed:" | grep -q "rc=0"; then
    echo "FAIL: prmi_smem_range_batch_packed did not return rc=0" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "prmi_smem_range_batch_packed:" | grep -q "all.*slots match single-key"; then
    echo "FAIL: prmi_smem_range_batch_packed batch/single-key mismatch check did not pass" >&2
    exit 1
fi
# Verify prmi_smem_range_long_read_packed appeared in output and returned rc=0
if ! echo "${OUTPUT}" | grep -q "prmi_smem_range_long_read_packed:"; then
    echo "FAIL: prmi_smem_range_long_read_packed line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "prmi_smem_range_long_read_packed:" | grep -q "rc=0"; then
    echo "FAIL: prmi_smem_range_long_read_packed did not return rc=0" >&2
    exit 1
fi
echo "=== prmi_smem_range_long_read_packed rc=0 — OK"

# Verify prmi_minimizer_32mer appeared in output and returned rc=0
if ! echo "${OUTPUT}" | grep -q "prmi_minimizer_32mer:"; then
    echo "FAIL: prmi_minimizer_32mer line missing from cpp_caller output" >&2
    exit 1
fi
if ! echo "${OUTPUT}" | grep "prmi_minimizer_32mer:" | grep -q "rc=0"; then
    echo "FAIL: prmi_minimizer_32mer did not return rc=0" >&2
    exit 1
fi
echo "=== prmi_minimizer_32mer rc=0 — OK"

echo "=== cpp_caller exited 0 — smoke PASSED"
