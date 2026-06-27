#!/usr/bin/env bash
# Run the prmi Criterion benches under a host-tuned target-cpu. The shipped
# release artifact is NOT tuned (portable, x86-64 baseline); this harness is for
# measurement only.
#
# Hermetic by design: this harness ALWAYS builds with exactly
# `-C target-cpu=native` and does not inherit ambient RUSTFLAGS — otherwise a
# stale/foreign RUSTFLAGS in the environment would silently change what is
# measured. To bench a specific floor instead of native (e.g. the documented
# distribution floor x86-64-v3 on x86, or neoverse-n1/v2 on Graviton), do NOT
# use this script; run the bench directly with your own flags, e.g.
#   RUSTFLAGS="-C target-cpu=x86-64-v3" cargo bench -p prmi
# The crate never pins target-cpu in .cargo/config.toml.
set -euo pipefail
if [[ -n "${RUSTFLAGS:-}" ]]; then
  printf 'bench-native.sh: overriding ambient RUSTFLAGS=%q with "-C target-cpu=native" (hermetic native bench)\n' \
    "$RUSTFLAGS" >&2
fi
export RUSTFLAGS="-C target-cpu=native"
exec cargo bench -p prmi "$@"
