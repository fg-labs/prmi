# Fork notice

`fg-labs/prmi` is a fork of [`learnedsystems/RMI`](https://github.com/learnedsystems/RMI)
(Ryan Marcus, MIT-licensed) at upstream commit
`8e147da1389b55b0fb74140187288957b832a2a7`.

The fork was renamed from `fg-labs/RMI` to `fg-labs/prmi` and extended by
Fulcrum Genomics with a genomic suffix-array trainer, a frozen on-disk
sidecar format, a runtime reader, and a C FFI (`prmi-sys`).

## Copyright headers

Every source file carries one of three headers:

1. **Unchanged upstream files** retain Marcus's original header verbatim:
   `// Copyright Ryan Marcus 2020`

2. **Modified upstream files** keep Marcus's header and add a second line on
   first modification:
   ```
   // Copyright Ryan Marcus 2020
   // Modified by Fulcrum Genomics 2026
   ```

3. **New files** authored by Fulcrum use only:
   ```
   // Copyright (C) 2026 Fulcrum Genomics LLC
   // SPDX-License-Identifier: MIT
   ```

The `LICENSE` and `COPYING` files at the repo root carry Marcus's MIT
license text unchanged; MIT terms apply to the whole repo including all
new files.
