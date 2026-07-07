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
   ```text
   // Copyright Ryan Marcus 2020
   // Modified by Fulcrum Genomics 2026
   ```

3. **New files** authored by Fulcrum use only:
   ```text
   // Copyright (C) 2026 Fulcrum Genomics LLC
   // SPDX-License-Identifier: MIT
   ```

The `LICENSE` and `COPYING` files at the repo root carry Marcus's MIT
license text; a `Copyright (c) 2026 Fulcrum Genomics LLC` line is added
alongside Marcus's original notice to reflect Fulcrum's authorship of the
new work. MIT terms apply to the whole repo including all new files.

## Acknowledgments

BWA-MEME (Jung & Han, Bioinformatics 2022) demonstrated the P-RMI
variant on suffix-array seeding and informed the algorithmic shape of
prmi's trainer. No code from BWA-MEME is used in prmi; the trainer is
Fulcrum-authored on top of Marcus's RMI primitives.
