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

## Inherited from BWA-MEME's RMI fork

This repository's `prmi/src/upstream/` tree absorbs additional code from
[`kaist-ina/BWA-MEME`](https://github.com/kaist-ina/BWA-MEME)'s `RMI/rmi_lib/`
subtree (MIT-licensed) at upstream commit
`42b2194b2c7d03249e300272dc12a0659d9cc3b1`. Specifically:

- `prmi/src/upstream/models/piecewiselinear.rs` (the P-RMI `pwl` model)
- The `train_partial_three_layer` codepath in `prmi/src/upstream/train/two_layer.rs`
- Related extensions to `LowerBoundCorrection` and the `TrainedRMI` struct

These additions are the published P-RMI training pipeline that BWA-MEME
deploys; we inherit them so prmi's reader can decode the same on-disk
layout (see the v0.1 brief §4.4).

## Tri-attribution headers for inherited BWA-MEME content

Source files that contain code copied or adapted from BWA-MEME's RMI fork
carry a three-line copyright preamble:

```
// Copyright Ryan Marcus 2020          (origin: learnedsystems/RMI)
// Copyright 2022 Youngmok Jung et al. (origin: kaist-ina/BWA-MEME RMI fork)
// Modified by Fulcrum Genomics 2026
// SPDX-License-Identifier: MIT
```

For files that contain content from only one upstream, drop the
corresponding line. New files authored entirely by Fulcrum use the
single-line Fulcrum header from §3.
