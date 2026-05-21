// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Runtime lookup machinery — implements the §4.4 lookup math the v0.1
//! sidecar format encodes. The `LearnedIndex` type lands in Task 20; this
//! module also exports `lookup_with_components`, the slice-based entrypoint
//! used by training (Task 17) and tests.

pub mod lookup;
