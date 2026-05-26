// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! On-disk sidecar format. The `magic` submodule lives here; the file-format
//! readers/writers (`meta`, `sa_file`, `model_file`, `skc_file`) land in
//! PR #3 (`feat/v0.1-sidecar`).

pub mod magic;
