// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! On-disk magic constants for the prmi sidecar. Bumped to v2 for the 2× SA.

/// Magic string written into the `.meta` TOML and reported by the C API
/// via `prmi_format_version()`.
pub const META_MAGIC: &str = "PRMIv2";

/// On-disk format version stored in every binary header.
pub const FORMAT_VERSION: u32 = 2;

/// Magic for `.sa` file header (ASCII "PRMS", little-endian).
pub const SA_MAGIC: u32 = u32::from_le_bytes(*b"PRMS");

/// Magic for `.l1` file header (ASCII "PML1", little-endian).
pub const L1_MAGIC: u32 = u32::from_le_bytes(*b"PML1");

/// Magic for `.l2` file header (ASCII "PML2", little-endian).
pub const L2_MAGIC: u32 = u32::from_le_bytes(*b"PML2");

/// Magic for the `.kmt` k-mer table file header (ASCII "PMKT", little-endian).
pub const KMT_MAGIC: u32 = u32::from_le_bytes(*b"PMKT");

/// Magic for the optional `.isa` inverse-suffix-array file header
/// (ASCII "PMIS", little-endian).
pub const ISA_MAGIC: u32 = u32::from_le_bytes(*b"PMIS");
