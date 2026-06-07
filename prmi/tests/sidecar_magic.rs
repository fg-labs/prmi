// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sidecar::magic::{ISA_MAGIC, L1_MAGIC, L2_MAGIC, META_MAGIC, SA_MAGIC};

#[test]
fn magics_match_spec_hex() {
    assert_eq!(META_MAGIC, "PRMIv2");
    assert_eq!(SA_MAGIC, 0x534D5250); // "PRMS" written LE on disk reads as this u32
    assert_eq!(L1_MAGIC, 0x314C4D50); // "PML1"
    assert_eq!(L2_MAGIC, 0x324C4D50); // "PML2"
    assert_eq!(ISA_MAGIC, 0x41534950); // "PISA"
}

#[test]
fn magics_on_disk_byte_order() {
    assert_eq!(SA_MAGIC.to_le_bytes(), *b"PRMS");
    assert_eq!(L1_MAGIC.to_le_bytes(), *b"PML1");
    assert_eq!(L2_MAGIC.to_le_bytes(), *b"PML2");
    assert_eq!(ISA_MAGIC.to_le_bytes(), *b"PISA");
}
