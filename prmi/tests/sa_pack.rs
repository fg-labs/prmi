// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::sa::{pack_position, unpack_position, MAX_PACKED_POSITION};

#[test]
fn roundtrip_zero() {
    let bytes = pack_position(0);
    assert_eq!(bytes, [0u8; 5]);
    assert_eq!(unpack_position(&bytes), 0);
}

#[test]
fn roundtrip_max() {
    let bytes = pack_position(MAX_PACKED_POSITION);
    assert_eq!(unpack_position(&bytes), MAX_PACKED_POSITION);
}

#[test]
fn layout_is_hi32_then_lo8_little_endian() {
    let bytes = pack_position(0x01_2345_6789);
    assert_eq!(bytes, [0x67, 0x45, 0x23, 0x01, 0x89]);
    assert_eq!(unpack_position(&bytes), 0x01_2345_6789);
}

#[test]
#[should_panic]
fn pack_overflow_panics() {
    pack_position(MAX_PACKED_POSITION + 1);
}
