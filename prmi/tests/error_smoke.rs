// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::Error;
use std::path::PathBuf;

#[test]
fn error_display_includes_kind() {
    let e = Error::BadMagic {
        file: PathBuf::new(),
        found: "XXXX".into(),
        expected: "PRMIv1".into(),
    };
    let s = format!("{e}");
    assert!(s.contains("PRMIv1"));
    assert!(s.contains("XXXX"));
}

#[test]
fn error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Error>();
}
