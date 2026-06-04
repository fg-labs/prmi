// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Thread-local last-error string for the C FFI.

use std::cell::RefCell;
use std::ffi::CString;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub fn set_last_error(s: &str) {
    let cleaned = s.replace('\0', "?");
    let c = CString::new(cleaned).unwrap_or_else(|_| {
        // Unreachable in practice — we just removed all NULs. Defensive
        // fallback to ensure we NEVER panic across an FFI boundary.
        CString::new("prmi: error message corrupted")
            .unwrap_or_else(|_| CString::new(b"".to_vec()).expect("empty bytes never contain NUL"))
    });
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = c;
    });
}

pub fn clear_last_error() {
    set_last_error("");
}

pub fn with_last_error<R>(f: impl FnOnce(&CString) -> R) -> R {
    LAST_ERROR.with(|cell| f(&cell.borrow()))
}
