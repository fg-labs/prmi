// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Thread-local last-error string for the C FFI.

use std::cell::RefCell;
use std::ffi::CString;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub fn set_last_error(s: &str) {
    let c = CString::new(s.replace('\0', "?")).unwrap();
    LAST_ERROR.with(|cell| *cell.borrow_mut() = c);
}

pub fn clear_last_error() {
    set_last_error("");
}

pub fn with_last_error<R>(f: impl FnOnce(&CString) -> R) -> R {
    LAST_ERROR.with(|cell| f(&cell.borrow()))
}
