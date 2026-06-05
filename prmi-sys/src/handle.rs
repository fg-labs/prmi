// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Opaque handle: `prmi_index_t` in C, wraps `Box<Handle>` in Rust.

use prmi::index::LearnedIndex;
use std::ffi::CString;

/// Opaque handle. cbindgen forward-declares it as a zero-sized struct in C.
///
/// cbindgen:opaque
#[repr(C)]
pub struct prmi_index_t {
    _opaque: [u8; 0],
}

pub(crate) struct Handle {
    pub idx: LearnedIndex,
    /// NUL-terminated copy of the sidecar's format magic (e.g. `"PRMIv2"`).
    /// Cached here so `prmi_format_version` can return a pointer that lives as
    /// long as the handle without allocating on each call.
    pub magic: CString,
}

impl Handle {
    /// Construct a `Handle` from a loaded index, caching its format magic as a
    /// `CString`. Panics internally on embedded NUL bytes, which cannot occur
    /// in a well-formed magic string; `unwrap_or_default` falls back to an empty
    /// string in the impossible case.
    pub fn new(idx: LearnedIndex) -> Self {
        let magic =
            CString::new(idx.format_version()).unwrap_or_else(|_| CString::new("").unwrap());
        Handle { idx, magic }
    }

    pub fn into_raw(self) -> *mut prmi_index_t {
        Box::into_raw(Box::new(self)) as *mut prmi_index_t
    }

    /// # Safety
    /// `p` must have been returned by `into_raw` and not yet freed.
    pub unsafe fn from_raw(p: *mut prmi_index_t) -> Box<Self> {
        unsafe { Box::from_raw(p as *mut Handle) }
    }

    /// # Safety
    /// `p` must be non-null and point to a valid Handle.
    pub unsafe fn as_ref<'a>(p: *const prmi_index_t) -> &'a Handle {
        unsafe { &*(p as *const Handle) }
    }
}
