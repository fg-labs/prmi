// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Opaque handle: `prmi_index_t` in C, wraps `Box<Handle>` in Rust.

use prmi::index::LearnedIndex;

/// Opaque handle. cbindgen forward-declares it as a zero-sized struct in C.
///
/// cbindgen:opaque
#[repr(C)]
pub struct prmi_index_t {
    _opaque: [u8; 0],
}

#[allow(dead_code)]
pub(crate) struct Handle(pub LearnedIndex);

impl Handle {
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
    #[allow(dead_code)]
    pub unsafe fn as_ref<'a>(p: *const prmi_index_t) -> &'a Handle {
        unsafe { &*(p as *const Handle) }
    }
}
