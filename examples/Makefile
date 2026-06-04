# Copyright (C) 2026 Fulcrum Genomics LLC
# SPDX-License-Identifier: MIT
#
# Build the cpp_caller example against libprmi_sys.a.
#
# Self-contained: `make -C examples` (or `make` from here) builds the static
# archive via cargo if it is missing, then compiles and links the example.

WORKSPACE    ?= ..
PRMI_LIBDIR  ?= $(WORKSPACE)/target/release
PRMI_ARCHIVE := $(PRMI_LIBDIR)/libprmi_sys.a

# Locate the generated header. prmi-sys/build.rs writes prmi.h to Cargo's
# OUT_DIR (authoritative) and only mirrors it into prmi-sys/include on a
# writable source tree, so prefer that mirror when present but fall back to the
# build-dir copy for read-only / vendored checkouts. Recursively expanded (`?=`)
# so it resolves on use — after the archive rule has generated the header.
# Override explicitly with `make PRMI_INCLUDE=<dir>`.
PRMI_INCLUDE ?= $(if $(wildcard $(WORKSPACE)/prmi-sys/include/prmi.h),$(WORKSPACE)/prmi-sys/include,$(dir $(firstword $(shell find $(WORKSPACE)/target -name prmi.h -path '*prmi-sys*' 2>/dev/null))))

CXX      ?= c++
CXXFLAGS := -std=c++17 -Wall -Wextra

# Link the static archive by full path (not -lprmi_sys): Cargo emits both
# libprmi_sys.a and a cdylib, and GNU ld prefers the shared object when both
# are on -L, producing a binary with a runtime libprmi_sys.so dependency that
# is not on the loader path. Naming the .a directly forces a static link on
# both platforms, matching this example's intent.
#
# The archive bundles libsais's OpenMP object code, so a static consumer must
# supply the OpenMP runtime itself (the cdylib had it linked in already):
# libomp via Homebrew on macOS, libgomp via -fopenmp on Linux.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(strip $(LIBOMP_PREFIX)),)
        $(error libomp not found; install it with `brew install libomp` (or set LIBOMP_PREFIX))
    endif
    LDFLAGS := $(PRMI_ARCHIVE) \
               -L$(LIBOMP_PREFIX)/lib -lomp \
               -framework Security -framework CoreFoundation
else
    LDFLAGS := $(PRMI_ARCHIVE) \
               -fopenmp -lpthread -ldl -lm
endif

.PHONY: all clean

all: cpp_caller

# Build the static archive (and generate prmi.h) via cargo if it is missing,
# so the example builds from a clean checkout with no prior cargo invocation.
$(PRMI_ARCHIVE):
	cargo build --release -p prmi-sys --manifest-path $(WORKSPACE)/Cargo.toml

# -I$(PRMI_INCLUDE) is resolved here (recipe time) so the header lookup runs
# after the archive rule above has generated/mirrored prmi.h.
cpp_caller: cpp_caller.cc $(PRMI_ARCHIVE)
	$(CXX) $(CXXFLAGS) -I$(PRMI_INCLUDE) $< -o $@ $(LDFLAGS)

clean:
	rm -f cpp_caller
