# Copyright (C) 2026 Fulcrum Genomics LLC
# SPDX-License-Identifier: MIT
#
# Build the cpp_caller example against libprmi_sys.a.
# Run from the workspace root:  make -C examples
# Or from this directory:       make

PRMI_LIBDIR ?= ../target/release
PRMI_INCLUDE ?= ../prmi-sys/include

CXX      ?= c++
CXXFLAGS := -std=c++17 -Wall -Wextra -I$(PRMI_INCLUDE)

# Platform-specific link flags
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    LDFLAGS := -L$(PRMI_LIBDIR) -lprmi_sys \
               -framework Security -framework CoreFoundation
else
    LDFLAGS := -L$(PRMI_LIBDIR) -lprmi_sys \
               -lpthread -ldl -lm
endif

.PHONY: all clean

all: cpp_caller

cpp_caller: cpp_caller.cc $(PRMI_LIBDIR)/libprmi_sys.a
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f cpp_caller
