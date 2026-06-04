# Installing prmi-sys for downstream C/C++ consumers

prmi-sys ships as a static library (`libprmi_sys.a`) plus a generated
header (`prmi.h`). It is built via Cargo and is not auto-installed by
`cargo install`; downstream projects need to place the artifacts where
their build system can find them.

## Prerequisites

### macOS

`prmi` uses libsais's OpenMP support for parallel suffix-array construction.
Apple Clang does not bundle `libomp`, so you must install it via Homebrew before
building:

```bash
brew install libomp
```

The workspace `.cargo/config.toml` sets `CFLAGS=-I$(brew --prefix libomp)/include`
for the default Homebrew prefixes (`/opt/homebrew` on Apple Silicon,
`/usr/local` on Intel). If your Homebrew prefix differs, override before
building:

```bash
CFLAGS="-I$(brew --prefix libomp)/include" cargo build --release -p prmi-sys
```

### Linux

No extra steps needed. GCC's `libgomp` is used automatically and is available
in any standard distribution.

## Build

```bash
cargo build --release -p prmi-sys
# Produces:
#   target/release/libprmi_sys.a
#   target/release/libprmi_sys.{dylib,so}
#   prmi-sys/include/prmi.h   (convenience mirror; only on a writable source tree)
```

The generated `prmi.h` is always written to the crate's Cargo `OUT_DIR` (the
authoritative copy, which works even for read-only or vendored source trees).
On a writable checkout the build also mirrors it to `prmi-sys/include/prmi.h`
for convenience. If your source tree is read-only, locate the header under the
build directory instead, e.g.:

```bash
header=$(find target -name prmi.h -path '*prmi-sys*' | head -n1)
```

## pkg-config

After building, substitute `prmi-sys.pc.in` (in this directory) into a
`.pc` file and place it on `PKG_CONFIG_PATH`:

```bash
prefix=$HOME/.local
mkdir -p $prefix/lib/pkgconfig $prefix/include $prefix/lib
sed -e "s|@PREFIX@|$prefix|" \
    -e "s|@VERSION@|0.1.0|" \
    -e "s|@PLATFORM_LIBS@|-lpthread -ldl -lm -lgomp|" \
    prmi-sys/prmi-sys.pc.in > $prefix/lib/pkgconfig/prmi-sys.pc

cp target/release/libprmi_sys.a $prefix/lib/
# Copy the generated header from its authoritative build-dir location (works even
# when prmi-sys/include/prmi.h was not mirrored, e.g. a read-only source tree).
header=$(find target -name prmi.h -path '*prmi-sys*' | head -n1)
cp "$header"                    $prefix/include/prmi.h

export PKG_CONFIG_PATH=$prefix/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --libs --cflags prmi-sys
```

Platform-specific link flags for `@PLATFORM_LIBS@`:
- Linux: `-lpthread -ldl -lm -lgomp`
- macOS: `-framework Security -framework CoreFoundation -L$(brew --prefix libomp)/lib -lomp`

The OpenMP runtime (`-lgomp` on Linux/GCC, `-lomp` on macOS) is required
because `libsais`'s OpenMP object code is statically embedded in
`libprmi_sys.a` (see below) — a consumer of the static archive must supply the
runtime itself. On macOS, `-L$(brew --prefix libomp)/lib` points the linker at
the Homebrew `libomp`.

## CMake

`prmi-sys/cmake/PrmiSysConfig.cmake.in` is a `find_package(PrmiSys)` template.
Substitute `@PREFIX@` and `@VERSION@`, then install and point
`CMAKE_PREFIX_PATH` at the installation root:

```bash
prefix=$HOME/.local
mkdir -p "$prefix/lib/cmake/PrmiSys"
sed -e "s|@PREFIX@|$prefix|" \
    -e "s|@VERSION@|0.1.0|" \
    prmi-sys/cmake/PrmiSysConfig.cmake.in \
    > $prefix/lib/cmake/PrmiSys/PrmiSysConfig.cmake

# Then in your CMakeLists.txt:
#   list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
#   find_package(PrmiSys REQUIRED)
#   target_link_libraries(my_target PRIVATE PrmiSys::prmi_sys)
```

`libsais` is statically embedded into `libprmi_sys.a`; no system libsais
is required.
