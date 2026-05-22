# Installing prmi-sys for downstream C/C++ consumers

prmi-sys ships as a static library (`libprmi_sys.a`) plus a generated
header (`prmi.h`). It is built via Cargo and is not auto-installed by
`cargo install`; downstream projects need to place the artifacts where
their build system can find them.

## Build

```bash
cargo build --release -p prmi-sys
# Produces:
#   target/release/libprmi_sys.a
#   target/release/libprmi_sys.{dylib,so}
#   prmi-sys/include/prmi.h
```

## pkg-config

After building, substitute `prmi-sys.pc.in` (in this directory) into a
`.pc` file and place it on `PKG_CONFIG_PATH`:

```bash
prefix=$HOME/.local
mkdir -p $prefix/lib/pkgconfig $prefix/include $prefix/lib
sed -e "s|@PREFIX@|$prefix|" \
    -e "s|@VERSION@|0.1.0|" \
    -e "s|@PLATFORM_LIBS@|-lpthread -ldl -lm|" \
    prmi-sys/prmi-sys.pc.in > $prefix/lib/pkgconfig/prmi-sys.pc

cp target/release/libprmi_sys.a $prefix/lib/
cp prmi-sys/include/prmi.h      $prefix/include/

export PKG_CONFIG_PATH=$prefix/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --libs --cflags prmi-sys
```

Platform-specific link flags for `@PLATFORM_LIBS@`:
- Linux: `-lpthread -ldl -lm`
- macOS: `-framework Security -framework CoreFoundation`

## CMake

`prmi-sys/cmake/PrmiSysConfig.cmake.in` is a `find_package(PrmiSys)` template.
Substitute `@PREFIX@` and `@VERSION@`, then install and point
`CMAKE_PREFIX_PATH` at the installation root:

```bash
prefix=$HOME/.local
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
