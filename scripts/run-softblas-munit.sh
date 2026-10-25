#!/bin/sh
# Run SoftBLAS's own munit test suite, unmodified, against the Rust port:
# the tests and headers are the vendored SoftBLAS ones, the library is
# libsdblas_cabi.a (Rust), and SoftFloat is linked only because the test
# helpers and softblas.h's inline NaN helpers call it.
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
V=$ROOT/crates/sdblas-oracle/vendor
SF=$ROOT/crates/sdfloat-oracle/vendor/SoftFloat/source
OUT=${OUT:-$ROOT/target/softblas-munit}
mkdir -p "$OUT/sf"

cargo build --release -p sdblas-cabi
LIB=$ROOT/target/release/libsdblas_cabi.a

CC=${CC:-cc}
SFFLAGS="-O2 -w -DSOFTFLOAT_FAST_INT64 -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32 -DTHREAD_LOCAL=_Thread_local -I$ROOT/crates/sdfloat-oracle/csrc -I$SF/include -I$SF/ARM-VFPv2"
# csrc/hoon-overflow replaces the vendored files of the same name, giving
# SoftFloat Hoon's overflow-to-infinity in every mode.
HOON=$ROOT/crates/sdfloat-oracle/csrc/hoon-overflow
if [ ! -f "$OUT/libsoftfloat.a" ] || [ -n "$(find "$HOON" -newer "$OUT/libsoftfloat.a")" ]; then
  for f in "$SF"/*.c "$SF"/ARM-VFPv2/*.c; do
    b=$(basename "$f")
    [ -f "$HOON/$b" ] && f=$HOON/$b
    $CC $SFFLAGS -c "$f" -o "$OUT/sf/${b%.c}.o"
  done
  ar rcs "$OUT/libsoftfloat.a" "$OUT"/sf/*.o
fi

TESTS="$V/SoftBLAS/tests/test_all.c $(find "$V/SoftBLAS/tests/blas" -name '*.c' | sort | tr '\n' ' ')"
case "$(uname)" in
  Darwin) SYSLIBS="-lm" ;;
  *)      SYSLIBS="-lm -lpthread -ldl" ;;
esac
# Tests are built at -O0 as SoftBLAS's Makefile does; at -O1 clang leaves a
# compound-literal array in test_cdotu_rounding_modes uninitialized on the
# first iteration (an issue in the test, not the library). -std=gnu17 because
# munit.c uses ATOMIC_VAR_INIT, which C23 removed and GCC 15 now rejects.
$CC -O0 -std=gnu17 -w -DSOFTFLOAT_FAST_INT64 -DTHREAD_LOCAL=_Thread_local \
  -I"$V/SoftBLAS/include" -I"$V/SoftBLAS/tests/blas/include" -I"$V/munit" \
  -I"$ROOT/crates/sdfloat-oracle/csrc" -I"$SF/include" -I"$SF/ARM-VFPv2" \
  $TESTS "$V/munit/munit.c" "$LIB" "$OUT/libsoftfloat.a" $SYSLIBS -o "$OUT/test_all"
"$OUT/test_all" "$@"
