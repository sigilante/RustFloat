Berkeley SoftFloat Release 3e by John R. Hauser, BSD-3-Clause (see COPYING.txt).

Copied 2026-09-12 from `~/urbit/SoftBLAS/SoftFloat/source/`. Only the files
listed by `build/Linux-x86_64-GCC/Makefile` (the `SOFTFLOAT_FAST_INT64` object
set) plus the `ARM-VFPv2` specialization are kept. ARM-VFPv2 is what vere links
(`vere/ext/softfloat/build.zig`); it only affects NaN payloads and the integer
returned for NaN / out-of-range float-to-int conversions.

Do not edit these files. `../csrc/` holds our own platform header and shim.
