# Hoon overflow

Copies of SoftFloat 3e's `s_roundPackToF{16,32,64,128}.c` with one change:
overflow returns infinity in every rounding mode, as Hoon `++fl` does
(urbit/urbit#7426). Stock SoftFloat returns the largest finite value in
modes that round toward zero. Each change is marked `hoon-overflow`.

Every oracle build compiles these files in place of the vendored ones:
`sdfloat-oracle`, `sdblas-oracle` (SoftFloat and SoftBLAS), and
`scripts/run-softblas-munit.sh`. Every overflow in SoftFloat, including
conversions and integer-to-float, passes through these four functions, so
the oracles produce Hoon's bits on every input.

`diff ../../vendor/SoftFloat/source/s_roundPackToF32.c s_roundPackToF32.c`
shows the change.
