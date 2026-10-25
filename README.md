# RustFloat

Pure-Rust IEEE 754 software floating point and BLAS with explicit rounding
modes, bit-identical to Berkeley SoftFloat 3e and to
[SoftBLAS](https://github.com/urbit/SoftBLAS) except on overflow, where it
follows Hoon. Built for NockApp jets of the
[urbit/numerics](https://github.com/urbit/numerics) suite, where a jet must
return exactly the bits the Hoon `++rh`/`++rs`/`++rd`/`++rq` cores and
`/lib/lagoon` return.

The `sdfloat` crate is `#![no_std]`, `#![forbid(unsafe_code)]`, has no
dependencies, allocates nothing, holds no global state, and uses no hardware
float type anywhere, tests included. Rounding mode is a parameter on every
inexact operation. Specification: [`doc/nockapp-softfloat-rs-spec.md`](doc/nockapp-softfloat-rs-spec.md).

## Layout

| Crate | Purpose |
|---|---|
| `crates/sdfloat/` | Scalar library: `F16`, `F32`, `F64`, `F128`, the `SoftFloat` trait, `Round`. |
| `crates/sdblas/` | BLAS Levels 1 to 3 over `sdfloat`, real and complex, all four widths; the SoftBLAS port. |
| `crates/sdblas-cabi/` | The SoftBLAS C ABI over `sdblas` as a static library: a drop-in `libsoftblas.a`, and how SoftBLAS's own test suite runs against the Rust. |
| `crates/sdfloat-oracle/` | Test-only FFI to vendored SoftFloat 3e C (ARM-VFPv2 specialization, as vere links), with Hoon overflow (`csrc/hoon-overflow/`). |
| `crates/sdblas-oracle/` | Test-only FFI to vendored SoftBLAS C, plus its munit test suite; links the same Hoon-overflow SoftFloat. |

The oracle crates are dev-dependencies only; nothing ships with C. Inside
`crates/sdfloat/src`, `pack.rs` is the single rounding point and each file in
`ops/` is one operation named after the SoftFloat source it replaces. In
`crates/sdblas/src`, each routine's doc names the SoftBLAS source it reproduces.

## Use

```rust
use sdfloat::{F32, Round, SoftFloat};

let a = F32(0x3f80_0000);               // 1.0
let b = F32(0x2edb_e6ff);               // 1e-10
assert_eq!(a.add(b, Round::NearEven), a);
assert_eq!(a.add(b, Round::Away), F32(0x3f80_0001));
assert_eq!(F32::MAX.mul(F32(0x4000_0000), Round::Zero), F32::INF); // as Hoon ++fl
```

BLAS routines are generic over the width and take the mode last, where
SoftBLAS takes `rndMode`:

```rust
use sdblas::{Round, Trans, F64};
let (a, b) = ([F64::ONE, F64::ONE], [F64::ONE, F64::ONE]);
let mut c = [F64::ZERO; 4];
sdblas::gemm(Trans::NoTrans, Trans::NoTrans, 2, 1, 2, F64::ONE, &a, 1, &b, 2, F64::ZERO, &mut c, 2, Round::NearEven);
```

Real routines: `asum axpy copy dot nrm2 nrm2_b rot rotg rotm rotmg scal swap
iamax sdsdot hsdot gemv gemm`; `sdblas::complex` has `axpy copy dotc dotu scal
swap rot asum nrm2 iamax gemv gemm` over `Complex<F>`. Strides, argument
validation, accumulation order and NaN handling follow SoftBLAS exactly; the
crate docs list the conventions.

`Round` has Hoon's five modes: `NearEven` (`%n`), `Up` (`%u`), `Down` (`%d`),
`Zero` (`%z`), `Away` (`%a`, directed away from zero). Operations: `add sub
mul div sqrt round_to_int eq lt le to_i32 to_i64 from_i32 from_i64 from_u32
from_u64 is_nan is_inf is_zero abs neg`, plus width conversions
(`to_f16(r)`, `to_f32(...)`, `to_f64(...)`, `to_f128()`, or the generic
`convert::<A, B>(x, r)`). `F128::to_u64_pair` gives nockvm's word order.

## Conventions beyond IEEE 754

- Any NaN result is the canonical NaN of its width (`0x7e00`, `0x7fc00000`,
  `0x7ff8…`, `0x7fff8…`), matching Hoon `++ff` and vere. Payloads never
  propagate; there is no signaling NaN.
- Overflow gives infinity in every mode, as Hoon `++fl` does. IEEE 754-2019
  §7.4, SoftFloat, and therefore Vere's C jets give the largest finite value
  instead under `%z`, `%d` for a positive result, and `%u` for a negative
  one; see urbit/urbit#7426. The oracles compile SoftFloat with Hoon's rule
  (`crates/sdfloat-oracle/csrc/hoon-overflow/`).
- Float to int: NaN gives 0, out of range saturates. This is SoftFloat under
  the ARM-VFPv2 specialization. The jet layer bails on those inputs instead.
- Signed zero follows IEEE §6.3: `x - x` is `+0` except under `Down`.

## Testing

```sh
cargo test --workspace                         # oracle fuzz, Hoon vectors, semantics, exhaustive f16 unary
SOFTFLOAT_ORACLE_N=2000000 cargo test --release -p sdfloat --test oracle
SDBLAS_ORACLE_N=40000 cargo test --release -p sdblas --test oracle
cargo test --release -p sdfloat --test exhaustive_f16 -- --ignored   # all 2^32 f16 pairs
scripts/run-softblas-munit.sh                  # SoftBLAS's 313 munit tests against the Rust port
scripts/no-hardware-floats.sh
```

For `sdfloat`, three layers, all run by `cargo test --workspace`:

1. **Oracle fuzz** (`tests/oracle.rs`): raw bits against SoftFloat for every
   op, width and mode on structured samples (zeros, subnormals, exponent
   boundaries, ties, near-cancellation pairs). `Away` is derived from
   SoftFloat's `round_max`/`round_min` by result sign. Exhaustive binary16
   coverage lives in `tests/exhaustive_f16.rs`; the 2^32-pair binary ops are
   `#[ignore]`d and run nightly in CI (they pass: ~5 minutes on 4 cores).
2. **Hoon vectors** (`tests/hoon_vectors.rs`, `tests/vectors/hoon_f*.txt`):
   1500 rows per width from the un-jetted `++ff` core via `urbit eval`,
   covering add/sub/mul/div/sqt/lth/lte/equ/toi/sun/san in all five modes.
   Regenerate with `URBIT=/path/to/urbit scripts/gen-hoon-vectors.py`.
   Arithmetic must match bit for bit, overflow included; between 106 and 633
   rows per width overflow where IEEE would saturate.
3. **Hand-checked semantics** (`tests/semantics.rs`): the conventions above.

For `sdblas`: a differential fuzz of every routine, width, stride pattern,
layout and transpose against SoftBLAS C (`crates/sdblas/tests/oracle.rs`,
four modes; SoftBLAS's `'a'` is nearest-ties-away, not Hoon's `%a`), the
complete SoftBLAS munit suite run unchanged through the C ABI (313 tests),
and hand-checked semantics. Two SoftBLAS quirks are documented rather than
copied: `rotg`/`rotmg` return raw NaN payloads there (this crate canonicalizes),
and `rotmg` loops forever on a negative `d1` there (this crate returns).

## Status

Phases 0–4 of the spec: `sdfloat` (all four widths, all operations) and
`sdblas` (all of SoftBLAS) are implemented and green against their oracles.
`lagoon-jets` (phase 5) lives in urbit/numerics under `lagoon/nockapp/`.
