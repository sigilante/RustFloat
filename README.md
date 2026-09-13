# RustFloat

Pure-Rust IEEE 754 software floating point with explicit rounding modes,
bit-identical to Berkeley SoftFloat 3e. Built for NockApp jets of the
[urbit/numerics](https://github.com/urbit/numerics) suite, where a jet must
return exactly the bits the Hoon `++rh`/`++rs`/`++rd`/`++rq` cores return.

The `sdfloat` crate is `#![no_std]`, `#![forbid(unsafe_code)]`, has no
dependencies, allocates nothing, holds no global state, and uses no hardware
float type anywhere, tests included. Rounding mode is a parameter on every
inexact operation. Specification: [`doc/nockapp-softfloat-rs-spec.md`](doc/nockapp-softfloat-rs-spec.md).

## Layout

| Crate | Purpose |
|---|---|
| `crates/sdfloat/` | The library: `F16`, `F32`, `F64`, `F128`, the `SoftFloat` trait, `Round`. |
| `crates/sdfloat-oracle/` | Test-only FFI to vendored SoftFloat 3e C (ARM-VFPv2 specialization, as vere links). Never a runtime dependency. |

Inside `crates/sdfloat/src`, `pack.rs` is the single rounding point; each file in
`ops/` is one operation and names the SoftFloat source it replaces.

## Use

```rust
use sdfloat::{F32, Round, SoftFloat};

let a = F32(0x3f80_0000);               // 1.0
let b = F32(0x2edb_e6ff);               // 1e-10
assert_eq!(a.add(b, Round::NearEven), a);
assert_eq!(a.add(b, Round::Away), F32(0x3f80_0001));
assert_eq!(F32::MAX.mul(F32(0x4000_0000), Round::Zero), F32::MAX); // saturates
assert_eq!(F32::MAX.mul(F32(0x4000_0000), Round::Up), F32::INF);
```

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
- Overflow saturates to the largest finite value in modes that do not round
  toward the overflowing infinity (IEEE 754-2019 §7.4). Hoon `++fl` currently
  returns infinity there; see urbit/urbit#7426.
- Float to int: NaN gives 0, out of range saturates. This is SoftFloat under
  the ARM-VFPv2 specialization. The jet layer bails on those inputs instead.
- Signed zero follows IEEE §6.3: `x - x` is `+0` except under `Down`.

## Testing

```sh
cargo test --workspace                         # oracle fuzz, semantics, exhaustive f16 unary
SOFTFLOAT_ORACLE_N=2000000 cargo test --release -p sdfloat --test oracle
cargo test --release -p sdfloat --test exhaustive_f16 -- --ignored   # all 2^32 f16 pairs
scripts/no-hardware-floats.sh
```

Three layers, all run by `cargo test --workspace`:

1. **Oracle fuzz** (`tests/oracle.rs`): raw bits against SoftFloat for every
   op, width and mode on structured samples (zeros, subnormals, exponent
   boundaries, ties, near-cancellation pairs). `Away` is derived from
   SoftFloat's `round_max`/`round_min` by result sign. Exhaustive binary16
   coverage lives in `tests/exhaustive_f16.rs`; the 2^32-pair binary ops are
   `#[ignore]`d and run nightly in CI (they pass: ~5 minutes on 4 cores).
2. **Hoon vectors** (`tests/hoon_vectors.rs`, `tests/vectors/hoon_f*.txt`):
   1500 rows per width from the un-jetted `++ff` core via `urbit eval`,
   covering add/sub/mul/div/sqt/lth/lte/equ/toi/sun/san in all five modes.
   Regenerate with `URBIT=/path/to/urbit scripts/gen-hoon-vectors.py`. Rows
   that hit the Hoon overflow bug are counted and must be exactly of that
   shape; everything else must match bit for bit.
3. **Hand-checked semantics** (`tests/semantics.rs`): the conventions above.

## Status

Phases 0–3 of the spec (all four widths, all operations) are implemented and
green against the oracle. `softblas` and `lagoon-jets` (phases 4–5) live in
urbit/numerics under `lagoon/nockapp/`, not here.
