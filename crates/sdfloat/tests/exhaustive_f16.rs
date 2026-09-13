//! Exhaustive binary16 checks against the oracle. Unary ops and conversions
//! cover all 65536 inputs in every mode on every run. Binary ops over all
//! 2^32 pairs are `#[ignore]`d; run them in release for the nightly job:
//!
//! ```text
//! cargo test --release -p sdfloat --test exhaustive_f16 -- --ignored
//! ```

#![allow(clippy::all)]

use sdfloat::{Format, Round, SoftFloat, F128, F16, F32, F64};
use sdfloat_oracle::{conv, f16 as orc, mode};

const MODES: [Round; 5] = [
    Round::NearEven,
    Round::Up,
    Round::Down,
    Round::Zero,
    Round::Away,
];

fn canon<F: Format>(bits: u128) -> u128 {
    if bits & F::ABS_MASK > F::INF_BITS {
        F::NAN_BITS
    } else {
        bits
    }
}

/// Oracle result under `r`; `Away` derived from the sign of the `MAX` result.
fn expect<F: Format>(r: Round, f: impl Fn(u8) -> u128) -> u128 {
    let raw = match r {
        Round::NearEven => f(mode::NEAR_EVEN),
        Round::Up => f(mode::MAX),
        Round::Down => f(mode::MIN),
        Round::Zero => f(mode::MIN_MAG),
        Round::Away => {
            let up = f(mode::MAX);
            if up & F::SIGN_MASK != 0 {
                f(mode::MIN)
            } else {
                up
            }
        }
    };
    canon::<F>(raw)
}

#[test]
fn unary_all_inputs_all_modes() {
    for a in 0..=u16::MAX {
        let x = F16(a);
        for r in MODES {
            assert_eq!(
                x.sqrt(r).0 as u128,
                expect::<F16>(r, |m| orc::orc_f16_sqrt(m, a) as u128),
                "sqrt {a:#x} {r:?}"
            );
            assert_eq!(
                x.round_to_int(r).0 as u128,
                expect::<F16>(r, |m| orc::orc_f16_round_to_int(m, a) as u128),
                "rint {a:#x} {r:?}"
            );
            let m = match r {
                Round::NearEven => mode::NEAR_EVEN,
                Round::Up => mode::MAX,
                Round::Down => mode::MIN,
                Round::Zero => mode::MIN_MAG,
                Round::Away => {
                    if a & 0x8000 != 0 {
                        mode::MIN
                    } else {
                        mode::MAX
                    }
                }
            };
            assert_eq!(x.to_i32(r), conv::f16_to_i32(m, a), "to_i32 {a:#x} {r:?}");
            assert_eq!(x.to_i64(r), conv::f16_to_i64(m, a), "to_i64 {a:#x} {r:?}");
        }
        // Exact widenings, mode-independent.
        assert_eq!(
            x.to_f32().0 as u128,
            canon::<F32>(conv::f16_to_f32(0, a) as u128),
            "to_f32 {a:#x}"
        );
        assert_eq!(
            x.to_f64().0 as u128,
            canon::<F64>(conv::f16_to_f64(0, a) as u128),
            "to_f64 {a:#x}"
        );
        assert_eq!(
            x.to_f128().0,
            canon::<F128>(conv::f16_to_f128(0, a)),
            "to_f128 {a:#x}"
        );
    }
}

#[test]
fn from_all_i32_low_range_and_edges() {
    // Every i32 whose magnitude fits in 17 bits (the interesting rounding
    // region for binary16 is |v| <= 2^16 anyway), plus the extremes.
    let mut vals: Vec<i32> = (-(1 << 17)..=(1 << 17)).collect();
    vals.extend([
        i32::MIN,
        i32::MAX,
        i32::MIN + 1,
        65504,
        65519,
        65520,
        65521,
        -65520,
    ]);
    for v in vals {
        for r in MODES {
            let neg = v < 0;
            let want = match r {
                Round::Away => conv::i32_to_f16(if neg { mode::MIN } else { mode::MAX }, v),
                Round::NearEven => conv::i32_to_f16(mode::NEAR_EVEN, v),
                Round::Up => conv::i32_to_f16(mode::MAX, v),
                Round::Down => conv::i32_to_f16(mode::MIN, v),
                Round::Zero => conv::i32_to_f16(mode::MIN_MAG, v),
            };
            assert_eq!(F16::from_i32(v, r).0, want, "from_i32 {v} {r:?}");
        }
    }
}

fn exhaustive_bin(name: &str, ours: fn(F16, F16, Round) -> F16, theirs: fn(u8, u16, u16) -> u16) {
    for a in 0..=u16::MAX {
        let x = F16(a);
        for b in 0..=u16::MAX {
            let y = F16(b);
            for r in MODES {
                let want = expect::<F16>(r, |m| theirs(m, a, b) as u16 as u128) as u16;
                let got = ours(x, y, r).0;
                assert_eq!(
                    got, want,
                    "{name} a={a:#x} b={b:#x} {r:?}: got {got:#x} want {want:#x}"
                );
            }
        }
    }
}

#[test]
#[ignore = "2^32 pairs x 5 modes; run in release for the nightly job"]
fn add_all_pairs() {
    exhaustive_bin("add", SoftFloat::add, orc::orc_f16_add)
}
#[test]
#[ignore = "2^32 pairs x 5 modes; run in release for the nightly job"]
fn sub_all_pairs() {
    exhaustive_bin("sub", SoftFloat::sub, orc::orc_f16_sub)
}
#[test]
#[ignore = "2^32 pairs x 5 modes; run in release for the nightly job"]
fn mul_all_pairs() {
    exhaustive_bin("mul", SoftFloat::mul, orc::orc_f16_mul)
}
#[test]
#[ignore = "2^32 pairs x 5 modes; run in release for the nightly job"]
fn div_all_pairs() {
    exhaustive_bin("div", SoftFloat::div, orc::orc_f16_div)
}

#[test]
fn compare_all_pairs_sampled_rows() {
    // Full 2^32 compare matrix is cheap enough per row; check every 97th
    // row against the oracle so the default run stays under a second.
    for a in (0..=u16::MAX).step_by(97) {
        let x = F16(a);
        for b in 0..=u16::MAX {
            let y = F16(b);
            assert_eq!(x.eq(y), orc::orc_f16_eq(a, b), "eq {a:#x} {b:#x}");
            assert_eq!(x.lt(y), orc::orc_f16_lt(a, b), "lt {a:#x} {b:#x}");
            assert_eq!(x.le(y), orc::orc_f16_le(a, b), "le {a:#x} {b:#x}");
        }
    }
}
