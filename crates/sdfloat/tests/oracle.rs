//! Differential test against Berkeley SoftFloat 3e (via `sdfloat-oracle`).
//!
//! For every op, width and mode, structured random operands are run through
//! both implementations and the raw bits compared (NaN results canonicalized
//! on the oracle side, since this crate only ever emits the canonical NaN).
//! `Away` has no SoftFloat mode; it is derived as `round_max` for
//! non-negative results and `round_min` for negative ones.
//!
//! Sample count per (op, width) is `SOFTFLOAT_ORACLE_N` (default 20000).

#![allow(clippy::all)]

use sdfloat::{Format, Round, SoftFloat, F128, F16, F32, F64};
use sdfloat_oracle::{conv, mode};

// ---------------------------------------------------------------- PRNG ----

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn bits128(&mut self) -> u128 {
        ((self.next() as u128) << 64) | self.next() as u128
    }
}

fn n_samples() -> usize {
    let n = std::env::var("SOFTFLOAT_ORACLE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    eprintln!("oracle: {n} samples x {} modes", MODES.len());
    n
}

// -------------------------------------------------------------- sampler ----

/// One operand, biased toward the places rounding goes wrong.
fn sample<F: Format>(rng: &mut Rng) -> u128 {
    let sign = if rng.below(2) == 0 { 0 } else { F::SIGN_MASK };
    let emax_field = (1u128 << F::EXP_BITS) - 1;
    let frac = |rng: &mut Rng| -> u128 {
        match rng.below(8) {
            0 => 0,
            1 => F::FRAC_MASK,
            2 => 1,
            3 => 1 << (F::FRAC_BITS - 1),
            4 => F::FRAC_MASK ^ 1,
            5 => rng.bits128() & F::FRAC_MASK & !((1u128 << rng.below(F::FRAC_BITS as u64)) - 1),
            _ => rng.bits128() & F::FRAC_MASK,
        }
    };
    let field = match rng.below(16) {
        0 => 0,                   // zero / subnormal
        1 => 1,                   // smallest normals
        2 => emax_field,          // inf / NaN
        3 => emax_field - 1,      // largest finite binade
        4 | 5 => F::BIAS as u128, // [1, 2)
        6 => {
            (F::BIAS as u128)
                .wrapping_add(rng.below(8) as u128)
                .wrapping_sub(4)
                & emax_field
        }
        7 => (F::BIAS as u128 + F::FRAC_BITS as u128 + rng.below(4) as u128).min(emax_field - 1), // integers
        8 => rng.below(4) as u128,                  // subnormal / tiny
        9 => emax_field - 1 - rng.below(4) as u128, // near overflow
        _ => rng.below(emax_field as u64 + 1) as u128,
    };
    let bits = (field << F::FRAC_BITS) | frac(rng);
    let bits = match rng.below(24) {
        0 => 0,
        1 => F::INF_BITS,
        2 => F::NAN_BITS,
        3 => F::NAN_BITS | (rng.bits128() & F::FRAC_MASK),
        4 => F::EXP_MASK | 1,   // NaN, low payload
        5 => 1,                 // min subnormal
        6 => F::FRAC_MASK,      // max subnormal
        7 => 1 << F::FRAC_BITS, // min normal
        8 => F::MAX_BITS,
        9 => (F::BIAS as u128) << F::FRAC_BITS,        // 1
        10 => ((F::BIAS + 1) as u128) << F::FRAC_BITS, // 2
        _ => bits,
    };
    sign | bits
}

/// A second operand correlated with the first: exact ties, cancellation,
/// and near-equal magnitudes are where add/sub/div need it.
fn sample_pair<F: Format>(rng: &mut Rng) -> (u128, u128) {
    let a = sample::<F>(rng);
    let b = match rng.below(8) {
        0 => a,
        1 => a ^ F::SIGN_MASK,
        2 => a.wrapping_add(1) & (F::SIGN_MASK | F::ABS_MASK),
        3 => a.wrapping_sub(1) & (F::SIGN_MASK | F::ABS_MASK),
        4 => {
            // same sign, exponent shifted by a few binades
            let d = rng.below(2 * F::PREC as u64 + 4) as u128;
            let e = (a & F::EXP_MASK) >> F::FRAC_BITS;
            let e2 = e.saturating_sub(d).min((1u128 << F::EXP_BITS) - 1);
            (a & !F::EXP_MASK) | (e2 << F::FRAC_BITS) | (rng.bits128() & F::FRAC_MASK)
        }
        _ => sample::<F>(rng),
    };
    (a, b)
}

// --------------------------------------------------------------- oracle ----

/// Oracle bindings for one width, over the common `u128` carrier.
trait Orc: Format {
    fn o_add(m: u8, a: u128, b: u128) -> u128;
    fn o_sub(m: u8, a: u128, b: u128) -> u128;
    fn o_mul(m: u8, a: u128, b: u128) -> u128;
    fn o_div(m: u8, a: u128, b: u128) -> u128;
    fn o_sqrt(m: u8, a: u128) -> u128;
    fn o_rint(m: u8, a: u128) -> u128;
    fn o_eq(a: u128, b: u128) -> bool;
    fn o_lt(a: u128, b: u128) -> bool;
    fn o_le(a: u128, b: u128) -> bool;
    fn o_to_i32(m: u8, a: u128) -> i32;
    fn o_to_i64(m: u8, a: u128) -> i64;
    fn o_from_i32(m: u8, v: i32) -> u128;
    fn o_from_i64(m: u8, v: i64) -> u128;
    fn o_from_u32(m: u8, v: u32) -> u128;
    fn o_from_u64(m: u8, v: u64) -> u128;
}

macro_rules! orc_scalar {
    ($F:ident, $t:ty, $m:ident, [$add:ident, $sub:ident, $mul:ident, $div:ident, $sqrt:ident, $rint:ident, $eq:ident, $lt:ident, $le:ident],
     [$to_i32:ident, $to_i64:ident, $from_i32:ident, $from_i64:ident, $from_u32:ident, $from_u64:ident]) => {
        impl Orc for $F {
            fn o_add(m: u8, a: u128, b: u128) -> u128 {
                sdfloat_oracle::$m::$add(m, a as $t, b as $t) as u128
            }
            fn o_sub(m: u8, a: u128, b: u128) -> u128 {
                sdfloat_oracle::$m::$sub(m, a as $t, b as $t) as u128
            }
            fn o_mul(m: u8, a: u128, b: u128) -> u128 {
                sdfloat_oracle::$m::$mul(m, a as $t, b as $t) as u128
            }
            fn o_div(m: u8, a: u128, b: u128) -> u128 {
                sdfloat_oracle::$m::$div(m, a as $t, b as $t) as u128
            }
            fn o_sqrt(m: u8, a: u128) -> u128 {
                sdfloat_oracle::$m::$sqrt(m, a as $t) as u128
            }
            fn o_rint(m: u8, a: u128) -> u128 {
                sdfloat_oracle::$m::$rint(m, a as $t) as u128
            }
            fn o_eq(a: u128, b: u128) -> bool {
                sdfloat_oracle::$m::$eq(a as $t, b as $t)
            }
            fn o_lt(a: u128, b: u128) -> bool {
                sdfloat_oracle::$m::$lt(a as $t, b as $t)
            }
            fn o_le(a: u128, b: u128) -> bool {
                sdfloat_oracle::$m::$le(a as $t, b as $t)
            }
            fn o_to_i32(m: u8, a: u128) -> i32 {
                conv::$to_i32(m, a as $t)
            }
            fn o_to_i64(m: u8, a: u128) -> i64 {
                conv::$to_i64(m, a as $t)
            }
            fn o_from_i32(m: u8, v: i32) -> u128 {
                conv::$from_i32(m, v) as u128
            }
            fn o_from_i64(m: u8, v: i64) -> u128 {
                conv::$from_i64(m, v) as u128
            }
            fn o_from_u32(m: u8, v: u32) -> u128 {
                conv::$from_u32(m, v) as u128
            }
            fn o_from_u64(m: u8, v: u64) -> u128 {
                conv::$from_u64(m, v) as u128
            }
        }
    };
}
orc_scalar!(
    F16,
    u16,
    f16,
    [
        orc_f16_add,
        orc_f16_sub,
        orc_f16_mul,
        orc_f16_div,
        orc_f16_sqrt,
        orc_f16_round_to_int,
        orc_f16_eq,
        orc_f16_lt,
        orc_f16_le
    ],
    [
        f16_to_i32,
        f16_to_i64,
        i32_to_f16,
        i64_to_f16,
        ui32_to_f16,
        ui64_to_f16
    ]
);
orc_scalar!(
    F32,
    u32,
    f32,
    [
        orc_f32_add,
        orc_f32_sub,
        orc_f32_mul,
        orc_f32_div,
        orc_f32_sqrt,
        orc_f32_round_to_int,
        orc_f32_eq,
        orc_f32_lt,
        orc_f32_le
    ],
    [
        f32_to_i32,
        f32_to_i64,
        i32_to_f32,
        i64_to_f32,
        ui32_to_f32,
        ui64_to_f32
    ]
);
orc_scalar!(
    F64,
    u64,
    f64,
    [
        orc_f64_add,
        orc_f64_sub,
        orc_f64_mul,
        orc_f64_div,
        orc_f64_sqrt,
        orc_f64_round_to_int,
        orc_f64_eq,
        orc_f64_lt,
        orc_f64_le
    ],
    [
        f64_to_i32,
        f64_to_i64,
        i32_to_f64,
        i64_to_f64,
        ui32_to_f64,
        ui64_to_f64
    ]
);

impl Orc for F128 {
    fn o_add(m: u8, a: u128, b: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_add(m, a, b)
    }
    fn o_sub(m: u8, a: u128, b: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_sub(m, a, b)
    }
    fn o_mul(m: u8, a: u128, b: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_mul(m, a, b)
    }
    fn o_div(m: u8, a: u128, b: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_div(m, a, b)
    }
    fn o_sqrt(m: u8, a: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_sqrt(m, a)
    }
    fn o_rint(m: u8, a: u128) -> u128 {
        sdfloat_oracle::f128::orc_f128_round_to_int(m, a)
    }
    fn o_eq(a: u128, b: u128) -> bool {
        sdfloat_oracle::f128::orc_f128_eq(a, b)
    }
    fn o_lt(a: u128, b: u128) -> bool {
        sdfloat_oracle::f128::orc_f128_lt(a, b)
    }
    fn o_le(a: u128, b: u128) -> bool {
        sdfloat_oracle::f128::orc_f128_le(a, b)
    }
    fn o_to_i32(m: u8, a: u128) -> i32 {
        conv::f128_to_i32(m, a)
    }
    fn o_to_i64(m: u8, a: u128) -> i64 {
        conv::f128_to_i64(m, a)
    }
    fn o_from_i32(m: u8, v: i32) -> u128 {
        conv::i32_to_f128(m, v)
    }
    fn o_from_i64(m: u8, v: i64) -> u128 {
        conv::i64_to_f128(m, v)
    }
    fn o_from_u32(m: u8, v: u32) -> u128 {
        conv::ui32_to_f128(m, v)
    }
    fn o_from_u64(m: u8, v: u64) -> u128 {
        conv::ui64_to_f128(m, v)
    }
}

// --------------------------------------------------------------- driver ----

const MODES: [Round; 5] = [
    Round::NearEven,
    Round::Up,
    Round::Down,
    Round::Zero,
    Round::Away,
];

fn c_mode(r: Round) -> u8 {
    match r {
        Round::NearEven => mode::NEAR_EVEN,
        Round::Up => mode::MAX,
        Round::Down => mode::MIN,
        Round::Zero => mode::MIN_MAG,
        Round::Away => unreachable!("derived"),
    }
}

/// Oracle result under `r`, deriving `Away` from the sign of the result
/// (`sign_hint` overrides that when the result's sign is not informative,
/// e.g. an integer 0).
fn expect<F: Orc>(r: Round, f: impl Fn(u8) -> u128, negative: Option<bool>) -> u128 {
    let raw = match r {
        Round::Away => {
            let up = f(mode::MAX);
            let neg = negative.unwrap_or(up & F::SIGN_MASK != 0);
            if neg {
                f(mode::MIN)
            } else {
                up
            }
        }
        _ => f(c_mode(r)),
    };
    canon::<F>(raw)
}

fn canon<F: Format>(bits: u128) -> u128 {
    if bits & F::ABS_MASK > F::INF_BITS {
        F::NAN_BITS
    } else {
        bits
    }
}

fn name<F: Format>() -> &'static str {
    match F::WIDTH {
        16 => "f16",
        32 => "f32",
        64 => "f64",
        _ => "f128",
    }
}

fn check_bin<F: Orc + SoftFloat + core::fmt::Debug>(
    op: &str,
    ours: fn(F, F, Round) -> F,
    theirs: fn(u8, u128, u128) -> u128,
) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ F::WIDTH as u64);
    for _ in 0..n_samples() {
        let (a, b) = sample_pair::<F>(&mut rng);
        for r in MODES {
            let want = expect::<F>(r, |m| theirs(m, a, b), None);
            let got = ours(F::from_raw(a), F::from_raw(b), r).raw();
            assert_eq!(
                got,
                want,
                "{} {op} {:?} a={a:#x} b={b:#x}: got {got:#x} want {want:#x}",
                name::<F>(),
                r
            );
        }
    }
}

fn check_un<F: Orc + SoftFloat + core::fmt::Debug>(
    op: &str,
    ours: fn(F, Round) -> F,
    theirs: fn(u8, u128) -> u128,
) {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ F::WIDTH as u64);
    for _ in 0..n_samples() {
        let a = sample::<F>(&mut rng);
        for r in MODES {
            let want = expect::<F>(r, |m| theirs(m, a), None);
            let got = ours(F::from_raw(a), r).raw();
            assert_eq!(
                got,
                want,
                "{} {op} {:?} a={a:#x}: got {got:#x} want {want:#x}",
                name::<F>(),
                r
            );
        }
    }
}

fn check_cmp<F: Orc + SoftFloat>() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ F::WIDTH as u64);
    for _ in 0..n_samples() {
        let (a, b) = sample_pair::<F>(&mut rng);
        let (x, y) = (F::from_raw(a), F::from_raw(b));
        assert_eq!(
            x.eq(y),
            F::o_eq(a, b),
            "{} eq a={a:#x} b={b:#x}",
            name::<F>()
        );
        assert_eq!(
            x.lt(y),
            F::o_lt(a, b),
            "{} lt a={a:#x} b={b:#x}",
            name::<F>()
        );
        assert_eq!(
            x.le(y),
            F::o_le(a, b),
            "{} le a={a:#x} b={b:#x}",
            name::<F>()
        );
    }
}

fn check_to_int<F: Orc + SoftFloat>() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF ^ F::WIDTH as u64);
    for _ in 0..n_samples() {
        let a = sample::<F>(&mut rng);
        let neg = a & F::SIGN_MASK != 0;
        for r in MODES {
            let m = match r {
                Round::Away => {
                    if neg {
                        mode::MIN
                    } else {
                        mode::MAX
                    }
                }
                _ => c_mode(r),
            };
            assert_eq!(
                F::from_raw(a).to_i32(r),
                F::o_to_i32(m, a),
                "{} to_i32 {:?} a={a:#x}",
                name::<F>(),
                r
            );
            assert_eq!(
                F::from_raw(a).to_i64(r),
                F::o_to_i64(m, a),
                "{} to_i64 {:?} a={a:#x}",
                name::<F>(),
                r
            );
        }
    }
}

fn check_from_int<F: Orc + SoftFloat>() {
    let mut rng = Rng(0xFEDC_BA98_7654_3210 ^ F::WIDTH as u64);
    for _ in 0..n_samples() {
        // Bias toward values needing rounding in narrow formats and toward the
        // signed extremes.
        let v: u64 = match rng.below(6) {
            0 => rng.next() >> rng.below(64),
            1 => (1u64 << rng.below(64)).wrapping_sub(rng.below(3)),
            2 => i64::MIN as u64,
            3 => i64::MAX as u64,
            4 => rng.below(4096),
            _ => rng.next(),
        };
        for r in MODES {
            let i = v as i64;
            let neg = i < 0;
            let want = expect::<F>(r, |m| F::o_from_i64(m, i), Some(neg));
            assert_eq!(
                F::from_i64(i, r).raw(),
                want,
                "{} from_i64 {:?} v={i}",
                name::<F>(),
                r
            );
            let want = expect::<F>(r, |m| F::o_from_u64(m, v), Some(false));
            assert_eq!(
                F::from_u64(v, r).raw(),
                want,
                "{} from_u64 {:?} v={v}",
                name::<F>(),
                r
            );
            let i32v = v as i32;
            let want = expect::<F>(r, |m| F::o_from_i32(m, i32v), Some(i32v < 0));
            assert_eq!(
                F::from_i32(i32v, r).raw(),
                want,
                "{} from_i32 {:?} v={i32v}",
                name::<F>(),
                r
            );
            let u32v = v as u32;
            let want = expect::<F>(r, |m| F::o_from_u32(m, u32v), Some(false));
            assert_eq!(
                F::from_u32(u32v, r).raw(),
                want,
                "{} from_u32 {:?} v={u32v}",
                name::<F>(),
                r
            );
        }
    }
}

macro_rules! suite {
    ($F:ident, $m:ident) => {
        mod $m {
            use super::*;
            #[test]
            fn add() {
                check_bin::<$F>("add", SoftFloat::add, $F::o_add)
            }
            #[test]
            fn sub() {
                check_bin::<$F>("sub", SoftFloat::sub, $F::o_sub)
            }
            #[test]
            fn mul() {
                check_bin::<$F>("mul", SoftFloat::mul, $F::o_mul)
            }
            #[test]
            fn div() {
                check_bin::<$F>("div", SoftFloat::div, $F::o_div)
            }
            #[test]
            fn sqrt() {
                check_un::<$F>("sqrt", SoftFloat::sqrt, $F::o_sqrt)
            }
            #[test]
            fn round_to_int() {
                check_un::<$F>("round_to_int", SoftFloat::round_to_int, $F::o_rint)
            }
            #[test]
            fn compare() {
                check_cmp::<$F>()
            }
            #[test]
            fn to_int() {
                check_to_int::<$F>()
            }
            #[test]
            fn from_int() {
                check_from_int::<$F>()
            }
        }
    };
}
suite!(F16, f16);
suite!(F32, f32);
suite!(F64, f64);
suite!(F128, f128);

// ---------------------------------------------------------- conversions ----

fn check_cvt<A: Orc + SoftFloat, B: Orc + SoftFloat>(theirs: fn(u8, u128) -> u128) {
    let mut rng = Rng(0xA5A5_5A5A_C3C3_3C3C ^ (A::WIDTH * 131 + B::WIDTH) as u64);
    for _ in 0..n_samples() {
        let a = sample::<A>(&mut rng);
        for r in MODES {
            let want = expect::<B>(r, |m| theirs(m, a), None);
            let got: B = sdfloat::convert(A::from_raw(a), r);
            assert_eq!(
                got.raw(),
                want,
                "{}->{} {:?} a={a:#x}: got {:#x} want {want:#x}",
                name::<A>(),
                name::<B>(),
                r,
                got.raw()
            );
        }
    }
}

mod cvt {
    use super::*;
    #[test]
    fn f16_f32() {
        check_cvt::<F16, F32>(|m, a| conv::f16_to_f32(m, a as u16) as u128)
    }
    #[test]
    fn f16_f64() {
        check_cvt::<F16, F64>(|m, a| conv::f16_to_f64(m, a as u16) as u128)
    }
    #[test]
    fn f16_f128() {
        check_cvt::<F16, F128>(|m, a| conv::f16_to_f128(m, a as u16))
    }
    #[test]
    fn f32_f16() {
        check_cvt::<F32, F16>(|m, a| conv::f32_to_f16(m, a as u32) as u128)
    }
    #[test]
    fn f32_f64() {
        check_cvt::<F32, F64>(|m, a| conv::f32_to_f64(m, a as u32) as u128)
    }
    #[test]
    fn f32_f128() {
        check_cvt::<F32, F128>(|m, a| conv::f32_to_f128(m, a as u32))
    }
    #[test]
    fn f64_f16() {
        check_cvt::<F64, F16>(|m, a| conv::f64_to_f16(m, a as u64) as u128)
    }
    #[test]
    fn f64_f32() {
        check_cvt::<F64, F32>(|m, a| conv::f64_to_f32(m, a as u64) as u128)
    }
    #[test]
    fn f64_f128() {
        check_cvt::<F64, F128>(|m, a| conv::f64_to_f128(m, a as u64))
    }
    #[test]
    fn f128_f16() {
        check_cvt::<F128, F16>(|m, a| conv::f128_to_f16(m, a) as u128)
    }
    #[test]
    fn f128_f32() {
        check_cvt::<F128, F32>(|m, a| conv::f128_to_f32(m, a) as u128)
    }
    #[test]
    fn f128_f64() {
        check_cvt::<F128, F64>(|m, a| conv::f128_to_f64(m, a) as u128)
    }
}
