//! Ground truth from Hoon: vectors produced by `scripts/gen-hoon-vectors.py`
//! from the un-jetted `++ff` core through `urbit eval`. This is the only
//! layer that can catch a SoftFloat-vs-Hoon divergence.
//!
//! Arithmetic must match bit for bit, overflow included: Hoon `++fl`
//! overflows to infinity in every mode (urbit/urbit#7426) and so does this
//! crate. Rows where IEEE would saturate instead are counted to show the
//! vectors exercise that case.
//!
//! Known divergences, asserted rather than ignored:
//! * Hoon has no float->float round-to-int; `toi` yields an integer and
//!   `san` of it loses the sign of zero. `round_to_int` is compared modulo
//!   the sign of a zero result.
//! * `toi` on NaN/inf is `~`; this crate's `to_i64` returns 0/saturates
//!   there, so those rows are skipped for `to_i64`.

#![allow(clippy::all)]

use sdfloat::{Format, Round, SoftFloat, F128, F16, F32, F64};

fn mode(c: &str) -> Round {
    match c {
        "n" => Round::NearEven,
        "u" => Round::Up,
        "d" => Round::Down,
        "z" => Round::Zero,
        "a" => Round::Away,
        _ => panic!("bad mode {c}"),
    }
}

fn hex(s: &str) -> u128 {
    u128::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
}

/// Hoon `@sd` text: `--5` is +5, `-5` is -5. Returns `(negative, magnitude)`;
/// `None` for `none` and for `big+`/`big-` (|toi| > 2^127).
fn hoon_int(s: &str) -> Option<(bool, u128)> {
    if s == "none" || s.starts_with("big") {
        return None;
    }
    let (neg, digits) = if let Some(d) = s.strip_prefix("--") {
        (false, d)
    } else {
        (true, &s[1..])
    };
    Some((neg, digits.parse().unwrap()))
}

/// Is `hoon` an infinity under a mode that does not round toward it? From
/// finite operands, not dividing by zero, that is an overflow where IEEE
/// would give MAX instead.
fn is_inf_toward_zero<F: Format>(hoon: u128, r: Round) -> bool {
    let sign = hoon & F::SIGN_MASK != 0;
    let toward_zero = match r {
        Round::Zero => true,
        Round::Down => !sign,
        Round::Up => sign,
        Round::NearEven | Round::Away => false,
    };
    hoon & F::ABS_MASK == F::INF_BITS && toward_zero
}

struct Stats {
    rows: usize,
    inf_toward_zero: usize,
    toi_skipped: usize,
}

fn run<F: Format + SoftFloat + core::fmt::Debug>(text: &str, name: &str) -> Stats {
    let mut st = Stats {
        rows: 0,
        inf_toward_zero: 0,
        toi_skipped: 0,
    };
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 15, "{name}: bad row {line}");
        let r = mode(f[0]);
        let (a, b, n) = (hex(f[1]), hex(f[2]), hex(f[3]));
        let (x, y) = (F::from_raw(a), F::from_raw(b));
        st.rows += 1;

        let arith = |st: &mut Stats, op: &str, finite: bool, ours: F, hoon: u128| {
            let ours = ours.raw();
            assert_eq!(
                ours, hoon,
                "{name} {op} {r:?} a={a:#x} b={b:#x} n={n:#x}: ours {ours:#x} hoon {hoon:#x}"
            );
            if finite && is_inf_toward_zero::<F>(hoon, r) {
                st.inf_toward_zero += 1;
            }
        };
        let fin = !x.is_inf() && !x.is_nan() && !y.is_inf() && !y.is_nan();
        arith(&mut st, "add", fin, x.add(y, r), hex(f[4]));
        arith(&mut st, "sub", fin, x.sub(y, r), hex(f[5]));
        arith(&mut st, "mul", fin, x.mul(y, r), hex(f[6]));
        arith(&mut st, "div", fin && !y.is_zero(), x.div(y, r), hex(f[7]));
        arith(&mut st, "sqt", fin, x.sqrt(r), hex(f[8]));

        assert_eq!(x.lt(y), f[9] == "1", "{name} lth a={a:#x} b={b:#x}");
        assert_eq!(x.le(y), f[10] == "1", "{name} lte a={a:#x} b={b:#x}");
        assert_eq!(x.eq(y), f[11] == "1", "{name} equ a={a:#x} b={b:#x}");

        match hoon_int(f[12]) {
            None if f[12] == "none" => {
                assert!(
                    x.is_nan() || x.is_inf(),
                    "{name} toi none for finite a={a:#x}"
                );
                st.toi_skipped += 1;
            }
            None => {
                // |toi| > 2^127 ("big+"/"big-"): integral already, and
                // to_i64 must saturate in the direction of the sign.
                assert_eq!(x.round_to_int(r), x, "{name} rint huge a={a:#x}");
                let neg = f[12] == "big-";
                assert_eq!(neg, a & F::SIGN_MASK != 0, "{name} big sign a={a:#x}");
                assert_eq!(
                    x.to_i64(r),
                    if neg { i64::MIN } else { i64::MAX },
                    "{name} to_i64 huge a={a:#x}"
                );
            }
            Some((neg, mag)) => {
                let fits = if neg {
                    mag <= 1u128 << 63
                } else {
                    mag <= i64::MAX as u128
                };
                if fits {
                    let v64 = if neg {
                        (mag as u64).wrapping_neg() as i64
                    } else {
                        mag as i64
                    };
                    assert_eq!(x.to_i64(r), v64, "{name} to_i64 {r:?} a={a:#x}");
                    let ri = x.round_to_int(r);
                    if v64 == 0 {
                        // Hoon's san(toi) cannot carry a zero's sign.
                        assert!(ri.is_zero(), "{name} rint zero {r:?} a={a:#x}: {ri:?}");
                    } else {
                        assert_eq!(
                            ri,
                            F::from_i64(v64, Round::NearEven),
                            "{name} rint {r:?} a={a:#x}"
                        );
                    }
                } else {
                    assert_eq!(
                        x.to_i64(r),
                        if neg { i64::MIN } else { i64::MAX },
                        "{name} to_i64 sat a={a:#x}"
                    );
                    assert_eq!(x.round_to_int(r), x, "{name} rint integral a={a:#x}");
                }
            }
        }

        // Integer conversions can overflow too (f16 from a 63-bit integer).
        arith(&mut st, "sun", true, F::from_u64(n as u64, r), hex(f[13]));
        arith(
            &mut st,
            "san",
            true,
            F::from_i64(-(n as i64), r),
            hex(f[14]),
        );
    }
    st
}

macro_rules! vectors {
    ($name:ident, $F:ident, $file:literal) => {
        #[test]
        fn $name() {
            let st = run::<$F>(include_str!($file), stringify!($F));
            eprintln!(
                "{}: {} rows, {} infinities where IEEE saturates, {} toi rows skipped",
                stringify!($F),
                st.rows,
                st.inf_toward_zero,
                st.toi_skipped
            );
            assert!(st.rows >= 1000, "vector file too small");
        }
    };
}
vectors!(f16, F16, "vectors/hoon_f16.txt");
vectors!(f32, F32, "vectors/hoon_f32.txt");
vectors!(f64, F64, "vectors/hoon_f64.txt");
vectors!(f128, F128, "vectors/hoon_f128.txt");
