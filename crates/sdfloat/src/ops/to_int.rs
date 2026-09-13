//! Float to signed integer, rounding per mode. Model: SoftFloat
//! `f32_to_i32.c` built with the ARM-VFPv2 specialization, which is what
//! vere links: NaN converts to 0, and out-of-range values saturate to
//! `MAX`/`MIN` of the target. Callers that need a crash on those inputs
//! (the jets) must test `is_nan`/range themselves.

use crate::format::{unpack, Class, Format};
use crate::round::Round;
use crate::wide::Wide;

/// Round `|a|` to an integer magnitude. `None` means the magnitude is at
/// least 2^`width` (out of any signed range of that width).
fn magnitude<F: Format>(a: u128, r: Round, width: u32) -> Option<(bool, u128)> {
    let ua = unpack::<F>(a);
    let sign = ua.sign;
    match ua.class {
        Class::Nan => return Some((false, 0)),
        Class::Zero => return Some((sign, 0)),
        Class::Inf => return None,
        Class::Finite { .. } => {}
    }
    let Class::Finite { e, sig } = ua.class else {
        unreachable!()
    };
    if e >= width as i32 {
        return None;
    }
    let p1 = F::PREC as i32 - 1;
    let sig = sig.to_u128();
    let mag = if e < 0 {
        let up = match r {
            Round::NearEven => e == -1 && sig != 1u128 << (F::PREC - 1),
            _ => r.rounds_away(sign),
        };
        up as u128
    } else if e >= p1 {
        sig << (e - p1) as u32
    } else {
        let n = (p1 - e) as u32;
        let q = sig >> n;
        let rem = sig & ((1u128 << n) - 1);
        let half = 1u128 << (n - 1);
        let up = match r {
            Round::NearEven => rem > half || (rem == half && q & 1 == 1),
            _ => rem != 0 && r.rounds_away(sign),
        };
        q + up as u128
    };
    Some((sign, mag))
}

pub fn to_i32<F: Format>(a: u128, r: Round) -> i32 {
    match magnitude::<F>(a, r, 32) {
        None => sat32(a & F::SIGN_MASK != 0),
        Some((sign, m)) => {
            if sign {
                if m > 1u128 << 31 {
                    i32::MIN
                } else {
                    (m as i64).wrapping_neg() as i32
                }
            } else if m > i32::MAX as u128 {
                i32::MAX
            } else {
                m as i32
            }
        }
    }
}

pub fn to_i64<F: Format>(a: u128, r: Round) -> i64 {
    match magnitude::<F>(a, r, 64) {
        None => sat64(a & F::SIGN_MASK != 0),
        Some((sign, m)) => {
            if sign {
                if m > 1u128 << 63 {
                    i64::MIN
                } else {
                    (m as u64).wrapping_neg() as i64
                }
            } else if m > i64::MAX as u128 {
                i64::MAX
            } else {
                m as i64
            }
        }
    }
}

#[inline]
fn sat32(negative: bool) -> i32 {
    if negative {
        i32::MIN
    } else {
        i32::MAX
    }
}
#[inline]
fn sat64(negative: bool) -> i64 {
    if negative {
        i64::MIN
    } else {
        i64::MAX
    }
}
