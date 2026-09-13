//! IEEE quiet comparisons on raw bits. Model: SoftFloat `f32_eq.c`,
//! `f32_lt.c`, `f32_le.c`. Any NaN operand makes every predicate false;
//! +0 and -0 compare equal.

use crate::format::{is_nan_bits, sign_of, Format};

pub fn eq<F: Format>(a: u128, b: u128) -> bool {
    if is_nan_bits::<F>(a) || is_nan_bits::<F>(b) {
        return false;
    }
    a == b || ((a | b) & F::ABS_MASK) == 0
}

pub fn lt<F: Format>(a: u128, b: u128) -> bool {
    if is_nan_bits::<F>(a) || is_nan_bits::<F>(b) {
        return false;
    }
    let (sa, sb) = (sign_of::<F>(a), sign_of::<F>(b));
    if sa != sb {
        sa && ((a | b) & F::ABS_MASK) != 0
    } else {
        a != b && (sa ^ (a < b))
    }
}

pub fn le<F: Format>(a: u128, b: u128) -> bool {
    if is_nan_bits::<F>(a) || is_nan_bits::<F>(b) {
        return false;
    }
    let (sa, sb) = (sign_of::<F>(a), sign_of::<F>(b));
    if sa != sb {
        sa || ((a | b) & F::ABS_MASK) == 0
    } else {
        a == b || (sa ^ (a < b))
    }
}
