//! Square root by digit recurrence (two radicand bits per step), streaming
//! zero bits below the significand so no double-width type is needed.
//! Model: SoftFloat `f32_sqrt.c` for the special cases; the core differs.

use crate::format::{nan_bits, unpack, Class, Format};
use crate::pack::norm_round_pack;
use crate::round::Round;
use crate::wide::Wide;

pub fn sqrt<F: Format>(a: u128, r: Round) -> u128 {
    let ua = unpack::<F>(a);
    let (e, sig) = match ua.class {
        Class::Nan => return nan_bits::<F>(),
        Class::Zero => return a,
        Class::Inf | Class::Finite { .. } if ua.sign => return nan_bits::<F>(),
        Class::Inf => return a,
        Class::Finite { e, sig } => (e, sig),
    };
    // value = sig * 2^x with x = e - (PREC - 1). Make x even by absorbing one
    // bit into the radicand, which then has PREC or PREC + 1 bits.
    let x = e - (F::PREC as i32 - 1);
    let (rad, x) = if x & 1 != 0 {
        (sig.shl(1), x - 1)
    } else {
        (sig, x)
    };
    // root = floor(sqrt(rad * 2^(2m))), chosen so root has PREC + 2 or
    // PREC + 3 bits: m = ceil((PREC + 3) / 2).
    let m = (F::PREC + 4) / 2;
    let pairs = (F::PREC + 2) / 2; // ceil((PREC + 1) / 2) radicand bit pairs
    let mut root = F::W::ZERO;
    let mut rem = F::W::ZERO;
    let one = F::W::ONE;
    let three = one.shl(2).wrapping_sub(one);
    for i in (0..pairs + m).rev() {
        let digit = if i >= m {
            rad.shr(2 * (i - m)).and(three)
        } else {
            F::W::ZERO
        };
        rem = rem.shl(2).or(digit);
        let trial = root.shl(2).or(one);
        root = root.shl(1);
        if rem >= trial {
            rem = rem.wrapping_sub(trial);
            root = root.or(one);
        }
    }
    if !rem.is_zero() {
        root = root.or(one);
    }
    // sqrt(value) = root * 2^(x/2 - m) = root * 2^(e_out - (PREC + 2)).
    let e_out = x / 2 - m as i32 + F::PREC as i32 + 2;
    norm_round_pack::<F>(false, e_out, root, r)
}
