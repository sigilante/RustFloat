//! The single rounding point. Every inexact operation ends here.
//!
//! Conventions: `sig` carries the significand with three extra low bits
//! (guard, round, sticky). A value is `sig * 2^(e - (PREC + 2))`, so a
//! normalized `sig` has its MSB at bit `PREC + 2`. Three guard bits with a
//! sticky are sufficient for correct rounding of every operation here,
//! provided the sticky bit reflects all discarded nonzero bits (which
//! `Wide::shr_jam` guarantees) and cancellation removes at most one leading
//! bit when any bits were discarded (true for add/sub with exponent gap >= 2).
//!
//! Model: SoftFloat `s_roundPackToF32.c` / `s_normRoundPackToF32.c`.

use crate::format::{inf_bits, max_bits, Format};
use crate::round::Round;
use crate::wide::Wide;

/// Round and pack a normalized significand (MSB at `PREC + 2`).
pub fn round_pack<F: Format>(sign: bool, e: i32, sig: F::W, r: Round) -> u128 {
    let one = F::W::ONE;
    let half = one.shl(2);
    let guard_mask = one.shl(3).wrapping_sub(one);
    let top = one.shl(F::PREC + 3);
    debug_assert!(sig >= one.shl(F::PREC + 2) && sig < top);

    let (mut e, mut sig) = (e, sig);
    if e < F::EMIN {
        // Subnormal result: shift into place, keeping a sticky bit.
        sig = sig.shr_jam((F::EMIN - e) as u32);
        e = F::EMIN;
    }
    let round_bits = sig.and(guard_mask);
    let inc = match r {
        Round::NearEven => half,
        _ if r.rounds_away(sign) => guard_mask,
        _ => F::W::ZERO,
    };
    let bumped = sig.wrapping_add(inc);
    if e > F::EMAX || (e == F::EMAX && bumped >= top) {
        // Overflow: infinity if this mode rounds toward it, else the
        // largest finite value (IEEE 754-2019 7.4).
        return if inc.is_zero() {
            max_bits::<F>(sign)
        } else {
            inf_bits::<F>(sign)
        };
    }
    let mut s = bumped.shr(3);
    if r == Round::NearEven && round_bits == half {
        s = s.and(one.not()); // tie: clear the LSB, making it even
    }
    // `field` is the biased exponent minus one; adding `s` (hidden bit
    // included) supplies the missing one and carries on round-up to a
    // power of two. At `e == EMIN` the field is zero, so a zero `s` packs a
    // signed zero and a hidden-bit `s` packs the smallest normal.
    let field = (e + F::BIAS - 1) as u128;
    let sign_bit = if sign { F::SIGN_MASK } else { 0 };
    sign_bit | ((field << F::FRAC_BITS) + s.to_u128())
}

/// Normalize a nonzero significand of any magnitude, then round and pack.
/// The value is `sig * 2^(e - (PREC + 2))` before and after normalization.
pub fn norm_round_pack<F: Format>(sign: bool, e: i32, sig: F::W, r: Round) -> u128 {
    debug_assert!(!sig.is_zero());
    let lz = sig.leading_zeros();
    let want = F::W::BITS - 1 - (F::PREC + 2);
    if lz > want {
        let d = lz - want;
        round_pack::<F>(sign, e - d as i32, sig.shl(d), r)
    } else if lz < want {
        let d = want - lz;
        round_pack::<F>(sign, e + d as i32, sig.shr_jam(d), r)
    } else {
        round_pack::<F>(sign, e, sig, r)
    }
}
