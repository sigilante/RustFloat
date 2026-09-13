//! The single rounding point. Every inexact operation ends here.
//!
//! Conventions: `sig` carries the significand with three extra low bits
//! (guard, round, sticky). A value is `sig * 2^(e - (PREC + 2))`, so a
//! normalized `sig` has its MSB at bit `PREC + 2`.
//!
//! Why three bits suffice (the odd-sticky lemma). Let `v` be the exact
//! scaled value and `s` the integer held. Every inexact reduction in this
//! crate (`Wide::shr_jam`, the `q |= 1` in div, the `root |= 1` in sqrt)
//! leaves `s` odd with `v` strictly inside `(s - 1, s + 1)`. Note that `s`
//! may exceed `v`: when `floor(v)` is even the jam yields `floor(v) + 1`.
//! Correctness does not need `s <= v`; it needs that no rounding boundary
//! separates `s` from `v`. The only integer in that open interval is `s`,
//! which is odd, so the interval contains no multiple of 4 and no multiple
//! of 8: `v` and `s` lie strictly between the same boundaries for rounding
//! at bit 3 and strictly on the same side of the midpoint (bit 2), and
//! neither is a tie. One normalizing left shift keeps this (2s is 2 mod 4,
//! and `(2s - 2, 2s + 2)` still holds no multiple of 4); two would not.
//! Every operation therefore guarantees that an inexact `sig` reaching
//! `norm_round_pack` has its MSB at `PREC + 1` or above: add/sub because a
//! gap of at least 4 (the only inexact case) bounds cancellation to one
//! bit, mul/div/sqrt by their output bit counts. `norm_round_pack` asserts
//! it in debug builds.
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
        // An odd `sig` is an inexact (jammed) one; see the module comment.
        debug_assert!(
            d <= 1 || sig.and(F::W::ONE).is_zero(),
            "inexact sig normalized by {d} bits"
        );
        round_pack::<F>(sign, e - d as i32, sig.shl(d), r)
    } else if lz < want {
        let d = want - lz;
        round_pack::<F>(sign, e + d as i32, sig.shr_jam(d), r)
    } else {
        round_pack::<F>(sign, e, sig, r)
    }
}
