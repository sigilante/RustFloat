//! Round to an integral value in the same format. Model: SoftFloat
//! `f32_roundToInt.c`. NaN canonicalizes; infinities and zeros pass through;
//! the sign of a zero result is the sign of the input (IEEE 754-2019 5.9).

use crate::format::{nan_bits, unpack, zero_bits, Class, Format};
use crate::round::Round;
use crate::wide::Wide;

pub fn round_to_int<F: Format>(a: u128, r: Round) -> u128 {
    let ua = unpack::<F>(a);
    let (e, sig) = match ua.class {
        Class::Nan => return nan_bits::<F>(),
        Class::Inf | Class::Zero => return a,
        Class::Finite { e, sig } => (e, sig),
    };
    let p1 = F::PREC as i32 - 1;
    if e >= p1 {
        return a; // already integral
    }
    let one_bits = ((F::BIAS as u128) << F::FRAC_BITS) | zero_bits::<F>(ua.sign);
    if e < 0 {
        // |a| < 1: the result is 0 or 1 in magnitude.
        let up = match r {
            Round::NearEven => e == -1 && sig != F::W::ONE.shl(F::PREC - 1),
            _ => r.rounds_away(ua.sign),
        };
        return if up {
            one_bits
        } else {
            zero_bits::<F>(ua.sign)
        };
    }
    // Discard n = p1 - e fraction bits.
    let n = (p1 - e) as u32;
    let q = sig.shr(n);
    let rem = sig.low_bits(n);
    let half = F::W::ONE.shl(n - 1);
    let up = match r {
        Round::NearEven => rem > half || (rem == half && q.and(F::W::ONE) == F::W::ONE),
        _ => !rem.is_zero() && r.rounds_away(ua.sign),
    };
    let q = if up { q.wrapping_add(F::W::ONE) } else { q };
    // q < 2^(PREC - n) + 1; if it carried to a power of two the exponent
    // grows by one. Pack exactly: the result is normal (e >= 0 >= EMIN).
    let (e, q) = if q == F::W::ONE.shl(F::PREC - n) {
        (e + 1, q.shr(1))
    } else {
        (e, q)
    };
    let sig = q.shl(n).to_u128() & F::FRAC_MASK;
    zero_bits::<F>(ua.sign) | (((e + F::BIAS) as u128) << F::FRAC_BITS) | sig
}
