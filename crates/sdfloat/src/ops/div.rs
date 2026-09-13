//! Division. Model: SoftFloat `f32_div.c`, but by restoring long division
//! for every width rather than SoftFloat's reciprocal approximations; the
//! quotient is exact to PREC + 2 bits with a sticky remainder.

use crate::format::{inf_bits, nan_bits, unpack, zero_bits, Class, Format};
use crate::pack::norm_round_pack;
use crate::round::Round;
use crate::wide::Wide;

pub fn div<F: Format>(a: u128, b: u128, r: Round) -> u128 {
    let ua = unpack::<F>(a);
    let ub = unpack::<F>(b);
    let sign = ua.sign ^ ub.sign;
    let (ea, sa, eb, sb) = match (ua.class, ub.class) {
        (Class::Nan, _) | (_, Class::Nan) => return nan_bits::<F>(),
        (Class::Inf, Class::Inf) | (Class::Zero, Class::Zero) => return nan_bits::<F>(),
        (Class::Inf, _) => return inf_bits::<F>(sign),
        (_, Class::Inf) => return zero_bits::<F>(sign),
        (_, Class::Zero) => return inf_bits::<F>(sign),
        (Class::Zero, _) => return zero_bits::<F>(sign),
        (Class::Finite { e: ea, sig: sa }, Class::Finite { e: eb, sig: sb }) => (ea, sa, eb, sb),
    };
    // sa/sb lies in (1/2, 2). Produce q = floor(sa * 2^k / sb) with
    // k = PREC + 2, so q has PREC + 2 or PREC + 3 bits. Then
    // value = q * 2^(ea - eb - k) = q * 2^(e - (PREC + 2)) with e = ea - eb.
    let (mut q, mut rem) = sa.div_rem(sb);
    for _ in 0..F::PREC + 2 {
        rem = rem.shl(1);
        q = q.shl(1);
        if rem >= sb {
            rem = rem.wrapping_sub(sb);
            q = q.or(F::W::ONE);
        }
    }
    if !rem.is_zero() {
        q = q.or(F::W::ONE);
    }
    norm_round_pack::<F>(sign, ea - eb, q, r)
}
