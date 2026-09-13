//! Multiplication. Model: SoftFloat `f32_mul.c`.

use crate::format::{inf_bits, nan_bits, unpack, zero_bits, Class, Format};
use crate::pack::norm_round_pack;
use crate::round::Round;
use crate::wide::Wide;

pub fn mul<F: Format>(a: u128, b: u128, r: Round) -> u128 {
    let ua = unpack::<F>(a);
    let ub = unpack::<F>(b);
    let sign = ua.sign ^ ub.sign;
    let (ea, sa, eb, sb) = match (ua.class, ub.class) {
        (Class::Nan, _) | (_, Class::Nan) => return nan_bits::<F>(),
        (Class::Inf, Class::Zero) | (Class::Zero, Class::Inf) => return nan_bits::<F>(),
        (Class::Inf, _) | (_, Class::Inf) => return inf_bits::<F>(sign),
        (Class::Zero, _) | (_, Class::Zero) => return zero_bits::<F>(sign),
        (Class::Finite { e: ea, sig: sa }, Class::Finite { e: eb, sig: sb }) => (ea, sa, eb, sb),
    };
    // Both significands have their MSB at PREC - 1, so the product has
    // 2*PREC - 1 or 2*PREC bits and equals prod * 2^(ea + eb - 2*(PREC-1)).
    // Reduce to at most PREC + 3 bits with a sticky; the exponent of the
    // reduced form under the pack convention is ea + eb + 1.
    let (hi, lo) = sa.widening_mul(sb);
    let sig = F::W::shr_jam2(hi, lo, F::PREC - 3);
    norm_round_pack::<F>(sign, ea + eb + 1, sig, r)
}
