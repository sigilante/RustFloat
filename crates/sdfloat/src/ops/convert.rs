//! Conversion between formats. Model: SoftFloat `f32_to_f16.c`,
//! `f16_to_f32.c`, and the rest. One routine serves every pair: unpack the
//! source, re-express the significand under the destination's pack
//! convention (with a sticky bit when narrowing), round and pack.
//! Widenings are exact, so the mode is irrelevant for them.

use crate::format::{inf_bits, nan_bits, unpack, zero_bits, Class, Format};
use crate::pack::round_pack;
use crate::round::Round;
use crate::wide::Wide;

pub fn convert<A: Format, B: Format>(a: u128, r: Round) -> u128 {
    let ua = unpack::<A>(a);
    let (e, sig) = match ua.class {
        Class::Nan => return nan_bits::<B>(),
        Class::Inf => return inf_bits::<B>(ua.sign),
        Class::Zero => return zero_bits::<B>(ua.sign),
        Class::Finite { e, sig } => (e, sig.to_u128()),
    };
    // Source MSB is at A::PREC - 1; the destination wants it at B::PREC + 2.
    let from = A::PREC - 1;
    let to = B::PREC + 2;
    let sig = if to >= from {
        sig << (to - from)
    } else {
        sig.shr_jam(from - to)
    };
    round_pack::<B>(ua.sign, e, B::W::from_u128(sig), r)
}
