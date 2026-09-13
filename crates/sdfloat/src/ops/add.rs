//! Addition and subtraction. Model: SoftFloat `f32_add.c`, `s_addMagsF32.c`,
//! `s_subMagsF32.c`, collapsed into one routine over the unpacked form.

use crate::format::{inf_bits, nan_bits, unpack, zero_bits, Class, Format};
use crate::pack::norm_round_pack;
use crate::round::Round;
use crate::wide::Wide;

pub fn add<F: Format>(a: u128, b: u128, r: Round) -> u128 {
    let ua = unpack::<F>(a);
    let ub = unpack::<F>(b);
    let (ea, siga, eb, sigb) = match (ua.class, ub.class) {
        (Class::Nan, _) | (_, Class::Nan) => return nan_bits::<F>(),
        (Class::Inf, Class::Inf) => {
            return if ua.sign == ub.sign {
                inf_bits::<F>(ua.sign)
            } else {
                nan_bits::<F>()
            }
        }
        (Class::Inf, _) => return a,
        (_, Class::Inf) => return b,
        (Class::Zero, Class::Zero) => {
            // IEEE 754-2019 6.3: the sum of two zeros of opposite sign is +0
            // except under roundTowardNegative.
            let sign = if ua.sign == ub.sign {
                ua.sign
            } else {
                r == Round::Down
            };
            return zero_bits::<F>(sign);
        }
        (Class::Zero, _) => return b,
        (_, Class::Zero) => return a,
        (Class::Finite { e: ea, sig: sa }, Class::Finite { e: eb, sig: sb }) => (ea, sa, eb, sb),
    };
    // Three guard bits below the significand.
    let siga = siga.shl(3);
    let sigb = sigb.shl(3);
    // Order so that `x` has the larger exponent; the smaller shifts right
    // with a sticky bit. Exponent gaps of 0 or 1 lose nothing.
    let (sx, ex, sigx, sy, sigy) = if ea >= eb {
        (ua.sign, ea, siga, ub.sign, sigb.shr_jam((ea - eb) as u32))
    } else {
        (ub.sign, eb, sigb, ua.sign, siga.shr_jam((eb - ea) as u32))
    };
    if sx == sy {
        return norm_round_pack::<F>(sx, ex, sigx.wrapping_add(sigy), r);
    }
    if sigx > sigy {
        norm_round_pack::<F>(sx, ex, sigx.wrapping_sub(sigy), r)
    } else if sigy > sigx {
        norm_round_pack::<F>(sy, ex, sigy.wrapping_sub(sigx), r)
    } else {
        // Exact cancellation: +0 in every mode but roundTowardNegative.
        zero_bits::<F>(r == Round::Down)
    }
}

pub fn sub<F: Format>(a: u128, b: u128, r: Round) -> u128 {
    add::<F>(a, b ^ F::SIGN_MASK, r)
}
