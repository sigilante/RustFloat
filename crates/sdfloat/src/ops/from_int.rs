//! Integer to float. Model: SoftFloat `i32_to_f32.c`, `ui64_to_f32.c`.
//! Zero always converts to +0.

use crate::format::{zero_bits, Format};
use crate::pack::round_pack;
use crate::round::Round;
use crate::wide::Wide;

/// `mag * 2^0`, with sign, rounded into `F`.
pub fn from_mag<F: Format>(sign: bool, mag: u128, r: Round) -> u128 {
    if mag == 0 {
        return zero_bits::<F>(false);
    }
    // Position the MSB at PREC + 2 (with a sticky bit if bits fall off).
    // The MSB index is the exponent under the pack convention.
    let msb = 127 - mag.leading_zeros();
    let target = F::PREC + 2;
    let sig = if msb > target {
        mag.shr_jam(msb - target)
    } else {
        mag << (target - msb)
    };
    round_pack::<F>(sign, msb as i32, F::W::from_u128(sig), r)
}

pub fn from_i64<F: Format>(v: i64, r: Round) -> u128 {
    from_mag::<F>(v < 0, v.unsigned_abs() as u128, r)
}
pub fn from_u64<F: Format>(v: u64, r: Round) -> u128 {
    from_mag::<F>(false, v as u128, r)
}
