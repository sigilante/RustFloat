//! Static description of an IEEE 754 binary interchange format and the
//! unpacked form the operations work on.

use crate::wide::Wide;

/// An IEEE 754 binary format: field widths plus the raw-bits container.
///
/// All generic code moves bit patterns around as `u128` (wide enough for
/// binary128) and does significand arithmetic in `Self::W`.
pub trait Format: Copy + Eq {
    /// Raw storage: `u16`, `u32`, `u64`, or `u128`.
    type Bits: Copy + Eq + core::fmt::Debug;
    /// Significand arithmetic width; must hold `PREC + 3` bits.
    type W: Wide;

    /// Exponent field width.
    const EXP_BITS: u32;
    /// Fraction field width (excludes the hidden bit).
    const FRAC_BITS: u32;

    /// Wrap raw bits.
    fn from_bits(b: Self::Bits) -> Self;
    /// Unwrap to raw bits.
    fn to_bits(self) -> Self::Bits;
    /// `NAN_BITS` as a value (the trait cannot build it in a const context).
    const NAN_VALUE: Self;
    /// `INF_BITS` as a value.
    const INF_VALUE: Self;
    /// `+0` as a value.
    const ZERO_VALUE: Self;
    /// `1.0` as a value.
    const ONE_VALUE: Self;
    /// `MAX_BITS` as a value.
    const MAX_VALUE: Self;
    /// Zero-extend raw bits to the common `u128` carrier.
    fn bits_to_u128(b: Self::Bits) -> u128;
    /// Truncating; the caller supplies a value that fits.
    fn bits_from_u128(v: u128) -> Self::Bits;

    /// Total width in bits.
    const WIDTH: u32 = 1 + Self::EXP_BITS + Self::FRAC_BITS;
    /// Precision `p`: significand bits including the hidden bit.
    const PREC: u32 = Self::FRAC_BITS + 1;
    /// Exponent bias.
    const BIAS: i32 = (1 << (Self::EXP_BITS - 1)) - 1;
    /// Largest unbiased exponent of a finite value.
    const EMAX: i32 = Self::BIAS;
    /// Smallest unbiased exponent of a normal value.
    const EMIN: i32 = 1 - Self::BIAS;

    /// Sign bit.
    const SIGN_MASK: u128 = 1 << (Self::WIDTH - 1);
    /// Exponent field.
    const EXP_MASK: u128 = ((1 << Self::EXP_BITS) - 1) << Self::FRAC_BITS;
    /// Fraction field.
    const FRAC_MASK: u128 = (1 << Self::FRAC_BITS) - 1;
    /// All bits below the sign.
    const ABS_MASK: u128 = Self::EXP_MASK | Self::FRAC_MASK;
    /// The one NaN this library ever produces: sign 0, exponent all ones,
    /// significand MSB set, everything else zero. Matches Hoon `++ff` `bif`
    /// and vere's `HALFNAN`/`SINGNAN`/`DOUBNAN`/`QUADNAN`.
    const NAN_BITS: u128 = Self::EXP_MASK | (1 << (Self::FRAC_BITS - 1));
    /// Positive infinity.
    const INF_BITS: u128 = Self::EXP_MASK;
    /// `1.0`: biased exponent `BIAS`, zero fraction.
    const ONE_BITS: u128 = (Self::BIAS as u128) << Self::FRAC_BITS;
    /// Largest finite magnitude.
    const MAX_BITS: u128 = Self::EXP_MASK - 1;

    /// Raw bits as `u128`.
    #[inline]
    fn raw(self) -> u128 {
        Self::bits_to_u128(self.to_bits())
    }
    /// From raw bits held in a `u128`.
    #[inline]
    fn from_raw(v: u128) -> Self {
        Self::from_bits(Self::bits_from_u128(v))
    }
}

/// Classification of a bit pattern with the significand normalized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class<W> {
    Nan,
    Inf,
    Zero,
    /// `sig` has its MSB at bit `PREC - 1` (hidden bit explicit); the value
    /// is `sig * 2^(e - (PREC - 1))`. Subnormal inputs are normalized here,
    /// so `e` may lie below `EMIN`.
    Finite {
        e: i32,
        sig: W,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Unpacked<W> {
    pub sign: bool,
    pub class: Class<W>,
}

#[inline]
pub fn sign_of<F: Format>(bits: u128) -> bool {
    bits & F::SIGN_MASK != 0
}

#[inline]
pub fn is_nan_bits<F: Format>(bits: u128) -> bool {
    bits & F::ABS_MASK > F::INF_BITS
}

pub fn unpack<F: Format>(bits: u128) -> Unpacked<F::W> {
    let sign = sign_of::<F>(bits);
    let exp_field = ((bits & F::EXP_MASK) >> F::FRAC_BITS) as i32;
    let frac = bits & F::FRAC_MASK;
    let class = if exp_field == (1 << F::EXP_BITS) - 1 {
        if frac == 0 {
            Class::Inf
        } else {
            Class::Nan
        }
    } else if exp_field == 0 {
        if frac == 0 {
            Class::Zero
        } else {
            // Subnormal: normalize so the MSB sits at PREC - 1.
            let sig = F::W::from_u128(frac);
            let shift = sig.leading_zeros() - (F::W::BITS - F::PREC);
            Class::Finite {
                e: F::EMIN - shift as i32,
                sig: sig.shl(shift),
            }
        }
    } else {
        let sig = F::W::from_u128(frac | (1 << F::FRAC_BITS));
        Class::Finite {
            e: exp_field - F::BIAS,
            sig,
        }
    };
    Unpacked { sign, class }
}

#[inline]
pub fn nan_bits<F: Format>() -> u128 {
    F::NAN_BITS
}
#[inline]
pub fn inf_bits<F: Format>(sign: bool) -> u128 {
    F::INF_BITS | if sign { F::SIGN_MASK } else { 0 }
}
#[inline]
pub fn zero_bits<F: Format>(sign: bool) -> u128 {
    if sign {
        F::SIGN_MASK
    } else {
        0
    }
}
#[inline]
pub fn max_bits<F: Format>(sign: bool) -> u128 {
    F::MAX_BITS | if sign { F::SIGN_MASK } else { 0 }
}
