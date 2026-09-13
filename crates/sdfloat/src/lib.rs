//! Software IEEE 754 binary floating point with explicit rounding modes.
//!
//! A pure-Rust analogue of Berkeley SoftFloat 3e for the four interchange
//! formats [`F16`], [`F32`], [`F64`], [`F128`], written so that Rust jets for
//! NockVM produce exactly the bits the Hoon `++rh`/`++rs`/`++rd`/`++rq`
//! cores produce. Every result is correctly rounded in each of the five
//! [`Round`] modes; every NaN produced is the canonical quiet NaN of its
//! width; no hardware float type appears anywhere in the crate.
//!
//! Semantics that go beyond IEEE 754 and are fixed here:
//!
//! * **NaN**: any NaN operand yields the canonical NaN (`F::NAN`), payloads
//!   are never propagated, and there is no signaling/quiet distinction.
//! * **Overflow** saturates to the largest finite value in modes that do not
//!   round toward the overflowing infinity (`Zero`; `Down` for positive;
//!   `Up` for negative), per IEEE 754-2019 7.4.
//! * **Float to int** on NaN returns 0 and on out-of-range input saturates,
//!   matching SoftFloat built with the ARM-VFPv2 specialization that vere
//!   links. Exception flags do not exist in this crate.
//! * [`Round::Away`] is Hoon's `%a`: directed rounding away from zero.
//!
//! Non-goals: exception flags, signaling NaNs, extF80, fused multiply-add,
//! decimal conversion.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::all)]

mod format;
mod ops;
mod pack;
mod round;
mod types;
mod wide;

pub use format::Format;
pub use round::Round;
pub use types::{convert, F128, F16, F32, F64};

/// The operations the numerics jets need, uniformly over the four widths.
///
/// Implemented for every [`Format`]. Method names follow the spec and the
/// Hoon arms; `eq`/`lt`/`le` are IEEE quiet comparisons, not `PartialEq`
/// (the derived `PartialEq` compares bits, so `-0 != +0` there and
/// `NaN == NaN`).
pub trait SoftFloat: Copy + Eq {
    /// Raw storage type.
    type Bits: Copy + Eq;
    /// Exponent field width.
    const EXP_BITS: u32;
    /// Fraction field width (significand bits excluding the hidden bit).
    const SIG_BITS: u32;
    /// The canonical NaN.
    const NAN: Self;
    /// Positive infinity.
    const INF: Self;
    /// Positive zero.
    const ZERO: Self;
    /// Largest finite value.
    const MAX: Self;

    /// From raw bits.
    fn from_bits(b: Self::Bits) -> Self;
    /// To raw bits.
    fn to_bits(self) -> Self::Bits;

    /// `self + o`.
    fn add(self, o: Self, r: Round) -> Self;
    /// `self - o`.
    fn sub(self, o: Self, r: Round) -> Self;
    /// `self * o`.
    fn mul(self, o: Self, r: Round) -> Self;
    /// `self / o`.
    fn div(self, o: Self, r: Round) -> Self;
    /// Square root; negative nonzero input gives NaN, `-0` gives `-0`.
    fn sqrt(self, r: Round) -> Self;
    /// Round to an integral value in the same format.
    fn round_to_int(self, r: Round) -> Self;

    /// IEEE equality: `-0 == +0`, NaN is unequal to everything.
    fn eq(self, o: Self) -> bool;
    /// IEEE less-than; false if either side is NaN.
    fn lt(self, o: Self) -> bool;
    /// IEEE less-or-equal; false if either side is NaN.
    fn le(self, o: Self) -> bool;

    /// To `i32`, rounding per `r`. NaN gives 0; out of range saturates.
    fn to_i32(self, r: Round) -> i32;
    /// To `i64`, rounding per `r`. NaN gives 0; out of range saturates.
    fn to_i64(self, r: Round) -> i64;
    /// From `i32`, rounding per `r` where the width cannot hold it.
    fn from_i32(v: i32, r: Round) -> Self;
    /// From `i64`, rounding per `r`.
    fn from_i64(v: i64, r: Round) -> Self;
    /// From `u32`, rounding per `r`.
    fn from_u32(v: u32, r: Round) -> Self;
    /// From `u64`, rounding per `r`.
    fn from_u64(v: u64, r: Round) -> Self;

    /// Any NaN, canonical or not.
    fn is_nan(self) -> bool;
    /// Either infinity.
    fn is_inf(self) -> bool;
    /// Either zero.
    fn is_zero(self) -> bool;
    /// Clear the sign bit.
    fn abs(self) -> Self;
    /// Flip the sign bit.
    fn neg(self) -> Self;
}

impl<F: Format> SoftFloat for F {
    type Bits = F::Bits;
    const EXP_BITS: u32 = F::EXP_BITS;
    const SIG_BITS: u32 = F::FRAC_BITS;
    const NAN: Self = F::NAN_VALUE;
    const INF: Self = F::INF_VALUE;
    const ZERO: Self = F::ZERO_VALUE;
    const MAX: Self = F::MAX_VALUE;

    #[inline]
    fn from_bits(b: F::Bits) -> Self {
        Format::from_bits(b)
    }
    #[inline]
    fn to_bits(self) -> F::Bits {
        Format::to_bits(self)
    }

    #[inline]
    fn add(self, o: Self, r: Round) -> Self {
        F::from_raw(ops::add::add::<F>(self.raw(), o.raw(), r))
    }
    #[inline]
    fn sub(self, o: Self, r: Round) -> Self {
        F::from_raw(ops::add::sub::<F>(self.raw(), o.raw(), r))
    }
    #[inline]
    fn mul(self, o: Self, r: Round) -> Self {
        F::from_raw(ops::mul::mul::<F>(self.raw(), o.raw(), r))
    }
    #[inline]
    fn div(self, o: Self, r: Round) -> Self {
        F::from_raw(ops::div::div::<F>(self.raw(), o.raw(), r))
    }
    #[inline]
    fn sqrt(self, r: Round) -> Self {
        F::from_raw(ops::sqrt::sqrt::<F>(self.raw(), r))
    }
    #[inline]
    fn round_to_int(self, r: Round) -> Self {
        F::from_raw(ops::round_to_int::round_to_int::<F>(self.raw(), r))
    }

    #[inline]
    fn eq(self, o: Self) -> bool {
        ops::compare::eq::<F>(self.raw(), o.raw())
    }
    #[inline]
    fn lt(self, o: Self) -> bool {
        ops::compare::lt::<F>(self.raw(), o.raw())
    }
    #[inline]
    fn le(self, o: Self) -> bool {
        ops::compare::le::<F>(self.raw(), o.raw())
    }

    #[inline]
    fn to_i32(self, r: Round) -> i32 {
        ops::to_int::to_i32::<F>(self.raw(), r)
    }
    #[inline]
    fn to_i64(self, r: Round) -> i64 {
        ops::to_int::to_i64::<F>(self.raw(), r)
    }
    #[inline]
    fn from_i32(v: i32, r: Round) -> Self {
        F::from_raw(ops::from_int::from_i64::<F>(v as i64, r))
    }
    #[inline]
    fn from_i64(v: i64, r: Round) -> Self {
        F::from_raw(ops::from_int::from_i64::<F>(v, r))
    }
    #[inline]
    fn from_u32(v: u32, r: Round) -> Self {
        F::from_raw(ops::from_int::from_u64::<F>(v as u64, r))
    }
    #[inline]
    fn from_u64(v: u64, r: Round) -> Self {
        F::from_raw(ops::from_int::from_u64::<F>(v, r))
    }

    #[inline]
    fn is_nan(self) -> bool {
        format::is_nan_bits::<F>(self.raw())
    }
    #[inline]
    fn is_inf(self) -> bool {
        self.raw() & F::ABS_MASK == F::INF_BITS
    }
    #[inline]
    fn is_zero(self) -> bool {
        self.raw() & F::ABS_MASK == 0
    }
    #[inline]
    fn abs(self) -> Self {
        F::from_raw(self.raw() & F::ABS_MASK)
    }
    #[inline]
    fn neg(self) -> Self {
        F::from_raw(self.raw() ^ F::SIGN_MASK)
    }
}
