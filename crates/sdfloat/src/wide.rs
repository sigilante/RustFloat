//! Unsigned integer widths used for significand arithmetic.
//!
//! Each format picks the narrowest native type that holds its significand
//! plus three guard bits (`Format::PREC + 3`): u32 for binary16/32, u64 for
//! binary64, u128 for binary128. Everything the rounding core needs is
//! expressed through this trait so the algorithms are written once.

pub trait Wide: Copy + Eq + Ord + core::fmt::Debug + private::Sealed {
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;

    /// Truncating conversion.
    fn from_u128(v: u128) -> Self;
    fn to_u128(self) -> u128;
    /// `n < BITS`.
    fn shl(self, n: u32) -> Self;
    /// `n < BITS`.
    fn shr(self, n: u32) -> Self;
    fn wrapping_add(self, o: Self) -> Self;
    fn wrapping_sub(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn not(self) -> Self;
    fn leading_zeros(self) -> u32;
    /// `(quotient, remainder)`; divisor nonzero.
    fn div_rem(self, o: Self) -> (Self, Self);
    /// Full-width product as `(hi, lo)`.
    fn widening_mul(self, o: Self) -> (Self, Self);

    #[inline]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
    /// The low `n` bits, `n <= BITS`.
    #[inline]
    fn low_bits(self, n: u32) -> Self {
        if n >= Self::BITS {
            self
        } else {
            self.and(Self::ONE.shl(n).wrapping_sub(Self::ONE))
        }
    }
    /// Shift right by `n` (any `n`), OR-ing the bits shifted out into the
    /// result's bit 0 as a sticky bit. SoftFloat's `softfloat_shiftRightJam`.
    #[inline]
    fn shr_jam(self, n: u32) -> Self {
        if n == 0 {
            self
        } else if n >= Self::BITS {
            if self.is_zero() {
                Self::ZERO
            } else {
                Self::ONE
            }
        } else {
            let s = self.shr(n);
            if self.low_bits(n).is_zero() {
                s
            } else {
                s.or(Self::ONE)
            }
        }
    }
    /// Shift the double-width value `hi:lo` right by `n` with a sticky bit,
    /// returning the low word. The caller guarantees the result fits, i.e.
    /// `hi >> n == 0` (for `n < BITS`) or `hi >> (n - BITS) == 0`.
    #[inline]
    fn shr_jam2(hi: Self, lo: Self, n: u32) -> Self {
        if n == 0 {
            debug_assert!(hi.is_zero());
            lo
        } else if n < Self::BITS {
            debug_assert!(hi.shr(n).is_zero());
            let s = lo.shr(n).or(hi.shl(Self::BITS - n));
            if lo.low_bits(n).is_zero() {
                s
            } else {
                s.or(Self::ONE)
            }
        } else {
            let s = hi.shr_jam(n - Self::BITS);
            if lo.is_zero() {
                s
            } else {
                s.or(Self::ONE)
            }
        }
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
}

macro_rules! impl_wide_native {
    ($t:ty, $bits:literal, $dbl:ty) => {
        impl Wide for $t {
            const BITS: u32 = $bits;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline]
            fn from_u128(v: u128) -> Self {
                v as $t
            }
            #[inline]
            fn to_u128(self) -> u128 {
                self as u128
            }
            #[inline]
            fn shl(self, n: u32) -> Self {
                self << n
            }
            #[inline]
            fn shr(self, n: u32) -> Self {
                self >> n
            }
            #[inline]
            fn wrapping_add(self, o: Self) -> Self {
                <$t>::wrapping_add(self, o)
            }
            #[inline]
            fn wrapping_sub(self, o: Self) -> Self {
                <$t>::wrapping_sub(self, o)
            }
            #[inline]
            fn or(self, o: Self) -> Self {
                self | o
            }
            #[inline]
            fn and(self, o: Self) -> Self {
                self & o
            }
            #[inline]
            fn not(self) -> Self {
                !self
            }
            #[inline]
            fn leading_zeros(self) -> u32 {
                <$t>::leading_zeros(self)
            }
            #[inline]
            fn div_rem(self, o: Self) -> (Self, Self) {
                (self / o, self % o)
            }
            #[inline]
            fn widening_mul(self, o: Self) -> (Self, Self) {
                let p = (self as $dbl) * (o as $dbl);
                ((p >> $bits) as $t, p as $t)
            }
        }
    };
}
impl_wide_native!(u32, 32, u64);
impl_wide_native!(u64, 64, u128);

impl Wide for u128 {
    const BITS: u32 = 128;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    #[inline]
    fn from_u128(v: u128) -> Self {
        v
    }
    #[inline]
    fn to_u128(self) -> u128 {
        self
    }
    #[inline]
    fn shl(self, n: u32) -> Self {
        self << n
    }
    #[inline]
    fn shr(self, n: u32) -> Self {
        self >> n
    }
    #[inline]
    fn wrapping_add(self, o: Self) -> Self {
        u128::wrapping_add(self, o)
    }
    #[inline]
    fn wrapping_sub(self, o: Self) -> Self {
        u128::wrapping_sub(self, o)
    }
    #[inline]
    fn or(self, o: Self) -> Self {
        self | o
    }
    #[inline]
    fn and(self, o: Self) -> Self {
        self & o
    }
    #[inline]
    fn not(self) -> Self {
        !self
    }
    #[inline]
    fn leading_zeros(self) -> u32 {
        u128::leading_zeros(self)
    }
    #[inline]
    fn div_rem(self, o: Self) -> (Self, Self) {
        (self / o, self % o)
    }
    /// Schoolbook 2x2 limb product; `u128::widening_mul` is not stable.
    #[inline]
    fn widening_mul(self, o: Self) -> (Self, Self) {
        const M: u128 = u64::MAX as u128;
        let (a0, a1) = (self & M, self >> 64);
        let (b0, b1) = (o & M, o >> 64);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + (p01 & M) + (p10 & M);
        let lo = (p00 & M) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        (hi, lo)
    }
}
