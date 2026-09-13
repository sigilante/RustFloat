//! The four interchange formats as newtypes over their raw bits.

use crate::format::Format;
use crate::round::Round;

macro_rules! define_format {
    ($(#[$doc:meta])* $name:ident, $bits:ty, $w:ty, $exp:literal, $frac:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(pub $bits);

        impl Format for $name {
            type Bits = $bits;
            type W = $w;
            const EXP_BITS: u32 = $exp;
            const FRAC_BITS: u32 = $frac;
            const NAN_VALUE: Self = $name(<Self as Format>::NAN_BITS as $bits);
            const INF_VALUE: Self = $name(<Self as Format>::INF_BITS as $bits);
            const ZERO_VALUE: Self = $name(0);
            const ONE_VALUE: Self = $name(<Self as Format>::ONE_BITS as $bits);
            const MAX_VALUE: Self = $name(<Self as Format>::MAX_BITS as $bits);
            #[inline] fn from_bits(b: $bits) -> Self { $name(b) }
            #[inline] fn to_bits(self) -> $bits { self.0 }
            #[inline] fn bits_to_u128(b: $bits) -> u128 { b as u128 }
            #[inline] fn bits_from_u128(v: u128) -> $bits { v as $bits }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($name), "({:#0w$x})"), self.0,
                       w = 2 + (<$name as Format>::WIDTH as usize) / 4)
            }
        }
    };
}

define_format!(
    /// IEEE 754 binary16 (Hoon `@rh`).
    F16, u16, u32, 5, 10);
define_format!(
    /// IEEE 754 binary32 (Hoon `@rs`).
    F32, u32, u32, 8, 23);
define_format!(
    /// IEEE 754 binary64 (Hoon `@rd`).
    F64, u64, u64, 11, 52);
define_format!(
    /// IEEE 754 binary128 (Hoon `@rq`).
    F128, u128, u128, 15, 112);

/// Convert between any two formats. Narrowing rounds per `r`; widening is
/// exact and ignores `r`.
#[inline]
pub fn convert<A: Format, B: Format>(a: A, r: Round) -> B {
    B::from_raw(crate::ops::convert::convert::<A, B>(a.raw(), r))
}

macro_rules! widen { ($($from:ident => $to:ident : $fn:ident;)*) => { $(
    impl $from {
        /// Exact widening.
        #[inline]
        pub fn $fn(self) -> $to { convert(self, Round::NearEven) }
    } )* }; }
macro_rules! narrow { ($($from:ident => $to:ident : $fn:ident;)*) => { $(
    impl $from {
        /// Narrowing conversion, rounded per `r`.
        #[inline]
        pub fn $fn(self, r: Round) -> $to { convert(self, r) }
    } )* }; }

widen! {
    F16 => F32: to_f32;  F16 => F64: to_f64;  F16 => F128: to_f128;
    F32 => F64: to_f64;  F32 => F128: to_f128;
    F64 => F128: to_f128;
}
narrow! {
    F32 => F16: to_f16;
    F64 => F16: to_f16;   F64 => F32: to_f32;
    F128 => F16: to_f16;  F128 => F32: to_f32;  F128 => F64: to_f64;
}

impl F128 {
    /// The two 64-bit words, low word first: the layout of SoftFloat's
    /// `float128_t` on little-endian hosts and of nockvm's `Atom::as_u64_pair`.
    #[inline]
    pub fn to_u64_pair(self) -> [u64; 2] {
        [self.0 as u64, (self.0 >> 64) as u64]
    }
    /// Inverse of [`F128::to_u64_pair`].
    #[inline]
    pub fn from_u64_pair(p: [u64; 2]) -> Self {
        F128((p[0] as u128) | ((p[1] as u128) << 64))
    }
}
