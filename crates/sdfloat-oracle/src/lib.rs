//! Test oracle: Berkeley SoftFloat 3e (C, vendored) behind a flat safe API.
//!
//! Every call takes the SoftFloat rounding mode explicitly. Values cross the
//! FFI as raw bit patterns (`u16`/`u32`/`u64`/`u128`). This crate exists only
//! as a dev-dependency of `sdfloat`; nothing here is for production.

#![allow(clippy::missing_safety_doc)]

/// SoftFloat rounding-mode codes (`softfloat_round_*`).
pub mod mode {
    pub const NEAR_EVEN: u8 = 0;
    pub const MIN_MAG: u8 = 1;
    pub const MIN: u8 = 2;
    pub const MAX: u8 = 3;
    pub const NEAR_MAXMAG: u8 = 4;
    pub const ODD: u8 = 6;
}

macro_rules! scalar_ffi {
    ($w:literal, $t:ty, $m:ident, [$($bin:ident),*], [$($un:ident),*], [$($cmp:ident),*]) => {
        pub mod $m {
            mod ffi {
                extern "C" {
                    $( pub fn $bin(m: u8, a: $t, b: $t) -> $t; )*
                    $( pub fn $un(m: u8, a: $t) -> $t; )*
                    $( pub fn $cmp(a: $t, b: $t) -> i32; )*
                }
            }
            $( pub fn $bin(m: u8, a: $t, b: $t) -> $t { unsafe { ffi::$bin(m, a, b) } } )*
            $( pub fn $un(m: u8, a: $t) -> $t { unsafe { ffi::$un(m, a) } } )*
            $( pub fn $cmp(a: $t, b: $t) -> bool { unsafe { ffi::$cmp(a, b) != 0 } } )*
        }
    };
}

scalar_ffi!(
    16,
    u16,
    f16,
    [orc_f16_add, orc_f16_sub, orc_f16_mul, orc_f16_div],
    [orc_f16_sqrt, orc_f16_round_to_int],
    [orc_f16_eq, orc_f16_lt, orc_f16_le]
);
scalar_ffi!(
    32,
    u32,
    f32,
    [orc_f32_add, orc_f32_sub, orc_f32_mul, orc_f32_div],
    [orc_f32_sqrt, orc_f32_round_to_int],
    [orc_f32_eq, orc_f32_lt, orc_f32_le]
);
scalar_ffi!(
    64,
    u64,
    f64,
    [orc_f64_add, orc_f64_sub, orc_f64_mul, orc_f64_div],
    [orc_f64_sqrt, orc_f64_round_to_int],
    [orc_f64_eq, orc_f64_lt, orc_f64_le]
);

mod ffi_int {
    extern "C" {
        pub fn orc_f16_to_i32(m: u8, a: u16) -> i32;
        pub fn orc_f16_to_i64(m: u8, a: u16) -> i64;
        pub fn orc_f32_to_i32(m: u8, a: u32) -> i32;
        pub fn orc_f32_to_i64(m: u8, a: u32) -> i64;
        pub fn orc_f64_to_i32(m: u8, a: u64) -> i32;
        pub fn orc_f64_to_i64(m: u8, a: u64) -> i64;
        pub fn orc_f128_to_i32(m: u8, a: *const u64) -> i32;
        pub fn orc_f128_to_i64(m: u8, a: *const u64) -> i64;
        pub fn orc_i32_to_f16(m: u8, a: i32) -> u16;
        pub fn orc_i64_to_f16(m: u8, a: i64) -> u16;
        pub fn orc_ui32_to_f16(m: u8, a: u32) -> u16;
        pub fn orc_ui64_to_f16(m: u8, a: u64) -> u16;
        pub fn orc_i32_to_f32(m: u8, a: i32) -> u32;
        pub fn orc_i64_to_f32(m: u8, a: i64) -> u32;
        pub fn orc_ui32_to_f32(m: u8, a: u32) -> u32;
        pub fn orc_ui64_to_f32(m: u8, a: u64) -> u32;
        pub fn orc_i32_to_f64(m: u8, a: i32) -> u64;
        pub fn orc_i64_to_f64(m: u8, a: i64) -> u64;
        pub fn orc_ui32_to_f64(m: u8, a: u32) -> u64;
        pub fn orc_ui64_to_f64(m: u8, a: u64) -> u64;
        pub fn orc_i32_to_f128(m: u8, a: i32, o: *mut u64);
        pub fn orc_i64_to_f128(m: u8, a: i64, o: *mut u64);
        pub fn orc_ui32_to_f128(m: u8, a: u32, o: *mut u64);
        pub fn orc_ui64_to_f128(m: u8, a: u64, o: *mut u64);
        pub fn orc_f16_to_f32(m: u8, a: u16) -> u32;
        pub fn orc_f16_to_f64(m: u8, a: u16) -> u64;
        pub fn orc_f32_to_f16(m: u8, a: u32) -> u16;
        pub fn orc_f32_to_f64(m: u8, a: u32) -> u64;
        pub fn orc_f64_to_f16(m: u8, a: u64) -> u16;
        pub fn orc_f64_to_f32(m: u8, a: u64) -> u32;
        pub fn orc_f16_to_f128(m: u8, a: u16, o: *mut u64);
        pub fn orc_f32_to_f128(m: u8, a: u32, o: *mut u64);
        pub fn orc_f64_to_f128(m: u8, a: u64, o: *mut u64);
        pub fn orc_f128_to_f16(m: u8, a: *const u64) -> u16;
        pub fn orc_f128_to_f32(m: u8, a: *const u64) -> u32;
        pub fn orc_f128_to_f64(m: u8, a: *const u64) -> u64;
        pub fn orc_f128_add(m: u8, a: *const u64, b: *const u64, o: *mut u64);
        pub fn orc_f128_sub(m: u8, a: *const u64, b: *const u64, o: *mut u64);
        pub fn orc_f128_mul(m: u8, a: *const u64, b: *const u64, o: *mut u64);
        pub fn orc_f128_div(m: u8, a: *const u64, b: *const u64, o: *mut u64);
        pub fn orc_f128_sqrt(m: u8, a: *const u64, o: *mut u64);
        pub fn orc_f128_round_to_int(m: u8, a: *const u64, o: *mut u64);
        pub fn orc_f128_eq(a: *const u64, b: *const u64) -> i32;
        pub fn orc_f128_lt(a: *const u64, b: *const u64) -> i32;
        pub fn orc_f128_le(a: *const u64, b: *const u64) -> i32;
    }
}

fn split(q: u128) -> [u64; 2] {
    [q as u64, (q >> 64) as u64]
}
fn join(p: [u64; 2]) -> u128 {
    (p[0] as u128) | ((p[1] as u128) << 64)
}

macro_rules! q_bin { ($($name:ident = $ffi:ident;)*) => { $(
    pub fn $name(m: u8, a: u128, b: u128) -> u128 {
        let (a, b, mut o) = (split(a), split(b), [0u64; 2]);
        unsafe { ffi_int::$ffi(m, a.as_ptr(), b.as_ptr(), o.as_mut_ptr()) }
        join(o)
    } )* }; }
macro_rules! q_un { ($($name:ident = $ffi:ident;)*) => { $(
    pub fn $name(m: u8, a: u128) -> u128 {
        let (a, mut o) = (split(a), [0u64; 2]);
        unsafe { ffi_int::$ffi(m, a.as_ptr(), o.as_mut_ptr()) }
        join(o)
    } )* }; }
macro_rules! q_cmp { ($($name:ident = $ffi:ident;)*) => { $(
    pub fn $name(a: u128, b: u128) -> bool {
        let (a, b) = (split(a), split(b));
        unsafe { ffi_int::$ffi(a.as_ptr(), b.as_ptr()) != 0 }
    } )* }; }
macro_rules! q_from { ($($name:ident($t:ty) = $ffi:ident;)*) => { $(
    pub fn $name(m: u8, a: $t) -> u128 {
        let mut o = [0u64; 2];
        unsafe { ffi_int::$ffi(m, a, o.as_mut_ptr()) }
        join(o)
    } )* }; }
macro_rules! q_to { ($($name:ident -> $t:ty = $ffi:ident;)*) => { $(
    pub fn $name(m: u8, a: u128) -> $t {
        let a = split(a);
        unsafe { ffi_int::$ffi(m, a.as_ptr()) }
    } )* }; }
macro_rules! plain { ($($name:ident($($an:ident: $at:ty),*) -> $rt:ty = $ffi:ident;)*) => { $(
    pub fn $name(m: u8, $($an: $at),*) -> $rt { unsafe { ffi_int::$ffi(m, $($an),*) } } )* }; }

pub mod f128 {
    use super::*;
    q_bin! { orc_f128_add = orc_f128_add; orc_f128_sub = orc_f128_sub;
    orc_f128_mul = orc_f128_mul; orc_f128_div = orc_f128_div; }
    q_un! { orc_f128_sqrt = orc_f128_sqrt; orc_f128_round_to_int = orc_f128_round_to_int; }
    q_cmp! { orc_f128_eq = orc_f128_eq; orc_f128_lt = orc_f128_lt; orc_f128_le = orc_f128_le; }
}

/// Integer and width conversions, all widths.
pub mod conv {
    use super::*;
    plain! {
        f16_to_i32(a: u16) -> i32 = orc_f16_to_i32; f16_to_i64(a: u16) -> i64 = orc_f16_to_i64;
        f32_to_i32(a: u32) -> i32 = orc_f32_to_i32; f32_to_i64(a: u32) -> i64 = orc_f32_to_i64;
        f64_to_i32(a: u64) -> i32 = orc_f64_to_i32; f64_to_i64(a: u64) -> i64 = orc_f64_to_i64;
        i32_to_f16(a: i32) -> u16 = orc_i32_to_f16; i64_to_f16(a: i64) -> u16 = orc_i64_to_f16;
        ui32_to_f16(a: u32) -> u16 = orc_ui32_to_f16; ui64_to_f16(a: u64) -> u16 = orc_ui64_to_f16;
        i32_to_f32(a: i32) -> u32 = orc_i32_to_f32; i64_to_f32(a: i64) -> u32 = orc_i64_to_f32;
        ui32_to_f32(a: u32) -> u32 = orc_ui32_to_f32; ui64_to_f32(a: u64) -> u32 = orc_ui64_to_f32;
        i32_to_f64(a: i32) -> u64 = orc_i32_to_f64; i64_to_f64(a: i64) -> u64 = orc_i64_to_f64;
        ui32_to_f64(a: u32) -> u64 = orc_ui32_to_f64; ui64_to_f64(a: u64) -> u64 = orc_ui64_to_f64;
        f16_to_f32(a: u16) -> u32 = orc_f16_to_f32; f16_to_f64(a: u16) -> u64 = orc_f16_to_f64;
        f32_to_f16(a: u32) -> u16 = orc_f32_to_f16; f32_to_f64(a: u32) -> u64 = orc_f32_to_f64;
        f64_to_f16(a: u64) -> u16 = orc_f64_to_f16; f64_to_f32(a: u64) -> u32 = orc_f64_to_f32;
    }
    q_to! { f128_to_i32 -> i32 = orc_f128_to_i32; f128_to_i64 -> i64 = orc_f128_to_i64;
    f128_to_f16 -> u16 = orc_f128_to_f16; f128_to_f32 -> u32 = orc_f128_to_f32;
    f128_to_f64 -> u64 = orc_f128_to_f64; }
    q_from! { i32_to_f128(i32) = orc_i32_to_f128; i64_to_f128(i64) = orc_i64_to_f128;
    ui32_to_f128(u32) = orc_ui32_to_f128; ui64_to_f128(u64) = orc_ui64_to_f128;
    f16_to_f128(u16) = orc_f16_to_f128; f32_to_f128(u32) = orc_f32_to_f128;
    f64_to_f128(u64) = orc_f64_to_f128; }
}
