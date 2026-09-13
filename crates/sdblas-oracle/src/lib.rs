//! Test oracle: SoftBLAS (C, vendored) on SoftFloat 3e, behind thin safe
//! wrappers over raw bit patterns. Rounding modes are SoftBLAS's `char`
//! codes: `b'n'`, `b'z'`, `b'u'`, `b'd'`, `b'a'` (note `'a'` is SoftFloat's
//! nearest-ties-away, not Hoon's `%a`). Dev-dependency of `sdblas` only.

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

/// SoftFloat `float128_t`: `[lo, hi]` words.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Q(pub [u64; 2]);

impl Q {
    pub fn from_u128(v: u128) -> Q {
        Q([v as u64, (v >> 64) as u64])
    }
    pub fn to_u128(self) -> u128 {
        (self.0[0] as u128) | ((self.0[1] as u128) << 64)
    }
}

/// SoftBLAS `complexN_t`: real part first.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct C<T>(pub T, pub T);

macro_rules! real_width {
    ($m:ident, $t:ty, $ct:ty,
     [$asum:ident, $axpy:ident, $copy:ident, $dot:ident, $nrm2:ident, $nrm2b:ident, $rot:ident, $rotg:ident,
      $rotm:ident, $rotmg:ident, $scal:ident, $swap:ident, $iamax:ident, $gemv:ident, $gemm:ident],
     [$caxpy:ident, $ccopy:ident, $cdotc:ident, $cdotu:ident, $cscal:ident, $cswap:ident, $crot:ident,
      $casum:ident, $cnrm2:ident, $ciamax:ident, $cgemv:ident, $cgemm:ident]) => {
        pub mod $m {
            use super::C;
            #[allow(unused_imports)]
            use super::Q;
            pub type T = $t;
            pub type CT = C<$ct>;
            mod ffi {
                use super::{CT, T};
                extern "C" {
                    pub fn $asum(n: u64, x: *const T, incx: u64, m: u8) -> T;
                    pub fn $axpy(n: u64, a: T, x: *mut T, incx: i64, y: *mut T, incy: i64, m: u8);
                    pub fn $copy(n: u64, x: *const T, incx: i64, y: *mut T, incy: i64, m: u8);
                    pub fn $dot(n: u64, x: *const T, incx: i64, y: *const T, incy: i64, m: u8)
                        -> T;
                    pub fn $nrm2(n: u64, x: *const T, incx: u64, m: u8) -> T;
                    pub fn $nrm2b(n: u64, x: *const T, incx: u64, m: u8) -> T;
                    pub fn $rot(
                        n: u64,
                        x: *mut T,
                        incx: u64,
                        y: *mut T,
                        incy: u64,
                        c: T,
                        s: T,
                        m: u8,
                    );
                    pub fn $rotg(a: *mut T, b: *mut T, c: *mut T, s: *mut T, m: u8);
                    pub fn $rotm(
                        n: u64,
                        x: *mut T,
                        incx: u64,
                        y: *mut T,
                        incy: u64,
                        p: *const T,
                        m: u8,
                    );
                    pub fn $rotmg(d1: *mut T, d2: *mut T, x1: *mut T, y1: T, p: *mut T, m: u8);
                    pub fn $scal(n: u64, a: T, x: *mut T, incx: u64, m: u8);
                    pub fn $swap(n: u64, x: *mut T, incx: i64, y: *mut T, incy: i64, m: u8);
                    pub fn $iamax(n: u64, x: *const T, incx: u64) -> u64;
                    pub fn $gemv(
                        layout: u8,
                        trans: u8,
                        mm: u64,
                        n: u64,
                        alpha: T,
                        a: *const T,
                        lda: u64,
                        x: *const T,
                        incx: i64,
                        beta: T,
                        y: *mut T,
                        incy: u64,
                        m: u8,
                    );
                    pub fn $gemm(
                        ta: u8,
                        tb: u8,
                        mm: u64,
                        n: u64,
                        p: u64,
                        alpha: T,
                        a: *const T,
                        lda: u64,
                        b: *const T,
                        ldb: u64,
                        beta: T,
                        c: *mut T,
                        ldc: u64,
                        m: u8,
                    );
                    pub fn $caxpy(
                        n: u64,
                        a: CT,
                        x: *mut CT,
                        incx: i64,
                        y: *mut CT,
                        incy: i64,
                        m: u8,
                    );
                    pub fn $ccopy(n: u64, x: *const CT, incx: i64, y: *mut CT, incy: i64, m: u8);
                    pub fn $cdotc(
                        n: u64,
                        x: *const CT,
                        incx: i64,
                        y: *const CT,
                        incy: i64,
                        m: u8,
                    ) -> CT;
                    pub fn $cdotu(
                        n: u64,
                        x: *const CT,
                        incx: i64,
                        y: *const CT,
                        incy: i64,
                        m: u8,
                    ) -> CT;
                    pub fn $cscal(n: u64, a: CT, x: *mut CT, incx: u64, m: u8);
                    pub fn $cswap(n: u64, x: *mut CT, incx: u64, y: *mut CT, incy: u64, m: u8);
                    pub fn $crot(
                        n: u64,
                        x: *mut CT,
                        incx: i64,
                        y: *mut CT,
                        incy: i64,
                        c: CT,
                        s: CT,
                        m: u8,
                    );
                    pub fn $casum(n: u64, x: *const CT, incx: i64, m: u8) -> T;
                    pub fn $cnrm2(n: u64, x: *const CT, incx: u64, m: u8) -> T;
                    pub fn $ciamax(n: u64, x: *const CT, incx: u64, m: u8) -> u64;
                    pub fn $cgemv(
                        layout: u8,
                        trans: u8,
                        mm: u64,
                        n: u64,
                        alpha: CT,
                        a: *const CT,
                        lda: u64,
                        x: *const CT,
                        incx: i64,
                        beta: CT,
                        y: *mut CT,
                        incy: u64,
                        m: u8,
                    );
                    pub fn $cgemm(
                        ta: u8,
                        tb: u8,
                        mm: u64,
                        n: u64,
                        p: u64,
                        alpha: CT,
                        a: *const CT,
                        lda: u64,
                        b: *const CT,
                        ldb: u64,
                        beta: CT,
                        c: *mut CT,
                        ldc: u64,
                        m: u8,
                    );
                }
            }
            pub fn asum(n: u64, x: &[T], incx: u64, m: u8) -> T {
                unsafe { ffi::$asum(n, x.as_ptr(), incx, m) }
            }
            pub fn axpy(n: u64, a: T, x: &[T], incx: i64, y: &mut [T], incy: i64, m: u8) {
                unsafe { ffi::$axpy(n, a, x.as_ptr() as *mut T, incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn copy(n: u64, x: &[T], incx: i64, y: &mut [T], incy: i64, m: u8) {
                unsafe { ffi::$copy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn dot(n: u64, x: &[T], incx: i64, y: &[T], incy: i64, m: u8) -> T {
                unsafe { ffi::$dot(n, x.as_ptr(), incx, y.as_ptr(), incy, m) }
            }
            pub fn nrm2(n: u64, x: &[T], incx: u64, m: u8) -> T {
                unsafe { ffi::$nrm2(n, x.as_ptr(), incx, m) }
            }
            pub fn nrm2_b(n: u64, x: &[T], incx: u64, m: u8) -> T {
                unsafe { ffi::$nrm2b(n, x.as_ptr(), incx, m) }
            }
            pub fn rot(n: u64, x: &mut [T], incx: u64, y: &mut [T], incy: u64, c: T, s: T, m: u8) {
                unsafe { ffi::$rot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s, m) }
            }
            pub fn rotg(a: &mut T, b: &mut T, c: &mut T, s: &mut T, m: u8) {
                unsafe { ffi::$rotg(a, b, c, s, m) }
            }
            pub fn rotm(n: u64, x: &mut [T], incx: u64, y: &mut [T], incy: u64, p: &[T; 5], m: u8) {
                unsafe { ffi::$rotm(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, p.as_ptr(), m) }
            }
            pub fn rotmg(d1: &mut T, d2: &mut T, x1: &mut T, y1: T, p: &mut [T; 5], m: u8) {
                unsafe { ffi::$rotmg(d1, d2, x1, y1, p.as_mut_ptr(), m) }
            }
            pub fn scal(n: u64, a: T, x: &mut [T], incx: u64, m: u8) {
                unsafe { ffi::$scal(n, a, x.as_mut_ptr(), incx, m) }
            }
            pub fn swap(n: u64, x: &mut [T], incx: i64, y: &mut [T], incy: i64, m: u8) {
                unsafe { ffi::$swap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn iamax(n: u64, x: &[T], incx: u64) -> u64 {
                unsafe { ffi::$iamax(n, x.as_ptr(), incx) }
            }
            pub fn gemv(
                layout: u8,
                trans: u8,
                mm: u64,
                n: u64,
                alpha: T,
                a: &[T],
                lda: u64,
                x: &[T],
                incx: i64,
                beta: T,
                y: &mut [T],
                incy: u64,
                m: u8,
            ) {
                unsafe {
                    ffi::$gemv(
                        layout,
                        trans,
                        mm,
                        n,
                        alpha,
                        a.as_ptr(),
                        lda,
                        x.as_ptr(),
                        incx,
                        beta,
                        y.as_mut_ptr(),
                        incy,
                        m,
                    )
                }
            }
            pub fn gemm(
                ta: u8,
                tb: u8,
                mm: u64,
                n: u64,
                p: u64,
                alpha: T,
                a: &[T],
                lda: u64,
                b: &[T],
                ldb: u64,
                beta: T,
                c: &mut [T],
                ldc: u64,
                m: u8,
            ) {
                unsafe {
                    ffi::$gemm(
                        ta,
                        tb,
                        mm,
                        n,
                        p,
                        alpha,
                        a.as_ptr(),
                        lda,
                        b.as_ptr(),
                        ldb,
                        beta,
                        c.as_mut_ptr(),
                        ldc,
                        m,
                    )
                }
            }
            pub fn caxpy(n: u64, a: CT, x: &[CT], incx: i64, y: &mut [CT], incy: i64, m: u8) {
                unsafe { ffi::$caxpy(n, a, x.as_ptr() as *mut CT, incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn ccopy(n: u64, x: &[CT], incx: i64, y: &mut [CT], incy: i64, m: u8) {
                unsafe { ffi::$ccopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn cdotc(n: u64, x: &[CT], incx: i64, y: &[CT], incy: i64, m: u8) -> CT {
                unsafe { ffi::$cdotc(n, x.as_ptr(), incx, y.as_ptr(), incy, m) }
            }
            pub fn cdotu(n: u64, x: &[CT], incx: i64, y: &[CT], incy: i64, m: u8) -> CT {
                unsafe { ffi::$cdotu(n, x.as_ptr(), incx, y.as_ptr(), incy, m) }
            }
            pub fn cscal(n: u64, a: CT, x: &mut [CT], incx: u64, m: u8) {
                unsafe { ffi::$cscal(n, a, x.as_mut_ptr(), incx, m) }
            }
            pub fn cswap(n: u64, x: &mut [CT], incx: u64, y: &mut [CT], incy: u64, m: u8) {
                unsafe { ffi::$cswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, m) }
            }
            pub fn crot(
                n: u64,
                x: &mut [CT],
                incx: i64,
                y: &mut [CT],
                incy: i64,
                c: CT,
                s: CT,
                m: u8,
            ) {
                unsafe { ffi::$crot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s, m) }
            }
            pub fn casum(n: u64, x: &[CT], incx: i64, m: u8) -> T {
                unsafe { ffi::$casum(n, x.as_ptr(), incx, m) }
            }
            pub fn cnrm2(n: u64, x: &[CT], incx: u64, m: u8) -> T {
                unsafe { ffi::$cnrm2(n, x.as_ptr(), incx, m) }
            }
            pub fn ciamax(n: u64, x: &[CT], incx: u64, m: u8) -> u64 {
                unsafe { ffi::$ciamax(n, x.as_ptr(), incx, m) }
            }
            pub fn cgemv(
                layout: u8,
                trans: u8,
                mm: u64,
                n: u64,
                alpha: CT,
                a: &[CT],
                lda: u64,
                x: &[CT],
                incx: i64,
                beta: CT,
                y: &mut [CT],
                incy: u64,
                m: u8,
            ) {
                unsafe {
                    ffi::$cgemv(
                        layout,
                        trans,
                        mm,
                        n,
                        alpha,
                        a.as_ptr(),
                        lda,
                        x.as_ptr(),
                        incx,
                        beta,
                        y.as_mut_ptr(),
                        incy,
                        m,
                    )
                }
            }
            pub fn cgemm(
                ta: u8,
                tb: u8,
                mm: u64,
                n: u64,
                p: u64,
                alpha: CT,
                a: &[CT],
                lda: u64,
                b: &[CT],
                ldb: u64,
                beta: CT,
                c: &mut [CT],
                ldc: u64,
                m: u8,
            ) {
                unsafe {
                    ffi::$cgemm(
                        ta,
                        tb,
                        mm,
                        n,
                        p,
                        alpha,
                        a.as_ptr(),
                        lda,
                        b.as_ptr(),
                        ldb,
                        beta,
                        c.as_mut_ptr(),
                        ldc,
                        m,
                    )
                }
            }
        }
    };
}

real_width!(
    h,
    u16,
    u16,
    [
        hasum, haxpy, hcopy, hdot, hnrm2, hnrm2_B, hrot, hrotg, hrotm, hrotmg, hscal, hswap,
        ihamax, hgemv, hgemm
    ],
    [iaxpy, icopy, idotc, idotu, iscal, iswap, ihrot, hiasum, hinrm2, iiamax, igemv, igemm]
);
real_width!(
    s,
    u32,
    u32,
    [
        sasum, saxpy, scopy, sdot, snrm2, snrm2_B, srot, srotg, srotm, srotmg, sscal, sswap,
        isamax, sgemv, sgemm
    ],
    [caxpy, ccopy, cdotc, cdotu, cscal, cswap, csrot, scasum, scnrm2, icamax, cgemv, cgemm]
);
real_width!(
    d,
    u64,
    u64,
    [
        dasum, daxpy, dcopy, ddot, dnrm2, dnrm2_B, drot, drotg, drotm, drotmg, dscal, dswap,
        idamax, dgemv, dgemm
    ],
    [zaxpy, zcopy, zdotc, zdotu, zscal, zswap, zdrot, dzasum, dznrm2, izamax, zgemv, zgemm]
);
real_width!(
    q,
    Q,
    Q,
    [
        qasum, qaxpy, qcopy, qdot, qnrm2, qnrm2_B, qrot, qrotg, qrotm, qrotmg, qscal, qswap,
        iqamax, qgemv, qgemm
    ],
    [vaxpy, vcopy, vdotc, vdotu, vscal, vswap, vqrot, qvasum, qvnrm2, ivamax, vgemv, vgemm]
);

mod ffi_extra {
    extern "C" {
        pub fn sdsdot(
            n: u64,
            alpha: u32,
            x: *const u32,
            incx: i64,
            y: *const u32,
            incy: i64,
            m: u8,
        ) -> u32;
        pub fn hsdot(
            n: u64,
            alpha: u16,
            x: *const u16,
            incx: i64,
            y: *const u16,
            incy: i64,
            m: u8,
        ) -> u32;
    }
}
pub fn sdsdot(n: u64, alpha: u32, x: &[u32], incx: i64, y: &[u32], incy: i64, m: u8) -> u32 {
    unsafe { ffi_extra::sdsdot(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy, m) }
}
pub fn hsdot(n: u64, alpha: u16, x: &[u16], incx: i64, y: &[u16], incy: i64, m: u8) -> u32 {
    unsafe { ffi_extra::hsdot(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy, m) }
}
