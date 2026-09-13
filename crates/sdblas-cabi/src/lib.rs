//! The SoftBLAS C ABI (`softblas.h`, every routine, every width) over
//! [`sdblas`]. Built as a static library it links wherever `libsoftblas.a`
//! did, and it is what lets SoftBLAS's own munit test suite run unchanged
//! against the Rust port (`scripts/run-softblas-munit.sh`).
//!
//! Differences from the C: the rounding code `'a'` maps to
//! [`Round::Away`] (Hoon's directed away-from-zero), not SoftFloat's
//! nearest-ties-away, and an unknown code means nearest-even rather than
//! "leave the global alone" (there is no global). Real `gemv`/`gemm`
//! reject `'C'` exactly as SoftBLAS does.
//!
//! Safety: callers pass raw pointers and BLAS-style sizes, as in C; every
//! function reads and writes exactly the elements the SoftBLAS routine
//! would. Pointers must be valid for that range.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments, non_snake_case)]

use core::ffi::c_char;
use core::ptr::NonNull;
use core::slice;
use sdblas::complex::Complex;
use sdblas::{BlasFloat, Layout, Round, Trans, F128, F16, F32, F64};

/// SoftFloat `float128_t`, `{lo, hi}`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct float128_t(pub [u64; 2]);

trait Abi: Copy {
    type F: BlasFloat;
    fn into_f(self) -> Self::F;
    fn from_f(f: Self::F) -> Self;
}
macro_rules! abi_bits {
    ($($t:ty => $F:ident),*) => { $(
        impl Abi for $t {
            type F = $F;
            #[inline] fn into_f(self) -> $F { $F(self) }
            #[inline] fn from_f(f: $F) -> Self { f.0 }
        }
    )* };
}
abi_bits!(u16 => F16, u32 => F32, u64 => F64);
impl Abi for float128_t {
    type F = F128;
    #[inline]
    fn into_f(self) -> F128 {
        F128::from_u64_pair(self.0)
    }
    #[inline]
    fn from_f(f: F128) -> Self {
        float128_t(f.to_u64_pair())
    }
}

/// SoftBLAS `complexN_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct complex_t<T>(pub T, pub T);

#[inline]
fn mode(c: u8) -> Round {
    match c {
        b'z' => Round::Zero,
        b'u' => Round::Up,
        b'd' => Round::Down,
        b'a' => Round::Away,
        _ => Round::NearEven,
    }
}

/// Elements a strided vector of `n` touches: `1 + (n-1)*|inc|`, 0 for `n == 0`.
#[inline]
fn vlen(n: u64, inc: i64) -> usize {
    if n == 0 {
        0
    } else {
        1 + (n as usize - 1) * inc.unsigned_abs() as usize
    }
}

/// A slice view of `len` elements at `p`; `p` may be anything when `len == 0`.
#[inline]
unsafe fn view<'a, T>(p: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(p, len)
    }
}
#[inline]
unsafe fn view_mut<'a, T>(p: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 {
        slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), 0)
    } else {
        slice::from_raw_parts_mut(p, len)
    }
}

/// `sdblas::F*` and the ABI type share layout (`repr(transparent)` newtype
/// over the same integer; `F128` over `u128` matches two little-endian
/// words), so pointer views reinterpret in place.
#[inline]
unsafe fn fview<'a, T: Abi>(p: *const T, len: usize) -> &'a [T::F] {
    view(p as *const T::F, len)
}
#[inline]
unsafe fn fview_mut<'a, T: Abi>(p: *mut T, len: usize) -> &'a mut [T::F] {
    view_mut(p as *mut T::F, len)
}
#[inline]
unsafe fn cview<'a, T: Abi>(p: *const complex_t<T>, len: usize) -> &'a [Complex<T::F>] {
    view(p as *const Complex<T::F>, len)
}
#[inline]
unsafe fn cview_mut<'a, T: Abi>(p: *mut complex_t<T>, len: usize) -> &'a mut [Complex<T::F>] {
    view_mut(p as *mut Complex<T::F>, len)
}
#[inline]
fn cx<T: Abi>(z: complex_t<T>) -> Complex<T::F> {
    Complex::new(z.0.into_f(), z.1.into_f())
}
#[inline]
fn uncx<T: Abi>(z: Complex<T::F>) -> complex_t<T> {
    complex_t(T::from_f(z.re), T::from_f(z.im))
}

fn layout(c: c_char) -> Option<Layout> {
    match c as u8 {
        b'C' | b'c' => Some(Layout::ColMajor),
        b'R' | b'r' => Some(Layout::RowMajor),
        _ => None,
    }
}
fn trans_real(c: c_char) -> Option<Trans> {
    match c as u8 {
        b'N' | b'n' => Some(Trans::NoTrans),
        b'T' | b't' => Some(Trans::Trans),
        _ => None,
    }
}
fn trans_cplx(c: c_char) -> Option<Trans> {
    match c as u8 {
        b'C' | b'c' => Some(Trans::ConjTrans),
        _ => trans_real(c),
    }
}

/// Extents the SoftBLAS gemv loops touch: `(a_len, x_len, y_len)`.
fn gemv_lens(
    layout: Layout,
    trans: Trans,
    m: u64,
    n: u64,
    lda: u64,
    incx: i64,
    incy: u64,
) -> (usize, usize, usize) {
    let (rows, inner) = if trans == Trans::NoTrans {
        (m, n)
    } else {
        (n, m)
    };
    let a = if m == 0 || n == 0 {
        0
    } else {
        match layout {
            Layout::ColMajor => (m - 1) + (n - 1) * lda + 1,
            Layout::RowMajor => (m - 1) * lda + (n - 1) + 1,
        }
    } as usize;
    (a, vlen(inner, incx), vlen(rows, incy as i64))
}
/// Extents the SoftBLAS gemm loops touch: `(a_len, b_len, c_len)`.
fn gemm_lens(
    ta: Trans,
    tb: Trans,
    m: u64,
    n: u64,
    p: u64,
    lda: u64,
    ldb: u64,
    ldc: u64,
) -> (usize, usize, usize) {
    let a = if m == 0 || n == 0 || p == 0 {
        0
    } else if ta == Trans::NoTrans {
        (n - 1) + (m - 1) * lda + 1
    } else {
        (m - 1) + (n - 1) * lda + 1
    } as usize;
    let b = if m == 0 || n == 0 || p == 0 {
        0
    } else if tb == Trans::NoTrans {
        (p - 1) + (n - 1) * ldb + 1
    } else {
        (n - 1) + (p - 1) * ldb + 1
    } as usize;
    let c = if m == 0 || p == 0 {
        0
    } else {
        (p - 1) + (m - 1) * ldc + 1
    } as usize;
    (a, b, c)
}

macro_rules! real_abi {
    ($T:ty, $asum:ident, $axpy:ident, $copy:ident, $dot:ident, $nrm2:ident, $nrm2b:ident, $rot:ident, $rotg:ident,
     $rotm:ident, $rotmg:ident, $scal:ident, $swap:ident, $iamax:ident, $gemv:ident, $gemm:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $asum(n: u64, x: *const $T, incx: u64, m: u8) -> $T {
            <$T>::from_f(sdblas::asum(
                n,
                fview(x, vlen(n, incx as i64)),
                incx,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $axpy(
            n: u64,
            a: $T,
            x: *mut $T,
            incx: i64,
            y: *mut $T,
            incy: i64,
            m: u8,
        ) {
            sdblas::axpy(
                n,
                a.into_f(),
                fview(x, vlen(n, incx)),
                incx,
                fview_mut(y, vlen(n, incy)),
                incy,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $copy(
            n: u64,
            x: *const $T,
            incx: i64,
            y: *mut $T,
            incy: i64,
            _m: u8,
        ) {
            sdblas::copy(
                n,
                fview(x, vlen(n, incx)),
                incx,
                fview_mut(y, vlen(n, incy)),
                incy,
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $dot(
            n: u64,
            x: *const $T,
            incx: i64,
            y: *const $T,
            incy: i64,
            m: u8,
        ) -> $T {
            <$T>::from_f(sdblas::dot(
                n,
                fview(x, vlen(n, incx)),
                incx,
                fview(y, vlen(n, incy)),
                incy,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $nrm2(n: u64, x: *const $T, incx: u64, m: u8) -> $T {
            <$T>::from_f(sdblas::nrm2(
                n,
                fview(x, vlen(n, incx as i64)),
                incx,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $nrm2b(n: u64, x: *const $T, incx: u64, m: u8) -> $T {
            <$T>::from_f(sdblas::nrm2_b(
                n,
                fview(x, vlen(n, incx as i64)),
                incx,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rot(
            n: u64,
            x: *mut $T,
            incx: u64,
            y: *mut $T,
            incy: u64,
            c: $T,
            s: $T,
            m: u8,
        ) {
            sdblas::rot(
                n,
                fview_mut(x, vlen(n, incx as i64)),
                incx,
                fview_mut(y, vlen(n, incy as i64)),
                incy,
                c.into_f(),
                s.into_f(),
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rotg(a: *mut $T, b: *mut $T, c: *mut $T, s: *mut $T, m: u8) {
            let (mut fa, mut fb, mut fc, mut fs) =
                ((*a).into_f(), (*b).into_f(), (*c).into_f(), (*s).into_f());
            sdblas::rotg(&mut fa, &mut fb, &mut fc, &mut fs, mode(m));
            *a = <$T>::from_f(fa);
            *b = <$T>::from_f(fb);
            *c = <$T>::from_f(fc);
            *s = <$T>::from_f(fs);
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rotm(
            n: u64,
            x: *mut $T,
            incx: u64,
            y: *mut $T,
            incy: u64,
            p: *const $T,
            m: u8,
        ) {
            let p: &[<$T as Abi>::F; 5] = &*(p as *const [<$T as Abi>::F; 5]);
            sdblas::rotm(
                n,
                fview_mut(x, vlen(n, incx as i64)),
                incx,
                fview_mut(y, vlen(n, incy as i64)),
                incy,
                p,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rotmg(
            d1: *mut $T,
            d2: *mut $T,
            x1: *mut $T,
            y1: $T,
            p: *mut $T,
            m: u8,
        ) {
            let p: &mut [<$T as Abi>::F; 5] = &mut *(p as *mut [<$T as Abi>::F; 5]);
            let (mut fd1, mut fd2, mut fx1) = ((*d1).into_f(), (*d2).into_f(), (*x1).into_f());
            sdblas::rotmg(&mut fd1, &mut fd2, &mut fx1, y1.into_f(), p, mode(m));
            *d1 = <$T>::from_f(fd1);
            *d2 = <$T>::from_f(fd2);
            *x1 = <$T>::from_f(fx1);
        }
        #[no_mangle]
        pub unsafe extern "C" fn $scal(n: u64, a: $T, x: *mut $T, incx: u64, m: u8) {
            sdblas::scal(
                n,
                a.into_f(),
                fview_mut(x, vlen(n, incx as i64)),
                incx,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $swap(
            n: u64,
            x: *mut $T,
            incx: i64,
            y: *mut $T,
            incy: i64,
            _m: u8,
        ) {
            sdblas::swap(
                n,
                fview_mut(x, vlen(n, incx)),
                incx,
                fview_mut(y, vlen(n, incy)),
                incy,
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $iamax(n: u64, x: *const $T, incx: u64) -> u64 {
            sdblas::iamax(n, fview(x, vlen(n, incx as i64)), incx)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $gemv(
            lay: c_char,
            tr: c_char,
            mm: u64,
            n: u64,
            alpha: $T,
            a: *const $T,
            lda: u64,
            x: *const $T,
            incx: i64,
            beta: $T,
            y: *mut $T,
            incy: u64,
            m: u8,
        ) {
            let (Some(lay), Some(tr)) = (layout(lay), trans_real(tr)) else {
                return;
            };
            if incx < 1 || incy < 1 {
                return;
            }
            let (al, xl, yl) = gemv_lens(lay, tr, mm, n, lda, incx, incy);
            sdblas::gemv(
                lay,
                tr,
                mm,
                n,
                alpha.into_f(),
                fview(a, al),
                lda,
                fview(x, xl),
                incx,
                beta.into_f(),
                fview_mut(y, yl),
                incy,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $gemm(
            ta: c_char,
            tb: c_char,
            mm: u64,
            n: u64,
            p: u64,
            alpha: $T,
            a: *const $T,
            lda: u64,
            b: *const $T,
            ldb: u64,
            beta: $T,
            c: *mut $T,
            ldc: u64,
            m: u8,
        ) {
            let (Some(ta), Some(tb)) = (trans_real(ta), trans_real(tb)) else {
                return;
            };
            let (al, bl, cl) = gemm_lens(ta, tb, mm, n, p, lda, ldb, ldc);
            sdblas::gemm(
                ta,
                tb,
                mm,
                n,
                p,
                alpha.into_f(),
                fview(a, al),
                lda,
                fview(b, bl),
                ldb,
                beta.into_f(),
                fview_mut(c, cl),
                ldc,
                mode(m),
            )
        }
    };
}

macro_rules! complex_abi {
    ($T:ty, $axpy:ident, $copy:ident, $dotc:ident, $dotu:ident, $scal:ident, $swap:ident, $rot:ident,
     $asum:ident, $nrm2:ident, $iamax:ident, $gemv:ident, $gemm:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $axpy(
            n: u64,
            a: complex_t<$T>,
            x: *mut complex_t<$T>,
            incx: i64,
            y: *mut complex_t<$T>,
            incy: i64,
            m: u8,
        ) {
            sdblas::complex::axpy(
                n,
                cx(a),
                cview(x, vlen(n, incx)),
                incx,
                cview_mut(y, vlen(n, incy)),
                incy,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $copy(
            n: u64,
            x: *const complex_t<$T>,
            incx: i64,
            y: *mut complex_t<$T>,
            incy: i64,
            _m: u8,
        ) {
            sdblas::complex::copy(
                n,
                cview(x, vlen(n, incx)),
                incx,
                cview_mut(y, vlen(n, incy)),
                incy,
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $dotc(
            n: u64,
            x: *const complex_t<$T>,
            incx: i64,
            y: *const complex_t<$T>,
            incy: i64,
            m: u8,
        ) -> complex_t<$T> {
            uncx(sdblas::complex::dotc(
                n,
                cview(x, vlen(n, incx)),
                incx,
                cview(y, vlen(n, incy)),
                incy,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $dotu(
            n: u64,
            x: *const complex_t<$T>,
            incx: i64,
            y: *const complex_t<$T>,
            incy: i64,
            m: u8,
        ) -> complex_t<$T> {
            uncx(sdblas::complex::dotu(
                n,
                cview(x, vlen(n, incx)),
                incx,
                cview(y, vlen(n, incy)),
                incy,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $scal(
            n: u64,
            a: complex_t<$T>,
            x: *mut complex_t<$T>,
            incx: u64,
            m: u8,
        ) {
            sdblas::complex::scal(n, cx(a), cview_mut(x, vlen(n, incx as i64)), incx, mode(m))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $swap(
            n: u64,
            x: *mut complex_t<$T>,
            incx: u64,
            y: *mut complex_t<$T>,
            incy: u64,
            _m: u8,
        ) {
            sdblas::complex::swap(
                n,
                cview_mut(x, vlen(n, incx as i64)),
                incx,
                cview_mut(y, vlen(n, incy as i64)),
                incy,
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rot(
            n: u64,
            x: *mut complex_t<$T>,
            incx: i64,
            y: *mut complex_t<$T>,
            incy: i64,
            c: complex_t<$T>,
            s: complex_t<$T>,
            m: u8,
        ) {
            sdblas::complex::rot(
                n,
                cview_mut(x, vlen(n, incx)),
                incx,
                cview_mut(y, vlen(n, incy)),
                incy,
                cx(c),
                cx(s),
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $asum(n: u64, x: *const complex_t<$T>, incx: i64, m: u8) -> $T {
            <$T>::from_f(sdblas::complex::asum(
                n,
                cview(x, vlen(n, incx)),
                incx,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $nrm2(n: u64, x: *const complex_t<$T>, incx: u64, m: u8) -> $T {
            <$T>::from_f(sdblas::complex::nrm2(
                n,
                cview(x, vlen(n, incx as i64)),
                incx,
                mode(m),
            ))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $iamax(n: u64, x: *const complex_t<$T>, incx: u64, m: u8) -> u64 {
            sdblas::complex::iamax(n, cview(x, vlen(n, incx as i64)), incx, mode(m))
        }
        #[no_mangle]
        pub unsafe extern "C" fn $gemv(
            lay: c_char,
            tr: c_char,
            mm: u64,
            n: u64,
            alpha: complex_t<$T>,
            a: *const complex_t<$T>,
            lda: u64,
            x: *const complex_t<$T>,
            incx: i64,
            beta: complex_t<$T>,
            y: *mut complex_t<$T>,
            incy: u64,
            m: u8,
        ) {
            let (Some(lay), Some(tr)) = (layout(lay), trans_cplx(tr)) else {
                return;
            };
            if incx < 1 || incy < 1 {
                return;
            }
            let (al, xl, yl) = gemv_lens(lay, tr, mm, n, lda, incx, incy);
            sdblas::complex::gemv(
                lay,
                tr,
                mm,
                n,
                cx(alpha),
                cview(a, al),
                lda,
                cview(x, xl),
                incx,
                cx(beta),
                cview_mut(y, yl),
                incy,
                mode(m),
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $gemm(
            ta: c_char,
            tb: c_char,
            mm: u64,
            n: u64,
            p: u64,
            alpha: complex_t<$T>,
            a: *const complex_t<$T>,
            lda: u64,
            b: *const complex_t<$T>,
            ldb: u64,
            beta: complex_t<$T>,
            c: *mut complex_t<$T>,
            ldc: u64,
            m: u8,
        ) {
            let (Some(ta), Some(tb)) = (trans_cplx(ta), trans_cplx(tb)) else {
                return;
            };
            let (al, bl, cl) = gemm_lens(ta, tb, mm, n, p, lda, ldb, ldc);
            sdblas::complex::gemm(
                ta,
                tb,
                mm,
                n,
                p,
                cx(alpha),
                cview(a, al),
                lda,
                cview(b, bl),
                ldb,
                cx(beta),
                cview_mut(c, cl),
                ldc,
                mode(m),
            )
        }
    };
}

real_abi!(
    u16, hasum, haxpy, hcopy, hdot, hnrm2, hnrm2_B, hrot, hrotg, hrotm, hrotmg, hscal, hswap,
    ihamax, hgemv, hgemm
);
real_abi!(
    u32, sasum, saxpy, scopy, sdot, snrm2, snrm2_B, srot, srotg, srotm, srotmg, sscal, sswap,
    isamax, sgemv, sgemm
);
real_abi!(
    u64, dasum, daxpy, dcopy, ddot, dnrm2, dnrm2_B, drot, drotg, drotm, drotmg, dscal, dswap,
    idamax, dgemv, dgemm
);
real_abi!(
    float128_t, qasum, qaxpy, qcopy, qdot, qnrm2, qnrm2_B, qrot, qrotg, qrotm, qrotmg, qscal,
    qswap, iqamax, qgemv, qgemm
);
complex_abi!(
    u16, iaxpy, icopy, idotc, idotu, iscal, iswap, ihrot, hiasum, hinrm2, iiamax, igemv, igemm
);
complex_abi!(
    u32, caxpy, ccopy, cdotc, cdotu, cscal, cswap, csrot, scasum, scnrm2, icamax, cgemv, cgemm
);
complex_abi!(
    u64, zaxpy, zcopy, zdotc, zdotu, zscal, zswap, zdrot, dzasum, dznrm2, izamax, zgemv, zgemm
);
complex_abi!(
    float128_t, vaxpy, vcopy, vdotc, vdotu, vscal, vswap, vqrot, qvasum, qvnrm2, ivamax, vgemv,
    vgemm
);

#[no_mangle]
pub unsafe extern "C" fn sdsdot(
    n: u64,
    alpha: u32,
    x: *const u32,
    incx: i64,
    y: *const u32,
    incy: i64,
    m: u8,
) -> u32 {
    sdblas::sdsdot(
        n,
        F32(alpha),
        fview(x, vlen(n, incx)),
        incx,
        fview(y, vlen(n, incy)),
        incy,
        mode(m),
    )
    .0
}
#[no_mangle]
pub unsafe extern "C" fn hsdot(
    n: u64,
    alpha: u16,
    x: *const u16,
    incx: i64,
    y: *const u16,
    incy: i64,
    m: u8,
) -> u32 {
    sdblas::hsdot(
        n,
        F16(alpha),
        fview(x, vlen(n, incx)),
        incx,
        fview(y, vlen(n, incy)),
        incy,
        mode(m),
    )
    .0
}
