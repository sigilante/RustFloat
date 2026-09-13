//! Complex routines over [`Complex<F>`]. Model: the SoftBLAS `c*.c`
//! sources and the `c32_*` macros in `softblas.h`; the `i`/`z`/`v` widths
//! are identical modulo type.

use crate::{at, gt, start, stride_overflows, BlasFloat, Layout, Round, Trans};

/// A complex number as two floats, real part first (SoftBLAS `complexN_t`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[repr(C)]
pub struct Complex<F> {
    /// Real part.
    pub re: F,
    /// Imaginary part.
    pub im: F,
}

impl<F: BlasFloat> Complex<F> {
    /// `0 + 0i`.
    pub const ZERO: Self = Complex {
        re: F::ZERO,
        im: F::ZERO,
    };

    /// Construct from parts.
    #[inline]
    pub const fn new(re: F, im: F) -> Self {
        Complex { re, im }
    }
    /// Componentwise sum (`c32_add`).
    #[inline]
    pub fn add(self, o: Self, r: Round) -> Self {
        Complex::new(self.re.add(o.re, r), self.im.add(o.im, r))
    }
    /// Componentwise difference (`c32_sub`).
    #[inline]
    pub fn sub(self, o: Self, r: Round) -> Self {
        Complex::new(self.re.sub(o.re, r), self.im.sub(o.im, r))
    }
    /// `(ar*br - ai*bi) + (ar*bi + ai*br)i`, in that order (`c32_mul`).
    #[inline]
    pub fn mul(self, o: Self, r: Round) -> Self {
        let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
        Complex::new(
            ar.mul(br, r).sub(ai.mul(bi, r), r),
            ar.mul(bi, r).add(ai.mul(br, r), r),
        )
    }
    /// Conjugate: flips the sign bit of the imaginary part (`c32_conj`).
    #[inline]
    pub fn conj(self) -> Self {
        Complex::new(self.re, self.im.neg())
    }
    /// `|re| + |im|`, the BLAS `scabs1` magnitude used by `asum`/`iamax`.
    #[inline]
    pub fn abs1(self, r: Round) -> F {
        self.re.abs().add(self.im.abs(), r)
    }
    /// True if either part is NaN (`nan_test_c`).
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    /// `nan_unify_c`: a NaN in either part makes both parts canonical NaN.
    #[inline]
    pub fn canon(self) -> Self {
        if self.is_nan() {
            Complex::new(F::NAN, F::NAN)
        } else {
            self
        }
    }
}

/// `caxpy`: `y <- a * x + y`.
pub fn axpy<F: BlasFloat>(
    n: u64,
    a: Complex<F>,
    x: &[Complex<F>],
    incx: i64,
    y: &mut [Complex<F>],
    incy: i64,
    r: Round,
) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let yi = &mut y[at(iy)];
        *yi = yi.add(a.mul(x[at(ix)], r), r).canon();
        ix += incx;
        iy += incy;
    }
}

/// `ccopy`: `y <- x`, canonicalizing NaNs.
pub fn copy<F: BlasFloat>(n: u64, x: &[Complex<F>], incx: i64, y: &mut [Complex<F>], incy: i64) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        y[at(iy)] = x[at(ix)].canon();
        ix += incx;
        iy += incy;
    }
}

/// `cdotc`: `sum conj(x_i) * y_i`.
pub fn dotc<F: BlasFloat>(
    n: u64,
    x: &[Complex<F>],
    incx: i64,
    y: &[Complex<F>],
    incy: i64,
    r: Round,
) -> Complex<F> {
    let mut acc = Complex::ZERO;
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        acc = acc.add(x[at(ix)].conj().mul(y[at(iy)], r), r);
        ix += incx;
        iy += incy;
    }
    acc.canon()
}

/// `cdotu`: `sum x_i * y_i`, unconjugated.
pub fn dotu<F: BlasFloat>(
    n: u64,
    x: &[Complex<F>],
    incx: i64,
    y: &[Complex<F>],
    incy: i64,
    r: Round,
) -> Complex<F> {
    let mut acc = Complex::ZERO;
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        acc = acc.add(x[at(ix)].mul(y[at(iy)], r), r);
        ix += incx;
        iy += incy;
    }
    acc.canon()
}

/// `cscal`: `x <- a * x`.
pub fn scal<F: BlasFloat>(n: u64, a: Complex<F>, x: &mut [Complex<F>], incx: u64, r: Round) {
    let mut ix = 0usize;
    for _ in 0..n {
        x[ix] = a.mul(x[ix], r).canon();
        ix += incx as usize;
    }
}

/// `cswap`: exchange `x` and `y` (unsigned strides), canonicalizing NaNs.
pub fn swap<F: BlasFloat>(
    n: u64,
    x: &mut [Complex<F>],
    incx: u64,
    y: &mut [Complex<F>],
    incy: u64,
) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let (xi, yi) = (x[ix], y[iy]);
        x[ix] = yi.canon();
        y[iy] = xi.canon();
        ix += incx as usize;
        iy += incy as usize;
    }
}

/// `csrot`: apply `[c s; -s c]` with complex `c` and `s`.
pub fn rot<F: BlasFloat>(
    n: u64,
    x: &mut [Complex<F>],
    incx: i64,
    y: &mut [Complex<F>],
    incy: i64,
    c: Complex<F>,
    s: Complex<F>,
    r: Round,
) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let (xi, yi) = (x[at(ix)], y[at(iy)]);
        let tmp = c.mul(xi, r).add(s.mul(yi, r), r);
        y[at(iy)] = c.mul(yi, r).sub(s.mul(xi, r), r).canon();
        x[at(ix)] = tmp.canon();
        ix += incx;
        iy += incy;
    }
}

/// `scasum`: `sum |re_i| + |im_i|`. Returns `+0` for `incx < 1` or an
/// overflowing stride.
pub fn asum<F: BlasFloat>(n: u64, x: &[Complex<F>], incx: i64, r: Round) -> F {
    let mut temp = F::ZERO;
    if incx < 1 || stride_overflows(n, incx as u64) {
        return crate::canon(temp);
    }
    for i in 0..n as usize {
        temp = temp.add(x[i * incx as usize].abs1(r), r);
    }
    crate::canon(temp)
}

/// `scnrm2`: the Euclidean norm over the `2n` real components by the
/// running-scale recurrence (here with the ratio squared, unlike the real
/// `nrm2`; both forms are SoftBLAS's).
pub fn nrm2<F: BlasFloat>(n: u64, x: &[Complex<F>], incx: u64, r: Round) -> F {
    let (zero, one) = (F::ZERO, F::ONE);
    if n < 1 || incx < 1 || stride_overflows(n, incx) {
        return zero;
    }
    let mut scale = zero;
    let mut ssq = one;
    let mut ix = 0usize;
    for _ in 0..n {
        for part in [x[ix].re, x[ix].im] {
            if crate::ne(part, zero) {
                let absxi = part.abs();
                if scale.lt(absxi) {
                    let q = scale.div(absxi, r);
                    ssq = one.add(ssq.mul(q.mul(q, r), r), r);
                    scale = absxi;
                } else {
                    let q = absxi.div(scale, r);
                    ssq = ssq.add(q.mul(q, r), r);
                }
            }
        }
        ix += incx as usize;
    }
    crate::canon(scale.mul(ssq.sqrt(r), r))
}

/// `icamax`: 0-based index of the first element of largest `|re| + |im|`,
/// 0 for `n < 2` or `incx == 0`. NaN compares greater (see [`crate`] docs).
pub fn iamax<F: BlasFloat>(n: u64, x: &[Complex<F>], incx: u64, r: Round) -> u64 {
    if n < 1 || incx == 0 || n == 1 {
        return 0;
    }
    let mut imax = 0u64;
    let mut ix = incx as usize;
    let mut smax = x[0].abs1(r);
    for i in 1..n {
        let mag = x[ix].abs1(r);
        if gt(mag, smax) {
            imax = i;
            smax = mag;
        }
        ix += incx as usize;
    }
    imax
}

/// `cgemv`: `y <- alpha * op(A) * x + beta * y`, with `op` one of `A`,
/// `A^T`, `A^H`.
pub fn gemv<F: BlasFloat>(
    layout: Layout,
    trans: Trans,
    m: u64,
    n: u64,
    alpha: Complex<F>,
    a: &[Complex<F>],
    lda: u64,
    x: &[Complex<F>],
    incx: i64,
    beta: Complex<F>,
    y: &mut [Complex<F>],
    incy: u64,
    r: Round,
) {
    if incx < 1 || incy < 1 {
        return;
    }
    let (incx, incy, lda) = (incx as usize, incy as usize, lda as usize);
    let colmajor = layout == Layout::ColMajor;
    let (rows, inner) = if trans.is_trans() { (n, m) } else { (m, n) };
    for i in 0..rows as usize {
        let mut dot = Complex::ZERO;
        for j in 0..inner as usize {
            let idx = match (trans.is_trans(), colmajor) {
                (false, true) => i + j * lda,
                (false, false) => i * lda + j,
                (true, true) => j + i * lda,
                (true, false) => j * lda + i,
            };
            let mut aij = a[idx];
            if trans.is_conj() {
                aij = aij.conj();
            }
            dot = dot.add(aij.mul(x[j * incx], r), r);
        }
        let yi = &mut y[i * incy];
        *yi = alpha.mul(dot, r).add(beta.mul(*yi, r), r).canon();
    }
}

/// `cgemm`: `C <- alpha * op(A) * op(B) + beta * C`, row-major as the real
/// [`crate::gemm`], with `op` one of `A`, `A^T`, `A^H`.
pub fn gemm<F: BlasFloat>(
    transa: Trans,
    transb: Trans,
    m: u64,
    n: u64,
    p: u64,
    alpha: Complex<F>,
    a: &[Complex<F>],
    lda: u64,
    b: &[Complex<F>],
    ldb: u64,
    beta: Complex<F>,
    c: &mut [Complex<F>],
    ldc: u64,
    r: Round,
) {
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
    let (ta, tb) = (transa.is_trans(), transb.is_trans());
    for i in 0..m as usize {
        for j in 0..p as usize {
            let mut dot = Complex::ZERO;
            for k in 0..n as usize {
                let ia = if ta { i + k * lda } else { k + i * lda };
                let ib = if tb { k + j * ldb } else { j + k * ldb };
                let mut av = a[ia];
                let mut bv = b[ib];
                if transa.is_conj() {
                    av = av.conj();
                }
                if transb.is_conj() {
                    bv = bv.conj();
                }
                dot = dot.add(av.mul(bv, r), r);
            }
            let cij = &mut c[j + i * ldc];
            *cij = alpha.mul(dot, r).add(beta.mul(*cij, r), r).canon();
        }
    }
}
