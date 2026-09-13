//! Level 1: vector routines. Each function names the SoftBLAS source it
//! reproduces (`s*.c`; the `d`/`h`/`q` variants are identical modulo type).

use crate::{at, canon, ge, gt, ne, start, stride_overflows, BlasFloat, Round, SoftFloat};

/// `sasum`: the sum of `|x_i|`. Returns `+0` for `incx == 0` or a stride
/// that would overflow.
pub fn asum<F: BlasFloat>(n: u64, x: &[F], incx: u64, r: Round) -> F {
    let mut temp = F::ZERO;
    if incx < 1 || stride_overflows(n, incx) {
        return canon(temp);
    }
    for i in 0..n {
        temp = temp.add(x[(i * incx) as usize].abs(), r);
    }
    canon(temp)
}

/// `saxpy`: `y <- a * x + y`. Negative strides walk the vector backwards
/// from its last element, as in reference BLAS.
pub fn axpy<F: BlasFloat>(n: u64, a: F, x: &[F], incx: i64, y: &mut [F], incy: i64, r: Round) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let yi = &mut y[at(iy)];
        *yi = canon((*yi).add(a.mul(x[at(ix)], r), r));
        ix += incx;
        iy += incy;
    }
}

/// `scopy`: `y <- x`, canonicalizing NaNs on the way.
pub fn copy<F: BlasFloat>(n: u64, x: &[F], incx: i64, y: &mut [F], incy: i64) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        y[at(iy)] = canon(x[at(ix)]);
        ix += incx;
        iy += incy;
    }
}

/// `sdot`: `sum x_i * y_i`, accumulated left to right from `+0`.
pub fn dot<F: BlasFloat>(n: u64, x: &[F], incx: i64, y: &[F], incy: i64, r: Round) -> F {
    let mut acc = F::ZERO;
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        acc = acc.add(x[at(ix)].mul(y[at(iy)], r), r);
        ix += incx;
        iy += incy;
    }
    canon(acc)
}

/// `sdsdot`/`hsdot` core: `alpha + sum x_i * y_i` accumulated in the wider
/// format `W`, the result narrowed back to `F` under `r`.
pub fn dot_wide<F: BlasFloat, W: BlasFloat>(
    n: u64,
    alpha: F,
    x: &[F],
    incx: i64,
    y: &[F],
    incy: i64,
    r: Round,
) -> F {
    let widen = |v: F| -> W { sdfloat::convert(v, r) };
    let mut acc: W = widen(alpha);
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        acc = acc.add(widen(x[at(ix)]).mul(widen(y[at(iy)]), r), r);
        ix += incx;
        iy += incy;
    }
    canon(sdfloat::convert::<W, F>(acc, r))
}

/// `sdsdot`: binary32 dot product accumulated in binary64, plus `alpha`.
pub fn sdsdot(n: u64, alpha: F32, x: &[F32], incx: i64, y: &[F32], incy: i64, r: Round) -> F32 {
    dot_wide::<F32, F64>(n, alpha, x, incx, y, incy, r)
}

/// `hsdot`: binary16 dot product accumulated in binary32, plus `alpha`,
/// returned in binary32 (SoftBLAS returns `float32_t` here).
pub fn hsdot(n: u64, alpha: F16, x: &[F16], incx: i64, y: &[F16], incy: i64, r: Round) -> F32 {
    let widen = |v: F16| -> F32 { v.to_f32() };
    let mut acc = widen(alpha);
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        acc = acc.add(widen(x[at(ix)]).mul(widen(y[at(iy)]), r), r);
        ix += incx;
        iy += incy;
    }
    canon(acc)
}

use crate::{F16, F32, F64};

/// `snrm2`: the Euclidean norm by the classic running-scale recurrence.
/// SoftBLAS forms `ssq` from the ratio of squares, not the square of the
/// ratio, and that order is kept. Returns `+0` for `n == 0` or a bad stride
/// and `|x_0|` for `n == 1`.
pub fn nrm2<F: BlasFloat>(n: u64, x: &[F], incx: u64, r: Round) -> F {
    if n < 1 || incx < 1 || stride_overflows(n, incx) {
        return canon(F::ZERO);
    }
    if n == 1 {
        return canon(x[0].abs());
    }
    let mut scale = F::ZERO;
    let mut ssq = F::ONE;
    let mut ix = 0usize;
    for _ in 0..n {
        let xi = x[ix];
        if ne(xi, F::ZERO) {
            let absxi = xi.abs();
            if scale.lt(absxi) {
                // ssq = 1 + ssq * ((scale * scale) / (absxi * absxi))
                let q = scale.mul(scale, r).div(absxi.mul(absxi, r), r);
                ssq = F::ONE.add(ssq.mul(q, r), r);
                scale = absxi;
            } else {
                // ssq = ssq + (absxi * absxi) / (scale * scale)
                let q = absxi.mul(absxi, r).div(scale.mul(scale, r), r);
                ssq = ssq.add(q, r);
            }
        }
        ix += incx as usize;
    }
    canon(scale.mul(ssq.sqrt(r), r))
}

/// `snrm2_B`: the Euclidean norm by E. Anderson's safe-scaling algorithm
/// (ACM TOMS Alg. 978), three fixed-threshold accumulators.
pub fn nrm2_b<F: BlasFloat>(n: u64, x: &[F], incx: u64, r: Round) -> F {
    let (zero, one) = (F::ZERO, F::ONE);
    if n < 1 || incx < 1 || stride_overflows(n, incx) {
        return canon(zero);
    }
    let (mut asml, mut amed, mut abig) = (zero, zero, zero);
    let mut notbig = true;
    let mut ix = 0usize;
    for _ in 0..n {
        let ax = x[ix].abs();
        if gt(ax, F::TBIG) {
            let t = ax.mul(F::SBIG, r);
            abig = abig.add(t.mul(t, r), r);
            notbig = false;
        } else if ax.lt(F::TSML) {
            if notbig {
                let t = ax.mul(F::SSML, r);
                asml = asml.add(t.mul(t, r), r);
            }
        } else {
            amed = amed.add(ax.mul(ax, r), r);
        }
        ix += incx as usize;
    }
    let (scl, sumsq);
    if gt(abig, zero) {
        if gt(amed, zero) || ne(amed, amed) {
            abig = abig.add(amed.mul(F::SBIG, r).mul(F::SBIG, r), r);
        }
        scl = one.div(F::SBIG, r);
        sumsq = abig;
    } else if gt(asml, zero) {
        if gt(amed, zero) || ne(amed, amed) {
            let am = amed.sqrt(r);
            let as_ = asml.sqrt(r).div(F::SSML, r);
            let (ymin, ymax) = if gt(as_, am) { (am, as_) } else { (as_, am) };
            let q = ymin.div(ymax, r);
            scl = one;
            sumsq = ymax.mul(ymax, r).mul(one.add(q.mul(q, r), r), r);
        } else {
            scl = one.div(F::SSML, r);
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }
    canon(scl.mul(sumsq.sqrt(r), r))
}

/// `srot`: apply the plane rotation `[c s; -s c]` to `(x_i, y_i)`.
pub fn rot<F: BlasFloat>(
    n: u64,
    x: &mut [F],
    incx: u64,
    y: &mut [F],
    incy: u64,
    c: F,
    s: F,
    r: Round,
) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let (xi, yi) = (x[ix], y[iy]);
        let tmp = c.mul(xi, r).add(s.mul(yi, r), r);
        y[iy] = canon(c.mul(yi, r).sub(s.mul(xi, r), r));
        x[ix] = canon(tmp);
        ix += incx as usize;
        iy += incy as usize;
    }
}

/// `srotg`: construct a Givens rotation zeroing `b`. On return `a` holds
/// `r = sign(roe) * sqrt(a^2 + b^2)` and `b` holds the reconstruction
/// scalar `z`.
///
/// SoftBLAS writes these four outputs without `nan_unify`, the only place
/// (with `rotmg`) it does so; a NaN input therefore comes back from the C
/// with SoftFloat's propagated payload and sign. This port returns the
/// canonical NaN, consistent with every other routine.
pub fn rotg<F: BlasFloat>(a: &mut F, b: &mut F, c: &mut F, s: &mut F, r: Round) {
    let (zero, one, neg) = (F::ZERO, F::ONE, F::NEG_ONE);
    let (av, bv) = (*a, *b);
    let (abs_a, abs_b) = (av.abs(), bv.abs());
    let roe = if gt(abs_a, abs_b) { av } else { bv };
    let scale = abs_a.add(abs_b, r);
    let (rr, z);
    if scale.eq(zero) {
        *c = one;
        *s = zero;
        rr = zero;
        z = zero;
    } else {
        let as_ = av.div(scale, r);
        let bs = bv.div(scale, r);
        let mag = scale.mul(as_.mul(as_, r).add(bs.mul(bs, r), r).sqrt(r), r);
        let signed = (if roe.lt(zero) { neg } else { one }).mul(mag, r);
        rr = signed;
        *c = av.div(rr, r);
        *s = bv.div(rr, r);
        let (cv, sv) = (*c, *s);
        let mut zz = one;
        if gt(abs_a, abs_b) {
            zz = sv;
        }
        if ge(abs_b, abs_a) && ne(cv, zero) {
            zz = one.div(cv, r);
        }
        z = zz;
    }
    *a = canon(rr);
    *b = canon(z);
    *c = canon(*c);
    *s = canon(*s);
}

/// `srotm`: apply the modified Givens rotation `H` selected by `p[0]`.
pub fn rotm<F: BlasFloat>(
    n: u64,
    x: &mut [F],
    incx: u64,
    y: &mut [F],
    incy: u64,
    p: &[F; 5],
    r: Round,
) {
    let flag = p[0];
    let (zero, one, neg_one, neg_two) = (F::ZERO, F::ONE, F::NEG_ONE, F::NEG_TWO);
    if flag.eq(neg_two) {
        return;
    }
    let (h11, h21, h12, h22) = if flag.eq(neg_one) {
        (p[1], p[2], p[3], p[4])
    } else if flag.eq(zero) {
        (one, p[2], p[3], one)
    } else {
        (p[1], neg_one, one, p[4])
    };
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let (w, z) = (x[ix], y[iy]);
        x[ix] = canon(w.mul(h11, r).add(z.mul(h12, r), r));
        y[iy] = canon(w.mul(h21, r).add(z.mul(h22, r), r));
        ix += incx as usize;
        iy += incy as usize;
    }
}

/// `srotmg`: construct a modified Givens transformation. Writes the flag to
/// `p[0]` and the `H` entries SoftBLAS writes for that flag; when
/// `d2 * y1 == 0` only `p[0] = -2` is written and `d1`, `d2`, `x1` are
/// left untouched. Outputs are NaN-canonicalized (SoftBLAS leaves them raw;
/// see [`rotg`]).
pub fn rotmg<F: BlasFloat>(d1: &mut F, d2: &mut F, x1: &mut F, y1: F, p: &mut [F; 5], r: Round) {
    let (zero, one, neg_one, neg_two) = (F::ZERO, F::ONE, F::NEG_ONE, F::NEG_TWO);
    let (gam, gamsq, rgam, rgamsq, inf) = (F::GAM, F::GAMSQ, F::RGAM, F::RGAMSQ, F::INF);
    let flag;
    let (mut dd1, mut dd2, mut xx1) = (*d1, *d2, *x1);
    let (mut h11, mut h21, mut h12, mut h22) = (zero, zero, zero, zero);
    if dd1.lt(zero) {
        flag = neg_one;
        dd1 = zero;
        dd2 = zero;
        xx1 = zero;
        // h's already zero
        write_back(d1, d2, x1, p, dd1, dd2, xx1, flag, h11, h21, h12, h22);
        return;
    }
    let p2 = dd2.mul(y1, r);
    if p2.eq(zero) {
        p[0] = neg_two;
        return;
    }
    let p1 = dd1.mul(xx1, r);
    let q2 = p2.mul(y1, r);
    let q1 = p1.mul(xx1, r);
    let mut fl;
    if gt(q1.abs(), q2.abs()) {
        h21 = neg_one.mul(y1, r).div(xx1, r);
        h12 = p2.div(p1, r);
        let u = one.sub(h12.mul(h21, r), r);
        if u.le(zero) {
            fl = neg_one;
            dd1 = zero;
            dd2 = zero;
            xx1 = zero;
            h11 = zero;
            h21 = zero;
            h12 = zero;
            h22 = zero;
        } else {
            fl = zero;
            dd1 = dd1.div(u, r);
            dd2 = dd2.div(u, r);
            xx1 = xx1.mul(u, r);
        }
    } else if q2.lt(zero) {
        fl = neg_one;
        dd1 = zero;
        dd2 = zero;
        xx1 = zero;
        h11 = zero;
        h21 = zero;
        h12 = zero;
        h22 = zero;
    } else {
        fl = one;
        h11 = p1.div(p2, r);
        h22 = xx1.div(y1, r);
        let u = one.add(h11.mul(h22, r), r);
        let tmp = dd2.div(u, r);
        dd2 = dd1.div(u, r);
        dd1 = tmp;
        xx1 = y1.mul(u, r);
    }
    // Rescale d1 into [1/gamsq, gamsq]. The upper loops are guarded by a
    // finiteness test because ge(inf, gamsq) is true forever. The first loop
    // is additionally restricted to positive d1: SoftBLAS's `d1 != 0 &&
    // d1 <= rgamsq` never exits for a negative d1 (which a negative d2 and
    // an underflowing q2 can produce), so on those inputs the C hangs and
    // this port returns the unscaled values instead.
    while zero.lt(dd1) && dd1.le(rgamsq) {
        fix_flag(
            &mut fl, &mut h11, &mut h21, &mut h12, &mut h22, zero, one, neg_one,
        );
        dd1 = dd1.mul(gamsq, r);
        xx1 = xx1.mul(rgam, r);
        h11 = h11.mul(rgam, r);
        h12 = h12.mul(rgam, r);
    }
    while dd1.abs().lt(inf) && ge(dd1, gamsq) {
        fix_flag(
            &mut fl, &mut h11, &mut h21, &mut h12, &mut h22, zero, one, neg_one,
        );
        dd1 = dd1.mul(rgamsq, r);
        xx1 = xx1.mul(gam, r);
        h11 = h11.mul(gam, r);
        h12 = h12.mul(gam, r);
    }
    // Rescale d2 into [1/gamsq, gamsq], by magnitude.
    while ne(dd2, zero) && dd2.abs().le(rgamsq) {
        fix_flag(
            &mut fl, &mut h11, &mut h21, &mut h12, &mut h22, zero, one, neg_one,
        );
        dd2 = dd2.mul(gamsq, r);
        h21 = h21.mul(rgam, r);
        h22 = h22.mul(rgam, r);
    }
    while dd2.abs().lt(inf) && ge(dd2.abs(), gamsq) {
        fix_flag(
            &mut fl, &mut h11, &mut h21, &mut h12, &mut h22, zero, one, neg_one,
        );
        dd2 = dd2.mul(rgamsq, r);
        h21 = h21.mul(gam, r);
        h22 = h22.mul(gam, r);
    }
    write_back(d1, d2, x1, p, dd1, dd2, xx1, fl, h11, h21, h12, h22);
}

/// The rescaling loops' flag promotion: once rescaled, `H` is stored in
/// full (`flag = -1`), with the implicit unit entries made explicit.
#[inline]
fn fix_flag<F: BlasFloat>(
    fl: &mut F,
    h11: &mut F,
    h21: &mut F,
    h12: &mut F,
    h22: &mut F,
    zero: F,
    one: F,
    neg_one: F,
) {
    if (*fl).eq(zero) {
        *fl = neg_one;
        *h11 = one;
        *h22 = one;
    } else {
        *fl = neg_one;
        *h21 = neg_one;
        *h12 = one;
    }
}

#[inline]
fn write_back<F: BlasFloat>(
    d1: &mut F,
    d2: &mut F,
    x1: &mut F,
    p: &mut [F; 5],
    dd1: F,
    dd2: F,
    xx1: F,
    flag: F,
    h11: F,
    h21: F,
    h12: F,
    h22: F,
) {
    *d1 = canon(dd1);
    *d2 = canon(dd2);
    *x1 = canon(xx1);
    p[0] = flag;
    if flag.eq(F::NEG_ONE) {
        p[1] = canon(h11);
        p[2] = canon(h21);
        p[3] = canon(h12);
        p[4] = canon(h22);
    } else if flag.eq(F::ZERO) {
        p[2] = canon(h21);
        p[3] = canon(h12);
    } else {
        p[1] = canon(h11);
        p[4] = canon(h22);
    }
}

/// `sscal`: `x <- a * x`.
pub fn scal<F: BlasFloat>(n: u64, a: F, x: &mut [F], incx: u64, r: Round) {
    let mut ix = 0usize;
    for _ in 0..n {
        x[ix] = canon(a.mul(x[ix], r));
        ix += incx as usize;
    }
}

/// `sswap`: exchange `x` and `y`, canonicalizing NaNs.
pub fn swap<F: BlasFloat>(n: u64, x: &mut [F], incx: i64, y: &mut [F], incy: i64) {
    if n == 0 {
        return;
    }
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let (xi, yi) = (x[at(ix)], y[at(iy)]);
        x[at(ix)] = canon(yi);
        y[at(iy)] = canon(xi);
        ix += incx;
        iy += incy;
    }
}

/// `isamax`: the 0-based index of the first element of largest magnitude,
/// 0 for `n < 2` or `incx == 0`. Because `gt` is `!le`, a NaN element is
/// taken as larger than the running maximum.
pub fn iamax<F: BlasFloat>(n: u64, x: &[F], incx: u64) -> u64 {
    if n < 1 || incx == 0 || n == 1 {
        return 0;
    }
    let mut imax = 0u64;
    let mut ix = incx as usize;
    let mut smax = x[0].abs();
    for i in 1..n {
        let ax = x[ix].abs();
        if gt(ax, smax) {
            imax = i;
            smax = ax;
        }
        ix += incx as usize;
    }
    imax
}
