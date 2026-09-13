//! Differential test of every sdblas routine against SoftBLAS (C), all four
//! widths, four rounding modes (SoftBLAS's `'a'` is not Hoon's `%a`, so
//! `Round::Away` has no oracle here). Vector lengths, strides (including
//! negative and zero where the routine accepts them), layouts, transposes
//! and operand bit patterns are randomized; every output element and return
//! value is compared bit for bit. `SDBLAS_ORACLE_N` sets trials per routine
//! and width (default 3000).

#![allow(clippy::all)]

use sdblas::complex::{self, Complex};
use sdblas::{BlasFloat, Layout, Round, Trans, F128, F16, F32, F64};
use sdblas_oracle::{C, Q};
use sdfloat::Format;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn bits128(&mut self) -> u128 {
        ((self.next() as u128) << 64) | self.next() as u128
    }
}

fn trials() -> usize {
    std::env::var("SDBLAS_ORACLE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000)
}

const MODES: [(Round, u8); 4] = [
    (Round::NearEven, b'n'),
    (Round::Zero, b'z'),
    (Round::Up, b'u'),
    (Round::Down, b'd'),
];

/// One operand: mostly modest finite values so accumulations stay
/// interesting, with specials mixed in.
fn sample<F: Format>(rng: &mut Rng) -> u128 {
    let sign = if rng.below(2) == 0 { 0 } else { F::SIGN_MASK };
    let emax = (1u128 << F::EXP_BITS) - 1;
    let bias = F::BIAS as u128;
    let field = match rng.below(20) {
        0 => 0,
        1 => emax,
        2 => emax - 1,
        3 => 1,
        4 => bias + F::FRAC_BITS as u128 - rng.below(3) as u128,
        5..=11 => bias.wrapping_add(rng.below(9) as u128).wrapping_sub(4),
        12 => bias.saturating_sub(20 + rng.below(30) as u128).max(1),
        13 => (bias + 20 + rng.below(30) as u128).min(emax - 1),
        _ => bias + rng.below(5) as u128 - 2,
    };
    let frac = match rng.below(6) {
        0 => 0,
        1 => F::FRAC_MASK,
        2 => 1 << (F::FRAC_BITS - 1),
        _ => rng.bits128() & F::FRAC_MASK,
    };
    let bits = match rng.below(40) {
        0 => 0,
        1 => F::INF_BITS,
        2 => F::NAN_BITS,
        3 => F::NAN_BITS | (rng.bits128() & F::FRAC_MASK),
        4 => F::MAX_BITS,
        5 => 1,
        6 => F::ONE_BITS,
        _ => (field << F::FRAC_BITS) | frac,
    };
    sign | bits
}

/// Width-specific oracle bindings over the common bit carriers.
trait Orc: BlasFloat + core::fmt::Debug {
    type T: Copy + PartialEq + core::fmt::Debug + Default;
    fn to_t(self) -> Self::T;
    fn from_t(t: Self::T) -> Self;
    fn o_asum(n: u64, x: &[Self::T], incx: u64, m: u8) -> Self::T;
    fn o_axpy(n: u64, a: Self::T, x: &[Self::T], incx: i64, y: &mut [Self::T], incy: i64, m: u8);
    fn o_copy(n: u64, x: &[Self::T], incx: i64, y: &mut [Self::T], incy: i64, m: u8);
    fn o_dot(n: u64, x: &[Self::T], incx: i64, y: &[Self::T], incy: i64, m: u8) -> Self::T;
    fn o_nrm2(n: u64, x: &[Self::T], incx: u64, m: u8) -> Self::T;
    fn o_nrm2_b(n: u64, x: &[Self::T], incx: u64, m: u8) -> Self::T;
    fn o_rot(
        n: u64,
        x: &mut [Self::T],
        incx: u64,
        y: &mut [Self::T],
        incy: u64,
        c: Self::T,
        s: Self::T,
        m: u8,
    );
    fn o_rotg(a: &mut Self::T, b: &mut Self::T, c: &mut Self::T, s: &mut Self::T, m: u8);
    fn o_rotm(
        n: u64,
        x: &mut [Self::T],
        incx: u64,
        y: &mut [Self::T],
        incy: u64,
        p: &[Self::T; 5],
        m: u8,
    );
    fn o_rotmg(
        d1: &mut Self::T,
        d2: &mut Self::T,
        x1: &mut Self::T,
        y1: Self::T,
        p: &mut [Self::T; 5],
        m: u8,
    );
    fn o_scal(n: u64, a: Self::T, x: &mut [Self::T], incx: u64, m: u8);
    fn o_swap(n: u64, x: &mut [Self::T], incx: i64, y: &mut [Self::T], incy: i64, m: u8);
    fn o_iamax(n: u64, x: &[Self::T], incx: u64) -> u64;
    fn o_gemv(
        layout: u8,
        trans: u8,
        mm: u64,
        n: u64,
        alpha: Self::T,
        a: &[Self::T],
        lda: u64,
        x: &[Self::T],
        incx: i64,
        beta: Self::T,
        y: &mut [Self::T],
        incy: u64,
        m: u8,
    );
    fn o_gemm(
        ta: u8,
        tb: u8,
        mm: u64,
        n: u64,
        p: u64,
        alpha: Self::T,
        a: &[Self::T],
        lda: u64,
        b: &[Self::T],
        ldb: u64,
        beta: Self::T,
        c: &mut [Self::T],
        ldc: u64,
        m: u8,
    );
    fn o_caxpy(
        n: u64,
        a: C<Self::T>,
        x: &[C<Self::T>],
        incx: i64,
        y: &mut [C<Self::T>],
        incy: i64,
        m: u8,
    );
    fn o_ccopy(n: u64, x: &[C<Self::T>], incx: i64, y: &mut [C<Self::T>], incy: i64, m: u8);
    fn o_cdotc(
        n: u64,
        x: &[C<Self::T>],
        incx: i64,
        y: &[C<Self::T>],
        incy: i64,
        m: u8,
    ) -> C<Self::T>;
    fn o_cdotu(
        n: u64,
        x: &[C<Self::T>],
        incx: i64,
        y: &[C<Self::T>],
        incy: i64,
        m: u8,
    ) -> C<Self::T>;
    fn o_cscal(n: u64, a: C<Self::T>, x: &mut [C<Self::T>], incx: u64, m: u8);
    fn o_cswap(n: u64, x: &mut [C<Self::T>], incx: u64, y: &mut [C<Self::T>], incy: u64, m: u8);
    fn o_crot(
        n: u64,
        x: &mut [C<Self::T>],
        incx: i64,
        y: &mut [C<Self::T>],
        incy: i64,
        c: C<Self::T>,
        s: C<Self::T>,
        m: u8,
    );
    fn o_casum(n: u64, x: &[C<Self::T>], incx: i64, m: u8) -> Self::T;
    fn o_cnrm2(n: u64, x: &[C<Self::T>], incx: u64, m: u8) -> Self::T;
    fn o_ciamax(n: u64, x: &[C<Self::T>], incx: u64, m: u8) -> u64;
    fn o_cgemv(
        layout: u8,
        trans: u8,
        mm: u64,
        n: u64,
        alpha: C<Self::T>,
        a: &[C<Self::T>],
        lda: u64,
        x: &[C<Self::T>],
        incx: i64,
        beta: C<Self::T>,
        y: &mut [C<Self::T>],
        incy: u64,
        m: u8,
    );
    fn o_cgemm(
        ta: u8,
        tb: u8,
        mm: u64,
        n: u64,
        p: u64,
        alpha: C<Self::T>,
        a: &[C<Self::T>],
        lda: u64,
        b: &[C<Self::T>],
        ldb: u64,
        beta: C<Self::T>,
        c: &mut [C<Self::T>],
        ldc: u64,
        m: u8,
    );
}

macro_rules! orc {
    ($F:ident, $T:ty, $m:ident, |$s:ident| $to:expr, |$t:ident| $from:expr) => {
        impl Orc for $F {
            type T = $T;
            fn to_t($s) -> $T { $to }
            fn from_t($t: $T) -> Self { $from }
            fn o_asum(n: u64, x: &[$T], incx: u64, m: u8) -> $T { sdblas_oracle::$m::asum(n, x, incx, m) }
            fn o_axpy(n: u64, a: $T, x: &[$T], incx: i64, y: &mut [$T], incy: i64, m: u8) { sdblas_oracle::$m::axpy(n, a, x, incx, y, incy, m) }
            fn o_copy(n: u64, x: &[$T], incx: i64, y: &mut [$T], incy: i64, m: u8) { sdblas_oracle::$m::copy(n, x, incx, y, incy, m) }
            fn o_dot(n: u64, x: &[$T], incx: i64, y: &[$T], incy: i64, m: u8) -> $T { sdblas_oracle::$m::dot(n, x, incx, y, incy, m) }
            fn o_nrm2(n: u64, x: &[$T], incx: u64, m: u8) -> $T { sdblas_oracle::$m::nrm2(n, x, incx, m) }
            fn o_nrm2_b(n: u64, x: &[$T], incx: u64, m: u8) -> $T { sdblas_oracle::$m::nrm2_b(n, x, incx, m) }
            fn o_rot(n: u64, x: &mut [$T], incx: u64, y: &mut [$T], incy: u64, c: $T, s: $T, m: u8) { sdblas_oracle::$m::rot(n, x, incx, y, incy, c, s, m) }
            fn o_rotg(a: &mut $T, b: &mut $T, c: &mut $T, s: &mut $T, m: u8) { sdblas_oracle::$m::rotg(a, b, c, s, m) }
            fn o_rotm(n: u64, x: &mut [$T], incx: u64, y: &mut [$T], incy: u64, p: &[$T; 5], m: u8) { sdblas_oracle::$m::rotm(n, x, incx, y, incy, p, m) }
            fn o_rotmg(d1: &mut $T, d2: &mut $T, x1: &mut $T, y1: $T, p: &mut [$T; 5], m: u8) { sdblas_oracle::$m::rotmg(d1, d2, x1, y1, p, m) }
            fn o_scal(n: u64, a: $T, x: &mut [$T], incx: u64, m: u8) { sdblas_oracle::$m::scal(n, a, x, incx, m) }
            fn o_swap(n: u64, x: &mut [$T], incx: i64, y: &mut [$T], incy: i64, m: u8) { sdblas_oracle::$m::swap(n, x, incx, y, incy, m) }
            fn o_iamax(n: u64, x: &[$T], incx: u64) -> u64 { sdblas_oracle::$m::iamax(n, x, incx) }
            fn o_gemv(layout: u8, trans: u8, mm: u64, n: u64, alpha: $T, a: &[$T], lda: u64, x: &[$T], incx: i64, beta: $T, y: &mut [$T], incy: u64, m: u8) {
                sdblas_oracle::$m::gemv(layout, trans, mm, n, alpha, a, lda, x, incx, beta, y, incy, m)
            }
            fn o_gemm(ta: u8, tb: u8, mm: u64, n: u64, p: u64, alpha: $T, a: &[$T], lda: u64, b: &[$T], ldb: u64, beta: $T, c: &mut [$T], ldc: u64, m: u8) {
                sdblas_oracle::$m::gemm(ta, tb, mm, n, p, alpha, a, lda, b, ldb, beta, c, ldc, m)
            }
            fn o_caxpy(n: u64, a: C<$T>, x: &[C<$T>], incx: i64, y: &mut [C<$T>], incy: i64, m: u8) { sdblas_oracle::$m::caxpy(n, a, x, incx, y, incy, m) }
            fn o_ccopy(n: u64, x: &[C<$T>], incx: i64, y: &mut [C<$T>], incy: i64, m: u8) { sdblas_oracle::$m::ccopy(n, x, incx, y, incy, m) }
            fn o_cdotc(n: u64, x: &[C<$T>], incx: i64, y: &[C<$T>], incy: i64, m: u8) -> C<$T> { sdblas_oracle::$m::cdotc(n, x, incx, y, incy, m) }
            fn o_cdotu(n: u64, x: &[C<$T>], incx: i64, y: &[C<$T>], incy: i64, m: u8) -> C<$T> { sdblas_oracle::$m::cdotu(n, x, incx, y, incy, m) }
            fn o_cscal(n: u64, a: C<$T>, x: &mut [C<$T>], incx: u64, m: u8) { sdblas_oracle::$m::cscal(n, a, x, incx, m) }
            fn o_cswap(n: u64, x: &mut [C<$T>], incx: u64, y: &mut [C<$T>], incy: u64, m: u8) { sdblas_oracle::$m::cswap(n, x, incx, y, incy, m) }
            fn o_crot(n: u64, x: &mut [C<$T>], incx: i64, y: &mut [C<$T>], incy: i64, c: C<$T>, s: C<$T>, m: u8) { sdblas_oracle::$m::crot(n, x, incx, y, incy, c, s, m) }
            fn o_casum(n: u64, x: &[C<$T>], incx: i64, m: u8) -> $T { sdblas_oracle::$m::casum(n, x, incx, m) }
            fn o_cnrm2(n: u64, x: &[C<$T>], incx: u64, m: u8) -> $T { sdblas_oracle::$m::cnrm2(n, x, incx, m) }
            fn o_ciamax(n: u64, x: &[C<$T>], incx: u64, m: u8) -> u64 { sdblas_oracle::$m::ciamax(n, x, incx, m) }
            fn o_cgemv(layout: u8, trans: u8, mm: u64, n: u64, alpha: C<$T>, a: &[C<$T>], lda: u64, x: &[C<$T>], incx: i64, beta: C<$T>, y: &mut [C<$T>], incy: u64, m: u8) {
                sdblas_oracle::$m::cgemv(layout, trans, mm, n, alpha, a, lda, x, incx, beta, y, incy, m)
            }
            fn o_cgemm(ta: u8, tb: u8, mm: u64, n: u64, p: u64, alpha: C<$T>, a: &[C<$T>], lda: u64, b: &[C<$T>], ldb: u64, beta: C<$T>, c: &mut [C<$T>], ldc: u64, m: u8) {
                sdblas_oracle::$m::cgemm(ta, tb, mm, n, p, alpha, a, lda, b, ldb, beta, c, ldc, m)
            }
        }
    };
}
orc!(F16, u16, h, |self| self.0, |t| F16(t));
orc!(F32, u32, s, |self| self.0, |t| F32(t));
orc!(F64, u64, d, |self| self.0, |t| F64(t));
orc!(F128, Q, q, |self| Q::from_u128(self.0), |t| F128(
    t.to_u128()
));

// ------------------------------------------------------------- helpers ----

fn gen<F: Orc>(rng: &mut Rng) -> F {
    F::from_raw(sample::<F>(rng))
}
fn vec_of<F: Orc>(rng: &mut Rng, len: usize) -> Vec<F> {
    (0..len).map(|_| gen::<F>(rng)).collect()
}
fn to_t<F: Orc>(v: &[F]) -> Vec<F::T> {
    v.iter().map(|x| x.to_t()).collect()
}
fn from_t<F: Orc>(v: &[F::T]) -> Vec<F> {
    v.iter().map(|&x| F::from_t(x)).collect()
}
fn cvec<F: Orc>(rng: &mut Rng, len: usize) -> Vec<Complex<F>> {
    (0..len)
        .map(|_| Complex::new(gen::<F>(rng), gen::<F>(rng)))
        .collect()
}
fn c_to_t<F: Orc>(v: &[Complex<F>]) -> Vec<C<F::T>> {
    v.iter().map(|z| C(z.re.to_t(), z.im.to_t())).collect()
}
fn c_from_t<F: Orc>(v: &[C<F::T>]) -> Vec<Complex<F>> {
    v.iter()
        .map(|z| Complex::new(F::from_t(z.0), F::from_t(z.1)))
        .collect()
}
/// Buffer length that a vector of `n` elements with stride `inc` touches
/// (plus slack so a wrong stride shows up as a mismatch, not a panic).
fn len_for(n: u64, inc: i64) -> usize {
    if n == 0 {
        1
    } else {
        1 + (n as usize - 1) * inc.unsigned_abs() as usize + 2
    }
}
fn signed_inc(rng: &mut Rng) -> i64 {
    match rng.below(6) {
        0 => -2,
        1 => -1,
        2 => 2,
        3 => 3,
        _ => 1,
    }
}
fn unsigned_inc(rng: &mut Rng) -> u64 {
    match rng.below(8) {
        0 => 0,
        1 => 2,
        2 => 3,
        _ => 1,
    }
}
fn name<F: Format>() -> &'static str {
    match F::WIDTH {
        16 => "f16",
        32 => "f32",
        64 => "f64",
        _ => "f128",
    }
}

macro_rules! check_eq {
    ($ours:expr, $theirs:expr, $($ctx:tt)*) => {
        assert_eq!($ours, $theirs, $($ctx)*)
    };
}

// ------------------------------------------------------------ level 1 ----

fn level1<F: Orc>() {
    let mut rng = Rng(0x1234_5678_9abc_def1 ^ F::WIDTH as u64);
    for trial in 0..trials() {
        let n = rng.below(7);
        for (r, m) in MODES {
            let ctx = || format!("{} trial {trial} n={n} mode {:?}", name::<F>(), r);
            // asum / nrm2 / nrm2_b / iamax: unsigned strides, incl. 0.
            {
                let inc = unsigned_inc(&mut rng);
                let x = vec_of::<F>(&mut rng, len_for(n, inc as i64));
                let xt = to_t::<F>(&x);
                check_eq!(
                    sdblas::asum(n, &x, inc, r).to_t(),
                    F::o_asum(n, &xt, inc, m),
                    "asum {} inc={inc} x={x:?}",
                    ctx()
                );
                check_eq!(
                    sdblas::nrm2(n, &x, inc, r).to_t(),
                    F::o_nrm2(n, &xt, inc, m),
                    "nrm2 {} inc={inc} x={x:?}",
                    ctx()
                );
                check_eq!(
                    sdblas::nrm2_b(n, &x, inc, r).to_t(),
                    F::o_nrm2_b(n, &xt, inc, m),
                    "nrm2_b {} inc={inc} x={x:?}",
                    ctx()
                );
                check_eq!(
                    sdblas::iamax(n, &x, inc),
                    F::o_iamax(n, &xt, inc),
                    "iamax {} inc={inc} x={x:?}",
                    ctx()
                );
            }
            // axpy / copy / dot / swap: signed strides.
            {
                let (incx, incy) = (signed_inc(&mut rng), signed_inc(&mut rng));
                let a = gen::<F>(&mut rng);
                let x = vec_of::<F>(&mut rng, len_for(n, incx));
                let y0 = vec_of::<F>(&mut rng, len_for(n, incy));
                let (xt, y0t) = (to_t::<F>(&x), to_t::<F>(&y0));
                let mut y = y0.clone();
                let mut yt = y0t.clone();
                sdblas::axpy(n, a, &x, incx, &mut y, incy, r);
                F::o_axpy(n, a.to_t(), &xt, incx, &mut yt, incy, m);
                check_eq!(
                    y,
                    from_t::<F>(&yt),
                    "axpy {} a={a:?} incx={incx} incy={incy} x={x:?} y0={y0:?}",
                    ctx()
                );
                check_eq!(
                    sdblas::dot(n, &x, incx, &y0, incy, r).to_t(),
                    F::o_dot(n, &xt, incx, &y0t, incy, m),
                    "dot {} incx={incx} incy={incy} x={x:?} y={y0:?}",
                    ctx()
                );
                let mut y = y0.clone();
                let mut yt = y0t.clone();
                sdblas::copy(n, &x, incx, &mut y, incy);
                F::o_copy(n, &xt, incx, &mut yt, incy, m);
                check_eq!(
                    y,
                    from_t::<F>(&yt),
                    "copy {} incx={incx} incy={incy}",
                    ctx()
                );
                let (mut xs, mut ys) = (x.clone(), y0.clone());
                let (mut xst, mut yst) = (xt.clone(), y0t.clone());
                sdblas::swap(n, &mut xs, incx, &mut ys, incy);
                F::o_swap(n, &mut xst, incx, &mut yst, incy, m);
                check_eq!(
                    (xs, ys),
                    (from_t::<F>(&xst), from_t::<F>(&yst)),
                    "swap {} incx={incx} incy={incy}",
                    ctx()
                );
            }
            // scal / rot / rotm: unsigned strides (rot/rotm accept 0).
            {
                let (incx, incy) = (unsigned_inc(&mut rng).max(0), unsigned_inc(&mut rng));
                let a = gen::<F>(&mut rng);
                let x0 = vec_of::<F>(&mut rng, len_for(n, incx as i64));
                let y0 = vec_of::<F>(&mut rng, len_for(n, incy as i64));
                let (x0t, y0t) = (to_t::<F>(&x0), to_t::<F>(&y0));
                let mut x = x0.clone();
                let mut xt = x0t.clone();
                sdblas::scal(n, a, &mut x, incx, r);
                F::o_scal(n, a.to_t(), &mut xt, incx, m);
                check_eq!(
                    x,
                    from_t::<F>(&xt),
                    "scal {} a={a:?} incx={incx} x0={x0:?}",
                    ctx()
                );
                let (c, s) = (gen::<F>(&mut rng), gen::<F>(&mut rng));
                let (mut x, mut y) = (x0.clone(), y0.clone());
                let (mut xt, mut yt) = (x0t.clone(), y0t.clone());
                sdblas::rot(n, &mut x, incx, &mut y, incy, c, s, r);
                F::o_rot(n, &mut xt, incx, &mut yt, incy, c.to_t(), s.to_t(), m);
                check_eq!(
                    (x, y),
                    (from_t::<F>(&xt), from_t::<F>(&yt)),
                    "rot {} c={c:?} s={s:?} x0={x0:?} y0={y0:?}",
                    ctx()
                );
                let flag = [F::NEG_TWO, F::NEG_ONE, F::ZERO, F::ONE, gen::<F>(&mut rng)]
                    [rng.below(5) as usize];
                let p = [
                    flag,
                    gen(&mut rng),
                    gen(&mut rng),
                    gen(&mut rng),
                    gen(&mut rng),
                ];
                let pt: [F::T; 5] = core::array::from_fn(|i| p[i].to_t());
                let (mut x, mut y) = (x0.clone(), y0.clone());
                let (mut xt, mut yt) = (x0t.clone(), y0t.clone());
                sdblas::rotm(n, &mut x, incx, &mut y, incy, &p, r);
                F::o_rotm(n, &mut xt, incx, &mut yt, incy, &pt, m);
                check_eq!(
                    (x, y),
                    (from_t::<F>(&xt), from_t::<F>(&yt)),
                    "rotm {} p={p:?} x0={x0:?} y0={y0:?}",
                    ctx()
                );
            }
            // rotg / rotmg: scalars.
            {
                let (a0, b0) = (gen::<F>(&mut rng), gen::<F>(&mut rng));
                let (mut a, mut b, mut c, mut s) = (a0, b0, F::ZERO, F::ZERO);
                let (mut at, mut bt, mut ct, mut st) =
                    (a0.to_t(), b0.to_t(), F::ZERO.to_t(), F::ZERO.to_t());
                sdblas::rotg(&mut a, &mut b, &mut c, &mut s, r);
                F::o_rotg(&mut at, &mut bt, &mut ct, &mut st, m);
                // SoftBLAS does not nan_unify rotg/rotmg outputs, so raw
                // SoftFloat NaN payloads and signs come back; compare modulo
                // NaN payload (sdblas canonicalizes, like every other routine).
                let cn = |v: F::T| -> F {
                    let f = F::from_t(v);
                    if f.is_nan() {
                        F::NAN
                    } else {
                        f
                    }
                };
                check_eq!(
                    [a, b, c, s],
                    [cn(at), cn(bt), cn(ct), cn(st)],
                    "rotg {} a={a0:?} b={b0:?}",
                    ctx()
                );
                let (d10, d20, x10, y1) = (
                    gen::<F>(&mut rng),
                    gen::<F>(&mut rng),
                    gen::<F>(&mut rng),
                    gen::<F>(&mut rng),
                );
                let p0: [F; 5] = core::array::from_fn(|_| gen(&mut rng));
                let (mut d1, mut d2, mut x1, mut p) = (d10, d20, x10, p0);
                let (mut d1t, mut d2t, mut x1t) = (d10.to_t(), d20.to_t(), x10.to_t());
                let mut pt: [F::T; 5] = core::array::from_fn(|i| p0[i].to_t());
                sdblas::rotmg(&mut d1, &mut d2, &mut x1, y1, &mut p, r);
                // A negative d1 on exit means SoftBLAS's first rescaling
                // loop would never terminate on this input; skip the oracle.
                if !d1.lt(F::ZERO) {
                    F::o_rotmg(&mut d1t, &mut d2t, &mut x1t, y1.to_t(), &mut pt, m);
                    // Entries rotmg leaves untouched keep their raw input NaNs
                    // on both sides; canonicalize both for the comparison.
                    let cf = |f: F| -> F {
                        if f.is_nan() {
                            F::NAN
                        } else {
                            f
                        }
                    };
                    let pr: [F; 5] = core::array::from_fn(|i| cn(pt[i]));
                    let po: [F; 5] = core::array::from_fn(|i| cf(p[i]));
                    check_eq!(
                        ([cf(d1), cf(d2), cf(x1)], po),
                        ([cn(d1t), cn(d2t), cn(x1t)], pr),
                        "rotmg {} d1={d10:?} d2={d20:?} x1={x10:?} y1={y1:?} p0={p0:?}",
                        ctx()
                    );
                }
            }
        }
    }
}

// ----------------------------------------------------- level 1 complex ----

fn level1_complex<F: Orc>() {
    let mut rng = Rng(0x0fed_cba9_8765_4321 ^ F::WIDTH as u64);
    for trial in 0..trials() {
        let n = rng.below(6);
        for (r, m) in MODES {
            let ctx = || format!("{} trial {trial} n={n} mode {:?}", name::<F>(), r);
            let (incx, incy) = (signed_inc(&mut rng), signed_inc(&mut rng));
            let a = Complex::new(gen::<F>(&mut rng), gen::<F>(&mut rng));
            let at = C(a.re.to_t(), a.im.to_t());
            let x = cvec::<F>(&mut rng, len_for(n, incx));
            let y0 = cvec::<F>(&mut rng, len_for(n, incy));
            let (xt, y0t) = (c_to_t::<F>(&x), c_to_t::<F>(&y0));
            let mut y = y0.clone();
            let mut yt = y0t.clone();
            complex::axpy(n, a, &x, incx, &mut y, incy, r);
            F::o_caxpy(n, at, &xt, incx, &mut yt, incy, m);
            check_eq!(
                y,
                c_from_t::<F>(&yt),
                "caxpy {} a={a:?} incx={incx} incy={incy} x={x:?} y0={y0:?}",
                ctx()
            );
            let mut y = y0.clone();
            let mut yt = y0t.clone();
            complex::copy(n, &x, incx, &mut y, incy);
            F::o_ccopy(n, &xt, incx, &mut yt, incy, m);
            check_eq!(y, c_from_t::<F>(&yt), "ccopy {}", ctx());
            let dc = complex::dotc(n, &x, incx, &y0, incy, r);
            let dct = F::o_cdotc(n, &xt, incx, &y0t, incy, m);
            check_eq!(
                dc,
                Complex::new(F::from_t(dct.0), F::from_t(dct.1)),
                "cdotc {} x={x:?} y={y0:?}",
                ctx()
            );
            let du = complex::dotu(n, &x, incx, &y0, incy, r);
            let dut = F::o_cdotu(n, &xt, incx, &y0t, incy, m);
            check_eq!(
                du,
                Complex::new(F::from_t(dut.0), F::from_t(dut.1)),
                "cdotu {} x={x:?} y={y0:?}",
                ctx()
            );
            let (c, s) = (
                Complex::new(gen::<F>(&mut rng), gen::<F>(&mut rng)),
                Complex::new(gen::<F>(&mut rng), gen::<F>(&mut rng)),
            );
            let (mut xr, mut yr) = (x.clone(), y0.clone());
            let (mut xrt, mut yrt) = (xt.clone(), y0t.clone());
            complex::rot(n, &mut xr, incx, &mut yr, incy, c, s, r);
            F::o_crot(
                n,
                &mut xrt,
                incx,
                &mut yrt,
                incy,
                C(c.re.to_t(), c.im.to_t()),
                C(s.re.to_t(), s.im.to_t()),
                m,
            );
            check_eq!(
                (xr, yr),
                (c_from_t::<F>(&xrt), c_from_t::<F>(&yrt)),
                "crot {} c={c:?} s={s:?}",
                ctx()
            );
            // asum takes a signed stride but rejects < 1.
            check_eq!(
                complex::asum(n, &x, incx, r).to_t(),
                F::o_casum(n, &xt, incx, m),
                "casum {} incx={incx} x={x:?}",
                ctx()
            );
            // unsigned-stride routines
            let (ux, uy) = (unsigned_inc(&mut rng), unsigned_inc(&mut rng));
            let x = cvec::<F>(&mut rng, len_for(n, ux as i64));
            let y0 = cvec::<F>(&mut rng, len_for(n, uy as i64));
            let (xt, y0t) = (c_to_t::<F>(&x), c_to_t::<F>(&y0));
            check_eq!(
                complex::nrm2(n, &x, ux, r).to_t(),
                F::o_cnrm2(n, &xt, ux, m),
                "cnrm2 {} inc={ux} x={x:?}",
                ctx()
            );
            check_eq!(
                complex::iamax(n, &x, ux, r),
                F::o_ciamax(n, &xt, ux, m),
                "ciamax {} inc={ux} x={x:?}",
                ctx()
            );
            let mut xs = x.clone();
            let mut xst = xt.clone();
            complex::scal(n, a, &mut xs, ux, r);
            F::o_cscal(n, at, &mut xst, ux, m);
            check_eq!(
                xs,
                c_from_t::<F>(&xst),
                "cscal {} a={a:?} inc={ux} x={x:?}",
                ctx()
            );
            let (mut xs, mut ys) = (x.clone(), y0.clone());
            let (mut xst, mut yst) = (xt.clone(), y0t.clone());
            complex::swap(n, &mut xs, ux, &mut ys, uy);
            F::o_cswap(n, &mut xst, ux, &mut yst, uy, m);
            check_eq!(
                (xs, ys),
                (c_from_t::<F>(&xst), c_from_t::<F>(&yst)),
                "cswap {} incx={ux} incy={uy}",
                ctx()
            );
        }
    }
}

// ------------------------------------------------------- levels 2 and 3 ----

fn layout_of(b: u8) -> Layout {
    if b == b'C' {
        Layout::ColMajor
    } else {
        Layout::RowMajor
    }
}
fn trans_of(b: u8) -> Trans {
    match b {
        b'N' => Trans::NoTrans,
        b'T' => Trans::Trans,
        _ => Trans::ConjTrans,
    }
}

fn level23<F: Orc>() {
    let mut rng = Rng(0x5555_aaaa_3333_cccc ^ F::WIDTH as u64);
    for trial in 0..trials() / 3 {
        let (mm, n, p) = (rng.below(5), rng.below(5), rng.below(5));
        for (r, m) in MODES {
            let ctx = || {
                format!(
                    "{} trial {trial} m={mm} n={n} p={p} mode {:?}",
                    name::<F>(),
                    r
                )
            };
            // gemv (real: N/T only; SoftBLAS rejects 'C' for real)
            let lay = if rng.below(2) == 0 { b'C' } else { b'R' };
            let tr = if rng.below(2) == 0 { b'N' } else { b'T' };
            let lda = mm.max(n).max(1) + rng.below(2);
            let (incx, incy) = (1 + rng.below(2) as i64, 1 + rng.below(2));
            let asz = (lda * (mm + n + 1) + 1) as usize;
            let (rows, inner) = if tr == b'N' { (mm, n) } else { (n, mm) };
            let a = vec_of::<F>(&mut rng, asz);
            let x = vec_of::<F>(&mut rng, len_for(inner, incx));
            let y0 = vec_of::<F>(&mut rng, len_for(rows, incy as i64));
            let (alpha, beta) = (gen::<F>(&mut rng), gen::<F>(&mut rng));
            let mut y = y0.clone();
            let mut yt = to_t::<F>(&y0);
            sdblas::gemv(
                layout_of(lay),
                trans_of(tr),
                mm,
                n,
                alpha,
                &a,
                lda,
                &x,
                incx,
                beta,
                &mut y,
                incy,
                r,
            );
            F::o_gemv(
                lay,
                tr,
                mm,
                n,
                alpha.to_t(),
                &to_t::<F>(&a),
                lda,
                &to_t::<F>(&x),
                incx,
                beta.to_t(),
                &mut yt,
                incy,
                m,
            );
            check_eq!(y, from_t::<F>(&yt), "gemv {} layout={} trans={} lda={lda} incx={incx} incy={incy} alpha={alpha:?} beta={beta:?} a={a:?} x={x:?} y0={y0:?}", ctx(), lay as char, tr as char);
            // gemm
            let (ta, tb) = (
                if rng.below(2) == 0 { b'N' } else { b'T' },
                if rng.below(2) == 0 { b'N' } else { b'T' },
            );
            let lda = mm.max(n).max(1) + rng.below(2);
            let ldb = n.max(p).max(1) + rng.below(2);
            let ldc = p.max(1) + rng.below(2);
            let a = vec_of::<F>(&mut rng, (lda * (mm + n + 1) + 1) as usize);
            let b = vec_of::<F>(&mut rng, (ldb * (n + p + 1) + 1) as usize);
            let c0 = vec_of::<F>(&mut rng, (ldc * (mm + 1) + 1) as usize);
            let mut c = c0.clone();
            let mut ct = to_t::<F>(&c0);
            sdblas::gemm(
                trans_of(ta),
                trans_of(tb),
                mm,
                n,
                p,
                alpha,
                &a,
                lda,
                &b,
                ldb,
                beta,
                &mut c,
                ldc,
                r,
            );
            F::o_gemm(
                ta,
                tb,
                mm,
                n,
                p,
                alpha.to_t(),
                &to_t::<F>(&a),
                lda,
                &to_t::<F>(&b),
                ldb,
                beta.to_t(),
                &mut ct,
                ldc,
                m,
            );
            check_eq!(
                c,
                from_t::<F>(&ct),
                "gemm {} ta={} tb={} lda={lda} ldb={ldb} ldc={ldc} alpha={alpha:?} beta={beta:?}",
                ctx(),
                ta as char,
                tb as char
            );
            // complex gemv / gemm, with 'C'
            let trc = [b'N', b'T', b'C'][rng.below(3) as usize];
            let (rows, inner) = if trc == b'N' { (mm, n) } else { (n, mm) };
            let lda = mm.max(n).max(1) + rng.below(2);
            let a = cvec::<F>(&mut rng, (lda * (mm + n + 1) + 1) as usize);
            let x = cvec::<F>(&mut rng, len_for(inner, incx));
            let y0 = cvec::<F>(&mut rng, len_for(rows, incy as i64));
            let (ca, cb) = (
                Complex::new(gen::<F>(&mut rng), gen::<F>(&mut rng)),
                Complex::new(gen::<F>(&mut rng), gen::<F>(&mut rng)),
            );
            let (cat, cbt) = (C(ca.re.to_t(), ca.im.to_t()), C(cb.re.to_t(), cb.im.to_t()));
            let mut y = y0.clone();
            let mut yt = c_to_t::<F>(&y0);
            complex::gemv(
                layout_of(lay),
                trans_of(trc),
                mm,
                n,
                ca,
                &a,
                lda,
                &x,
                incx,
                cb,
                &mut y,
                incy,
                r,
            );
            F::o_cgemv(
                lay,
                trc,
                mm,
                n,
                cat,
                &c_to_t::<F>(&a),
                lda,
                &c_to_t::<F>(&x),
                incx,
                cbt,
                &mut yt,
                incy,
                m,
            );
            check_eq!(
                y,
                c_from_t::<F>(&yt),
                "cgemv {} layout={} trans={}",
                ctx(),
                lay as char,
                trc as char
            );
            let (ta, tb) = (
                [b'N', b'T', b'C'][rng.below(3) as usize],
                [b'N', b'T', b'C'][rng.below(3) as usize],
            );
            let ldb = n.max(p).max(1) + rng.below(2);
            let ldc = p.max(1) + rng.below(2);
            let b = cvec::<F>(&mut rng, (ldb * (n + p + 1) + 1) as usize);
            let c0 = cvec::<F>(&mut rng, (ldc * (mm + 1) + 1) as usize);
            let mut c = c0.clone();
            let mut ct = c_to_t::<F>(&c0);
            complex::gemm(
                trans_of(ta),
                trans_of(tb),
                mm,
                n,
                p,
                ca,
                &a,
                lda,
                &b,
                ldb,
                cb,
                &mut c,
                ldc,
                r,
            );
            F::o_cgemm(
                ta,
                tb,
                mm,
                n,
                p,
                cat,
                &c_to_t::<F>(&a),
                lda,
                &c_to_t::<F>(&b),
                ldb,
                cbt,
                &mut ct,
                ldc,
                m,
            );
            check_eq!(
                c,
                c_from_t::<F>(&ct),
                "cgemm {} ta={} tb={}",
                ctx(),
                ta as char,
                tb as char
            );
        }
    }
}

#[test]
fn sdsdot_and_hsdot() {
    let mut rng = Rng(0x7777_1111_2222_3333);
    for _ in 0..trials() {
        let n = rng.below(7);
        let (incx, incy) = (signed_inc(&mut rng), signed_inc(&mut rng));
        for (r, m) in MODES {
            let alpha = gen::<F32>(&mut rng);
            let x = vec_of::<F32>(&mut rng, len_for(n, incx));
            let y = vec_of::<F32>(&mut rng, len_for(n, incy));
            let ours = sdblas::sdsdot(n, alpha, &x, incx, &y, incy, r);
            let theirs = sdblas_oracle::sdsdot(
                n,
                alpha.0,
                &to_t::<F32>(&x),
                incx,
                &to_t::<F32>(&y),
                incy,
                m,
            );
            assert_eq!(
                ours.0, theirs,
                "sdsdot n={n} {r:?} alpha={alpha:?} x={x:?} y={y:?}"
            );
            let alpha = gen::<F16>(&mut rng);
            let x = vec_of::<F16>(&mut rng, len_for(n, incx));
            let y = vec_of::<F16>(&mut rng, len_for(n, incy));
            let ours = sdblas::hsdot(n, alpha, &x, incx, &y, incy, r);
            let theirs = sdblas_oracle::hsdot(
                n,
                alpha.0,
                &to_t::<F16>(&x),
                incx,
                &to_t::<F16>(&y),
                incy,
                m,
            );
            assert_eq!(
                ours.0, theirs,
                "hsdot n={n} {r:?} alpha={alpha:?} x={x:?} y={y:?}"
            );
        }
    }
}

macro_rules! suite {
    ($F:ident, $m:ident) => {
        mod $m {
            use super::*;
            #[test]
            fn level1() {
                super::level1::<$F>()
            }
            #[test]
            fn level1_complex() {
                super::level1_complex::<$F>()
            }
            #[test]
            fn level23() {
                super::level23::<$F>()
            }
        }
    };
}
suite!(F16, f16);
suite!(F32, f32);
suite!(F64, f64);
suite!(F128, f128);
