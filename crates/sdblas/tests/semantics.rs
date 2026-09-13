//! Hand-checked behaviour: the SoftBLAS conventions this crate promises,
//! stated without the oracle so a reader can see them.

use sdblas::complex::{self, Complex};
use sdblas::{Layout, Round, SoftFloat, Trans, F16, F32, F64};

const ONE: F32 = F32(0x3f80_0000);
const TWO: F32 = F32(0x4000_0000);
const HALF_ULP_ABOVE_ONE: F32 = F32(0x3380_0000); // 2^-24

fn f(v: u32) -> F32 {
    F32(v)
}

#[test]
fn axpy_rounding_mode_reaches_every_op() {
    // 1 + 2^-24 rounds up only toward +inf; SoftBLAS test_saxpy_rounding.
    for (r, want) in [
        (Round::NearEven, ONE),
        (Round::Zero, ONE),
        (Round::Down, ONE),
        (Round::Up, F32(0x3f80_0001)),
        (Round::Away, F32(0x3f80_0001)),
    ] {
        let mut y = [ONE];
        sdblas::axpy(1, ONE, &[HALF_ULP_ABOVE_ONE], 1, &mut y, 1, r);
        assert_eq!(y[0], want, "{r:?}");
    }
}

#[test]
fn negative_strides_walk_backwards() {
    // y = [5,4,3,2,1] with incy = -1 pairs y[4] with x[0]: SoftBLAS
    // test_saxpy_neg_stride expects [15,12,9,6,3] for a = 2, x = [1..5].
    let x: Vec<F32> = (1..=5).map(|i| F32::from_i32(i, Round::NearEven)).collect();
    let mut y: Vec<F32> = (1..=5)
        .rev()
        .map(|i| F32::from_i32(i, Round::NearEven))
        .collect();
    sdblas::axpy(5, TWO, &x, 1, &mut y, -1, Round::NearEven);
    let want: Vec<F32> = [15, 12, 9, 6, 3]
        .iter()
        .map(|&i| F32::from_i32(i, Round::NearEven))
        .collect();
    assert_eq!(y, want);
}

#[test]
fn nan_canonicalization() {
    let junk = f(0xffff_1234);
    let mut y = [junk, ONE];
    sdblas::scal(2, ONE, &mut y, 1, Round::NearEven);
    assert_eq!(y, [F32::NAN, ONE]);
    let mut c = [junk];
    sdblas::copy(1, &[junk], 1, &mut c, 1);
    assert_eq!(c, [F32::NAN]);
    assert_eq!(
        sdblas::dot(2, &[ONE, junk], 1, &[ONE, ONE], 1, Round::NearEven),
        F32::NAN
    );
    // Complex: one NaN part poisons both.
    let mut z = [Complex::new(ONE, junk)];
    complex::scal(1, Complex::new(ONE, F32::ZERO), &mut z, 1, Round::NearEven);
    assert_eq!(z[0], Complex::new(F32::NAN, F32::NAN));
}

#[test]
fn nan_compares_greater_in_iamax() {
    // gt is !le, so a NaN element beats the running maximum, and the element
    // after it beats the NaN; from then on the comparison is ordinary.
    let x = [ONE, F32::NAN, TWO, ONE];
    assert_eq!(sdblas::iamax(4, &x, 1), 2);
    assert_eq!(sdblas::iamax(3, &x, 1), 2);
    assert_eq!(sdblas::iamax(2, &x, 1), 1);
    assert_eq!(sdblas::iamax(1, &x, 1), 0);
    assert_eq!(sdblas::iamax(4, &x, 0), 0);
}

#[test]
fn invalid_strides_are_rejected_not_panicked() {
    let x = [ONE, TWO];
    assert_eq!(sdblas::asum(2, &x, 0, Round::NearEven), F32::ZERO);
    assert_eq!(sdblas::nrm2(2, &x, 0, Round::NearEven), F32::ZERO);
    let huge = 1u64 << 36; // (huge - 1) * huge overflows u64
    assert_eq!(
        sdblas::asum(huge, &x[..1], huge, Round::NearEven),
        F32::ZERO
    );
    assert_eq!(
        sdblas::nrm2(huge, &x[..1], huge, Round::NearEven),
        F32::ZERO
    );
    let mut y = [ONE];
    sdblas::gemv(
        Layout::RowMajor,
        Trans::NoTrans,
        1,
        1,
        ONE,
        &x,
        1,
        &x,
        0,
        ONE,
        &mut y,
        1,
        Round::NearEven,
    );
    assert_eq!(y, [ONE], "incx = 0 makes gemv a no-op");
}

#[test]
fn gemm_row_major_conventions() {
    // A = [1 2; 3 4], B = [5 6; 7 8] row-major; C = A B = [19 22; 43 50].
    let i = |v: i32| F32::from_i32(v, Round::NearEven);
    let a = [i(1), i(2), i(3), i(4)];
    let b = [i(5), i(6), i(7), i(8)];
    let mut c = [F32::ZERO; 4];
    sdblas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        2,
        2,
        2,
        ONE,
        &a,
        2,
        &b,
        2,
        F32::ZERO,
        &mut c,
        2,
        Round::NearEven,
    );
    assert_eq!(c, [i(19), i(22), i(43), i(50)]);
    // Transposed storage of both operands gives the same product.
    let at = [i(1), i(3), i(2), i(4)];
    let bt = [i(5), i(7), i(6), i(8)];
    let mut c = [F32::ZERO; 4];
    sdblas::gemm(
        Trans::Trans,
        Trans::Trans,
        2,
        2,
        2,
        ONE,
        &at,
        2,
        &bt,
        2,
        F32::ZERO,
        &mut c,
        2,
        Round::NearEven,
    );
    assert_eq!(c, [i(19), i(22), i(43), i(50)]);
    // beta * C is always formed: NaN in C survives beta = 0.
    let mut c = [F32::NAN, F32::ZERO, F32::ZERO, F32::ZERO];
    sdblas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        2,
        2,
        2,
        ONE,
        &a,
        2,
        &b,
        2,
        F32::ZERO,
        &mut c,
        2,
        Round::NearEven,
    );
    assert_eq!(c[0], F32::NAN);
}

#[test]
fn gemv_layouts_agree() {
    let i = |v: i32| F64::from_i32(v, Round::NearEven);
    let one = i(1);
    // A = [1 2 3; 4 5 6] (2x3), x = [1 1 1]: A x = [6 15].
    let row = [i(1), i(2), i(3), i(4), i(5), i(6)];
    let col = [i(1), i(4), i(2), i(5), i(3), i(6)];
    let x = [one, one, one];
    let mut y = [F64::ZERO; 2];
    sdblas::gemv(
        Layout::RowMajor,
        Trans::NoTrans,
        2,
        3,
        one,
        &row,
        3,
        &x,
        1,
        F64::ZERO,
        &mut y,
        1,
        Round::NearEven,
    );
    assert_eq!(y, [i(6), i(15)]);
    let mut y = [F64::ZERO; 2];
    sdblas::gemv(
        Layout::ColMajor,
        Trans::NoTrans,
        2,
        3,
        one,
        &col,
        2,
        &x,
        1,
        F64::ZERO,
        &mut y,
        1,
        Round::NearEven,
    );
    assert_eq!(y, [i(6), i(15)]);
    // A^T x for x = [1 1]: [5 7 9].
    let mut y = [F64::ZERO; 3];
    sdblas::gemv(
        Layout::RowMajor,
        Trans::Trans,
        2,
        3,
        one,
        &row,
        3,
        &x[..2],
        1,
        F64::ZERO,
        &mut y,
        1,
        Round::NearEven,
    );
    assert_eq!(y, [i(5), i(7), i(9)]);
}

#[test]
fn complex_conjugate_transpose() {
    // x = [1+2i, 3+4i], y = [5+6i, 7+8i]: dotu = -18+68i, dotc = 70-8i.
    let c = |re: i32, im: i32| {
        Complex::new(
            F32::from_i32(re, Round::NearEven),
            F32::from_i32(im, Round::NearEven),
        )
    };
    let x = [c(1, 2), c(3, 4)];
    let y = [c(5, 6), c(7, 8)];
    assert_eq!(complex::dotu(2, &x, 1, &y, 1, Round::NearEven), c(-18, 68));
    assert_eq!(complex::dotc(2, &x, 1, &y, 1, Round::NearEven), c(70, -8));
    // gemm with A^H: A = [[1+2i]] (1x1), B = [[1]]: A^H B = [[1-2i]].
    let mut out = [Complex::ZERO];
    complex::gemm(
        Trans::ConjTrans,
        Trans::NoTrans,
        1,
        1,
        1,
        c(1, 0),
        &[c(1, 2)],
        1,
        &[c(1, 0)],
        1,
        Complex::ZERO,
        &mut out,
        1,
        Round::NearEven,
    );
    assert_eq!(out, [c(1, -2)]);
}

#[test]
fn rotmg_identity_flag_writes_only_p0() {
    // d2 * y1 == 0 -> P[0] = -2, nothing else touched.
    let (mut d1, mut d2, mut x1) = (TWO, F32::ZERO, ONE);
    let mut p = [ONE; 5];
    sdblas::rotmg(&mut d1, &mut d2, &mut x1, ONE, &mut p, Round::NearEven);
    assert_eq!((d1, d2, x1), (TWO, F32::ZERO, ONE));
    assert_eq!(p, [F32(0xc000_0000), ONE, ONE, ONE, ONE]);
}

#[test]
fn nrm2_variants_agree_on_easy_input() {
    // [3, 4] -> 5 exactly, both algorithms, all widths shown for f16.
    let x = [F16(0x4200), F16(0x4400)]; // 3, 4
    let five = F16(0x4500);
    assert_eq!(sdblas::nrm2(2, &x, 1, Round::NearEven), five);
    assert_eq!(sdblas::nrm2_b(2, &x, 1, Round::NearEven), five);
    assert_eq!(sdblas::nrm2(1, &x[1..], 1, Round::NearEven), x[1]);
    assert_eq!(sdblas::nrm2(0, &x, 1, Round::NearEven), F16::ZERO);
}

#[test]
fn sdsdot_accumulates_in_double() {
    // alpha = 1 plus four terms of 2^-25. Accumulated in f32 each term is
    // below half an ulp of 1 and is lost; in f64 they sum to 2^-23, and the
    // result narrows to the f32 just above 1.
    let tiny = F32(0x3300_0000); // 2^-25
    let x = [tiny; 4];
    let y = [ONE; 4];
    let f32_fold = x.iter().fold(ONE, |acc, &t| {
        acc.add(t.mul(ONE, Round::NearEven), Round::NearEven)
    });
    assert_eq!(f32_fold, ONE);
    assert_eq!(
        sdblas::sdsdot(4, ONE, &x, 1, &y, 1, Round::NearEven),
        F32(0x3f80_0001)
    );
}
