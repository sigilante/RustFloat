//! Level 2: `gemv`. Model: SoftBLAS `sgemv.c`.

use crate::{canon, BlasFloat, Layout, Round, Trans};

/// `sgemv`: `y <- alpha * op(A) * x + beta * y`, `A` being `m x n`.
///
/// Each output element is `alpha * dot + beta * y_i` where `dot` is the row
/// (or column) inner product accumulated left to right from `+0`. Strides
/// below 1 make the call a no-op, as in SoftBLAS.
pub fn gemv<F: BlasFloat>(
    layout: Layout,
    trans: Trans,
    m: u64,
    n: u64,
    alpha: F,
    a: &[F],
    lda: u64,
    x: &[F],
    incx: i64,
    beta: F,
    y: &mut [F],
    incy: u64,
    r: Round,
) {
    if incx < 1 || incy < 1 {
        return;
    }
    let (incx, incy, lda) = (incx as usize, incy as usize, lda as usize);
    let colmajor = layout == Layout::ColMajor;
    // Untransposed: y has length m, inner length n. Transposed: swapped.
    let (rows, inner) = if trans.is_trans() { (n, m) } else { (m, n) };
    for i in 0..rows as usize {
        let mut dot = F::ZERO;
        for j in 0..inner as usize {
            let idx = match (trans.is_trans(), colmajor) {
                (false, true) => i + j * lda,
                (false, false) => i * lda + j,
                (true, true) => j + i * lda,
                (true, false) => j * lda + i,
            };
            dot = dot.add(a[idx].mul(x[j * incx], r), r);
        }
        let yi = &mut y[i * incy];
        *yi = canon(alpha.mul(dot, r).add(beta.mul(*yi, r), r));
    }
}
