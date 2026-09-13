//! Level 3: `gemm`. Model: SoftBLAS `sgemm.c`.

use crate::{canon, BlasFloat, Round, Trans};

/// `sgemm`: `C <- alpha * op(A) * op(B) + beta * C`.
///
/// Row-major, as SoftBLAS: `op(A)` is `m x n`, `op(B)` is `n x p`, `C` is
/// `m x p` with `C[i][j]` at `j + i * ldc`. Untransposed `A[i][k]` is at
/// `k + i * lda`; transposed it is at `i + k * lda`; likewise `B`. Each
/// output element is `alpha * dot + beta * c_ij` with the inner product
/// accumulated over `k` left to right from `+0`.
pub fn gemm<F: BlasFloat>(
    transa: Trans,
    transb: Trans,
    m: u64,
    n: u64,
    p: u64,
    alpha: F,
    a: &[F],
    lda: u64,
    b: &[F],
    ldb: u64,
    beta: F,
    c: &mut [F],
    ldc: u64,
    r: Round,
) {
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
    let (ta, tb) = (transa.is_trans(), transb.is_trans());
    for i in 0..m as usize {
        for j in 0..p as usize {
            let mut dot = F::ZERO;
            for k in 0..n as usize {
                let ia = if ta { i + k * lda } else { k + i * lda };
                let ib = if tb { k + j * ldb } else { j + k * ldb };
                dot = dot.add(a[ia].mul(b[ib], r), r);
            }
            let cij = &mut c[j + i * ldc];
            *cij = canon(alpha.mul(dot, r).add(beta.mul(*cij, r), r));
        }
    }
}
