//! BLAS Levels 1 to 3 over [`sdfloat`], bit-identical to
//! [SoftBLAS](https://github.com/urbit/SoftBLAS).
//!
//! Every routine is generic over the four IEEE widths through
//! [`BlasFloat`] and takes its rounding mode as the last argument, exactly
//! where SoftBLAS takes `rndMode`. Operation order, accumulation order,
//! argument validation, and NaN canonicalization follow the SoftBLAS C
//! sources routine by routine, so a Rust jet built on this crate produces
//! the bits the vere C jets produce. Real routines live at the crate root;
//! complex ones in [`complex`].
//!
//! Conventions inherited from SoftBLAS that differ from reference BLAS:
//!
//! * Strides are signed (`i64`) only where SoftBLAS makes them signed
//!   (`axpy`, `copy`, `dot`, `swap`, and the complex `axpy`/`copy`/`dot*`/
//!   `rot`); elsewhere they are `u64`, and a routine given a stride it
//!   rejects returns zero or does nothing rather than panicking.
//! * `iamax` returns a 0-based index and 0 for an empty or invalid vector.
//! * Comparisons are SoftBLAS's macros: `gt(a, b)` is `!le(a, b)`, so a
//!   NaN compares greater than everything. This matters for `iamax`,
//!   `rotg`, and `nrm2_b`.
//! * Every stored or returned value is NaN-canonicalized; a complex value
//!   with a NaN in either part becomes NaN in both. SoftBLAS skips this in
//!   `rotg` and `rotmg` only, where it leaks SoftFloat's propagated NaN
//!   payloads; this crate canonicalizes there too.
//! * `gemv`/`gemm` compute `alpha * dot + beta * y` with no shortcut for
//!   `beta == 0`, so a NaN or infinity already in the output propagates.
//! * `Round::Away` is Hoon's directed away-from-zero. SoftBLAS's `'a'`
//!   is SoftFloat's nearest-ties-away instead; the four other modes match.
//!
//! Matrices are dense slices with an explicit leading dimension. `gemm` is
//! row-major with `A` being `M x N`, `B` being `N x P`, `C` being `M x P`;
//! `gemv` takes a [`Layout`].

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::all)]
// BLAS signatures are long by definition.
#![allow(clippy::too_many_arguments)]

pub use sdfloat::{Round, SoftFloat, F128, F16, F32, F64};

pub mod complex;
mod consts;
mod level1;
mod level2;
mod level3;

pub use consts::BlasFloat;
pub use level1::{
    asum, axpy, copy, dot, dot_wide, hsdot, iamax, nrm2, nrm2_b, rot, rotg, rotm, rotmg, scal,
    sdsdot, swap,
};
pub use level2::gemv;
pub use level3::gemm;

/// Storage order of a matrix argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// Element `(i, j)` at `i + j * ld`.
    ColMajor,
    /// Element `(i, j)` at `i * ld + j`.
    RowMajor,
}

/// The operation applied to a matrix argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trans {
    /// `op(A) = A`.
    NoTrans,
    /// `op(A) = A^T`.
    Trans,
    /// `op(A) = A^H`. Real routines treat it as [`Trans::Trans`]; SoftBLAS
    /// rejects `'C'` for real matrices, an error path no jet reaches.
    ConjTrans,
}

impl Trans {
    #[inline]
    fn is_trans(self) -> bool {
        !matches!(self, Trans::NoTrans)
    }
    #[inline]
    fn is_conj(self) -> bool {
        matches!(self, Trans::ConjTrans)
    }
}

/// SoftBLAS `SB_STRIDE_OVERFLOWS`: would `(n - 1) * inc` overflow `u64`?
#[inline]
pub(crate) fn stride_overflows(n: u64, inc: u64) -> bool {
    n > 1 && inc > u64::MAX / (n - 1)
}

/// SoftBLAS start offset for a possibly negative stride: `(1 - n) * inc`
/// when `inc < 0`, else 0. Computed in two's complement exactly as the C
/// `(-N + 1) * incX` does.
#[inline]
pub(crate) fn start(n: u64, inc: i64) -> i64 {
    if inc < 0 {
        (1i64.wrapping_sub(n as i64)).wrapping_mul(inc)
    } else {
        0
    }
}

/// Index a slice with a signed offset that must be non-negative.
#[inline]
pub(crate) fn at(i: i64) -> usize {
    debug_assert!(i >= 0, "negative vector index {i}");
    i as usize
}

/// SoftBLAS comparison macros. `gt` is `!le` and `ge` is `!lt`, so NaN
/// compares greater; `ne` is `!eq`, so NaN is unequal to itself.
#[inline]
pub(crate) fn gt<F: SoftFloat>(a: F, b: F) -> bool {
    !a.le(b)
}
#[inline]
pub(crate) fn ge<F: SoftFloat>(a: F, b: F) -> bool {
    !a.lt(b)
}
#[inline]
pub(crate) fn ne<F: SoftFloat>(a: F, b: F) -> bool {
    !a.eq(b)
}

/// `nan_unify_*`: any NaN becomes the canonical NaN of its width.
#[inline]
pub(crate) fn canon<F: SoftFloat>(a: F) -> F {
    if a.is_nan() {
        F::NAN
    } else {
        a
    }
}
