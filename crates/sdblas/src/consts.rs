//! The per-width constants SoftBLAS hard-codes: the safe-scaling thresholds
//! of `nrm2_B` and the rescaling gammas of `rotmg`.

use sdfloat::{Format, SoftFloat, F128, F16, F32, F64};

/// A float width usable by every routine in this crate.
///
/// The constants are copied from the SoftBLAS sources; `hrotmg` uses a
/// reduced gamma of 16 because binary16 cannot hold 2^24.
pub trait BlasFloat: SoftFloat + Format {
    /// `-1`.
    const NEG_ONE: Self;
    /// `-2`, the `rotm`/`rotmg` "identity" flag.
    const NEG_TWO: Self;
    /// `nrm2_b` small threshold.
    const TSML: Self;
    /// `nrm2_b` big threshold.
    const TBIG: Self;
    /// `nrm2_b` small scale.
    const SSML: Self;
    /// `nrm2_b` big scale.
    const SBIG: Self;
    /// `rotmg` gamma.
    const GAM: Self;
    /// `rotmg` gamma squared.
    const GAMSQ: Self;
    /// `rotmg` 1 / gamma.
    const RGAM: Self;
    /// `rotmg` 1 / gamma squared.
    const RGAMSQ: Self;
}

impl BlasFloat for F16 {
    const NEG_ONE: Self = F16(0xbc00);
    const NEG_TWO: Self = F16(0xc000);
    const TSML: Self = F16(0x2000); // 2^-7
    const TBIG: Self = F16(0x4800); // 2^3
    const SSML: Self = F16(0x6c00); // 2^12
    const SBIG: Self = F16(0x0800); // 2^-13
    const GAM: Self = F16(0x4c00); // 16
    const GAMSQ: Self = F16(0x5c00); // 256
    const RGAM: Self = F16(0x2c00); // 1/16
    const RGAMSQ: Self = F16(0x1c00); // 1/256
}

impl BlasFloat for F32 {
    const NEG_ONE: Self = F32(0xbf80_0000);
    const NEG_TWO: Self = F32(0xc000_0000);
    const TSML: Self = F32(0x2000_0000); // 2^-63
    const TBIG: Self = F32(0x5980_0000); // 2^52
    const SSML: Self = F32(0x6500_0000); // 2^75
    const SBIG: Self = F32(0x1980_0000); // 2^-76
    const GAM: Self = F32(0x4580_0000); // 4096
    const GAMSQ: Self = F32(0x4b80_0000); // 2^24
    const RGAM: Self = F32(0x3980_0000); // 2^-12
    const RGAMSQ: Self = F32(0x3380_0000); // 2^-24
}

impl BlasFloat for F64 {
    const NEG_ONE: Self = F64(0xbff0_0000_0000_0000);
    const NEG_TWO: Self = F64(0xc000_0000_0000_0000);
    const TSML: Self = F64(0x2000_0000_0000_0000); // 2^-511
    const TBIG: Self = F64(0x5e50_0000_0000_0000); // 2^486
    const SSML: Self = F64(0x6180_0000_0000_0000); // 2^537
    const SBIG: Self = F64(0x1e50_0000_0000_0000); // 2^-538
    const GAM: Self = F64(0x40b0_0000_0000_0000); // 4096
    const GAMSQ: Self = F64(0x4170_0000_0000_0000); // 2^24
    const RGAM: Self = F64(0x3f30_0000_0000_0000); // 2^-12
    const RGAMSQ: Self = F64(0x3e70_0000_0000_0000); // 2^-24
}

const fn q(hi: u64) -> F128 {
    F128((hi as u128) << 64)
}

impl BlasFloat for F128 {
    const NEG_ONE: Self = q(0xbfff_0000_0000_0000);
    const NEG_TWO: Self = q(0xc000_0000_0000_0000);
    const TSML: Self = q(0x2000_0000_0000_0000); // 2^-8191
    const TBIG: Self = q(0x5fc7_0000_0000_0000); // 2^8136
    const SSML: Self = q(0x6036_0000_0000_0000); // 2^8247
    const SBIG: Self = q(0x1fc7_0000_0000_0000); // 2^-8248
    const GAM: Self = q(0x400b_0000_0000_0000); // 4096
    const GAMSQ: Self = q(0x4017_0000_0000_0000); // 2^24
    const RGAM: Self = q(0x3ff3_0000_0000_0000); // 2^-12
    const RGAMSQ: Self = q(0x3fe7_0000_0000_0000); // 2^-24
}
