//! Hand-checked cases for the conventions in the spec (section 4) and the
//! Hoon probes recorded during orientation. These do not need the oracle.

use sdfloat::{Round, SoftFloat, F128, F16, F32, F64};

const MODES: [Round; 5] = [
    Round::NearEven,
    Round::Up,
    Round::Down,
    Round::Zero,
    Round::Away,
];

#[test]
fn canonical_nan_per_width() {
    assert_eq!(F16::NAN.0, 0x7e00);
    assert_eq!(F32::NAN.0, 0x7fc0_0000);
    assert_eq!(F64::NAN.0, 0x7ff8_0000_0000_0000);
    assert_eq!(F128::NAN.0, 0x7fff_8000_0000_0000_0000_0000_0000_0000);
}

#[test]
fn nan_inputs_canonicalize_and_never_propagate_payload() {
    let junk = F32(0xffff_1234);
    for r in MODES {
        assert_eq!(junk.add(F32(0x3f80_0000), r), F32::NAN);
        assert_eq!(F32(0x3f80_0000).mul(junk, r), F32::NAN);
        assert_eq!(junk.sqrt(r), F32::NAN);
        assert_eq!(junk.round_to_int(r), F32::NAN);
    }
    assert!(junk.is_nan());
    assert!(!junk.eq(junk));
}

#[test]
fn signed_zero_rules() {
    let one = F32(0x3f80_0000);
    let pz = F32(0);
    let nz = F32(0x8000_0000);
    for r in MODES {
        let want = if r == Round::Down { nz } else { pz };
        assert_eq!(one.sub(one, r), want, "{r:?}");
        assert_eq!(one.neg().add(one, r), want, "{r:?}");
        assert_eq!(pz.add(nz, r), want, "{r:?}");
        assert_eq!(nz.add(nz, r), nz, "{r:?}");
        assert_eq!(pz.add(pz, r), pz, "{r:?}");
        assert_eq!(one.neg().mul(pz, r), nz, "{r:?}");
        assert_eq!(nz.sqrt(r), nz, "{r:?}");
        assert_eq!(nz.round_to_int(r), nz, "{r:?}");
    }
}

#[test]
fn overflow_is_infinity_in_every_mode() {
    // Hoon ++fl, not IEEE 754 (which gives MAX under Zero, Down for a
    // positive result, Up for a negative one): urbit/urbit#7426.
    let max = F32::MAX;
    let two = F32(0x4000_0000);
    for r in MODES {
        assert_eq!(max.mul(two, r), F32::INF, "{r:?}");
        assert_eq!(max.neg().mul(two, r), F32::INF.neg(), "{r:?}");
        assert_eq!(max.add(max, r), F32::INF, "{r:?}");
        assert_eq!(F16::from_u64(u64::MAX, r), F16::INF, "{r:?}");
        assert_eq!(F64::MAX.to_f32(r), F32::INF, "{r:?}");
    }
    // Below 2^(emax+1) nothing overflows: MAX plus half an ulp rounds to
    // MAX unless the mode rounds away.
    let half_ulp = F32(0x7300_0000); // 2^103 = ulp(MAX) / 2
    assert_eq!(max.add(half_ulp, Round::Zero), max);
    assert_eq!(max.add(half_ulp, Round::Down), max);
    assert_eq!(max.add(half_ulp, Round::Up), F32::INF);
    assert_eq!(max.neg().sub(half_ulp, Round::Up), max.neg());
}

#[test]
fn away_is_directed_not_nearest() {
    // From `urbit eval` on ++ff with %a: toi 2.25 -> 3; 1 + 1e-10 -> 1+ulp.
    let x = F32(0x4010_0000); // 2.25
    assert_eq!(x.round_to_int(Round::Away), F32(0x4040_0000)); // 3
    assert_eq!(x.round_to_int(Round::NearEven), F32(0x4000_0000)); // 2
    assert_eq!(x.to_i32(Round::Away), 3);
    let one = F32(0x3f80_0000);
    let tiny = F32(0x2edb_e6ff); // 1e-10
    assert_eq!(one.add(tiny, Round::Away), F32(0x3f80_0001));
    assert_eq!(one.neg().add(tiny.neg(), Round::Away), F32(0xbf80_0001));
    assert_eq!(one.add(tiny, Round::NearEven), one);
}

#[test]
fn round_to_int_ties_and_small_values() {
    let half = F32(0x3f00_0000);
    let one = F32(0x3f80_0000);
    let three_halves = F32(0x3fc0_0000);
    let two = F32(0x4000_0000);
    assert_eq!(half.round_to_int(Round::NearEven), F32(0));
    assert_eq!(half.neg().round_to_int(Round::NearEven), F32(0x8000_0000));
    assert_eq!(three_halves.round_to_int(Round::NearEven), two);
    assert_eq!(half.round_to_int(Round::Away), one);
    assert_eq!(half.round_to_int(Round::Up), one);
    assert_eq!(half.neg().round_to_int(Round::Up), F32(0x8000_0000));
    assert_eq!(half.neg().round_to_int(Round::Down), one.neg());
    assert_eq!(F32(0x3e80_0000).round_to_int(Round::Zero), F32(0)); // 0.25 -> +0
    assert_eq!(F32::INF.round_to_int(Round::Zero), F32::INF);
}

#[test]
fn to_int_saturation_matches_arm_vfpv2_softfloat() {
    assert_eq!(F32::NAN.to_i32(Round::NearEven), 0);
    assert_eq!(F32::NAN.to_i64(Round::Zero), 0);
    assert_eq!(F32::INF.to_i32(Round::NearEven), i32::MAX);
    assert_eq!(F32::INF.neg().to_i32(Round::NearEven), i32::MIN);
    assert_eq!(F32(0x4f00_0000).to_i32(Round::Zero), i32::MAX); // 2^31
    assert_eq!(F32(0xcf00_0000).to_i32(Round::Zero), i32::MIN); // -2^31 exact
    assert_eq!(F64(0x43e0_0000_0000_0000).to_i64(Round::Zero), i64::MAX); // 2^63
    assert_eq!(F64(0xc3e0_0000_0000_0000).to_i64(Round::Zero), i64::MIN);
    assert_eq!(F32(0xbf00_0000).to_i32(Round::Away), -1); // -0.5 away
    assert_eq!(F32(0xbf00_0000).to_i32(Round::NearEven), 0);
}

#[test]
fn compare_basics() {
    let pz = F64(0);
    let nz = F64(0x8000_0000_0000_0000);
    let one = F64(0x3ff0_0000_0000_0000);
    assert!(pz.eq(nz));
    assert!(!pz.lt(nz));
    assert!(pz.le(nz));
    assert!(one.neg().lt(nz));
    assert!(!F64::NAN.le(F64::NAN));
    assert!(!F64::NAN.lt(one));
    assert!(!one.lt(F64::NAN));
    assert!(F64::INF.neg().lt(F64::MAX.neg()));
}

#[test]
fn f128_word_pair_round_trip() {
    let x = F128(0x4000_1234_5678_9abc_def0_1122_3344_5566);
    assert_eq!(
        x.to_u64_pair(),
        [0xdef0_1122_3344_5566, 0x4000_1234_5678_9abc]
    );
    assert_eq!(F128::from_u64_pair(x.to_u64_pair()), x);
}

#[test]
fn exact_widening_and_narrowing_examples() {
    assert_eq!(F16(0x3c00).to_f32(), F32(0x3f80_0000));
    assert_eq!(F16(0x0001).to_f32(), F32(0x3380_0000)); // min subnormal 2^-24
    assert_eq!(F32(0x3f80_0000).to_f64(), F64(0x3ff0_0000_0000_0000));
    assert_eq!(
        F64(0x3ff0_0000_0000_0000).to_f128(),
        F128(0x3fff_0000_0000_0000_0000_0000_0000_0000)
    );
    assert_eq!(F32(0x477f_e000).to_f16(Round::NearEven), F16(0x7bff)); // 65504
    assert_eq!(F32(0x4780_0000).to_f16(Round::NearEven), F16(0x7c00)); // 65536 -> inf
    assert_eq!(F32(0x4780_0000).to_f16(Round::Zero), F16(0x7c00)); // Hoon: inf, IEEE: MAX
}
