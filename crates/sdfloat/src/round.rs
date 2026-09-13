/// IEEE 754 rounding-direction attribute, one variant per Hoon `++fl` mode.
///
/// Every operation that can be inexact takes a `Round` explicitly; there is
/// no global mode. The five variants are Hoon's `%n %u %d %z %a`. Note that
/// Hoon's `%a` (and therefore `Away`) is *directed* rounding away from zero,
/// the magnitude analogue of `Up`; it is not SoftFloat's `near_maxMag`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Round {
    /// `%n`: round to nearest, ties to even. IEEE `roundTiesToEven`.
    NearEven,
    /// `%u`: round toward +infinity. IEEE `roundTowardPositive`.
    Up,
    /// `%d`: round toward -infinity. IEEE `roundTowardNegative`.
    Down,
    /// `%z`: round toward zero. IEEE `roundTowardZero`.
    Zero,
    /// `%a`: round away from zero (directed: the magnitude ceiling).
    Away,
}

impl Round {
    /// Whether an inexact result of the given sign rounds to the larger
    /// magnitude under this mode. Nearest-even is excluded; it is decided
    /// by the round bits, not by direction alone.
    #[inline]
    pub(crate) fn rounds_away(self, negative: bool) -> bool {
        match self {
            Round::NearEven | Round::Zero => false,
            Round::Away => true,
            Round::Up => !negative,
            Round::Down => negative,
        }
    }
}
