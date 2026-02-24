from strategies.s01_momentum_dip import MomentumDip
from strategies.s02_cross_asset_momentum import CrossAssetMomentum
from strategies.s03_factor_alpha import FactorAlpha
from strategies.s04_earnings_drift import EarningsDrift
from strategies.s05_short_term_reversal import ShortTermReversal
from strategies.s06_vix_term_structure import VIXTermStructure
from strategies.s07_macro_regime import MacroRegime
from strategies.s08_covered_calls import CoveredCalls
from strategies.s09_dollar_carry import DollarCarry
from strategies.s10_vol_surface import VolSurface

ALL_STRATEGIES = {
    "s01": MomentumDip,
    "s02": CrossAssetMomentum,
    "s03": FactorAlpha,
    "s04": EarningsDrift,
    "s05": ShortTermReversal,
    "s06": VIXTermStructure,
    "s07": MacroRegime,
    "s09": DollarCarry,
    "s10": VolSurface,
}
# S08 is an overlay — instantiated separately
