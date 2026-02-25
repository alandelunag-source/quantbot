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
from strategies.s11_congressional import CongressionalTrades
from strategies.s12_index_inclusion import IndexInclusion
from strategies.s13_pre_earnings_drift import PreEarningsDrift
from strategies.s14_gamma_wall import GammaWall
from strategies.s15_short_flow import ShortFlow
from strategies.s16_overnight_carry import OvernightCarry

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
    "s11": CongressionalTrades,
    "s12": IndexInclusion,
    "s13": PreEarningsDrift,
    "s14": GammaWall,
    "s15": ShortFlow,
    "s16": OvernightCarry,
}
# S08 is an overlay — instantiated separately
