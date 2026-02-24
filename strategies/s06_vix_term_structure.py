"""
S06 — VIX Term Structure Carry

Alpha thesis:
  VIX futures are almost always in contango (futures > spot). When VIX9D
  (9-day VIX) is significantly BELOW VIX (30-day), the term structure is
  steep — there's a "vol carry" opportunity: buy SPY (markets calm short-term
  despite elevated longer-term fear). When VIX9D > VIX, the term is in
  backwardation = near-term panic = defensive posture.

Signal logic:
  1. Compute roll = (VIX − VIX9D) / VIX9D  (vol term premium)
  2. Z-score the roll over 60 days
  3. When z > 1.0: steep contango → long SPY (vol carry is attractive)
  4. When z < -1.0: backwardation → long SHY (near-term fear elevated)
  5. Neutral zone: 60% SPY + 40% TLT

Academic: Simon & Campasano (2014) show VIX9D/VIX ratio predicts 1-week
equity returns. Whaley (2013) confirms vol term structure is a forward-looking
fear indicator.

Novel enhancement: combine with VIX absolute level.
  High VIX absolute (>28) + backwardation = max risk-off (60% SHY + 40% GLD)
  This creates a regime-conditional VRP harvest strategy.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import z_score

logger = logging.getLogger(__name__)

REGIME_CONTANGO = "contango"
REGIME_NEUTRAL = "neutral"
REGIME_BACKWARDATION = "backwardation"

ALLOCATIONS = {
    REGIME_CONTANGO:       {"SPY": 1.0},
    REGIME_NEUTRAL:        {"SPY": 0.60, "TLT": 0.40},
    REGIME_BACKWARDATION:  {"SHY": 0.60, "GLD": 0.40},
}

EXTREME_RISK_OFF = {"SHY": 0.60, "GLD": 0.40}


class VIXTermStructure(Strategy):
    name = "s06_vix_term_structure"
    rebalance_freq = "daily"
    max_positions = 3

    Z_THRESH = 1.0     # z-score threshold for regime switch
    VIX_PANIC = 28.0   # Absolute VIX level for extreme risk-off

    def get_universe(self) -> list[str]:
        return ["SPY", "TLT", "SHY", "GLD"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        prices should contain SPY, TLT, SHY, GLD.
        VIX/VIX9D passed via kwargs['vix'] and kwargs['vix9d'].
        """
        vix = kwargs.get("vix")
        vix9d = kwargs.get("vix9d")

        if prices.empty:
            return pd.DataFrame()

        # If VIX data unavailable, fall back to flat neutral
        if vix is None or vix9d is None:
            logger.warning("VIX/VIX9D not provided — defaulting to neutral regime")
            signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for t, w in ALLOCATIONS[REGIME_NEUTRAL].items():
                if t in signals.columns:
                    signals[t] = w
            return signals

        # Roll = (VIX - VIX9D) / VIX9D
        roll = (vix - vix9d) / vix9d.replace(0, np.nan)
        roll_z = z_score(roll, window=60)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date in prices.index:
            if date not in roll_z.index:
                continue

            z = roll_z.get(date, np.nan)
            vix_level = vix.get(date, np.nan)

            if np.isnan(z):
                regime = REGIME_NEUTRAL
            elif not np.isnan(vix_level) and vix_level > self.VIX_PANIC and z < 0:
                # Panic + backwardation = maximum defensiveness
                alloc = EXTREME_RISK_OFF
                for t, w in alloc.items():
                    if t in signals.columns:
                        signals.loc[date, t] = w
                continue
            elif z > self.Z_THRESH:
                regime = REGIME_CONTANGO
            elif z < -self.Z_THRESH:
                regime = REGIME_BACKWARDATION
            else:
                regime = REGIME_NEUTRAL

            alloc = ALLOCATIONS[regime]
            for t, w in alloc.items():
                if t in signals.columns:
                    signals.loc[date, t] = w

        return signals

    def get_regime(self, vix: float, vix9d: float, roll_z: float) -> str:
        if np.isnan(roll_z):
            return REGIME_NEUTRAL
        if vix > self.VIX_PANIC and roll_z < 0:
            return "extreme_risk_off"
        if roll_z > self.Z_THRESH:
            return REGIME_CONTANGO
        if roll_z < -self.Z_THRESH:
            return REGIME_BACKWARDATION
        return REGIME_NEUTRAL

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        return {t: float(w) for t, w in signals[signals > 0].items()}
