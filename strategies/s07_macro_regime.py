"""
S07 — Macro Regime Switcher

Alpha thesis:
  Market regimes are persistent and predictable from macro signals.
  The combination of equity trend, credit spreads, yield curve shape,
  and VIX level gives a robust regime classifier. Historically:
  - Risk-On periods: equities outperform bonds by ~15%/yr
  - Risk-Off periods: bonds + gold outperform by ~10%/yr
  Low turnover (regime changes are infrequent) = minimal transaction costs.

Inputs (checked weekly):
  VIX level          | 2Y-10Y yield spread
  SPY vs SMA(200)    | HYG/LQD credit ratio (proxy for credit risk appetite)

Regime rules:
  Risk-On:  VIX < 18 AND curve > 0 AND SPY > SMA(200)              -> 100% QQQ
  Neutral:  mixed signals (default)                                  -> 60% SPY + 40% TLT
  Risk-Off: VIX > 28 OR (curve < -0.5 AND SPY < SMA(200))          -> 60% SHY + 40% GLD

Novel enhancement: use HYG/LQD ratio as a "credit risk appetite" tiebreaker.
When HYG/LQD rises (credit risk-on), boost equity allocation +10%.
When HYG/LQD falls sharply (-3% 10d), reduce equity allocation -15%.

Regime rebalances only on regime CHANGE (not calendar), minimizing turnover.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import sma

logger = logging.getLogger(__name__)

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"

REGIME_ALLOC = {
    RISK_ON:   {"QQQ": 1.00},
    NEUTRAL:   {"SPY": 0.60, "TLT": 0.40},
    RISK_OFF:  {"SHY": 0.60, "GLD": 0.40},
}


class MacroRegime(Strategy):
    name = "s07_macro_regime"
    rebalance_freq = "weekly"
    max_positions = 2

    VIX_RISK_ON = 18.0
    VIX_RISK_OFF = 28.0
    CURVE_RISK_OFF = -0.50
    HYG_LQD_LOOKBACK = 10   # days for credit ratio momentum

    def get_universe(self) -> list[str]:
        return ["QQQ", "SPY", "TLT", "SHY", "GLD", "HYG", "LQD"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        kwargs expected:
          vix: pd.Series  (VIX levels)
          yield_spread: pd.Series  (2Y-10Y in pct, e.g. 0.5 = 50bps positive)
        """
        if prices.empty or len(prices) < 210:
            return pd.DataFrame()

        vix = kwargs.get("vix", pd.Series(dtype=float))
        yield_spread = kwargs.get("yield_spread", pd.Series(dtype=float))

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        spy_sma200 = sma(prices["SPY"], 200) if "SPY" in prices.columns else pd.Series(dtype=float)

        # Credit risk appetite proxy: HYG/LQD ratio momentum
        hyg_lqd_ratio = pd.Series(dtype=float)
        if "HYG" in prices.columns and "LQD" in prices.columns:
            hyg_lqd_ratio = prices["HYG"] / prices["LQD"]

        current_regime = NEUTRAL
        weekly_idx = prices.resample("W").last().index

        for date in weekly_idx:
            if date not in prices.index:
                idx = prices.index.searchsorted(date) - 1
                if idx < 0:
                    continue
                date = prices.index[idx]

            loc = prices.index.get_loc(date)
            if loc < 200:
                continue

            # --- Classify regime ---
            vix_now = vix.get(date, np.nan) if not vix.empty else np.nan
            curve_now = yield_spread.get(date, np.nan) if not yield_spread.empty else np.nan
            spy_now = prices["SPY"].get(date, np.nan) if "SPY" in prices.columns else np.nan
            sma200_now = spy_sma200.get(date, np.nan) if not spy_sma200.empty else np.nan

            # Credit tiebreaker
            credit_risk_on = False
            credit_risk_off = False
            if not hyg_lqd_ratio.empty and date in hyg_lqd_ratio.index:
                ratio_now = hyg_lqd_ratio.get(date)
                idx_loc = hyg_lqd_ratio.index.get_loc(date)
                if idx_loc >= self.HYG_LQD_LOOKBACK:
                    ratio_10d_ago = hyg_lqd_ratio.iloc[idx_loc - self.HYG_LQD_LOOKBACK]
                    credit_chg = (ratio_now - ratio_10d_ago) / ratio_10d_ago
                    credit_risk_on = credit_chg > 0.01   # +1% = credit improving
                    credit_risk_off = credit_chg < -0.03  # -3% = credit deteriorating

            new_regime = _classify(vix_now, curve_now, spy_now, sma200_now,
                                   self.VIX_RISK_ON, self.VIX_RISK_OFF, self.CURVE_RISK_OFF)

            # Override with credit signals
            if credit_risk_off and new_regime == RISK_ON:
                new_regime = NEUTRAL
            elif credit_risk_on and new_regime == NEUTRAL:
                new_regime = RISK_ON

            alloc = REGIME_ALLOC[new_regime].copy()

            # Zero entire window first so stale regime weights don't bleed through
            next_loc = min(loc + 5, len(prices))
            signals.iloc[loc:next_loc, :] = 0.0
            for t, w in alloc.items():
                if t in signals.columns:
                    col_idx = signals.columns.get_loc(t)
                    signals.iloc[loc:next_loc, col_idx] = w

            if new_regime != current_regime:
                logger.info("[MacroRegime] %s -> %s on %s", current_regime, new_regime, date.date())
                current_regime = new_regime

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        return {t: float(w) for t, w in signals[signals > 0].items()}

    def current_regime(self, vix: float, curve: float, spy: float, sma200: float) -> str:
        return _classify(vix, curve, spy, sma200,
                         self.VIX_RISK_ON, self.VIX_RISK_OFF, self.CURVE_RISK_OFF)


def _classify(
    vix: float, curve: float, spy: float, sma200: float,
    vix_on: float, vix_off: float, curve_off: float,
) -> str:
    has_vix = not np.isnan(vix) if vix is not None else False
    has_curve = not np.isnan(curve) if curve is not None else False
    has_spy = not (np.isnan(spy) or np.isnan(sma200)) if (spy is not None and sma200 is not None) else False

    if has_vix and has_curve and has_spy:
        if vix < vix_on and curve > 0 and spy > sma200:
            return RISK_ON
        if vix > vix_off or (curve < curve_off and spy < sma200):
            return RISK_OFF
    return NEUTRAL
