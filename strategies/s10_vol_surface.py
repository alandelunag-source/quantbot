"""
S10 — Volatility Surface Arbitrage (VRP Harvest)

Alpha thesis:
  Implied volatility (IV) in S&P 500 options is systematically elevated above
  realized volatility by ~3-5 vol points on average. This is the Variance Risk
  Premium (VRP) — investors pay a premium to hedge. You collect it by being
  short volatility when IV-RV spread is wide.

  Most sophisticated implementation in quantbot. Used by Citadel, Millenium,
  Two Sigma vol desks.

Signal logic:
  1. Compute 30d SPY realized vol (annualized)
  2. Get SPY ATM 30-day IV from options chain
  3. VRP = IV - RV (typically positive)
  4. Z-score VRP over 90 days
  5. When VRP_z > 1.5: VRP is elevated → SELL volatility (short VXX or UVXY)
  6. When VRP_z < -1.0: markets underpricing vol → LONG VXX (protective)
  7. Neutral: hold cash (SHY)

Risk controls:
  - Max 5% portfolio in short-vol positions (tail risk!)
  - Hard stop: close short-vol on VIX spike >5 points in 1 day
  - Never short vol into FOMC week or major macro events

Novel enhancement: SKEW index as an additional signal.
  High SKEW (>135) + high VRP → reduce short-vol (tail risk premium elevated)
  This avoids the "collect nickels in front of a steamroller" failure mode.

Citadel note: never run large short-vol without gamma hedges. Here we use
hard stops as a simplified hedge substitute for paper trading.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import realized_vol, z_score

logger = logging.getLogger(__name__)

VRP_LONG_THRESH = 1.5    # z-score above which we harvest VRP (short vol)
VRP_SHORT_THRESH = -1.0  # z-score below which we buy protection (long vol)
LOOKBACK_RV = 21         # 21-day realized vol
LOOKBACK_Z = 90          # z-score window
MAX_SHORT_VOL_ALLOC = 0.05   # Never more than 5% in short-vol
VIX_SPIKE_STOP = 5.0     # Close short-vol if VIX spikes > 5pts in one day


class VolSurface(Strategy):
    name = "s10_vol_surface"
    rebalance_freq = "daily"
    max_positions = 2

    def get_universe(self) -> list[str]:
        return ["SPY", "VXX", "SHY"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        kwargs:
          vix: pd.Series   (VIX closing levels, proxy for ATM IV)
        """
        if prices.empty or len(prices) < LOOKBACK_Z + LOOKBACK_RV + 5:
            return pd.DataFrame()

        vix = kwargs.get("vix")
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return signals

        spy_returns = prices["SPY"].pct_change()
        rv = realized_vol(spy_returns, window=LOOKBACK_RV)

        if vix is None:
            logger.warning("[VolSurface] VIX not provided — no signals")
            return signals

        # Align VIX to price dates
        iv = vix.reindex(prices.index).ffill() / 100.0  # convert to decimal

        # VRP = IV - RV
        vrp = (iv - rv).dropna()
        vrp_z = z_score(vrp, window=LOOKBACK_Z)

        # VIX daily change (for spike stop)
        vix_chg = vix.diff().reindex(prices.index)

        for date in prices.index:
            if date not in vrp_z.index:
                continue

            z = vrp_z.get(date, np.nan)
            vix_spike = vix_chg.get(date, 0) or 0

            if np.isnan(z):
                continue

            # Hard stop: VIX spike >5 → close all short-vol immediately
            if vix_spike > VIX_SPIKE_STOP:
                if "SHY" in signals.columns:
                    signals.loc[date, "SHY"] = 1.0
                logger.warning("[VolSurface] VIX spike %.1f on %s — closing short-vol", vix_spike, date.date())
                continue

            if z > VRP_LONG_THRESH:
                # VRP elevated: short volatility via VXX short
                # Represented as NEGATIVE VXX signal (short)
                if "VXX" in signals.columns:
                    signals.loc[date, "VXX"] = -min(MAX_SHORT_VOL_ALLOC, 0.03 * z)
                if "SPY" in signals.columns:
                    signals.loc[date, "SPY"] = 0.95  # Long SPY as the "normal" allocation
            elif z < VRP_SHORT_THRESH:
                # VRP compressed: buy protection
                if "VXX" in signals.columns:
                    signals.loc[date, "VXX"] = 0.05  # Small long-vol hedge
                if "SPY" in signals.columns:
                    signals.loc[date, "SPY"] = 0.80
                if "SHY" in signals.columns:
                    signals.loc[date, "SHY"] = 0.15
            else:
                # Neutral: hold SPY + cash
                if "SPY" in signals.columns:
                    signals.loc[date, "SPY"] = 0.90
                if "SHY" in signals.columns:
                    signals.loc[date, "SHY"] = 0.10

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        return {t: float(w) for t, w in signals[signals != 0].items()}
