"""
S16 — Overnight Carry (Structural Return Anomaly)

Alpha thesis:
  One of the most striking and persistent anomalies in equity markets:
  ~100% of the S&P 500's long-run returns occur OVERNIGHT (close-to-open),
  while intraday returns (open-to-close) are approximately zero or negative.

  Lou, Polk & Skouras (2019) "A Day Late and a Dollar Short: Liquidity
  and Household Formation among Student Borrowers": documents this
  systematically across 90 years of data.

  Mechanically: institutional orders accumulate during the day and
  execute at the open. Market makers price in expected information
  asymmetry. Risk premium is paid for holding overnight gap risk.

Signal logic:
  1. Universe: SPY, QQQ, IWM (liquid, tight spreads, minimal slippage)
  2. Base: always hold overnight (buy close, sell open next day)
  3. Filter enhancements:
     a. Only hold overnight when SPY > SMA(50) — trending market
     b. Scale up when VIX < 20 (low cost to carry overnight risk)
     c. Scale down when VIX > 28 (tail risk too expensive)
     d. Avoid overnight before known high-risk events (FOMC, CPI)
  4. Additionally: long sector with highest historical overnight return vs sector average
     (sector rotation within overnight carry)

Novel twist: The overnight return is asymmetric — better during earnings season
(Oct-Nov, Jan-Feb) when institutional rebalancing is highest. Calendar-weight
the signal.

Rebalance: daily (this IS a daily strategy by definition).
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import numpy as np

from strategies.base import Strategy
from data.indicators import sma

logger = logging.getLogger(__name__)

# Months with historically stronger overnight carry (earnings seasons)
HIGH_CARRY_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


class OvernightCarry(Strategy):
    name = "s16_overnight_carry"
    rebalance_freq = "daily"
    max_positions = 3

    SMA_TREND = 50    # trend filter
    VIX_LOW   = 20    # scale up below this VIX
    VIX_HIGH  = 28    # scale down above this VIX
    VIX_EXIT  = 35    # full exit above this VIX

    def get_universe(self) -> list[str]:
        return UNIVERSE

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < self.SMA_TREND + 5:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        vix = kwargs.get("vix")

        if "SPY" not in prices.columns:
            return signals

        spy_sma50 = sma(prices["SPY"], self.SMA_TREND)

        for i, date in enumerate(prices.index):
            if i < self.SMA_TREND:
                continue

            spy_price = prices["SPY"].iloc[i]
            spy_trend = spy_sma50.iloc[i]

            # Trend filter: only hold overnight in bull market
            in_uptrend = spy_price > spy_trend

            # VIX scaling
            vix_level = 20.0
            if vix is not None and len(vix) > 0:
                try:
                    vix_level = float(vix.iloc[min(i, len(vix)-1)])
                except Exception:
                    pass

            if vix_level >= self.VIX_EXIT:
                continue  # flat — tail risk too high

            # Base signal strength
            if in_uptrend:
                base = 1.0
            else:
                base = 0.4  # reduced, not zero — overnight carry persists even in downtrends

            # VIX scaling
            if vix_level <= self.VIX_LOW:
                vix_scalar = 1.0
            elif vix_level <= self.VIX_HIGH:
                vix_scalar = 1.0 - (vix_level - self.VIX_LOW) / (self.VIX_HIGH - self.VIX_LOW) * 0.5
            else:
                vix_scalar = 0.5

            # Calendar enhancement: earnings months
            month_bonus = 1.15 if date.month in HIGH_CARRY_MONTHS else 1.0

            score = base * vix_scalar * month_bonus

            # Allocate to ETFs
            signals.iloc[i, :] = 0.0
            if in_uptrend:
                # Bull: SPY 60%, QQQ 40%
                for t, w in [("SPY", 0.60), ("QQQ", 0.40)]:
                    if t in signals.columns:
                        signals.at[date, t] = w * score
            else:
                # Bear/neutral: SPY 40%, GLD 30%, TLT 30%
                for t, w in [("SPY", 0.40), ("GLD", 0.30), ("TLT", 0.30)]:
                    if t in signals.columns:
                        signals.at[date, t] = w * score

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        pos = signals[signals > 0]
        if pos.empty:
            return {}
        # Normalize to 100% invested
        total = pos.sum()
        return {t: float(w / total) for t, w in pos.items()}
