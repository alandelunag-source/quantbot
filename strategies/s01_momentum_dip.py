"""
S01 — Momentum Dip (Mean Reversion in Uptrend)

Alpha thesis:
  Stocks that dip 3-8% on elevated volume in a strong uptrend are over-sold
  relative to their trend. Institutional accumulation typically follows.
  This captures the fear-driven dip in a healthy uptrend.

Signal logic:
  1. Universe: S&P 500 components
  2. Uptrend filter: close > SMA(200)
  3. Dip: -3% to -8% today vs prior close
  4. Volume surge: today volume > 1.5× 20d avg
  5. RSI(14) < 40 (oversold)
  6. Score = momentum(-dip_pct) × volume_ratio × (40 - RSI) / 40

Entry: next open. Exit: +2.5% target OR 7 calendar days.
Sizing: 2% per position, max 10 concurrent.

Citadel signal quality:
  - IC(21d): historically ~0.05-0.08 on this factor combo
  - Works best in moderate-volatility regimes (VIX 15-25)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import sma, rsi, volume_ratio

logger = logging.getLogger(__name__)


class MomentumDip(Strategy):
    name = "s01_momentum_dip"
    rebalance_freq = "daily"
    max_positions = 10

    DIP_MIN = 0.03
    DIP_MAX = 0.08
    VOL_SURGE = 1.5
    RSI_THRESH = 40
    SMA_TREND = 200
    EXIT_TARGET = 0.025   # +2.5% profit target
    STOP_LOSS   = 0.030   # -3.0% stop-loss (momentum dips that keep falling = fundamental issue)
    EXIT_DAYS   = 7

    def get_universe(self) -> list[str]:
        from data.universe import SP100  # Use SP100 as proxy (liquid subset of SP500)
        return SP100

    def generate_signals(self, prices: pd.DataFrame, volume: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Returns a signal DataFrame. Last row = current signal.
        Positive values = buy candidate, magnitude = confidence.
        """
        if prices.empty or len(prices) < self.SMA_TREND + 5:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for ticker in prices.columns:
            try:
                p = prices[ticker].dropna()
                if len(p) < self.SMA_TREND + 5:
                    continue

                # Trend filter
                trend = sma(p, self.SMA_TREND)
                above_trend = p > trend

                # Dip today
                ret_1d = p.pct_change()
                dip = ret_1d.between(-self.DIP_MAX, -self.DIP_MIN)

                # RSI oversold
                r = rsi(p, 14)
                oversold = r < self.RSI_THRESH

                # Volume surge (if volume provided)
                if volume is not None and ticker in volume.columns:
                    vol_r = volume_ratio(volume[ticker].dropna(), 20)
                    vol_ok = vol_r > self.VOL_SURGE
                else:
                    vol_ok = pd.Series(True, index=p.index)

                # Signal: all conditions met
                trigger = above_trend & dip & oversold & vol_ok

                # Score: higher RSI underrun + deeper dip = stronger signal
                dip_pct = (-ret_1d).clip(self.DIP_MIN, self.DIP_MAX)
                rsi_underrun = (self.RSI_THRESH - r.clip(0, self.RSI_THRESH)) / self.RSI_THRESH
                vol_factor = vol_r.clip(1.0, 5.0) / 5.0 if volume is not None else pd.Series(0.5, index=p.index)

                score = trigger.astype(float) * (dip_pct * 10 + rsi_underrun * 0.5 + vol_factor * 0.2)
                signals[ticker] = score.reindex(signals.index)

            except Exception as exc:
                logger.debug("Signal error for %s: %s", ticker, exc)

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """2% per position, max 10."""
        longs = signals[signals > 0].nlargest(self.max_positions)
        return {t: 0.02 for t in longs.index}

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        """
        Exit when:
          - Profit target reached: +2.5% (momentum inflection captured)
          - Stop-loss: -3.0% (if dip is not reverting, cut and move on)
          - Time stop: 7 calendar days (reversal window has passed)
        """
        ret = (current_price - entry_price) / entry_price
        return ret >= self.EXIT_TARGET or ret <= -self.STOP_LOSS or days_held >= self.EXIT_DAYS
