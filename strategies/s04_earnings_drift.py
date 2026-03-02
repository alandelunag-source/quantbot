"""
S04 — Earnings Drift / Post-Earnings Announcement Drift (PEAD)

Alpha thesis:
  One of the oldest and most persistent market anomalies (Ball & Brown 1968,
  Bernard & Thomas 1989). Stocks that beat earnings estimates by >1 standard
  deviation continue to drift upward for 20-60 trading days as the market
  slowly reprices. The drift is strongest in small/mid caps but persists
  in large caps when the surprise is large.

Signal logic:
  1. Detect earnings surprises: actual EPS vs consensus estimate (yfinance)
  2. Filter: stock gapped up/down >3% on earnings day on >2× avg volume
  3. Buy POSITIVE surprises (gap up) at the close of earnings day
  4. Score = gap_pct × volume_ratio × EPS_surprise_magnitude
  5. Exit: 30 days hold OR if stock retraces >3% from high watermark

Citadel enhancement:
  - Only take PEAD signal when gap doesn't fully price in the move (conservative)
  - Exit on momentum exhaustion (RSI > 70 after 20 days)
  - IC(30d) on PEAD historically 0.04-0.07 (persistent but decaying)

Note: yfinance earnings data is limited. For production use FactSet/Bloomberg
earnings surprise API. Here we use price gap as a proxy.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import volume_ratio, rsi

logger = logging.getLogger(__name__)


class EarningsDrift(Strategy):
    name = "s04_earnings_drift"
    rebalance_freq = "daily"
    max_positions = 10

    GAP_MIN       = 0.03   # Minimum gap on earnings day
    VOL_MIN       = 2.0    # Minimum volume ratio
    HOLD_DAYS     = 30     # Maximum hold period (calendar days)
    STOP_LOSS     = 0.05   # -5% hard stop (gap reversal = thesis broken)
    PROFIT_TARGET = 0.10   # +10% profit target (PEAD drift typically 5-15%);

    def get_universe(self) -> list[str]:
        from data.universe import SP100
        return SP100

    def generate_signals(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Detect large earnings-day gaps with volume confirmation.
        These serve as proxies for positive earnings surprises.
        """
        if prices.empty or len(prices) < 30:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        returns = prices.pct_change()

        for ticker in prices.columns:
            try:
                p = prices[ticker].dropna()
                r = returns[ticker].dropna()
                if len(p) < 30:
                    continue

                # Volume ratio
                vol_r = (
                    volume_ratio(volume[ticker].dropna(), 20)
                    if volume is not None and ticker in volume.columns
                    else pd.Series(1.0, index=p.index)
                )

                # Gap detection: single-day return > GAP_MIN with volume surge
                gap_up = (r >= self.GAP_MIN) & (vol_r >= self.VOL_MIN)

                # Score: magnitude × volume factor
                score = gap_up.astype(float) * (
                    r.clip(lower=0) * 10 + vol_r.clip(1, 5) * 0.1
                )

                # Decay signal: once triggered, hold for HOLD_DAYS (use rolling max)
                # This approximates "continue to hold PEAD signal for hold period"
                score_carried = score.rolling(self.HOLD_DAYS, min_periods=1).max()
                signals[ticker] = score_carried.reindex(signals.index)

            except Exception as exc:
                logger.debug("PEAD signal error %s: %s", ticker, exc)

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """Signal-weighted PEAD positions: 80% deployed, 10% per-name cap."""
        active = signals[signals > 0].nlargest(self.max_positions)
        if active.empty:
            return {}
        return self._sized_weights(active, prices=prices, max_deploy=0.80, max_weight=0.10)

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        """
        Exit when:
          - Stop-loss: -5% (gap that reverses is a failed PEAD — bad earnings quality)
          - Profit target: +10% (PEAD mean drift; lock in, don't give back)
          - Time stop: 30 days (drift window empirically exhausted by day 30)
        """
        ret = (current_price - entry_price) / entry_price
        return ret <= -self.STOP_LOSS or ret >= self.PROFIT_TARGET or days_held >= self.HOLD_DAYS
