"""
S02 — Cross-Asset Momentum (Trend Following ETFs)

Alpha thesis:
  Asset class trends persist for 3-12 months. Absolute momentum (positive
  12mo return) avoids bear markets. Equal-weighting top 3 assets monthly
  captures cross-asset leadership rotation.

Signal logic:
  Universe: SPY, QQQ, IWM, GLD, TLT, DBC, VNQ, EFA, IEMG, SHY
  Score per asset = 12mo total return − 1mo total return (skip-1 momentum)
  Monthly rebalance: go equal-weight top 3 assets with positive absolute momentum.
  If no asset has positive 12mo return → 100% SHY (bear market protection).

Academic grounding: Moskowitz, Ooi, Pedersen (2012) time-series momentum.
Citadel-grade enhancement: dual-momentum (absolute + relative) avoids the
worst drawdowns vs pure relative momentum.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy

logger = logging.getLogger(__name__)


class CrossAssetMomentum(Strategy):
    name = "s02_cross_asset_mom"
    rebalance_freq = "monthly"
    max_positions = 3

    SAFE_HAVEN = "SHY"
    N_TOP = 3
    LOOKBACK_LONG = 252     # 12 months
    LOOKBACK_SHORT = 21     # 1 month (skip)

    def get_universe(self) -> list[str]:
        from data.universe import ETF_UNIVERSE
        return ETF_UNIVERSE

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < self.LOOKBACK_LONG + 5:
            return pd.DataFrame()

        # Only compute on month-end dates (for backtest efficiency)
        monthly_idx = prices.resample("ME").last().index
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date in monthly_idx:
            if date not in prices.index:
                # Find nearest date
                idx = prices.index.searchsorted(date) - 1
                if idx < 0:
                    continue
                date = prices.index[idx]

            if prices.index.get_loc(date) < self.LOOKBACK_LONG:
                continue

            snap = prices.loc[:date]
            ret_12m = snap.iloc[-1] / snap.iloc[-self.LOOKBACK_LONG] - 1
            ret_1m = snap.iloc[-1] / snap.iloc[-self.LOOKBACK_SHORT] - 1

            # Skip-1 momentum
            score = ret_12m - ret_1m

            # Absolute momentum gate: only rank assets with positive 12m return
            score_filtered = score.where(ret_12m > 0, other=-999)

            if (score_filtered == -999).all():
                # All assets in downtrend → cash
                signal_today = pd.Series(0.0, index=prices.columns)
                if self.SAFE_HAVEN in prices.columns:
                    signal_today[self.SAFE_HAVEN] = 1.0
            else:
                signal_today = score_filtered.clip(lower=0)

            # Apply this signal to all dates until next month-end
            loc = prices.index.get_loc(date)
            next_loc = min(loc + 22, len(prices))
            signals.iloc[loc:next_loc] = signal_today.values

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """Signal-weighted, 95% deployed, 50% per-asset cap. Falls back to safe-haven if no longs."""
        longs = signals[signals > 0].nlargest(self.N_TOP)
        if longs.empty:
            return {self.SAFE_HAVEN: 0.95} if self.SAFE_HAVEN else {}
        return self._sized_weights(longs, prices=prices, max_deploy=0.95, max_weight=0.50)
