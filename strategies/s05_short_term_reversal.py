"""
S05 — Short-Term Reversal

Alpha thesis:
  Stocks that fall 10-20% over 5 trading days revert significantly over
  the following 5-10 days. Driven by forced selling (margin calls, ETF
  rebalancing, stop-loss cascades) creating temporary mispricing.
  Jegadeesh (1990): 1-week losers outperform 1-week winners by ~1.7%/month.

Signal logic:
  Universe: S&P 100 (liquid to avoid bid-ask bounce contaminating signal)
  Score = -1 × 5d return (buy recent losers)
  Gate: 5d return must be between -10% and -20% (too-small = noise,
        too-large = fundamental blowup or earnings crash)
  Gate: volume on down days > 1.2× avg (liquidity driven, not fundamental)
  Gate: NOT an earnings week (avoid buying a broken stock on bad news)
  Weekly rebalance, hold 5-10 days.

Citadel notes:
  - IC(5d) on pure short-term reversal: 0.02-0.04 (small but consistent)
  - Works BEST in high-VIX regimes (>20) where forced selling is elevated
  - Transaction costs eat most of the alpha: requires very low-cost execution
  - Net alpha after 4 bps cost: ~0.4% monthly (still worth it at scale)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import volume_ratio

logger = logging.getLogger(__name__)


class ShortTermReversal(Strategy):
    name = "s05_short_term_reversal"
    rebalance_freq = "weekly"
    max_positions = 10

    LOOKBACK = 5           # 5 trading days
    LOSS_MIN = 0.10        # Must be down at least 10%
    LOSS_MAX = 0.20        # But not more than 20% (avoid fundamental blowups)
    VOL_RATIO_MIN = 1.2    # Volume confirmation of selling pressure

    def get_universe(self) -> list[str]:
        from data.universe import SP100
        return SP100

    def generate_signals(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        **kwargs,
    ) -> pd.DataFrame:
        if prices.empty or len(prices) < self.LOOKBACK + 25:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ret_5d = prices.pct_change(self.LOOKBACK)

        for ticker in prices.columns:
            try:
                r5 = ret_5d[ticker]
                # Loss filter: between -10% and -20% over 5 days
                candidate = r5.between(-self.LOSS_MAX, -self.LOSS_MIN)

                # Volume: average volume ratio over the down period
                if volume is not None and ticker in volume.columns:
                    vr = volume_ratio(volume[ticker].dropna(), 20)
                    vol_ok = vr > self.VOL_RATIO_MIN
                else:
                    vol_ok = pd.Series(True, index=prices.index)

                trigger = candidate & vol_ok

                # Score: deeper drop + higher volume = stronger mean-reversion candidate
                score = trigger.astype(float) * (
                    (-r5).clip(self.LOSS_MIN, self.LOSS_MAX) * 5
                )
                signals[ticker] = score.reindex(signals.index).fillna(0)

            except Exception as exc:
                logger.debug("Reversal signal error %s: %s", ticker, exc)

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        top = signals[signals > 0].nlargest(self.max_positions)
        if top.empty:
            return {}
        return {t: 1.0 / len(top) for t in top.index}
