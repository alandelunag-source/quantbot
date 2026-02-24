"""
Risk manager — portfolio-level guard rails.

Rules applied before any order goes out:
  1. Max 5% per stock
  2. Max 20% per strategy
  3. Portfolio drawdown stop: if down >12% from peak → halve all positions
  4. Correlation cap: if 2 positions >0.85 → reduce both to 60% of target

Applied in forward testing and paper trading.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.peak_value = settings.INITIAL_CAPITAL
        self._halved = False

    def apply(
        self,
        target_weights: dict[str, float],
        portfolio_value: float,
        price_history: Optional[pd.DataFrame] = None,
        strategy_name: str = "",
    ) -> dict[str, float]:
        """
        Apply all risk rules to target_weights.
        Returns adjusted weights (may be scaled down).
        """
        if not target_weights:
            return {}

        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        # 1. Per-stock cap
        adjusted = {t: min(abs(w), settings.MAX_POSITION_PCT) * np.sign(w)
                    for t, w in target_weights.items()}

        # 2. Per-strategy cap
        total_w = sum(abs(w) for w in adjusted.values())
        if total_w > settings.MAX_STRATEGY_PCT:
            scale = settings.MAX_STRATEGY_PCT / total_w
            adjusted = {t: w * scale for t, w in adjusted.items()}
            logger.debug("[Risk] Strategy cap hit for %s — scaled by %.2f", strategy_name, scale)

        # 3. Drawdown stop
        drawdown = (portfolio_value - self.peak_value) / self.peak_value
        if drawdown < -settings.DRAWDOWN_STOP_PCT:
            if not self._halved:
                logger.warning("[Risk] Portfolio drawdown %.1f%% — halving all positions", drawdown * 100)
                self._halved = True
            adjusted = {t: w * 0.5 for t, w in adjusted.items()}
        else:
            self._halved = False

        # 4. Correlation cap
        if price_history is not None and len(adjusted) > 1:
            adjusted = self._apply_correlation_cap(adjusted, price_history)

        return adjusted

    def _apply_correlation_cap(
        self,
        weights: dict[str, float],
        price_history: pd.DataFrame,
        lookback: int = 60,
    ) -> dict[str, float]:
        """Reduce pairs with correlation > CORRELATION_CAP."""
        tickers = [t for t in weights if t in price_history.columns]
        if len(tickers) < 2:
            return weights

        rets = price_history[tickers].pct_change().iloc[-lookback:]
        corr = rets.corr()

        adjusted = weights.copy()
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                c = corr.loc[t1, t2] if t1 in corr.index and t2 in corr.columns else 0
                if c > settings.CORRELATION_CAP:
                    logger.debug("[Risk] Corr cap: %s <> %s corr=%.2f", t1, t2, c)
                    adjusted[t1] = adjusted.get(t1, 0) * 0.6
                    adjusted[t2] = adjusted.get(t2, 0) * 0.6

        return adjusted
