"""
Abstract base strategy.

Every strategy must implement:
  generate_signals() -> pd.DataFrame (index=date, cols=ticker, values=signal score -1..1)
  get_universe()     -> list[str]
  name               -> str

Optional overrides:
  rebalance_freq     -> "daily" | "weekly" | "monthly"
  max_positions      -> int
  position_sizing(signals) -> dict[str, float]  (ticker -> target weight)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Strategy(ABC):
    # Override in subclasses
    rebalance_freq: str = "daily"   # "daily" | "weekly" | "monthly"
    max_positions: int = 20

    # Citadel-style sizing defaults — override in subclasses
    MAX_DEPLOY: float = 0.95   # max fraction of portfolio to invest (keep 5% reserve)
    MAX_WEIGHT: float = 0.20   # max single-name weight

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_universe(self) -> list[str]: ...

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Given a price history DataFrame (index=dates, cols=tickers),
        return a signal DataFrame with the same shape.
        Signal values: positive = long, negative = short, 0 = flat.
        """
        ...

    # ------------------------------------------------------------------
    # Sizing utilities
    # ------------------------------------------------------------------

    def _sized_weights(
        self,
        signals: pd.Series,
        prices: pd.DataFrame = None,
        max_deploy: float = None,
        max_weight: float = None,
    ) -> dict[str, float]:
        """
        Citadel-style position sizing:
          1. Signal-proportional weights  — higher conviction → more capital
          2. Inverse-vol tilt             — equal risk contribution per name
          3. Capital reserve              — total deployed ≤ max_deploy
          4. Single-name cap              — no position > max_weight

        Args:
            signals:    Cross-sectional signal scores (positive = long candidate).
            prices:     Price history DataFrame (index=dates, cols=tickers).
                        If provided, applies 20-day realized-vol inverse-weighting.
            max_deploy: Maximum fraction of portfolio to deploy (default: MAX_DEPLOY).
            max_weight: Per-name cap (default: MAX_WEIGHT).
        """
        max_deploy = max_deploy if max_deploy is not None else self.MAX_DEPLOY
        max_weight = max_weight if max_weight is not None else self.MAX_WEIGHT

        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}

        # Step 1: signal-proportional raw weights (sum → 1)
        total_sig = longs.sum()
        raw_w = longs / total_sig

        # Step 2: inverse-vol tilt (equal risk contribution)
        if prices is not None and not prices.empty:
            tickers_present = [t for t in longs.index if t in prices.columns]
            if len(tickers_present) >= 2:
                ret = prices[tickers_present].pct_change().iloc[-21:]
                ann_vol = ret.std() * np.sqrt(252)
                inv_vol = (1.0 / ann_vol.clip(lower=0.05)).reindex(longs.index).fillna(1.0)
                raw_w = raw_w * inv_vol
                raw_w = raw_w / raw_w.sum()   # renormalize after vol tilt

        # Step 3: scale to max_deploy
        raw_w = raw_w * max_deploy

        # Step 4: clip to max_weight (excess stays as cash reserve)
        raw_w = raw_w.clip(upper=max_weight)

        return {t: float(w) for t, w in raw_w.items()}

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """
        Convert a cross-sectional signal series (date=now, index=tickers)
        into target portfolio weights.

        Default: signal-weighted top N, capped at MAX_DEPLOY total and MAX_WEIGHT per name.
        Override in subclasses for strategy-specific parameters.
        """
        if signals.empty:
            return {}
        return self._sized_weights(signals, prices=prices)

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        """
        Return True if a position should be force-closed (stop-loss, profit target, time stop).
        Called by ForwardTest on every mark-to-market, regardless of rebalance schedule.

        Override in subclasses with strategy-specific thresholds.
        Default: no forced exit (signal-driven exit only).
        """
        return False

    def get_description(self) -> str:
        return f"{self.name} | freq={self.rebalance_freq} | max_pos={self.max_positions}"


class LongShortStrategy(Strategy):
    """Base for long-short strategies with separate long/short books."""
    long_count: int = 10
    short_count: int = 10

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        if signals.empty:
            return {}
        longs = signals.nlargest(self.long_count)
        shorts = signals.nsmallest(self.short_count)
        weights = {}
        if not longs.empty:
            w = 0.5 / len(longs)
            for t in longs.index:
                weights[t] = w
        if not shorts.empty:
            w = 0.5 / len(shorts)
            for t in shorts.index:
                weights[t] = -w
        return weights
