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

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        """
        Convert a cross-sectional signal series (date=now, index=tickers)
        into target portfolio weights.

        Default: equal-weight top N long signals.
        Override for custom sizing (Kelly, vol-targeting, etc.)
        """
        if signals.empty:
            return {}

        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}

        weight = 1.0 / len(longs)
        return {ticker: weight for ticker in longs.index}

    def get_description(self) -> str:
        return f"{self.name} | freq={self.rebalance_freq} | max_pos={self.max_positions}"


class LongShortStrategy(Strategy):
    """Base for long-short strategies with separate long/short books."""
    long_count: int = 10
    short_count: int = 10

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
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
