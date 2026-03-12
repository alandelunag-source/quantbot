"""
S19 — Turn-of-Month

Alpha thesis:
  Stock returns are significantly elevated in the last ~4 trading days of
  the month and first ~3 of the next (Ariel 1987, Lakonishok & Smidt 1988).
  Driven by institutional rebalancing flows, pension fund/401k cash deployment,
  and window dressing creating predictable month-end buying pressure.
  ~0.4–0.6% per 8-day window, ~9 cycles/year. Low turnover, minimal costs.
  Effect is persistent post-publication and robust across global markets.

Signal logic:
  Window : last TOM_ENTRY_DAYS trading days of month + first TOM_EXIT_DAYS of next
  In window  AND VIX <= VIX_MAX  ->  100% SPY
  Otherwise                      ->  100% SHY

VIX gate:
  Skip the TOM trade when VIX > 30. During macro crises the seasonal premium
  is overwhelmed by directional selling; cash is safer than riding the window.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.base import Strategy

logger = logging.getLogger(__name__)


class TurnOfMonth(Strategy):
    name            = "s19_turn_of_month"
    rebalance_freq  = "daily"
    max_positions   = 1

    TOM_ENTRY_DAYS = 4     # last N trading days of the month (go long SPY)
    TOM_EXIT_DAYS  = 3     # first M trading days of next month (stay long SPY)
    VIX_MAX        = 30.0  # skip TOM trade if VIX exceeds this (crisis override)

    def get_universe(self) -> list[str]:
        return ["SPY", "SHY"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        vix = kwargs.get("vix", pd.Series(dtype=float))
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Build TOM window using calendar-based counts so mid-month dates are
        # never misclassified when the month is still in progress.
        tom_dates: set = set()
        for date in prices.index:
            # Last TOM_ENTRY_DAYS of the month: count business days from date to month-end
            month_end = date + pd.offsets.BMonthEnd(0)
            bdays_to_end = len(pd.bdate_range(date, month_end))
            if bdays_to_end <= self.TOM_ENTRY_DAYS:
                tom_dates.add(date)
                continue
            # First TOM_EXIT_DAYS of the month: count business days from month-start to date
            month_start = pd.Timestamp(date.year, date.month, 1)
            bdays_from_start = len(pd.bdate_range(month_start, date))
            if bdays_from_start <= self.TOM_EXIT_DAYS:
                tom_dates.add(date)

        for i, date in enumerate(prices.index):
            vix_level = np.nan
            if not vix.empty:
                try:
                    vix_level = float(vix.iloc[min(i, len(vix) - 1)])
                except Exception:
                    pass

            in_window = date in tom_dates
            vix_ok    = np.isnan(vix_level) or vix_level <= self.VIX_MAX

            if in_window and vix_ok:
                if "SPY" in signals.columns:
                    signals.at[date, "SPY"] = 1.0
            else:
                if "SHY" in signals.columns:
                    signals.at[date, "SHY"] = 1.0

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """100% in SPY (TOM window) or 100% in SHY (out of window / crisis)."""
        active = signals[signals > 0]
        if active.empty:
            return {"SHY": 1.0}
        if "SPY" in active.index:
            return {"SPY": 1.0}
        return {str(active.index[0]): 1.0}
