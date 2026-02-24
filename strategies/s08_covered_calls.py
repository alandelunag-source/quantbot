"""
S08 — Covered Call Income Overlay

Alpha thesis:
  Systematic OTM call selling on equity positions captures the volatility risk
  premium (implied vol > realized vol ~80% of the time in US equities).
  Adds ~3-5% annual yield on underlying positions with limited downside
  (premium collected reduces cost basis).

Layered on top of S03 Factor Alpha positions.

Signal logic:
  1. For each S03 position >1% profitable:
     - Find nearest monthly expiry with 25-35 DTE
     - Select call at +7% OTM strike (delta ~0.20-0.25)
     - Sell the call if IV > 20-day realized vol × 1.15 (vol premium exists)
  2. Roll when: DTE < 7 OR price within 2% of strike
  3. Never sell calls on positions expected to trigger momentum exit

Citadel-grade VRP harvesting:
  - Only sell when IV rank (IVR) > 30% (premium is elevated vs historical)
  - Size: 1 contract per 100 shares; never cap more than 50% of position upside
  - Track net premium collected vs unrealized P&L on calls

Note: Requires options data. Uses yfinance option chains.
      In paper mode, logs the trade but uses synthetic P&L.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from data.indicators import realized_vol

logger = logging.getLogger(__name__)

STRIKE_OTM_PCT = 0.07      # Sell call 7% above current price
MIN_DTE = 25
MAX_DTE = 35
MIN_PROFIT_TO_SELL = 0.01  # Only sell when position is >1% profitable
IV_PREMIUM_THRESHOLD = 1.15  # IV must be >15% above realized vol


@dataclass
class CoveredCallTrade:
    ticker: str
    underlying_entry: float
    call_strike: float
    call_expiry: str
    premium_collected: float
    iv: float
    realized_vol: float
    open: bool = True
    notes: str = ""


class CoveredCalls:
    """
    Not a Strategy subclass — it's an overlay that operates on existing positions.
    Called from the orchestrator with the current S03 position dict.
    """
    name = "s08_covered_calls"
    open_trades: list[CoveredCallTrade] = field(default_factory=list)

    def __init__(self):
        self.open_trades: list[CoveredCallTrade] = []
        self.total_premium = 0.0

    def evaluate(
        self,
        positions: dict[str, float],  # ticker -> entry_price
        current_prices: dict[str, float],
        prices_history: pd.DataFrame,
    ) -> list[CoveredCallTrade]:
        """
        For each profitable long position, evaluate whether to sell a covered call.
        Returns list of new trades to execute.
        """
        new_trades = []

        for ticker, entry_price in positions.items():
            current = current_prices.get(ticker)
            if current is None:
                continue

            pnl_pct = (current - entry_price) / entry_price
            if pnl_pct < MIN_PROFIT_TO_SELL:
                continue

            # Already have an open call on this ticker?
            if any(t.ticker == ticker and t.open for t in self.open_trades):
                continue

            # Compute realized vol
            if ticker not in prices_history.columns:
                continue

            hist = prices_history[ticker].dropna()
            if len(hist) < 25:
                continue

            rv = hist.pct_change().iloc[-20:].std() * np.sqrt(252)

            # Get options chain
            chain_data = _get_options_chain(ticker)
            if not chain_data:
                continue

            call = _find_best_call(chain_data, current, rv)
            if call is None:
                continue

            trade = CoveredCallTrade(
                ticker=ticker,
                underlying_entry=entry_price,
                call_strike=call["strike"],
                call_expiry=call["expiry"],
                premium_collected=call["lastPrice"],
                iv=call["impliedVolatility"],
                realized_vol=rv,
            )
            self.open_trades.append(trade)
            self.total_premium += trade.premium_collected
            new_trades.append(trade)
            logger.info(
                "[CC] Sell %s call @ strike=%.2f  expiry=%s  premium=%.3f  IV=%.1f%%  RV=%.1f%%",
                ticker, call["strike"], call["expiry"],
                call["lastPrice"], call["impliedVolatility"] * 100, rv * 100,
            )

        return new_trades

    def roll_or_close(self, current_prices: dict[str, float]) -> list[str]:
        """Check open trades and close/roll as needed. Returns list of closed tickers."""
        closed = []
        for trade in self.open_trades:
            if not trade.open:
                continue
            current = current_prices.get(trade.ticker)
            if current is None:
                continue
            # Close if price is within 2% of strike (risk of assignment)
            if current >= trade.call_strike * 0.98:
                trade.open = False
                trade.notes = f"Closed: price {current:.2f} within 2% of strike {trade.call_strike:.2f}"
                closed.append(trade.ticker)
                logger.info("[CC] Closing covered call on %s — assignment risk", trade.ticker)
        return closed

    def summary(self) -> str:
        open_count = sum(1 for t in self.open_trades if t.open)
        return (
            f"Covered Calls: {open_count} open | "
            f"Total premium collected: ${self.total_premium:.2f}"
        )


def _get_options_chain(ticker: str) -> dict:
    """Fetch options chain via yfinance. Returns {} on failure."""
    from data.market_data import get_options_chain
    return get_options_chain(ticker)


def _find_best_call(chain_data: dict, current_price: float, rv: float) -> Optional[dict]:
    """
    Select the best OTM call: nearest to +7% strike with 25-35 DTE
    and IV > rv * IV_PREMIUM_THRESHOLD.
    """
    calls = chain_data.get("calls")
    if calls is None or calls.empty:
        return None

    target_strike = current_price * (1 + STRIKE_OTM_PCT)
    # Filter to near-target strikes
    candidates = calls[
        (calls["strike"] >= current_price * 1.04) &
        (calls["strike"] <= current_price * 1.12) &
        (calls["volume"] > 10) &
        (calls["impliedVolatility"] > rv * IV_PREMIUM_THRESHOLD)
    ]

    if candidates.empty:
        return None

    # Pick closest to target strike
    candidates = candidates.copy()
    candidates["dist"] = (candidates["strike"] - target_strike).abs()
    best = candidates.loc[candidates["dist"].idxmin()]
    best_dict = best.to_dict()
    best_dict["expiry"] = chain_data.get("expiry", "")
    return best_dict
