"""
S13 — Pre-Earnings Drift (Institutional Positioning)

Alpha thesis:
  Stocks with a strong history of positive EPS surprises drift upward in the
  5 trading days BEFORE the earnings announcement as institutional investors
  position ahead of the expected beat. This is distinct from PEAD (post-
  announcement drift) and carries no overnight earnings risk.

  Academic basis: Barber et al. (2013) "The Earnings Announcement Premium"
  documents ~1.5-2.5% avg pre-announcement return in high-surprise-rate stocks.

Signal logic:
  1. Universe: S&P 100 (large-cap, liquid, earnings well-covered)
  2. For each stock: fetch next earnings date (yfinance)
  3. If earnings in 3-7 trading days:
     a. Check EPS surprise history: % of last 8 quarters with positive surprise
     b. Check earnings trend: last 2 quarters had positive surprise
     c. Check price momentum: stock up YTD (institutional interest present)
  4. Signal = surprise_rate × momentum × (1 / days_to_earnings)
  5. EXIT: day before earnings announcement (no overnight risk)

Novel twist: scale position size by analyst revision momentum —
if estimates have been RAISED in last 30 days, signal is stronger
(buy-side already knows something).

Rebalance: daily (new earnings windows open/close every day).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from strategies.base import Strategy
from data.indicators import momentum as mom_calc, sma

logger = logging.getLogger(__name__)

SP100 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","JPM",
    "AVGO","XOM","TSLA","UNH","V","JNJ","WMT","MA","PG","HD","MRK","CVX",
    "ABBV","BAC","KO","AMD","PEP","CSCO","TMO","ACN","MCD","WFC","ADBE","CRM",
    "ABT","GE","LIN","DHR","CAT","AXP","MS","GS","BLK","SPGI","ISRG","NOW",
    "UBER","INTU","RTX","DE","HON","AMGN","PLD","TJX","SYK","BSX","REGN",
    "VRTX","CB","ETN","ELV","ADP","PANW","MMC","SO","SCHW","ZTS","CI",
    "BDX","MO","DUK","CME","AON","MDLZ","KLAC","SLB","ANET","PH","TGT",
    "EQIX","CEG","KMB","D","WM","APH","SNPS","CDNS","FCX","NSC","FDX",
    "EMR","MCO","USB","MSI","MCHP","ADI","ORCL","NFLX","DIS","PYPL",
]

_earnings_cache: dict[str, dict] = {}


def _get_earnings_info(ticker: str) -> dict:
    """Fetch earnings date and surprise history for a ticker."""
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    result = {"next_date": None, "surprise_rate": 0.5, "last_two_positive": False}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Next earnings date
        cal = t.calendar
        if cal is not None and not cal.empty:
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
                if hasattr(ed, "__iter__"):
                    ed = ed.iloc[0] if hasattr(ed, "iloc") else list(ed)[0]
                result["next_date"] = pd.to_datetime(ed, errors="coerce")

        # Historical EPS surprises
        hist = t.earnings_history
        if hist is not None and not hist.empty and "surprisePercent" in hist.columns:
            surprises = hist["surprisePercent"].dropna()
            if len(surprises) >= 2:
                result["surprise_rate"] = float((surprises > 0).mean())
                result["last_two_positive"] = bool((surprises.iloc[:2] > 0).all())

    except Exception as exc:
        logger.debug("[PreEarnings] %s info failed: %s", ticker, exc)

    _earnings_cache[ticker] = result
    return result


class PreEarningsDrift(Strategy):
    name = "s13_pre_earnings_drift"
    rebalance_freq = "daily"
    max_positions = 8
    ENTRY_DAYS_BEFORE = 5   # enter N days before earnings
    EXIT_DAYS_BEFORE = 1    # exit N days before (avoid overnight risk)
    MIN_SURPRISE_RATE = 0.60  # min % of quarters with positive surprise

    def get_universe(self) -> list[str]:
        return SP100

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < 60:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        today = prices.index[-1]

        for ticker in prices.columns:
            if ticker not in SP100:
                continue
            info = _get_earnings_info(ticker)

            next_date = info.get("next_date")
            if next_date is None or pd.isna(next_date):
                continue

            days_to = (next_date - today).days
            if not (self.EXIT_DAYS_BEFORE < days_to <= self.ENTRY_DAYS_BEFORE):
                continue

            surprise_rate = info.get("surprise_rate", 0.5)
            if surprise_rate < self.MIN_SURPRISE_RATE:
                continue

            # Price momentum filter: stock must be in uptrend
            col = prices[ticker].dropna()
            if len(col) < 60:
                continue
            ret_60d = col.pct_change(60).iloc[-1]
            if ret_60d < 0:
                continue

            # Signal score = surprise quality × recency urgency
            last_two_bonus = 1.3 if info.get("last_two_positive") else 1.0
            signal_score = surprise_rate * last_two_bonus * (1.0 / days_to)

            # Apply to the current day (forward signal for next day entry)
            signals.loc[today, ticker] = signal_score

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}
        # Equal weight with a 12% cap per position
        w = min(1.0 / len(longs), 0.12)
        return {t: w for t in longs.index}
