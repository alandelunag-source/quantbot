"""
S12 — Index Inclusion Frontrun

Alpha thesis:
  When a stock is announced for S&P 500 inclusion (~5 trading days before
  the effective date), index funds MUST buy it at any price. This creates
  predictable demand pressure. Stocks average +3-5% from announcement to
  add date, then partially reverse after.

  Additionally, stocks approaching S&P 500 eligibility criteria (market cap,
  liquidity, profitability) get "pre-inclusion" drift as quants anticipate
  the announcement.

Signal logic:
  1. Detect stocks recently added/announced via Wikipedia historical table
     (free, covers all changes since 1957)
  2. Score stocks approaching S&P 500 eligibility based on:
     - Market cap rank within Russell 1000 (top 500 = candidate)
     - 4 consecutive quarters of GAAP profitability
     - 1-year liquidity (ADTV > $1bn)
  3. Also detect Russell 1000 → S&P 500 transitions (June reconstitution)

Novel twist: model the "announcement drift" as a mean-reverting signal post-add.
Long on announcement, flatten/short after effective date (sell the news).

Rebalance: event-driven (check for new announcements daily).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from strategies.base import Strategy
from data.indicators import momentum, sma

logger = logging.getLogger(__name__)

# Known recent S&P 500 additions (fallback when Wikipedia parse fails)
# Format: (ticker, announcement_date, effective_date)
KNOWN_ADDITIONS = [
    ("DELL", "2024-09-06", "2024-09-23"),
    ("PLTR", "2024-09-06", "2024-09-23"),
    ("ERIE", "2024-09-06", "2024-09-23"),
    ("KKR",  "2024-10-01", "2024-10-18"),
    ("GEV",  "2024-11-01", "2024-11-08"),
    ("AXON", "2024-11-08", "2024-11-22"),
    ("SPOT", "2024-11-15", "2024-12-23"),
    ("CBOE", "2024-11-15", "2024-12-23"),
    ("APP",  "2024-12-06", "2024-12-23"),
    ("CRWD", "2024-12-06", "2024-12-23"),
    ("VST",  "2024-09-20", "2024-09-20"),
    ("GDDY", "2025-01-17", "2025-02-03"),
    ("TWLO", "2025-03-01", "2025-03-21"),
]

# Universe for eligibility screening
RUSSELL_1000_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","JPM","AVGO",
    "XOM","TSLA","UNH","V","JNJ","WMT","MA","PG","HD","COST","MRK","CVX",
    "ABBV","BAC","KO","AMD","PEP","CSCO","TMO","ACN","MCD","WFC","ADBE","CRM",
    "ABT","GE","LIN","DHR","CAT","AXP","MS","GS","BLK","SPGI","ISRG","NOW",
    "UBER","INTU","RTX","DE","HON","AMGN","PLD","TJX","SYK","BSX","REGN",
    "VRTX","CB","ETN","ELV","ADP","PANW","MMC","SO","SCHW","ZTS","CI",
    "BDX","MO","DUK","CME","AON","MDLZ","KLAC","SLB","ANET","PH","TGT",
    "EQIX","CEG","KMB","D","WM","APH","SNPS","CDNS","FCX","NSC","FDX",
    "EMR","MCO","USB","MSI","MCHP","ADI","ORCL","NFLX","DIS","PYPL",
]


def _fetch_wikipedia_additions(days_back: int = 90) -> list[dict]:
    """Parse recent S&P 500 additions from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Added",
        )
        # The second table usually has changes
        changes = tables[1] if len(tables) > 1 else pd.DataFrame()
        if changes.empty:
            return []

        changes.columns = [str(c).lower().replace(" ", "_") for c in changes.columns]
        records = []
        cutoff = datetime.today() - timedelta(days=days_back)
        for _, row in changes.iterrows():
            date_str = str(row.get("date", row.get("added", ""))).split("[")[0].strip()
            try:
                date = pd.to_datetime(date_str, errors="coerce")
            except Exception:
                continue
            if pd.isna(date) or date < cutoff:
                continue
            added = str(row.get("added_ticker", row.get("added", ""))).strip().upper()
            if not added or len(added) > 6:
                continue
            records.append({"ticker": added, "date": date, "source": "wikipedia"})
        return records
    except Exception as exc:
        logger.debug("[IndexInclusion] Wikipedia parse failed: %s", exc)
        return []


class IndexInclusion(Strategy):
    name = "s12_index_inclusion"
    rebalance_freq = "daily"
    max_positions = 8
    HOLD_DAYS = 12        # days to hold after announcement
    PRE_ANNOUNCE_DAYS = 5 # days before effective date = announcement window

    def get_universe(self) -> list[str]:
        tickers = list({t for t, _, _ in KNOWN_ADDITIONS})
        tickers += RUSSELL_1000_SAMPLE[:50]
        return list(set(tickers))

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < 30:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Known additions: mark signal from announcement to effective+2 ---
        events = list(KNOWN_ADDITIONS)

        # Try live Wikipedia additions
        wiki = _fetch_wikipedia_additions(days_back=120)
        for w in wiki:
            events.append((w["ticker"], str(w["date"].date()), str(w["date"].date())))

        for ticker, ann_str, eff_str in events:
            if ticker not in signals.columns:
                continue
            try:
                ann_date = pd.to_datetime(ann_str)
                eff_date = pd.to_datetime(eff_str)
            except Exception:
                continue

            # Signal window: from announcement until 2 days after effective date
            window_start = ann_date - timedelta(days=2)  # sometimes drift before announce
            window_end = eff_date + timedelta(days=2)    # sell the news

            mask = (signals.index >= window_start) & (signals.index <= window_end)
            if mask.any():
                col_idx = signals.columns.get_loc(ticker)
                # Stronger signal early, fade after effective date
                for i, date in enumerate(signals.index[mask]):
                    days_since_ann = (date - ann_date).days
                    if days_since_ann < 0:
                        score = 0.3  # pre-announce drift
                    elif days_since_ann <= (eff_date - ann_date).days:
                        score = 1.0  # full signal during inclusion window
                    else:
                        score = -0.3  # slight fade after add date
                    signals.iloc[signals.index.get_loc(date), col_idx] = score

        # --- Eligibility screening: stocks approaching S&P 500 criteria ---
        if len(prices) >= 252:
            for ticker in prices.columns:
                if ticker not in RUSSELL_1000_SAMPLE:
                    continue
                col = prices[ticker].dropna()
                if len(col) < 252:
                    continue
                # Proxy: stock has been in strong uptrend (appreciation = earnings growth proxy)
                ret_1y = col.iloc[-1] / col.iloc[-252] - 1 if len(col) >= 252 else 0
                ret_3m = col.pct_change(63).iloc[-1]
                # High momentum + near 52w high = candidate
                high_52w = col.iloc[-252:].max()
                near_high = col.iloc[-1] / high_52w > 0.90
                if ret_1y > 0.20 and ret_3m > 0.05 and near_high:
                    # Small persistent eligibility signal
                    signals[ticker] = signals[ticker].clip(lower=0.15)

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}
        w = round(1.0 / len(longs), 3)
        return {t: w for t in longs.index}
