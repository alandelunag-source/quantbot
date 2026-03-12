"""
S11 — Congressional Trade Follower

Alpha thesis:
  US Congress members legally must disclose stock trades within 45 days.
  Academic research (Ziobrowski et al. 2004, 2011) shows House members
  outperform the market by ~6-10% annually; Senators by ~12%.
  The edge persists even after the STOCK Act (2012) mandated faster disclosure.

  We exploit the information signal: politicians with inside access to
  regulatory decisions, government contracts, and macro policy buy stocks
  they expect to benefit from those decisions.

Signal logic:
  1. Fetch all House + Senate disclosures via FMP API (financialmodelingprep.com)
  2. Filter for PURCHASES of Stocks only (not sales, bonds, ETFs)
  3. Score by: recency (fresher = stronger), number of distinct politicians buying,
     aggregate dollar size, Senate > House bonus
  4. Universe: any ticker disclosed in last 30 days
  5. Rank by composite score, go long top N

Data source: Financial Modeling Prep /stable/senate-latest and /stable/house-latest
  (free tier, 250 req/day — well within budget at 3 sessions/day)

Rebalance: daily (new disclosures arrive any day).
Hold: up to 60 days per position.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import numpy as np

from strategies.base import Strategy

logger = logging.getLogger(__name__)

# Cache disclosures for the session to avoid redundant API calls
_disclosure_cache: dict = {}
_cache_date: str = ""


def _fmp_key() -> str:
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        # Try loading from .env manually (no python-dotenv required)
        try:
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FMP_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        break
        except Exception:
            pass
    return key


def _fetch_disclosures(days_back: int = 45) -> pd.DataFrame:
    """Fetch recent House + Senate stock purchase disclosures via FMP API."""
    global _disclosure_cache, _cache_date

    today = datetime.today().strftime("%Y-%m-%d")
    if _cache_date == today and _disclosure_cache.get(days_back) is not None:
        return _disclosure_cache[days_back]

    import requests
    key = _fmp_key()
    if not key:
        logger.warning("[Congressional] FMP_API_KEY not set — skipping S11")
        return pd.DataFrame()

    cutoff = datetime.today() - timedelta(days=days_back)
    records = []

    for chamber, endpoint in [("senate", "senate-latest"), ("house", "house-latest")]:
        for page in range(1):  # free tier: page 0 only (100 records per chamber)
            try:
                url = f"https://financialmodelingprep.com/stable/{endpoint}?page={page}&apikey={key}"
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for row in data:
                    # Only stocks, only purchases
                    if row.get("assetType", "").lower() not in ("stock", "stocks"):
                        continue
                    tx_type = row.get("type", "").lower()
                    if "purchase" not in tx_type and "buy" not in tx_type:
                        continue
                    disc_date = pd.to_datetime(row.get("disclosureDate"), errors="coerce")
                    if pd.isna(disc_date) or disc_date < cutoff:
                        continue
                    ticker = (row.get("symbol") or "").strip().upper()
                    if not ticker or len(ticker) > 5:
                        continue
                    records.append({
                        "disclosure_date": disc_date,
                        "transaction_date": pd.to_datetime(row.get("transactionDate"), errors="coerce"),
                        "ticker": ticker,
                        "chamber": chamber,
                        "amount": row.get("amount", ""),
                        "politician": f"{row.get('firstName','')} {row.get('lastName','')}".strip(),
                    })
                # If we got fewer than 100 records, no more pages
                if len(data) < 100:
                    break
            except Exception as exc:
                logger.warning("[Congressional] FMP %s page %d failed: %s", chamber, page, exc)
                break

    df = pd.DataFrame(records) if records else pd.DataFrame()
    _disclosure_cache[days_back] = df
    _cache_date = today
    return df


def _amount_to_score(amount_str: str) -> float:
    """Convert STOCK Act amount range to a rough dollar midpoint score."""
    mapping = {
        "$1,001 - $15,000":       8_000,
        "$15,001 - $50,000":     32_500,
        "$50,001 - $100,000":    75_000,
        "$100,001 - $250,000":  175_000,
        "$250,001 - $500,000":  375_000,
        "$500,001 - $1,000,000": 750_000,
        "$1,000,001 - $5,000,000": 3_000_000,
        "over $5,000,000":       7_500_000,
    }
    s = str(amount_str).lower()
    for key, val in mapping.items():
        if key.lower() in s:
            return float(val)
    return 10_000.0


class CongressionalTrades(Strategy):
    name = "s11_congressional"
    rebalance_freq = "daily"
    max_positions  = 10
    LOOKBACK_DAYS  = 30

    STOP_LOSS      = 0.05   # -5%
    PROFIT_TARGET  = 0.15   # +15% (avg congressional excess return, Ziobrowski 2011)
    TIME_STOP_DAYS = 60     # 60-day hold max

    def get_universe(self) -> list[str]:
        """Dynamically return tickers from recent congressional purchases."""
        try:
            df = _fetch_disclosures(days_back=self.LOOKBACK_DAYS + 10)
            if not df.empty and "ticker" in df.columns:
                tickers = df["ticker"].dropna().unique().tolist()
                if tickers:
                    logger.info("[Congressional] Universe: %d tickers from disclosures", len(tickers))
                    return tickers
        except Exception as exc:
            logger.warning("[Congressional] get_universe failed: %s", exc)
        return ["SPY"]  # fallback — signals will be empty but won't crash

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        disclosures = _fetch_disclosures(days_back=self.LOOKBACK_DAYS + 10)
        if disclosures.empty:
            logger.info("[Congressional] No recent disclosures.")
            return pd.DataFrame()

        # Keep purchases only — sales carry no long signal
        if "type" in disclosures.columns:
            disclosures = disclosures[
                disclosures["type"].str.lower().str.contains("purchase|buy", na=False)
            ]
        if disclosures.empty:
            return pd.DataFrame()

        today = datetime.today()
        disclosures["days_old"] = (today - disclosures["disclosure_date"]).dt.days.clip(lower=1)
        disclosures["recency_score"] = np.exp(-disclosures["days_old"] / 15.0)
        disclosures["size_score"]    = disclosures["amount"].apply(_amount_to_score).apply(np.log1p)
        disclosures["senate_bonus"]  = (disclosures["chamber"] == "senate").astype(float) * 0.5

        disclosures["score"] = (
            disclosures["recency_score"] * (1 + disclosures["senate_bonus"])
            + disclosures["size_score"] / 20.0
        )

        ticker_scores = (
            disclosures.groupby("ticker")["score"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total_score", "count": "n_politicians"})
        )
        ticker_scores["final_score"] = (
            ticker_scores["total_score"] * np.log1p(ticker_scores["n_politicians"])
        )

        # Filter to tickers present in price data
        valid = [t for t in ticker_scores.index if t in prices.columns]
        if not valid:
            logger.info("[Congressional] No disclosed tickers in price data. Top picks: %s",
                        ticker_scores.nlargest(5, "final_score").index.tolist())
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        top = ticker_scores.loc[valid].nlargest(self.max_positions, "final_score")
        for ticker in top.index:
            signals[ticker] = top.loc[ticker, "final_score"]

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """Signal-weighted congressional trades: 85% deployed, 15% per-name cap."""
        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}
        return self._sized_weights(longs, prices=prices, max_deploy=0.85, max_weight=0.15)

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        ret = (current_price - entry_price) / entry_price
        return ret <= -self.STOP_LOSS or ret >= self.PROFIT_TARGET or days_held >= self.TIME_STOP_DAYS
