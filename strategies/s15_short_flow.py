"""
S15 — Institutional Short Flow (Dark Pool Signal)

Alpha thesis:
  FINRA publishes daily short selling volume data (free) for all US equity
  venues including ADF (Alternative Display Facility) — the primary venue
  for dark pool prints. Short volume as % of total volume is a leading
  indicator of institutional bearish conviction.

  Key insight: SHORT VOLUME != SHORT INTEREST.
  Short volume is a DAILY flow measure. High short volume ratio (>55%) means
  institutions are actively selling short today. When sustained over 5+ days,
  it predicts continued downside.

  Contrarian edge: when short volume ratio spikes to extreme (>70%) AND
  the stock has already fallen significantly, the short-squeeze probability
  is high -> fade the shorts.

Signal logic:
  Fetch FINRA daily short volume data from finra.org
  Compute 5-day rolling short volume ratio per ticker
  HIGH sustained ratio (>55% for 3 days) -> BEARISH signal (short or avoid)
  EXTREME ratio (>70%) after large drop -> CONTRARIAN BULLISH (squeeze setup)
  LOW ratio (<35%) with price strength -> CONFIRM long momentum

Novel twist: combine with options put/call ratio for confluence.
When both short volume and put/call ratio are extreme, conviction doubles.

Rebalance: daily.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

from strategies.base import Strategy
from data.indicators import momentum as mom_calc

logger = logging.getLogger(__name__)

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

_flow_cache: dict[str, pd.DataFrame] = {}


def _fetch_finra_short_volume(date: datetime) -> pd.DataFrame:
    """Fetch FINRA short volume data for a specific date."""
    date_str = date.strftime("%Y%m%d")
    if date_str in _flow_cache:
        return _flow_cache[date_str]

    url = FINRA_URL.format(date=date_str)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        df = pd.read_csv(
            io.StringIO(resp.text),
            sep="|",
            usecols=["Symbol", "ShortVolume", "TotalVolume"],
            dtype={"Symbol": str, "ShortVolume": float, "TotalVolume": float},
        )
        df = df.dropna()
        df["short_ratio"] = df["ShortVolume"] / df["TotalVolume"].replace(0, np.nan)
        df = df.set_index("Symbol")
        _flow_cache[date_str] = df
        return df
    except Exception as exc:
        logger.debug("[ShortFlow] FINRA fetch failed for %s: %s", date_str, exc)
        return pd.DataFrame()


def _get_short_ratios(tickers: list[str], lookback_days: int = 10) -> pd.DataFrame:
    """
    Return DataFrame of short_ratio indexed by date, columns by ticker.
    """
    today = datetime.today()
    rows = []
    for d in range(lookback_days, 0, -1):
        date = today - timedelta(days=d)
        if date.weekday() >= 5:  # skip weekends
            continue
        df = _fetch_finra_short_volume(date)
        if df.empty:
            continue
        row = {"date": date.date()}
        for t in tickers:
            if t in df.index:
                row[t] = df.loc[t, "short_ratio"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).set_index("date")
    return result


UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "HD", "BAC", "KO",
    "AMD", "CSCO", "MCD", "WFC", "ADBE", "CRM", "ABT",
    "SPY", "QQQ", "IWM",
]


class ShortFlow(Strategy):
    name = "s15_short_flow"
    rebalance_freq = "daily"
    max_positions = 8

    # Thresholds
    BEARISH_RATIO  = 0.55   # sustained selling -> avoid
    SQUEEZE_RATIO  = 0.68   # extreme short + drop -> long contrarian
    BULLISH_RATIO  = 0.38   # low short flow + momentum -> confirm long

    STOP_LOSS      = 0.06   # -6%: short covering not materializing; shorts may be right
    PROFIT_TARGET  = 0.12   # +12%: short squeeze can be violent; take profits before unwind ends
    TIME_STOP_DAYS = 30     # 30-day max (squeeze events resolve within a month)

    def get_universe(self) -> list[str]:
        return UNIVERSE

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < 20:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        tickers = [t for t in prices.columns if t in UNIVERSE]

        if not tickers:
            return signals

        # Fetch short flow data for live signals
        try:
            flow_df = _get_short_ratios(tickers, lookback_days=15)
        except Exception as exc:
            logger.warning("[ShortFlow] Data fetch failed: %s", exc)
            flow_df = pd.DataFrame()

        if flow_df.empty:
            # Fallback: use volume-based proxy (high volume + declining = bearish)
            volume = kwargs.get("volume")
            if volume is not None and not volume.empty:
                for ticker in tickers:
                    if ticker not in prices.columns or ticker not in volume.columns:
                        continue
                    vol_ratio = volume[ticker] / volume[ticker].rolling(20).mean()
                    price_change = prices[ticker].pct_change(5)
                    # High volume + falling price = institutional selling
                    squeeze_mask = (vol_ratio > 2.0) & (price_change < -0.08)
                    confirm_mask = (vol_ratio > 1.5) & (price_change > 0.03)
                    signals.loc[squeeze_mask[squeeze_mask].index, ticker] = 0.6
                    signals.loc[confirm_mask[confirm_mask].index, ticker] = 0.4
            return signals

        # Apply live short flow signals to latest date
        latest = prices.index[-1]
        for ticker in tickers:
            if ticker not in flow_df.columns:
                continue
            ratios = flow_df[ticker].dropna()
            if len(ratios) < 3:
                continue

            current_ratio = ratios.iloc[-1]
            avg_ratio_3d = ratios.iloc[-3:].mean()

            # Check price context
            if ticker not in prices.columns:
                continue
            price_5d = prices[ticker].pct_change(5).iloc[-1]

            if current_ratio >= self.SQUEEZE_RATIO and price_5d < -0.05:
                # Extreme shorts + big drop -> squeeze setup
                signals.loc[latest, ticker] = 0.8
            elif avg_ratio_3d <= self.BULLISH_RATIO and price_5d > 0:
                # Low short pressure + positive price -> confirm long
                signals.loc[latest, ticker] = 0.5
            elif avg_ratio_3d >= self.BEARISH_RATIO:
                # Sustained selling -> bearish (use negative to signal avoid)
                signals.loc[latest, ticker] = -0.3

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """Signal-weighted short-squeeze candidates: 80% deployed, 20% per-name cap."""
        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}
        return self._sized_weights(longs, prices=prices, max_deploy=0.80, max_weight=0.20)

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        """
        Exit when:
          - Stop-loss: -6% (short squeeze thesis failed; momentum may have reversed)
          - Profit target: +12% (squeeze captured; lock in before short covering exhausts)
          - Time stop: 30 days (squeeze events are short-lived; stale signal = different regime)
        """
        ret = (current_price - entry_price) / entry_price
        return ret <= -self.STOP_LOSS or ret >= self.PROFIT_TARGET or days_held >= self.TIME_STOP_DAYS
