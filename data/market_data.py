"""
Market data layer.

Primary source: yfinance (free, no key needed)
Supplementary: Alpaca REST API for intraday / live bars

All data returned as pandas DataFrames with DatetimeIndex.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def get_bars(
    tickers: list[str],
    days: int = 365,
    interval: str = "1d",
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for multiple tickers.

    Returns:
        MultiIndex DataFrame (cols: Open/High/Low/Close/Volume × ticker).
        Adjusted close used where available.
    """
    end = end or datetime.today()
    start = end - timedelta(days=days + 30)  # extra buffer for weekends/holidays

    try:
        df = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    if df.empty:
        logger.warning("No data returned for tickers: %s", tickers)
    return df


def get_close(tickers: list[str], days: int = 365) -> pd.DataFrame:
    """Return adjusted close prices as a simple DataFrame (columns = tickers)."""
    df = get_bars(tickers, days=days)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].dropna(how="all")
    return df[["Close"]].rename(columns={"Close": tickers[0]}).dropna()


def get_volume(tickers: list[str], days: int = 365) -> pd.DataFrame:
    """Return daily volume as a DataFrame (columns = tickers)."""
    df = get_bars(tickers, days=days)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        return df["Volume"].dropna(how="all")
    return df[["Volume"]].rename(columns={"Volume": tickers[0]}).dropna()


# ---------------------------------------------------------------------------
# Fundamental / macro data
# ---------------------------------------------------------------------------

def get_vix(days: int = 365) -> pd.Series:
    """Return daily VIX closing values."""
    df = get_close(["^VIX"], days=days)
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rename("VIX")


def get_yield_spread(days: int = 365) -> pd.Series:
    """
    Return 2Y-10Y yield spread (bps).
    Uses TNX (10Y) and IRX (13-week, proxy for 2Y) from yfinance.
    """
    df = get_close(["^TNX", "^IRX"], days=days)
    if df.empty or len(df.columns) < 2:
        return pd.Series(dtype=float)
    return (df["^TNX"] - df["^IRX"]).rename("yield_spread")


def get_earnings_calendar(ticker: str) -> Optional[pd.DataFrame]:
    """Return upcoming earnings dates and EPS estimates via yfinance."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return None
        return cal
    except Exception:
        return None


def get_options_chain(ticker: str, expiry: Optional[str] = None) -> dict:
    """
    Return {calls: DataFrame, puts: DataFrame} for the nearest expiry (or specified).
    """
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {}
        target = expiry or expiries[0]
        chain = t.option_chain(target)
        return {"calls": chain.calls, "puts": chain.puts, "expiry": target}
    except Exception as exc:
        logger.debug("Options chain fetch failed for %s: %s", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# Alpaca live bars (paper or live)
# ---------------------------------------------------------------------------

def get_alpaca_bars(
    tickers: list[str],
    days: int = 5,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """
    Fetch recent bars from Alpaca REST API.
    Falls back to yfinance if Alpaca creds not set.
    """
    from config import settings
    if not settings.ALPACA_API_KEY:
        logger.debug("Alpaca creds not set, falling back to yfinance")
        return get_close(tickers, days=max(days, 10))

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(
            settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
        )
        tf = TimeFrame(1, TimeFrameUnit.Day) if timeframe == "1Day" else TimeFrame(1, TimeFrameUnit.Hour)
        end = datetime.now()
        start = end - timedelta(days=days + 5)
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame()
        # Pivot to wide format: columns = tickers
        close = bars["close"].unstack(level=0)
        return close
    except Exception as exc:
        logger.warning("Alpaca bars failed: %s — falling back to yfinance", exc)
        return get_close(tickers, days=max(days, 10))
