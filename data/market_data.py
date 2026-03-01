"""
Market data layer.

Primary source: yfinance (free, no key needed)
Supplementary: Alpaca REST API for intraday / live bars

All data returned as pandas DataFrames with DatetimeIndex.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BATCH_RETRIES = 3
_TICKER_RETRIES = 3


def _download_single(ticker: str, kwargs: dict) -> pd.DataFrame:
    """Download one ticker and normalize to simple (non-MultiIndex) columns."""
    df = yf.download([ticker], **{**kwargs, "threads": False})
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        if ticker in lvl1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            df.columns = df.columns.droplevel(1)
    return df


def _batch_with_fallback(tickers: list[str], kwargs: dict) -> pd.DataFrame:
    """
    Try a bulk yf.download with retries. If the batch fails or returns empty,
    fall back to per-ticker downloads so one bad ticker cannot kill the rest.
    """
    # --- batch attempts ---
    for attempt in range(_BATCH_RETRIES):
        try:
            df = yf.download(tickers, **kwargs)
            if not df.empty:
                return df
            logger.warning("yfinance batch returned empty (attempt %d/%d)", attempt + 1, _BATCH_RETRIES)
        except Exception as exc:
            logger.warning("yfinance batch attempt %d/%d failed: %s", attempt + 1, _BATCH_RETRIES, exc)
        if attempt < _BATCH_RETRIES - 1:
            time.sleep(2 ** attempt)   # 1 s, 2 s

    # --- per-ticker fallback ---
    logger.warning("Batch download failed — falling back to per-ticker for %d tickers", len(tickers))
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        for attempt in range(_TICKER_RETRIES):
            try:
                df = _download_single(ticker, kwargs)
                if not df.empty:
                    frames[ticker] = df
                    break
            except Exception as exc:
                logger.warning("yfinance %s attempt %d/%d: %s", ticker, attempt + 1, _TICKER_RETRIES, exc)
            if attempt < _TICKER_RETRIES - 1:
                time.sleep(1)
        else:
            logger.error("yfinance: all retries failed for %s — skipping", ticker)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return list(frames.values())[0]

    # Rebuild MultiIndex (field, ticker) DataFrame from per-ticker simple DataFrames
    combined: dict[tuple, pd.Series] = {}
    for ticker, df in frames.items():
        for col in df.columns:
            combined[(col, ticker)] = df[col]
    result = pd.DataFrame(combined)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


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

    kwargs = dict(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    df = _batch_with_fallback(tickers, kwargs)
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


def get_vix9d(days: int = 365) -> pd.Series:
    """Return daily VIX9D closing values (9-day expected volatility)."""
    df = get_close(["^VIX9D"], days=days)
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rename("VIX9D")


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
    for attempt in range(_TICKER_RETRIES):
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or cal.empty:
                return None
            return cal
        except Exception as exc:
            logger.warning("get_earnings_calendar %s attempt %d/%d: %s", ticker, attempt + 1, _TICKER_RETRIES, exc)
            if attempt < _TICKER_RETRIES - 1:
                time.sleep(1)
    return None


def get_options_chain(ticker: str, expiry: Optional[str] = None) -> dict:
    """
    Return {calls: DataFrame, puts: DataFrame} for the nearest expiry (or specified).
    """
    for attempt in range(_TICKER_RETRIES):
        try:
            t = yf.Ticker(ticker)
            expiries = t.options
            if not expiries:
                return {}
            target = expiry or expiries[0]
            chain = t.option_chain(target)
            return {"calls": chain.calls, "puts": chain.puts, "expiry": target}
        except Exception as exc:
            logger.debug("Options chain fetch failed for %s attempt %d/%d: %s", ticker, attempt + 1, _TICKER_RETRIES, exc)
            if attempt < _TICKER_RETRIES - 1:
                time.sleep(1)
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
