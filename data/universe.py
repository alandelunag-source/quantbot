"""
Ticker universes used by each strategy.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def _wiki_read_html(url: str) -> list:
    """Fetch a Wikipedia page and parse HTML tables, using a browser User-Agent."""
    import io
    import requests
    import pandas as pd
    headers = {"User-Agent": "Mozilla/5.0 (compatible; quantbot/1.0; +research)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text), header=0)


@functools.lru_cache(maxsize=1)
def get_sp500() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia (cached per session)."""
    try:
        tbl = _wiki_read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers = tbl["Symbol"].str.replace(".", "-", regex=False).dropna().tolist()
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return sorted(tickers)
    except Exception as exc:
        logger.warning("S&P 500 Wikipedia fetch failed (%s) — falling back to SP100", exc)
        return list(SP100)


@functools.lru_cache(maxsize=1)
def get_nasdaq100() -> list[str]:
    """Fetch current NASDAQ-100 constituents from Wikipedia (cached per session)."""
    try:
        tables = _wiki_read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for tbl in tables:
            for col in ("Ticker", "Symbol", "Ticker symbol"):
                if col in tbl.columns:
                    tickers = (tbl[col].str.replace(".", "-", regex=False)
                               .dropna().tolist())
                    tickers = [t for t in tickers if isinstance(t, str) and len(t) <= 6]
                    logger.info("Fetched %d NASDAQ-100 tickers from Wikipedia", len(tickers))
                    return sorted(tickers)
    except Exception as exc:
        logger.warning("NASDAQ-100 Wikipedia fetch failed (%s) — falling back to SP100", exc)
    return list(SP100)


@functools.lru_cache(maxsize=1)
def get_large_cap_universe() -> list[str]:
    """Union of S&P 500 + NASDAQ-100, deduplicated and sorted (~550 tickers)."""
    return sorted(set(get_sp500()) | set(get_nasdaq100()))

SP100 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","JPM","JNJ",
    "XOM","V","UNH","PG","MA","HD","CVX","LLY","ABBV","MRK","PEP","AVGO",
    "KO","COST","MCD","TMO","ACN","CSCO","CRM","BAC","ABT","NFLX","WMT",
    "DHR","LIN","NEE","TXN","PM","UPS","AMGN","RTX","QCOM","HON","LOW",
    "IBM","GE","DE","CAT","SBUX","GILD","BMY","MDT","SPGI","BLK","AXP",
    "ISRG","C","GS","NOW","BKNG","ELV","MO","TJX","ZTS","VRTX","PLD",
    "SYK","CB","REGN","SO","CL","CI","DUK","MMC","BSX","ITW","ADI","MDLZ",
    "APD","EQIX","CME","TGT","USB","MMM","NSC","PNC","FDX","AON","KLAC",
    "EOG","MPC","PSA","SLB","ORLY","AFL","AIG","AEP","WM","EW",
]

ETF_UNIVERSE = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "GLD",   # Gold
    "TLT",   # 20yr Treasuries
    "DBC",   # Commodities
    "VNQ",   # REITs
    "EFA",   # Developed ex-US
    "IEMG",  # EM
    "SHY",   # 1-3yr Treasuries (safe haven / cash proxy)
]

MACRO_TICKERS = [
    "SPY",   # equities
    "TLT",   # long bonds
    "GLD",   # gold
    "SHY",   # short bonds / cash
    "QQQ",   # growth equities
    "HYG",   # high yield credit
    "LQD",   # investment grade credit
    "UUP",   # US dollar
    "FXE",   # Euro
    "DBC",   # commodities
]

VOL_TICKERS = [
    "VXX",   # VIX short-term futures ETN
    "UVXY",  # 1.5x VIX
    "SVXY",  # -0.5x VIX
]

# VIX proxies (yfinance)
VIX_TICKER = "^VIX"
VIX9D_TICKER = "^VIX9D"
TNX_TICKER = "^TNX"   # 10Y yield
IRX_TICKER = "^IRX"   # 3M yield (proxy for 2Y)
