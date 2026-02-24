"""
Ticker universes used by each strategy.
"""
from __future__ import annotations

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
