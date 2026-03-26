"""Retroactive March 12 open session — fetches 9:35 ET intraday prices then runs open session."""
import json, sys, os
os.chdir(r'C:\Users\aland\quantbot')
sys.path.insert(0, r'C:\Users\aland\quantbot')

import pandas as pd
import yfinance as yf

import backtesting.forward_test as ftest_mod
ftest_mod.is_trading_day = lambda dt=None: True

from strategies import ALL_STRATEGIES
from backtesting.forward_test import ForwardTest

# ─── Ticker universe (current positions + common strategy universe) ───────────
TICKERS = [
    # Currently held (from state files)
    "AME","AXP","C","CIEN","COO","DBC","EA","FIX","GEV","GLD","GOOG","GOOGL",
    "HOLX","HWM","IEMG","IFF","IWM","JNJ","KR","LII","LRCX","MSFT","MU",
    "NEM","NTAP","PPG","PRU","PWR","SHY","SNDK","SPY","STX","TER","UAL",
    "UUP","WBD","WDC",
    # Likely new signals
    "CCL","FCX","FDX","FE","GLW","L","LUV","MCHP","NRG","PG","PH","QQQ",
    "RTX","TFC","TLT","TSLA","UHS","V","WFC","XLB","XLE","XLF","XLI","XLK",
    "XLP","XLU","XLV","XLY","VXX",
]

PRICE_JSON = 'state/intraday_prices_2026-03-12.json'

def price_at(df: pd.DataFrame, hour: int, minute: int) -> dict:
    """Return last 1-min close at or before HH:MM ET as {ticker: price}."""
    mask = (df.index.hour < hour) | ((df.index.hour == hour) & (df.index.minute <= minute))
    subset = df[mask]
    if subset.empty:
        return {}
    row = subset.iloc[-1]
    return {col: round(float(row[col]), 4) for col in df.columns if pd.notna(row[col])}

# ─── Fetch intraday data ──────────────────────────────────────────────────────
if not os.path.exists(PRICE_JSON):
    print("Fetching 1-min intraday data for 2026-03-12 ...")
    sys.stdout.flush()
    raw = yf.download(TICKERS, period="1d", interval="1m", progress=False, auto_adjust=True)
    if raw.empty:
        print("ERROR: No intraday data returned from yfinance")
        sys.exit(1)

    if isinstance(raw.columns, pd.MultiIndex):
        close_data = raw["Close"]
    else:
        close_data = raw[["Close"]].rename(columns={"Close": TICKERS[0]})

    close_data.index = close_data.index.tz_convert("America/New_York")

    prices_open   = price_at(close_data, 9,  35)
    prices_midday = price_at(close_data, 12, 30)
    prices_close  = price_at(close_data, 15, 45)

    intraday = {"open": prices_open, "midday": prices_midday, "close": prices_close}
    with open(PRICE_JSON, 'w') as f:
        json.dump(intraday, f, indent=2)
    print(f"Saved {PRICE_JSON}  (open={len(prices_open)}, midday={len(prices_midday)}, close={len(prices_close)} tickers)")
else:
    print(f"Loading existing {PRICE_JSON}")

with open(PRICE_JSON) as f:
    intraday = json.load(f)

# ─── Run open session only ────────────────────────────────────────────────────
STRATS = [s for s in ALL_STRATEGIES if s not in ('s04','s06','s08','s16')]

print(f"\nRunning OPEN session for {len(STRATS)} strategies on 2026-03-12")
print(f"(9:35 ET prices: {len(intraday['open'])} tickers)")
sys.stdout.flush()

session_prices = intraday['open']

for sid in STRATS:
    cls = ALL_STRATEGIES[sid]
    ft  = ForwardTest(cls())
    prices = dict(session_prices)
    for ticker, pos in ft._state['positions'].items():
        if ticker not in prices:
            prices[ticker] = pos.get('entry_price', 0)
    try:
        summary = ft.update(prices, session='open')
        pv    = summary.get('portfolio_value', 0)
        ret   = (pv / 100_000 - 1) * 100
        n_pos = len(summary.get('positions', {}))
        trades = summary.get('trades', [])
        recent = [t for t in trades if t.get('date') == '2026-03-12']
        trade_str = f"  {len(recent)} trade(s): " + ", ".join(
            f"{t['action']} {t['ticker']}" for t in recent[:3]
        ) if recent else ""
        print(f"  {sid:<28} ${pv:>10,.2f}  ({ret:+.2f}%)  [{n_pos} pos]{trade_str}")
    except Exception as e:
        print(f"  {sid:<28} ERROR: {e}")
    sys.stdout.flush()

print("\nOpen session complete. Run midday at 12:30 ET and close at 15:45 ET.")
