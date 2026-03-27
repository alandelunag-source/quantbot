"""Retroactive March 26 session runner — fetches 1-min intraday, runs open/midday/close."""
import json, sys, os, csv
os.chdir(r'C:\Users\aland\quantbot')
sys.path.insert(0, r'C:\Users\aland\quantbot')

import pandas as pd
import yfinance as yf

import backtesting.forward_test as ftest_mod
ftest_mod.is_trading_day = lambda dt=None: True

from datetime import datetime as _real_datetime
class _FakeDateTime(_real_datetime):
    @classmethod
    def today(cls):
        return cls(2026, 3, 26)
import backtesting.forward_test
backtesting.forward_test.datetime = _FakeDateTime

from strategies import ALL_STRATEGIES
from backtesting.forward_test import ForwardTest

# ─── Ticker universe ──────────────────────────────────────────────────────────
TICKERS = sorted(set([
    # Currently held (from state files as of March 25 close)
    "C","CIEN","DBC","EA","FE","FIX","GLD","GOOG","GOOGL","HOLX","IEMG","IWM",
    "JNJ","L","LRCX","MSFT","MU","NEM","SHY","SNDK","SPY","STX","TER","UUP",
    "V","WBD","WDC",
    # Likely new signals
    "AME","AXP","CCL","COO","FCX","FDX","GEV","GLW","HWM","IFF","KR","LII","LUV",
    "MCHP","NRG","NTAP","PG","PH","PPG","PRU","PWR","QQQ","RTX","TFC","TLT",
    "TSLA","UAL","UHS","WFC","XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","VXX",
]))

PRICE_JSON = 'state/intraday_prices_2026-03-26.json'
DATE_STR   = '2026-03-26'

def price_at(df: pd.DataFrame, hour: int, minute: int) -> dict:
    mask = (df.index.hour < hour) | ((df.index.hour == hour) & (df.index.minute <= minute))
    subset = df[mask]
    if subset.empty:
        return {}
    row = subset.iloc[-1]
    return {col: round(float(row[col]), 4) for col in df.columns if pd.notna(row[col])}

# ─── Fetch intraday data ──────────────────────────────────────────────────────
if not os.path.exists(PRICE_JSON):
    print(f"Fetching 1-min intraday data for {DATE_STR} ...")
    sys.stdout.flush()
    raw = yf.download(TICKERS, start=DATE_STR, end='2026-03-27',
                      interval='1m', progress=False, auto_adjust=True)
    if raw.empty:
        print("ERROR: No intraday data returned from yfinance")
        sys.exit(1)
    close_data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']].rename(columns={'Close': TICKERS[0]})
    close_data.index = close_data.index.tz_convert('America/New_York')
    prices_open   = price_at(close_data, 9,  35)
    prices_midday = price_at(close_data, 12, 30)
    prices_close  = price_at(close_data, 15, 45)
    intraday = {'open': prices_open, 'midday': prices_midday, 'close': prices_close}
    with open(PRICE_JSON, 'w') as f:
        json.dump(intraday, f, indent=2)
    print(f"Saved {PRICE_JSON}  (open={len(prices_open)}, midday={len(prices_midday)}, close={len(prices_close)} tickers)")
else:
    print(f"Loading existing {PRICE_JSON}")

with open(PRICE_JSON) as f:
    intraday = json.load(f)

# ─── Run all 3 sessions ───────────────────────────────────────────────────────
STRATS   = [s for s in ALL_STRATEGIES if s not in ('s04','s06','s08','s16','s17')]
SESSIONS = [('open', intraday['open']), ('midday', intraday['midday']), ('close', intraday['close'])]

print(f"\nRunning 3 sessions for {len(STRATS)} strategies on {DATE_STR}")
sys.stdout.flush()

for session_name, session_prices in SESSIONS:
    print(f"\n=== SESSION: {session_name.upper()} ({DATE_STR}) ===")
    sys.stdout.flush()
    for sid in STRATS:
        ft = ForwardTest(ALL_STRATEGIES[sid]())
        prices = dict(session_prices)
        for ticker, pos in ft._state['positions'].items():
            if ticker not in prices:
                prices[ticker] = pos.get('entry_price', 0)
        try:
            summary = ft.update(prices, session=session_name)
            pv    = summary.get('portfolio_value', 0)
            ret   = (pv / 100_000 - 1) * 100
            n_pos = len(summary.get('positions', {}))
            print(f"  {sid:<28} ${pv:>10,.2f}  ({ret:+.2f}%)  [{n_pos} pos]")
        except Exception as e:
            print(f"  {sid:<28} ERROR: {e}")
            import traceback; traceback.print_exc()
        sys.stdout.flush()

# ─── Re-read final close values ───────────────────────────────────────────────
results = {}
for sid in STRATS:
    ft = ForwardTest(ALL_STRATEGIES[sid]())
    results[sid] = ft._state.get('portfolio_value', 0)

# ─── Append to paper_log.csv ──────────────────────────────────────────────────
with open('state/paper_log.csv', 'a', newline='') as f:
    w = csv.writer(f)
    for sid in STRATS:
        w.writerow([DATE_STR, sid, round(results[sid], 2)])

total = sum(results.values())
basis = len(STRATS) * 100_000
print(f"\nAppended {len(STRATS)} rows to paper_log.csv for {DATE_STR}")
print(f"Total portfolio: ${total:,.2f}  ({(total/basis - 1)*100:+.2f}% vs ${basis/1e6:.1f}M basis)")

# ─── Trade summary (reads from state files) ───────────────────────────────────
print(f"\n{'='*60}")
print(f"TRADE SUMMARY -- {DATE_STR}")
print(f"{'='*60}")
import glob as _glob
any_trade = False
for path in sorted(_glob.glob('state/*_state.json')):
    if 's17' in path: continue
    with open(path) as f:
        state = json.load(f)
    sid = os.path.basename(path).replace('_state.json', '')
    day_trades = [t for t in state.get('trades', []) if t.get('date') == DATE_STR]
    if day_trades:
        any_trade = True
        print(f"\n  {sid}:")
        for t in day_trades:
            val  = t.get('dollar_value', 0)
            note = f"  [{t['note']}]" if t.get('note') else ''
            print(f"    {t['action']:<8} {t['ticker']:<6}  ${val:,.0f}{note}")
if not any_trade:
    print("\n  No trades executed on March 26.")
