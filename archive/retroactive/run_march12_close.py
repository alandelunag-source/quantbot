"""March 12 close session — 15:45 ET prices. Full update + rebalance + daily log."""
import json, sys, os
os.chdir(r'C:\Users\aland\quantbot')
sys.path.insert(0, r'C:\Users\aland\quantbot')

import backtesting.forward_test as ftest_mod
ftest_mod.is_trading_day = lambda dt=None: True

from strategies import ALL_STRATEGIES
from backtesting.forward_test import ForwardTest

PRICE_JSON = 'state/intraday_prices_2026-03-12.json'

with open(PRICE_JSON) as f:
    intraday = json.load(f)

STRATS = [s for s in ALL_STRATEGIES if s not in ('s04','s06','s08','s16')]

print(f"Running CLOSE session for {len(STRATS)} strategies on 2026-03-12")
print(f"(15:45 ET prices: {len(intraday['close'])} tickers)")
sys.stdout.flush()

session_prices = intraday['close']

for sid in STRATS:
    cls = ALL_STRATEGIES[sid]
    ft  = ForwardTest(cls())
    prices = dict(session_prices)
    for ticker, pos in ft._state['positions'].items():
        if ticker not in prices:
            prices[ticker] = pos.get('entry_price', 0)
    try:
        summary = ft.update(prices, session='close')
        pv    = summary.get('portfolio_value', 0)
        ret   = (pv / 100_000 - 1) * 100
        n_pos = len(summary.get('positions', {}))
        trades = summary.get('trades', [])
        recent = [t for t in trades if t.get('date') == '2026-03-12']
        trade_str = f"  {len(recent)} trade(s): " + ", ".join(
            f"{t['action']} {t['ticker']}" for t in recent[:4]
        ) if recent else ""
        print(f"  {sid:<28} ${pv:>10,.2f}  ({ret:+.2f}%)  [{n_pos} pos]{trade_str}")
    except Exception as e:
        print(f"  {sid:<28} ERROR: {e}")
    sys.stdout.flush()

print("\nClose session complete.")
