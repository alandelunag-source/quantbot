"""
Retroactive March 9 session runner.
Runs open (9:35), midday (12:30), close (3:45) with fetched intraday prices.
Today IS 2026-03-09 on the machine, so no datetime patching needed.
"""
import json, sys, os, csv
os.chdir(r'C:\Users\aland\quantbot')
sys.path.insert(0, r'C:\Users\aland\quantbot')

import backtesting.forward_test as ftest_mod
ftest_mod.is_trading_day = lambda dt=None: True

from strategies import ALL_STRATEGIES
from backtesting.forward_test import ForwardTest

with open('state/intraday_prices_2026-03-09.json') as f:
    intraday = json.load(f)

SESSIONS = [('open', intraday['open']), ('midday', intraday['midday']), ('close', intraday['close'])]
STRATS   = [s for s in ALL_STRATEGIES if s not in ('s04','s06','s08','s16')]

print(f"Running 3 sessions for {len(STRATS)} strategies on 2026-03-09")
sys.stdout.flush()

results = {}
for session_name, session_prices in SESSIONS:
    print(f"\n=== SESSION: {session_name.upper()} ===")
    sys.stdout.flush()
    for sid in STRATS:
        cls  = ALL_STRATEGIES[sid]
        ft   = ForwardTest(cls())
        prices = dict(session_prices)
        for ticker, pos in ft._state['positions'].items():
            if ticker not in prices:
                prices[ticker] = pos.get('entry_price', 0)
        try:
            summary = ft.update(prices, session=session_name)
            pv      = summary.get('portfolio_value', 0)
            ret     = (pv / 100_000 - 1) * 100
            n_pos   = len(summary.get('positions', {}))
            print(f"  {sid:<28} ${pv:>10,.2f}  ({ret:+.2f}%)  [{n_pos} pos]")
        except Exception as e:
            print(f"  {sid:<28} ERROR: {e}")
        sys.stdout.flush()
        if session_name == 'close':
            results[sid] = summary.get('portfolio_value', 0)

# Append close values to paper_log
with open('state/paper_log.csv', 'a', newline='') as f:
    w = csv.writer(f)
    for sid in STRATS:
        w.writerow(['2026-03-09', sid, round(results[sid], 2)])

total = sum(results.values())
print(f"\nAppended {len(STRATS)} rows to paper_log.csv")
print(f"Total portfolio: ${total:,.2f}  ({(total/1_300_000-1)*100:+.2f}% since inception)")
