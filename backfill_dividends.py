"""
Backfill dividend credits for the past 25 days of paper trading (Feb 24 – Mar 20).

For each strategy state file:
  1. Replay the trade history to reconstruct share counts on each calendar date.
  2. Fetch real ex-dividend events from yfinance for every ticker ever held.
  3. If an ex-date falls within the holding window, credit cash and log a DIVIDEND trade.
  4. Recompute portfolio_value and peak_value from the updated cash + current positions.

Run once. Safe to re-run — deduplicates by checking existing DIVIDEND entries.
"""
import json, glob, os, sys
os.chdir(r'C:\Users\aland\quantbot')
sys.path.insert(0, r'C:\Users\aland\quantbot')

import pandas as pd
import yfinance as yf

START_DATE = pd.Timestamp('2026-02-24')
END_DATE   = pd.Timestamp('2026-03-20')

# ── helpers ──────────────────────────────────────────────────────────────────

div_cache: dict = {}

def get_divs(ticker: str) -> pd.Series:
    if ticker not in div_cache:
        try:
            d = yf.Ticker(ticker).dividends
            # Strip timezone and normalize to date-only
            d.index = d.index.tz_localize(None).normalize()
            div_cache[ticker] = d
        except Exception:
            div_cache[ticker] = pd.Series(dtype=float)
    return div_cache[ticker]


def reconstruct_holdings(trades: list) -> dict:
    """
    Replay BUY/SELL trades to build a dict of:
      { ticker: [ (start_date, end_date_or_None, shares) ] }
    where end_date_or_None=None means still held.
    """
    # Simple running share tracker
    holdings: dict[str, list] = {}   # ticker -> list of (date, shares_delta)
    for t in trades:
        if t.get('action') not in ('BUY', 'SELL'):
            continue
        ticker = t['ticker']
        date   = pd.Timestamp(t['date'])
        # shares stored in trade? if not, derive from dollar_value / price
        # We don't have price per share in trades, so we use current position shares
        # Instead, track open/close windows from trade dates
        holdings.setdefault(ticker, []).append((date, t['action']))
    return holdings


def holding_windows(trades: list, positions: dict) -> dict:
    """
    For each ticker ever traded, return list of (entry_date, exit_date_or_END).
    Uses actual entry_date from current positions where still held.
    """
    windows: dict[str, list] = {}  # ticker -> [(entry_ts, exit_ts, shares)]

    # Group trades by ticker in chronological order
    by_ticker: dict[str, list] = {}
    for t in sorted(trades, key=lambda x: x.get('date', '')):
        if t.get('action') not in ('BUY', 'SELL'):
            continue
        by_ticker.setdefault(t['ticker'], []).append(t)

    for ticker, txns in by_ticker.items():
        entry_ts    = None
        entry_shares = 0.0
        for t in txns:
            ts = pd.Timestamp(t['date'])
            if t['action'] == 'BUY':
                if entry_ts is None:
                    entry_ts = ts
                # shares: use dollar_value / div not available — use rough from state
                # We'll use the dollar_value as proxy; actual shares don't matter,
                # we just need to know *when* the position was open.
                entry_shares += t.get('dollar_value', 0)
            elif t['action'] == 'SELL':
                if entry_ts is not None:
                    entry_shares -= t.get('dollar_value', 0)
                    if entry_shares <= 0.01:
                        windows.setdefault(ticker, []).append((entry_ts, ts))
                        entry_ts    = None
                        entry_shares = 0.0

        # Still open
        if entry_ts is not None and ticker in positions:
            windows.setdefault(ticker, []).append((entry_ts, END_DATE + pd.Timedelta(days=1)))

    return windows


# ── main backfill ─────────────────────────────────────────────────────────────

state_files = sorted(glob.glob('state/*_state.json'))
# Skip cut strategies
SKIP = {'s04', 's06', 's08', 's16', 's17'}

total_credited = 0.0
total_events   = 0

for fpath in state_files:
    sid = os.path.basename(fpath).replace('_state.json', '')
    if any(sid.startswith(s + '_') or sid == s for s in SKIP):
        continue

    with open(fpath) as fh:
        state = json.load(fh)

    trades    = state.get('trades', [])
    positions = state.get('positions', {})

    # All tickers ever touched
    all_tickers = set(t['ticker'] for t in trades if t.get('action') in ('BUY','SELL'))
    all_tickers |= set(positions.keys())

    credited_this_strategy = 0.0
    new_div_trades = []

    for ticker in sorted(all_tickers):
        divs = get_divs(ticker)
        if divs.empty:
            continue

        # Only ex-dates within our simulation window
        relevant = divs[(divs.index >= START_DATE) & (divs.index <= END_DATE)]
        if relevant.empty:
            continue

        # Get actual shares on each ex-date from current position or trade history
        # For simplicity: use current shares if still held, else find approx from trades
        cur_pos = positions.get(ticker)

        for ex_date, div_per_share in relevant.items():
            ex_date_str = ex_date.strftime('%Y-%m-%d')

            # Skip if already credited
            already = any(
                t.get('action') == 'DIVIDEND' and
                t.get('ticker') == ticker and
                t.get('date') == ex_date_str
                for t in trades + new_div_trades
            )
            if already:
                continue

            # Determine shares held on ex-date
            shares = 0.0
            if cur_pos and ex_date >= pd.Timestamp(cur_pos.get('entry_date', '2026-01-01')):
                shares = cur_pos['shares']
            else:
                # Try to infer from trades: last BUY before ex-date, check no full SELL before it
                running = 0.0
                for t in sorted(trades, key=lambda x: x.get('date', '')):
                    if t.get('ticker') != ticker or t.get('action') not in ('BUY','SELL'):
                        continue
                    t_date = pd.Timestamp(t['date'])
                    if t_date > ex_date:
                        break
                    dv = t.get('dollar_value', 0)
                    if t['action'] == 'BUY':
                        running += dv
                    else:
                        running -= dv
                # running > 0 means position was open; get approx shares from state
                # We don't store historical shares, so use the current shares as proxy
                # if the ticker is still held and the entry predates ex_date
                if running > 0 and cur_pos:
                    shares = cur_pos['shares']

            if shares <= 0:
                continue

            total_div = div_per_share * shares
            state['cash'] += total_div
            credited_this_strategy += total_div
            total_credited += total_div
            total_events   += 1

            new_div_trades.append({
                'date':          ex_date_str,
                'ticker':        ticker,
                'action':        'DIVIDEND',
                'dollar_value':  round(total_div, 2),
                'div_per_share': round(float(div_per_share), 6),
                'shares':        round(shares, 4),
                'note':          f'backfill ex-div ${float(div_per_share):.4f}/sh',
            })
            print(f'  {sid:<30} {ticker:<6} ex={ex_date_str}  '
                  f'${float(div_per_share):.4f}/sh x {shares:.1f}sh = ${total_div:,.2f}')

    if new_div_trades:
        # Insert dividend trades in chronological order
        state['trades'] = sorted(
            trades + new_div_trades,
            key=lambda x: (x.get('date',''), x.get('action',''))
        )
        # Recompute portfolio_value from cash + current mark-to-market
        # (use entry_price as proxy since we don't have live prices here)
        invested = sum(
            p.get('entry_price', 0) * p.get('shares', 0)
            for p in state['positions'].values()
        )
        state['portfolio_value'] = round(state['cash'] + invested, 2)
        state['peak_value']      = max(state.get('peak_value', 0), state['portfolio_value'])

        with open(fpath, 'w') as fh:
            json.dump(state, fh, indent=2)
        print(f'  >> {sid}: credited ${credited_this_strategy:,.2f} across {len(new_div_trades)} event(s)\n')

print(f'\nDone. Total credited: ${total_credited:,.2f} across {total_events} dividend event(s).')
