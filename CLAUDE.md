# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Tests
venv\Scripts\python.exe -m pytest tests/ -q
venv\Scripts\python.exe -m pytest tests/test_strategies.py::TestMomentumDip -q  # single class

# Paper trading (all active strategies, 3 sessions)
venv\Scripts\python.exe main.py paper --strategies s01,s02,s03,s05,s07,s09,s10,s11,s12,s13,s14,s15,s17,s19 --once
venv\Scripts\python.exe main.py paper --strategies s01,... --once --session open   # open/midday/close

# Scan signals
venv\Scripts\python.exe main.py scan

# Backtest a strategy
venv\Scripts\python.exe main.py backtest --strategy s01

# Dashboard
venv\Scripts\streamlit.exe run dashboard.py

# Retroactive session simulation (when sessions missed)
# See run_marchN.py pattern — fetches 1-min intraday, extracts 9:35/12:30/15:45 ET prices
powershell.exe -ExecutionPolicy Bypass -File run_marchN.ps1
```

**Never run tests automatically** — user runs them manually at end of day.

## Architecture

### Strategy lifecycle

Every strategy inherits from `strategies/base.py:Strategy` and implements three methods:

```python
def get_universe() -> list[str]          # tickers to consider
def generate_signals(prices, **kwargs)   # DataFrame: index=dates, cols=tickers, values=scores
def position_sizing(signals, prices)     # dict: {ticker: weight}  (weights sum ≤ MAX_DEPLOY=0.95)
def exit_rules(entry_price, current_price, days_held) -> bool  # stop-loss / profit target / time
```

Key class attributes: `rebalance_freq` ("daily"/"weekly"/"monthly"), `max_positions`, `PROFIT_LOCK` (reset cost basis at threshold, used by S02/S09 at 5%), `ENTRIES_ONLY` (S05/S17 — only add, never trim; exits via `exit_rules` only).

### Paper trading state machine (`backtesting/forward_test.py`)

`ForwardTest(strategy_instance)` loads/saves JSON from `state/{name}_state.json`. The `update(prices, session)` call:

1. Recomputes `portfolio_value` from `cash + Σ(shares × price)` — **this overwrites the stored value**
2. Fires exit rules and profit lock on **every session** (open/midday/close)
3. Runs signal generation + rebalance on **close only**
4. Logs to `daily_log` on **close only**
5. Saves state

**Critical**: `portfolio_value` stored in state = pre-rebalance value (set at step 1, not updated after rebalance). The actual live value is always `cash + invested`. Negative target weights are **not supported** — `_rebalance` pops the position and adds phantom cash. Keep all weights ≥ 0.

### Session model

- **Open (9:35 ET) / Midday (12:30 ET)**: exit rules only, no rebalance, no daily log
- **Close (15:45 ET)**: full update — signals, rebalance, `daily_log` entry appended

When sessions are missed, simulate retroactively: fetch 1-min yfinance data, extract prices at the three session times, run sessions in order. See `run_march10.py` as the canonical pattern.

### Active strategies

`strategies/__init__.py:ALL_STRATEGIES` — 14 active keys: `s01,s02,s03,s05,s07,s09,s10,s11,s12,s13,s14,s15,s17,s19`. Excluded from paper trading: `s04,s06,s08,s16` (cut or overlay).

### Data layer

`data/market_data.py` — all market data via yfinance (primary) with Alpaca fallback. Key functions: `get_close()`, `get_volume()`, `get_vix()`, `get_yield_spread()`. Batch download with 3-retry + per-ticker fallback.

`data/indicators.py` — stateless vectorized indicators (`sma`, `ema`, `rsi`, `realized_vol`, `z_score`, `ic`, etc.).

`data/universe.py` — `ETF_UNIVERSE`, `SP100`, `get_sp500()`, `get_large_cap_universe()`.

### State files (`state/`)

- `{strategy}_state.json` — persistent positions, cash, trades, daily_log
- `paper_log.csv` — daily close values per strategy (date, sid, value)
- `paper_trading.log` — append-only trading log (FileHandler, not batch redirect)
- `intraday_prices_YYYY-MM-DD.json` — `{open: {ticker: price}, midday: {...}, close: {...}}`

### Config

`config/settings.py` — all settings env-overridable via `.env`. Key values: `INITIAL_CAPITAL=100_000`, `DRAWDOWN_STOP_PCT=0.12`, `TOTAL_COST_PCT=0.0004` (4 bps round-trip).

## Known gotchas

- **Windows execution**: use `powershell.exe -ExecutionPolicy Bypass -File script.ps1` — direct bash invocation of `.exe` paths gives exit 127
- **ForwardTest takes instances**, not classes: `ForwardTest(MyStrategy())` not `ForwardTest(MyStrategy)`
- **yfinance rate limits**: occasional `YFRateLimitError`; `RE` ticker is delisted — skip safely
- **S10 short positions**: `_rebalance` does not support negative weights — any `target_w ≤ 0` removes the position and adds phantom cash. Fixed in S10 (VXX short replaced with SHY 5%)
- **portfolio_value in state** is stale after rebalance — recompute as `cash + Σ(shares × price)` for accurate current value
