"""
Simulate paper trading for missed days using historical price data.

Replays Feb 25, 26, 27 as if the scheduler had been running.
- Fetches price data capped to each simulated date
- Monkeypatches datetime.today() so ForwardTest uses the right date
- Runs full signal generation + rebalancing for each strategy each day
- Replaces the price-only backfill entries in daily_log with real ones
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

SIMULATE_DATES = ["2026-02-25", "2026-02-26", "2026-02-27"]

STRATEGY_CODES = [
    "s01", "s02", "s03", "s04", "s05", "s06", "s07",
    "s09", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17",
]

STATE_DIR = Path("state")


# ---------------------------------------------------------------------------
# Historical price fetcher — capped to a specific date
# ---------------------------------------------------------------------------

_price_cache: dict[str, pd.DataFrame] = {}

def _fetch_capped(tickers: list[str], days: int, as_of: datetime) -> pd.DataFrame:
    """Fetch up to `days` of OHLCV data ending on `as_of` date."""
    key = f"{','.join(sorted(tickers))}|{days}|{as_of.date()}"
    if key in _price_cache:
        return _price_cache[key]

    start = as_of - timedelta(days=days + 30)
    end   = as_of + timedelta(days=1)  # yfinance end is exclusive

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"  WARN: batch download failed ({exc}), falling back to per-ticker")
        frames = {}
        for t in tickers:
            try:
                df = yf.download([t], start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False, threads=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    frames[t] = df
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            raw = list(frames.values())[0]
        else:
            combined = {}
            for t, df in frames.items():
                for col in df.columns:
                    combined[(col, t)] = df[col]
            raw = pd.DataFrame(combined)
            raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    _price_cache[key] = raw
    return raw


def _get_close_capped(tickers: list[str], days: int, as_of: datetime) -> pd.DataFrame:
    raw = _fetch_capped(tickers, days, as_of)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].dropna(how="all")
    return raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna()


def _get_volume_capped(tickers: list[str], days: int, as_of: datetime) -> pd.DataFrame:
    raw = _fetch_capped(tickers, days, as_of)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Volume"].dropna(how="all")
    return raw[["Volume"]].rename(columns={"Volume": tickers[0]}).dropna()


def _get_spot_prices(tickers: list[str], as_of: datetime) -> dict[str, float]:
    """Get the closing price on `as_of` date for each ticker."""
    df = _get_close_capped(tickers, 10, as_of)
    if df.empty:
        return {}
    date_str = as_of.strftime("%Y-%m-%d")
    df.index = pd.to_datetime(df.index)
    row = df[df.index.strftime("%Y-%m-%d") <= date_str].iloc[-1] if not df.empty else pd.Series()
    return {t: float(row[t]) for t in df.columns if t in row.index and not pd.isna(row[t])}


# ---------------------------------------------------------------------------
# Strip price-only backfill entries so simulate can overwrite
# ---------------------------------------------------------------------------

def _strip_backfill_entries(state: dict, dates: list[str]) -> dict:
    """Remove daily_log entries for the given dates (marked as 'backfill')."""
    state["daily_log"] = [
        e for e in state["daily_log"]
        if not (e["date"] in dates and e.get("note") == "backfill")
    ]
    return state


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate():
    from strategies import ALL_STRATEGIES
    from backtesting.forward_test import ForwardTest
    from config import settings

    print("\n" + "="*60)
    print("  HISTORICAL SIMULATION: Feb 25 / 26 / 27")
    print("="*60)

    # Pre-load states and strip old backfill entries
    print("\nStripping price-only backfill entries...")
    for code in STRATEGY_CODES:
        cls = ALL_STRATEGIES.get(code)
        if not cls:
            continue
        strategy = cls()
        sf = STATE_DIR / f"{strategy.name}_state.json"
        if not sf.exists():
            continue
        state = json.loads(sf.read_text())
        state = _strip_backfill_entries(state, SIMULATE_DATES)
        sf.write_text(json.dumps(state, indent=2, default=str))

    # Simulate each day in order
    for date_str in SIMULATE_DATES:
        sim_dt = datetime.strptime(date_str, "%Y-%m-%d")

        print(f"\n{'-'*60}")
        print(f"  Simulating {date_str}")
        print(f"{'-'*60}")

        # Patch datetime.today() globally in forward_test and data modules
        fake_today = sim_dt

        with patch("backtesting.forward_test.datetime") as mock_ft_dt, \
             patch("data.market_data.datetime") as mock_md_dt:

            # Make datetime behave normally except .today()
            mock_ft_dt.today.return_value  = fake_today
            mock_ft_dt.now.return_value    = fake_today
            mock_ft_dt.strptime            = datetime.strptime
            mock_ft_dt.fromisoformat       = datetime.fromisoformat
            mock_md_dt.today.return_value  = fake_today
            mock_md_dt.now.return_value    = fake_today
            mock_md_dt.strptime            = datetime.strptime

            for code in STRATEGY_CODES:
                cls = ALL_STRATEGIES.get(code)
                if not cls:
                    continue
                strategy = cls()
                ft = ForwardTest(strategy)

                universe = strategy.get_universe()
                spot = _get_spot_prices(universe, sim_dt)
                if not spot:
                    print(f"  [{strategy.name}] no price data — skip")
                    continue

                # Patch data.market_data functions directly — forward_test imports
                # them with `from data.market_data import ...` inside the method,
                # so we must patch the source module, not the forward_test namespace.
                with patch("data.market_data.get_close",
                           side_effect=lambda t, days=300, **kw: _get_close_capped(
                               t if isinstance(t, list) else [t], days, sim_dt)), \
                     patch("data.market_data.get_volume",
                           side_effect=lambda t, days=300, **kw: _get_volume_capped(
                               t if isinstance(t, list) else [t], days, sim_dt)), \
                     patch("data.market_data.get_vix",
                           side_effect=lambda days=300, **kw: _get_close_capped(
                               ["^VIX"], days, sim_dt).iloc[:, 0].rename("VIX")
                               if not _get_close_capped(["^VIX"], days, sim_dt).empty
                               else pd.Series(dtype=float)), \
                     patch("data.market_data.get_vix9d",
                           side_effect=lambda days=300, **kw: _get_close_capped(
                               ["^VIX9D"], days, sim_dt).iloc[:, 0].rename("VIX9D")
                               if not _get_close_capped(["^VIX9D"], days, sim_dt).empty
                               else pd.Series(dtype=float)), \
                     patch("data.market_data.get_yield_spread",
                           side_effect=lambda days=300, **kw: pd.Series(dtype=float)):

                    try:
                        summary = ft.update(spot)
                        pv = summary["portfolio_value"]
                        ret = (pv / 100_000 - 1) * 100
                        n_pos = summary["n_positions"]
                        print(f"  [{strategy.name}] ${pv:,.2f} ({ret:+.3f}%)  {n_pos} positions")
                    except Exception as exc:
                        print(f"  [{strategy.name}] ERROR: {exc}")
                        logging.exception("Simulation error for %s on %s", strategy.name, date_str)

    # Final: update portfolio_value in each state to latest daily_log
    print("\nFinalising state files...")
    for code in STRATEGY_CODES:
        cls = ALL_STRATEGIES.get(code)
        if not cls:
            continue
        strategy = cls()
        sf = STATE_DIR / f"{strategy.name}_state.json"
        if not sf.exists():
            continue
        state = json.loads(sf.read_text())
        log = sorted(state.get("daily_log", []), key=lambda x: x["date"])
        if log:
            state["daily_log"] = log
            state["portfolio_value"] = log[-1]["pv"]
            state["peak_value"] = max(state.get("peak_value", 0),
                                      max(e["pv"] for e in log))
            sf.write_text(json.dumps(state, indent=2, default=str))

    print("\nDone. Run `python main.py status` to verify.\n")


if __name__ == "__main__":
    simulate()
