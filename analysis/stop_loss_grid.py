#!/usr/bin/env python3
"""
Stop-Loss / Profit-Target Grid Search

Finds the Sharpe-maximizing (stop-loss, profit-target) pair for each
stock-picking strategy using historical signal data and trade simulation.

BacktestEngine is vectorized and does NOT call exit_rules(), so this script
uses a simulation-based approach:
  1. Call generate_signals() to find entry events
  2. Simulate each trade forward with different stop/target combos
  3. Compute trade-level Sharpe, win rate, profit factor per combo

Usage:
    python analysis/stop_loss_grid.py --strategies s05,s17 --days 365
    python analysis/stop_loss_grid.py --strategies s01,s03,s04,s05,s11,s12,s15,s17 --days 500
"""

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.market_data import (
    get_close,
    get_volume,
    get_vix,
    get_vix9d,
    get_yield_spread,
)
from strategies import ALL_STRATEGIES

# ── Grid parameters ────────────────────────────────────────────────────────────
STOP_LOSSES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

# Includes all current strategy configs (0.025 for S01, 0.03 for S17, 0.12 for S15)
PROFIT_TARGETS = [0.025, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, None]

IS_FRAC = 0.70        # 70% in-sample, 30% out-of-sample
RF_ANNUAL = 0.04      # Risk-free rate for Sharpe calculation

# ── Strategies eligible for grid search ───────────────────────────────────────
# Regime/ETF strategies (s02,s06,s07,s09,s10,s13,s14,s16) are skipped
ELIGIBLE = {"s01", "s03", "s04", "s05", "s11", "s12", "s15", "s17"}


# ── Config helpers ─────────────────────────────────────────────────────────────

def get_strategy_config(strat):
    """Extract (stop_loss, profit_target, max_days) from a strategy instance.

    Handles non-standard attribute names (S01 uses EXIT_TARGET / EXIT_DAYS).
    """
    sl = getattr(strat, "STOP_LOSS", None)
    pt = getattr(strat, "PROFIT_TARGET", None)
    if pt is None:
        pt = getattr(strat, "EXIT_TARGET", None)  # S01 compat
    max_d = (
        getattr(strat, "TIME_STOP_DAYS", None)
        or getattr(strat, "EXIT_DAYS", None)      # S01 compat
        or getattr(strat, "HOLD_DAYS", None)
        or 30
    )
    return sl, pt, int(max_d)


def fmt_sl(sl):
    return "n/a" if sl is None else f"-{sl * 100:.1f}%"


def fmt_pt(pt):
    return "none" if pt is None else f"+{pt * 100:.1f}%"


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trade(entry_price: float, fwd_prices: np.ndarray,
                   stop_loss: float, profit_target, max_days: int):
    """
    Simulate a single trade and return (trade_return, days_held, exit_reason).

    Parameters
    ----------
    entry_price   : price on the entry date
    fwd_prices    : close prices for days AFTER entry (index 0 = day+1)
    stop_loss     : positive fraction, e.g. 0.04 means exit at -4%
    profit_target : positive fraction or None (no profit target, only time stop)
    max_days      : maximum holding period (time stop)
    """
    n = min(max_days, len(fwd_prices))
    if n == 0:
        return 0.0, 0, "time"

    for d in range(n):
        ret = fwd_prices[d] / entry_price - 1
        if ret <= -stop_loss:
            return ret, d + 1, "stop"
        if profit_target is not None and ret >= profit_target:
            return ret, d + 1, "target"

    # Time stop — use the last available price up to max_days
    ret = fwd_prices[n - 1] / entry_price - 1
    return ret, n, "time"


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_sharpe(rets: list, avg_hold_days: float) -> float:
    """Annualized Sharpe (rf=4%) from a list of per-trade returns."""
    if len(rets) < 3:
        return np.nan
    arr = np.array(rets, dtype=float)
    mu = arr.mean()
    sig = arr.std(ddof=1)
    if sig == 0.0:
        return np.nan
    periods_per_year = 252.0 / max(avg_hold_days, 1.0)
    rf_per_trade = RF_ANNUAL / periods_per_year
    return (mu - rf_per_trade) / sig * np.sqrt(periods_per_year)


def compute_metrics(trades: list) -> dict:
    """Aggregate trade-level stats for a (stop, target) combo."""
    if not trades:
        return dict(n=0, win_rate=np.nan, profit_factor=np.nan,
                    avg_ret=np.nan, sharpe=np.nan)
    rets  = [t["ret"]  for t in trades]
    holds = [t["days"] for t in trades]
    avg_hold = float(np.mean(holds)) if holds else 1.0

    winners = [r for r in rets if r > 0]
    losers  = [r for r in rets if r <= 0]
    win_rate      = len(winners) / len(rets)
    loser_sum     = sum(losers)
    profit_factor = (sum(winners) / abs(loser_sum)
                     if losers and loser_sum != 0 else np.inf)

    return dict(
        n=len(rets),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_ret=float(np.mean(rets)),
        sharpe=compute_sharpe(rets, avg_hold),
    )


# ── Main per-strategy runner ───────────────────────────────────────────────────

def run_grid(strategy_key: str, days: int):
    """Run the stop/target grid for one strategy. Returns summary dict or None."""
    if strategy_key not in ALL_STRATEGIES:
        print(f"\n  [{strategy_key}] Unknown strategy — skipping")
        return None
    if strategy_key not in ELIGIBLE:
        print(f"\n  [{strategy_key}] Regime/ETF strategy — no per-position exits, skipping")
        return None

    strat     = ALL_STRATEGIES[strategy_key]()
    cur_sl, cur_pt, cur_max_days = get_strategy_config(strat)

    print(f"\n{'=' * 72}")
    print(f"  {strategy_key.upper()}  {strat.name}")
    print(f"  Current config: SL={fmt_sl(cur_sl)}  PT={fmt_pt(cur_pt)}  "
          f"max_days={cur_max_days}")
    print(f"{'=' * 72}")

    # ── Fetch market data ──────────────────────────────────────────────────────
    print("  Fetching universe and price data...")
    universe   = strat.get_universe()
    fetch_days = days + 90  # extra warmup buffer for signal computation

    try:
        prices = get_close(universe, fetch_days)
        volume = get_volume(universe, fetch_days)
        vix    = get_vix(fetch_days)
        vix9d  = get_vix9d(fetch_days)
        ys     = get_yield_spread(fetch_days)
    except Exception as exc:
        print(f"  ERROR fetching data: {exc}")
        return None

    if prices is None or prices.empty:
        print("  ERROR: no price data returned")
        return None

    # ── Generate signals ───────────────────────────────────────────────────────
    print("  Generating signals...")
    try:
        signals = strat.generate_signals(
            prices,
            volume=volume,
            vix=vix,
            vix9d=vix9d,
            yield_spread=ys,
        )
    except Exception as exc:
        print(f"  ERROR in generate_signals: {exc}")
        return None

    if signals is None or signals.empty:
        print("  ERROR: generate_signals returned empty DataFrame")
        return None

    # Align to common dates
    common = signals.index.intersection(prices.index)
    signals = signals.loc[common]
    prices  = prices.loc[common]

    # ── Extract entry events ───────────────────────────────────────────────────
    print("  Extracting entry events...")
    entries = []
    for ticker in signals.columns:
        if ticker not in prices.columns:
            continue
        sig = signals[ticker].fillna(0)
        px  = prices[ticker].dropna()

        # New entry: signal transitions from <=0 to >0
        new_entry = (sig > 0) & (sig.shift(1).fillna(0) <= 0)
        for entry_date in sig[new_entry].index:
            if entry_date not in px.index:
                continue
            ep = px.loc[entry_date]
            if pd.isna(ep) or ep <= 0:
                continue
            # Forward prices (days after entry)
            future = px.loc[px.index > entry_date]
            if len(future) < 2:
                continue
            entries.append({
                "ticker":      ticker,
                "entry_date":  entry_date,
                "entry_price": float(ep),
                "fwd_prices":  future.values.astype(float),
            })

    if not entries:
        print("  ERROR: no entry events found — check signal generation")
        return None

    # Sort chronologically and split IS / OOS
    entries.sort(key=lambda x: x["entry_date"])
    split_idx   = int(len(entries) * IS_FRAC)
    is_entries  = entries[:split_idx]
    oos_entries = entries[split_idx:]
    oos_n       = len(oos_entries)

    print(f"  Total entries: {len(entries)}   IS: {len(is_entries)}   OOS: {oos_n}")

    if oos_n < 2:
        print("  Skipping grid — fewer than 2 OOS trades")
        return None
    if oos_n < 20:
        print(f"  WARNING: only {oos_n} OOS trades — Sharpe estimates may be noisy")

    # ── Grid search on OOS trades ──────────────────────────────────────────────
    max_days = cur_max_days
    grid: dict = {}  # (sl, pt) -> metrics dict

    for sl, pt in product(STOP_LOSSES, PROFIT_TARGETS):
        trades = []
        for e in oos_entries:
            ret, days_held, reason = simulate_trade(
                e["entry_price"], e["fwd_prices"], sl, pt, max_days
            )
            trades.append({"ret": ret, "days": days_held, "reason": reason})
        grid[(sl, pt)] = compute_metrics(trades)

    # ── Find optimal combo ─────────────────────────────────────────────────────
    best_sharpe = -np.inf
    best_combo  = None
    for (sl, pt), m in grid.items():
        sh = m["sharpe"]
        if not np.isnan(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_combo  = (sl, pt)

    cur_sharpe = grid.get((cur_sl, cur_pt), {}).get("sharpe", np.nan)

    # ── Print grid table ───────────────────────────────────────────────────────
    sl_labels = [f"{int(sl * 100)}%" for sl in STOP_LOSSES]
    col_w = 8  # chars per cell

    print(f"\n  OOS Sharpe grid  [{oos_n} OOS trades | max_days={max_days}]")
    print(f"  {'PT \\ SL':>8}  " + "  ".join(f"{lbl:>{col_w}}" for lbl in sl_labels))
    print("  " + "-" * (10 + len(STOP_LOSSES) * (col_w + 2)))

    for pt in PROFIT_TARGETS:
        pt_str    = fmt_pt(pt)
        is_cur_pt = (pt == cur_pt)
        is_opt_pt = (best_combo is not None and pt == best_combo[1])

        if is_opt_pt and is_cur_pt:
            prefix = ">>*"
        elif is_opt_pt:
            prefix = ">> "
        elif is_cur_pt:
            prefix = " * "
        else:
            prefix = "   "

        row = f"  {prefix} {pt_str:>6}   "
        for sl in STOP_LOSSES:
            m  = grid[(sl, pt)]
            sh = m["sharpe"]
            is_cur = (sl == cur_sl and pt == cur_pt)
            is_opt = (best_combo == (sl, pt))

            if np.isnan(sh):
                cell = "  nan  "
            elif is_opt and is_cur:
                cell = f"[{sh:5.2f}]*"
            elif is_opt:
                cell = f"[{sh:5.2f}] "
            elif is_cur:
                cell = f" {sh:5.2f} *"
            else:
                cell = f" {sh:5.2f}  "

            row += f" {cell}"
        print(row)

    print()
    if not np.isnan(cur_sharpe):
        print(f"  Current config:  SL={fmt_sl(cur_sl)}  PT={fmt_pt(cur_pt)}"
              f"  (OOS Sharpe: {cur_sharpe:.3f})")
    else:
        print(f"  Current config:  SL={fmt_sl(cur_sl)}  PT={fmt_pt(cur_pt)}"
              f"  (current params not on grid)")

    if best_combo is not None:
        bsl, bpt = best_combo
        print(f"  Optimal OOS:     SL={fmt_sl(bsl)}  PT={fmt_pt(bpt)}"
              f"  (OOS Sharpe: {best_sharpe:.3f})")
        if best_combo != (cur_sl, cur_pt):
            gain = best_sharpe - (cur_sharpe if not np.isnan(cur_sharpe) else best_sharpe)
            print(f"  Sharpe gain:     {'+' if gain >= 0 else ''}{gain:.3f}  <-- recommend updating")
        else:
            print("  Current config is already optimal on OOS data.")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = []
    for (sl, pt), m in grid.items():
        rows.append({
            "stop_loss":      sl,
            "profit_target":  "none" if pt is None else pt,
            "n_trades":       m["n"],
            "win_rate":       round(m["win_rate"], 4)      if not np.isnan(m.get("win_rate", np.nan))      else np.nan,
            "profit_factor":  round(m["profit_factor"], 4) if not np.isnan(m.get("profit_factor", np.nan)) else np.nan,
            "avg_ret":        round(m["avg_ret"], 6)       if not np.isnan(m.get("avg_ret", np.nan))       else np.nan,
            "sharpe":         round(m["sharpe"], 4)        if not np.isnan(m["sharpe"])                     else np.nan,
            "is_current":     (sl == cur_sl and pt == cur_pt),
            "is_optimal":     (best_combo == (sl, pt)),
        })

    csv_path = Path(__file__).parent / f"stop_loss_grid_{strategy_key.upper()}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path.name}")

    return {
        "strategy":   strategy_key,
        "name":       strat.name,
        "cur_sl":     cur_sl,
        "cur_pt":     cur_pt,
        "opt_sl":     best_combo[0] if best_combo else None,
        "opt_pt":     best_combo[1] if best_combo else None,
        "cur_sharpe": cur_sharpe,
        "opt_sharpe": best_sharpe,
        "oos_n":      oos_n,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stop-Loss / Profit-Target Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategies",
        default=",".join(sorted(ELIGIBLE)),
        help="Comma-separated strategy keys (default: all 8 eligible)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=500,
        help="Days of historical data to use (default: 500)",
    )
    args = parser.parse_args()

    strategy_keys = [k.strip().lower() for k in args.strategies.split(",") if k.strip()]

    print("=" * 72)
    print("  STOP-LOSS / PROFIT-TARGET GRID SEARCH")
    print("=" * 72)
    print(f"  Strategies      : {', '.join(strategy_keys)}")
    print(f"  Historical days : {args.days}")
    print(f"  Stop grid       : {[f'{int(s*100)}%' for s in STOP_LOSSES]}")
    print(f"  Target grid     : {[fmt_pt(p) for p in PROFIT_TARGETS]}")
    print(f"  IS / OOS split  : {IS_FRAC*100:.0f}% / {(1 - IS_FRAC)*100:.0f}%")
    print(f"  Combos per strat: {len(STOP_LOSSES) * len(PROFIT_TARGETS)}")

    summaries = []
    for key in strategy_keys:
        result = run_grid(key, args.days)
        if result:
            summaries.append(result)

    # ── Summary table ──────────────────────────────────────────────────────────
    if not summaries:
        print("\nNo results to summarize.")
        return

    print(f"\n\n{'=' * 80}")
    print("  OPTIMAL STOP-LOSS / PROFIT-TARGET RECOMMENDATIONS")
    print(f"{'=' * 80}")

    hdr = (f"  {'Strategy':<25}  {'CurSL':>6}  {'OptSL':>6}  "
           f"{'CurPT':>6}  {'OptPT':>6}  {'CurSh':>6}  {'OptSh':>6}  "
           f"{'Gain':>6}  {'OOS_n':>5}  {'Action'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for s in summaries:
        cur_sh = s["cur_sharpe"]
        opt_sh = s["opt_sharpe"]
        gain   = (opt_sh - cur_sh) if not (np.isnan(cur_sh) or np.isnan(opt_sh)) else np.nan
        gain_s = f"{gain:+.3f}" if not np.isnan(gain) else "  n/a"
        same   = (s["opt_sl"] == s["cur_sl"] and s["opt_pt"] == s["cur_pt"])
        action = "OK (current=optimal)" if same else "<= UPDATE RECOMMENDED"
        cur_sh_s = f"{cur_sh:.3f}" if not np.isnan(cur_sh) else "  n/a"
        print(
            f"  {s['name']:<25}  {fmt_sl(s['cur_sl']):>6}  {fmt_sl(s['opt_sl']):>6}  "
            f"{fmt_pt(s['cur_pt']):>6}  {fmt_pt(s['opt_pt']):>6}  "
            f"{cur_sh_s:>6}  {opt_sh:>6.3f}  "
            f"{gain_s:>6}  {s['oos_n']:>5}  {action}"
        )

    print(f"\n  Grid CSVs saved in: analysis/stop_loss_grid_<STRATEGY>.csv")
    print()


if __name__ == "__main__":
    main()
