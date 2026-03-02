#!/usr/bin/env python3
"""
Back-adjust paper trading state after stop-loss / profit-target param changes.

Replays each strategy from its start date using updated exit parameters,
correctly applying signal-weighted sizing and dry-powder rules so that
any cash freed by early exits is redeployed into new entries on subsequent
signal rebalances -- exactly as the live system would have done.

Usage:
    python analysis/backfill_exit_params.py
"""
import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.market_data import get_close, get_volume, get_vix, get_vix9d, get_yield_spread
from config import settings

STATE_DIR       = Path("state")
TOTAL_COST_PCT  = settings.TOTAL_COST_PCT
DRAWDOWN_STOP   = settings.DRAWDOWN_STOP_PCT
INITIAL_CAPITAL = settings.INITIAL_CAPITAL


# ── Rebalance (exact copy of ForwardTest._rebalance) ─────────────────────────

def _rebalance(state: dict, target_weights: dict, prices: dict,
               pv: float, date_str: str) -> None:
    current_weights: dict = {}
    for t, pos in state["positions"].items():
        price = prices.get(t, pos["entry_price"])
        current_weights[t] = (price * pos["shares"]) / pv if pv else 0.0

    for ticker in set(target_weights) | set(current_weights):
        target_w  = target_weights.get(ticker, 0.0)
        current_w = current_weights.get(ticker, 0.0)
        delta_w   = target_w - current_w
        if abs(delta_w) < 0.005:
            continue
        price = prices.get(ticker)
        if not price:
            continue
        dollar_delta = delta_w * pv
        cost = abs(dollar_delta) * TOTAL_COST_PCT

        if delta_w > 0:
            state["cash"] -= dollar_delta + cost
            shares_add = dollar_delta / price
            if ticker in state["positions"]:
                old   = state["positions"][ticker]
                total = old["shares"] + shares_add
                avg   = (old["entry_price"] * old["shares"] + price * shares_add) / total
                state["positions"][ticker] = {
                    "weight": target_w, "shares": total,
                    "entry_price": avg, "entry_date": old["entry_date"],
                }
            else:
                state["positions"][ticker] = {
                    "weight": target_w, "shares": dollar_delta / price,
                    "entry_price": price, "entry_date": date_str,
                }
        else:
            state["cash"] += abs(dollar_delta) - cost
            if target_w <= 0:
                state["positions"].pop(ticker, None)
            else:
                pos = state["positions"].get(ticker, {})
                remaining = pos.get("shares", 0) - abs(dollar_delta) / price
                if remaining > 0:
                    state["positions"][ticker] = {**pos, "weight": target_w, "shares": remaining}
                else:
                    state["positions"].pop(ticker, None)

        state["trades"].append({
            "date": date_str, "ticker": ticker,
            "action": "BUY" if delta_w > 0 else "SELL",
            "delta_weight": round(delta_w, 4),
            "dollar_value": round(abs(dollar_delta), 2),
            "cost": round(cost, 4),
        })


def _should_rebalance(freq: str, last_reb: str, today: pd.Timestamp) -> bool:
    if not last_reb:
        return True
    last_dt = pd.Timestamp(last_reb)
    if freq == "daily":
        return True
    if freq == "weekly":
        return (today - last_dt).days >= 5
    if freq == "monthly":
        return today.month != last_dt.month
    return True


# ── Core replay engine ────────────────────────────────────────────────────────

def replay_strategy(strat, start_date_str: str, prices_df: pd.DataFrame,
                    volume_df, vix_s, vix9d_s, ys_s,
                    original_start_ts: str) -> dict:
    """
    Replay a strategy day-by-day from start_date_str using its current params.

    Mirrors ForwardTest.update() exactly:
      1. Mark-to-market (pre-rebalance)
      2. Drawdown stop check
      3. exit_rules -> close positions
      4. Signal rebalance (if frequency allows) -> open/resize positions
      5. Log day

    Cash freed by exits is automatically redeployed on the next signal
    rebalance via position_sizing(), which scales to current portfolio value.
    """
    start_dt     = pd.Timestamp(start_date_str)
    replay_dates = sorted([d for d in prices_df.index if d >= start_dt])

    state = {
        "positions":      {},
        "cash":           INITIAL_CAPITAL,
        "portfolio_value": INITIAL_CAPITAL,
        "peak_value":     INITIAL_CAPITAL,
        "trades":         [],
        "daily_log":      [],
        "start_date":     original_start_ts,
        "last_rebalance": "",
    }
    prev_pv = INITIAL_CAPITAL   # PRE-rebalance pv from previous day (mirrors ForwardTest)

    for date in replay_dates:
        date_str    = date.strftime("%Y-%m-%d")
        px_hist     = prices_df.loc[:date]
        vol_hist    = volume_df.loc[:date]   if volume_df is not None else None
        vix_h       = vix_s.loc[:date]       if vix_s   is not None else None
        vix9d_h     = vix9d_s.loc[:date]     if vix9d_s is not None else None
        ys_h        = ys_s.loc[:date]        if ys_s   is not None else None
        today_px    = px_hist.iloc[-1].dropna().to_dict()

        # 1. Mark-to-market (pre-rebalance)
        pv = state["cash"] + sum(
            today_px.get(t, pos["entry_price"]) * pos["shares"]
            for t, pos in state["positions"].items()
        )
        state["portfolio_value"] = pv
        state["peak_value"]      = max(state["peak_value"], pv)

        # 2. Drawdown stop
        dd = (pv - state["peak_value"]) / state["peak_value"] if state["peak_value"] else 0.0
        if dd < -DRAWDOWN_STOP:
            print(f"    [{date_str}] DRAWDOWN STOP ({dd:.1%}) -- liquidating")
            for t, pos in list(state["positions"].items()):
                pr = today_px.get(t, pos["entry_price"])
                state["cash"] += pr * pos["shares"]
                state["trades"].append({"date": date_str, "ticker": t, "action": "SELL",
                                        "delta_weight": -pos["weight"],
                                        "dollar_value": round(pr * pos["shares"], 2), "cost": 0.0})
            state["positions"] = {}
            state["daily_log"].append({"date": date_str, "pv": round(pv, 2),
                                       "ret_pct": round((pv-prev_pv)/prev_pv if prev_pv else 0, 6),
                                       "n_positions": 0, "note": "DRAWDOWN_STOP"})
            prev_pv = pv
            continue

        # 3. Exit rules (fires every day, any frequency)
        exits = {}
        for ticker, pos in state["positions"].items():
            ep        = pos["entry_price"]
            cp        = today_px.get(ticker, ep)
            days_held = (date - pd.Timestamp(pos["entry_date"])).days
            if strat.exit_rules(ep, cp, days_held):
                exits[ticker] = 0.0
                reason = "PT" if (cp / ep - 1) >= 0 else "SL" if (cp / ep - 1) <= -strat.STOP_LOSS else "TIME"
                print(f"    [{date_str}] EXIT {reason} {ticker}: "
                      f"{cp/ep-1:+.2%} in {days_held}d  (freed ${cp*pos['shares']:,.0f})")
        if exits:
            _rebalance(state, exits, today_px, pv, date_str)

        # 4. Signal-driven rebalance (respects frequency; deploys freed cash)
        if _should_rebalance(strat.rebalance_freq, state.get("last_rebalance", ""), date):
            try:
                signals = strat.generate_signals(
                    px_hist, volume=vol_hist, vix=vix_h, vix9d=vix9d_h, yield_spread=ys_h
                )
                if not signals.empty:
                    latest = signals.iloc[-1]
                    target = strat.position_sizing(latest, prices=px_hist)
                    if target:
                        _rebalance(state, target, today_px, pv, date_str)
            except Exception as exc:
                print(f"    [{date_str}] signal error: {exc}")
            state["last_rebalance"] = date_str

        # 5. Log (use PRE-rebalance pv, matching ForwardTest)
        ret_pct = (pv - prev_pv) / prev_pv if prev_pv else 0.0
        state["daily_log"].append({
            "date": date_str, "pv": round(pv, 2), "ret_pct": round(ret_pct, 6),
            "n_positions": len(state["positions"]), "note": "",
        })
        prev_pv = pv

    return state


def _print_result(state: dict, label: str, state_file: Path) -> None:
    pv = state["portfolio_value"]
    print(f"\n  Result: PV=${pv:,.2f}  ({(pv - INITIAL_CAPITAL):+,.2f}  "
          f"{(pv/INITIAL_CAPITAL - 1):+.2%})")
    pos_str = ", ".join(f"{t} {p['weight']:.0%}" for t, p in state["positions"].items())
    print(f"  Positions ({len(state['positions'])}): {pos_str or 'flat (all cash)'}")
    print(f"  Trades: {len(state['trades'])}")
    print(f"  Daily log:")
    for row in state["daily_log"]:
        print(f"    {row['date']}  PV=${row['pv']:>12,.2f}  ret={row['ret_pct']:+.4%}  "
              f"pos={row['n_positions']}  {row.get('note','')}")
    print(f"  State written: {state_file}")


# ── Strategy-specific runners ─────────────────────────────────────────────────

def backfill_s05():
    print("\n-- S05 ShortTermReversal -- full universe replay ----------------------")
    from strategies.s05_short_term_reversal import ShortTermReversal
    strat = ShortTermReversal()
    print(f"  Params: SL={strat.STOP_LOSS:.1%}  PT={strat.PROFIT_TARGET:.1%}  "
          f"TIME_STOP={strat.TIME_STOP_DAYS}d  rebalance={strat.rebalance_freq}")
    print("  Fetching full SP100 universe prices...")

    universe  = strat.get_universe()
    prices_df = get_close(universe,  days=60)
    volume_df = get_volume(universe, days=60)

    # Show the two entry positions' path before replaying
    period = prices_df.loc[prices_df.index >= pd.Timestamp("2026-02-24"), ["IBM","ACN"]].dropna()
    IBM_ENTRY, ACN_ENTRY = 223.35, 201.18
    print(f"\n  IBM / ACN path vs entry prices:")
    for date, row in period.iterrows():
        ibm_r = row["IBM"] / IBM_ENTRY - 1
        acn_r = row["ACN"] / ACN_ENTRY - 1
        ibm_f = " PT " if ibm_r >=  strat.PROFIT_TARGET else (" SL " if ibm_r <= -strat.STOP_LOSS else "    ")
        acn_f = " PT " if acn_r >=  strat.PROFIT_TARGET else (" SL " if acn_r <= -strat.STOP_LOSS else "    ")
        print(f"    {date.date()}  IBM ${row['IBM']:>7.2f} ({ibm_r:+.2%}){ibm_f}"
              f"   ACN ${row['ACN']:>7.2f} ({acn_r:+.2%}){acn_f}")

    print("\n  Replaying...")
    state      = replay_strategy(strat, "2026-02-24", prices_df, volume_df,
                                 None, None, None, "2026-02-24T17:57:18.077471")
    state_file = STATE_DIR / "s05_short_term_reversal_state.json"
    state_file.write_text(json.dumps(state, indent=2, default=str))
    _print_result(state, "S05", state_file)


def backfill_s17():
    print("\n-- S17 PanicReversal -- full universe replay --------------------------")
    from strategies.s17_panic_reversal import PanicReversal
    strat = PanicReversal()
    print(f"  Params: SL={strat.STOP_LOSS:.1%}  PT={strat.PROFIT_TARGET:.1%}  "
          f"HOLD={strat.HOLD_DAYS}d  rebalance={strat.rebalance_freq}")
    print("  Fetching SP100 + SPY universe prices...")

    universe  = strat.get_universe()
    prices_df = get_close(universe,  days=120)
    volume_df = get_volume(universe, days=120)
    vix_s     = get_vix(days=120)
    vix9d_s   = get_vix9d(days=120)
    ys_s      = get_yield_spread(days=120)

    print("  Replaying...")
    state      = replay_strategy(strat, "2026-02-24", prices_df, volume_df,
                                 vix_s, vix9d_s, ys_s, "2026-02-24T20:56:08.100470")
    state_file = STATE_DIR / "s17_panic_reversal_state.json"
    state_file.write_text(json.dumps(state, indent=2, default=str))
    _print_result(state, "S17", state_file)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  BACK-ADJUST PAPER TRADING STATE")
    print("  Replays S05 + S17 with updated exit params + correct sizing rules")
    print("  Cash freed by early exits is redeployed on next signal rebalance")
    print("=" * 70)

    backfill_s05()
    backfill_s17()

    print("\nDone. Run 'python main.py paper --once' to continue from today.")
