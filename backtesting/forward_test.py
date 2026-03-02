"""
Forward (paper) testing harness.

Each strategy runs independently with its own JSON state file in state/.
Call update() once per trading day (via --once mode in main.py).

State is fully persistent across runs — safe to restart at any time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

STATE_DIR = Path("state")


def is_trading_day(dt: datetime = None) -> bool:
    """Return True if dt (default: today) is a weekday (Mon-Fri)."""
    d = dt or datetime.today()
    return d.weekday() < 5  # 0=Mon, 4=Fri


class ForwardTest:
    def __init__(self, strategy, capital: float = None):
        self.strategy = strategy
        self.capital = capital or settings.INITIAL_CAPITAL
        self.state_file = STATE_DIR / f"{strategy.name}_state.json"
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        STATE_DIR.mkdir(exist_ok=True)
        if self.state_file.exists():
            try:
                s = json.loads(self.state_file.read_text())
                # Back-compat: ensure cash key exists
                if "cash" not in s:
                    s["cash"] = s.get("portfolio_value", self.capital)
                return s
            except Exception:
                pass
        return {
            "positions":       {},        # ticker -> {weight, shares, entry_price, entry_date}
            "cash":            self.capital,
            "portfolio_value": self.capital,
            "peak_value":      self.capital,
            "trades":          [],
            "daily_log":       [],        # [{date, pv, ret_pct, regime/positions}]
            "start_date":      datetime.today().isoformat(),
        }

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self._state, indent=2, default=str))

    # ------------------------------------------------------------------
    # Core daily update
    # ------------------------------------------------------------------

    def update(self, current_prices: dict[str, float], extra_kwargs: dict = None) -> dict:
        """
        Run one daily update. Idempotent if called twice on the same date.
        1. Skip if already ran today
        2. Mark-to-market
        3. Check drawdown stop
        4. Generate signals if rebalance due
        5. Rebalance with proper cash tracking
        6. Log and persist
        """
        today = datetime.today().strftime("%Y-%m-%d")
        extra = extra_kwargs or {}

        # Idempotency: skip if already ran today
        last_log = self._state["daily_log"][-1] if self._state["daily_log"] else {}
        if last_log.get("date") == today:
            logger.debug("[FT:%s] Already ran today (%s), skipping", self.strategy.name, today)
            return self._summary(today)

        # Mark-to-market
        pv = self._compute_portfolio_value(current_prices)
        prev_pv = self._state["portfolio_value"]
        self._state["portfolio_value"] = pv
        self._state["peak_value"] = max(self._state["peak_value"], pv)

        drawdown = (pv - self._state["peak_value"]) / self._state["peak_value"]

        # Drawdown stop
        if drawdown < -settings.DRAWDOWN_STOP_PCT:
            logger.warning("[FT:%s] Drawdown stop %.1f%% — liquidating", self.strategy.name, drawdown * 100)
            self._liquidate(current_prices, reason="drawdown_stop")
            self._log_day(today, pv, prev_pv, note="DRAWDOWN_STOP")
            self._save_state()
            return self._summary(today)

        # Exit rules (always checked, regardless of rebalance schedule)
        self._apply_exit_rules(current_prices, pv, today)

        # Rebalance?
        if self._should_rebalance(today):
            self._run_signals_and_rebalance(current_prices, pv, today, extra)
            self._state["last_rebalance"] = today

        self._log_day(today, pv, prev_pv)
        self._save_state()
        return self._summary(today)

    def _apply_exit_rules(
        self,
        current_prices: dict[str, float],
        portfolio_value: float,
        today: str,
    ) -> None:
        """
        Check exit_rules() for every held position and immediately close any that trigger.
        Runs on every mark-to-market (not just rebalance days) so stop-losses fire intraday.
        """
        exit_weights = {}
        for ticker, pos in self._state["positions"].items():
            entry_price = pos["entry_price"]
            current_price = current_prices.get(ticker, entry_price)
            try:
                days_held = (pd.Timestamp(today) - pd.Timestamp(pos.get("entry_date", today))).days
            except Exception:
                days_held = 0
            if self.strategy.exit_rules(entry_price, current_price, days_held):
                exit_weights[ticker] = 0.0
                logger.info(
                    "[FT:%s] exit_rules triggered for %s (held %dd, entry=%.2f, now=%.2f)",
                    self.strategy.name, ticker, days_held, entry_price, current_price,
                )
        if exit_weights:
            self._rebalance(exit_weights, current_prices, portfolio_value, today)

    def _run_signals_and_rebalance(
        self,
        current_prices: dict[str, float],
        portfolio_value: float,
        today: str,
        extra: dict,
    ) -> None:
        from data.market_data import get_close, get_volume, get_vix, get_vix9d, get_yield_spread

        universe = self.strategy.get_universe()
        prices_df = get_close(universe, days=300)
        volume_df = get_volume(universe, days=300)
        vix      = get_vix(days=300)
        vix9d    = get_vix9d(days=300)
        ys       = get_yield_spread(days=300)

        if prices_df.empty:
            return

        extra.update({"volume": volume_df, "vix": vix, "vix9d": vix9d, "yield_spread": ys})
        signals_df = self.strategy.generate_signals(prices_df, **extra)
        if signals_df.empty:
            return

        latest = signals_df.iloc[-1]
        target_weights = self.strategy.position_sizing(latest, prices=prices_df)
        self._rebalance(target_weights, current_prices, portfolio_value, today)

    def _compute_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Cash + mark-to-market of all positions."""
        cash = self._state["cash"]
        invested = sum(
            current_prices.get(t, pos["entry_price"]) * pos["shares"]
            for t, pos in self._state["positions"].items()
        )
        return cash + invested

    def _rebalance(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        portfolio_value: float,
        today: str,
    ) -> None:
        """
        Rebalance to target_weights.
        Cash tracking: BUY reduces cash, SELL increases cash, both net of costs.
        """
        # Current weights from live mark-to-market
        current_weights: dict[str, float] = {}
        for t, pos in self._state["positions"].items():
            price = current_prices.get(t, pos["entry_price"])
            current_weights[t] = (price * pos["shares"]) / portfolio_value

        all_tickers = set(target_weights) | set(current_weights)

        for ticker in all_tickers:
            target_w  = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            delta_w   = target_w - current_w

            if abs(delta_w) < 0.005:   # < 0.5% → skip (below min trade threshold)
                continue

            price = current_prices.get(ticker)
            if not price:
                continue

            dollar_delta = delta_w * portfolio_value
            cost = abs(dollar_delta) * settings.TOTAL_COST_PCT

            if delta_w > 0:
                # BUY: deduct cost + principal from cash
                self._state["cash"] -= (dollar_delta + cost)
                shares_to_add = dollar_delta / price
                if ticker in self._state["positions"]:
                    old = self._state["positions"][ticker]
                    total_shares = old["shares"] + shares_to_add
                    avg_entry = (old["entry_price"] * old["shares"] + price * shares_to_add) / total_shares
                    self._state["positions"][ticker] = {
                        "weight": target_w, "shares": total_shares,
                        "entry_price": avg_entry, "entry_date": old["entry_date"],
                    }
                else:
                    self._state["positions"][ticker] = {
                        "weight": target_w,
                        "shares": dollar_delta / price,
                        "entry_price": price,
                        "entry_date": today,
                    }
            else:
                # SELL: add proceeds (net of cost) to cash
                shares_to_sell = abs(dollar_delta) / price
                self._state["cash"] += (abs(dollar_delta) - cost)
                if target_w <= 0:
                    self._state["positions"].pop(ticker, None)
                else:
                    pos = self._state["positions"].get(ticker, {})
                    remaining = pos.get("shares", 0) - shares_to_sell
                    if remaining > 0:
                        self._state["positions"][ticker] = {
                            **pos, "weight": target_w, "shares": remaining,
                        }
                    else:
                        self._state["positions"].pop(ticker, None)

            self._state["trades"].append({
                "date": today, "ticker": ticker,
                "action": "BUY" if delta_w > 0 else "SELL",
                "delta_weight": round(delta_w, 4),
                "dollar_value": round(abs(dollar_delta), 2),
                "cost": round(cost, 4),
            })

    def _liquidate(self, current_prices: dict, reason: str = "") -> None:
        for ticker, pos in list(self._state["positions"].items()):
            price = current_prices.get(ticker, pos["entry_price"])
            self._state["cash"] += price * pos["shares"]
        self._state["positions"] = {}
        logger.info("[FT:%s] Liquidated — %s", self.strategy.name, reason)

    def _should_rebalance(self, today: str) -> bool:
        freq = self.strategy.rebalance_freq
        last = self._state.get("last_rebalance", "")
        if not last:
            return True
        if freq == "daily":
            return True
        last_dt = pd.Timestamp(last)
        now_dt  = pd.Timestamp(today)
        if freq == "weekly":
            return (now_dt - last_dt).days >= 5
        if freq == "monthly":
            return now_dt.month != last_dt.month
        return True

    def _log_day(self, today: str, pv: float, prev_pv: float, note: str = "") -> None:
        ret_pct = (pv - prev_pv) / prev_pv if prev_pv else 0.0
        self._state["daily_log"].append({
            "date": today,
            "pv": round(pv, 2),
            "ret_pct": round(ret_pct, 6),
            "n_positions": len(self._state["positions"]),
            "note": note,
        })

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _summary(self, today: str) -> dict:
        pv = self._state["portfolio_value"]
        total_return = (pv - self.capital) / self.capital
        drawdown = (pv - self._state["peak_value"]) / self._state["peak_value"]
        return {
            "date": today,
            "strategy": self.strategy.name,
            "portfolio_value": round(pv, 2),
            "total_return": round(total_return, 4),
            "drawdown_from_peak": round(drawdown, 4),
            "n_positions": len(self._state["positions"]),
            "positions": {t: round(p["weight"], 3) for t, p in self._state["positions"].items()},
        }

    def print_status(self) -> None:
        pv   = self._state["portfolio_value"]
        ret  = (pv - self.capital) / self.capital
        peak = self._state["peak_value"]
        dd   = (pv - peak) / peak

        # Daily return from log
        log  = self._state["daily_log"]
        day_ret = log[-1]["ret_pct"] if log else 0.0

        # Trade count
        n_trades = len(self._state["trades"])

        positions = self._state["positions"]
        print(f"\n  [{self.strategy.name}]")
        print(f"    Value   : ${pv:>12,.2f}  ({ret:+.2%} total | {day_ret:+.2%} today)")
        print(f"    Peak DD : {dd:.2%}   Trades: {n_trades}")
        if positions:
            print(f"    Positions ({len(positions)}):", end="")
            for t, p in positions.items():
                entry = p.get("entry_price", 0)
                print(f"  {t} {p['weight']:.0%}@{entry:.2f}", end="")
            print()
        else:
            print("    Positions: flat (cash)")

    def sharpe(self) -> float:
        log = self._state["daily_log"]
        if len(log) < 5:
            return float("nan")
        rets = np.array([d["ret_pct"] for d in log])
        if rets.std() == 0:
            return float("nan")
        return (rets.mean() / rets.std()) * np.sqrt(252)

    def max_drawdown(self) -> float:
        log = self._state["daily_log"]
        if not log:
            return 0.0
        pvs = np.array([d["pv"] for d in log])
        roll_max = np.maximum.accumulate(pvs)
        return float(((pvs - roll_max) / roll_max).min())
