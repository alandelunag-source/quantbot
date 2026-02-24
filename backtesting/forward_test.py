"""
Forward (paper) testing harness.

Simulates live trading using today's data with the same logic as backtest.
Tracks positions, P&L, and vs benchmark in real time.

Each strategy runs independently via its own ForwardTest instance.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

STATE_DIR = Path("state")  # JSON state files persisted per strategy


class ForwardTest:
    def __init__(self, strategy, capital: float = None):
        self.strategy = strategy
        self.capital = capital or settings.INITIAL_CAPITAL
        self.state_file = STATE_DIR / f"{strategy.name}_state.json"
        self._state = self._load_state()

    # --- State persistence ---

    def _load_state(self) -> dict:
        STATE_DIR.mkdir(exist_ok=True)
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {
            "positions": {},           # ticker -> {weight, entry_price, entry_date}
            "portfolio_value": self.capital,
            "peak_value": self.capital,
            "trades": [],
            "daily_returns": [],
            "start_date": datetime.today().isoformat(),
        }

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self._state, indent=2, default=str))

    # --- Core update loop ---

    def update(self, current_prices: dict[str, float], extra_kwargs: dict = None) -> dict:
        """
        Called once per day with current prices.
        1. Revalue current positions
        2. Generate new signals
        3. Rebalance if needed (respecting rebalance_freq)
        4. Apply transaction costs
        5. Persist state

        Returns summary dict.
        """
        extra = extra_kwargs or {}
        today = datetime.today().strftime("%Y-%m-%d")

        # Revalue
        pv = self._compute_portfolio_value(current_prices)
        self._state["portfolio_value"] = pv
        self._state["peak_value"] = max(self._state["peak_value"], pv)

        # Drawdown stop
        drawdown = (pv - self._state["peak_value"]) / self._state["peak_value"]
        if drawdown < -settings.DRAWDOWN_STOP_PCT:
            logger.warning("[FT:%s] Drawdown stop triggered: %.1f%% — liquidating",
                           self.strategy.name, drawdown * 100)
            self._liquidate(current_prices, reason="drawdown_stop")
            self._save_state()
            return self._summary(today, pv, drawdown)

        # Check rebalance schedule
        if not self._should_rebalance(today):
            self._save_state()
            return self._summary(today, pv, drawdown)

        # Get fresh price history for signal generation
        universe = self.strategy.get_universe()
        from data.market_data import get_close, get_volume, get_vix, get_yield_spread
        prices_df = get_close(universe, days=300)
        volume_df = get_volume(universe, days=300)
        vix = get_vix(days=300)
        ys = get_yield_spread(days=300)
        extra.update({"volume": volume_df, "vix": vix, "yield_spread": ys})

        if prices_df.empty:
            return self._summary(today, pv, drawdown)

        # Generate signals
        signals_df = self.strategy.generate_signals(prices_df, **extra)
        if signals_df.empty:
            return self._summary(today, pv, drawdown)

        latest_signals = signals_df.iloc[-1]
        target_weights = self.strategy.position_sizing(latest_signals)

        # Rebalance
        self._rebalance(target_weights, current_prices, pv, today)
        self._save_state()
        return self._summary(today, pv, drawdown)

    def _compute_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Mark-to-market current positions + cash."""
        cash = self._state.get("cash", self.capital)
        invested = 0.0
        for ticker, pos in self._state["positions"].items():
            price = current_prices.get(ticker, pos.get("entry_price", 0))
            shares = pos.get("shares", 0)
            invested += price * shares
        return cash + invested

    def _rebalance(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        portfolio_value: float,
        today: str,
    ) -> None:
        """Compute trades needed to hit target weights. Apply cost drag."""
        current_weights = {}
        for ticker, pos in self._state["positions"].items():
            price = current_prices.get(ticker, pos.get("entry_price", 0))
            shares = pos.get("shares", 0)
            current_weights[ticker] = (price * shares) / portfolio_value

        all_tickers = set(target_weights) | set(current_weights)
        total_cost = 0.0

        for ticker in all_tickers:
            target_w = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            delta_w = target_w - current_w

            if abs(delta_w) < 0.005:  # < 0.5% change → skip (min trade size)
                continue

            price = current_prices.get(ticker)
            if not price:
                continue

            dollar_trade = delta_w * portfolio_value
            cost = abs(dollar_trade) * settings.TOTAL_COST_PCT
            total_cost += cost

            # Update position
            if target_w <= 0:
                self._state["positions"].pop(ticker, None)
            else:
                shares = (target_w * portfolio_value) / price
                self._state["positions"][ticker] = {
                    "weight": target_w,
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": today,
                }

            self._state["trades"].append({
                "date": today,
                "ticker": ticker,
                "action": "BUY" if delta_w > 0 else "SELL",
                "delta_weight": round(delta_w, 4),
                "cost": round(cost, 4),
            })

        # Deduct costs from cash
        cash = self._state.get("cash", portfolio_value)
        self._state["cash"] = cash - total_cost

    def _liquidate(self, current_prices: dict, reason: str = "") -> None:
        for ticker in list(self._state["positions"].keys()):
            price = current_prices.get(ticker, 0)
            pos = self._state["positions"][ticker]
            self._state["cash"] = self._state.get("cash", 0) + price * pos.get("shares", 0)
        self._state["positions"] = {}
        logger.info("[FT:%s] Liquidated — %s", self.strategy.name, reason)

    def _should_rebalance(self, today: str) -> bool:
        freq = self.strategy.rebalance_freq
        last = self._state.get("last_rebalance", "")
        if not last:
            return True
        if freq == "daily":
            return True
        if freq == "weekly":
            last_dt = pd.Timestamp(last)
            now_dt = pd.Timestamp(today)
            return (now_dt - last_dt).days >= 5
        if freq == "monthly":
            last_dt = pd.Timestamp(last)
            now_dt = pd.Timestamp(today)
            return now_dt.month != last_dt.month
        return True

    def _summary(self, today: str, pv: float, drawdown: float) -> dict:
        start_val = self.capital
        total_return = (pv - start_val) / start_val
        n_positions = len(self._state["positions"])
        return {
            "date": today,
            "strategy": self.strategy.name,
            "portfolio_value": round(pv, 2),
            "total_return": round(total_return, 4),
            "drawdown_from_peak": round(drawdown, 4),
            "n_positions": n_positions,
            "positions": {t: round(p["weight"], 3) for t, p in self._state["positions"].items()},
        }

    def print_status(self) -> None:
        pv = self._state["portfolio_value"]
        start = self.capital
        ret = (pv - start) / start
        peak = self._state["peak_value"]
        dd = (pv - peak) / peak
        positions = self._state["positions"]

        print(f"\n--- {self.strategy.name} ---")
        print(f"  Value: ${pv:,.2f}  ({ret:+.2%} since start)")
        print(f"  Drawdown from peak: {dd:.2%}")
        print(f"  Positions ({len(positions)}):")
        for t, p in positions.items():
            print(f"    {t}: {p['weight']:.1%}  entry={p['entry_price']:.2f}")
        print()
