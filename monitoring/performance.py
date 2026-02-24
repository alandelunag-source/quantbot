"""
Performance tracker — daily printout, Citadel-style dashboard.

Tracks each strategy independently AND aggregate portfolio.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, initial_capital: float, benchmark_tickers: list[str] = None):
        self.initial_capital = initial_capital
        self.benchmarks = benchmark_tickers or ["SPY", "QQQ"]
        self._strategy_values: dict[str, list[float]] = {}
        self._strategy_dates: list[str] = []
        self._portfolio_values: list[float] = [initial_capital]
        self._dates: list[str] = []

    def update(self, date: str, portfolio_value: float, strategy_values: dict[str, float]) -> None:
        self._dates.append(date)
        self._portfolio_values.append(portfolio_value)
        for strat, val in strategy_values.items():
            if strat not in self._strategy_values:
                self._strategy_values[strat] = []
            self._strategy_values[strat].append(val)

    def _returns(self, values: list[float]) -> pd.Series:
        s = pd.Series(values)
        return s.pct_change().dropna()

    def sharpe(self, values: list[float], rf: float = 0.04) -> float:
        r = self._returns(values)
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        excess = r - rf / 252
        return excess.mean() / excess.std() * np.sqrt(252)

    def max_drawdown(self, values: list[float]) -> float:
        s = pd.Series(values)
        roll_max = s.cummax()
        dd = (s - roll_max) / roll_max
        return dd.min()

    def total_return(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / values[0]

    def win_rate(self, values: list[float]) -> float:
        r = self._returns(values)
        if r.empty:
            return float("nan")
        return (r > 0).mean()

    def get_benchmark_return(self, ticker: str, days: int = None) -> Optional[float]:
        """Fetch benchmark return over same period."""
        try:
            from data.market_data import get_close
            n = days or len(self._dates) or 30
            df = get_close([ticker], days=n + 10)
            if df.empty:
                return None
            col = df.columns[0]
            return (df[col].iloc[-1] / df[col].iloc[0]) - 1
        except Exception:
            return None

    def print_dashboard(self) -> None:
        pv = self._portfolio_values[-1]
        port_ret = self.total_return(self._portfolio_values)
        port_sharpe = self.sharpe(self._portfolio_values)
        port_dd = self.max_drawdown(self._portfolio_values)
        port_wr = self.win_rate(self._portfolio_values)

        spy_ret = self.get_benchmark_return("SPY")
        qqq_ret = self.get_benchmark_return("QQQ")

        alpha_spy = (port_ret - spy_ret) if spy_ret is not None else float("nan")
        alpha_qqq = (port_ret - qqq_ret) if qqq_ret is not None else float("nan")

        print(f"\n{'='*52}")
        print(f"  QUANTBOT PERFORMANCE  {datetime.today().strftime('%Y-%m-%d')}")
        print(f"{'='*52}")
        print(f"  Portfolio : ${pv:>12,.2f}  ({port_ret:+.2%})")
        print(f"  vs SPY    : {spy_ret:+.2%}  Alpha: {alpha_spy:+.2%}" if spy_ret else "  vs SPY: n/a")
        print(f"  vs QQQ    : {qqq_ret:+.2%}  Alpha: {alpha_qqq:+.2%}" if qqq_ret else "  vs QQQ: n/a")
        print(f"  Sharpe    : {port_sharpe:.2f}")
        print(f"  Max DD    : {port_dd:.2%}")
        print(f"  Win Rate  : {port_wr:.1%}")

        if self._strategy_values:
            print(f"\n  --- Per-Strategy ---")
            for strat, vals in self._strategy_values.items():
                if len(vals) < 2:
                    continue
                ret = self.total_return(vals)
                sh = self.sharpe(vals)
                dd = self.max_drawdown(vals)
                print(f"  {strat:<28}  {ret:+.2%}  Sharpe={sh:.2f}  DD={dd:.2%}")

        print(f"{'='*52}\n")
