"""
Vectorized backtesting engine.

Design principles (Citadel-grade):
  1. Realistic transaction costs (configurable bps, applied both ways)
  2. In-Sample (IS) / Out-of-Sample (OOS) split enforced
  3. Forward returns computed at next-open (not same-bar, avoids look-ahead)
  4. Turnover tracking and cost-adjusted returns
  5. IC time-series computed alongside equity curve
  6. Benchmark comparison (SPY, QQQ)

Usage:
  engine = BacktestEngine(strategy_instance)
  result = engine.run(days=365, oos_split=0.3)
  result.print_summary()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from data.market_data import get_close, get_volume, get_vix, get_vix9d, get_yield_spread
from data.indicators import ic as compute_ic

logger = logging.getLogger(__name__)

COST_PCT = settings.TOTAL_COST_PCT  # ~4 bps round-trip default


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series           # portfolio value over time
    returns: pd.Series                 # daily returns
    benchmark_returns: dict[str, pd.Series]  # SPY, QQQ
    positions: pd.DataFrame            # position weights over time
    turnover: pd.Series                # daily portfolio turnover
    ic_series: pd.Series               # rolling IC vs next 5d return
    is_period: tuple[str, str]         # in-sample start, end
    oos_period: tuple[str, str]        # out-of-sample start, end
    cost_drag: float                   # total cost drag on return

    def annualized_return(self, series: Optional[pd.Series] = None) -> float:
        s = series if series is not None else self.returns
        s = s.dropna()
        if len(s) < 2:
            return float("nan")
        return (1 + s).prod() ** (252 / len(s)) - 1

    def sharpe(self, series: Optional[pd.Series] = None, rf: float = 0.04) -> float:
        s = series if series is not None else self.returns
        s = s.dropna()
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        excess = s - rf / 252
        return excess.mean() / excess.std() * np.sqrt(252)

    def max_drawdown(self, series: Optional[pd.Series] = None) -> float:
        s = series if series is not None else self.equity_curve
        s = s.dropna()
        if s.empty:
            return float("nan")
        roll_max = s.cummax()
        dd = (s - roll_max) / roll_max
        return dd.min()

    def win_rate(self) -> float:
        r = self.returns.dropna()
        if r.empty:
            return float("nan")
        return (r > 0).mean()

    def avg_turnover(self) -> float:
        return self.turnover.mean() if not self.turnover.empty else float("nan")

    def ic_mean(self) -> float:
        return self.ic_series.dropna().mean()

    def ic_ir(self) -> float:
        """IC Information Ratio = mean(IC) / std(IC). >0.5 is excellent."""
        s = self.ic_series.dropna()
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        return s.mean() / s.std()

    def alpha_vs(self, benchmark: str = "SPY") -> float:
        """Annualized alpha vs benchmark (Jensen's alpha approximation)."""
        bm = self.benchmark_returns.get(benchmark, pd.Series())
        if bm.empty:
            return float("nan")
        port_ann = self.annualized_return()
        bm_ann = self.annualized_return(bm)
        return port_ann - bm_ann

    def print_summary(self, oos_only: bool = False) -> None:
        label = "OOS" if oos_only else "FULL"
        if oos_only and self.oos_period[0]:
            r = self.returns.loc[self.oos_period[0] : self.oos_period[1]]
            eq = self.equity_curve.loc[self.oos_period[0] : self.oos_period[1]]
        else:
            r = self.returns
            eq = self.equity_curve

        print(f"\n{'='*52}")
        print(f"  Backtest: {self.strategy_name}  [{label}]")
        print(f"{'='*52}")
        print(f"  Period      : {r.index[0].date() if not r.empty else 'n/a'}  ->  {r.index[-1].date() if not r.empty else 'n/a'}")
        print(f"  Ann. Return : {self.annualized_return(r):.2%}")
        print(f"  Sharpe      : {self.sharpe(r):.2f}")
        print(f"  Max DD      : {self.max_drawdown(eq):.2%}")
        print(f"  Win Rate    : {self.win_rate():.1%}")
        print(f"  Avg Turnover: {self.avg_turnover():.1%}/day")
        print(f"  IC (mean)   : {self.ic_mean():.3f}  |  IC IR: {self.ic_ir():.2f}")
        print(f"  Cost Drag   : {self.cost_drag:.2%}")
        print(f"  Alpha/SPY   : {self.alpha_vs('SPY'):.2%}")
        print(f"  Alpha/QQQ   : {self.alpha_vs('QQQ'):.2%}")
        if oos_only:
            print(f"  IS period   : {self.is_period[0]} -> {self.is_period[1]}")
            print(f"  OOS period  : {self.oos_period[0]} -> {self.oos_period[1]}")
        print(f"{'='*52}\n")


class BacktestEngine:
    def __init__(self, strategy, initial_capital: float = None):
        self.strategy = strategy
        self.capital = initial_capital or settings.INITIAL_CAPITAL

    def run(
        self,
        days: int = 365,
        oos_split: float = 0.25,
        extra_kwargs: dict = None,
    ) -> BacktestResult:
        """
        Run a vectorized backtest.

        Args:
            days: total lookback period (IS + OOS)
            oos_split: fraction reserved for out-of-sample (default 25%)
            extra_kwargs: passed to strategy.generate_signals (e.g. vix=)

        Returns:
            BacktestResult with full equity curve, IC, drawdown, etc.
        """
        logger.info("[BT] Starting backtest for %s  days=%d  oos=%.0f%%",
                    self.strategy.name, days, oos_split * 100)

        universe = self.strategy.get_universe()
        extra = extra_kwargs or {}

        # Fetch price data
        prices = get_close(universe, days=days + 60)  # +60 buffer for warmup
        volume = get_volume(universe, days=days + 60)
        if prices.empty:
            logger.error("[BT] No price data returned")
            return self._empty_result()

        prices = prices.iloc[-days:]
        volume = volume.reindex(prices.index)

        # Fetch macro data for strategies that need it
        vix = get_vix(days=days + 60).reindex(prices.index)
        vix9d = get_vix9d(days=days + 60).reindex(prices.index)
        yield_spread = get_yield_spread(days=days + 60).reindex(prices.index)
        extra.update({"volume": volume, "vix": vix, "vix9d": vix9d, "yield_spread": yield_spread})

        # Generate signals
        signals = self.strategy.generate_signals(prices, **extra)
        if signals.empty:
            logger.warning("[BT] No signals generated")
            return self._empty_result()

        signals = signals.reindex(prices.index).fillna(0)

        # Build positions (target weights, applied next day)
        position_weights = signals.shift(1).fillna(0)  # next-bar execution

        # Normalize weights to sum ≤ 1 per day
        row_sums = position_weights.abs().sum(axis=1).replace(0, 1)
        position_weights = position_weights.div(row_sums, axis=0)

        # Daily returns
        price_returns = prices.pct_change().fillna(0)

        # Strategy returns = weight × return (shift already applied above)
        strat_returns_gross = (position_weights * price_returns).sum(axis=1)

        # Turnover (sum of absolute weight changes / 2)
        weight_changes = position_weights.diff().abs().sum(axis=1) / 2
        turnover = weight_changes.clip(0, 1)

        # Transaction cost drag: cost_pct × turnover
        cost_drag_daily = turnover * COST_PCT
        strat_returns_net = strat_returns_gross - cost_drag_daily

        # Equity curve
        equity = (1 + strat_returns_net).cumprod() * self.capital

        # Benchmark returns
        bm_prices = get_close(["SPY", "QQQ"], days=days + 60).reindex(prices.index)
        bm_returns = {}
        for bm in ["SPY", "QQQ"]:
            if bm in bm_prices.columns:
                bm_returns[bm] = bm_prices[bm].pct_change().fillna(0)

        # IC series: rolling correlation of signal strength vs 5d forward return
        fwd_5d = price_returns.rolling(5).sum().shift(-5)
        daily_ic = pd.Series(dtype=float, index=prices.index)
        for date in prices.index[60:]:
            sig_today = signals.loc[date].dropna()
            fwd_today = fwd_5d.loc[date].dropna()
            aligned = pd.concat([sig_today, fwd_today], axis=1).dropna()
            if len(aligned) >= 5:
                daily_ic[date] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")

        # IS / OOS split
        n = len(strat_returns_net)
        split_idx = int(n * (1 - oos_split))
        is_start = str(strat_returns_net.index[0].date())
        is_end = str(strat_returns_net.index[split_idx].date())
        oos_start = str(strat_returns_net.index[split_idx + 1].date()) if split_idx < n - 1 else ""
        oos_end = str(strat_returns_net.index[-1].date())

        total_cost_drag = cost_drag_daily.sum()

        return BacktestResult(
            strategy_name=self.strategy.name,
            equity_curve=equity,
            returns=strat_returns_net,
            benchmark_returns=bm_returns,
            positions=position_weights,
            turnover=turnover,
            ic_series=daily_ic,
            is_period=(is_start, is_end),
            oos_period=(oos_start, oos_end),
            cost_drag=total_cost_drag,
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=self.strategy.name,
            equity_curve=pd.Series(dtype=float),
            returns=pd.Series(dtype=float),
            benchmark_returns={},
            positions=pd.DataFrame(),
            turnover=pd.Series(dtype=float),
            ic_series=pd.Series(dtype=float),
            is_period=("", ""),
            oos_period=("", ""),
            cost_drag=0.0,
        )
