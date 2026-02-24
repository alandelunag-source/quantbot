#!/usr/bin/env python3
"""
Quantbot — systematic equity + macro trading bot.

Each strategy is INDEPENDENT. Run any one standalone, or run all together.

Usage:
  python main.py scan                        # Signal scan all strategies
  python main.py scan --strategy s01         # Single strategy scan
  python main.py backtest --strategy s01 --days 365  # Backtest one strategy
  python main.py backtest --all --days 365   # Backtest all, compare
  python main.py paper                       # Start paper trading (all)
  python main.py paper --strategy s03        # Paper trade one strategy
  python main.py status                      # Current positions + performance

Strategy codes: s01 s02 s03 s04 s05 s06 s07 s09 s10
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

from monitoring.performance import PerformanceTracker
from config import settings


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_strategy(code: str):
    from strategies import ALL_STRATEGIES
    cls = ALL_STRATEGIES.get(code)
    if cls is None:
        print(f"Unknown strategy: {code}")
        print(f"Available: {', '.join(ALL_STRATEGIES.keys())}")
        sys.exit(1)
    return cls()


def _get_current_prices(universe: list[str]) -> dict[str, float]:
    from data.market_data import get_close
    prices = get_close(universe, days=5)
    if prices.empty:
        return {}
    return prices.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    from strategies import ALL_STRATEGIES
    from data.market_data import get_close, get_volume, get_vix, get_yield_spread

    codes = [args.strategy] if args.strategy else list(ALL_STRATEGIES.keys())

    for code in codes:
        strategy = _get_strategy(code)
        print(f"\n[{strategy.name}] Fetching data...")

        universe = strategy.get_universe()
        prices = get_close(universe, days=300)
        volume = get_volume(universe, days=300)
        vix = get_vix(days=300)
        ys = get_yield_spread(days=300)

        if prices.empty:
            print(f"  No price data for {strategy.name}")
            continue

        signals_df = strategy.generate_signals(
            prices, volume=volume, vix=vix, yield_spread=ys
        )
        if signals_df.empty:
            print(f"  No signals generated.")
            continue

        latest = signals_df.iloc[-1]
        positions = strategy.position_sizing(latest)

        print(f"  [{datetime.today().date()}] {strategy.name}")
        if positions:
            for ticker, weight in sorted(positions.items(), key=lambda x: -abs(x[1])):
                direction = "LONG " if weight > 0 else "SHORT"
                print(f"    {direction}  {ticker:<6}  {weight:+.1%}")
        else:
            print("    No actionable signals today.")


def cmd_backtest(args: argparse.Namespace) -> None:
    from strategies import ALL_STRATEGIES
    from backtesting.engine import BacktestEngine

    codes = list(ALL_STRATEGIES.keys()) if args.all else [args.strategy or "s01"]
    days = args.days or 365
    oos = args.oos or 0.25

    results = []
    for code in codes:
        strategy = _get_strategy(code)
        print(f"\n[BACKTEST] {strategy.name}  days={days}  OOS={oos:.0%}...")
        engine = BacktestEngine(strategy)
        result = engine.run(days=days, oos_split=oos)
        result.print_summary(oos_only=True)
        results.append(result)

    if len(results) > 1:
        _print_comparison(results)


def _print_comparison(results: list) -> None:
    print("\n" + "="*70)
    print(f"  {'Strategy':<28}  {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'IC':>6}")
    print("="*70)
    for r in sorted(results, key=lambda x: x.sharpe(), reverse=True):
        print(f"  {r.strategy_name:<28}  "
              f"{r.annualized_return():>8.2%}  "
              f"{r.sharpe():>7.2f}  "
              f"{r.max_drawdown():>7.2%}  "
              f"{r.ic_mean():>6.3f}")
    print("="*70 + "\n")


def cmd_paper(args: argparse.Namespace) -> None:
    from strategies import ALL_STRATEGIES
    from backtesting.forward_test import ForwardTest

    codes = [args.strategy] if args.strategy else list(ALL_STRATEGIES.keys())
    strategies = [_get_strategy(c) for c in codes]
    forward_tests = [ForwardTest(s) for s in strategies]

    print(f"Starting paper trading for: {[s.name for s in strategies]}")
    print("Press Ctrl+C to stop.\n")

    tracker = PerformanceTracker(
        initial_capital=settings.INITIAL_CAPITAL * len(strategies)
    )

    try:
        while True:
            today = datetime.today().strftime("%Y-%m-%d")
            strategy_values = {}

            for ft in forward_tests:
                universe = ft.strategy.get_universe()
                prices = _get_current_prices(universe)
                summary = ft.update(prices)
                ft.print_status()
                strategy_values[ft.strategy.name] = summary["portfolio_value"]

            agg_value = sum(strategy_values.values())
            tracker.update(today, agg_value, strategy_values)
            tracker.print_dashboard()

            # Sleep until next market open (simplified: just wait 60s in paper mode)
            interval = 60
            print(f"Next update in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading. Final status:")
        tracker.print_dashboard()


def cmd_status(args: argparse.Namespace) -> None:
    from strategies import ALL_STRATEGIES
    from backtesting.forward_test import ForwardTest

    codes = [args.strategy] if args.strategy else list(ALL_STRATEGIES.keys())
    for code in codes:
        strategy = _get_strategy(code)
        ft = ForwardTest(strategy)
        ft.print_status()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quantbot",
        description="Systematic trading bot: 10 independent strategies, backtest + paper trading",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # scan
    scan_p = sub.add_parser("scan", help="One-shot signal scan")
    scan_p.add_argument("--strategy", help="Strategy code (e.g. s01). Default: all")

    # backtest
    bt_p = sub.add_parser("backtest", help="Backtest one or all strategies")
    bt_p.add_argument("--strategy", help="Strategy code (e.g. s03)")
    bt_p.add_argument("--all", action="store_true", help="Backtest all strategies")
    bt_p.add_argument("--days", type=int, default=365)
    bt_p.add_argument("--oos", type=float, default=0.25, help="OOS fraction (default 0.25)")

    # paper
    paper_p = sub.add_parser("paper", help="Paper trading (live forward test)")
    paper_p.add_argument("--strategy", help="Single strategy (default: all)")

    # status
    status_p = sub.add_parser("status", help="Current positions + P&L")
    status_p.add_argument("--strategy", help="Single strategy (default: all)")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
