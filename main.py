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
    from backtesting.forward_test import ForwardTest, is_trading_day

    # --strategies s09,s02,s06 overrides --strategy
    if hasattr(args, "strategies") and args.strategies:
        codes = [c.strip() for c in args.strategies.split(",")]
    elif args.strategy:
        codes = [args.strategy]
    else:
        codes = list(ALL_STRATEGIES.keys())

    strategies = [_get_strategy(c) for c in codes]
    forward_tests = [ForwardTest(s) for s in strategies]

    tracker = PerformanceTracker(initial_capital=settings.INITIAL_CAPITAL * len(strategies))

    if args.once:
        # Single daily update — designed for Task Scheduler / cron
        _paper_run_once(forward_tests, tracker)
    else:
        # Daemon: sleeps and wakes up once per trading day at ~4pm
        print(f"Paper trading daemon started for: {[s.name for s in strategies]}")
        print("Runs once per trading day at ~16:00. Press Ctrl+C to stop.\n")
        try:
            while True:
                if is_trading_day():
                    now = datetime.now()
                    # Run between 16:00 and 16:30 (after market close)
                    if 16 <= now.hour < 17:
                        _paper_run_once(forward_tests, tracker)
                        # Sleep past the 16:00-16:30 window
                        time.sleep(3600)
                        continue
                time.sleep(300)  # check every 5 min
        except KeyboardInterrupt:
            print("\nStopping. Final status:")
            cmd_status(args)


def _paper_run_once(forward_tests, tracker) -> None:
    """Execute one daily paper trading update across all strategies."""
    from backtesting.forward_test import is_trading_day

    today = datetime.today().strftime("%Y-%m-%d")

    if not is_trading_day():
        print(f"[{today}] Not a trading day — skipping.")
        return

    print(f"\n{'='*56}")
    print(f"  PAPER TRADING UPDATE  {today}")
    print(f"{'='*56}")

    strategy_values = {}
    for ft in forward_tests:
        universe = ft.strategy.get_universe()
        prices = _get_current_prices(universe)
        if not prices:
            print(f"  [{ft.strategy.name}] No price data — skipping")
            continue
        summary = ft.update(prices)
        ft.print_status()
        strategy_values[ft.strategy.name] = summary["portfolio_value"]

    if strategy_values:
        agg_value = sum(strategy_values.values())
        tracker.update(today, agg_value, strategy_values)
        tracker.print_dashboard()

    # Append to CSV log for 1-month review
    _append_csv_log(today, strategy_values)


def _append_csv_log(today: str, strategy_values: dict) -> None:
    """Append one row per strategy to state/paper_log.csv for 1-month review."""
    from pathlib import Path
    log_path = Path("state/paper_log.csv")
    log_path.parent.mkdir(exist_ok=True)
    header_needed = not log_path.exists()
    with open(log_path, "a") as f:
        if header_needed:
            f.write("date,strategy,portfolio_value\n")
        for strat, pv in strategy_values.items():
            f.write(f"{today},{strat},{pv:.2f}\n")


def cmd_status(args: argparse.Namespace) -> None:
    from strategies import ALL_STRATEGIES
    from backtesting.forward_test import ForwardTest

    if hasattr(args, "strategies") and args.strategies:
        codes = [c.strip() for c in args.strategies.split(",")]
    elif args.strategy:
        codes = [args.strategy]
    else:
        codes = list(ALL_STRATEGIES.keys())

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
    paper_p.add_argument("--strategy", help="Single strategy code (e.g. s09)")
    paper_p.add_argument("--strategies", help="Comma-separated strategy codes (e.g. s09,s02,s06)")
    paper_p.add_argument("--once", action="store_true", help="Run one daily update and exit (for Task Scheduler)")

    # status
    status_p = sub.add_parser("status", help="Current positions + P&L")
    status_p.add_argument("--strategy", help="Single strategy (default: all)")
    status_p.add_argument("--strategies", help="Comma-separated strategy codes (e.g. s09,s02,s06)")

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
