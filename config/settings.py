"""Quantbot settings — all env-overridable."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Capital
INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "100000"))

# Risk
MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
MAX_STRATEGY_PCT: float = float(os.getenv("MAX_STRATEGY_PCT", "0.20"))
DRAWDOWN_STOP_PCT: float = float(os.getenv("DRAWDOWN_STOP_PCT", "0.12"))
CORRELATION_CAP: float = float(os.getenv("CORRELATION_CAP", "0.85"))

# Transaction cost assumptions (backtesting realism)
TRANSACTION_COST_BPS: float = float(os.getenv("TRANSACTION_COST_BPS", "3"))
SLIPPAGE_BPS: float = float(os.getenv("SLIPPAGE_BPS", "1"))
TOTAL_COST_PCT: float = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

# Benchmark tickers
BENCHMARKS = ["SPY", "QQQ"]

# Strategy allocations (independent — each can be run solo)
STRATEGY_ALLOCATIONS: dict[str, float] = {
    "s01_momentum_dip":       0.10,
    "s02_cross_asset_mom":    0.15,
    "s03_factor_alpha":       0.15,
    "s04_earnings_drift":     0.10,
    "s05_short_term_reversal":0.10,
    "s06_vix_term_structure": 0.10,
    "s07_macro_regime":       0.15,
    "s08_covered_calls":      0.00,  # overlay on s03, not additive capital
    "s09_dollar_carry":       0.10,
    "s10_vol_surface":        0.05,
}
