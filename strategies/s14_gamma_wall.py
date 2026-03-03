"""
S14 — Gamma Wall Breakout / Pinning

Alpha thesis:
  Options market makers (dealers) delta-hedge their books continuously.
  When dealers are net LONG gamma (they sold options to clients), they
  sell into rallies and buy into dips — SUPPRESSING volatility and
  pinning price near high-OI strikes ("gamma walls").

  When dealers are net SHORT gamma (bought options from clients), they
  must buy as price rises and sell as it falls — AMPLIFYING moves.

  GEX (Gamma Exposure) = sum over all strikes of:
    (call_OI - put_OI) × gamma × spot² × 100 × contract_size

  Positive GEX -> pinning, mean-revert trades
  Negative GEX -> trending, breakout trades
  Zero-crossing -> transition, expect vol expansion

Signal logic:
  1. Compute GEX from SPY options chain (yfinance, nearest 2 expiries)
  2. Compute "gamma walls" = strikes with >5% of total call OI
  3. If |GEX| > threshold AND spot near gamma wall (within 1%):
     -> Mean reversion signal: sell moves, expect pin
  4. If spot breaks through gamma wall with volume:
     -> Breakout signal: momentum trade
  5. Regime: +1 = long equities (pinning supports grind up), -1 = reduce

Novel twist: combine GEX with VIX term structure to detect when
gamma is about to flip negative (spike risk). Short UVXY when pinning,
go to SHY/GLD when GEX goes negative.

Rebalance: daily.
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from strategies.base import Strategy
from data.indicators import sma

logger = logging.getLogger(__name__)


def _black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    from math import log, sqrt, exp
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        gamma = exp(-0.5 * d1**2) / (S * sigma * sqrt(T) * (2 * np.pi)**0.5)
        return gamma
    except Exception:
        return 0.0


def _compute_gex(ticker: str = "SPY") -> dict:
    """
    Compute net gamma exposure from options chain.
    Returns dict with gex, spot, gamma_walls, regime.
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    result = {"gex": 0.0, "spot": 0.0, "gamma_walls": [], "regime": "neutral"}

    try:
        t = yf.Ticker(ticker)
        spot = t.fast_info.get("lastPrice") or t.fast_info.get("last_price")
        if not spot:
            hist = t.history(period="2d")
            if hist.empty:
                return result
            spot = float(hist["Close"].iloc[-1])
        result["spot"] = float(spot)

        expiries = t.options[:3]  # nearest 3 expiries
        if not expiries:
            return result

        today = datetime.today()
        total_gex = 0.0
        strike_gex: dict[float, float] = {}

        for expiry in expiries:
            try:
                exp_date = pd.to_datetime(expiry)
                T = max((exp_date - today).days / 365.0, 1 / 365.0)
                chain = t.option_chain(expiry)

                for df, sign in [(chain.calls, 1), (chain.puts, -1)]:
                    if df is None or df.empty:
                        continue
                    for _, row in df.iterrows():
                        K = float(row.get("strike", 0))
                        if K <= 0:
                            continue
                        iv = float(row.get("impliedVolatility", 0.20) or 0.20)
                        oi = float(row.get("openInterest", 0) or 0)
                        if oi <= 0:
                            continue

                        gamma = _black_scholes_gamma(spot, K, T, 0.05, max(iv, 0.05))
                        gex_contribution = sign * gamma * oi * spot**2 * 100

                        total_gex += gex_contribution
                        strike_gex[K] = strike_gex.get(K, 0) + gex_contribution

            except Exception as exc:
                logger.debug("[GammaWall] expiry %s failed: %s", expiry, exc)

        result["gex"] = total_gex

        # Find gamma walls: strikes where |GEX| > 5% of total
        if strike_gex and total_gex != 0:
            max_gex = max(abs(v) for v in strike_gex.values())
            walls = [k for k, v in strike_gex.items() if abs(v) > 0.05 * max_gex]
            result["gamma_walls"] = sorted(walls)

        # Regime
        if total_gex > 1e8:
            result["regime"] = "pinning"    # dealers long gamma, suppress vol
        elif total_gex < -1e8:
            result["regime"] = "trending"   # dealers short gamma, amplify moves
        else:
            result["regime"] = "neutral"

    except Exception as exc:
        logger.warning("[GammaWall] GEX compute failed: %s", exc)

    return result


class GammaWall(Strategy):
    name = "s14_gamma_wall"
    rebalance_freq = "daily"
    max_positions = 3

    # ETF proxies per regime
    PINNING_ALLOC  = {"SPY": 0.70, "QQQ": 0.30}          # suppressed vol -> long equities
    TRENDING_ALLOC = {"UVXY": 0.30, "SPY": 0.40, "GLD": 0.30}  # amplified moves -> vol hedge
    NEUTRAL_ALLOC  = {"SPY": 0.60, "SHY": 0.40}

    def get_universe(self) -> list[str]:
        return ["SPY", "QQQ", "GLD", "SHY", "UVXY", "TLT"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < 20:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # For backtesting: use VIX as GEX proxy (VIX low = positive GEX likely)
        vix = kwargs.get("vix")
        vix9d = kwargs.get("vix9d")

        for i, date in enumerate(prices.index):
            if i < 20:
                continue

            # Live mode: compute actual GEX on the last date
            if date == prices.index[-1]:
                gex_data = _compute_gex("SPY")
                regime = gex_data["regime"]
                logger.info("[GammaWall] GEX=%.2e regime=%s spot=%.2f",
                            gex_data["gex"], regime, gex_data["spot"])
            else:
                # Backtest proxy: use VIX + VIX9D term structure
                if vix is not None and vix9d is not None:
                    v = vix.get(date, vix.iloc[min(i, len(vix)-1)] if len(vix) > 0 else 20)
                    v9 = vix9d.get(date, vix9d.iloc[min(i, len(vix9d)-1)] if len(vix9d) > 0 else 20)
                    if v < 15:
                        regime = "pinning"
                    elif v > 25 or (hasattr(v9, '__float__') and float(v9) > float(v) * 1.1):
                        regime = "trending"
                    else:
                        regime = "neutral"
                elif vix is not None:
                    try:
                        v = float(vix.iloc[min(i, len(vix)-1)])
                        regime = "pinning" if v < 15 else ("trending" if v > 25 else "neutral")
                    except Exception:
                        regime = "neutral"
                else:
                    regime = "neutral"

            alloc = {
                "pinning": self.PINNING_ALLOC,
                "trending": self.TRENDING_ALLOC,
                "neutral": self.NEUTRAL_ALLOC,
            }[regime]

            signals.iloc[i, :] = 0.0
            for t, w in alloc.items():
                if t in signals.columns:
                    signals.at[date, t] = w

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        return {t: float(w) for t, w in signals[signals > 0].items()}
