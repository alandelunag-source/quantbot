"""
S09 — Dollar Carry & FX Momentum

Alpha thesis:
  The carry trade (borrow low-yield, invest high-yield) is one of the most
  documented anomalies in FX markets. Combined with FX momentum (trend
  following on exchange rates), it generates a Sharpe of ~0.6-0.9 historically.

  US-focused ETF implementation: uses UUP (USD), FXE (EUR), FXY (JPY),
  FXB (GBP), EWJ (Japan equities), EWG (Germany equities) as proxies.

Signal logic:
  1. Dollar strength signal: compare UUP 20d vs 60d momentum
  2. If dollar is strengthening (UUP 20d > 60d): go long UUP + DBC (commodities)
  3. If dollar is weakening: go long EFA (international) + EEM + GLD
  4. Carry component: 2Y Treasury yields vs 0 (if US yield curve positive → risk)
  5. Score = dollar_momentum × carry_signal × liquidity_regime

Novel twist: combine dollar direction with equity international momentum.
When USD weakens + EM momentum positive → double signal on EEM.
When USD strengthens + EM negative → skip or short EM.

Rebalance: monthly.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging

from strategies.base import Strategy
from data.indicators import momentum, sma

logger = logging.getLogger(__name__)

RISK_ON_ALLOC  = {"EFA": 0.30, "IEMG": 0.30, "GLD": 0.20, "DBC": 0.20}
RISK_OFF_ALLOC = {"UUP": 0.40, "DBC": 0.30, "SHY": 0.30}
NEUTRAL_ALLOC  = {"EFA": 0.25, "SPY": 0.25, "GLD": 0.25, "DBC": 0.25}


class DollarCarry(Strategy):
    name = "s09_dollar_carry"
    rebalance_freq = "monthly"
    max_positions = 4

    MOM_SHORT = 20     # short-term USD momentum window (trading days)
    MOM_LONG  = 60     # long-term USD momentum window

    def get_universe(self) -> list[str]:
        return ["UUP", "EFA", "IEMG", "GLD", "DBC", "SHY", "SPY"]

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < self.MOM_LONG + 5:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        monthly_idx = prices.resample("ME").last().index

        for date in monthly_idx:
            if date not in prices.index:
                idx = prices.index.searchsorted(date) - 1
                if idx < 0:
                    continue
                date = prices.index[idx]

            loc = prices.index.get_loc(date)
            if loc < self.MOM_LONG:
                continue

            # Dollar momentum
            if "UUP" in prices.columns:
                uup = prices["UUP"].iloc[:loc+1]
                uup_20d = uup.pct_change(self.MOM_SHORT).iloc[-1]
                uup_60d = uup.pct_change(self.MOM_LONG).iloc[-1]
                dollar_strengthening = (uup_20d > 0) and (uup_20d > uup_60d * 0.8)
            else:
                dollar_strengthening = False

            # Regime
            if dollar_strengthening:
                alloc = RISK_OFF_ALLOC
            else:
                # Check international momentum
                intf_ok = False
                if "EFA" in prices.columns:
                    efa_ret = prices["EFA"].pct_change(self.MOM_SHORT).iloc[loc]
                    intf_ok = efa_ret > 0

                alloc = RISK_ON_ALLOC if intf_ok else NEUTRAL_ALLOC

            next_loc = min(loc + 22, len(prices))
            for t, w in alloc.items():
                if t in signals.columns:
                    signals[t].iloc[loc:next_loc] = w

        return signals

    def position_sizing(self, signals: pd.Series) -> dict[str, float]:
        return {t: float(w) for t, w in signals[signals > 0].items()}
