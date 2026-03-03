"""
S17 -- Market Panic Reversal

Alpha thesis (data-driven from analysis/loser_recovery.py, 2 years, 102 stocks):

  CONVENTIONAL WISDOM (WRONG):
    Buy stocks that crashed hard on massive volume with a hammer close.
    Result from data: avg -0.51% over 3 days, wr=44%. Negative edge.

  WHAT THE DATA ACTUALLY SHOWS:
    [1] VIX regime is the single biggest driver of reversal edge:
          VIX > 25 -> avg +1.98% over 3 days, wr=67%, t=8.51  <-- enormous
          VIX < 15 -> avg +0.14%, wr=52%  <-- near zero
    [2] Co-movement beats idiosyncratic:
          Stock fell WITH market (rel_loss -1% to 0%) -> +1.32%, wr=65%, t=7.56
          Stock fell MORE than market (idiosyncratic) -> +0.13%, wr=51%  <-- weak
    [3] Volume sweet spot is 1x-2x, NOT extreme:
          1x-2x vol -> +0.51%, wr=56%, t=6.94
          >3x vol   -> +0.08%, wr=46%  <-- capitulation is a continuation signal!
    [4] Holding 5 days (vs. 3) meaningfully improves edge: +0.82% vs +0.39%

  INTERPRETATION:
    During high-VIX risk-off episodes, indiscriminate selling hits ALL stocks --
    including quality large-caps that have no company-specific bad news.
    Forced deleveraging, risk-parity unwinds, and ETF redemptions drive
    correlated liquidation regardless of fundamentals.

    These stocks revert because the selling pressure was systematic/mechanical
    rather than information-driven. The reversion happens as VIX mean-reverts.

    Idiosyncratic losers (company-specific drops) do NOT revert -- they may
    carry real information (earnings, guidance, legal) so we avoid them.

Signal logic:
  1. Universe: S&P 100 (liquid, high-quality, tight spreads)
  2. Daily: identify stocks where:
       a. Day return <= -2% (meaningful drop)
       b. Relative loss vs SPY in (-2%, 0%) -- fell WITH market, not alone
       c. VIX > 20 (elevated fear, indiscriminate selling regime)
       d. Volume ratio 0.7x - 2.5x (normal to elevated, not panic extreme)
  3. Score each candidate:
         score = |ret| * vix_factor * comovement_quality
       where vix_factor = (vix - 20) / 20 for vix > 20
             comovement_quality = 1 - abs(rel_loss) / 0.02  (closer to 0 = better)
  4. Rank, go long top 8
  5. Hold 5 trading days, then exit

Rebalance: daily (new signals added, old positions aged out).
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from strategies.base import Strategy
from data.indicators import sma

logger = logging.getLogger(__name__)

SP100 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","JPM",
    "AVGO","XOM","TSLA","UNH","V","JNJ","WMT","MA","PG","HD","MRK","CVX",
    "ABBV","BAC","KO","AMD","PEP","CSCO","TMO","ACN","MCD","WFC","ADBE","CRM",
    "ABT","GE","LIN","DHR","CAT","AXP","MS","GS","BLK","SPGI","ISRG","NOW",
    "UBER","INTU","RTX","DE","HON","AMGN","PLD","TJX","SYK","BSX","REGN",
    "VRTX","CB","ETN","ELV","ADP","PANW","MMC","SO","SCHW","ZTS","CI",
    "BDX","MO","DUK","CME","AON","MDLZ","KLAC","SLB","ANET","PH","TGT",
    "EQIX","CEG","KMB","D","WM","APH","SNPS","CDNS","FCX","NSC","FDX",
    "EMR","MCO","USB","MSI","MCHP","ADI","ORCL","NFLX","DIS","PYPL","IBM","TXN",
]


class PanicReversal(Strategy):
    """
    Buy quality stocks that dropped with the market (not idiosyncratically)
    during elevated-VIX panic environments. Hold 5 days.

    Validated parameters from loser_recovery.py analysis (2 years, 102 stocks):
      - VIX gate >= 20: captures the high-VIX regime with wr=67%, avg=+1.98%
      - rel_loss in (-2%, 0%): co-movement filter, avoids idiosyncratic drops
      - vol_ratio in (0.7, 2.5): elevated but not capitulation extreme
      - hold_days = 5: optimal from forward-return analysis
    """
    name = "s17_panic_reversal"
    rebalance_freq = "daily"
    max_positions = 8
    HOLD_DAYS = 5

    # Signal thresholds (from empirical analysis)
    MIN_DROP      = 0.020   # stock must be down at least 2%
    VIX_GATE      = 20.0    # only trade in elevated VIX environment
    MAX_REL_LOSS  = 0.020   # rel. to SPY: must be within 2% (co-movement)
    VOL_MIN       = 0.70    # not a dead market
    VOL_MAX       = 2.50    # not panic capitulation

    # Exit rules (grid-optimised: stop_loss_grid.py, 500d, 222 OOS trades)
    STOP_LOSS     = 0.03    # cut at -3% from entry (hard stop)
    PROFIT_TARGET = 0.025   # take profit at +2.5%

    def get_universe(self) -> list[str]:
        from data.universe import get_large_cap_universe
        return get_large_cap_universe() + ["SPY"]   # SPY needed for market-relative calc

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if prices.empty or len(prices) < 25:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        vix = kwargs.get("vix")
        volume = kwargs.get("volume")

        if "SPY" not in prices.columns:
            return signals

        spy_ret = prices["SPY"].pct_change()

        for i, date in enumerate(prices.index):
            if i < 21:
                continue

            # VIX gate -- only signal during elevated fear
            vix_level = 20.0
            if vix is not None and len(vix) > 0:
                try:
                    vix_level = float(vix.iloc[min(i, len(vix) - 1)])
                except Exception:
                    pass

            if vix_level < self.VIX_GATE:
                continue   # low fear = no indiscriminate selling = no edge

            spy_day_ret = float(spy_ret.iloc[i]) if not pd.isna(spy_ret.iloc[i]) else 0.0
            vix_factor = (vix_level - 20.0) / 20.0   # 0 at vix=20, 1 at vix=40

            for ticker in prices.columns:
                if ticker == "SPY":
                    continue

                col = prices[ticker]
                if pd.isna(col.iloc[i]) or col.iloc[i] < 10:
                    continue

                # Day return
                if i == 0 or pd.isna(col.iloc[i - 1]) or col.iloc[i - 1] == 0:
                    continue
                day_ret = float(col.iloc[i] / col.iloc[i - 1] - 1)

                if day_ret > -self.MIN_DROP:
                    continue   # not a significant loser

                # Relative loss vs SPY (co-movement check)
                rel_loss = day_ret - spy_day_ret

                # We want co-movement: stock fell WITH market (rel_loss near 0)
                # Not idiosyncratic drop (rel_loss << 0)
                if rel_loss < -self.MAX_REL_LOSS:
                    continue   # stock-specific problem, avoid

                # Volume filter
                if volume is not None and ticker in volume.columns:
                    vol_20d = volume[ticker].iloc[max(0, i - 20):i].mean()
                    if vol_20d > 0:
                        vol_ratio = float(volume[ticker].iloc[i]) / vol_20d
                        if not (self.VOL_MIN <= vol_ratio <= self.VOL_MAX):
                            continue

                # Score: loss magnitude x VIX intensity x co-movement quality
                comovement_q = 1.0 - abs(rel_loss) / self.MAX_REL_LOSS  # 1 = perfect, 0 = edge
                score = abs(day_ret) * (1.0 + vix_factor) * comovement_q
                signals.at[date, ticker] = score

        # ── Carry signals forward for HOLD_DAYS ──────────────────────────────
        # A position entered on day T must stay alive on T+1 … T+HOLD_DAYS-1.
        # We carry the original score forward, but clear it early on:
        #   - stop-loss  : price dropped > STOP_LOSS from entry
        #   - profit target: price rose  > PROFIT_TARGET from entry
        # A stronger NEW signal on a later day overwrites the carry.
        for ticker in list(signals.columns):
            if ticker == "SPY":
                continue
            col_sig   = signals[ticker]
            col_price = prices[ticker] if ticker in prices.columns else None
            event_dates = col_sig[col_sig > 0].index.tolist()

            for event_date in event_dates:
                event_idx   = prices.index.get_loc(event_date)
                entry_price = float(prices[ticker].iloc[event_idx]) if col_price is not None else 0.0
                base_score  = float(col_sig[event_date])

                for d in range(1, self.HOLD_DAYS):
                    carry_idx = event_idx + d
                    if carry_idx >= len(prices.index):
                        break
                    carry_date  = prices.index[carry_idx]
                    carry_price = float(prices[ticker].iloc[carry_idx]) if col_price is not None else entry_price

                    # Stop-loss: exit early, no carry for remaining days
                    if entry_price > 0 and (carry_price / entry_price - 1) < -self.STOP_LOSS:
                        break

                    # Profit target: exit early
                    if entry_price > 0 and (carry_price / entry_price - 1) > self.PROFIT_TARGET:
                        break

                    # Only carry if no stronger signal already on this day
                    if signals.at[carry_date, ticker] < base_score:
                        signals.at[carry_date, ticker] = base_score

        return signals

    def position_sizing(self, signals: pd.Series, prices: pd.DataFrame = None) -> dict[str, float]:
        """Signal-weighted panic reversals: 72% deployed, 12% per-name cap (reserve for daily entries)."""
        longs = signals[signals > 0].nlargest(self.max_positions)
        if longs.empty:
            return {}
        return self._sized_weights(longs, prices=prices, max_deploy=0.72, max_weight=0.12)

    def exit_rules(self, entry_price: float, current_price: float, days_held: int) -> bool:
        """
        Three exit conditions (mirrored in generate_signals carry logic):
          1. Time stop  : held >= HOLD_DAYS (5 days) — exit regardless of P&L
          2. Profit take: up >= PROFIT_TARGET (+2.5%)
          3. Stop-loss  : down >= STOP_LOSS  (-3%)
        """
        if days_held >= self.HOLD_DAYS:
            return True
        if current_price >= entry_price * (1 + self.PROFIT_TARGET):
            return True
        if current_price <= entry_price * (1 - self.STOP_LOSS):
            return True
        return False
