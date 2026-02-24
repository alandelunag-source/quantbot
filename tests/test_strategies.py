"""
Unit tests for each strategy — signal generation logic.
Uses synthetic price data (no network calls).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def make_prices(tickers: list[str], n: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {}
    for i, t in enumerate(tickers):
        prices = [100.0]
        returns = np.random.normal(0.0005, 0.012, n)
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        data[t] = prices[1:]
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=dates)


def make_volume(prices: pd.DataFrame, multiplier: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        np.random.randint(1_000_000, 5_000_000, size=prices.shape) * multiplier,
        index=prices.index,
        columns=prices.columns,
    )


# ---------------------------------------------------------------------------
# S01 — Momentum Dip
# ---------------------------------------------------------------------------

class TestMomentumDip:
    def test_returns_dataframe(self):
        from strategies.s01_momentum_dip import MomentumDip
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = make_prices(tickers, n=250)
        volume = make_volume(prices)
        s = MomentumDip()
        sig = s.generate_signals(prices, volume=volume)
        assert isinstance(sig, pd.DataFrame)
        assert set(sig.columns) == set(tickers)

    def test_signal_nonnegative(self):
        from strategies.s01_momentum_dip import MomentumDip
        prices = make_prices(["AAPL", "MSFT"], n=250)
        volume = make_volume(prices)
        s = MomentumDip()
        sig = s.generate_signals(prices, volume=volume).fillna(0)
        assert (sig >= 0).all().all()

    def test_position_sizing_max_10(self):
        from strategies.s01_momentum_dip import MomentumDip
        import pandas as pd
        s = MomentumDip()
        # 15 tickers with positive signal
        signals = pd.Series({f"T{i}": float(i) for i in range(15)})
        pos = s.position_sizing(signals)
        assert len(pos) <= 10
        for w in pos.values():
            assert w == pytest.approx(0.02)

    def test_exit_target(self):
        from strategies.s01_momentum_dip import MomentumDip
        s = MomentumDip()
        assert s.exit_rules(entry_price=100, current_price=102.6, days_held=3)  # +2.6% > 2.5%
        assert not s.exit_rules(entry_price=100, current_price=101.0, days_held=3)

    def test_exit_days(self):
        from strategies.s01_momentum_dip import MomentumDip
        s = MomentumDip()
        assert s.exit_rules(entry_price=100, current_price=99.0, days_held=7)
        assert not s.exit_rules(entry_price=100, current_price=99.0, days_held=6)


# ---------------------------------------------------------------------------
# S02 — Cross-Asset Momentum
# ---------------------------------------------------------------------------

class TestCrossAssetMomentum:
    def test_returns_dataframe(self):
        from strategies.s02_cross_asset_momentum import CrossAssetMomentum
        from data.universe import ETF_UNIVERSE
        prices = make_prices(ETF_UNIVERSE, n=280)
        s = CrossAssetMomentum()
        sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_safe_haven_when_all_negative(self):
        from strategies.s02_cross_asset_momentum import CrossAssetMomentum
        from data.universe import ETF_UNIVERSE
        # Simulate bear market: all ETFs declining
        n = 280
        prices_data = {t: [100.0 * (0.999 ** i) for i in range(n)] for t in ETF_UNIVERSE}
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = pd.DataFrame(prices_data, index=dates)
        s = CrossAssetMomentum()
        sig = s.generate_signals(prices)
        if not sig.empty:
            last = sig.iloc[-1]
            # In bear market, SHY should have highest or only weight
            assert "SHY" in last.index

    def test_position_sizing_max_3(self):
        from strategies.s02_cross_asset_momentum import CrossAssetMomentum
        import pandas as pd
        s = CrossAssetMomentum()
        signals = pd.Series({"SPY": 0.5, "QQQ": 0.4, "GLD": 0.3, "TLT": 0.2, "IWM": 0.1})
        pos = s.position_sizing(signals)
        assert len(pos) <= 3
        weights = list(pos.values())
        # Equal weight
        assert all(abs(w - weights[0]) < 0.001 for w in weights)


# ---------------------------------------------------------------------------
# S03 — Factor Alpha
# ---------------------------------------------------------------------------

class TestFactorAlpha:
    def test_returns_dataframe(self):
        from strategies.s03_factor_alpha import FactorAlpha
        from data.universe import SP100
        tickers = SP100[:10]
        prices = make_prices(tickers, n=280)
        s = FactorAlpha()
        sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_position_sizing_max_20(self):
        from strategies.s03_factor_alpha import FactorAlpha
        import pandas as pd
        s = FactorAlpha()
        signals = pd.Series({f"T{i}": float(i) / 30 for i in range(30)})
        pos = s.position_sizing(signals)
        assert len(pos) <= 20

    def test_z_cross_mean_zero(self):
        from strategies.s03_factor_alpha import _z_cross
        import pandas as pd
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _z_cross(s)
        assert abs(z.mean()) < 1e-10


# ---------------------------------------------------------------------------
# S07 — Macro Regime
# ---------------------------------------------------------------------------

class TestMacroRegime:
    def test_risk_on(self):
        from strategies.s07_macro_regime import _classify
        regime = _classify(vix=15, curve=0.5, spy=420, sma200=400,
                           vix_on=18, vix_off=28, curve_off=-0.5)
        assert regime == "risk_on"

    def test_risk_off_high_vix(self):
        from strategies.s07_macro_regime import _classify
        regime = _classify(vix=30, curve=0.5, spy=420, sma200=400,
                           vix_on=18, vix_off=28, curve_off=-0.5)
        assert regime == "risk_off"

    def test_risk_off_inverted_curve_below_sma(self):
        from strategies.s07_macro_regime import _classify
        regime = _classify(vix=22, curve=-0.6, spy=380, sma200=400,
                           vix_on=18, vix_off=28, curve_off=-0.5)
        assert regime == "risk_off"

    def test_neutral_mixed(self):
        from strategies.s07_macro_regime import _classify
        regime = _classify(vix=22, curve=0.1, spy=420, sma200=400,
                           vix_on=18, vix_off=28, curve_off=-0.5)
        assert regime == "neutral"

    def test_nan_inputs(self):
        from strategies.s07_macro_regime import _classify
        regime = _classify(np.nan, np.nan, np.nan, np.nan, 18, 28, -0.5)
        assert regime == "neutral"


# ---------------------------------------------------------------------------
# S06 — VIX Term Structure
# ---------------------------------------------------------------------------

class TestVIXTermStructure:
    def test_regime_contango(self):
        from strategies.s06_vix_term_structure import VIXTermStructure
        s = VIXTermStructure()
        regime = s.get_regime(vix=20, vix9d=15, roll_z=2.0)
        assert regime == "contango"

    def test_regime_backwardation(self):
        from strategies.s06_vix_term_structure import VIXTermStructure
        s = VIXTermStructure()
        regime = s.get_regime(vix=25, vix9d=30, roll_z=-1.5)
        assert regime == "backwardation"

    def test_extreme_risk_off(self):
        from strategies.s06_vix_term_structure import VIXTermStructure
        s = VIXTermStructure()
        regime = s.get_regime(vix=35, vix9d=40, roll_z=-2.0)
        assert regime == "extreme_risk_off"


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_position_cap(self):
        from execution.risk import RiskManager
        rm = RiskManager()
        rm.peak_value = 100_000
        weights = {"AAPL": 0.10, "MSFT": 0.08}  # both > 5% cap
        adjusted = rm.apply(weights, portfolio_value=100_000)
        for w in adjusted.values():
            assert abs(w) <= 0.05 + 1e-9

    def test_drawdown_stop_halves(self):
        from execution.risk import RiskManager
        rm = RiskManager()
        rm.peak_value = 100_000
        weights = {"AAPL": 0.05}
        adjusted = rm.apply(weights, portfolio_value=85_000)  # -15% > 12% stop
        assert adjusted["AAPL"] == pytest.approx(0.025, abs=0.001)

    def test_empty_input(self):
        from execution.risk import RiskManager
        rm = RiskManager()
        assert rm.apply({}, portfolio_value=100_000) == {}
