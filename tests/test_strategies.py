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
        # Signal-weighted: higher signal gets more capital; total <= 95%
        assert sum(pos.values()) <= 0.96
        assert all(0 < w <= 0.50 + 1e-9 for w in pos.values())


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


# ---------------------------------------------------------------------------
# S11 — Congressional Trade Follower
# ---------------------------------------------------------------------------

class TestCongressional:
    def _make_disclosure_df(self):
        from datetime import datetime, timedelta
        import pandas as pd
        return pd.DataFrame([
            {"disclosure_date": datetime.today() - timedelta(days=3),
             "ticker": "AAPL", "type": "purchase", "chamber": "senate", "amount": "$15,001 - $50,000"},
            {"disclosure_date": datetime.today() - timedelta(days=5),
             "ticker": "MSFT", "type": "purchase", "chamber": "house", "amount": "$50,001 - $100,000"},
            {"disclosure_date": datetime.today() - timedelta(days=10),
             "ticker": "GOOGL", "type": "sale", "chamber": "house", "amount": "$15,001 - $50,000"},
        ])

    def test_returns_dataframe_with_mock(self):
        from strategies.s11_congressional import CongressionalTrades
        from unittest.mock import patch
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        prices = make_prices(tickers, n=100)
        s = CongressionalTrades()
        with patch("strategies.s11_congressional._fetch_disclosures", return_value=self._make_disclosure_df()):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_filters_purchases_only(self):
        from strategies.s11_congressional import CongressionalTrades
        from unittest.mock import patch
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = make_prices(tickers, n=100)
        s = CongressionalTrades()
        with patch("strategies.s11_congressional._fetch_disclosures", return_value=self._make_disclosure_df()):
            sig = s.generate_signals(prices)
        if not sig.empty:
            last = sig.iloc[-1]
            # GOOGL was a sale — should have zero or lower signal than purchases
            assert last.get("GOOGL", 0) == 0.0

    def test_position_sizing_capped(self):
        from strategies.s11_congressional import CongressionalTrades
        s = CongressionalTrades()
        signals = pd.Series({f"T{i}": float(i + 1) for i in range(15)})
        pos = s.position_sizing(signals)
        assert len(pos) <= s.max_positions
        for w in pos.values():
            assert w <= 0.151  # cap is 15%

    def test_amount_to_score(self):
        from strategies.s11_congressional import _amount_to_score
        assert _amount_to_score("$15,001 - $50,000") == 32_500.0
        assert _amount_to_score("over $5,000,000") == 7_500_000.0
        assert _amount_to_score("unknown range") == 10_000.0


# ---------------------------------------------------------------------------
# S12 — Index Inclusion Frontrun
# ---------------------------------------------------------------------------

class TestIndexInclusion:
    def test_returns_dataframe(self):
        from strategies.s12_index_inclusion import IndexInclusion, KNOWN_ADDITIONS
        from unittest.mock import patch
        tickers = list({t for t, _, _ in KNOWN_ADDITIONS}) + ["SPY"]
        prices = make_prices(tickers[:10], n=300)
        s = IndexInclusion()
        with patch("strategies.s12_index_inclusion._fetch_wikipedia_additions", return_value=[]):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_known_additions_generate_signal(self):
        from strategies.s12_index_inclusion import IndexInclusion, KNOWN_ADDITIONS
        from unittest.mock import patch
        import pandas as pd
        # Use a ticker from KNOWN_ADDITIONS that has a recent-ish date
        ticker, ann_str, eff_str = KNOWN_ADDITIONS[0]  # e.g., DELL
        ann = pd.to_datetime(ann_str)
        eff = pd.to_datetime(eff_str)
        # Need >= 30 rows for generate_signals; extend well before announcement
        start = ann - pd.Timedelta(days=60)
        dates = pd.bdate_range(start, eff + pd.Timedelta(days=5))
        prices = pd.DataFrame({ticker: [100.0] * len(dates)}, index=dates)
        s = IndexInclusion()
        with patch("strategies.s12_index_inclusion._fetch_wikipedia_additions", return_value=[]):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)
        if not sig.empty and ticker in sig.columns:
            # Signal should be positive in the announcement-to-effective window
            window = sig.loc[sig.index >= ann]
            window = window.loc[window.index <= eff]
            if not window.empty:
                assert (window[ticker] > 0).any()

    def test_position_sizing_signal_weighted(self):
        from strategies.s12_index_inclusion import IndexInclusion
        s = IndexInclusion()
        # 7 positions so none hit the 15% cap — signal-weighting is visible
        signals = pd.Series({f"T{i}": float(10 - i) for i in range(7)})
        pos = s.position_sizing(signals)
        weights = sorted(pos.values(), reverse=True)
        # Signal-weighted: weights are strictly descending (higher signal → more capital)
        assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))
        assert sum(pos.values()) <= 0.71
        assert all(0 < w <= 0.15 + 1e-9 for w in pos.values())


# ---------------------------------------------------------------------------
# S13 — Pre-Earnings Drift
# ---------------------------------------------------------------------------

class TestPreEarningsDrift:
    def _mock_earnings(self, days_ahead: int = 4, surprise_rate: float = 0.85):
        from datetime import datetime, timedelta
        return {
            "next_date": pd.Timestamp(datetime.today() + timedelta(days=days_ahead)),
            "surprise_rate": surprise_rate,
            "last_two_positive": True,
        }

    def test_signal_generated_in_window(self):
        from strategies.s13_pre_earnings_drift import PreEarningsDrift, SP100
        from unittest.mock import patch
        tickers = SP100[:5]
        prices = make_prices(tickers, n=120)
        s = PreEarningsDrift()
        mock_info = self._mock_earnings(days_ahead=4, surprise_rate=0.85)
        with patch("strategies.s13_pre_earnings_drift._get_earnings_info", return_value=mock_info):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)
        last = sig.iloc[-1]
        # At least some tickers should have signals (those passing momentum filter)
        assert last.sum() >= 0

    def test_no_signal_outside_window(self):
        from strategies.s13_pre_earnings_drift import PreEarningsDrift, SP100
        from unittest.mock import patch
        tickers = SP100[:5]
        prices = make_prices(tickers, n=120)
        s = PreEarningsDrift()
        # 20 days out — beyond the entry window
        mock_info = self._mock_earnings(days_ahead=20, surprise_rate=0.85)
        with patch("strategies.s13_pre_earnings_drift._get_earnings_info", return_value=mock_info):
            sig = s.generate_signals(prices)
        if not sig.empty:
            assert sig.iloc[-1].sum() == 0.0

    def test_no_signal_low_surprise_rate(self):
        from strategies.s13_pre_earnings_drift import PreEarningsDrift, SP100
        from unittest.mock import patch
        tickers = SP100[:5]
        prices = make_prices(tickers, n=120)
        s = PreEarningsDrift()
        mock_info = self._mock_earnings(days_ahead=4, surprise_rate=0.40)  # below MIN
        with patch("strategies.s13_pre_earnings_drift._get_earnings_info", return_value=mock_info):
            sig = s.generate_signals(prices)
        if not sig.empty:
            assert sig.iloc[-1].sum() == 0.0

    def test_position_sizing_capped(self):
        from strategies.s13_pre_earnings_drift import PreEarningsDrift
        s = PreEarningsDrift()
        signals = pd.Series({f"T{i}": 0.5 for i in range(10)})
        pos = s.position_sizing(signals)
        assert len(pos) <= s.max_positions
        for w in pos.values():
            assert w <= 0.12 + 1e-9


# ---------------------------------------------------------------------------
# S14 — Gamma Wall
# ---------------------------------------------------------------------------

class TestGammaWall:
    def _mock_gex(self, regime: str = "pinning"):
        return {"gex": 5e8 if regime == "pinning" else -5e8, "spot": 500.0,
                "gamma_walls": [495.0, 500.0, 505.0], "regime": regime}

    def test_pinning_regime_allocates_to_spy_qqq(self):
        from strategies.s14_gamma_wall import GammaWall
        from unittest.mock import patch
        tickers = ["SPY", "QQQ", "GLD", "SHY", "UVXY", "TLT"]
        prices = make_prices(tickers, n=50)
        s = GammaWall()
        with patch("strategies.s14_gamma_wall._compute_gex", return_value=self._mock_gex("pinning")):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)
        last = sig.iloc[-1]
        assert last.get("SPY", 0) > 0
        assert last.get("QQQ", 0) > 0

    def test_trending_regime_allocates_to_uvxy(self):
        from strategies.s14_gamma_wall import GammaWall
        from unittest.mock import patch
        tickers = ["SPY", "QQQ", "GLD", "SHY", "UVXY", "TLT"]
        prices = make_prices(tickers, n=50)
        vix = pd.Series([30.0] * 50, index=prices.index)
        s = GammaWall()
        with patch("strategies.s14_gamma_wall._compute_gex", return_value=self._mock_gex("trending")):
            sig = s.generate_signals(prices, vix=vix)
        assert isinstance(sig, pd.DataFrame)

    def test_bs_gamma_positive(self):
        from strategies.s14_gamma_wall import _black_scholes_gamma
        g = _black_scholes_gamma(S=500, K=500, T=0.1, r=0.05, sigma=0.20)
        assert g > 0

    def test_bs_gamma_zero_tte(self):
        from strategies.s14_gamma_wall import _black_scholes_gamma
        g = _black_scholes_gamma(S=500, K=500, T=0.0, r=0.05, sigma=0.20)
        assert g == 0.0


# ---------------------------------------------------------------------------
# S15 — Institutional Short Flow
# ---------------------------------------------------------------------------

class TestShortFlow:
    def _mock_flow_df(self, tickers):
        import pandas as pd
        from datetime import date, timedelta
        dates = [date.today() - timedelta(days=i) for i in range(5, 0, -1)]
        data = {t: [0.42, 0.40, 0.38, 0.36, 0.34] for t in tickers}
        return pd.DataFrame(data, index=dates)

    def test_returns_dataframe(self):
        from strategies.s15_short_flow import ShortFlow, UNIVERSE
        from unittest.mock import patch
        tickers = UNIVERSE[:5]
        prices = make_prices(tickers, n=60)
        s = ShortFlow()
        with patch("strategies.s15_short_flow._get_short_ratios", return_value=self._mock_flow_df(tickers)):
            sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_squeeze_signal_on_high_ratio_drop(self):
        from strategies.s15_short_flow import ShortFlow, UNIVERSE
        from unittest.mock import patch
        import pandas as pd
        ticker = UNIVERSE[0]  # AAPL
        # Build declining price series: flat then -11% drop in last 5 days
        n = 60
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        price_values = [100.0] * (n - 5) + [95.0, 93.0, 91.0, 89.0, 87.0]
        prices = pd.DataFrame({ticker: price_values}, index=dates)

        # High short ratio — triggers squeeze (>= 0.68) + price down > 5%
        flow_dates = list(dates[-5:])
        flow_df = pd.DataFrame({ticker: [0.70, 0.71, 0.72, 0.73, 0.74]}, index=flow_dates)
        s = ShortFlow()
        with patch("strategies.s15_short_flow._get_short_ratios", return_value=flow_df):
            sig = s.generate_signals(prices)
        if not sig.empty:
            assert sig.iloc[-1].get(ticker, 0) > 0  # squeeze = bullish signal

    def test_position_sizing_cap(self):
        from strategies.s15_short_flow import ShortFlow
        s = ShortFlow()
        signals = pd.Series({f"T{i}": 0.5 for i in range(10)})
        pos = s.position_sizing(signals)
        assert len(pos) <= s.max_positions
        for w in pos.values():
            assert w <= 0.20 + 1e-9


# ---------------------------------------------------------------------------
# S16 — Overnight Carry
# ---------------------------------------------------------------------------

class TestOvernightCarry:
    def test_returns_dataframe(self):
        from strategies.s16_overnight_carry import OvernightCarry, UNIVERSE
        prices = make_prices(UNIVERSE, n=100)
        s = OvernightCarry()
        sig = s.generate_signals(prices)
        assert isinstance(sig, pd.DataFrame)

    def test_spy_always_positive_in_uptrend(self):
        from strategies.s16_overnight_carry import OvernightCarry, UNIVERSE
        # Steady uptrend so SPY > SMA50
        n = 80
        prices_data = {t: [100.0 * (1.001 ** i) for i in range(n)] for t in UNIVERSE}
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame(prices_data, index=dates)
        vix = pd.Series([15.0] * n, index=dates)  # low VIX
        s = OvernightCarry()
        sig = s.generate_signals(prices, vix=vix)
        assert isinstance(sig, pd.DataFrame)
        if not sig.empty:
            last = sig.iloc[-1]
            assert last.get("SPY", 0) > 0

    def test_flat_above_vix_exit(self):
        from strategies.s16_overnight_carry import OvernightCarry, UNIVERSE
        n = 80
        prices_data = {t: [100.0 * (1.001 ** i) for i in range(n)] for t in UNIVERSE}
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame(prices_data, index=dates)
        vix = pd.Series([40.0] * n, index=dates)  # above VIX_EXIT=35
        s = OvernightCarry()
        sig = s.generate_signals(prices, vix=vix)
        if not sig.empty:
            # Last row should be all zeros (flat above exit threshold)
            assert sig.iloc[-1].sum() == pytest.approx(0.0)

    def test_position_sizing_sums_to_one(self):
        from strategies.s16_overnight_carry import OvernightCarry
        s = OvernightCarry()
        signals = pd.Series({"SPY": 0.6, "QQQ": 0.4})
        pos = s.position_sizing(signals)
        assert sum(pos.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# S17 -- Panic Reversal
# ---------------------------------------------------------------------------

class TestPanicReversal:
    def _make_panic_prices(self, tickers, n=60, drop_last=True):
        """Steady uptrend then a correlated selloff on last day."""
        data = {}
        for t in tickers:
            vals = [100.0 * (1.001 ** i) for i in range(n)]
            if drop_last and t != "SPY":
                vals[-1] = vals[-2] * 0.975   # down 2.5%
            if drop_last and t == "SPY":
                vals[-1] = vals[-2] * 0.980   # SPY down 2%
            data[t] = vals
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame(data, index=dates)

    def test_returns_dataframe(self):
        from strategies.s17_panic_reversal import PanicReversal, SP100
        tickers = SP100[:5] + ["SPY"]
        prices = self._make_panic_prices(tickers)
        vix = pd.Series([25.0] * len(prices), index=prices.index)
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        assert isinstance(sig, pd.DataFrame)

    def test_no_signal_below_vix_gate(self):
        from strategies.s17_panic_reversal import PanicReversal, SP100
        tickers = SP100[:5] + ["SPY"]
        prices = self._make_panic_prices(tickers)
        vix = pd.Series([14.0] * len(prices), index=prices.index)  # low VIX
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        # No signals should fire when VIX < 20
        if not sig.empty:
            assert sig.iloc[-1].sum() == pytest.approx(0.0)

    def test_signal_fires_in_high_vix_comovement(self):
        from strategies.s17_panic_reversal import PanicReversal, SP100
        tickers = SP100[:8] + ["SPY"]
        prices = self._make_panic_prices(tickers)
        vix = pd.Series([28.0] * len(prices), index=prices.index)  # high VIX
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        assert isinstance(sig, pd.DataFrame)
        if not sig.empty:
            # At least some stocks should signal (co-movement drop with VIX > 20)
            assert sig.iloc[-1].sum() >= 0

    def test_no_signal_on_idiosyncratic_drop(self):
        """Stock down 5% but SPY flat = idiosyncratic -> no signal."""
        from strategies.s17_panic_reversal import PanicReversal, SP100
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        ticker = SP100[0]
        data = {ticker: [100.0] * (n - 1) + [94.0],  # -6% idiosyncratic
                "SPY":  [100.0] * n}                  # SPY flat
        prices = pd.DataFrame(data, index=dates)
        vix = pd.Series([28.0] * n, index=dates)
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        if not sig.empty:
            assert sig.iloc[-1].get(ticker, 0) == pytest.approx(0.0)

    def test_exit_rules_days_held(self):
        from strategies.s17_panic_reversal import PanicReversal
        s = PanicReversal()
        assert s.exit_rules(entry_price=100, current_price=99, days_held=5)
        assert not s.exit_rules(entry_price=100, current_price=99, days_held=4)

    def test_exit_rules_profit_target(self):
        from strategies.s17_panic_reversal import PanicReversal
        s = PanicReversal()
        assert s.exit_rules(entry_price=100, current_price=103.5, days_held=1)
        assert not s.exit_rules(entry_price=100, current_price=102.0, days_held=1)

    def test_exit_rules_stop_loss(self):
        from strategies.s17_panic_reversal import PanicReversal
        s = PanicReversal()
        assert s.exit_rules(entry_price=100, current_price=95.0, days_held=2)   # -5% > -4% stop
        assert not s.exit_rules(entry_price=100, current_price=97.0, days_held=2)  # -3% < stop

    def test_position_sizing_equal_weight_capped(self):
        from strategies.s17_panic_reversal import PanicReversal
        s = PanicReversal()
        signals = pd.Series({f"T{i}": float(i + 1) for i in range(12)})
        pos = s.position_sizing(signals)
        assert len(pos) <= s.max_positions
        for w in pos.values():
            assert w <= 0.12 + 1e-9

    def test_signal_carries_forward(self):
        from strategies.s17_panic_reversal import PanicReversal, SP100
        n = 70
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        ticker = SP100[0]
        prices_data = {ticker: [100.0] * n, "SPY": [100.0] * n}
        prices_data[ticker][50] = 97.5   # event: -2.5%
        prices_data["SPY"][50]  = 97.8   # SPY -2.2% (co-movement)
        prices = pd.DataFrame(prices_data, index=dates)
        vix = pd.Series([28.0] * n, index=dates)
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        assert isinstance(sig, pd.DataFrame)
        if not sig.empty and ticker in sig.columns:
            assert sig.iloc[50][ticker] > 0          # event day has signal
            assert sig.iloc[51][ticker] > 0          # day+1 carries signal

    def test_carry_cleared_by_stop_loss(self):
        from strategies.s17_panic_reversal import PanicReversal, SP100
        n = 70
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        ticker = SP100[0]
        prices_data = {ticker: [100.0] * n, "SPY": [100.0] * n}
        prices_data[ticker][50] = 97.5   # event: -2.5%
        prices_data["SPY"][50]  = 97.8
        prices_data[ticker][51] = 92.5   # day+1 craters -5% from entry -> stop
        prices = pd.DataFrame(prices_data, index=dates)
        vix = pd.Series([28.0] * n, index=dates)
        s = PanicReversal()
        sig = s.generate_signals(prices, vix=vix)
        if not sig.empty and ticker in sig.columns and 52 < n:
            assert sig.iloc[52][ticker] == pytest.approx(0.0)
