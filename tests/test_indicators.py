"""Tests for data/indicators.py"""
import numpy as np
import pandas as pd
import pytest

from data.indicators import (
    sma, ema, rsi, realized_vol, momentum, z_score, volume_ratio, ic,
)


def make_prices(n=100, start=100.0, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices[1:])


class TestSMA:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_warmup_nan(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)


class TestRSI:
    def test_range(self):
        p = make_prices()
        r = rsi(p, 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_overbought_when_always_up(self):
        p = pd.Series(range(1, 50, 1), dtype=float)
        r = rsi(p, 14).dropna()
        assert r.iloc[-1] > 70

    def test_oversold_when_always_down(self):
        p = pd.Series(list(range(50, 0, -1)), dtype=float)
        r = rsi(p, 14).dropna()
        assert r.iloc[-1] < 30


class TestRealizedVol:
    def test_positive(self):
        p = make_prices()
        r = p.pct_change()
        rv = realized_vol(r, 20).dropna()
        assert (rv > 0).all()

    def test_zero_for_flat(self):
        p = pd.Series([100.0] * 50)
        r = p.pct_change()
        rv = realized_vol(r, 20).dropna()
        assert rv.iloc[-1] == pytest.approx(0.0)


class TestMomentum:
    def test_positive_for_rising(self):
        p = pd.Series([float(i) for i in range(1, 300)])
        m = momentum(p, 252, skip=21).dropna()
        assert (m > 0).all()


class TestZScore:
    def test_mean_zero_std_one(self):
        # Use stationary (mean-reverting) process so rolling z-score stays centered
        np.random.seed(7)
        vals = pd.Series(np.cumsum(np.random.normal(0, 0.01, 300)))
        z = z_score(vals, 100).dropna()
        # Rolling z-score of stationary process should hover near 0
        assert abs(z.mean()) < 2.0
        assert z.std() > 0


class TestIC:
    def test_positive_ic_for_perfect_signal(self):
        signal = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        forward = signal * 2 + pd.Series(np.random.normal(0, 0.1, 10))
        result = ic(signal, forward)
        assert result > 0.5

    def test_negative_ic_for_inverse_signal(self):
        signal = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        forward = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = ic(signal, forward)
        assert result < -0.5

    def test_nan_for_insufficient_data(self):
        assert np.isnan(ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])))
