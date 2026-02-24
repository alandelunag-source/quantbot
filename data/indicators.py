"""
Technical indicators — all vectorized via pandas/numpy.
No TA-lib dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # When loss = 0 (all gains), RSI = 100
    rs = gain / loss.where(loss != 0, other=np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(loss != 0, other=100.0)  # loss=0 → RSI=100
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def realized_vol(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Annualized realized volatility."""
    rv = returns.rolling(window).std()
    return rv * np.sqrt(252) if annualize else rv


def momentum(prices: pd.Series, lookback: int, skip: int = 1) -> pd.Series:
    """
    Standard momentum: return over [lookback, skip] months.
    skip=1 excludes the most recent month (avoids short-term reversal bias).
    """
    return prices.pct_change(lookback) - prices.pct_change(skip)


def z_score(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of a signal."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume / N-day average volume."""
    return volume / volume.rolling(window).mean()


def bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(prices, window)
    std = prices.rolling(window).std()
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * std, "lower": mid - n_std * std})


def ic(signal: pd.Series, forward_return: pd.Series) -> float:
    """
    Information Coefficient (rank correlation of signal vs next-period return).
    Citadel-style signal quality metric.
    """
    aligned = pd.concat([signal, forward_return], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")


def rolling_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Rolling cross-sectional IC (mean rank-corr across assets per day).
    Standard Citadel/AQR signal evaluation tool.
    """
    dates = signal.index.intersection(forward_returns.index)
    ics = []
    for i, date in enumerate(dates):
        if i < window:
            ics.append(float("nan"))
            continue
        window_dates = dates[i - window : i]
        sig_w = signal.loc[window_dates].iloc[-1]
        ret_w = forward_returns.loc[window_dates].iloc[-1]
        aligned = pd.concat([sig_w, ret_w], axis=1).dropna()
        if len(aligned) < 5:
            ics.append(float("nan"))
        else:
            ics.append(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))
    return pd.Series(ics, index=dates)
