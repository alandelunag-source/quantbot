"""
Loser Recovery Edge Analysis
=============================
Hypothesis: stocks that are significant single-day losers exhibit a
mean-reverting bounce over the following 1-5 days.

We segment the analysis by:
  1. Loss magnitude buckets (-1% to -2%, -2% to -3%, -3% to -5%, < -5%)
  2. Volume ratio (abnormal vs. normal volume  exhaustion signal)
  3. Intraday range position: where the close lands within the day's range
     close_position = (close - low) / (high - low)   1.0 = close at high
  4. Sector-relative loss (loss vs. SPY on the same day)
  5. VIX regime (low < 15, normal 15-25, elevated > 25)

Run: venv/Scripts/python.exe analysis/loser_recovery.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

#  Universe 
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","JNJ",
    "WMT","MA","PG","HD","MRK","CVX","ABBV","BAC","KO","AMD","PEP",
    "CSCO","TMO","ACN","MCD","WFC","ADBE","CRM","ABT","GE","LIN","DHR",
    "CAT","AXP","MS","GS","BLK","ISRG","NOW","UBER","INTU","RTX","HON",
    "AMGN","TJX","SYK","VRTX","CB","ETN","ADP","PANW","ZTS","KLAC",
    "ANET","SNPS","CDNS","ORCL","NFLX","DIS","IBM","TXN","QCOM","INTC",
    "MU","LRCX","MRVL","AMAT","TER","ONTO","ENPH","SEDG","FSLR","NKE",
    "SBUX","CMG","YUM","DPZ","MKC","GIS","CPB","HRL","SJM",
    "XOM","OXY","COP","HAL","SLB","BKR","DVN","MPC","VLO","PSX",
    "UNH","ELV","HUM","CVS","MCK","CI","MOH","CNC","HCA","DGX",
]

LOOKBACK_DAYS = 730   # 2 years
FORWARD_WINDOWS = [1, 2, 3, 5, 10]
MIN_PRICE = 10.0      # skip penny stocks
MIN_VOLUME = 500_000  # skip illiquid days


def fetch_data(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for each ticker. Returns {ticker: OHLCV DataFrame}."""
    print(f"Fetching {len(tickers)} tickers ({days} days)...")
    result = {}
    raw = yf.download(
        tickers,
        period=f"{days}d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return result
    for t in tickers:
        try:
            df = pd.DataFrame({
                "open":   raw["Open"][t],
                "high":   raw["High"][t],
                "low":    raw["Low"][t],
                "close":  raw["Close"][t],
                "volume": raw["Volume"][t],
            }).dropna()
            if len(df) > 100:
                result[t] = df
        except Exception:
            pass
    return result


def build_event_table(
    ohlcv: dict[str, pd.DataFrame],
    vix: pd.Series,
    spy_ret: pd.Series,
) -> pd.DataFrame:
    """
    For every (ticker, date) where the single-day return < -1%,
    compute features and forward returns.
    """
    rows = []
    max_fwd = max(FORWARD_WINDOWS)

    for ticker, df in ohlcv.items():
        if len(df) < 30:
            continue

        df = df.copy()
        df["ret_1d"] = df["close"].pct_change()
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["close_pos"] = (
            (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        )
        # Forward returns
        for n in FORWARD_WINDOWS:
            df[f"fwd_{n}d"] = df["close"].pct_change(n).shift(-n)

        df = df.dropna()

        for date, row in df.iterrows():
            ret = row["ret_1d"]
            if ret >= -0.01 or row["close"] < MIN_PRICE or row["volume"] < MIN_VOLUME:
                continue

            vix_val = float(vix.reindex([date], method="ffill").iloc[0]) if date in vix.index or True else 20.0
            spy_val = float(spy_ret.reindex([date], method="ffill").iloc[0]) if date in spy_ret.index or True else 0.0

            rows.append({
                "ticker":      ticker,
                "date":        date,
                "ret_1d":      ret,
                "vol_ratio":   row["vol_ratio"],
                "close_pos":   row["close_pos"],   # 0 = close at low, 1 = at high
                "vix":         vix_val,
                "spy_ret":     spy_val,
                "rel_loss":    ret - spy_val,       # loss vs. market
                **{f"fwd_{n}d": row[f"fwd_{n}d"] for n in FORWARD_WINDOWS},
            })

    return pd.DataFrame(rows)


def segment_analysis(events: pd.DataFrame) -> None:
    """Print mean forward returns and win rates across key segments."""

    def stats(df: pd.DataFrame, label: str, n: int = 3) -> None:
        if df.empty:
            return
        fwd = df[f"fwd_{n}d"]
        mean_ret = fwd.mean()
        win_rate = (fwd > 0).mean()
        t_stat   = fwd.mean() / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        print(f"  {label:<45}  n={len(df):>5}  "
              f"avg_fwd_{n}d={mean_ret:+.2%}  wr={win_rate:.0%}  t={t_stat:.2f}")

    print("\n" + "=" * 90)
    print("  LOSER RECOVERY ANALYSIS   forward returns after single-day decline")
    print("=" * 90)

    #  1. By loss magnitude 
    print("\n[1] BY LOSS MAGNITUDE  (3-day forward return)")
    buckets = [
        ("  -1% to -2%",  (-0.02, -0.01)),
        ("  -2% to -3%",  (-0.03, -0.02)),
        ("  -3% to -5%",  (-0.05, -0.03)),
        ("  < -5%",       (-1.00, -0.05)),
    ]
    for label, (lo, hi) in buckets:
        sub = events[(events["ret_1d"] >= lo) & (events["ret_1d"] < hi)]
        stats(sub, label)

    #  2. By forward window (best loss bucket: -3% to -5%) 
    print("\n[2] BY FORWARD WINDOW  (loss -3% to -5%)")
    core = events[(events["ret_1d"] >= -0.05) & (events["ret_1d"] < -0.03)]
    for n in FORWARD_WINDOWS:
        fwd = core[f"fwd_{n}d"]
        wr  = (fwd > 0).mean()
        t   = fwd.mean() / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        print(f"  {n}-day fwd: avg={fwd.mean():+.2%}  wr={wr:.0%}  t={t:.2f}  n={len(fwd)}")

    #  3. Volume exhaustion filter 
    print("\n[3] VOLUME EXHAUSTION  (loss < -2%, 3-day fwd)")
    sub = events[events["ret_1d"] < -0.02]
    stats(sub[sub["vol_ratio"] <  1.0], "  Normal volume  (ratio < 1x)")
    stats(sub[sub["vol_ratio"].between(1.0, 2.0)], "  Elevated volume (1x-2x)")
    stats(sub[sub["vol_ratio"] >  2.0], "  Exhaustion vol  (> 2x)")
    stats(sub[sub["vol_ratio"] >  3.0], "  Capitulation    (> 3x)")

    #  4. Intraday close position (hammer candle) 
    print("\n[4] INTRADAY CLOSE POSITION  (loss < -2%, 3-day fwd)")
    sub = events[events["ret_1d"] < -0.02]
    stats(sub[sub["close_pos"] <  0.30], "  Closed near low   (0-30% of range)")
    stats(sub[sub["close_pos"].between(0.30, 0.60)], "  Closed mid-range  (30-60%)")
    stats(sub[sub["close_pos"] >  0.60], "  Closed near high  (60-100%)  hammer")

    #  5. Combined: capitulation + hammer 
    print("\n[5] COMBINED SIGNAL  (loss < -3%, vol > 2x, close_pos > 0.5)")
    combined = events[
        (events["ret_1d"]   <  -0.03) &
        (events["vol_ratio"] >   2.0) &
        (events["close_pos"] >   0.5)
    ]
    print(f"  Events matching: {len(combined)}")
    for n in FORWARD_WINDOWS:
        fwd = combined[f"fwd_{n}d"]
        wr  = (fwd > 0).mean()
        t   = fwd.mean() / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        print(f"  {n}-day fwd: avg={fwd.mean():+.2%}  wr={wr:.0%}  t={t:.2f}")

    #  6. VIX regime 
    print("\n[6] VIX REGIME  (loss < -2%, 3-day fwd)")
    sub = events[events["ret_1d"] < -0.02]
    stats(sub[sub["vix"] <  15], "  Low VIX    (< 15)")
    stats(sub[sub["vix"].between(15, 25)], "  Normal VIX (15-25)")
    stats(sub[sub["vix"] >  25], "  High VIX   (> 25)")

    #  7. Sector-relative loss 
    print("\n[7] SECTOR-RELATIVE LOSS  (3-day fwd)")
    sub = events[events["ret_1d"] < -0.02]
    stats(sub[sub["rel_loss"] < -0.03], "  Underperformed market by > 3%   idiosyncratic")
    stats(sub[sub["rel_loss"].between(-0.03, -0.01)], "  Underperformed by 1-3%")
    stats(sub[sub["rel_loss"] > -0.01], "  In-line with market (co-movement)")

    #  8. Best composite signal 
    print("\n[8] BEST COMPOSITE SIGNAL  (rel_loss < -3%, vol > 1.5x, close_pos > 0.4, VIX < 25)")
    best = events[
        (events["rel_loss"]   <  -0.03) &
        (events["vol_ratio"]  >   1.5)  &
        (events["close_pos"]  >   0.4)  &
        (events["vix"]        <   25.0)
    ]
    print(f"  Events matching: {len(best)}")
    for n in FORWARD_WINDOWS:
        fwd = best[f"fwd_{n}d"]
        wr  = (fwd > 0).mean()
        t   = fwd.mean() / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        print(f"  {n}-day fwd: avg={fwd.mean():+.2%}  wr={wr:.0%}  t={t:.2f}")

    print("\n" + "=" * 90)


def main() -> None:
    # Fetch data
    ohlcv = fetch_data(UNIVERSE, LOOKBACK_DAYS)
    print(f"Loaded {len(ohlcv)} tickers")

    # VIX
    vix_raw = yf.download("^VIX", period=f"{LOOKBACK_DAYS}d", auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze() if not vix_raw.empty else pd.Series(dtype=float)

    # SPY daily return (market reference)
    spy_raw = yf.download("SPY", period=f"{LOOKBACK_DAYS}d", auto_adjust=True, progress=False)
    spy_ret = spy_raw["Close"].squeeze().pct_change() if not spy_raw.empty else pd.Series(dtype=float)

    # Build event table
    print("Building event table...")
    events = build_event_table(ohlcv, vix, spy_ret)
    print(f"Total loser events (ret < -1%): {len(events):,}")

    # Run analysis
    segment_analysis(events)

    # Save for strategy tuning
    out = Path(__file__).parent / "loser_events.csv"
    events.to_csv(out)
    print(f"\nEvent table saved -> {out}")


if __name__ == "__main__":
    main()
