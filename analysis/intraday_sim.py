"""
Simulate what would have happened if open (9:35) and midday (12:30) sessions
ran on time today. Fetches 1-minute intraday data and checks exit rules.
"""
import json
import glob
import sys
from datetime import datetime, date
import pandas as pd
import yfinance as yf

TODAY = date.today().isoformat()

# Exit rule parameters per strategy (from strategy files)
STOP_PARAMS = {
    "s01_momentum_dip":        {"stop": 0.05,  "profit": 0.10, "days": 30},
    "s02_cross_asset_mom":     {"stop": 0.10,  "profit": 0.25, "days": 90},
    "s03_factor_alpha":        {"stop": 0.07,  "profit": 0.20, "days": 60},
    "s04_earnings_drift":      {"stop": 0.05,  "profit": 0.08, "days": 10},
    "s05_short_term_reversal": {"stop": 0.05,  "profit": 0.05, "days": 10},
    "s06_vix_term_structure":  {"stop": 0.10,  "profit": 0.20, "days": 30},
    "s07_macro_regime":        {"stop": 0.10,  "profit": 0.20, "days": 60},
    "s09_dollar_carry":        {"stop": 0.05,  "profit": 0.10, "days": 30},
    "s10_vol_surface":         {"stop": 0.08,  "profit": 0.15, "days": 30},
    "s11_congressional":       {"stop": 0.05,  "profit": 0.15, "days": 60},
    "s12_index_inclusion":     {"stop": 0.08,  "profit": 0.20, "days": 45},
    "s14_gamma_wall":          {"stop": 0.05,  "profit": 0.10, "days": 30},
    "s15_short_flow":          {"stop": 0.07,  "profit": 0.15, "days": 21},
    "s16_overnight_carry":     {"stop": 0.05,  "profit": 0.15, "days": 30},
    "s17_panic_reversal":      {"stop": 0.04,  "profit": 0.06, "days": 5},
}

def days_held(entry_date: str) -> int:
    try:
        ed = datetime.strptime(entry_date, "%Y-%m-%d").date()
        return (date.today() - ed).days
    except Exception:
        return 0

def check_exit(strategy: str, entry_price: float, price: float, entry_date: str) -> str | None:
    params = STOP_PARAMS.get(strategy, {"stop": 0.05, "profit": 0.15, "days": 60})
    ret = (price - entry_price) / entry_price
    dh = days_held(entry_date)
    if ret <= -params["stop"]:
        return f"STOP-LOSS ({ret:+.1%})"
    if ret >= params["profit"]:
        return f"PROFIT-TARGET ({ret:+.1%})"
    if dh >= params["days"]:
        return f"TIME-STOP ({dh}d)"
    return None

# ---- Load all positions from state files ----
positions: dict[str, dict] = {}  # ticker -> {strategy, entry_price, entry_date}
for f in sorted(glob.glob("state/*_state.json")):
    strat = f.replace("state/", "").replace("_state.json", "")
    with open(f) as fh:
        s = json.load(fh)
    for ticker, pos in s.get("positions", {}).items():
        positions.setdefault(ticker, []).append({
            "strategy": strat,
            "entry_price": pos["entry_price"],
            "entry_date": pos["entry_date"],
        })

if not positions:
    print("No open positions found.")
    sys.exit(0)

tickers = list(positions.keys())
print(f"Checking {len(tickers)} tickers: {tickers}\n")

# ---- Fetch today's 1-minute data ----
print("Fetching 1-minute intraday data...")
raw = yf.download(tickers, period="1d", interval="1m", progress=False, auto_adjust=True)
if raw.empty:
    print("No intraday data returned.")
    sys.exit(1)

close_data = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
close_data.index = close_data.index.tz_convert("America/New_York")

# ---- Get prices at target times ----
def price_at(df: pd.DataFrame, target_time: str) -> pd.Series:
    """Get the last 1-min close at or before target_time (HH:MM)."""
    h, m = map(int, target_time.split(":"))
    mask = (df.index.hour < h) | ((df.index.hour == h) & (df.index.minute <= m))
    subset = df[mask]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[-1]

prices_935  = price_at(close_data, "9:35")
prices_1230 = price_at(close_data, "12:30")
prices_eod  = close_data.iloc[-1]

print(f"{'Strategy':<28} {'Ticker':<6} {'Entry':>8} {'@9:35':>8} {'@12:30':>8} {'@EOD':>8}  {'9:35 Exit?':<22} {'12:30 Exit?'}")
print("-" * 115)

should_have_fired = []

for ticker, pos_list in sorted(positions.items()):
    for pos in pos_list:
        ep = pos["entry_price"]
        ed = pos["entry_date"]
        strat = pos["strategy"]

        p935  = prices_935.get(ticker, float("nan"))
        p1230 = prices_1230.get(ticker, float("nan"))
        peod  = prices_eod.get(ticker, float("nan"))

        exit_935  = check_exit(strat, ep, p935,  ed) if not pd.isna(p935)  else None
        exit_1230 = check_exit(strat, ep, p1230, ed) if not pd.isna(p1230) else None

        flag = ""
        if exit_935 or exit_1230:
            flag = " <-- SHOULD HAVE FIRED"
            should_have_fired.append({
                "strategy": strat, "ticker": ticker,
                "entry_price": ep, "entry_date": ed,
                "exit_935": exit_935, "exit_1230": exit_1230,
                "p935": p935, "p1230": p1230, "peod": peod,
            })

        print(f"{strat:<28} {ticker:<6} {ep:>8.2f} {p935:>8.2f} {p1230:>8.2f} {peod:>8.2f}  "
              f"{str(exit_935 or ''):22} {str(exit_1230 or '')}{flag}")

print(f"\n{'='*60}")
print(f"Positions that SHOULD have exited earlier: {len(should_have_fired)}")
for x in should_have_fired:
    print(f"  {x['strategy']} / {x['ticker']}: 9:35={x['exit_935'] or '-'}, 12:30={x['exit_1230'] or '-'}")
