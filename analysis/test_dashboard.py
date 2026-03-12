"""
Smoke-test every Plotly figure in dashboard.py outside of Streamlit.
Run: venv\Scripts\python.exe analysis\test_dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

STATE_DIR  = Path(__file__).parent.parent / "state"
START_CASH = 100_000

STRATEGIES = [
    "s01_momentum_dip","s02_cross_asset_mom","s03_factor_alpha",
    "s04_earnings_drift","s05_short_term_reversal","s06_vix_term_structure",
    "s07_macro_regime","s09_dollar_carry","s10_vol_surface",
    "s11_congressional","s12_index_inclusion","s13_pre_earnings_drift",
    "s14_gamma_wall","s15_short_flow","s16_overnight_carry","s17_panic_reversal",
]
SHORT = {
    "s01_momentum_dip":"Momentum Dip","s02_cross_asset_mom":"Cross-Asset Mom",
    "s03_factor_alpha":"Factor Alpha","s04_earnings_drift":"Earnings Drift",
    "s05_short_term_reversal":"Short-Term Rev","s06_vix_term_structure":"VIX Term Struct",
    "s07_macro_regime":"Macro Regime","s09_dollar_carry":"Dollar Carry",
    "s10_vol_surface":"Vol Surface","s11_congressional":"Congressional",
    "s12_index_inclusion":"Index Inclusion","s13_pre_earnings_drift":"Pre-Earnings",
    "s14_gamma_wall":"Gamma Wall","s15_short_flow":"Short Flow",
    "s16_overnight_carry":"Overnight Carry","s17_panic_reversal":"Panic Reversal",
}
C = {
    "bg":"#0f1117","card":"#1a1d27","border":"#2a2d3a","green":"#00d4aa",
    "red":"#ff4d6d","cyan":"#38bdf8","orange":"#fb923c","purple":"#a78bfa",
    "text":"#f1f5f9","muted":"#94a3b8","dimmed":"#475569",
}
CHART_COLORS = [
    "#38bdf8","#00d4aa","#a78bfa","#fb923c","#f472b6",
    "#34d399","#fbbf24","#60a5fa","#f87171","#4ade80",
    "#e879f9","#22d3ee","#fb7185","#a3e635","#c084fc","#f9a8d4",
]
PLOTLY_LAYOUT = dict(
    paper_bgcolor = C["bg"],
    plot_bgcolor  = C["card"],
    font          = dict(family="Inter, Arial, sans-serif", size=12, color=C["text"]),
    xaxis         = dict(gridcolor=C["border"], zeroline=False, tickcolor=C["muted"], linecolor=C["border"]),
    yaxis         = dict(gridcolor=C["border"], zeroline=False, tickcolor=C["muted"], linecolor=C["border"]),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
    hovermode     = "x unified",
    hoverlabel    = dict(bgcolor=C["card"], bordercolor=C["border"], font_color=C["text"]),
    margin        = dict(l=0, r=0, t=40, b=0),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kwargs})
    return fig

def ok(name):
    print(f"  [OK] {name}")

def fail(name, e):
    print(f"  [FAIL] {name}: {e}")
    raise

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading states...")
states = {}
for s in STRATEGIES:
    p = STATE_DIR / f"{s}_state.json"
    if p.exists():
        states[s] = json.load(open(p))
print(f"  {len(states)} states loaded")

print("Fetching SPY...")
raw   = yf.download("SPY", start="2026-02-24",
                    end=(datetime.today()+timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)
close = raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
sq = close.squeeze()
if not isinstance(sq, pd.Series):
    sq = pd.Series([float(sq)], index=raw.index)
spy_raw = sq.dropna()
print(f"  SPY: {spy_raw.values}  empty={spy_raw.empty}")

# ── Build summary ─────────────────────────────────────────────────────────────
print("Building summary...")
rows = []
for s, st_ in states.items():
    pv        = st_.get("portfolio_value", START_CASH)
    peak      = st_.get("peak_value", pv)
    daily     = st_.get("daily_log", [])
    positions = st_.get("positions", {})
    trades    = st_.get("trades", [])
    max_dd    = (pv / peak - 1) * 100 if peak > 0 else 0.0
    sharpe    = None
    if len(daily) > 2:
        rets   = pd.Series([e["pv"] for e in daily]).pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    spy_ret   = 0.0
    start_dt  = st_.get("start_date", "")[:10]
    if not spy_raw.empty and start_dt:
        spy_sub = spy_raw[spy_raw.index >= start_dt]
        if len(spy_sub) >= 2:
            spy_ret = (spy_sub.iloc[-1] / spy_sub.iloc[0] - 1) * 100
    total_ret = (pv / START_CASH - 1) * 100
    rows.append({
        "key":s, "Strategy":s, "Value":pv, "Return":total_ret,
        "Alpha":total_ret-spy_ret, "SPY Ret":spy_ret, "Max DD":max_dd,
        "Sharpe":sharpe, "Positions":len(positions), "Trades":len(trades),
    })
summary = pd.DataFrame(rows).sort_values("Return", ascending=False).reset_index(drop=True)
fs = summary.copy()
labels_short = [SHORT.get(k, k) for k in fs["key"]]
spy_ret_val  = summary["SPY Ret"].iloc[0] if not summary.empty else 0.0
print("  OK")

# ── Test every figure ─────────────────────────────────────────────────────────
print("\nTesting figures...")

# 1. Overview bar
try:
    bar_colors = [C["green"] if r >= 0 else C["red"] for r in fs["Return"]]
    fig = go.Figure(go.Bar(
        y=labels_short, x=fs["Return"], orientation="h",
        marker=dict(color=bar_colors, opacity=0.9, line=dict(width=0)),
        text=[f"{v:+.3f}%" for v in fs["Return"]],
        textposition="outside", textfont=dict(size=11, color=C["text"]),
    ))
    apply_layout(fig, height=420, margin=dict(l=0, r=60, t=10, b=0),
                 yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(size=11), tickcolor=C["muted"], linecolor=C["border"]))
    fig.add_vline(x=0, line_color=C["border"], line_width=1)
    ok("overview bar")
except Exception as e:
    fail("overview bar", e)

# 2. Alpha bar
try:
    alpha_colors = [C["green"] if a >= 0 else C["red"] for a in fs["Alpha"]]
    fig = go.Figure(go.Bar(
        y=labels_short, x=fs["Alpha"], orientation="h",
        marker=dict(color=alpha_colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:+.3f}%" for v in fs["Alpha"]],
        textposition="outside", textfont=dict(size=11, color=C["text"]), name="Alpha",
    ))
    apply_layout(fig, height=420, xaxis_title="Alpha (%)",
                 margin=dict(l=0, r=60, t=10, b=0),
                 yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(size=11), linecolor=C["border"]))
    fig.add_vline(x=0, line_color=C["border"], line_width=1)
    ok("alpha bar")
except Exception as e:
    fail("alpha bar", e)

# 3. Return vs Alpha scatter
try:
    sc_colors = [C["green"] if a >= 0 else C["red"] for a in fs["Alpha"]]
    fig = go.Figure()
    fig.add_vline(x=spy_ret_val, line_color=C["orange"], line_width=1.5,
                  line_dash="dot", annotation_text="SPY",
                  annotation_font_color=C["orange"], annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=fs["Return"], y=fs["Alpha"], mode="markers+text",
        text=[SHORT.get(k, "") for k in fs["key"]],
        textposition="top center", textfont=dict(size=10, color=C["muted"]),
        marker=dict(size=14, color=sc_colors, opacity=0.85,
                    line=dict(width=1, color=C["border"])),
        hovertemplate="<b>%{text}</b><br>Return: %{x:+.3f}%<br>Alpha: %{y:+.3f}%<extra></extra>",
    ))
    apply_layout(fig, height=380, xaxis_title="Total Return (%)", yaxis_title="Alpha vs SPY (%)")
    fig.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0.5, y0=0.5, x1=1, y1=1,
                  fillcolor="rgba(0,212,170,0.04)", line_width=0)
    ok("scatter")
except Exception as e:
    fail("scatter", e)

# 4. Equity curves (flat for day 1)
try:
    fig = go.Figure()
    for i, k in enumerate(list(states.keys())[:3]):
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp("2026-02-24")], y=[100.0],
            mode="markers", name=SHORT.get(k, k),
            marker=dict(size=8, color=CHART_COLORS[i % len(CHART_COLORS)]),
        ))
    apply_layout(fig, height=380)
    fig.add_hline(y=100, line_dash="dot", line_color=C["border"], line_width=1)
    ok("equity curves (flat)")
except Exception as e:
    fail("equity curves", e)

# 5. Treemap + exposure
try:
    all_pos = []
    for k, st_ in states.items():
        pv = st_.get("portfolio_value", START_CASH)
        for ticker, p in st_.get("positions", {}).items():
            all_pos.append({"Strategy": SHORT.get(k, k), "Ticker": ticker,
                            "Notional": p["weight"] * pv})
    if all_pos:
        pos_df = pd.DataFrame(all_pos)
        fig = px.treemap(pos_df, path=["Strategy","Ticker"], values="Notional",
                         color="Notional",
                         color_continuous_scale=[[0, C["card"]], [0.5, C["cyan"]], [1, C["purple"]]])
        fig.update_traces(textfont=dict(size=13, color=C["text"]),
                          marker=dict(line=dict(width=2, color=C["bg"])))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
                          coloraxis_showscale=False, height=380,
                          margin=dict(l=0, r=0, t=10, b=0))
        ok("treemap")

        ticker_exp = pos_df.groupby("Ticker")["Notional"].sum().sort_values(ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=ticker_exp.values, y=ticker_exp.index, orientation="h",
            marker=dict(color=list(ticker_exp.values),
                        colorscale=[[0, C["border"]], [1, C["cyan"]]], line=dict(width=0)),
            text=[f"${v:,.0f}" for v in ticker_exp.values],
            textposition="outside", textfont=dict(size=10, color=C["muted"]),
        ))
        apply_layout(fig, height=380, margin=dict(l=0, r=60, t=10, b=0),
                     xaxis=dict(gridcolor=C["border"], tickfont=dict(size=10)),
                     yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)))
        ok("ticker exposure bar")
except Exception as e:
    fail("treemap/exposure", e)

# 6. Trade volume bar
try:
    all_trades = []
    for k, st_ in states.items():
        for t in st_.get("trades", []):
            all_trades.append({"Strategy": SHORT.get(k, k), "Action": t.get("action",""),
                               "Value": t.get("dollar_value", 0)})
    if all_trades:
        td   = pd.DataFrame(all_trades)
        tvol = td.groupby("Strategy")["Value"].sum().sort_values(ascending=True)
        fig  = go.Figure(go.Bar(
            x=tvol.values, y=tvol.index, orientation="h",
            marker=dict(color=C["purple"], opacity=0.85, line=dict(width=0)),
            text=[f"${v:,.0f}" for v in tvol.values],
            textposition="outside", textfont=dict(size=10, color=C["muted"]),
        ))
        apply_layout(fig, height=350, margin=dict(l=0, r=60, t=10, b=0),
                     yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                     xaxis=dict(tickfont=dict(size=10)))
        ok("trade volume bar")

        act_ct = td.groupby(["Strategy","Action"])["Value"].count().unstack(fill_value=0)
        fig2   = go.Figure()
        if "BUY"  in act_ct.columns:
            fig2.add_trace(go.Bar(name="BUY",  x=act_ct.index, y=act_ct["BUY"],  marker_color=C["green"]))
        if "SELL" in act_ct.columns:
            fig2.add_trace(go.Bar(name="SELL", x=act_ct.index, y=act_ct["SELL"], marker_color=C["red"]))
        fig2.update_layout(barmode="group")
        apply_layout(fig2, height=350, margin=dict(l=0, r=0, t=10, b=0),
                     yaxis_title="Count",
                     xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                     legend=dict(orientation="h", y=1.1))
        ok("buy/sell bar")
except Exception as e:
    fail("trades charts", e)

# 7. Combined portfolio
try:
    eq_rows = []
    for s, st_ in states.items():
        for e in st_.get("daily_log", []):
            eq_rows.append({"date": e["date"], "pv": e["pv"]})
    if eq_rows:
        eq_df    = pd.DataFrame(eq_rows)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        combined = eq_df.groupby("date")["pv"].sum().reset_index()
        combined["rebased"] = combined["pv"] / combined["pv"].iloc[0] * 100
        fig = go.Figure(go.Scatter(
            x=combined["date"], y=combined["rebased"],
            fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
            line=dict(color=C["cyan"], width=2), name="Combined",
        ))
        apply_layout(fig, height=200, margin=dict(l=0, r=0, t=30, b=0), title="Combined Portfolio")
        fig.add_hline(y=100, line_dash="dot", line_color=C["border"], line_width=1)
        ok("combined equity")
except Exception as e:
    fail("combined equity", e)

print("\nAll tests passed!")
