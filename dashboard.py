"""
Quantbot Paper Trading Dashboard  ·  venv/Scripts/streamlit.exe run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from config import settings

# ── Constants & config ────────────────────────────────────────────────────────
STATE_DIR   = Path(__file__).parent / "state"
START_CASH  = 100_000

STRATEGIES  = [
    "s01_momentum_dip", "s02_cross_asset_mom", "s03_factor_alpha",
    "s04_earnings_drift", "s05_short_term_reversal", "s06_vix_term_structure",
    "s07_macro_regime", "s09_dollar_carry", "s10_vol_surface",
    "s11_congressional", "s12_index_inclusion", "s13_pre_earnings_drift",
    "s14_gamma_wall", "s15_short_flow", "s16_overnight_carry",
    "s19_turn_of_month",
]

LABELS = {
    "s01_momentum_dip":        "S01 · Momentum Dip",
    "s02_cross_asset_mom":     "S02 · Cross-Asset Mom",
    "s03_factor_alpha":        "S03 · Factor Alpha",
    "s04_earnings_drift":      "S04 · Earnings Drift",
    "s05_short_term_reversal": "S05 · Short-Term Rev",
    "s06_vix_term_structure":  "S06 · VIX Term Struct",
    "s07_macro_regime":        "S07 · Macro Regime",
    "s09_dollar_carry":        "S09 · Dollar Carry",
    "s10_vol_surface":         "S10 · Vol Surface",
    "s11_congressional":       "S11 · Congressional",
    "s12_index_inclusion":     "S12 · Index Inclusion",
    "s13_pre_earnings_drift":  "S13 · Pre-Earnings",
    "s14_gamma_wall":          "S14 · Gamma Wall",
    "s15_short_flow":          "S15 · Short Flow",
    "s16_overnight_carry":     "S16 · Overnight Carry",
    "s19_turn_of_month":       "S19 · Turn-of-Month",
}

SHORT = {k: v.split(" · ")[1] for k, v in LABELS.items()}

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":        "#0f1117",
    "card":      "#1a1d27",
    "border":    "#2a2d3a",
    "green":     "#00d4aa",
    "red":       "#ff4d6d",
    "cyan":      "#38bdf8",
    "orange":    "#fb923c",
    "purple":    "#a78bfa",
    "text":      "#f1f5f9",
    "muted":     "#94a3b8",
    "dimmed":    "#475569",
}

CHART_COLORS = [
    "#38bdf8","#00d4aa","#a78bfa","#fb923c","#f472b6",
    "#34d399","#fbbf24","#60a5fa","#f87171","#4ade80",
    "#e879f9","#22d3ee","#fb7185","#a3e635","#c084fc",
    "#f9a8d4",
]

# ── Plotly dark template ───────────────────────────────────────────────────────
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


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── global ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {C['bg']};
        color: {C['text']};
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {C['card']};
        border-right: 1px solid {C['border']};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    div[data-testid="stTabs"] button {{
        color: {C['muted']};
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {C['cyan']};
        border-bottom-color: {C['cyan']};
    }}

    /* ── metric card ── */
    .qcard {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 18px 20px 14px;
        margin-bottom: 4px;
    }}
    .qcard-label {{
        font-size: 11px;
        font-weight: 600;
        color: {C['muted']};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }}
    .qcard-value {{
        font-size: 26px;
        font-weight: 700;
        color: {C['text']};
        font-family: 'Courier New', monospace;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }}
    .qcard-delta {{
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }}
    .up   {{ color: {C['green']}; }}
    .down {{ color: {C['red']};   }}
    .flat {{ color: {C['muted']}; }}

    /* ── section header ── */
    .section-hdr {{
        font-size: 13px;
        font-weight: 700;
        color: {C['muted']};
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid {C['border']};
        padding-bottom: 6px;
        margin: 16px 0 10px;
    }}

    /* ── pill badge ── */
    .pill {{
        display:inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }}
    .pill-green {{ background: rgba(0,212,170,0.12); color:{C['green']}; }}
    .pill-red   {{ background: rgba(255,77,109,0.12); color:{C['red']};   }}
    .pill-cyan  {{ background: rgba(56,189,248,0.12); color:{C['cyan']};  }}

    /* ── dataframe override ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {C['border']};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ── strategy guide ── */
    .guide-tag {{
        display:inline-block; padding:3px 12px; border-radius:20px;
        font-size:11px; font-weight:700; margin:0 6px 8px 0; letter-spacing:0.4px;
    }}
    .gt-equity {{ background:rgba(56,189,248,.15);  color:#38bdf8; border:1px solid rgba(56,189,248,.3); }}
    .gt-macro  {{ background:rgba(167,139,250,.15); color:#a78bfa; border:1px solid rgba(167,139,250,.3); }}
    .gt-event  {{ background:rgba(251,146,60,.15);  color:#fb923c; border:1px solid rgba(251,146,60,.3); }}
    .gt-vol    {{ background:rgba(244,114,182,.15); color:#f472b6; border:1px solid rgba(244,114,182,.3); }}
    .gt-flow   {{ background:rgba(52,211,153,.15);  color:#34d399; border:1px solid rgba(52,211,153,.3); }}
    .gt-struct {{ background:rgba(251,191,36,.15);  color:#fbbf24; border:1px solid rgba(251,191,36,.3); }}
    .guide-h {{ font-size:11px; font-weight:700; color:{C['muted']}; text-transform:uppercase;
                letter-spacing:0.9px; margin:16px 0 6px; }}
    .guide-body {{ font-size:14px; color:#e2e8f0; line-height:1.75; }}
    .exit-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:8px 0 4px; }}
    .exit-cell {{ background:{C['bg']}; border:1px solid {C['border']}; border-radius:8px;
                  padding:12px 14px; }}
    .exit-cell-label {{ font-size:10px; color:{C['muted']}; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:4px; }}
    .exit-cell-val {{ font-size:22px; font-weight:800; font-family:'Courier New',monospace; }}
    .exit-stop   {{ color:{C['red']};   }}
    .exit-target {{ color:{C['green']}; }}
    .exit-time   {{ color:{C['cyan']};  }}

    /* streamlit default metric tweak */
    [data-testid="stMetric"] {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    [data-testid="stMetricLabel"] > div {{
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {C['muted']} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Courier New', monospace;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: {C['text']} !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 13px !important;
        font-weight: 600 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def card(label: str, value: str, delta: str | None = None, delta_sign: int = 0) -> str:
    cls = "up" if delta_sign > 0 else ("down" if delta_sign < 0 else "flat")
    delta_html = f'<div class="qcard-delta {cls}">{delta}</div>' if delta else ""
    return f"""
    <div class="qcard">
        <div class="qcard-label">{label}</div>
        <div class="qcard-value">{value}</div>
        {delta_html}
    </div>"""


def pct_color(v: float) -> str:
    if v > 0:   return C["green"]
    if v < 0:   return C["red"]
    return C["muted"]


def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_states() -> dict[str, dict]:
    out = {}
    for s in STRATEGIES:
        p = STATE_DIR / f"{s}_state.json"
        if p.exists():
            out[s] = json.load(open(p))
    return out


@st.cache_data(ttl=300)
def fetch_current_prices(tickers: tuple) -> dict[str, float]:
    """Fetch latest close price for a set of tickers."""
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="5d", auto_adjust=True, progress=False)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        closes = closes.dropna(how="all")
        if closes.empty:
            return {}
        last = closes.iloc[-1]
        return {t: float(last[t]) for t in tickers if t in last.index and not pd.isna(last[t])}
    except Exception:
        return {}


@st.cache_data(ttl=120)
def fetch_spy(start: str, end: str) -> pd.Series:
    try:
        raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"]
        # yfinance returns MultiIndex columns for single ticker — flatten
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        sq = close.squeeze()
        # squeeze() on 1-row DataFrame gives scalar — wrap back into Series
        if not isinstance(sq, pd.Series):
            sq = pd.Series([float(sq)], index=raw.index)
        result = sq.dropna()
        # Return empty if we got fewer than 2 points (can't compute returns)
        if len(result) < 2:
            return pd.Series(dtype=float)
        return result
    except Exception:
        return pd.Series(dtype=float)


def equity_curves(states: dict) -> pd.DataFrame:
    rows = []
    for s, st_ in states.items():
        for e in st_.get("daily_log", []):
            rows.append({"date": e["date"], "strategy": s, "pv": e["pv"]})
    if not rows:
        return pd.DataFrame(columns=["date", "strategy", "pv"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # De-duplicate (keep last entry per strategy per date) then forward-fill
    # so strategies that didn't log on a given day retain their prior value.
    piv = (df.sort_values("date")
             .groupby(["date", "strategy"])["pv"].last()
             .unstack("strategy")
             .ffill())
    df = piv.stack("strategy").reset_index()
    df.columns = ["date", "strategy", "pv"]
    return df.sort_values("date")


def build_summary(states: dict, spy_series: pd.Series) -> pd.DataFrame:
    rows = []
    for s, st_ in states.items():
        pv        = st_.get("portfolio_value", START_CASH)
        peak      = st_.get("peak_value", pv)
        positions = st_.get("positions", {})
        trades    = st_.get("trades", [])
        daily     = st_.get("daily_log", [])
        start_dt = start_d  # always compare vs SPY from global inception date (Feb 24)

        total_ret = (pv / START_CASH - 1) * 100

        # max drawdown
        pvs = [e["pv"] for e in daily]
        if len(pvs) > 1:
            peak_r, max_dd = pvs[0], 0.0
            for v in pvs:
                peak_r = max(peak_r, v)
                max_dd = min(max_dd, (v / peak_r - 1) * 100)
        else:
            max_dd = (pv / peak - 1) * 100 if peak > 0 else 0.0

        # Sharpe (annualized) from daily returns
        if len(daily) > 2:
            rets = pd.Series([e["pv"] for e in daily]).pct_change().dropna()
            sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        else:
            sharpe = None

        # Annualized return from daily_log date range
        ann_ret = None
        if len(daily) >= 2:
            try:
                d0 = pd.Timestamp(daily[0]["date"])
                d1 = pd.Timestamp(daily[-1]["date"])
                cal_days = (d1 - d0).days
                if cal_days > 0:
                    total_ret_dec = pv / START_CASH - 1
                    ann_ret = ((1 + total_ret_dec) ** (365 / cal_days) - 1) * 100
            except Exception:
                pass

        # SPY return over same window
        spy_ret = 0.0
        if not spy_series.empty and start_dt:
            try:
                spy_sub = spy_series[spy_series.index >= start_dt]
                if len(spy_sub) >= 2:
                    spy_ret = (spy_sub.iloc[-1] / spy_sub.iloc[0] - 1) * 100
            except Exception:
                pass

        alpha = total_ret - spy_ret

        # Invested vs cash breakdown
        cash      = st_.get("cash", 0.0)
        invested  = pv - cash   # total notional in positions

        rows.append({
            "key":        s,
            "Strategy":   LABELS.get(s, s),
            "Value":      pv,
            "Invested":   invested,
            "Cash":       cash,
            "Return":     total_ret,
            "Ann Ret":    ann_ret,
            "Alpha":      alpha,
            "SPY Ret":    spy_ret,
            "Max DD":     max_dd,
            "Sharpe":     sharpe,
            "Positions":  len(positions),
            "Trades":     len(trades),
        })
    return pd.DataFrame(rows).sort_values("Return", ascending=False).reset_index(drop=True)


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Quantbot",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)
inject_css()

states = load_states()
if not states:
    st.error("No state files found — run paper trading first.")
    st.stop()

eq_df   = equity_curves(states)
start_d = "2026-02-24"
spy_raw = fetch_spy(start_d, (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"))
summary = build_summary(states, spy_raw)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:10px 0 20px;">
        <div style="font-size:22px;font-weight:800;color:{C['text']};letter-spacing:-0.5px;">
            📈 Quantbot
        </div>
        <div style="font-size:11px;color:{C['muted']};margin-top:2px;">Paper Trading Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    total_aum = summary["Value"].sum()
    total_ret = (total_aum / (START_CASH * len(states)) - 1) * 100
    spy_ret_g = summary["SPY Ret"].mean()
    alpha_g   = total_ret - spy_ret_g

    sign = lambda v: "▲" if v > 0 else ("▼" if v < 0 else "—")
    col  = lambda v: C["green"] if v > 0 else (C["red"] if v < 0 else C["muted"])

    st.markdown(f"""
    <div class="qcard" style="margin-bottom:10px;">
        <div class="qcard-label">Total AUM</div>
        <div class="qcard-value">${total_aum:,.0f}</div>
        <div class="qcard-delta" style="color:{col(total_ret)};">{sign(total_ret)} {total_ret:+.3f}%</div>
    </div>
    <div class="qcard" style="margin-bottom:10px;">
        <div class="qcard-label">Alpha vs SPY</div>
        <div class="qcard-value" style="color:{col(alpha_g)};">{alpha_g:+.3f}%</div>
        <div class="qcard-delta flat">SPY: {spy_ret_g:+.3f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    active = summary[summary["Positions"] > 0].shape[0]
    total_pos = int(summary["Positions"].sum())
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="color:{C['muted']};font-size:12px;">Active strategies</span>
        <span style="color:{C['text']};font-weight:700;">{active} / {len(states)}</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="color:{C['muted']};font-size:12px;">Open positions</span>
        <span style="color:{C['text']};font-weight:700;">{total_pos}</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="color:{C['muted']};font-size:12px;">Last updated</span>
        <span style="color:{C['text']};font-weight:700;">{datetime.now().strftime('%H:%M:%S')}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    all_labels = [LABELS[s] for s in STRATEGIES if s in states]
    selected   = st.multiselect("Filter strategies", all_labels, default=all_labels)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid {C['border']};padding-bottom:12px;margin-bottom:20px;">
    <div>
        <span style="font-size:28px;font-weight:800;color:{C['text']};letter-spacing:-1px;">
            Paper Trading
        </span>
        <span style="font-size:14px;color:{C['muted']};margin-left:12px;">
            Since {start_d}
        </span>
    </div>
    <div style="font-size:12px;color:{C['muted']};">
        Auto-refresh · {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Portfolio hero banner ─────────────────────────────────────────────────────
calendar_days   = max((datetime.today() - datetime.strptime(start_d, "%Y-%m-%d")).days, 1)
ann_ret         = total_ret * (365 / calendar_days)
pret_col        = C["green"] if total_ret >= 0 else C["red"]
ann_col         = C["green"] if ann_ret   >= 0 else C["red"]
pret_sign       = "▲" if total_ret >= 0 else "▼"
ann_sign        = "▲" if ann_ret   >= 0 else "▼"

st.markdown(f"""
<div style="background:linear-gradient(135deg,{C['card']} 0%,{C['bg']} 100%);
            border:1px solid {C['border']};border-radius:14px;
            padding:28px 36px;margin-bottom:18px;
            display:flex;align-items:center;gap:0;">
  <div style="flex:1;border-right:1px solid {C['border']};padding-right:40px;">
    <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;
                letter-spacing:1.2px;margin-bottom:4px;">Total Portfolio Return</div>
    <div style="font-size:62px;font-weight:800;color:{pret_col};
                font-family:'Courier New',monospace;letter-spacing:-3px;line-height:1;">
      {pret_sign} {abs(total_ret):.2f}%
    </div>
    <div style="font-size:13px;color:{C['muted']};margin-top:6px;">
      across ${total_aum:,.0f} AUM &nbsp;·&nbsp; {len(states)} strategies
    </div>
  </div>
  <div style="flex:1;padding:0 40px;border-right:1px solid {C['border']};">
    <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;
                letter-spacing:1.2px;margin-bottom:4px;">Implied Annual Return</div>
    <div style="font-size:50px;font-weight:800;color:{ann_col};
                font-family:'Courier New',monospace;letter-spacing:-2px;line-height:1;">
      {ann_sign} {abs(ann_ret):.1f}%
    </div>
    <div style="font-size:13px;color:{C['muted']};margin-top:6px;">
      extrapolated from {calendar_days} days &nbsp;·&nbsp; short-history estimate
    </div>
  </div>
  <div style="padding-left:40px;text-align:center;flex:0 0 auto;">
    <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;
                letter-spacing:1.2px;margin-bottom:4px;">Live Since</div>
    <div style="font-size:44px;font-weight:800;color:{C['text']};
                font-family:'Courier New',monospace;line-height:1;">
      Day {calendar_days}
    </div>
    <div style="font-size:13px;color:{C['muted']};margin-top:6px;">{start_d}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────────────────────
best_row  = summary.iloc[0]
worst_row = summary.iloc[-1]
sharpe_valid = summary["Sharpe"].dropna()
avg_sharpe = sharpe_valid.mean() if len(sharpe_valid) > 0 else None
avg_dd    = summary["Max DD"].mean()

k1, k2, k3, k4, k5 = st.columns(5)

def sign_int(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)

with k1:
    st.markdown(card("Total AUM", f"${total_aum:,.0f}",
                     f"{'▲' if total_ret>0 else '▼'} {total_ret:+.3f}%", sign_int(total_ret)),
                unsafe_allow_html=True)
with k2:
    st.markdown(card("Alpha vs SPY", f"{alpha_g:+.3f}%",
                     f"SPY {spy_ret_g:+.3f}%", sign_int(alpha_g)),
                unsafe_allow_html=True)
with k3:
    sh_str = f"{avg_sharpe:.2f}" if avg_sharpe is not None else "—"
    st.markdown(card("Avg Sharpe", sh_str, "annualized", 0), unsafe_allow_html=True)
with k4:
    st.markdown(card("Avg Max DD", f"{avg_dd:.3f}%", None, sign_int(avg_dd)),
                unsafe_allow_html=True)
with k5:
    st.markdown(card("Best Strategy",
                     SHORT.get(best_row["key"], best_row["Strategy"]),
                     f"{best_row['Return']:+.3f}%", sign_int(best_row["Return"])),
                unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_eq, tab_bench, tab_pos, tab_trades, tab_pnl, tab_guide = st.tabs([
    "  Overview  ", "  Equity Curves  ", "  Benchmark vs SPY  ",
    "  Positions  ", "  Trades  ", "  P&L Breakdown  ", "  Strategy Guide  ",
])

filt_keys    = [k for k, v in LABELS.items() if v in selected and k in states]
filt_summary = summary[summary["key"].isin(filt_keys)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    # ── P&L waterfall bar ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Strategy Returns</div>', unsafe_allow_html=True)

    bar_colors = [C["green"] if r >= 0 else C["red"] for r in filt_summary["Return"]]
    labels_short = [SHORT.get(k, k) for k in filt_summary["key"]]

    fig_bar = go.Figure(go.Bar(
        y=labels_short,
        x=filt_summary["Return"],
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.9, line=dict(width=0)),
        text=[f"{v:+.3f}%" for v in filt_summary["Return"]],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
    ))
    apply_layout(fig_bar, height=420, margin=dict(l=0, r=60, t=10, b=0),
                 yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(size=11), tickcolor=C["muted"], linecolor=C["border"]))
    fig_bar.add_vline(x=0, line_color=C["border"], line_width=1)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Summary table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Performance Table</div>', unsafe_allow_html=True)

    tbl = filt_summary[["Strategy","Value","Invested","Cash","Return","Ann Ret","Alpha","SPY Ret","Max DD","Sharpe","Positions","Trades"]].copy()
    tbl.columns = ["Strategy","Value ($)","Invested ($)","Cash ($)","Return (%)","Ann Ret (%)","Alpha (%)","SPY (%)","Max DD (%)","Sharpe","Pos","Trades"]

    # Format helpers
    def fmt_pct(v):
        if pd.isna(v): return "—"
        color = C["green"] if v > 0 else (C["red"] if v < 0 else C["muted"])
        sym   = "▲" if v > 0 else ("▼" if v < 0 else "")
        return f'<span style="color:{color};font-weight:600;">{sym}{abs(v):.3f}%</span>'

    def fmt_ann_ret(v):
        if v is None or pd.isna(v): return "—"
        color = C["green"] if v > 0 else C["red"]
        sym   = "▲" if v > 0 else "▼"
        return f'<span style="color:{color};font-weight:600;">{sym}{abs(v):.1f}%</span>'

    def fmt_sharpe(v):
        if v is None or pd.isna(v): return "—"
        color = C["green"] if v > 1 else (C["orange"] if v > 0 else C["red"])
        return f'<span style="color:{color};">{v:.2f}</span>'

    def fmt_cash(v):
        color = C["muted"] if v > 1000 else C["orange"]   # flag low cash
        return f'<span style="color:{color};">${v:,.0f}</span>'

    # Fetch current prices for all open positions (cached 5 min)
    all_open_tickers: set[str] = set()
    for k in filt_keys:
        all_open_tickers.update(states.get(k, {}).get("positions", {}).keys())
    live_prices = fetch_current_prices(tuple(sorted(all_open_tickers)))

    # Build position detail rows for each strategy (used in accordion dropdown)
    def _pos_detail_html(key: str) -> str:
        pos = states.get(key, {}).get("positions", {})
        if not pos:
            return f'<span style="color:{C["muted"]};font-style:italic;">No open positions (flat / cash)</span>'
        inner = ""
        for ticker, p in sorted(pos.items(), key=lambda x: -x[1]["weight"]):
            entry   = p.get("entry_price", 0)
            shares  = p.get("shares", 0)
            weight  = p.get("weight", 0) * 100
            notional = entry * shares
            edate   = p.get("entry_date", "")
            cur     = live_prices.get(ticker)
            if cur is not None and entry:
                unreal_d = (cur - entry) * shares
                unreal_p = (cur / entry - 1) * 100
                pnl_color = C["green"] if unreal_d >= 0 else C["red"]
                arrow = "&#9650;" if unreal_d >= 0 else "&#9660;"
                pnl_str  = f'<span style="color:{pnl_color};font-weight:700;">{arrow} ${abs(unreal_d):,.0f} ({unreal_p:+.2f}%)</span>'
                cur_str  = f"${cur:.2f}"
            else:
                pnl_str = f'<span style="color:{C["muted"]};">—</span>'
                cur_str = "—"
            inner += f"""
            <tr>
                <td style="font-weight:700;color:{C['text']}">{ticker}</td>
                <td style="color:{C['cyan']}">{weight:.0f}%</td>
                <td>${entry:.2f}</td>
                <td>{cur_str}</td>
                <td>{shares:.2f}</td>
                <td style="color:{C['cyan']}">${notional:,.0f}</td>
                <td>{pnl_str}</td>
                <td style="color:{C['muted']}">{edate}</td>
            </tr>"""
        hdr = "".join(f'<th style="padding:5px 12px;color:{C["muted"]};font-size:10px;'
                      f'text-transform:uppercase;letter-spacing:0.6px;">{c}</th>'
                      for c in ["Ticker","Weight","Entry $","Current $","Shares","Notional","Unreal P&L","Entry Date"])
        return (f'<table style="width:100%;border-collapse:collapse;font-size:12px;'
                f'font-family:\'Courier New\',monospace;">'
                f'<thead><tr>{hdr}</tr></thead><tbody>{inner}</tbody></table>')

    html_rows = ""
    NCOLS = 13  # strategy + 11 data cols + expand arrow
    for _, row in tbl.iterrows():
        key = filt_summary.loc[filt_summary["Strategy"] == row["Strategy"], "key"].values[0]
        safe_key = key.replace("_", "-")
        n_pos = int(row["Pos"])
        arrow_color = C["cyan"] if n_pos > 0 else C["dimmed"]

        html_rows += f"""
        <tr class="strategy-row" onclick="toggleDetail('{safe_key}')"
            style="cursor:pointer;" title="Click to expand positions">
            <td style="color:{C['text']};font-weight:500;">
                <span id="arrow-{safe_key}" style="color:{arrow_color};margin-right:8px;font-size:10px;">&#9654;</span>{row['Strategy']}
            </td>
            <td style="font-family:'Courier New',monospace;">${row['Value ($)']:,.0f}</td>
            <td style="color:{C['cyan']};font-family:'Courier New',monospace;">${row['Invested ($)']:,.0f}</td>
            <td>{fmt_cash(row['Cash ($)'])}</td>
            <td>{fmt_pct(row['Return (%)'])}</td>
            <td>{fmt_ann_ret(row['Ann Ret (%)'])}</td>
            <td>{fmt_pct(row['Alpha (%)'])}</td>
            <td style="color:{C['muted']};">{row['SPY (%)']:+.3f}%</td>
            <td style="color:{C['red'] if row['Max DD (%)'] < 0 else C['muted']};">{row['Max DD (%)']:.3f}%</td>
            <td>{fmt_sharpe(row['Sharpe'])}</td>
            <td style="color:{C['cyan'] if n_pos>0 else C['muted']};">{n_pos}</td>
            <td style="color:{C['muted']};">{int(row['Trades'])}</td>
        </tr>
        <tr id="detail-{safe_key}" style="display:none;">
            <td colspan="{NCOLS}" style="padding:0 14px 14px 36px;background:{C['bg']};border-bottom:1px solid {C['border']};">
                {_pos_detail_html(key)}
            </td>
        </tr>"""

    header_cols = ["Strategy","Value ($)","Invested ($)","Cash ($)","Return","Ann Ret","Alpha","SPY","Max DD","Sharpe","Pos","Trades"]
    th = "".join(f"<th>{c}</th>" for c in header_cols)

    n_rows = len(filt_summary)
    table_height = 60 + n_rows * 44 + 20   # header + rows + hint; expands dynamically via JS

    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; padding:0; background:{C['bg']}; font-family:'Courier New',monospace; }}
      .perf-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
      .perf-table th {{
          background:{C['border']}; color:{C['muted']}; padding:10px 14px;
          text-align:left; font-size:11px; font-weight:700;
          text-transform:uppercase; letter-spacing:0.7px;
          border-bottom:2px solid {C['cyan']};
      }}
      .perf-table td {{
          padding:9px 14px; border-bottom:1px solid {C['border']};
          color:{C['text']}; font-size:13px;
      }}
      .strategy-row {{ cursor:pointer; }}
      .strategy-row:hover td {{ background:{C['border']}; }}
      .detail-row td {{ background:{C['bg']}; padding:0 14px 14px 38px; }}
      .pos-table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }}
      .pos-table th {{ color:{C['muted']}; font-size:10px; text-transform:uppercase;
                       letter-spacing:0.6px; padding:4px 10px; text-align:left;
                       border-bottom:1px solid {C['border']}; }}
      .pos-table td {{ padding:5px 10px; border-bottom:1px solid {C['border']}22;
                       color:{C['text']}; }}
      .pos-table tr:last-child td {{ border-bottom:none; }}
      .hint {{ font-size:11px; color:{C['dimmed']}; margin-top:6px; padding-left:4px; }}
      .arrow {{ color:{C['cyan']}; margin-right:8px; font-size:10px; display:inline-block;
                width:10px; transition: transform 0.15s; }}
    </style>
    </head>
    <body>
    <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
    <table class="perf-table">
      <thead><tr>{th}</tr></thead>
      <tbody>{html_rows}</tbody>
    </table>
    </div>
    <div class="hint">&#9654; Click any row to expand positions</div>

    <script>
    function toggleDetail(key) {{
        var detail = document.getElementById('detail-' + key);
        var arrow  = document.getElementById('arrow-' + key);
        if (!detail) return;
        var open = detail.style.display === 'table-row';
        detail.style.display = open ? 'none' : 'table-row';
        if (arrow) arrow.innerHTML = open ? '&#9654;' : '&#9660;';
        // Notify parent iframe to resize
        var newH = document.body.scrollHeight + 20;
        window.parent.document.querySelectorAll('iframe').forEach(function(f) {{
            if (f.contentWindow === window) f.style.height = newH + 'px';
        }});
    }}
    </script>
    </body></html>
    """, height=table_height, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EQUITY CURVES
# ─────────────────────────────────────────────────────────────────────────────
with tab_eq:
    if eq_df.empty or len(eq_df["date"].unique()) < 2:
        st.info("Need at least 2 days of data for equity curves. Check back after the next paper trading run.")
        # Show flat lines for day 1
        st.markdown('<div class="section-hdr">Equity Curves (Day 1 — awaiting more data)</div>',
                    unsafe_allow_html=True)
        fig_flat = go.Figure()
        for i, k in enumerate(filt_keys):
            pv = states[k].get("portfolio_value", START_CASH)
            fig_flat.add_trace(go.Scatter(
                x=[pd.Timestamp(start_d)],
                y=[100.0],
                mode="markers",
                name=SHORT.get(k, k),
                marker=dict(size=8, color=CHART_COLORS[i % len(CHART_COLORS)]),
            ))
        apply_layout(fig_flat, height=380, title="Portfolio Value (rebased to 100)")
        fig_flat.add_hline(y=100, line_dash="dot", line_color=C["border"], line_width=1)
        st.plotly_chart(fig_flat, use_container_width=True)
    else:
        col_l, col_r = st.columns([3, 1])
        with col_r:
            view = st.radio("View", ["Rebased (100)", "Absolute ($)"], index=0)
            show_spy = st.checkbox("Show SPY", value=True)

        eq_filt = eq_df[eq_df["strategy"].isin(filt_keys)].copy()
        fig_eq  = go.Figure()

        # SPY overlay
        if show_spy and not spy_raw.empty:
            spy_sub = spy_raw[spy_raw.index >= start_d]
            if len(spy_sub) > 0:
                spy_base = spy_sub.iloc[0]
                spy_vals = spy_sub / spy_base * 100 if "Rebased" in view else spy_sub * (START_CASH / spy_base)
                fig_eq.add_trace(go.Scatter(
                    x=spy_sub.index, y=spy_vals,
                    name="SPY",
                    line=dict(color=C["orange"], width=2, dash="dot"),
                    opacity=0.8,
                ))

        for i, k in enumerate(filt_keys):
            sub = eq_filt[eq_filt["strategy"] == k].sort_values("date")
            if sub.empty: continue
            base  = sub["pv"].iloc[0]
            yvals = (sub["pv"] / base * 100) if "Rebased" in view else sub["pv"]
            fig_eq.add_trace(go.Scatter(
                x=sub["date"], y=yvals,
                name=SHORT.get(k, k),
                mode="lines+markers",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                marker=dict(size=4),
            ))

        ylab = "Portfolio Value (rebased 100 = $100k)" if "Rebased" in view else "Portfolio Value ($)"
        apply_layout(fig_eq, height=460, yaxis_title=ylab,
                     legend=dict(orientation="h", y=1.05, x=0))
        fig_eq.add_hline(y=100 if "Rebased" in view else START_CASH,
                         line_dash="dot", line_color=C["border"], line_width=1)
        with col_l:
            st.plotly_chart(fig_eq, use_container_width=True)

        # Combined portfolio
        combined = eq_filt.groupby("date")["pv"].sum().reset_index()
        combined["rebased"] = combined["pv"] / combined["pv"].iloc[0] * 100
        yc = combined["rebased"] if "Rebased" in view else combined["pv"]
        fig_comb = go.Figure()
        fig_comb.add_trace(go.Scatter(
            x=combined["date"], y=yc,
            fill="tozeroy",
            fillcolor=f"rgba(56,189,248,0.07)",
            line=dict(color=C["cyan"], width=2),
            name="Combined",
        ))
        apply_layout(fig_comb, height=200, margin=dict(l=0,r=0,t=30,b=0),
                     title="Combined Portfolio")
        fig_comb.add_hline(y=100 if "Rebased" in view else combined["pv"].iloc[0],
                           line_dash="dot", line_color=C["border"], line_width=1)
        st.plotly_chart(fig_comb, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: BENCHMARK vs SPY
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    spy_ret_val = summary["SPY Ret"].iloc[0] if not summary.empty else 0.0

    # ── Alpha bar chart ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Alpha vs SPY by Strategy</div>', unsafe_allow_html=True)

    fs = filt_summary.copy()
    alpha_colors = [C["green"] if a >= 0 else C["red"] for a in fs["Alpha"]]
    labels_s = [SHORT.get(k, k) for k in fs["key"]]

    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Bar(
        y=labels_s, x=fs["Alpha"],
        orientation="h",
        marker=dict(color=alpha_colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:+.3f}%" for v in fs["Alpha"]],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
        name="Alpha",
    ))
    apply_layout(fig_alpha, height=420,
                 xaxis_title="Alpha (%)",
                 margin=dict(l=0, r=60, t=10, b=0),
                 yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                            tickfont=dict(size=11), linecolor=C["border"]))
    fig_alpha.add_vline(x=0, line_color=C["border"], line_width=1)
    st.plotly_chart(fig_alpha, use_container_width=True)

    # ── Return vs SPY scatter ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Return Distribution vs SPY Benchmark</div>',
                unsafe_allow_html=True)

    col_sc, col_meta = st.columns([2, 1])

    with col_sc:
        sc_colors = [C["green"] if a >= 0 else C["red"] for a in fs["Alpha"]]
        fig_sc = go.Figure()

        # SPY reference line
        fig_sc.add_vline(x=spy_ret_val, line_color=C["orange"], line_width=1.5,
                         line_dash="dot", annotation_text="SPY",
                         annotation_font_color=C["orange"],
                         annotation_position="top right")

        fig_sc.add_trace(go.Scatter(
            x=fs["Return"],
            y=fs["Alpha"],
            mode="markers+text",
            text=[SHORT.get(k,"") for k in fs["key"]],
            textposition="top center",
            textfont=dict(size=10, color=C["muted"]),
            marker=dict(
                size=14,
                color=sc_colors,
                opacity=0.85,
                line=dict(width=1, color=C["border"]),
            ),
            hovertemplate="<b>%{text}</b><br>Return: %{x:+.3f}%<br>Alpha: %{y:+.3f}%<extra></extra>",
        ))
        apply_layout(fig_sc, height=380, xaxis_title="Total Return (%)", yaxis_title="Alpha vs SPY (%)")
        fig_sc.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
        fig_sc.add_vline(x=0, line_dash="dot", line_color=C["border"], line_width=1)
        # Quadrant shading
        fig_sc.add_shape(type="rect", xref="paper", yref="paper",
                         x0=0.5, y0=0.5, x1=1, y1=1,
                         fillcolor=f"rgba(0,212,170,0.04)", line_width=0)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_meta:
        n_beat  = (fs["Alpha"] > 0).sum()
        n_trail = (fs["Alpha"] < 0).sum()
        avg_alpha = fs["Alpha"].mean()
        best_a    = fs.loc[fs["Alpha"].idxmax()]
        worst_a   = fs.loc[fs["Alpha"].idxmin()]

        st.markdown(f"""
        <div class="qcard" style="margin-bottom:8px;">
            <div class="qcard-label">Strategies Beating SPY</div>
            <div class="qcard-value" style="color:{C['green']};">{n_beat} / {len(fs)}</div>
        </div>
        <div class="qcard" style="margin-bottom:8px;">
            <div class="qcard-label">Average Alpha</div>
            <div class="qcard-value" style="color:{C['green'] if avg_alpha>0 else C['red']};">
                {avg_alpha:+.3f}%
            </div>
        </div>
        <div class="qcard" style="margin-bottom:8px;">
            <div class="qcard-label">SPY Return (period)</div>
            <div class="qcard-value">{spy_ret_val:+.3f}%</div>
        </div>
        <div class="qcard" style="margin-bottom:8px;">
            <div class="qcard-label">Best Alpha</div>
            <div class="qcard-value" style="color:{C['green']};font-size:18px;">
                {SHORT.get(best_a['key'],'')}
            </div>
            <div class="qcard-delta up">{best_a['Alpha']:+.3f}%</div>
        </div>
        <div class="qcard">
            <div class="qcard-label">Worst Alpha</div>
            <div class="qcard-value" style="color:{C['red']};font-size:18px;">
                {SHORT.get(worst_a['key'],'')}
            </div>
            <div class="qcard-delta down">{worst_a['Alpha']:+.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SPY vs strategies return table ────────────────────────────────────────
    st.markdown('<div class="section-hdr">Full Benchmark Breakdown</div>', unsafe_allow_html=True)

    bench_rows = ""
    for _, row in fs.iterrows():
        beat = row["Alpha"] >= 0
        badge_cls = "pill-green" if beat else "pill-red"
        badge_txt = "BEAT" if beat else "TRAIL"
        bench_rows += f"""
        <tr>
            <td style="color:{C['text']};font-weight:500;">{row['Strategy']}</td>
            <td style="color:{pct_color(row['Return'])};">{row['Return']:+.3f}%</td>
            <td style="color:{C['muted']};">{row['SPY Ret']:+.3f}%</td>
            <td style="color:{pct_color(row['Alpha'])};font-weight:700;">{row['Alpha']:+.3f}%</td>
            <td><span class="pill {badge_cls}">{badge_txt}</span></td>
        </tr>"""

    bench_th = "".join(f"<th>{c}</th>" for c in ["Strategy","Return","SPY Return","Alpha","Status"])
    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
    <table class="perf-table"><thead><tr>{bench_th}</tr></thead><tbody>{bench_rows}</tbody></table>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_pos:
    all_pos = []
    for k in filt_keys:
        st_ = states[k]
        pv  = st_.get("portfolio_value", START_CASH)
        for ticker, p in st_.get("positions", {}).items():
            all_pos.append({
                "Strategy":   SHORT.get(k, k),
                "Ticker":     ticker,
                "Weight %":   p["weight"] * 100,
                "Entry $":    p["entry_price"],
                "Shares":     p.get("shares"),
                "Notional $": p["weight"] * pv,
                "Entry Date": p.get("entry_date", ""),
            })

    if not all_pos:
        st.info("No open positions across selected strategies.")
    else:
        pos_df = pd.DataFrame(all_pos)

        col_t, col_b = st.columns([3, 2])
        with col_t:
            st.markdown('<div class="section-hdr">Position Treemap</div>', unsafe_allow_html=True)
            fig_tree = px.treemap(
                pos_df, path=["Strategy","Ticker"], values="Notional $",
                color="Notional $",
                color_continuous_scale=[[0, C["card"]], [0.5, C["cyan"]], [1, C["purple"]]],
            )
            fig_tree.update_traces(
                textfont=dict(size=13, color=C["text"]),
                marker=dict(line=dict(width=2, color=C["bg"])),
            )
            fig_tree.update_layout(
                paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
                coloraxis_showscale=False, height=380,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-hdr">Top Ticker Exposure</div>', unsafe_allow_html=True)
            ticker_exp = (pos_df.groupby("Ticker")["Notional $"]
                          .sum().sort_values(ascending=True).tail(15))
            fig_exp = go.Figure(go.Bar(
                x=ticker_exp.values, y=ticker_exp.index,
                orientation="h",
                marker=dict(
                    color=ticker_exp.values,
                    colorscale=[[0, C["border"]], [1, C["cyan"]]],
                    line=dict(width=0),
                ),
                text=[f"${v:,.0f}" for v in ticker_exp.values],
                textposition="outside",
                textfont=dict(size=10, color=C["muted"]),
            ))
            apply_layout(fig_exp, height=380, margin=dict(l=0,r=60,t=10,b=0),
                         xaxis=dict(gridcolor=C["border"], tickfont=dict(size=10)),
                         yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)))
            st.plotly_chart(fig_exp, use_container_width=True)

        # Full positions table
        st.markdown('<div class="section-hdr">All Open Positions</div>', unsafe_allow_html=True)
        pos_rows_html = ""
        for _, row in pos_df.sort_values(["Strategy","Notional $"], ascending=[True,False]).iterrows():
            pos_rows_html += f"""
            <tr>
                <td style="color:{C['cyan']};font-weight:500;">{row['Strategy']}</td>
                <td style="font-weight:700;">{row['Ticker']}</td>
                <td>{row['Weight %']:.0f}%</td>
                <td>${row['Entry $']:.2f}</td>
                <td>{row['Shares']:.2f}</td>
                <td style="color:{C['cyan']};">${row['Notional $']:,.0f}</td>
                <td style="color:{C['muted']};">{row['Entry Date']}</td>
            </tr>"""
        pos_th = "".join(f"<th>{c}</th>" for c in
                         ["Strategy","Ticker","Weight","Entry Price","Shares","Notional","Entry Date"])
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
        <table class="perf-table"><thead><tr>{pos_th}</tr></thead><tbody>{pos_rows_html}</tbody></table>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: TRADES
# ─────────────────────────────────────────────────────────────────────────────
with tab_trades:
    # Build position lifecycle records: pair each BUY with its matching SELL
    # (FIFO within each strategy × ticker) so closed positions show realized P&L
    # and open positions show unrealized P&L.  Raw BUY/SELL rows are never shown
    # directly — that was the source of $0.00 entries and NaN columns.
    pos_records: list[dict] = []

    for k in filt_keys:
        open_pos   = states[k].get("positions", {})
        raw_trades = states[k].get("trades", [])
        label      = SHORT.get(k, k)

        # AVCO (average-cost) tracker: proportional cost basis per position.
        # Handles partial rebalancing trims correctly — FIFO falsely matched a
        # tiny partial sell to the full buy lot, showing huge fake losses in S04.
        trackers: dict[str, dict] = {}  # ticker -> running position state

        for t in raw_trades:
            ticker   = t.get("ticker", "")
            action   = t.get("action", "")
            notional = t.get("dollar_value", 0.0)
            cost     = t.get("cost", 0.0)
            date     = t.get("date", "")
            delta_w  = abs(t.get("delta_weight", 0.0))

            if action == "BUY":
                # Start fresh tracker if ticker is new or was fully closed
                if ticker not in trackers or trackers[ticker]["weight"] < 0.001:
                    trackers[ticker] = {
                        "weight":          0.0,
                        "basis":           0.0,   # cost basis of currently held shares
                        "first_date":      date,
                        "total_pnl":       0.0,
                        "total_invested":  0.0,
                        "total_proceeds":  0.0,
                    }
                tr = trackers[ticker]
                tr["weight"]         += delta_w
                tr["basis"]          += notional + cost   # full cost (principal + fee)
                tr["total_invested"] += notional

            elif action == "SELL" and ticker in trackers and trackers[ticker]["weight"] >= 0.001:
                tr = trackers[ticker]
                remaining_weight = max(0.0, tr["weight"] - delta_w)
                if remaining_weight < 0.005:
                    # Final close — consume all remaining basis so the full loss/gain is captured
                    cost_portion = tr["basis"]
                    tr["basis"]  = 0.0
                    tr["weight"] = 0.0
                else:
                    sell_frac    = min(1.0, delta_w / tr["weight"])
                    cost_portion = tr["basis"] * sell_frac
                    tr["basis"]  = max(0.0, tr["basis"] - cost_portion)
                    tr["weight"] = remaining_weight
                tr["total_proceeds"] += notional
                tr["total_pnl"]      += notional - cost_portion - cost

                if tr["weight"] < 0.005:   # position fully closed
                    invested = tr["total_invested"]
                    realized_pct = tr["total_pnl"] / invested * 100 if invested else 0.0
                    try:
                        days = (pd.Timestamp(date) - pd.Timestamp(tr["first_date"])).days
                    except Exception:
                        days = None
                    pos_records.append({
                        "_key":        k,
                        "Strategy":    label,
                        "Entry Date":  tr["first_date"],
                        "Exit Date":   date,
                        "Ticker":      ticker,
                        "Invested":    invested,
                        "Proceeds":    tr["total_proceeds"],
                        "Real $":      tr["total_pnl"],
                        "Real %":      realized_pct,
                        "Status":      "CLOSED",
                        "Days":        days,
                        "Entry Price": None,
                        "Cur Price":   None,
                        "Shares":      None,
                        "Unreal $":    None,
                        "Unreal %":    None,
                    })
                    del trackers[ticker]

        # Remaining trackers are currently open positions
        for ticker, tr in trackers.items():
            if ticker not in open_pos:
                continue   # orphaned tracker — position fully closed, skip
            pos = open_pos[ticker]
            entry_date = pos.get("entry_date", tr["first_date"])
            try:
                days = (datetime.today().date() - pd.Timestamp(entry_date).date()).days
            except Exception:
                days = None
            pos_records.append({
                "_key":        k,
                "Strategy":    label,
                "Entry Date":  entry_date,
                "Exit Date":   None,
                "Ticker":      ticker,
                "Invested":    tr["basis"],   # cost basis of currently held shares
                "Proceeds":    None,
                "Real $":      None,
                "Real %":      None,
                "Status":      "OPEN",
                "Days":        days,
                "Entry Price": pos.get("entry_price"),
                "Cur Price":   None,
                "Shares":      pos.get("shares"),
                "Unreal $":    None,
                "Unreal %":    None,
            })

    if not pos_records:
        st.info("No trades recorded yet.")
    else:
        # Fetch live prices for open positions only
        open_tickers_set = {r["Ticker"] for r in pos_records if r["Status"] == "OPEN"}
        prices = fetch_current_prices(tuple(sorted(open_tickers_set)))

        for r in pos_records:
            if r["Status"] != "OPEN":
                continue
            ep     = r["Entry Price"]
            shares = r["Shares"]
            cur    = prices.get(r["Ticker"])
            if cur and ep and shares:
                r["Cur Price"] = cur
                r["Unreal $"]  = (cur - ep) * shares
                r["Unreal %"]  = (cur / ep - 1) * 100

        # Sort: OPEN first (entry date desc), then CLOSED (exit date desc)
        td = pd.DataFrame(pos_records)
        td["_sort_date"] = td["Exit Date"].fillna(td["Entry Date"])
        td = td.sort_values(["Status", "_sort_date"], ascending=[True, False]).drop(columns="_sort_date")

        open_td   = td[td["Status"] == "OPEN"]
        closed_td = td[td["Status"] == "CLOSED"]

        # ── KPI row ──────────────────────────────────────────────────────────
        total_unreal   = open_td["Unreal $"].dropna().sum()
        real_series    = closed_td["Real $"].dropna()
        gross_profit   = real_series[real_series > 0].sum()
        gross_loss     = real_series[real_series < 0].sum()
        total_realized = gross_profit + gross_loss
        n_wins  = int((real_series > 0).sum())
        n_loss  = int((real_series < 0).sum())

        ka, kb, kc, kd, ke = st.columns(5)
        with ka:
            st.markdown(card("Open Positions", str(len(open_td)), None, 0),
                        unsafe_allow_html=True)
        with kb:
            sign = 1 if total_unreal > 0 else (-1 if total_unreal < 0 else 0)
            st.markdown(card("Unrealized P&L",
                             f"{'▲' if total_unreal>=0 else '▼'} ${abs(total_unreal):,.0f}",
                             None, sign), unsafe_allow_html=True)
        with kc:
            st.markdown(card("Realized Gains",
                             f"▲ ${gross_profit:,.0f}",
                             f"{n_wins}W", 1), unsafe_allow_html=True)
        with kd:
            st.markdown(card("Realized Losses",
                             f"▼ ${abs(gross_loss):,.0f}",
                             f"{n_loss}L", -1 if gross_loss < 0 else 0), unsafe_allow_html=True)
        with ke:
            sign = 1 if total_realized > 0 else (-1 if total_realized < 0 else 0)
            st.markdown(card("Net Realized",
                             f"{'▲' if total_realized>=0 else '▼'} ${abs(total_realized):,.0f}",
                             None, sign), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Charts ───────────────────────────────────────────────────────────
        col_tv, col_by = st.columns(2)
        with col_tv:
            st.markdown('<div class="section-hdr">Unrealized P&L by Strategy</div>',
                        unsafe_allow_html=True)
            pnl_by_strat = (open_td.groupby("Strategy")["Unreal $"]
                            .sum().sort_values(ascending=True).dropna())
            if not pnl_by_strat.empty:
                pnl_colors = [C["green"] if v >= 0 else C["red"] for v in pnl_by_strat]
                fig_pnl = go.Figure(go.Bar(
                    x=pnl_by_strat.values, y=pnl_by_strat.index, orientation="h",
                    marker=dict(color=pnl_colors, opacity=0.85, line=dict(width=0)),
                    text=[f"${v:+,.0f}" for v in pnl_by_strat.values],
                    textposition="outside", textfont=dict(size=10, color=C["text"]),
                ))
                apply_layout(fig_pnl, height=350, margin=dict(l=0, r=70, t=10, b=0),
                             yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                             xaxis=dict(tickfont=dict(size=10)))
                fig_pnl.add_vline(x=0, line_color=C["border"], line_width=1)
                st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.caption("No open positions with price data.")

        with col_by:
            st.markdown('<div class="section-hdr">Realized P&L by Strategy</div>',
                        unsafe_allow_html=True)
            real_by_strat = (closed_td.groupby("Strategy")["Real $"]
                             .sum().sort_values(ascending=True).dropna())
            if not real_by_strat.empty:
                real_colors = [C["green"] if v >= 0 else C["red"] for v in real_by_strat]
                fig_real = go.Figure(go.Bar(
                    x=real_by_strat.values, y=real_by_strat.index, orientation="h",
                    marker=dict(color=real_colors, opacity=0.85, line=dict(width=0)),
                    text=[f"${v:+,.0f}" for v in real_by_strat.values],
                    textposition="outside", textfont=dict(size=10, color=C["text"]),
                ))
                apply_layout(fig_real, height=350, margin=dict(l=0, r=70, t=10, b=0),
                             yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                             xaxis=dict(tickfont=dict(size=10)))
                fig_real.add_vline(x=0, line_color=C["border"], line_width=1)
                st.plotly_chart(fig_real, use_container_width=True)
            else:
                st.caption("No closed positions yet.")

        # ── Position log table ────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Position Log</div>', unsafe_allow_html=True)
        trade_rows = ""
        for _, row in td.iterrows():
            status     = row["Status"]
            days_str   = f"{int(row['Days'])}d" if pd.notna(row["Days"]) else "—"

            if status == "OPEN":
                ep  = row["Entry Price"]
                cur = row["Cur Price"]
                ep_str  = f"${ep:.2f}"  if pd.notna(ep)  else "—"
                cur_str = f"${cur:.2f}" if pd.notna(cur) else "mkt closed"
                date_str = row["Entry Date"]

                if pd.notna(row["Unreal $"]):
                    sign_u   = 1 if row["Unreal $"] >= 0 else -1
                    uc       = C["green"] if sign_u > 0 else C["red"]
                    arrow    = "▲" if sign_u > 0 else "▼"
                    pnl_str  = (f'<span style="color:{uc};font-weight:700;">'
                                f'{arrow} ${abs(row["Unreal $"]):,.0f}'
                                f' ({row["Unreal %"]:+.2f}%)</span>')
                else:
                    pnl_str = f'<span style="color:{C["muted"]};">mkt closed</span>'

                detail_str = f"{ep_str} &rarr; {cur_str}"
                status_html = f'<span class="pill pill-green">OPEN</span>'

            else:  # CLOSED
                inv  = row["Invested"]
                proc = row["Proceeds"]
                inv_str  = f"${inv:,.0f}"  if pd.notna(inv)  else "—"
                proc_str = f"${proc:,.0f}" if pd.notna(proc) else "—"
                detail_str = f"{inv_str} &rarr; {proc_str}"

                exit_dt  = row["Exit Date"] if pd.notna(row["Exit Date"]) else "?"
                date_str = f'{row["Entry Date"]} &rarr; {exit_dt}'

                if pd.notna(row["Real $"]):
                    sign_r  = 1 if row["Real $"] >= 0 else -1
                    rc      = C["green"] if sign_r > 0 else C["red"]
                    arrow   = "▲" if sign_r > 0 else "▼"
                    outcome = "PROFIT" if sign_r > 0 else "LOSS"
                    pnl_str = (f'<span style="color:{rc};font-weight:700;">'
                               f'{arrow} ${abs(row["Real $"]):,.0f}'
                               f' ({row["Real %"]:+.2f}%)</span>')
                    pill_cls = "pill-green" if sign_r > 0 else "pill-red"
                    status_html = f'<span class="pill {pill_cls}">{outcome}</span>'
                else:
                    pnl_str     = f'<span style="color:{C["muted"]};">—</span>'
                    status_html = f'<span class="pill pill-cyan">CLOSED</span>'

            trade_rows += f"""
            <tr>
                <td style="color:{C['cyan']};">{row['Strategy']}</td>
                <td style="color:{C['muted']};">{date_str}</td>
                <td style="font-weight:700;">{row['Ticker']}</td>
                <td style="color:{C['muted']};">{detail_str}</td>
                <td>{pnl_str}</td>
                <td style="color:{C['muted']};">{days_str}</td>
                <td>{status_html}</td>
            </tr>"""

        tr_th = "".join(f"<th>{c}</th>" for c in
                        ["Strategy", "Date", "Ticker", "Entry -> Exit",
                         "P&L", "Held", "Status"])
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
        <table class="perf-table"><thead><tr>{tr_th}</tr></thead><tbody>{trade_rows}</tbody></table>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: P&L BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
with tab_pnl:
    # ── Aggregate transaction costs from raw trades ───────────────────────────
    pnl_rows: list[dict] = []
    for k in filt_keys:
        raw_trades = states[k].get("trades", [])
        label      = SHORT.get(k, k)
        init_cap   = settings.INITIAL_CAPITAL

        total_costs   = sum(t.get("cost", 0.0) for t in raw_trades)
        gross_gains   = sum(r["Real $"] for r in pos_records if r["_key"] == k and (r.get("Real $") or 0) > 0)
        gross_losses  = sum(r["Real $"] for r in pos_records if r["_key"] == k and (r.get("Real $") or 0) < 0)
        net_realized  = gross_gains + gross_losses
        unrealized    = sum((r.get("Unreal $") or 0) for r in pos_records if r["_key"] == k and r["Status"] == "OPEN")
        total_pnl     = net_realized + unrealized
        pv            = states[k].get("portfolio_value", init_cap)

        pnl_rows.append({
            "Strategy":       label,
            "Gross Gains $":  gross_gains,
            "Gross Losses $": gross_losses,
            "Net Realized $": net_realized,
            "Transaction $":  -total_costs,
            "Unrealized $":   unrealized,
            "Total P&L $":    total_pnl,
            "Total P&L %":    total_pnl / init_cap * 100 if init_cap else 0,
        })

    pnl_df = pd.DataFrame(pnl_rows).sort_values("Total P&L $", ascending=False)

    # ── Portfolio totals ──────────────────────────────────────────────────────
    tot_gains   = pnl_df["Gross Gains $"].sum()
    tot_losses  = pnl_df["Gross Losses $"].sum()
    tot_net_r   = pnl_df["Net Realized $"].sum()
    tot_costs   = pnl_df["Transaction $"].sum()
    tot_unreal  = pnl_df["Unrealized $"].sum()
    tot_pnl     = pnl_df["Total P&L $"].sum()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-hdr">Portfolio P&L Breakdown</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        sign = 1 if tot_pnl >= 0 else -1
        st.markdown(card("Total P&L",
                         f"{'▲' if tot_pnl>=0 else '▼'} ${abs(tot_pnl):,.0f}",
                         None, sign), unsafe_allow_html=True)
    with c2:
        st.markdown(card("Gross Gains", f"▲ ${tot_gains:,.0f}", None, 1), unsafe_allow_html=True)
    with c3:
        st.markdown(card("Gross Losses", f"▼ ${abs(tot_losses):,.0f}", None, -1 if tot_losses < 0 else 0),
                    unsafe_allow_html=True)
    with c4:
        sign = 1 if tot_net_r >= 0 else -1
        st.markdown(card("Net Realized",
                         f"{'▲' if tot_net_r>=0 else '▼'} ${abs(tot_net_r):,.0f}",
                         None, sign), unsafe_allow_html=True)
    with c5:
        st.markdown(card("Transaction Costs", f"▼ ${abs(tot_costs):,.0f}", None, -1),
                    unsafe_allow_html=True)
    with c6:
        sign = 1 if tot_unreal >= 0 else -1
        st.markdown(card("Unrealized P&L",
                         f"{'▲' if tot_unreal>=0 else '▼'} ${abs(tot_unreal):,.0f}",
                         None, sign), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Stacked waterfall chart ───────────────────────────────────────────────
    col_wa, col_br = st.columns(2)

    with col_wa:
        st.markdown(f'<div class="section-hdr">P&L Waterfall (Portfolio)</div>', unsafe_allow_html=True)
        components    = ["Gross Gains", "Gross Losses", "Transaction Costs", "Unrealized", "Net P&L"]
        comp_vals     = [tot_gains, tot_losses, tot_costs, tot_unreal, tot_pnl]
        comp_colors   = [C["green"], C["red"], C["orange"], C["cyan"],
                         C["green"] if tot_pnl >= 0 else C["red"]]
        fig_wf = go.Figure(go.Bar(
            x=components, y=comp_vals,
            marker=dict(color=comp_colors, opacity=0.85, line=dict(width=0)),
            text=[f"${v:,.0f}" for v in comp_vals],
            textposition="outside", textfont=dict(size=11, color=C["text"]),
        ))
        apply_layout(fig_wf, height=300, showlegend=False,
                     yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_br:
        st.markdown(f'<div class="section-hdr">Net P&L by Strategy</div>', unsafe_allow_html=True)
        strat_pnl = pnl_df.set_index("Strategy")["Total P&L $"].sort_values()
        colors    = [C["green"] if v >= 0 else C["red"] for v in strat_pnl]
        fig_sp = go.Figure(go.Bar(
            x=strat_pnl.values, y=strat_pnl.index, orientation="h",
            marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
            text=[f"${v:,.0f}" for v in strat_pnl.values],
            textposition="outside", textfont=dict(size=10, color=C["text"]),
        ))
        apply_layout(fig_sp, height=300, showlegend=False,
                     xaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig_sp, use_container_width=True)

    # ── Per-strategy breakdown table ─────────────────────────────────────────
    st.markdown(f'<div class="section-hdr">Per-Strategy Breakdown</div>', unsafe_allow_html=True)

    def _pnl_color(v: float) -> str:
        if v > 0:  return C["green"]
        if v < 0:  return C["red"]
        return C["muted"]

    def _fmt(v: float) -> str:
        sign = "+" if v > 0 else ""
        return f"{sign}${v:,.0f}"

    cols = ["Strategy", "Gross Gains $", "Gross Losses $", "Net Realized $",
            "Transaction $", "Unrealized $", "Total P&L $", "Total P&L %"]
    th_cells = "".join(f"<th>{c.replace(' $','').replace(' %','')}</th>" for c in cols)
    trows = ""
    for _, row in pnl_df.iterrows():
        trows += "<tr>"
        trows += f"<td>{row['Strategy']}</td>"
        trows += f"<td style='color:{C['green']}'>{_fmt(row['Gross Gains $'])}</td>"
        trows += f"<td style='color:{C['red']}'>{_fmt(row['Gross Losses $'])}</td>"
        trows += f"<td style='color:{_pnl_color(row['Net Realized $'])}'>{_fmt(row['Net Realized $'])}</td>"
        trows += f"<td style='color:{C['orange']}'>{_fmt(row['Transaction $'])}</td>"
        trows += f"<td style='color:{_pnl_color(row['Unrealized $'])}'>{_fmt(row['Unrealized $'])}</td>"
        trows += f"<td style='color:{_pnl_color(row['Total P&L $'])};font-weight:600'>{_fmt(row['Total P&L $'])}</td>"
        trows += f"<td style='color:{_pnl_color(row['Total P&L %'])};font-weight:600'>{row['Total P&L %']:+.2f}%</td>"
        trows += "</tr>"

    # Totals row
    trows += f"""<tr style='border-top:1px solid {C['border']};font-weight:700'>
        <td>TOTAL</td>
        <td style='color:{C['green']}'>{_fmt(tot_gains)}</td>
        <td style='color:{C['red']}'>{_fmt(tot_losses)}</td>
        <td style='color:{_pnl_color(tot_net_r)}'>{_fmt(tot_net_r)}</td>
        <td style='color:{C['orange']}'>{_fmt(tot_costs)}</td>
        <td style='color:{_pnl_color(tot_unreal)}'>{_fmt(tot_unreal)}</td>
        <td style='color:{_pnl_color(tot_pnl)}'>{_fmt(tot_pnl)}</td>
        <td style='color:{_pnl_color(tot_pnl)}'>{tot_pnl / (settings.INITIAL_CAPITAL * len(filt_keys)) * 100:+.2f}%</td>
    </tr>"""

    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;margin-top:8px">
    <table class="perf-table"><thead><tr>{th_cells}</tr></thead><tbody>{trows}</tbody></table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: STRATEGY GUIDE
# ─────────────────────────────────────────────────────────────────────────────
GUIDE = [
    {
        "key": "s01_momentum_dip",
        "title": "S01 · Momentum Dip Buyer",
        "tags": [("Equity Long", "equity"), ("Mean Reversion", "equity"), ("7-Day Hold", "struct")],
        "what": "Buys S&P 500 + NASDAQ-100 stocks (~517 names) that dip 3–8% on high volume in a single day, while the stock is still in a long-term uptrend. Expects a quick bounce within a week.",
        "edge": "Healthy stocks in uptrends occasionally get hit by panic sellers, stop-loss cascades, or margin calls — not because anything fundamentally changed. Institutions tend to step in and buy these dips, pushing the price back up. The strategy captures that mechanical recovery.",
        "entry": [
            "Stock must be above its 200-day moving average (long-term uptrend intact)",
            "Single-day drop between -3% and -8% (big enough to be a real dip, small enough to not be a crash)",
            "Today's trading volume is at least 1.5× the 20-day average (confirms active selling, not just a quiet drift down)",
            "RSI indicator below 40 (stock is technically oversold — stretched too far down)",
        ],
        "exit_stop": "-3.0%", "exit_target": "+2.5%", "exit_time": "7 days",
        "sizing": "2% of portfolio per stock, up to 10 stocks simultaneously (~20% total deployed at once)",
        "notes": "Works best when VIX is between 15–25. Very high VIX dips may not bounce as quickly.",
    },
    {
        "key": "s02_cross_asset_mom",
        "title": "S02 · Cross-Asset Momentum",
        "tags": [("Macro", "macro"), ("Trend Following", "macro"), ("Monthly", "struct")],
        "what": "Each month, selects the 3 best-performing asset classes (stocks, bonds, gold, commodities, etc.) based on their 12-month trend and invests equally in them. Moves to safety (short-term treasuries) when nothing is trending up.",
        "edge": "Asset class trends persist for months, not days — winners keep winning. By only owning assets with positive absolute momentum, the strategy naturally avoids bear markets and crashes. Lower turnover means lower costs.",
        "entry": [
            "Rank 10 ETFs (SPY, QQQ, IWM, GLD, TLT, DBC, VNQ, EFA, IEMG, SHY) by 12-month return minus most recent 1-month return",
            "Only buy assets with a positive 12-month return (absolute momentum filter — avoids downtrends)",
            "Buy the top 3 qualifying assets in equal weight",
            "If nothing qualifies (all assets in downtrend), go 100% short-term treasuries (SHY)",
        ],
        "exit_stop": "None", "exit_target": "None", "exit_time": "Monthly rebalance",
        "sizing": "Equal weight across top 3 assets, max 50% per asset, 95% of portfolio deployed",
        "notes": "This strategy doesn't use stop-losses — the monthly signal itself will exit a losing position when it falls out of the top 3.",
    },
    {
        "key": "s03_factor_alpha",
        "title": "S03 · Factor Alpha",
        "tags": [("Equity Long", "equity"), ("Multi-Factor", "equity"), ("60-Day Hold", "struct")],
        "what": "Owns the top 20 stocks from the S&P 500 + NASDAQ-100 universe (~517 names) scoring highest on two academically proven factors: strong 12-month price momentum AND low volatility. Rebalances weekly.",
        "edge": "Decades of academic research across 50+ countries confirm these two factors generate persistent excess returns. Combining them reduces drawdowns (low-vol dampens momentum's crashes) while preserving alpha.",
        "entry": [
            "Score each stock in the S&P 500 + NASDAQ-100 universe on momentum: 12-month return minus last month (skip-1 momentum), z-scored across the universe",
            "Score each stock on low volatility: negative of 90-day realized volatility, z-scored",
            "Combine: 60% momentum score + 40% low-vol score",
            "Buy the top 20 stocks weekly",
        ],
        "exit_stop": "-7.0%", "exit_target": "None (let winners run)", "exit_time": "60 days",
        "sizing": "Signal-weighted sizing, max 8% per stock, 80% of portfolio deployed. No profit target — momentum winners are held until the factor signal decays.",
        "notes": "Only stop-loss and time stop exist — there is intentionally no profit target because momentum stocks tend to keep rising.",
    },
    {
        "key": "s04_earnings_drift",
        "title": "S04 · Post-Earnings Drift (PEAD)",
        "tags": [("Equity Long", "equity"), ("Event-Driven", "event"), ("30-Day Hold", "struct")],
        "what": "Scans S&P 500 + NASDAQ-100 (~517 stocks) for names that gap up 3%+ on more than double their normal volume on earnings day — a strong signal that results beat expectations. Holds for up to 30 days while the market slowly reprices.",
        "edge": "Post-Earnings Announcement Drift (PEAD) is one of the most documented anomalies in finance: markets underreact to earnings surprises. Stocks with positive surprise continue drifting up for weeks as slow-moving institutional investors accumulate.",
        "entry": [
            "Stock gaps up 3% or more in a single day (earnings surprise proxy)",
            "That day's volume is at least 2× the 20-day average (confirms institutional reaction, not noise)",
            "Signal stays active for 30 days from the gap-up date",
        ],
        "exit_stop": "-5.0%", "exit_target": "+10.0%", "exit_time": "30 days",
        "sizing": "Signal-weighted, max 10% per stock, up to 10 concurrent positions, 80% deployed",
        "notes": "The strategy trims positions during rebalance as weights drift. A small trim is NOT a full exit — only a full sell brings a position to zero.",
    },
    {
        "key": "s05_short_term_reversal",
        "title": "S05 · Short-Term Reversal",
        "tags": [("Equity Long", "equity"), ("Mean Reversion", "equity"), ("10-Day Hold", "struct")],
        "what": "Buys S&P 500 + NASDAQ-100 stocks (~517 names) that crash 10–20% over 5 trading days due to forced selling, then bets on a bounce within 10 days. Scans daily for new opportunities but never disturbs existing positions.",
        "edge": "Jegadeesh (1990) documented that 1-week losers outperform 1-week winners by ~1.7%/month. The driver: margin calls, ETF rebalancing, and stop-loss cascades create temporary prices below fair value — not because the company deteriorated. These reverse when the mechanical selling ends.",
        "entry": [
            "5-day return between -10% and -20% (too small = noise; too large = real fundamental problem)",
            "Volume on the down days above 1.2× average (confirms selling is liquidity-driven, not informational)",
            "Not during an earnings week (avoids buying stocks with genuine bad news)",
            "Daily scan: new qualifying stocks are added each day as new entries only",
        ],
        "exit_stop": "-5.0%", "exit_target": "+5.0%", "exit_time": "10 days",
        "sizing": "Signal-weighted, max 12% per stock, 65% deployed (reserves capital for daily new entries). Never trims or rebalances existing positions — exits only via stop/target/time.",
        "notes": "Works best in high-VIX environments (>20) where forced selling is elevated. In calm markets, fewer qualifying signals appear.",
    },
    {
        "key": "s06_vix_term_structure",
        "title": "S06 · VIX Term Structure Carry",
        "tags": [("Macro", "macro"), ("Volatility", "vol"), ("Daily Regime", "struct")],
        "what": "Reads the 'shape' of the fear curve — comparing short-term fear (VIX9D, 9-day) to medium-term fear (VIX, 30-day) — to decide each day whether to be in stocks, bonds, or gold.",
        "edge": "VIX futures are almost always in contango (future fear priced higher than current). When the curve is steep, short-term fear is subdued and stocks thrive. When the curve inverts (short-term panic spikes above long-term), a flight to safety is warranted.",
        "entry": [
            "Calculate the 'roll' = (VIX − VIX9D) / VIX9D and z-score it over 60 days",
            "Steep contango (z > 1.0): go 100% SPY — markets calm, carry stocks",
            "Near-term panic / backwardation (z < -1.0): go 60% short-term treasuries + 40% gold",
            "Neutral zone: 60% SPY + 40% long-term bonds (TLT)",
            "Extreme fear override (VIX > 28 AND curve inverted): go defensive immediately",
        ],
        "exit_stop": "Regime change", "exit_target": "Regime change", "exit_time": "Daily rebalance",
        "sizing": "Fixed weights per regime: 40-100% SPY in calm, 60% SHY + 40% GLD in panic",
        "notes": "No per-position stop-losses. The regime signal itself is the risk manager.",
    },
    {
        "key": "s07_macro_regime",
        "title": "S07 · Macro Regime Switcher",
        "tags": [("Macro", "macro"), ("Regime", "macro"), ("Low Turnover", "struct")],
        "what": "Switches between full-risk (QQQ), balanced (SPY + bonds), and defensive (treasuries + gold) allocations based on 4 macro signals: VIX level, yield curve shape, stock market trend, and credit market health.",
        "edge": "Macro regimes persist for weeks to months, not days. Being fully in equities during a prolonged bear market can destroy years of returns. This strategy avoids those regimes by monitoring 4 independent signals, triggering a regime change only when multiple confirm.",
        "entry": [
            "Risk-On (all 4 green: VIX < 18, positive yield curve, SPY above 200-day MA, credit spreads benign): 100% QQQ",
            "Neutral (mixed signals — default): 60% SPY + 40% long-term bonds",
            "Risk-Off (VIX > 28 OR inverted yield curve + SPY downtrend): 60% short-term treasuries + 40% gold",
            "Credit tiebreaker: HYG/LQD ratio momentum can upgrade or downgrade equity allocation",
        ],
        "exit_stop": "Regime change only", "exit_target": "Regime change only", "exit_time": "Weekly check (min turnover)",
        "sizing": "Fixed weights per regime. Only 2 positions at a time. Very low turnover by design.",
        "notes": "Designed to be boring most of the time. Regime changes are infrequent — the strategy earns by avoiding crashes, not by trading often.",
    },
    {
        "key": "s09_dollar_carry",
        "title": "S09 · Dollar Carry & FX Momentum",
        "tags": [("Macro", "macro"), ("FX", "macro"), ("Monthly", "struct")],
        "what": "Uses the direction of the US dollar (UUP ETF) as a barometer for global risk appetite, then positions in the appropriate assets: dollar strengthening = go defensive; dollar weakening = go international and commodities.",
        "edge": "The dollar is the world's reserve currency — when it strengthens, global liquidity tightens and risk assets sell off. When it weakens, international stocks and commodities thrive. This is one of the most reliable macro relationships documented across 50+ years.",
        "entry": [
            "Compare UUP (dollar ETF) 20-day vs 60-day momentum",
            "Dollar strengthening: go 40% UUP + 30% commodities (DBC) + 30% short-term treasuries",
            "Dollar weakening + international stocks rising (EFA positive): 30% European stocks + 30% EM + 20% gold + 20% commodities",
            "Dollar weakening + international stocks flat/down: balanced 25% each in EFA/SPY/GLD/DBC",
        ],
        "exit_stop": "Regime change", "exit_target": "Regime change", "exit_time": "Monthly rebalance",
        "sizing": "Fixed weights per regime, 4 positions max, fully invested",
        "notes": "Low trading frequency reduces transaction costs. The strategy rebalances once a month unless the dollar regime flips dramatically.",
    },
    {
        "key": "s10_vol_surface",
        "title": "S10 · Volatility Risk Premium Harvest",
        "tags": [("Volatility", "vol"), ("Carry", "vol"), ("Daily Regime", "struct")],
        "what": "Profits from the persistent gap between how much fear the options market prices in (implied volatility / VIX) and how much volatility actually materializes (realized volatility). When options are overpriced, the strategy collects that premium by being positioned against excessive fear.",
        "edge": "Implied volatility is consistently 3–5 points higher than realized volatility over time — a well-documented Variance Risk Premium (VRP). Like an insurance company charging more than expected claims, this strategy earns by selling overpriced fear.",
        "entry": [
            "Calculate VRP = VIX / 100 (implied vol) minus 21-day realized volatility",
            "Z-score VRP over 90 days",
            "Very expensive options (z > 1.5): 95% SPY — short vol implicit, very long equities",
            "Cheap options / market stressed (z < -1.0): 80% SPY + 5% VXX protection + 15% treasuries",
            "Neutral: 90% SPY + 10% treasuries",
            "Hard emergency stop: if VIX spikes more than 5 points in a single day, exit all risk immediately",
        ],
        "exit_stop": "VIX spike > 5 pts/day", "exit_target": "Regime change", "exit_time": "Daily rebalance",
        "sizing": "Fixed per regime, max 5% in volatility instruments, primarily SPY exposure",
        "notes": "The VIX spike stop is critical — in a market crash, being short volatility without a hard stop can cause catastrophic losses in hours.",
    },
    {
        "key": "s11_congressional",
        "title": "S11 · Congressional Trade Follower",
        "tags": [("Event-Driven", "event"), ("Alternative Data", "flow"), ("60-Day Hold", "struct")],
        "what": "Mirrors stock purchases made by US Congress members. Politicians must legally disclose trades within 45 days, and academic research shows they consistently earn excess returns — likely from policy information advantages.",
        "edge": "Ziobrowski (2004, 2011) found House members earn ~6–10% excess annual returns; Senators earn ~12%. While some of this edge has narrowed post-STOCK Act, the signal remains statistically significant, especially for Senators near key policy committees.",
        "entry": [
            "Fetch daily House and Senate trade disclosures from public sources",
            "Filter for purchases only (sales are less informative — often forced by diversification requirements)",
            "Score each stock: freshness of disclosure (recent = better) × transaction size × number of distinct politicians buying",
            "Senators get 50% more weight than House members (historically stronger signal)",
            "Buy top 10 scored stocks",
        ],
        "exit_stop": "-5.0%", "exit_target": "+15.0%", "exit_time": "60 days",
        "sizing": "Signal-weighted, max 15% per stock, up to 10 positions, 85% deployed",
        "notes": "Data feed can fail (House/Senate websites go down). When data is unavailable, the strategy stays flat rather than guessing.",
    },
    {
        "key": "s12_index_inclusion",
        "title": "S12 · Index Inclusion Front-Run",
        "tags": [("Event-Driven", "event"), ("Structural", "struct"), ("30-Day Hold", "struct")],
        "what": "Buys stocks newly announced for S&P 500 inclusion before the massive wave of index fund buying that must happen when they officially join. Sells after the passive buying pressure is absorbed.",
        "edge": "When a stock joins the S&P 500, every index fund in the world must buy it — at any price. This creates a predictable, mechanical demand event. Front-running this flow historically generates 3–5% in the window between announcement and effective date.",
        "entry": [
            "Monitor S&P 500 additions (announcement typically 5 trading days before effective date)",
            "Buy within 2 days of announcement; signal peaks on the effective date then fades",
            "Also tracks Russell 1000 stocks approaching S&P 500 eligibility (>20% 1-year return, near 52-week high) for pre-announcement drift",
        ],
        "exit_stop": "-3.0%", "exit_target": "+5.0%", "exit_time": "30 days",
        "sizing": "Signal-weighted, max 15% per name, up to 8 positions, 70% deployed",
        "notes": "The signal inverts after inclusion (score goes negative) — the strategy exits as passive buying is complete and 'sell the news' dynamics emerge.",
    },
    {
        "key": "s13_pre_earnings_drift",
        "title": "S13 · Pre-Earnings Drift",
        "tags": [("Event-Driven", "event"), ("Earnings", "event"), ("5-Day Hold", "struct")],
        "what": "Scans S&P 500 + NASDAQ-100 (~517 stocks) for high-quality earnings-beaters in the 3–7 trading days before their earnings announcement, then exits the day before to avoid overnight announcement risk. No earnings gap risk.",
        "edge": "Barber et al. (2013) documents ~1.5–2.5% average return in the 5 days before earnings for consistent earnings-beaters. Institutional investors quietly accumulate positions ahead of announcements they expect to be positive, lifting the price before the news breaks.",
        "entry": [
            "Next earnings date is 3–7 trading days away",
            "Stock has beaten earnings estimates in at least 60% of the last 8 quarters",
            "Bonus: both of the last 2 quarters were positive surprises",
            "Stock is in an uptrend (positive 60-day return — institutional interest present)",
            "Score increases as the earnings date approaches (signal strongest 3 days out)",
        ],
        "exit_stop": "Day before earnings", "exit_target": "+10.0%", "exit_time": "Day before earnings",
        "sizing": "Signal-weighted, max 12% per stock, up to 8 positions, 70% deployed",
        "notes": "The strategy always exits the day before earnings — never holds through an announcement. This is by design: the edge is in the pre-announcement drift, not the earnings reaction.",
    },
    {
        "key": "s14_gamma_wall",
        "title": "S14 · Gamma Wall (Options Market Structure)",
        "tags": [("Derivatives", "vol"), ("Market Structure", "struct"), ("Daily Regime", "struct")],
        "what": "Reads the options market to determine whether the stock market is likely to stay pinned near a level (low vol) or break out dramatically (high vol), then positions accordingly.",
        "edge": "Options market makers must continuously delta-hedge their books. When they hold large positive gamma positions (net long options), they act as shock absorbers — buying dips and selling rallies — which suppresses volatility and pins prices. When they're net short gamma, they amplify moves. GEX (Gamma Exposure) measures this.",
        "entry": [
            "Calculate SPY Gamma Exposure (GEX) from the options chain across 3 nearest expiries",
            "Positive GEX / pinning regime (VIX < 15 proxy): 70% SPY + 30% QQQ — low vol environment, own equities",
            "Negative GEX / trending regime (VIX > 25 proxy): 30% UVXY (vol ETF) + 40% SPY + 30% gold — expect big moves",
            "Neutral regime: 60% SPY + 40% short-term treasuries",
        ],
        "exit_stop": "Regime change", "exit_target": "Regime change", "exit_time": "Daily rebalance",
        "sizing": "Fixed weights per regime, 3 positions max",
        "notes": "Most retail investors have never heard of GEX. It's a quant tool used by institutional options desks to predict market pinning. UVXY is a 1.5× leveraged VIX ETF — only held in trending/high-vol regimes.",
    },
    {
        "key": "s15_short_flow",
        "title": "S15 · Institutional Short Flow",
        "tags": [("Alternative Data", "flow"), ("Contrarian", "equity"), ("30-Day Hold", "struct")],
        "what": "Uses FINRA's daily short volume data to detect when institutions are aggressively shorting a stock. Plays the contrarian squeeze when extreme pessimism meets a big price drop — historically a setup for rapid reversals.",
        "edge": "When short interest is extreme (>68% of volume is short) AND the stock has already dropped significantly, the setup is ripe for a short squeeze: any positive news or buying pressure forces shorts to cover, amplifying the move upward.",
        "entry": [
            "Extreme contrarian buy: short ratio ≥ 68% of daily volume AND stock down >5% over 5 days → score 0.8 (squeeze setup)",
            "Momentum confirmation: short ratio ≤ 38% AND stock rising → score 0.5 (low shorts, price going up = clean trend)",
            "Sustained bearish selling (3-day avg ≥ 55%): negative signal — avoid or short",
            "Fallback (no FINRA data): uses volume-ratio + price-return proxy",
        ],
        "exit_stop": "-6.0%", "exit_target": "+12.0%", "exit_time": "30 days",
        "sizing": "Signal-weighted, max 20% per stock, up to 8 positions, 80% deployed",
        "notes": "Short squeezes can be violent and fast — hence the wide +12% profit target to capture the full move before short-covering exhausts itself.",
    },
    {
        "key": "s16_overnight_carry",
        "title": "S16 · Overnight Carry",
        "tags": [("Structural", "struct"), ("Market Microstructure", "flow"), ("Daily", "struct")],
        "what": "Captures the well-documented 'overnight premium': nearly all of the stock market's long-run returns occur between market close and the next morning's open — not during the trading day itself.",
        "edge": "Lou, Polk & Skouras (2019) documented that 100%+ of the S&P 500's long-run returns occurred close-to-open over 90 years of data. Institutional investors accumulate orders during the day and execute at the open, paying a premium for overnight gap risk that retail investors can harvest.",
        "entry": [
            "Always hold overnight when SPY is above its 50-day moving average (uptrend confirmed)",
            "In uptrend: 60% SPY + 40% QQQ — maximize overnight equity exposure",
            "Below 50-day MA (downtrend): shift to 40% SPY + 30% gold + 30% long bonds — defensive overnight carry",
            "Scale back when VIX > 20 (reduce equity portion); scale back further at VIX > 28",
            "Full exit if VIX ≥ 35 (tail risk too high to hold overnight)",
        ],
        "exit_stop": "VIX ≥ 35 full exit", "exit_target": "None", "exit_time": "Daily rebalance",
        "sizing": "40-60% equities, rest in defensive assets. Normalized to 100% invested.",
        "notes": "This strategy essentially holds positions permanently — the 'overnight carry' is structural, not a trade. It earns by being in the right mix of assets at each close, not by timing entry/exit.",
    },
    {
        "key": "s19_turn_of_month",
        "title": "S19 · Turn-of-Month",
        "tags": [("Seasonal", "macro"), ("Calendar", "struct"), ("Low Turnover", "struct")],
        "what": "Holds 100% SPY during the last 4 and first 3 trading days of each month (the 'turn-of-month window'), and holds 100% SHY (short-term treasuries) for the rest of the month. Skips the equity position entirely when VIX > 30.",
        "edge": "One of the most replicated seasonal anomalies in finance (Ariel 1987, Lakonishok & Smidt 1988). Institutional rebalancing, pension fund cash deployment, 401k contributions, and fund manager window dressing create predictable buying pressure in the ~8-day turn-of-month window. Roughly 0.4–0.6% premium per cycle, ~9 cycles/year, with minimal transaction costs due to very low turnover.",
        "entry": [
            "Last 4 trading days of the month: buy SPY at close",
            "First 3 trading days of the next month: stay long SPY",
            "All other days: hold SHY (short-term treasuries, ~cash equivalent)",
            "VIX gate: if VIX > 30, skip SPY and stay in SHY — macro crises overwhelm the seasonal",
        ],
        "exit_stop": "None (calendar-driven exit only)", "exit_target": "None", "exit_time": "Close of trading day +3 of new month",
        "sizing": "100% SPY in window, 100% SHY out of window. Single position at all times.",
        "notes": "Extremely low turnover — only 2 trades per month-turn cycle (~18 trades/year). The VIX gate adds robustness during crisis periods where the seasonal premium historically disappears.",
    },
]

with tab_guide:
    st.markdown(f"""
    <div style="font-size:14px;color:{C['muted']};margin-bottom:20px;line-height:1.7;">
        Plain-language breakdown of every strategy — what it does, why it works, and how it manages risk.
        Written so you can explain each one to someone with no finance background.
    </div>
    """, unsafe_allow_html=True)

    for g in GUIDE:
        pv_g  = states.get(g["key"], {}).get("portfolio_value", START_CASH)
        ret_g = (pv_g / START_CASH - 1) * 100
        ret_arrow = "▲" if ret_g >= 0 else "▼"
        ret_col_g = C["green"] if ret_g >= 0 else C["red"]
        label_g = f"{g['title']}  ·  " \
                  f"<span style='color:{ret_col_g};font-weight:700;'>{ret_arrow} {ret_g:+.2f}%</span>"

        with st.expander(g["title"] + f"   {ret_arrow} {ret_g:+.2f}%", expanded=False):
            # Tags
            tags_html = "".join(
                f'<span class="guide-tag gt-{cls}">{name}</span>'
                for name, cls in g["tags"]
            )
            st.markdown(f'<div style="margin-bottom:14px;">{tags_html}</div>',
                        unsafe_allow_html=True)

            # What it does
            st.markdown(f'<div class="guide-h">What it does</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="guide-body">{g["what"]}</div>', unsafe_allow_html=True)

            # Why it works
            st.markdown(f'<div class="guide-h">Why it works</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="guide-body">{g["edge"]}</div>', unsafe_allow_html=True)

            # Entry rules
            st.markdown(f'<div class="guide-h">Entry rules — all must be true to open a position</div>',
                        unsafe_allow_html=True)
            bullets = "".join(f"<li style='margin-bottom:6px;color:#e2e8f0;font-size:14px;'>{e}</li>"
                              for e in g["entry"])
            st.markdown(f'<ul style="margin:4px 0 0 0;padding-left:20px;line-height:1.7;">'
                        f'{bullets}</ul>', unsafe_allow_html=True)

            # Exit rules
            st.markdown(f'<div class="guide-h">Exit rules</div>', unsafe_allow_html=True)
            sl_col  = C["red"]   if g["exit_stop"]   != "None" else C["muted"]
            pt_col  = C["green"] if g["exit_target"]  != "None" else C["muted"]
            ts_col  = C["cyan"]
            st.markdown(f"""
            <div class="exit-grid">
              <div class="exit-cell">
                <div class="exit-cell-label">Stop-Loss</div>
                <div class="exit-cell-val exit-stop" style="color:{sl_col};">{g["exit_stop"]}</div>
                <div style="font-size:11px;color:{C['muted']};margin-top:4px;">
                  Cut the loss — thesis is wrong
                </div>
              </div>
              <div class="exit-cell">
                <div class="exit-cell-label">Profit Target</div>
                <div class="exit-cell-val exit-target" style="color:{pt_col};">{g["exit_target"]}</div>
                <div style="font-size:11px;color:{C['muted']};margin-top:4px;">
                  Lock in gains — don't get greedy
                </div>
              </div>
              <div class="exit-cell">
                <div class="exit-cell-label">Time Stop</div>
                <div class="exit-cell-val exit-time">{g["exit_time"]}</div>
                <div style="font-size:11px;color:{C['muted']};margin-top:4px;">
                  Exit regardless — signal expired
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Sizing & risk
            st.markdown(f'<div class="guide-h">Sizing & risk management</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="guide-body">{g["sizing"]}</div>', unsafe_allow_html=True)

            # Notes
            if g.get("notes"):
                st.markdown(f"""
                <div style="background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.2);
                            border-radius:8px;padding:12px 16px;margin-top:14px;">
                  <span style="font-size:11px;font-weight:700;color:{C['cyan']};
                               text-transform:uppercase;letter-spacing:0.8px;">Note  </span>
                  <span style="font-size:13px;color:{C['muted']};">{g['notes']}</span>
                </div>
                """, unsafe_allow_html=True)
