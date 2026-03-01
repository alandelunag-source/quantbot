"""
Quantbot Paper Trading Dashboard  ·  venv\Scripts\streamlit.exe run dashboard.py
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
import yfinance as yf

# ── Constants & config ────────────────────────────────────────────────────────
STATE_DIR   = Path(__file__).parent / "state"
START_CASH  = 100_000

STRATEGIES  = [
    "s01_momentum_dip", "s02_cross_asset_mom", "s03_factor_alpha",
    "s04_earnings_drift", "s05_short_term_reversal", "s06_vix_term_structure",
    "s07_macro_regime", "s09_dollar_carry", "s10_vol_surface",
    "s11_congressional", "s12_index_inclusion", "s13_pre_earnings_drift",
    "s14_gamma_wall", "s15_short_flow", "s16_overnight_carry", "s17_panic_reversal",
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
    "s17_panic_reversal":      "S17 · Panic Reversal",
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


@st.cache_data(ttl=3600)
def fetch_spy(start: str, end: str) -> pd.Series:
    try:
        raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"]
        # yfinance returns MultiIndex columns for single ticker — flatten
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        sq = close.squeeze()
        # squeeze() on 1-row DataFrame gives scalar — wrap back into Series
        if not isinstance(sq, pd.Series):
            sq = pd.Series([float(sq)], index=raw.index)
        return sq.dropna()
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
    return df.sort_values("date")


def build_summary(states: dict, spy_series: pd.Series) -> pd.DataFrame:
    rows = []
    for s, st_ in states.items():
        pv        = st_.get("portfolio_value", START_CASH)
        peak      = st_.get("peak_value", pv)
        positions = st_.get("positions", {})
        trades    = st_.get("trades", [])
        daily     = st_.get("daily_log", [])
        start_dt  = st_.get("start_date", "")[:10]

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
tab_ov, tab_eq, tab_bench, tab_pos, tab_trades = st.tabs([
    "  Overview  ", "  Equity Curves  ", "  Benchmark vs SPY  ",
    "  Positions  ", "  Trades  ",
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

    tbl = filt_summary[["Strategy","Value","Invested","Cash","Return","Alpha","SPY Ret","Max DD","Sharpe","Positions","Trades"]].copy()
    tbl.columns = ["Strategy","Value ($)","Invested ($)","Cash ($)","Return (%)","Alpha (%)","SPY (%)","Max DD (%)","Sharpe","Pos","Trades"]

    # Format helpers
    def fmt_pct(v):
        if pd.isna(v): return "—"
        color = C["green"] if v > 0 else (C["red"] if v < 0 else C["muted"])
        sym   = "▲" if v > 0 else ("▼" if v < 0 else "")
        return f'<span style="color:{color};font-weight:600;">{sym}{abs(v):.3f}%</span>'

    def fmt_sharpe(v):
        if v is None or pd.isna(v): return "—"
        color = C["green"] if v > 1 else (C["orange"] if v > 0 else C["red"])
        return f'<span style="color:{color};">{v:.2f}</span>'

    def fmt_cash(v):
        color = C["muted"] if v > 1000 else C["orange"]   # flag low cash
        return f'<span style="color:{color};">${v:,.0f}</span>'

    html_rows = ""
    for _, row in tbl.iterrows():
        html_rows += f"""
        <tr>
            <td style="color:{C['text']};font-weight:500;">{row['Strategy']}</td>
            <td style="font-family:'Courier New',monospace;">${row['Value ($)']:,.0f}</td>
            <td style="color:{C['cyan']};font-family:'Courier New',monospace;">${row['Invested ($)']:,.0f}</td>
            <td>{fmt_cash(row['Cash ($)'])}</td>
            <td>{fmt_pct(row['Return (%)'])}</td>
            <td>{fmt_pct(row['Alpha (%)'])}</td>
            <td style="color:{C['muted']};">{row['SPY (%)']:+.3f}%</td>
            <td style="color:{C['red'] if row['Max DD (%)'] < 0 else C['muted']};">{row['Max DD (%)']:.3f}%</td>
            <td>{fmt_sharpe(row['Sharpe'])}</td>
            <td style="color:{C['cyan'] if row['Pos']>0 else C['muted']};">{int(row['Pos'])}</td>
            <td style="color:{C['muted']};">{int(row['Trades'])}</td>
        </tr>"""

    header_cols = ["Strategy","Value ($)","Invested ($)","Cash ($)","Return","Alpha","SPY","Max DD","Sharpe","Pos","Trades"]
    th = "".join(f"<th>{c}</th>" for c in header_cols)

    st.markdown(f"""
    <style>
    .perf-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .perf-table th {{
        background:{C['border']}; color:{C['muted']}; padding:10px 14px;
        text-align:left; font-size:11px; font-weight:700;
        text-transform:uppercase; letter-spacing:0.7px;
        border-bottom:1px solid {C['cyan']};
    }}
    .perf-table td {{
        padding:9px 14px; border-bottom:1px solid {C['border']};
        color:{C['text']}; font-size:13px; font-family:'Courier New',monospace;
    }}
    .perf-table tr:hover td {{ background:{C['border']}; }}
    .perf-table tr:last-child td {{ border-bottom:none; }}
    </style>
    <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
    <table class="perf-table"><thead><tr>{th}</tr></thead><tbody>{html_rows}</tbody></table>
    </div>
    """, unsafe_allow_html=True)

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
                "Shares":     p["shares"],
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
    all_trades = []
    for k in filt_keys:
        for t in states[k].get("trades", []):
            all_trades.append({
                "Strategy": SHORT.get(k, k),
                "Date":     t.get("date",""),
                "Ticker":   t.get("ticker",""),
                "Action":   t.get("action",""),
                "$ Value":  t.get("dollar_value", 0),
                "Cost $":   t.get("cost", 0),
            })

    if not all_trades:
        st.info("No trades recorded yet.")
    else:
        td = pd.DataFrame(all_trades).sort_values(["Date","Strategy"], ascending=[False,True])

        col_tv, col_by = st.columns(2)
        with col_tv:
            st.markdown('<div class="section-hdr">Volume by Strategy</div>', unsafe_allow_html=True)
            tvol = td.groupby("Strategy")["$ Value"].sum().sort_values(ascending=True)
            fig_tv = go.Figure(go.Bar(
                x=tvol.values, y=tvol.index, orientation="h",
                marker=dict(color=C["purple"], opacity=0.85, line=dict(width=0)),
                text=[f"${v:,.0f}" for v in tvol.values],
                textposition="outside",
                textfont=dict(size=10, color=C["muted"]),
            ))
            apply_layout(fig_tv, height=350, margin=dict(l=0,r=60,t=10,b=0),
                         yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                         xaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_tv, use_container_width=True)

        with col_by:
            st.markdown('<div class="section-hdr">Buy vs Sell Count</div>', unsafe_allow_html=True)
            action_ct = td.groupby(["Strategy","Action"]).size().unstack(fill_value=0)
            fig_bs = go.Figure()
            if "BUY" in action_ct.columns:
                fig_bs.add_trace(go.Bar(name="BUY", x=action_ct.index,
                                        y=action_ct["BUY"], marker_color=C["green"]))
            if "SELL" in action_ct.columns:
                fig_bs.add_trace(go.Bar(name="SELL", x=action_ct.index,
                                        y=action_ct["SELL"], marker_color=C["red"]))
            fig_bs.update_layout(barmode="group")
            apply_layout(fig_bs, height=350, margin=dict(l=0,r=0,t=10,b=0),
                         yaxis_title="Count",
                         xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                         legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_bs, use_container_width=True)

        # Trades table
        st.markdown('<div class="section-hdr">Trade Log</div>', unsafe_allow_html=True)
        trade_rows = ""
        for _, row in td.iterrows():
            act_color = C["green"] if row["Action"]=="BUY" else C["red"]
            trade_rows += f"""
            <tr>
                <td style="color:{C['cyan']};">{row['Strategy']}</td>
                <td style="color:{C['muted']};">{row['Date']}</td>
                <td style="font-weight:700;">{row['Ticker']}</td>
                <td style="color:{act_color};font-weight:700;">{row['Action']}</td>
                <td>${row['$ Value']:,.0f}</td>
                <td style="color:{C['muted']};">${row['Cost $']:.2f}</td>
            </tr>"""
        tr_th = "".join(f"<th>{c}</th>" for c in
                        ["Strategy","Date","Ticker","Action","$ Value","Cost"])
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;">
        <table class="perf-table"><thead><tr>{tr_th}</tr></thead><tbody>{trade_rows}</tbody></table>
        </div>
        """, unsafe_allow_html=True)
