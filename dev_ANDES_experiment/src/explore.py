#!/usr/bin/env python3
"""
explore.py — interactive Plotly viewer for QSTS results and firm trajectories.

Complements viz.py (static matplotlib figures for the thesis) with zoom/pan/hover HTML
for inspecting the time series. Writes a self-contained .html next to each CSV, so you
can keep the interactive figure alongside the data exactly as before.

Auto-detects the file type from its columns:
  - trajectory CSV (has P_net): P_actual / P_forecast / P_net on top, BESS + curtailment below
  - QSTS CSV     (has pv_mw): PV / synchronous / interconnector + load, penetration, voltage band

Usage (in the ANDES venv; needs plotly):
    python src/explore.py                                   # every results CSV it can find
    python src/explore.py --csv results/trajectories/rule_RUGBYR1.csv
    python src/explore.py --cdn                             # smaller HTML, needs internet to view
Requires: pip install plotly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _x_axis(df: pd.DataFrame, n: int):
    """Datetime axis if the file carries usable timestamps, else hour-of-day."""
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
        if dt.notna().any():
            return dt, "time"
    col = "t" if "t" in df.columns else None
    idx = df[col].to_numpy() if col else np.arange(n)
    return idx * 5.0 / 60.0, "hour of day"


def _write(fig, out: Path, inline: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=("inline" if inline else "cdn"),
                   full_html=True)
    print(f"  wrote {out}")
    return out


def explore_trajectory(csv: Path, inline=True) -> Path:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    df = pd.read_csv(csv)
    x, xlab = _x_axis(df, len(df))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.65, 0.35],
                        subplot_titles=("POI power (pu of PV peak)", "BESS & curtailment (pu)"))
    fig.add_trace(go.Scatter(x=x, y=df.P_actual, name="P_actual (available PV)",
                             line=dict(color="orange", width=1)), row=1, col=1)
    if "P_forecast" in df:
        fig.add_trace(go.Scatter(x=x, y=df.P_forecast, name="P_forecast",
                                 line=dict(color="#16a2b8", width=1, dash="dot")), row=1, col=1)
    if "P_dfc" in df:
        fig.add_trace(go.Scatter(x=x, y=df.P_dfc, name="P_dfc (dispatch target)",
                                 line=dict(color="#999999", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df.P_net, name="P_net (delivered)",
                             line=dict(color="#1f4e79", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df.P_curtail, name="curtailed", fill="tozeroy",
                             line=dict(color="grey", width=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df.P_bess_cmd, name="BESS (+dis / −chg)",
                             line=dict(color="green", width=1)), row=2, col=1)
    fig.update_layout(title=f"Trajectory — {csv.stem}", hovermode="x unified",
                      template="plotly_white", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(title_text=xlab, rangeslider=dict(visible=True), row=2, col=1)
    return _write(fig, csv.with_suffix(".html"), inline)


def explore_qsts(csv: Path, inline=True) -> Path:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    df = pd.read_csv(csv)
    x, xlab = _x_axis(df, len(df))
    load = df.syn_mw + df.slack_mw + df.pv_mw - df.loss_mw

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.5, 0.22, 0.28],
                        subplot_titles=("Power (MW)", "PV penetration (%)", "Bus voltage (pu)"))
    fig.add_trace(go.Scatter(x=x, y=df.pv_mw, name="PV-BESS", line=dict(color="#f2c14e")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df.syn_mw, name="Synchronous", line=dict(color="#4d7ea8")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df.slack_mw, name="Interconnector", line=dict(color="#c0392b")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=load, name="Load", line=dict(color="black", dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df.pen * 100, name="penetration", fill="tozeroy",
                             line=dict(color="#27ae60")), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df.vmin, name="Vmin", line=dict(color="#8e44ad")), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=df.vmax, name="Vmax", line=dict(color="#8e44ad", dash="dot")), 3, 1)
    for y in (0.95, 1.05):
        fig.add_hline(y=y, line=dict(color="red", width=0.7, dash="dash"), row=3, col=1)
    fig.update_layout(title=f"QSTS — {csv.stem}", hovermode="x unified",
                      template="plotly_white", legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text=xlab, rangeslider=dict(visible=True), row=3, col=1)
    return _write(fig, csv.with_suffix(".html"), inline)


def compare_methods(date: str, inline=True) -> Path | None:
    """Overlay the FLEET delivered power (sum over the 9 buses, MW) for every method that
    has data for `date`: A1 raw PV, A2 rule, A3 agent, A4 MPC — one interactive axis."""
    import plotly.graph_objects as go
    import pv_data as P
    import trajectory as T

    plants = P.load_all()
    x = np.arange(C.INTERVALS_PER_DAY) * 5.0 / 60.0
    series = {}
    try:
        series["A1 raw PV"] = P.day_injection_matrix(plants, date, "scada").sum(axis=1)
    except Exception:
        pass
    for pref, lab in (("rule", "A2 rule"), ("agent", "A3 DRL agent"), ("mpc", "A4 MPC")):
        try:
            tj = T.load_profile_trajectories(C.RESULTS / "trajectories", pref)
            y = T.injection_from_trajectories(tj, date=date, column="P_net").sum(axis=1)
        except Exception:
            continue
        if float(np.nanmax(y)) < 1e-6:           # method has no data for this day -> skip
            print(f"  compare: '{pref}' has no energy on {date} (day not generated?) — skipped")
            continue
        series[lab] = y
    if len(series) < 2:
        print(f"  compare: need >=2 methods for {date}; found {list(series)}")
        return None
    colors = {"A1 raw PV": "#f4a259", "A2 rule": "#2a9d8f",
              "A3 DRL agent": "#1f4e79", "A4 MPC": "#9b5de5"}
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                                 line=dict(width=1.6, color=colors.get(name))))
    fig.update_layout(title=f"Fleet delivered power (P_net) by method — {date}",
                      xaxis_title="hour of day", yaxis_title="Total PV-BESS injection (MW)",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(rangeslider=dict(visible=True), range=[0, 24])
    return _write(fig, C.RESULTS / "figures" / f"compare_{date}.html", inline)


def explore(csv: Path, inline=True) -> Path | None:
    cols = pd.read_csv(csv, nrows=1).columns
    if "P_net" in cols:
        return explore_trajectory(csv, inline)
    if "pv_mw" in cols:
        return explore_qsts(csv, inline)
    print(f"  skip {csv.name} (unrecognized columns)")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Interactive Plotly viewer for QSTS / trajectory CSVs.")
    ap.add_argument("--csv", type=Path, default=None, help="a single CSV (else scan results/)")
    ap.add_argument("--compare", default=None, metavar="YYYY-MM-DD",
                    help="overlay fleet P_net for A1/A2/A3/A4 on one axis for this day")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from CDN (smaller HTML)")
    args = ap.parse_args(argv)
    try:
        import plotly  # noqa: F401
    except ImportError:
        sys.exit("plotly not installed. In the venv:  pip install plotly")

    inline = not args.cdn
    if args.compare:
        return 0 if compare_methods(args.compare, inline) else 1
    if args.csv:
        targets = [args.csv]
    else:
        targets = sorted(C.RESULTS.glob("qsts_*.csv")) + \
                  sorted((C.RESULTS / "trajectories").glob("*.csv"))
    made = [p for c in targets if (p := explore(c, inline))]
    if not made:
        print("No CSVs found. Run study_a_qsts.py / make_firm_trajectory.py first.")
        return 1
    print(f"Generated {len(made)} interactive HTML view(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
