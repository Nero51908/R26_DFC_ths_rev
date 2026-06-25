#!/usr/bin/env python3
"""
injection_drivers.py — Study A mining of the FLEET net-injection drivers (ANDES-free).

The transmission network sees the fleet's net injection; before any power flow, the
quantities that drive line-flow movement and voltage swing are the injection's level and
its time variability. This module builds the fleet net-injection series the QSTS would
inject (sum over the nine PV-BESS sites, MW) for each scenario and reports:

  peak_MW / mean_MW   fleet net injection level
  ramp_mean / ramp_p95  |Δ fleet MW| per 5-min  (the upstream driver of |Δ line MVA|)
  totvar_MW           Σ|Δ| over the day (total variation) — overall injection "jumpiness"

Framing (thesis): DFC's value is **dependability / schedulability** — letting the operator
treat PV-BESS as a *dispatchable generator* rather than a negative load (see
availability_by_plant for the firm-capacity-at-reliability and dependability metrics). These
injection drivers are the complementary *firmness-of-injection* view. NOTE: the MPC (A4) is
explicitly ramp-penalised in its QP, so it is the benchmark for raw injection SMOOTHNESS;
the DRL agent's edge is dependability, not smoothing (smoothing is a separable reward term
that can be added later). This module quantifies both so the distinction is explicit.

    python src/injection_drivers.py                 # period p2, all scenarios + agent vs MPC
    python src/injection_drivers.py --plot          # also write interactive Plotly HTML
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import pv_data as P
import trajectory as T

SCENARIOS = [("a1", "raw", "A1 raw PV"), ("a2", "rule", "A2 rule"),
             ("a4", "mpc", "A4 MPC"), ("a3", "agent", "A3 DRL agent")]
PREFIX = {s: p for s, p, _ in SCENARIOS}
LABEL = {s: l for s, _, l in SCENARIOS}
N = C.INTERVALS_PER_DAY


def _traj_days(traj: dict[str, pd.DataFrame]) -> set[str]:
    s = None
    for name in C.PROFILES:
        d = pd.to_datetime(traj[name].datetime, errors="coerce").dt.date
        days = {str(x) for x, c in d.value_counts().items() if c >= N}
        s = days if s is None else (s & days)
    return s or set()


def load(period):
    plants = P.load_all(period)
    traj = {pref: T.load_profile_trajectories(C.RESULTS / "trajectories", pref)
            for pref in ("rule", "agent", "mpc")}
    pv_days = {str(d.date()) for d in P.common_complete_days(plants)}
    common = sorted(pv_days & _traj_days(traj["rule"])
                    & _traj_days(traj["agent"]) & _traj_days(traj["mpc"]))
    return plants, traj, common


def fleet_series(scen: str, day, plants, traj) -> np.ndarray:
    """Fleet net injection (MW, 288) — sum over the nine sites, mirroring study_a_qsts."""
    if scen == "a1":
        inj = P.day_injection_matrix(plants, pd.Timestamp(day), column="scada")
    else:
        inj = T.injection_from_trajectories(traj[PREFIX[scen]], date=pd.Timestamp(day),
                                            column="P_net")
    return inj.sum(axis=1)


def _day_drivers(f: np.ndarray) -> dict:
    d = np.abs(np.diff(f))
    return {"peak": f.max(), "mean": f.mean(), "ramp_mean": d.mean(),
            "ramp_p95": np.percentile(d, 95), "totvar": d.sum()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Study A fleet net-injection drivers.")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    ap.add_argument("--compare", nargs=2, default=["a3", "a4"], metavar=("A", "B"),
                    help="scenarios for the paired daily test (default agent vs MPC)")
    ap.add_argument("--plot", action="store_true", help="also write interactive Plotly HTML")
    args = ap.parse_args(argv)

    plants, traj, common = load(args.period)
    # per-scenario, per-day drivers
    perday = {s: pd.DataFrame([_day_drivers(fleet_series(s, d, plants, traj)) for d in common],
                              index=common) for s, _, _ in SCENARIOS}
    rows = [{"scenario": LABEL[s], "peak_MW": round(perday[s]["peak"].mean()),
             "mean_MW": round(perday[s]["mean"].mean()),
             "ramp_mean_MW": round(perday[s]["ramp_mean"].mean(), 1),
             "ramp_p95_MW": round(perday[s]["ramp_p95"].mean(), 1),
             "totvar_MW": round(perday[s]["totvar"].mean())} for s, _, _ in SCENARIOS]
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 160)
    print(f"\nFleet net-injection drivers — period {args.period}, {len(common)} common days\n")
    print(df.to_string(index=False))

    a, b = args.compare
    print(f"\nPaired daily test, {LABEL[a]} vs {LABEL[b]} (lower ramp/totvar = firmer injection):")
    for metric in ("ramp_mean", "totvar"):
        x, y = perday[a][metric].to_numpy(), perday[b][metric].to_numpy()
        diff = x - y
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        print(f"  {metric:9s}: {LABEL[a]} {x.mean():.0f}  {LABEL[b]} {y.mean():.0f}  "
              f"diff {diff.mean():+.0f}  t={t:+.2f}  {LABEL[a]}-lower days={np.mean(x < y):.2f}")

    out = C.RESULTS / "injection_drivers.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  peak/mean = fleet net injection MW; ramp_* = |Δ fleet MW|/5-min; totvar = Σ|Δ|/day")
    print(f"  A4 MPC is ramp-penalised by construction (injection-smoothness benchmark);")
    print(f"  the DRL agent's value is dependability (see availability_by_plant), not smoothing.")
    print(f"  -> {out}")
    if args.plot:
        _plot(args.period, plants, traj, common, perday, out.with_suffix(".html"))
    return 0


def _plot(period, plants, traj, common, perday, out: Path):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  (plotly not installed: skipping HTML)")
        return
    # representative day = highest raw-PV total variation (most volatile)
    day = perday["a1"]["totvar"].idxmax()
    colors = {"a1": "#999999", "a2": "#ff7f0e", "a4": "#1f77b4", "a3": "#2ca02c"}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=(f"Fleet net injection — {day} (most volatile day)",
                                        "Daily total variation (all days)"))
    grid = pd.date_range(pd.Timestamp(day).normalize(), periods=N, freq="5min")
    for s, _, lab in SCENARIOS:
        fig.add_trace(go.Scatter(x=grid, y=fleet_series(s, day, plants, traj), name=lab,
                                 line=dict(color=colors[s])), row=1, col=1)
        fig.add_trace(go.Box(y=perday[s]["totvar"], name=lab, marker_color=colors[s],
                            boxmean=True, showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="Σ|Δ| MW/day", row=1, col=2)
    fig.update_layout(template="plotly_white", height=480, width=1150,
                      legend=dict(orientation="h", y=1.12),
                      title=f"Study A injection firmness — period {period}, {len(common)} days")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
