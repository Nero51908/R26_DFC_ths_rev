#!/usr/bin/env python3
"""
viz.py — standard figures for the QSTS / DFC study (matplotlib, headless-safe).

Reads the per-interval QSTS result CSVs (results/qsts_<scenario>_<date>.csv) and the
canonical trajectory CSVs, and writes PNGs to results/figures/.

Implemented figures:
  1. dispatch_stack  — intraday PV + synchronous + slack(import/export) vs load
  2. penetration     — instantaneous PV share of load over the day
  3. voltage_band    — min/max bus voltage over the day with limit lines
  4. balancing_bar   — fleet balancing duty (MWh) compared across scenarios A1/A3/A4
  5. poi_dispatch    — P_forecast vs P_actual vs P_DFC (+ curtailment/BESS) for a trajectory

Usage:
    python src/viz.py                       # auto: every results CSV it can find
    python src/viz.py --qsts results/qsts_a1_2024-04-26.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIGDIR = C.RESULTS / "figures"
H = lambda t: np.asarray(t) * 5.0 / 60.0          # interval index -> hours


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def dispatch_stack(qsts_csv: Path):
    df = pd.read_csv(qsts_csv)
    load = df.syn_mw + df.slack_mw + df.pv_mw - df.loss_mw
    t = H(df.t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(t, df.pv_mw, df.syn_mw,
                 labels=["PV-BESS", "Synchronous"], colors=["#f2c14e", "#4d7ea8"], alpha=.9)
    ax.plot(t, load, "k--", lw=1.2, label="Load")
    ax.plot(t, df.slack_mw, color="#c0392b", lw=1, label="Slack / interconnector")
    ax.axhline(0, color="grey", lw=.6)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (MW)")
    ax.set_title(f"Intraday dispatch — {qsts_csv.stem}")
    ax.legend(loc="upper right", fontsize=8); ax.set_xlim(0, 24)
    return _save(fig, f"dispatch_{qsts_csv.stem}.png")


def penetration(qsts_csv: Path):
    df = pd.read_csv(qsts_csv)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(H(df.t), df.pen * 100, color="#27ae60", alpha=.85)
    ax.axhline(44, color="k", ls=":", lw=1, label="44% annual-avg target")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("PV penetration (%)")
    ax.set_title(f"Instantaneous PV penetration — {qsts_csv.stem}")
    ax.legend(fontsize=8); ax.set_xlim(0, 24); ax.set_ylim(0, max(100, df.pen.max()*100*1.1))
    return _save(fig, f"penetration_{qsts_csv.stem}.png")


def voltage_band(qsts_csv: Path):
    df = pd.read_csv(qsts_csv)
    t = H(df.t)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(t, df.vmin, df.vmax, color="#8e44ad", alpha=.25, label="bus V range")
    ax.plot(t, df.vmin, color="#8e44ad", lw=1); ax.plot(t, df.vmax, color="#8e44ad", lw=1)
    for y in (0.95, 1.05):
        ax.axhline(y, color="r", ls="--", lw=.8)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Voltage (pu)")
    ax.set_title(f"Bus voltage band — {qsts_csv.stem}")
    ax.legend(fontsize=8); ax.set_xlim(0, 24)
    return _save(fig, f"voltage_{qsts_csv.stem}.png")


def balancing_bar(qsts_csvs: list[Path]):
    names, vals = [], []
    for c in qsts_csvs:
        df = pd.read_csv(c)
        nonpv = (df.syn_mw + df.slack_mw).to_numpy()
        vals.append(float(np.sum(np.abs(np.diff(nonpv)))) * 5 / 60)
        names.append(c.stem.replace("qsts_", ""))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color="#34495e")
    ax.set_ylabel("Fleet balancing duty (MWh/day)")
    ax.set_title("Balancing energy by scenario (lower = firmer)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    return _save(fig, "balancing_comparison.png")


def poi_dispatch(traj_csv: Path, day_intervals: int = 288):
    df = pd.read_csv(traj_csv).head(day_intervals)
    t = H(range(len(df)))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(t, df.P_actual, color="orange", lw=1, label=r"$P_{actual}$ (available PV)")
    a1.plot(t, df.P_forecast, color="cyan", lw=1, label=r"$P_{forecast}$")
    if "P_dfc" in df:
        a1.plot(t, df.P_dfc, color="grey", lw=0.8, ls=":", label=r"$P_{dfc}$ (dispatch target)")
    a1.plot(t, df.P_net, color="blue", lw=1.4, label=r"$P_{net}$ (delivered)")
    a1.set_ylabel("Power (pu)"); a1.legend(fontsize=8); a1.set_ylim(0, 1)
    a1.set_title(f"POI dispatch — {traj_csv.stem}")
    a2.fill_between(t, df.P_curtail, color="grey", alpha=.6, label="curtailed")
    a2.plot(t, df.P_bess_cmd, color="green", lw=1, label="BESS (+dis/-chg)")
    a2.axhline(0, color="k", lw=.5); a2.set_ylabel("pu"); a2.set_xlabel("Hour of day")
    a2.legend(fontsize=8, ncol=2); a2.set_xlim(0, 24)
    return _save(fig, f"poi_{traj_csv.stem}.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate standard QSTS/DFC figures.")
    ap.add_argument("--qsts", type=Path, default=None, help="a single QSTS CSV")
    ap.add_argument("--traj", type=Path, default=None, help="a single trajectory CSV")
    args = ap.parse_args(argv)

    made = []
    qsts_files = [args.qsts] if args.qsts else sorted(C.RESULTS.glob("qsts_*.csv"))
    for c in qsts_files:
        made += [dispatch_stack(c), penetration(c), voltage_band(c)]
    if len(qsts_files) > 1:
        made.append(balancing_bar(qsts_files))

    traj_files = [args.traj] if args.traj else sorted((C.RESULTS / "trajectories").glob("*.csv"))
    for c in traj_files:
        made.append(poi_dispatch(c))

    if not made:
        print("No results CSVs found. Run study_a_qsts.py first.")
        return 1
    print(f"Generated {len(made)} figure(s) in {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
