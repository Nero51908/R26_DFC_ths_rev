#!/usr/bin/env python3
"""
tracker.py — experiment journal for the grid-value (Study A QSTS) chapter.

Mirrors the structure of the DRL chapters' evaluation journal
(dev_DRL_experiment/evaluation_journal_*.csv) so the logging convention is consistent
across the thesis, then appends the grid-level metrics this chapter produces.

Backbone columns (same as the DRL evaluation journal):
    dataset, actor, fixed_c, BESS_capacity, BESS_charg_lim, BESS_disch_lim,
    filename, mean_curtailment_ratio, mean_curtailment, perfect_rate, rmse, mae, spotlight
Appended grid columns:
    fleet_balancing_mwh, interconnector_swing_mw, peak_pen, mean_pen, mean_loss_mw,
    vmin, vmax, n_converged, n_intervals, day, day_class

Notes on the mapping (one row per QSTS run):
  - dataset  : the three-plant fleet (RUGBYR1+BANN1+EDENVSF1)
  - actor    : the dispatch method  (a1=raw PV, a3=DFC agent, a4=MPC)
  - BESS_*   : the storage config behind the firm trajectory (blank for raw-PV A1)
  - curtailment/perfect_rate/rmse/mae : trajectory-quality fields carried from the firm
    source. For the rule-based STAND-IN, delivered == committed firm power, so
    perfect_rate=100 / rmse=mae=0; these become the trained agent's real numbers once
    the agent trajectories replace the stand-in.

CLI:
    python src/tracker.py            # (re)build the journal from results/qsts_*.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIELDS = [
    "dataset", "actor", "fixed_c", "BESS_capacity", "BESS_charg_lim", "BESS_disch_lim",
    "filename", "mean_curtailment_ratio", "mean_curtailment", "deliv_frac", "ramp_std_net",
    "bess_throughput_puh", "perfect_rate", "rmse", "mae", "spotlight",
    "fleet_balancing_mwh", "interconnector_swing_mw", "peak_pen", "mean_pen",
    "mean_loss_mw", "vmin", "vmax", "n_converged", "n_intervals", "day", "day_class",
]
JOURNAL = C.ROOT / "experiment_tracker.csv"
ACTOR = {"a1": "raw_pv", "a2": "rule_firming", "a3": "dfc_agent", "a4": "mpc"}
PREFIX = {"a2": "rule", "a3": "agent", "a4": "mpc"}          # firm-trajectory file prefixes
PERFECT_BAND = 0.092            # pu (~6 MW on the 65 MW RUGBYR1 scale; the Ch.5 tolerance)
SIGNAL_KEYS = ["mean_curtailment_ratio", "mean_curtailment", "deliv_frac", "ramp_std_net",
               "bess_throughput_puh", "perfect_rate", "rmse", "mae"]


def grid_metrics(qsts_csv: Path) -> dict:
    rows = list(csv.DictReader(open(qsts_csv)))
    syn = np.array([float(r["syn_mw"]) for r in rows])
    slack = np.array([float(r["slack_mw"]) for r in rows])
    loss = np.array([float(r["loss_mw"]) for r in rows])
    pen = np.array([float(r["pen"]) for r in rows])
    nonpv = syn + slack
    return {
        "fleet_balancing_mwh": round(float(np.abs(np.diff(nonpv)).sum()) * 5 / 60, 1),
        "interconnector_swing_mw": round(float(slack.max() - slack.min()), 1),
        "peak_pen": round(float(pen.max()), 3),
        "mean_pen": round(float(pen.mean()), 3),
        "mean_loss_mw": round(float(loss.mean()), 1),
        "vmin": round(min(float(r["vmin"]) for r in rows), 4),
        "vmax": round(max(float(r["vmax"]) for r in rows), 4),
        "n_converged": int(sum(str(r["converged"]) == "True" for r in rows)),
        "n_intervals": len(rows),
    }


def traj_metrics(prefix: str, date) -> dict:
    """DFC-signal quality metrics for `date`, aggregated over the three plant trajectories.

    Beyond curtailment, this captures the firmness-vs-yield trade-off that distinguishes
    the methods: how firm the delivered power is (ramp_std_net), how much of the available
    PV is actually delivered (deliv_frac) vs spilled (curtailment), battery cycling
    (throughput), and how accurately delivery meets the commitment (rmse/mae/perfect_rate
    of P_net vs P_dfc, using the Ch.5 tolerance band).
    """
    import pandas as pd
    tdir = C.RESULTS / "trajectories"
    NET, DFC, ACT, CUR, BESS = [], [], [], [], []
    for name in C.PROFILES:
        path = tdir / f"{prefix}_{name}.csv"
        if not path.exists():
            return {k: "" for k in SIGNAL_KEYS}
        df = pd.read_csv(path)
        d = pd.to_datetime(df.get("datetime"), format="mixed", errors="coerce").dt.date
        day = df[d == pd.Timestamp(date).date()]
        if not len(day):                       # single-day file (no/empty datetime)
            day = df
        ACT.append(day["P_actual"].to_numpy()); NET.append(day["P_net"].to_numpy())
        DFC.append(day["P_dfc"].to_numpy()); CUR.append(day["P_curtail"].to_numpy())
        BESS.append(day["P_bess_cmd"].to_numpy())

    act, net = np.concatenate(ACT), np.concatenate(NET)
    dfc, cur, bess = np.concatenate(DFC), np.concatenate(CUR), np.concatenate(BESS)
    avail = max(float(act.sum()), 1e-9)
    dl = act > 1e-3
    err = net[dl] - dfc[dl]
    # ramp std per-plant then averaged (concatenating plants would add fake boundary jumps)
    ramps = [np.std(np.diff(n[a > 1e-3])) for n, a in zip(NET, ACT) if (a > 1e-3).sum() > 2]
    return {
        "mean_curtailment_ratio": round(float(cur.sum() / avail), 5),
        "mean_curtailment": round(float(cur.mean()), 5),
        "deliv_frac": round(float(net.sum() / avail), 4),
        "ramp_std_net": round(float(np.mean(ramps)), 5) if ramps else "",
        "bess_throughput_puh": round(float(np.abs(bess).sum() * (5 / 60) / len(C.PROFILES)), 4),
        "perfect_rate": round(float(100 * np.mean(np.abs(err) < PERFECT_BAND)), 2) if dl.any() else "",
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 5) if dl.any() else "",
        "mae": round(float(np.mean(np.abs(err))), 5) if dl.any() else "",
    }


def day_class_of(date) -> str:
    import pv_data as P
    import pandas as pd
    try:                                   # may be absent in the active period's data
        tbl = P.day_class_table(P.load_all(), [pd.Timestamp(date)])
        return tbl.iloc[0]["klass"] if len(tbl) else ""
    except (KeyError, IndexError):
        return ""


def build_row(scenario: str, date_str: str, qsts_csv: Path) -> dict:
    g = grid_metrics(qsts_csv)
    is_firm = scenario in PREFIX
    if is_firm:
        sig = traj_metrics(PREFIX[scenario], date_str)
    else:                                                # a1 raw PV: delivers all PV, no BESS
        sig = {"mean_curtailment_ratio": 0.0, "mean_curtailment": 0.0, "deliv_frac": 1.0,
               "ramp_std_net": "", "bess_throughput_puh": 0.0,
               "perfect_rate": "", "rmse": "", "mae": ""}
    row = {
        "dataset": "RUGBYR1+BANN1+EDENVSF1",
        "actor": ACTOR.get(scenario, scenario),
        "fixed_c": "",                                   # learned/rule-based -> blank
        "BESS_capacity": C.BESS_ENERGY_PU if is_firm else "",
        "BESS_charg_lim": C.BESS_ERATE if is_firm else "",
        "BESS_disch_lim": -C.BESS_ERATE if is_firm else "",
        "filename": qsts_csv.name,
        **sig,
        "spotlight": False,
        **g,
        "day": date_str,
        "day_class": day_class_of(date_str),
    }
    return row


def rebuild() -> int:
    runs = sorted(C.RESULTS.glob("qsts_*.csv"))
    rows = []
    for f in runs:
        parts = f.stem.split("_")                        # qsts_<scenario>_<date>
        if len(parts) != 3 or parts[1] == "selftest":
            continue
        scenario, date_str = parts[1], parts[2]
        rows.append(build_row(scenario, date_str, f))
    with open(JOURNAL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows -> {JOURNAL}")
    for r in rows:
        print(f"  {r['actor']:18s} {r['day']} ({r['day_class']:8s}): "
              f"balancing {r['fleet_balancing_mwh']:.0f} MWh, "
              f"interconnector swing {r['interconnector_swing_mw']:.0f} MW, "
              f"peak pen {r['peak_pen']*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(rebuild())
