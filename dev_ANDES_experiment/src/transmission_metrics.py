#!/usr/bin/env python3
"""
transmission_metrics.py — transmission-network value of DFC, from QSTS network logs.

Reads results/qsts_net_<scenario>_<date>.npz (written by study_a_qsts: per-interval line
%loading and bus voltages) and computes, per scenario, how firm/reliable DFC dispatch
shapes the transmission network:

  - peak line loading and CONGESTION FREQUENCY (% intervals with any line > threshold),
  - number of distinct lines ever stressed,
  - network FLOW VARIABILITY (total |Δ MVA| summed over lines & intervals) — how much the
    dispatch makes the transmission flows move (firm DFC should move them less),
  - voltage band: min/max bus V, # bus-intervals outside [0.95, 1.05], max |ΔV|,
  - losses are reported by study_a_qsts (qsts_<scenario>_<date>.csv).

Hypothesis link: a dependable DFC (the agent) lets the operator schedule against a firm
availability, yielding flatter, less-congested, more predictable transmission conditions.

    python src/transmission_metrics.py                 # every qsts_net_*.npz it finds
    python src/transmission_metrics.py --load-thresh 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

ACTOR = {"a1": "raw_pv", "a2": "forecast-DFC", "a3": "DRL_agent", "a4": "MPC"}


def scen_metrics(npz_path: Path, load_thresh: float, vlo: float, vhi: float) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    s = C.LINE_RATE_SCALE                         # reinforce legacy IEEE39 ratings (config; DFC_RATE_SCALE)
    pct = d["loadpct"].astype(float) / s         # (T, L) % of REINFORCED rating (nan where unrated)
    vmag = d["vmag"].astype(float)               # (T, B)
    rate = d["rate"].astype(float) * s
    mva = pct / 100.0 * rate                     # physical MVA (scale cancels) for flow variability
    over = np.nan_to_num(pct, nan=0.0) > load_thresh
    return {
        "peak_load_%": round(float(np.nanmax(pct)), 1),
        "congestion_freq_%": round(float(100 * np.mean(over.any(axis=1))), 2),
        "lines_stressed": int(over.any(axis=0).sum()),
        "flow_var_GVA": round(float(np.nansum(np.abs(np.diff(mva, axis=0)))) / 1000, 2),
        "vmin": round(float(vmag.min()), 4),
        "vmax": round(float(vmag.max()), 4),
        "v_excursions": int(((vmag < vlo) | (vmag > vhi)).sum()),
        "max_dV": round(float(np.max(np.abs(vmag - 1.0))), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transmission-network metrics from QSTS logs.")
    ap.add_argument("--load-thresh", type=float, default=90.0, help="line loading %% for congestion")
    ap.add_argument("--vlo", type=float, default=0.95)
    ap.add_argument("--vhi", type=float, default=1.05)
    args = ap.parse_args(argv)

    files = sorted(C.qsts_dir().rglob("qsts_net_*.npz"))
    if not files:
        print(f"No qsts_net_*.npz under {C.qsts_dir()}/. Run study_a_qsts.py (records network observables).")
        return 1
    rows = []
    for f in files:
        parts = f.stem.split("_")                # qsts_net_<scenario>_<date>
        scen, date = parts[2], parts[3]
        m = scen_metrics(f, args.load_thresh, args.vlo, args.vhi)
        rows.append({"scenario": ACTOR.get(scen, scen), "day": date, **m})

    df = pd.DataFrame(rows).sort_values(["day", "scenario"])
    out = C.RESULTS / "transmission_metrics.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"\nTransmission-network metrics (congestion threshold {args.load_thresh:g}% loading)")
    print(df.to_string(index=False))
    print(f"\n  congestion_freq = % intervals with any line over threshold")
    print(f"  flow_var_GVA = total |delta MVA| over lines & intervals (lower = firmer flows)")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
