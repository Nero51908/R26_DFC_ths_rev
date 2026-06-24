#!/usr/bin/env python3
"""
make_firm_trajectory.py — rule-based BESS firming baseline (scenario A2).

Scenario A2 in the design (experiment_design.md §3): the plant commits to the PV
**forecast** and uses the BESS to cover the forecast error — the forecast+BESS firming
benchmark that A3 (DRL agent) and A4 (MPC) are compared against. It is NOT the agent; it
is the simplest defensible firming rule (Ch. 4-5 benchmark), and a cvxpy-free way to
exercise the firm-injection path end-to-end.

Battery physics come from the shared dfc_plant.DFCPlant (the faithful Ch.5 DeepComp
port), so A2/A3/A4 all sit on identical storage dynamics. Output is the canonical schema
(idx, datetime, P_forecast, P_actual, P_DFC, P_bess_cmd, P_curtail) written to
results/trajectories/rule_<profile>.csv, which study_a_qsts consumes as scenario A2.

Run (ANDES-free):
    python src/make_firm_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import pv_data as P
from dfc_plant import DFCPlant

PREFIX = "rule"


def firm_profile(df: pd.DataFrame) -> pd.DataFrame:
    """A2 benchmark (design §3): the plant commits to the PV FORECAST and the BESS covers
    the forecast error. Target = P_forecast each step; the shared DeepComp battery charges
    when actual PV exceeds the forecast and discharges when it falls short. Because the
    forecast tracks actual PV closely, only the (small) forecast error is buffered, so
    curtailment is minimal — unlike an aggressive ramp-smoothing target."""
    fc = df["forecast"].to_numpy(float)
    ac = df["scada"].to_numpy(float)
    plant = DFCPlant()
    soc = 50.0
    P_NET, P_DFC, P_BESS, P_CURT = [], [], [], []
    for pf, pav in zip(fc, ac):
        target = float(min(max(pf, 0.0), 1.0))           # commit to the PV forecast
        soc, Ppv, Pb, Pnet, actual_c = plant.step(Pm=float(pav), Pdfc=target, c=0.0, soc_pct=soc)
        P_NET.append(Pnet); P_DFC.append(target)
        P_BESS.append(Pb); P_CURT.append(float(pav) * actual_c)
    return pd.DataFrame({
        "idx": np.arange(len(ac)),
        "datetime": df.index.strftime("%Y-%m-%d %H:%M:%S"),
        "P_forecast": fc, "P_actual": ac,
        "P_net": np.array(P_NET), "P_dfc": np.array(P_DFC),
        "P_bess_cmd": np.array(P_BESS), "P_curtail": np.array(P_CURT),
    })


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="A2 rule-based firming baseline.")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    args = ap.parse_args(argv)
    out_dir = C.RESULTS / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in P.load_all(args.period).items():
        traj = firm_profile(df)
        out = out_dir / f"{PREFIX}_{name}.csv"
        traj.to_csv(out, index=False)
        raw = np.std(np.diff(traj["P_actual"])); firm = np.std(np.diff(traj["P_net"]))
        print(f"  {name:9s}: {len(traj)} rows -> {out.name}  "
              f"ramp std {raw:.4f} -> {firm:.4f} ({100*(1-firm/raw):.0f}% smoother), "
              f"curtail {traj['P_curtail'].mean():.4f}")
    print(f"\nWrote A2 rule-firming trajectories ({args.period}) to {out_dir}")
    print("Run A2:  python src/study_a_qsts.py --scenario a2 --klass mixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
