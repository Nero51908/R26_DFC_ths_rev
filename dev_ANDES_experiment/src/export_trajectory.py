#!/usr/bin/env python3
"""
export_trajectory.py — convert a DRL-agent environment evaluation CSV into the canonical
trajectory schema (scenario A3), runnable in the ANDES environment (pure pandas).

The DRL eval records DAYTIME intervals only; this converter reindexes onto the full
5-min grid and zero-fills the night, so the output is complete 288-interval days like the
rule/MPC trajectories.

Canonical columns (per-unit of PV peak):
    idx, datetime, P_forecast, P_actual, P_net, P_dfc, P_bess_cmd, P_curtail
where
    P_actual = available PV (pv_potential)            P_forecast = PV forecast
    P_dfc    = dispatch target the engine commits to   (recording 'Pdfc')
    P_net    = ACTUAL plant delivery, PV + BESS net    (recording 'Pnet')  <-- injected by QSTS
    P_bess_cmd = battery power (+charge / -discharge)   P_curtail = curtailed PV

The QSTS injects P_net (what the plant actually delivers to the grid); P_dfc is retained
as the scheduled/forecast value for reference and future schedule-vs-delivery analysis.

    python src/export_trajectory.py --in agent_eval.csv --out results/trajectories/agent_RUGBYR1.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

ALIASES = {
    "Pf":   ["env_state_pv_forecast", "pv_forecast", "Pf"],
    "Pm":   ["env_state_pv_potential", "pv_potential", "Pm"],
    "Pdfc": ["Pdfc", "pdfc"],
    "Pnet": ["Pnet", "pnet"],
    "bess": ["env_state_bess_power", "bess_power"],
    "ac":   ["actual_cr", "actual_c", "ac"],
    "dt":   ["env_state_datetime", "datetime", "DATETIME"],
}
INTERVALS_PER_DAY = 288
FREQ = "5min"


def pick(df, key, required=True):
    for c in ALIASES[key]:
        if c in df.columns:
            return df[c]
    if required:
        raise KeyError(f"none of {ALIASES[key]} in columns: {list(df.columns)}")
    return None


def convert(in_csv: str, out_csv: str) -> str:
    df = pd.read_csv(in_csv)
    Pf, Pm = pick(df, "Pf"), pick(df, "Pm")
    Pdfc, Pnet = pick(df, "Pdfc"), pick(df, "Pnet")
    ac = pick(df, "ac")
    bess = pick(df, "bess", required=False)
    dt = pd.to_datetime(pick(df, "dt").astype(str).str.replace("/", "-"),
                        format="mixed", errors="coerce")

    canon = pd.DataFrame({
        "P_forecast": Pf.to_numpy(), "P_actual": Pm.to_numpy(),
        "P_net": Pnet.to_numpy(), "P_dfc": Pdfc.to_numpy(),
        "P_bess_cmd": (bess.to_numpy() if bess is not None
                       else (Pm * (1 - ac) - Pnet).to_numpy()),
        "P_curtail": (Pm * ac).to_numpy(),
    }, index=dt)

    # reindex onto the full continuous 5-min grid (daytime-only -> full days, night = 0)
    grid = pd.date_range(canon.index.min().normalize(),
                         canon.index.max().normalize() + pd.Timedelta(days=1)
                         - pd.Timedelta(FREQ), freq=FREQ)
    canon = canon[~canon.index.duplicated(keep="first")].reindex(grid, fill_value=0.0)

    out = canon.reset_index().rename(columns={"index": "datetime"})
    out.insert(0, "idx", np.arange(len(out)))
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out[["idx", "datetime", "P_forecast", "P_actual", "P_net", "P_dfc",
               "P_bess_cmd", "P_curtail"]]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"export_trajectory(): {len(out)} rows ({len(out)//INTERVALS_PER_DAY} full days) "
          f"-> {out_csv}  P_net {out.P_net.min():.3f}..{out.P_net.max():.3f}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    a = ap.parse_args()
    convert(a.in_csv, a.out_csv)
