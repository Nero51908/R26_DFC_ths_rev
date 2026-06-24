#!/usr/bin/env python3
"""
trajectory.py — adapter from DRL/MPC canonical trajectories to QSTS injection matrices.

A canonical trajectory CSV has per-unit columns:
    idx, datetime, P_forecast, P_actual, P_net, P_dfc, P_bess_cmd, P_curtail
where P_net is the ACTUAL plant delivery (PV + BESS net) and P_dfc the dispatch
forecast/target. The QSTS injects P_net — the power the plant actually puts on the grid.

This module scales the chosen column pu -> MW by per-site nameplate and maps the three
plant profiles round-robin onto the nine buses (identical mapping to add_pv_bess and
pv_data), producing the (INTERVALS_PER_DAY, 9) MW matrix that study_a_qsts injects for
scenario A2 (rule), A3 (DRL agent) or A4 (MPC). The BESS action is already embedded in
P_net, so the QSTS simply injects it — no SoC tracking needed grid-side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def load_trajectory(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "datetime" in df.columns and df["datetime"].astype(str).str.len().gt(0).any():
        df["datetime"] = pd.to_datetime(df["datetime"].astype(str).str.replace("/", "-"),
                                        format="mixed", errors="coerce")
    return df


def _day_slice(df: pd.DataFrame, date, column: str) -> np.ndarray:
    """Return the day's INTERVALS_PER_DAY values of `column` (pu), aligned to the full
    5-min grid. Handles full-day files (rule/mpc, 288 rows) AND daytime-only files (the
    DRL agent eval, which omits night intervals) by reindexing onto the day grid and
    zero-filling the missing (night) intervals."""
    n = C.INTERVALS_PER_DAY
    if date is not None and "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
        if dt.notna().any():
            mask = dt.dt.date == pd.Timestamp(date).date()
            day = pd.Series(df.loc[mask, column].to_numpy(),
                            index=dt[mask].dt.floor("5min"))
            day = day[~day.index.duplicated(keep="first")]
            grid = pd.date_range(pd.Timestamp(date).normalize(), periods=n, freq="5min")
            return day.reindex(grid, fill_value=0.0).to_numpy()
    # no datetimes: assume a single day already; take first n, zero-pad if short
    arr = df[column].to_numpy()[:n]
    return np.concatenate([arr, np.zeros(n - len(arr))]) if len(arr) < n else arr


def injection_from_trajectories(traj_by_profile: dict[str, pd.DataFrame],
                                date=None, column: str = "P_net") -> np.ndarray:
    """(INTERVALS_PER_DAY, 9) MW matrix, profiles mapped round-robin to PV_BUSES."""
    profiles = list(C.PLANT_FILES_P1.keys())     # RUGBYR1, BANN1, EDENVSF1 (same order)
    per_profile = {name: _day_slice(traj_by_profile[name], date, column)
                   for name in profiles}
    cols = []
    for i, _bus in enumerate(C.PV_BUSES):
        prof = profiles[i % len(profiles)]
        cols.append(per_profile[prof] * C.NAMEPLATE_PER_SITE_MW)
    return np.column_stack(cols)                  # (288, 9) MW


def load_profile_trajectories(traj_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    """Load <prefix>_<PROFILE>.csv for each of the three profiles."""
    out = {}
    for name in C.PLANT_FILES_P1:
        path = traj_dir / f"{prefix}_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"trajectory not found: {path}")
        out[name] = load_trajectory(path)
    return out
