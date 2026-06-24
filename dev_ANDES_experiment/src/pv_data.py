#!/usr/bin/env python3
"""
pv_data.py — PV time-series loader for the QSTS study (ANDES-free, unit-testable).

Reads the three Australian plant profiles (RUGBYR1, BANN1, EDENVSF1, period 1),
normalises the two datetime formats, finds the days for which ALL three plants have a
complete 288-interval record, classifies those days (clear / mixed / overcast), and
produces per-interval MW injection matrices (288 x 9 buses) scaled by nameplate.

CLI:
    python src/pv_data.py                 # summary: common days + class counts
    python src/pv_data.py --list clear    # list common clear days
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def load_plant(plant: str, period: str | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by datetime with columns ['forecast','scada'] (pu)."""
    path = C.PV_DATA_DIR / C.plant_files(period)[plant]
    df = pd.read_csv(path)
    # the two files differ: RUGBYR1 uses '/', the others '-'; pandas handles both
    df["DATETIME"] = pd.to_datetime(df["DATETIME"].str.strip().str.replace("/", "-"),
                                    format="mixed")
    df = df.rename(columns={"FORECAST_POE50": "forecast", "SCADAVALUE": "scada"})
    return df.set_index("DATETIME")[["forecast", "scada"]].sort_index()


def common_complete_days(plants: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Calendar dates for which every plant has all INTERVALS_PER_DAY records."""
    day_sets = []
    for df in plants.values():
        counts = df.groupby(df.index.normalize()).size()
        full = set(counts[counts >= C.INTERVALS_PER_DAY].index)
        day_sets.append(full)
    return sorted(set.intersection(*day_sets))


def classify_day(agg_scada: np.ndarray) -> str:
    """Classify a day from the fleet-aggregate per-unit profile.

    cf    = daily capacity factor (24 h mean)            -> separates overcast
    fluct = mean |5-min change| during daylight          -> separates clear vs mixed
    Thresholds calibrated on the 204 common period-1 days (see calibration in commit
    history): clear=9, mixed=162, overcast=33. Clear days are rare because all three
    geographically-spread sites must be cloudless on the same day.
    """
    peak = agg_scada.max()
    if peak < 1e-3:
        return "overcast"
    cf = float(agg_scada.mean())
    daylight = agg_scada > 0.05 * peak
    fluct = np.mean(np.abs(np.diff(agg_scada[daylight]))) if daylight.sum() > 2 else 0.0
    if cf < 0.17:
        return "overcast"
    if fluct < 0.035 and cf > 0.26:
        return "clear"
    return "mixed"


def day_class_table(plants: dict[str, pd.DataFrame],
                    days: list[pd.Timestamp]) -> pd.DataFrame:
    """Per-day class label from the mean of the three plants' SCADA profiles."""
    rows = []
    for d in days:
        mats = []
        for df in plants.values():
            day = df.loc[str(d.date())]["scada"].to_numpy()[: C.INTERVALS_PER_DAY]
            mats.append(day)
        agg = np.mean(mats, axis=0)
        rows.append({"date": d.date(), "klass": classify_day(agg),
                     "cf": round(float(agg.mean()), 3)})
    return pd.DataFrame(rows)


def day_injection_matrix(plants: dict[str, pd.DataFrame], date,
                         column: str = "scada") -> np.ndarray:
    """288 x 9 MW injection matrix for `date`, mapping the 3 profiles round-robin to the
    9 buses (matching add_pv_bess) and scaling pu -> MW by per-site nameplate."""
    profile_names = list(C.PLANT_FILES_P1.keys())
    per_profile = {}
    for name, df in plants.items():
        series = df.loc[str(pd.Timestamp(date).date())][column].to_numpy()[: C.INTERVALS_PER_DAY]
        per_profile[name] = series
    cols = []
    for i, _bus in enumerate(C.PV_BUSES):
        prof = profile_names[i % len(profile_names)]
        cols.append(per_profile[prof] * C.NAMEPLATE_PER_SITE_MW)
    return np.column_stack(cols)   # shape (288, 9), MW


def load_all(period: str | None = None) -> dict[str, pd.DataFrame]:
    return {p: load_plant(p, period) for p in C.PROFILES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", choices=["clear", "mixed", "overcast"], default=None)
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    args = ap.parse_args(argv)

    plants = load_all(args.period)
    print(f"period: {args.period}")
    for name, df in plants.items():
        print(f"  {name:9s}: {len(df):6d} rows, {df.index.min()} -> {df.index.max()}")
    days = common_complete_days(plants)
    print(f"\nCommon complete days across all 3 plants: {len(days)}")
    tbl = day_class_table(plants, days)
    print("Day-class counts:", tbl["klass"].value_counts().to_dict())

    if args.list:
        sel = tbl[tbl["klass"] == args.list]
        print(f"\n{args.list} days ({len(sel)}):")
        for _, r in sel.iterrows():
            print(f"   {r['date']}  cf={r['cf']}")
    else:
        # show one representative of each class
        print("\nRepresentative day per class:")
        for k in ["clear", "mixed", "overcast"]:
            sel = tbl[tbl["klass"] == k]
            if len(sel):
                r = sel.iloc[len(sel) // 2]
                print(f"   {k:9s}: {r['date']}  cf={r['cf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
