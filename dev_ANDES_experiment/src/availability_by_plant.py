#!/usr/bin/env python3
"""
availability_by_plant.py — DFC-as-availability, broken out PER PLANT (ANDES-free).

availability.py scores the FLEET (sum of the nine sites). This module instead scores each
plant profile on its own, so a weak site is not hidden by strong ones — and adds the
statistical machinery to answer "does method X beat method Y for THIS plant?":

  * per-plant, per-method table on the common days (generating intervals only):
      committed / delivered      mean firm capacity offered (P_dfc) and delivered (P_net)
      firm@95 / firm@99          QUANTITY: delivered MW beaten in 95% / 99% of intervals
                                 (capacity credit; commitment-agnostic, the cleanest measure)
      dependability_%            RELIABILITY: % of intervals where P_net >= P_dfc - tol
      shortfall_energy_%         unmet-commitment energy / available PV energy (depth-weighted)
      curtail_energy_%           deliberately shaved PV energy / available
      breach_meanDepth/p95       when the commitment IS missed, how deep (tail risk)
  * head-to-head significance (default A3 agent vs A4 MPC):
      - day-bootstrap 95% CI on the firm@95 / firm@99 difference
      - paired daily t on dependability and shortfall

All values are % OF PLANT NAMEPLATE (per-unit x 100), so plants of different size compare
directly. Metric definitions (GEN_THRESH, MET_TOL, firm = quantile of delivered net) are
identical to availability.py.

    python src/availability_by_plant.py                       # all plants, agent-vs-MPC tests
    python src/availability_by_plant.py --plant BANN1
    python src/availability_by_plant.py --compare agent rule --bootstrap 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# (scenario, file prefix, label) — mirrors availability.py
METHODS = {"a1": ("raw", "A1 raw PV"), "a2": ("rule", "A2 forecast-DFC"),
           "a3": ("agent", "A3 DRL agent"), "a4": ("mpc", "A4 MPC")}
PREFIX = {v[0]: k for k, v in METHODS.items()}           # 'agent' -> 'a3', etc.
GEN_THRESH = 0.02            # actual > 2% of capacity => "generating" interval (per-unit)
MET_TOL = 0.005             # commitment met if (commit - net) <= this (per-unit)
N = C.INTERVALS_PER_DAY


def _load(plant: str, prefix: str, period: str | None) -> dict[str, pd.DataFrame]:
    """{date_str: day_df[net,commit,avail]} for one plant+method, full 288-interval days only."""
    if prefix == "raw":
        import pv_data as P
        df = P.load_plant(plant, period).reset_index().rename(columns={"DATETIME": "datetime"})
        df = df.assign(net=df["scada"], commit=df["forecast"], avail=df["scada"])
    else:
        df = pd.read_csv(C.traj_dir() / f"{prefix}_{plant}.csv")
        df = df.rename(columns={"P_net": "net", "P_dfc": "commit", "P_actual": "avail"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["_d"] = df["datetime"].dt.date
    return {str(d): g.iloc[:N] for d, g in df.groupby("_d") if len(g) >= N}


def _gen(day_df: pd.DataFrame):
    a = day_df["avail"].to_numpy()
    m = a > GEN_THRESH
    return day_df["net"].to_numpy()[m], day_df["commit"].to_numpy()[m], a[m]


def _metrics(net, commit, avail) -> dict:
    deficit = np.maximum(commit - net, 0.0)
    breach = deficit[deficit > MET_TOL]
    return {
        "committed": round(100 * commit.mean(), 2),
        "delivered": round(100 * net.mean(), 2),
        "firm@95": round(100 * np.quantile(net, 0.05), 2),
        "firm@99": round(100 * np.quantile(net, 0.01), 2),
        "depend_%": round(100 * np.mean(deficit <= MET_TOL), 2),
        "shortfall_%": round(100 * deficit.sum() / max(avail.sum(), 1e-9), 2),
        "curtail_%": round(100 * max(avail.sum() - net.sum(), 0) / max(avail.sum(), 1e-9), 2),
        "breach_depth": round(100 * breach.mean(), 2) if len(breach) else 0.0,
        "breach_p95": round(100 * np.percentile(breach, 95), 2) if len(breach) else 0.0,
    }


def plant_report(plant: str, methods: list[str], period: str | None):
    """Return (table_df, common_days, day_arrays) for one plant over the days common to all
    `methods` (each item is a file prefix: raw/rule/mpc/agent)."""
    loaded = {pref: _load(plant, pref, period) for pref in methods}
    common = sorted(set.intersection(*[set(loaded[p]) for p in methods])) if methods else []
    rows, day_arr = [], {p: {} for p in methods}
    for pref in methods:
        nets, coms, avs = [], [], []
        for d in common:
            n, c, a = _gen(loaded[pref][d])
            nets.append(n); coms.append(c); avs.append(a)
            day_arr[pref][d] = (n, c, a)
        net, commit, avail = np.concatenate(nets), np.concatenate(coms), np.concatenate(avs)
        rows.append({"plant": plant, "method": METHODS[PREFIX[pref]][1], **_metrics(net, commit, avail)})
    return pd.DataFrame(rows), common, day_arr


def head_to_head(common, day_arr, a_pref, b_pref, n_boot, rng) -> list[str]:
    """Significance of a_pref vs b_pref: firm bootstrap CI + paired daily dependability/shortfall."""
    lines = []
    net_a = {d: day_arr[a_pref][d][0] for d in common}
    net_b = {d: day_arr[b_pref][d][0] for d in common}

    def firm(net_map, days, q):
        return 100 * np.quantile(np.concatenate([net_map[d] for d in days]), q)

    if n_boot > 0:
        for q, lab in [(0.05, "firm@95"), (0.01, "firm@99")]:
            diffs = []
            for _ in range(n_boot):
                s = rng.choice(common, len(common), replace=True)
                diffs.append(firm(net_a, s, q) - firm(net_b, s, q))
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            lines.append(f"    {lab}: {a_pref}-{b_pref} = {diffs.mean():+.2f} %cap  "
                         f"95%CI [{lo:+.2f}, {hi:+.2f}]  P({a_pref}>{b_pref})={np.mean(diffs > 0):.3f}")

    # paired daily dependability & shortfall
    da, db, sa, sb = [], [], [], []
    for d in common:
        na, ca, aa = day_arr[a_pref][d]
        nb, cb, ab = day_arr[b_pref][d]
        da.append(100 * np.mean(np.maximum(ca - na, 0) <= MET_TOL))
        db.append(100 * np.mean(np.maximum(cb - nb, 0) <= MET_TOL))
        sa.append(100 * np.maximum(ca - na, 0).sum() / max(aa.sum(), 1e-9))
        sb.append(100 * np.maximum(cb - nb, 0).sum() / max(ab.sum(), 1e-9))
    for name, x, y, hi_good in [("dependability", np.array(da), np.array(db), True),
                                ("shortfall", np.array(sa), np.array(sb), False)]:
        diff = x - y
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        win = np.mean(x > y) if hi_good else np.mean(x < y)
        lines.append(f"    {name}: {a_pref} {x.mean():.2f}  {b_pref} {y.mean():.2f}  "
                     f"diff {diff.mean():+.2f}  t={t:+.2f}  {a_pref}-better days={win:.2f}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-plant DFC availability + head-to-head tests.")
    ap.add_argument("--plant", choices=[*C.PROFILES, "all"], default="all")
    ap.add_argument("--methods", nargs="+", default=["raw", "rule", "mpc", "agent"],
                    choices=["raw", "rule", "mpc", "agent"], help="file prefixes to score")
    ap.add_argument("--compare", nargs=2, default=["agent", "mpc"],
                    metavar=("A", "B"), help="head-to-head pair (file prefixes)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="day-bootstrap reps (0 = skip CI)")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    plants = C.PROFILES if args.plant == "all" else [args.plant]
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 170, "display.max_columns", 25)
    print(f"\nPer-plant DFC availability — period {args.period}, generating intervals, % of nameplate")

    all_rows, h2h_blocks = [], []
    for plant in plants:
        df, common, day_arr = plant_report(plant, args.methods, args.period)
        all_rows.append(df)
        print(f"\n{plant}  ({len(common)} common days across {', '.join(args.methods)}):")
        print(df.to_string(index=False))
        a, b = args.compare
        if a in args.methods and b in args.methods and common:
            print(f"  head-to-head  {a} vs {b}:")
            block = head_to_head(common, day_arr, a, b, args.bootstrap, rng)
            for ln in block:
                print(ln)
            h2h_blocks.append((plant, block))

    out = pd.concat(all_rows, ignore_index=True)
    path = C.RESULTS / "availability_by_plant.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print("\n  firm@R = delivered MW beaten in R% of intervals (capacity credit; higher=better)")
    print("  depend_% higher=better; shortfall_%/breach_depth lower=better (tail risk)")
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
