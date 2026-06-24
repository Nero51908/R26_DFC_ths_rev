#!/usr/bin/env python3
"""
availability.py — DFC-as-availability analysis (ANDES-free).

Reframes DFC as a *firm availability product*: the plant commits a maximum dispatchable
availability (P_dfc) and delivers P_net, shaving below the commitment when the grid needs
less. This module scores each dispatch method (A2 forecast-as-DFC, A3 DRL agent, A4 MPC,
and A1 raw PV for reference) on the two axes of the hypothesis:

  QUANTITY  — how much firm capacity it can offer, reliability-weighted:
      capacity_credit(R) = the fleet power delivered for at least R% of generating
      intervals (the R-reliable firm MW; the BESS lifts the lower tail above raw PV).
  RELIABILITY — how dependably it honours the committed availability:
      dependability = P(P_net >= P_dfc - tol)  during generating intervals
      shortfall energy = unmet commitment / available energy.

All at the FLEET level (sum over the nine sites in MW) so geographic decorrelation is
captured — that is the grid-relevant firm capacity. Run:

    python src/availability.py                         # default days, R = 95% and 99%
    python src/availability.py --days 2025-01-29 2025-02-09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

METHODS = [("a1", "raw", "A1 raw PV"),
           ("a2", "rule", "A2 forecast-DFC"),
           ("a4", "mpc", "A4 MPC"),
           ("a3", "agent", "A3 DRL agent")]
GEN_THRESH = 0.02            # fraction of fleet nameplate above which the fleet is "generating"
MET_TOL = 0.005             # commitment met if (commit - net)/fleet_nameplate <= this


_CACHE: dict[str, dict] = {}
N = C.INTERVALS_PER_DAY


def _method_by_day(prefix: str) -> dict[str, dict]:
    """{date_str: {net,commit,avail}} fleet-MW 288-arrays. Each source file is read ONCE
    and grouped by date; only dates with a full 288-interval record on every plant kept.

    'raw' (A1): net=avail=scada, commit=forecast (no BESS). Others: canonical trajectories
    (net=P_net, commit=P_dfc, avail=P_actual)."""
    if prefix in _CACHE:
        return _CACHE[prefix]
    nameplate = C.NAMEPLATE_PER_SITE_MW

    if prefix == "raw":
        import pv_data as P
        plants = P.load_all()
        cols = {"net": "scada", "commit": "forecast", "avail": "scada"}
        framed = {n: plants[n].assign(_d=plants[n].index.date) for n in C.PROFILES}
    else:
        tdir = C.RESULTS / "trajectories"
        cols = {"net": "P_net", "commit": "P_dfc", "avail": "P_actual"}
        framed = {}
        for n in C.PROFILES:
            df = pd.read_csv(tdir / f"{prefix}_{n}.csv")
            df["_d"] = pd.to_datetime(df.datetime, errors="coerce").dt.date
            framed[n] = df

    # group each plant by date once
    grp = {n: {str(d): g for d, g in framed[n].groupby("_d")} for n in C.PROFILES}
    common = set.intersection(*[set(grp[n]) for n in C.PROFILES])
    out = {}
    for day in common:
        fleet = {}
        ok = True
        for key, col in cols.items():
            mats = []
            for n in C.PROFILES:
                a = grp[n][day][col].to_numpy()[:N]
                if len(a) != N:
                    ok = False
                    break
                mats.append(a)
            if not ok:
                break
            fleet[key] = np.sum(mats, axis=0) * nameplate
        if ok:
            out[day] = fleet
    _CACHE[prefix] = out
    return out


def _fleet(prefix, days, key):
    by_day = _method_by_day(prefix)
    miss = [d for d in days if d not in by_day]
    if miss:
        raise FileNotFoundError(f"{prefix}: no complete data for {miss[:3]}{'...' if len(miss)>3 else ''}")
    return np.concatenate([by_day[d][key] for d in days])


def metrics(prefix: str, days, reliabilities) -> dict:
    net = _fleet(prefix, days, "net")
    commit = _fleet(prefix, days, "commit")
    avail = _fleet(prefix, days, "avail")
    fleet_cap = C.NAMEPLATE_PER_SITE_MW * len(C.PROFILES)

    gen = avail > GEN_THRESH * fleet_cap                 # generating intervals only
    net_g, commit_g, avail_g = net[gen], commit[gen], avail[gen]
    deficit = np.maximum(commit_g - net_g, 0.0)

    out = {
        "committed_MW": float(commit_g.mean()),
        "delivered_MW": float(net_g.mean()),
        "dependability_%": float(100 * np.mean(deficit <= MET_TOL * fleet_cap)),
        "shortfall_energy_%": float(100 * deficit.sum() / max(avail_g.sum(), 1e-9)),
        "curtail_energy_%": float(100 * max(avail_g.sum() - net_g.sum(), 0) / max(avail_g.sum(), 1e-9)),
    }
    for R in reliabilities:
        out[f"firmMW@{int(R*100)}"] = float(np.quantile(net_g, 1 - R))   # R-reliable delivered MW
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DFC-as-availability analysis (fleet level).")
    ap.add_argument("--days", nargs="+", default=["2025-01-29", "2025-02-09"])
    ap.add_argument("--all-days", action="store_true",
                    help="use every complete period-2 day (methods lacking coverage are skipped)")
    ap.add_argument("--R", nargs="+", type=float, default=[0.95, 0.99])
    args = ap.parse_args(argv)

    if args.all_days:
        import pv_data as P
        args.days = [str(d.date()) for d in P.common_complete_days(P.load_all())]
        print(f"  using {len(args.days)} complete period-2 days")

    rows = []
    for scen, pref, lab in METHODS:
        try:
            m = metrics(pref, args.days, args.R)
        except FileNotFoundError:
            print(f"  (skip {lab}: trajectory files for '{pref}' not found)")
            continue
        m["method"] = lab
        rows.append(m)

    firm_cols = [f"firmMW@{int(R*100)}" for R in args.R]
    cols = ["method", "committed_MW", "delivered_MW", *firm_cols,
            "dependability_%", "shortfall_energy_%", "curtail_energy_%"]
    df = pd.DataFrame(rows)[cols]
    out = C.RESULTS / "availability.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 140, "display.max_columns", 20)
    span = (f"{len(args.days)} days {args.days[0]}..{args.days[-1]}"
            if len(args.days) > 4 else str(args.days))
    print(f"\nDFC availability — fleet level, {span}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    fleet_cap = C.NAMEPLATE_PER_SITE_MW * len(C.PROFILES)
    print(f"\nColumn meanings  (fleet = sum of the {len(C.PROFILES)} sites, {fleet_cap:,.0f} MW nameplate;")
    print(f"                  'generating intervals' = the {GEN_THRESH:.0%}-of-nameplate daylight hours only,")
    print(f"                  so night zeros never inflate the averages)")
    print( "  method            the dispatch rule producing the firm signal")
    print( "                      A1 raw PV (no battery) · A2 forecast-as-DFC · A3 DRL agent · A4 MPC")
    print( "  committed_MW      avg firm capacity OFFERED to the grid  (the P_dfc commitment)")
    print( "  delivered_MW      avg power ACTUALLY delivered           (P_net, after the battery)")
    def _ord(n: int) -> str:
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    for R in args.R:
        pct = int(R * 100)
        print(f"  firmMW@{pct:<2d}        QUANTITY: firm MW the fleet beats in {pct}% of generating intervals —")
        print(f"                      i.e. capacity you could SELL at {pct}% reliability (the {_ord(100-pct)}-percentile")
        print(f"                      delivery; capacity credit). Higher = more bankable firm capacity.")
    print( "  dependability_%   RELIABILITY: % of generating intervals where delivery met the commitment")
    print( "                      (P_net >= P_dfc - tol). Higher = the offered MW is more trustworthy.")
    print( "  shortfall_energy_%  energy the fleet promised but failed to deliver, as % of available PV energy.")
    print( "                      Lower = fewer/smaller broken commitments.")
    print( "  curtail_energy_%  available PV energy deliberately shaved (stored or spilled), as % of available.")
    print( "                      Not a fault — it is the headroom spent to firm the commitment.")
    print(f"\n  Hypothesis read: a stronger method offers a HIGHER firmMW@{int(max(args.R)*100)} (quantity) at")
    print( "  HIGHER dependability (reliability) than the forecast/MPC baselines.")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
