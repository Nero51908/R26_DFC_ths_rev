#!/usr/bin/env python3
"""
study_a_qsts.py — QSTS prototype for Study A (steady-state, power-flow loop).

A QSTS is a sequence of independent power-flow solves over time. For one representative
day (288 x 5-min intervals) this script:
  1. loads the high-penetration case (config.PVBESS_CASE),
  2. at each interval sets the nine PV-BESS injections from the real plant profiles,
  3. re-dispatches the synchronous fleet to cover (load - PV), slack/interconnector
     absorbing the remainder,
  4. solves the power flow and records grid metrics.

Scenario A1 (this prototype): plants inject the RAW measured PV (no BESS, no DFC) — the
volatile baseline. The firm DFC scenario (A3) and the MPC benchmark (A4) reuse this same
loop with a different per-interval injection series and BESS SoC carried between steps.

Sized for a laptop: one day x one scenario = 288 power flows (seconds). Scale to more
days / scenarios on HPC later.

Run (in the ANDES venv):
    python src/study_a_qsts.py                 # auto-pick a clear day, scenario A1
    python src/study_a_qsts.py --klass mixed   # a volatile day instead
    python src/study_a_qsts.py --date 2023-10-13
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import pv_data as P
import trajectory as T


def pick_day(plants, days, klass: str | None, date_str: str | None):
    import pandas as pd
    if date_str:
        return pd.Timestamp(date_str)
    tbl = P.day_class_table(plants, days)
    sel = tbl if klass is None else tbl[tbl["klass"] == klass]
    if not len(sel):
        sys.exit(f"No '{klass}' day found among the common days.")
    row = sel.iloc[len(sel) // 2]               # a representative middle one
    return __import__("pandas").Timestamp(row["date"])


def load_case():
    import andes
    if not C.PVBESS_CASE.exists():
        sys.exit(f"PV-BESS case not found: {C.PVBESS_CASE}. Run add_pv_bess.py first.")
    andes.config_logger(stream_level=40)        # ERROR only (quiet inside the loop)
    ss = andes.load(str(C.PVBESS_CASE), setup=True, no_output=True)
    ss.config.freq = C.FREQ_HZ
    ss.PFlow.run()
    return ss


def split_pq(ss):
    """Return (pvbess_idx_by_bus, real_load_total_MW). PV-BESS plants are the PQ entries
    whose idx starts with 'PVBESS_'; everything else is real load."""
    idxs = list(ss.PQ.idx.v)
    p0 = np.asarray(ss.PQ.p0.v, float)
    pvbess = {}
    load_total = 0.0
    for i, idx in enumerate(idxs):
        if str(idx).startswith("PVBESS_"):
            bus = int(str(idx).split("_")[1])
            pvbess[bus] = idx
        else:
            load_total += p0[i] * C.SBASE_MVA
    return pvbess, load_total


def econ_dispatch(prev, residual, pmin, pmax, ramp, cost):
    """AEMO-style 5-min merit-order economic dispatch with ramp + min/max limits.

    Synchronous units fill `residual` (= load - PV) cheapest-first, each constrained to
    [max(pmin, prev-ramp), min(pmax, prev+ramp)]. Whatever the fleet cannot supply within
    those limits is left for the slack/interconnector (import if short, export if it can't
    back down fast enough). All MW.
    """
    lo = np.maximum(pmin, prev - ramp)
    hi = np.minimum(pmax, prev + ramp)
    lo = np.minimum(lo, hi)
    disp = lo.copy()
    remaining = residual - lo.sum()
    if remaining > 0:
        for u in np.argsort(cost):               # cheapest first
            add = min(hi[u] - disp[u], remaining)
            disp[u] += add
            remaining -= add
            if remaining <= 1e-6:
                break
    return disp


def get_caps(ss, syn_base_mw):
    """Per-unit Pmax/Pmin/ramp/cost for the synchronous fleet (MW)."""
    pmax = np.asarray(ss.PV.pmax.v, float) * C.SBASE_MVA
    # guard against ANDES placeholder limits (e.g. 999 pu): fall back to headroom on base
    bad = (pmax <= 0) | (pmax > 5000)
    pmax = np.where(bad, syn_base_mw * 1.6, pmax)
    pmin = C.GEN_MIN_FRAC * pmax
    ramp = C.GEN_RAMP_FRAC_PER_MIN * pmax * 5.0   # per 5-min interval
    buses = [int(b) for b in ss.PV.bus.v]
    cost = np.array([C.SYNC_MERIT_COST.get(b, 30.0) for b in buses], float)
    return pmax, pmin, ramp, cost


def run_day(ss, inj_mw: np.ndarray, real_load_mw: float, record_net: bool = True):
    """Loop over the intervals; return (rows, net).

    inj_mw: (N, 9) MW injections aligned to config.PV_BUSES.
    net: dict of per-interval transmission observables (line %loading, bus V) for the
         transmission analysis, or None if record_net is False.
    """
    import netobs
    pvbess_idx, _ = split_pq(ss)
    syn_idx = list(ss.PV.idx.v)
    syn_base = np.asarray(ss.PV.p0.v, float) * C.SBASE_MVA
    pmax, pmin, ramp, cost = get_caps(ss, syn_base)
    meta = netobs.line_meta(ss) if record_net else None

    # warm-start the fleet at an unconstrained merit dispatch for interval 0,
    # so there is no artificial startup ramp from the saved 44% snapshot.
    resid0 = max(0.0, real_load_mw - float(inj_mw[0].sum()))
    prev = econ_dispatch(syn_base, resid0, pmin, pmax, pmax, cost)  # ramp=pmax -> unconstrained

    rows, loadpct, vmag = [], [], []
    n = inj_mw.shape[0]
    t0 = time.time()
    for t in range(n):
        pv_row = inj_mw[t]
        pv_total = float(pv_row.sum())

        for j, bus in enumerate(C.PV_BUSES):
            ss.PQ.set("p0", pvbess_idx[bus], -pv_row[j] / C.SBASE_MVA, attr="v")

        residual = max(0.0, real_load_mw - pv_total)
        disp = econ_dispatch(prev, residual, pmin, pmax, ramp, cost)
        for k, idx in enumerate(syn_idx):
            ss.PV.set("p0", idx, disp[k] / C.SBASE_MVA, attr="v")
        prev = disp

        ss.PFlow.run()
        conv = bool(ss.PFlow.converged)
        v = np.asarray(ss.Bus.v.v, float)
        syn_total = float(np.sum(ss.PV.p.v)) * C.SBASE_MVA
        slack = float(np.sum(ss.Slack.p.v)) * C.SBASE_MVA
        loss = syn_total + slack + pv_total - real_load_mw
        rows.append(dict(t=t, pv_mw=pv_total, syn_mw=syn_total, slack_mw=slack,
                         loss_mw=loss, vmin=float(v.min()), vmax=float(v.max()),
                         converged=conv, pen=pv_total / real_load_mw))
        if record_net:
            _, pct = netobs.line_flows(ss, meta)
            loadpct.append(pct)
            vmag.append(v.copy())
        if (t + 1) % 48 == 0 or t + 1 == n:      # progress: bounded to one day (N=288)
            print(f"    interval {t + 1}/{n}  ({time.time() - t0:.1f}s)", flush=True)

    net = None
    if record_net:
        net = dict(loadpct=np.array(loadpct), vmag=np.array(vmag),
                   line_idx=np.array([str(i) for i in meta["line_idx"]]),
                   rate=meta["rate"], bus_idx=np.array(meta["busidx"]),
                   line_from=np.array([meta["busidx"][i] for i in meta["f"]]),
                   line_to=np.array([meta["busidx"][i] for i in meta["t"]]))
    return rows, net


def summarize(rows, date, real_load_mw: float, scenario: str = "a1") -> None:
    import csv
    out = C.RESULTS / f"qsts_{scenario}_{date.date()}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    nonpv = np.array([r["syn_mw"] + r["slack_mw"] for r in rows])
    balancing_mwh = float(np.sum(np.abs(np.diff(nonpv)))) * (5.0 / 60.0)
    pens = np.array([r["pen"] for r in rows])
    nconv = sum(r["converged"] for r in rows)
    vmin = min(r["vmin"] for r in rows); vmax = max(r["vmax"] for r in rows)

    print(f"  day                 : {date.date()}  ({len(rows)} intervals, load {real_load_mw:.0f} MW)")
    print(f"  converged           : {nconv}/{len(rows)}")
    print(f"  PV penetration      : peak {pens.max()*100:.0f}%  mean {pens.mean()*100:.0f}%")
    print(f"  fleet balancing duty: {balancing_mwh:.0f} MWh (Sigma|delta non-PV gen|)")
    print(f"  voltage range (day) : {vmin:.3f} .. {vmax:.3f} pu")
    print(f"  per-interval CSV    : {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="QSTS Study A (a1 raw-PV / a2 rule-firming / a3 DRL agent / a4 MPC).")
    ap.add_argument("--scenario", choices=["a1", "a2", "a3", "a4"], default="a1",
                    help="a1=raw PV injection; a2/a3/a4=firm trajectory (rule/DRL/MPC)")
    ap.add_argument("--klass", choices=["clear", "mixed", "overcast"], default="clear")
    ap.add_argument("--date", default=None, help="explicit YYYY-MM-DD (overrides --klass)")
    ap.add_argument("--traj-dir", type=Path, default=C.RESULTS / "trajectories",
                    help="dir of canonical trajectory CSVs for a2/a3/a4")
    ap.add_argument("--traj-prefix", default=None,
                    help="trajectory filename prefix (default: rule/agent/mpc for a2/a3/a4)")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD,
                    help="PV data period (p2 = unseen evaluation set, default)")
    ap.add_argument("--max-intervals", type=int, default=None,
                    help="cap intervals for a quick smoke test")
    ap.add_argument("--no-record-net", action="store_true",
                    help="skip per-interval line/voltage logging (no qsts_net_*.npz)")
    args = ap.parse_args(argv)

    # firm-trajectory scenarios and their default canonical-file prefixes
    firm_prefix = {"a2": "rule", "a3": "agent", "a4": "mpc"}

    plants = P.load_all(args.period)
    days = P.common_complete_days(plants)
    day = pick_day(plants, days, args.klass, args.date)

    if args.scenario == "a1":
        inj = P.day_injection_matrix(plants, day, column="scada")     # raw PV (volatile)
        label = "raw-PV injection"
    else:
        prefix = args.traj_prefix or firm_prefix[args.scenario]
        traj = T.load_profile_trajectories(args.traj_dir, prefix)
        inj = T.injection_from_trajectories(traj, date=day, column="P_net")
        label = f"firm trajectory ({prefix}, P_net delivered)"
    print(f"Study A QSTS  scenario={args.scenario.upper()}  day={day.date()}  {label}")

    if args.max_intervals:
        inj = inj[: args.max_intervals]
    print(f"  intervals to simulate: {inj.shape[0]} (one day = {C.INTERVALS_PER_DAY})")

    ss = load_case()
    _, real_load = split_pq(ss)
    rows, net = run_day(ss, inj, real_load, record_net=not args.no_record_net)
    summarize(rows, day, real_load, args.scenario)
    if net is not None:
        netout = C.RESULTS / f"qsts_net_{args.scenario}_{day.date()}.npz"
        np.savez_compressed(netout, **net)
        print(f"  network observables : {netout.name} "
              f"(lines {net['loadpct'].shape[1]}, buses {net['vmag'].shape[1]})")
    ok = all(r["converged"] for r in rows)
    print("  RESULT              :", "PASS — QSTS day solved" if ok else
          "PARTIAL — some intervals failed to converge")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
