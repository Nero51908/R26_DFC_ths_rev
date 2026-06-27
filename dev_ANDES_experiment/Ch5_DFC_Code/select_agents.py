#!/usr/bin/env python3
"""
select_agents.py — pick the best DRL agent per (plant, bcap) for Study A.

To avoid selecting on the unseen period-2 test set (selection bias), candidates are RANKED on
their PERIOD-1 (*_old, training-era) DFC metrics, and the winner's PERIOD-2 (*_new) metrics are
REPORTED — alongside the period-2 across-seed median/IQR, so the pick is visible in the context of
seed variance. Reads models/run_manifest.csv (run_id, reward_fn, seed, bcap, dataset) + each
candidate's eval trajectory. The metric definitions MIRROR ../src/availability_by_plant.py
(generating intervals only, % of nameplate) so selection is consistent with the final Study-A scoring.

Primary ranking key: shortfall_pct = under-delivery energy / available PV energy (depth-weighted) —
the firmness failure the reward targets. committed (mean Pdfc) and track_mae (two-sided prediction
error) are REPORTED, not ranked, so a degenerate low-committer is visible. perfect_rate = the
commitment-met % (the thesis name for the reliability metric).

Outputs:  models/selection.csv (the winners, with period-2 report metrics) and
models/agent_metrics.csv (PERIOD-2 metrics for EVERY candidate, with a `selected` flag). Both carry
the breach tail — breach_depth (mean) / breach_p95 / breach_p99 / breach_max (worst under-delivery,
% of nameplate) — kept for Study-B contingency sizing.

numpy/pandas only — runs anywhere (no torch).

    python select_agents.py                                # all firm agents, rank on *_old
    python select_agents.py --plants BANN1 --bcap 0.30
    python select_agents.py --primary breach_p95           # rank by a different key
After it prints the export commands, run them to refresh ../results/trajectories/agent_<PLANT>.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# definitions mirror ../src/availability_by_plant.py — do NOT let these drift.
GEN_THRESH = 0.02     # "generating" interval: available PV potential > 2% of nameplate
MET_TOL    = 0.005    # commitment met if deficit = max(Pdfc - Pnet, 0) <= this
COL = {"avail": "env_state_pv_potential", "commit": "Pdfc", "net": "Pnet"}   # eval-recorder schema

# metrics where SMALLER is better (controls --primary sort direction)
LOWER_BETTER = {"shortfall_pct", "breach_depth", "breach_p95", "breach_p99", "breach_max",
                "track_mae", "curtail_pct"}


def dfc_metrics(eval_csv: str) -> dict:
    """Canonical per-plant DFC metrics on generating intervals (% of nameplate)."""
    df = pd.read_csv(eval_csv)
    avail = df[COL["avail"]].to_numpy(float)
    g = avail > GEN_THRESH
    commit, net, avail = df[COL["commit"]].to_numpy(float)[g], df[COL["net"]].to_numpy(float)[g], avail[g]
    deficit = np.maximum(commit - net, 0.0)
    breach = deficit[deficit > MET_TOL]
    asum = max(avail.sum(), 1e-9)
    return {
        "n_gen":         int(g.sum()),
        "shortfall_pct": 100 * deficit.sum() / asum,           # PRIMARY (firmness failure energy)
        "perfect_rate":  100 * float(np.mean(deficit <= MET_TOL)),  # commitment-met % (thesis name)
        "committed":     100 * float(commit.mean()),           # mean Pdfc — guard vs low-committer
        "delivered":     100 * float(net.mean()),
        "track_mae":     100 * float(np.mean(np.abs(commit - net))),  # two-sided accuracy (context)
        "breach_depth":  100 * float(breach.mean()) if breach.size else 0.0,
        "breach_p95":    100 * float(np.percentile(breach, 95)) if breach.size else 0.0,
        "breach_p99":    100 * float(np.percentile(breach, 99)) if breach.size else 0.0,
        "breach_max":    100 * float(breach.max()) if breach.size else 0.0,   # worst under-delivery (Study-B contingency size)
        "curtail_pct":   100 * max(avail.sum() - net.sum(), 0.0) / asum,
    }


def eval_path(eval_dir: str, plant: str, suffix: str, run_id: str) -> str:
    ds = f"{plant}_{suffix}"
    return os.path.join(eval_dir, ds, f"{run_id}_{ds}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="models/run_manifest.csv")
    ap.add_argument("--eval-dir", default="evaluation")
    ap.add_argument("--reward", default="firm", help="only consider agents trained with this reward")
    ap.add_argument("--select-suffix", default="old", help="period to RANK on (period-1, training-era)")
    ap.add_argument("--report-suffix", default="new", help="period to REPORT/export (period-2, unseen)")
    ap.add_argument("--bcap", type=float, default=None, help="restrict to one BESS size")
    ap.add_argument("--plants", nargs="+", default=None, help="restrict to these plants")
    ap.add_argument("--primary", default="shortfall_pct", help="metric to rank on (default shortfall_pct)")
    ap.add_argument("--traj-dir", default="../results/trajectories", help="export_trajectory output dir")
    ap.add_argument("--out", default="models/selection.csv")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    if "reward_fn" not in man.columns:
        man["reward_fn"] = "firm"
    man = man[man.reward_fn == args.reward].copy()
    if man.empty:
        sys.exit(f"no '{args.reward}' agents in {args.manifest}")
    man["plant"] = man["dataset"].astype(str).str.rsplit("_", n=1).str[0]   # "<plant>_old" -> "<plant>"
    if args.bcap is not None:
        man = man[np.isclose(man["bcap"].astype(float), args.bcap)]
    if args.plants:
        man = man[man.plant.isin(args.plants)]
    if "timestamp" in man.columns:                 # one run per (plant,bcap,seed): newest supersedes
        man = man.sort_values("timestamp")
    man = man.drop_duplicates(["plant", "bcap", "seed"], keep="last")

    # 1. score every candidate on the SELECT period
    rows, missing = [], []
    for _, r in man.iterrows():
        p, b, rid = r["plant"], float(r["bcap"]), str(r["run_id"])
        sp = eval_path(args.eval_dir, p, args.select_suffix, rid)
        if not os.path.isfile(sp):
            missing.append((rid, p, sp)); continue
        m = dfc_metrics(sp)
        m.update(run_id=rid, plant=p, bcap=b, seed=r.get("seed", ""))
        rows.append(m)
    for rid, p, sp in missing:
        print(f"  skip {rid} ({p}): no {args.select_suffix} eval at {sp}")
    if not rows:
        sys.exit(f"no candidates had a '{args.select_suffix}' eval CSV — run the period-{args.select_suffix} "
                 "evaluation first (the training job emits it; for existing models eval with --datae <plant>_old).")
    cand = pd.DataFrame(rows)
    if args.primary not in cand.columns:
        sys.exit(f"--primary {args.primary} not a metric; choose from {sorted(set(cand.columns)-{'run_id','plant','bcap','seed'})}")

    asc = args.primary in LOWER_BETTER
    # 2. rank within (plant, bcap): primary, then breach_p95 (tail), then committed desc (anti-degenerate)
    winners = []
    print(f"\n=== candidate ranking on period-1 (*_{args.select_suffix}), primary={args.primary} "
          f"({'lower' if asc else 'higher'} better) ===")
    for (p, b), grp in cand.groupby(["plant", "bcap"]):
        grp = grp.sort_values([args.primary, "breach_p95", "committed"],
                              ascending=[asc, True, False]).reset_index(drop=True)
        winners.append(grp.iloc[0])
        show = ["seed", "run_id", args.primary, "perfect_rate", "committed", "track_mae", "breach_p95"]
        print(f"\n{p} (bcap {b}) — {len(grp)} seeds:")
        print(grp[show].round(3).to_string(index=False))
        print(f"  -> winner seed {grp.iloc[0]['seed']} ({grp.iloc[0]['run_id']})")

    # 3. PERIOD-2 (held-out) report. Compute period-2 metrics for EVERY candidate so the breach
    #    depth is KEPT for all agents (-> models/agent_metrics.csv, for Study-B contingency sizing),
    #    then summarise each winner against its across-seed median/IQR.
    print(f"\n=== winners — period-2 (*_{args.report_suffix}) report (held-out) ===")
    winner_ids = {w["run_id"] for w in winners}
    win_by_group = {(w["plant"], float(w["bcap"])): w for w in winners}
    report_rows, sel_rows = [], []
    for (p, b), grp in cand.groupby(["plant", "bcap"]):
        rep_by_rid = {}
        for _, r in grp.iterrows():
            rid = str(r["run_id"])
            rp = eval_path(args.eval_dir, p, args.report_suffix, rid)
            m2 = dfc_metrics(rp) if os.path.isfile(rp) else {}
            rep_by_rid[rid] = m2
            report_rows.append({"plant": p, "bcap": b, "seed": r["seed"], "run_id": rid,
                                "selected": rid in winner_ids,
                                **{k: round(float(v), 4) for k, v in m2.items() if k != "n_gen"}})
        win = win_by_group[(p, b)]
        rid = win["run_id"]; rep = rep_by_rid.get(rid, {})
        dist = [m[args.primary] for m in rep_by_rid.values() if args.primary in m]
        med = float(np.median(dist)) if dist else float("nan")
        lo, hi = (float(np.percentile(dist, 25)), float(np.percentile(dist, 75))) if dist else (float("nan"),) * 2
        sel_rows.append({
            "plant": p, "bcap": b, "seed": win["seed"], "run_id": rid,
            f"sel_{args.primary}": round(float(win[args.primary]), 4),
            **{f"rep_{k}": round(float(v), 4) for k, v in rep.items() if k != "n_gen"},
            f"rep_{args.primary}_seedmedian": round(med, 4),
            f"rep_{args.primary}_seedIQR": f"[{lo:.4g},{hi:.4g}]",
        })
        print(f"  {p} bcap {b}: winner {rid} (seed {win['seed']})  period-2 {args.primary}="
              f"{rep.get(args.primary, float('nan')):.4g} vs seed median {med:.4g} IQR [{lo:.4g},{hi:.4g}]"
              f"   breach mean {rep.get('breach_depth', float('nan')):.3g} / p95 {rep.get('breach_p95', float('nan')):.3g}"
              f" / max {rep.get('breach_max', float('nan')):.3g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(sel_rows).to_csv(args.out, index=False)
    print(f"\nselection (winners) -> {args.out}")
    metrics_out = os.path.join(os.path.dirname(args.out) or ".", "agent_metrics.csv")
    pd.DataFrame(report_rows).to_csv(metrics_out, index=False)
    print(f"per-candidate period-2 metrics (breach depth KEPT for all agents) -> {metrics_out}")

    # 4. emit the export_trajectory command to promote each winner's PERIOD-2 eval to Study A
    print("\n=== promote winners to Study-A trajectories (run these) ===")
    for win in winners:
        p, rid = win["plant"], win["run_id"]
        src = eval_path(args.eval_dir, p, args.report_suffix, rid)
        print(f"python ../src/export_trajectory.py --in {src} "
              f"--out {os.path.join(args.traj_dir, f'agent_{p}.csv')}")


if __name__ == "__main__":
    main()
