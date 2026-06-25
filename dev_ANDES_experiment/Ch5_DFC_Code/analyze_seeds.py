#!/usr/bin/env python3
"""
analyze_seeds.py — reduce a multi-seed reward ablation (e.g. firm vs n3 on BANN1) into a
median/IQR table plus a PAIRED-by-seed contrast. Reads the run manifest (run_id -> seed,
reward_fn, bcap) and each run's evaluation trajectory, recomputes the DFC deep-tail metrics
with the SAME definitions used in the verification, and aggregates per reward_fn.

Dependency-light (numpy + pandas; scipy OPTIONAL, only for the paired p-values; NO torch/gymnasium),
so it runs on the laptop after you rsync the results back from Bunya — consistent with the CLAUDE.md
"runs anywhere" guardrail.

Typical use after retrieving /scratch/.../dfc_train/{models,evaluation} into this dir:
    python analyze_seeds.py --merge-shards --dataset BANN1_new --out models/seed_compare.csv

  --merge-shards   concat models/run_manifest_*.csv (the SLURM per-task shards) into the
                   canonical models/run_manifest.csv before analysing (de-dupes by run_id).
  --dataset        the EVAL dataset name (period-2), used to locate evaluation/<ds>/<run_id>_<ds>.csv
  --bcap           optional: restrict to one BESS size (e.g. 0.30)
  --out            write the per-run metric table here (default models/seed_compare.csv)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# scipy is OPTIONAL — it powers the paired significance tests (Wilcoxon / paired t) in the by-seed
# contrast. The script still runs without it (the numpy/pandas-only "runs anywhere" guardrail); the
# p-value column just shows "—" and prints a hint to `pip install scipy`.
try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:   # pragma: no cover
    ttest_rel = wilcoxon = None

# deep-tail metric set (per-unit of nameplate); shortfall = max(Pdfc - Pnet, 0).
GEN_THRESH = 0.02   # "generating interval": available PV potential > 2% nameplate
MET_TOL    = 0.005  # availability_by_plant.py dependability tolerance


def run_metrics(eval_csv: str) -> dict:
    """Recompute the DFC claim metrics for one evaluation trajectory CSV."""
    df = pd.read_csv(eval_csv)
    Pdfc = df["Pdfc"].to_numpy(float)
    Pnet = df["Pnet"].to_numpy(float)
    pot  = df["env_state_pv_potential"].to_numpy(float)
    cr   = df["actual_cr"].to_numpy(float)
    s = np.maximum(Pdfc - Pnet, 0.0)                       # shortfall (under-delivery)
    gen = pot > GEN_THRESH
    net_g = Pnet[gen]
    deficit_g = np.maximum(Pdfc[gen] - net_g, 0.0)
    return {
        "n_steps":            len(df),
        "perfect_rate_092":   100.0 * np.mean(s <= 0.092),
        "any_shortfall_pct":  100.0 * np.mean(s > 0),
        "tot_shortfall_energy": float(s.sum()),
        "mean_shortfall_short": float(s[s > 0].mean()) if np.any(s > 0) else 0.0,
        "max_breach_pu":      float(s.max()),
        "p99_shortfall_pu":   float(np.percentile(s, 99)),
        "deep_gt_005":        int(np.sum(s > 0.05)),
        "deep_gt_020":        int(np.sum(s > 0.20)),
        "dependability_pct":  100.0 * float(np.mean(deficit_g <= MET_TOL)) if net_g.size else float("nan"),
        "firm95_pu":          float(np.quantile(net_g, 0.05)) if net_g.size else float("nan"),
        "mean_Pdfc":          float(Pdfc.mean()),
        "mean_curtailment":   float(cr.mean()),
    }


# metrics where SMALLER is better (for the paired "firm beats n3" sign summary)
LOWER_BETTER = {"any_shortfall_pct", "tot_shortfall_energy", "mean_shortfall_short",
                "max_breach_pu", "p99_shortfall_pu", "deep_gt_005", "deep_gt_020"}
TAIL_KEYS = ["tot_shortfall_energy", "deep_gt_005", "deep_gt_020", "max_breach_pu",
             "p99_shortfall_pu", "dependability_pct", "firm95_pu", "perfect_rate_092"]


def paired_p(fk, nk, test):
    """Two-sided paired p-value (firm vs n3) for one metric's per-seed values. Returns NaN when
    scipy is absent, fewer than 2 usable pairs remain after dropping NaNs, or every per-seed
    difference is zero (no signal to test). `test` is scipy's wilcoxon or ttest_rel (or None)."""
    if test is None:
        return float("nan")
    mask = ~(np.isnan(fk) | np.isnan(nk))
    a, b = fk[mask], nk[mask]
    if len(a) < 2 or np.allclose(a - b, 0.0):
        return float("nan")
    try:
        return float(test(a, b).pvalue)
    except ValueError:           # e.g. all differences zero after wilcoxon's zero-handling
        return float("nan")


def fmt_p(p):
    """Render a p-value with a significance star (* p<0.05, ** p<0.01); em-dash when undefined."""
    if np.isnan(p):
        return "—"
    return f"{p:.4g}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")


def merge_shards(models_dir: str) -> str:
    """Concat models/run_manifest_*.csv shards (+ any base) into models/run_manifest.csv."""
    shards = sorted(glob.glob(os.path.join(models_dir, "run_manifest_*.csv")))
    base = os.path.join(models_dir, "run_manifest.csv")
    frames = [pd.read_csv(p) for p in shards if os.path.getsize(p) > 0]
    if os.path.isfile(base) and os.path.getsize(base) > 0:
        frames.append(pd.read_csv(base))
    if not frames:
        sys.exit(f"merge-shards: no run_manifest_*.csv shards or base manifest under {models_dir}")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="run_id", keep="last")
    merged.to_csv(base, index=False)
    print(f"merge-shards: {len(shards)} shard(s) -> {base} ({len(merged)} unique runs)")
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="models/run_manifest.csv")
    ap.add_argument("--eval-dir", default="evaluation")
    ap.add_argument("--dataset",  default="BANN1_new", help="EVAL (period-2) dataset name")
    ap.add_argument("--bcap",     type=float, default=None, help="restrict to this BESS size")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--merge-shards", action="store_true", help="concat run_manifest_*.csv first")
    ap.add_argument("--out", default="models/seed_compare.csv")
    ap.add_argument("--ttest", action="store_true",
                    help="also show a paired t-test p-value column (parametric; Wilcoxon is default)")
    args = ap.parse_args()

    manifest = merge_shards(args.models_dir) if args.merge_shards else args.manifest
    if not os.path.isfile(manifest):
        sys.exit(f"manifest not found: {manifest} (did you --merge-shards / retrieve results?)")
    man = pd.read_csv(manifest)
    if "reward_fn" not in man.columns:
        man["reward_fn"] = "firm"   # legacy manifests predate the column; all were the firm reward
    if args.bcap is not None:
        man = man[np.isclose(man["bcap"].astype(float), args.bcap)]

    # Collapse to ONE run per (reward_fn, seed). Re-running a seed (a smoke test, a local one-off, a
    # resubmitted array) leaves several manifest rows sharing (reward_fn, seed) but with different
    # run_ids — which makes the paired-by-seed contrast below subtract unequal-length vectors
    # (ValueError: operands could not be broadcast ... (5,) (4,)). Keep the most RECENT run per seed
    # (max timestamp; a rerun supersedes the old one); fall back to file order if there's no timestamp.
    dedup_key = ["reward_fn", "seed"]
    if "timestamp" in man.columns:
        man = man.sort_values("timestamp")
    dropped = man[man.duplicated(subset=dedup_key, keep="last")]
    if len(dropped):
        print(f"[dedupe] dropping {len(dropped)} older duplicate (reward_fn, seed) run(s); "
              f"kept the most recent per seed:")
        for _, d in dropped.iterrows():
            ts = f"  {d['timestamp']}" if "timestamp" in man.columns else ""
            print(f"    drop {d['run_id']}  reward={d['reward_fn']} seed={d['seed']}{ts}")
    man = man.drop_duplicates(subset=dedup_key, keep="last")

    rows = []
    for _, r in man.iterrows():
        run_id = str(r["run_id"])
        eval_csv = os.path.join(args.eval_dir, args.dataset, f"{run_id}_{args.dataset}.csv")
        if not os.path.isfile(eval_csv):
            print(f"  skip {run_id}: no eval CSV at {eval_csv}")
            continue
        m = run_metrics(eval_csv)
        m.update(run_id=run_id, reward_fn=r.get("reward_fn", "firm"),
                 seed=r.get("seed", ""), bcap=r.get("bcap", ""))
        rows.append(m)
    if not rows:
        sys.exit("no runs matched manifest x eval CSVs — check --dataset / retrieval")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out, index=False)
    print(f"\nper-run metrics ({len(per_run)} runs) -> {args.out}")

    # ---- median / IQR by reward_fn ------------------------------------------------------------
    print(f"\n=== median [IQR] by reward_fn on {args.dataset} "
          f"(bcap={args.bcap if args.bcap is not None else 'all'}) ===")
    hdr = f"{'metric':24s}" + "".join(f"{g:>22s}" for g in sorted(per_run['reward_fn'].unique()))
    print(hdr); print("-" * len(hdr))
    for k in TAIL_KEYS:
        line = f"{k:24s}"
        for g in sorted(per_run["reward_fn"].unique()):
            v = per_run.loc[per_run.reward_fn == g, k].astype(float)
            line += f"{f'{v.median():.4g} [{v.quantile(.25):.4g},{v.quantile(.75):.4g}]':>22s}"
        print(line)
    counts = per_run.groupby("reward_fn").size().to_dict()
    print(f"\nn runs per reward: {counts}")

    # ---- paired-by-seed contrast: does firm beat n3 on the SAME seed? --------------------------
    if {"firm", "n3"}.issubset(set(per_run["reward_fn"])):
        f = per_run[per_run.reward_fn == "firm"].set_index("seed")
        n = per_run[per_run.reward_fn == "n3"].set_index("seed")
        seeds = sorted(set(f.index) & set(n.index))
        if seeds:
            print(f"\n=== paired firm-vs-n3 by seed (n={len(seeds)} paired seeds) ===")
            hdr = (f"{'metric':24s}{'firm better in':>16s}{'median Δ(firm−n3)':>20s}"
                   f"{'median %vs n3':>16s}{'wilcoxon p':>14s}")
            if args.ttest:
                hdr += f"{'paired-t p':>14s}"
            print(hdr); print("-" * len(hdr))
            for k in TAIL_KEYS:
                fk = f.loc[seeds, k].astype(float).to_numpy()
                nk = n.loc[seeds, k].astype(float).to_numpy()
                d = fk - nk
                rel = (nk - fk) if k in LOWER_BETTER else (fk - nk)
                wins = int(np.sum(fk < nk if k in LOWER_BETTER else fk > nk))
                with np.errstate(divide="ignore", invalid="ignore"):   # n3==0 -> % undefined (use Δ col)
                    ratios = np.where(nk != 0, rel / np.abs(nk) * 100.0, np.nan)
                pct = float(np.nanmedian(ratios)) if np.any(~np.isnan(ratios)) else float("nan")
                row = (f"{k:24s}{f'{wins}/{len(seeds)}':>16s}{np.median(d):>20.4g}"
                       f"{pct:>15.1f}%{fmt_p(paired_p(fk, nk, wilcoxon)):>14s}")
                if args.ttest:
                    row += f"{fmt_p(paired_p(fk, nk, ttest_rel)):>14s}"
                print(row)
            note = ("\n(firm 'better' = lower for shortfall/breach metrics, higher for "
                    "perfect_rate/dependability/firm95. A gap that holds across most seeds is "
                    "real; one that flips seed-to-seed is PPO noise.)")
            if wilcoxon is None:
                note += "\n(wilcoxon p column needs scipy: pip install scipy)"
            else:
                note += ("\n(p = two-sided Wilcoxon signed-rank on the per-seed firm−n3 differences; "
                         "* p<0.05, ** p<0.01. Needs n≥6 paired seeds to even reach p<0.05.)")
            if args.ttest:
                note += ("\n(paired-t p assumes ~normal differences — unreliable at small n and prone "
                         "to over-claiming; prefer the Wilcoxon column as the headline test.)")
            print(note)
        else:
            print("\n(no shared seeds between firm and n3 — cannot pair)")


if __name__ == "__main__":
    main()
