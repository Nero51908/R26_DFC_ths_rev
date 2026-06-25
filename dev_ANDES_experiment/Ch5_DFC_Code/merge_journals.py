#!/usr/bin/env python3
"""
merge_journals.py — reduce the per-task evaluation-journal SHARDS that the SLURM array writes into
one canonical journal per dataset.

On the HPC each array task journals to its OWN file, suffixed with DFC_OUTPUT_TAG=<jobid>_<task>:

    evaluation_journal_<dataset>_<kind>_<jobid>_<task>.csv      (kind = model | dummy)

so concurrent tasks never append to the same file and race. After you fetch them back
(WANT_JOURNALS=1 in slurm/fetch_results_sftp.sh they land in the checkout root), this concatenates
each (dataset, kind) group into the canonical, un-suffixed file:

    evaluation_journal_<dataset>_<kind>.csv

de-duped by run_id (the `actor` column). It is the journal twin of `analyze_seeds.py --merge-shards`
(which does the same for models/run_manifest.csv). numpy/pandas only — runs on the laptop.

    python merge_journals.py                 # merge every <dataset>/<kind> group found in .
    python merge_journals.py --dir .         # explicit dir (default: current)
    python merge_journals.py --keep-shards      # KEEP the per-task shards (default: delete them post-merge)
    python merge_journals.py --no-manifest      # skip the reward_fn/seed enrichment
    python merge_journals.py --latest-per-seed  # ALSO write *_model_latest.csv (1 run/seed; full log kept)

Notes:
  * `model` and `dummy` stay SEPARATE — the dummy is the naive POE50 baseline, not an agent.
  * Only files WITH a <jobid>_<task> tag are treated as shards; the canonical base and any untagged
    local journals are read in (so their rows are preserved) but never globbed as shards.
  * The journal has no reward_fn/seed; if models/run_manifest.csv is present, the merged MODEL
    journal is left-joined with reward_fn + seed on run_id so every row says which arm it came from.
  * Default output is the COMPLETE log (every run, de-duped only by run_id). --latest-per-seed
    ADDITIONALLY writes evaluation_journal_<dataset>_model_latest.csv with just the most recent run
    per (reward_fn, seed) — same rule as analyze_seeds.py — leaving the full log untouched.
  * After a successful merge the per-task shards are DELETED by default (their rows are now in the
    canonical journal, and they are re-fetchable from the HPC). Pass --keep-shards to retain them.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

# evaluation_journal_<dataset>_<kind>_<jobid>_<task>.csv   (tag = jobid_task, both numeric)
SHARD_RE = re.compile(
    r"^evaluation_journal_(?P<dataset>.+)_(?P<kind>model|dummy)_(?P<job>\d+)_(?P<task>\d+)\.csv$"
)
RUN_KEY = "actor"   # the journal's run_id column


def _finalize(df):
    """Tidy a merged journal for writing: move reward_fn/seed up next to actor, drop the timestamp
    helper (a manifest detail, not journal-native), and sort arm→seed→actor for readability."""
    df = df.drop(columns=[c for c in ("timestamp",) if c in df.columns])
    front = [c for c in ["dataset", RUN_KEY, "reward_fn", "seed"] if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    sort_cols = [c for c in ["dataset", "reward_fn", "seed", RUN_KEY] if c in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True) if sort_cols else df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="dir holding the journal CSVs (the checkout root)")
    ap.add_argument("--manifest", default="models/run_manifest.csv",
                    help="run manifest used to enrich model journals with reward_fn/seed")
    ap.add_argument("--no-manifest", action="store_true",
                    help="do NOT enrich model journals with reward_fn/seed")
    ap.add_argument("--keep-shards", action="store_true",
                    help="KEEP the per-task journal shards (default: delete them after a successful "
                         "merge — their rows are preserved in the canonical journal)")
    ap.add_argument("--latest-per-seed", action="store_true",
                    help="ALSO write evaluation_journal_<ds>_model_latest.csv keeping only the most "
                         "recent run per (reward_fn, seed) — matches analyze_seeds.py; the full log "
                         "is left intact")
    args = ap.parse_args()

    # 1. discover shards, grouped by (dataset, kind); skip the canonical base + untagged journals.
    groups: dict[tuple[str, str], list[str]] = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "evaluation_journal_*.csv"))):
        m = SHARD_RE.match(os.path.basename(path))
        if m:
            groups.setdefault((m["dataset"], m["kind"]), []).append(path)
    if not groups:
        sys.exit("no journal shards (evaluation_journal_<dataset>_<kind>_<jobid>_<task>.csv) found "
                 f"in {os.path.abspath(args.dir)} — already merged & cleaned (shards are deleted by "
                 "default), or not fetched yet (WANT_JOURNALS=1)?")

    # 2. load the manifest once (for the reward_fn/seed enrichment of model journals).
    man = None
    if not args.no_manifest:
        mpath = args.manifest if os.path.isabs(args.manifest) else os.path.join(args.dir, args.manifest)
        if os.path.isfile(mpath) and os.path.getsize(mpath) > 0:
            man = pd.read_csv(mpath)

    # 3. merge each (dataset, kind) group into its canonical file.
    for (dataset, kind), shards in sorted(groups.items()):
        base = os.path.join(args.dir, f"evaluation_journal_{dataset}_{kind}.csv")
        frames = [pd.read_csv(p) for p in shards if os.path.getsize(p) > 0]
        if os.path.isfile(base) and os.path.getsize(base) > 0:
            frames.append(pd.read_csv(base))   # preserve existing/local rows
        merged = pd.concat(frames, ignore_index=True)

        # Drop columns left over from a PRIOR enrichment so re-runs are idempotent AND self-healing.
        # The canonical file already carries reward_fn/seed, and an older non-idempotent merge may
        # have left reward_fn_x/_y, seed_x/_y. These are all DERIVED from the manifest and re-added
        # fresh below; keeping them makes the manifest merge collide (pandas MergeError on duplicate
        # columns). The raw journal has no reward_fn*/seed*/run_id columns, so this only sheds cruft.
        derived = [c for c in merged.columns
                   if c in ("reward_fn", "seed", "run_id") or c.startswith(("reward_fn_", "seed_"))]
        if derived:
            merged = merged.drop(columns=derived)

        dupes = 0
        if RUN_KEY in merged.columns:
            n0 = len(merged)
            merged = merged.drop_duplicates(subset=RUN_KEY, keep="last")   # a re-run supersedes
            dupes = n0 - len(merged)

        # enrich the MODEL journal with reward_fn + seed from the manifest (run_id == actor); also
        # pull timestamp when we'll need it to order the --latest-per-seed trim.
        enriched = ""
        if kind == "model" and man is not None and RUN_KEY in merged.columns and "run_id" in man.columns:
            cols = [c for c in ["run_id", "reward_fn", "seed"] if c in man.columns]
            if args.latest_per_seed and "timestamp" in man.columns:
                cols.append("timestamp")
            merged = merged.merge(man[cols].drop_duplicates("run_id"),
                                  how="left", left_on=RUN_KEY, right_on="run_id")
            merged = merged.drop(columns=["run_id"])
            n_missing = int(merged["reward_fn"].isna().sum()) if "reward_fn" in merged.columns else 0
            enriched = "; +reward_fn/seed from manifest" + (f" ({n_missing} unmatched)" if n_missing else "")

        # write the FULL log (every run, de-duped only by run_id) to the canonical file.
        full = _finalize(merged)
        full.to_csv(base, index=False)
        msg = f"{dataset}/{kind}: {len(shards)} shard(s) -> {base}  ({len(full)} runs"
        msg += f", {dupes} dup run_id dropped" if dupes else ""
        print(msg + ")" + enriched)
        if kind == "model" and {"reward_fn", "seed"}.issubset(full.columns):
            print(f"    by reward_fn: {full.groupby('reward_fn').size().to_dict()}")

        # --latest-per-seed: ALSO emit a trimmed view keeping the most RECENT run per (reward_fn,
        # seed) — same rule as analyze_seeds.py, so it agrees with the paired stats. Written to a
        # SEPARATE *_latest.csv; the full log above is left intact.
        if args.latest_per_seed:
            if {"reward_fn", "seed"}.issubset(merged.columns):
                trim = (merged.sort_values("timestamp", na_position="first")   # real ts beat NaN
                        if "timestamp" in merged.columns else merged)
                trim = _finalize(trim.drop_duplicates(subset=["reward_fn", "seed"], keep="last"))
                latest_path = os.path.join(args.dir, f"evaluation_journal_{dataset}_{kind}_latest.csv")
                trim.to_csv(latest_path, index=False)
                print(f"    latest-per-seed: kept {len(trim)} of {len(full)} runs -> {latest_path}")
            else:
                print("    latest-per-seed: skipped (no reward_fn/seed — need the manifest)")

        # By default, delete the per-task shards once their rows are safely in the canonical journal
        # (re-fetchable from the HPC if ever needed). --keep-shards retains them.
        if not args.keep_shards:
            if os.path.isfile(base) and os.path.getsize(base) > 0:
                for p in shards:
                    os.remove(p)
                print(f"    deleted {len(shards)} merged shard(s) -> rows now in {base} (--keep-shards to retain)")
            else:
                print(f"    KEPT {len(shards)} shard(s): merged file {base} missing/empty — refusing to delete")


if __name__ == "__main__":
    main()
