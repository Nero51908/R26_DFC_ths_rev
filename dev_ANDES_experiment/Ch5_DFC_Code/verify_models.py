#!/usr/bin/env python3
"""
verify_models.py — check that DRL models fetched from the HPC reconstruct into working agents.

Two levels:
  LEVEL 1 (always): load every models/<run_id>/model.zip with PPO.load and run one deterministic
    predict. Proves the Bunya-trained weights deserialize into a working policy in THIS environment.
    Reports the training stack SB3 embedded in each model (system_info.txt) vs the local stack, and
    flags the training-state caveat — if clip_range/lr_schedule fail to deserialize across an SB3
    skew, that is harmless for EVALUATION but means you cannot RESUME training without matching SB3.
  LEVEL 2 (--roundtrip RUN_ID): reload that model, re-run the deterministic evaluation locally, and
    compare the regenerated trajectory to the fetched evaluation/<ds>/<run_id>_<ds>.csv. A match to
    float tolerance proves FAITHFUL reconstruction. Non-mutating: it restores the fetched CSV + journal.

Unlike analyze_seeds.py / merge_journals.py, this NEEDS the training stack (SB3 + torch + gymnasium,
and the env for --roundtrip) — run it in the venv, not the numpy/pandas-only analysis environment.

    python verify_models.py                                   # level 1 over models/run_manifest.csv
    python verify_models.py --roundtrip guxb3clv --bcap 0.30  # + faithful round-trip for one run
    DFC_DATA_ROOT=$PWD/../data python verify_models.py --roundtrip <id>   # if data lives elsewhere
"""
import argparse
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd

STACK_KEYS = ("Python", "Stable-Baselines3", "PyTorch", "Numpy", "Gymnasium")
CMP_COLS = ["Pnet", "Pdfc", "actual_cr", "env_state_pv_potential"]


def model_stub(models_dir, run_id):
    return os.path.join(models_dir, run_id, "model")   # PPO.load appends .zip


def training_stack(zip_path):
    """Versions SB3 embedded at save time (system_info.txt) — the model's provenance."""
    info = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "system_info.txt" in zf.namelist():
                for line in zf.read("system_info.txt").decode().splitlines():
                    line = line.lstrip("- ").strip()
                    if ":" in line:
                        k, v = line.split(":", 1)
                        info[k.strip()] = v.strip()
    except (zipfile.BadZipFile, OSError):
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--manifest", default="models/run_manifest.csv")
    ap.add_argument("--roundtrip", metavar="RUN_ID", help="also do a faithful eval round-trip for this run")
    ap.add_argument("--dataset", default="BANN1_new", help="eval dataset for --roundtrip")
    ap.add_argument("--bcap", type=float, default=0.30, help="BESS size for --roundtrip (match the run)")
    ap.add_argument("--tol", type=float, default=1e-4, help="max|Δ| tolerance for a 'faithful' verdict")
    args = ap.parse_args()

    # SB3/torch import is heavy and noisy; defer it so --help stays fast and dependency-free.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import gymnasium
        import stable_baselines3 as sb3
        import torch
        from stable_baselines3 import PPO
    print(f"local load env: SB3 {sb3.__version__}, torch {torch.__version__}, gymnasium {gymnasium.__version__}\n")

    if not os.path.isfile(args.manifest):
        sys.exit(f"manifest not found: {args.manifest} (did you fetch + merge the run shards?)")
    run_ids = [str(r) for r in pd.read_csv(args.manifest)["run_id"]]

    # ---- LEVEL 1: load + predict every model ------------------------------------------------------
    ok, bad, train_skew, sb3_versions = [], [], False, {}
    for rid in run_ids:
        zp = model_stub(args.models_dir, rid) + ".zip"
        if not os.path.isfile(zp):
            bad.append((rid, "no model.zip")); continue
        ver = training_stack(zp).get("Stable-Baselines3", "?")
        sb3_versions[ver] = sb3_versions.get(ver, 0) + 1
        try:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                model = PPO.load(model_stub(args.models_dir, rid), device="cpu")
            if any("deserialize" in str(w.message) for w in wl):
                train_skew = True
            action, _ = model.predict(model.observation_space.sample(), deterministic=True)
            assert model.action_space.contains(action), "predicted action outside action_space"
            ok.append(rid)
        except Exception as e:                                # noqa: BLE001 - report any load failure
            bad.append((rid, f"{type(e).__name__}: {e}"))

    print(f"LEVEL 1  reconstructed (PPO.load + predict): {len(ok)}/{len(run_ids)} OK")
    for rid, why in bad:
        print(f"  FAIL {rid}: {why}")
    if not bad:
        print("  -> every fetched model reloads into a working PPO agent here ✓")

    print(f"\n  model provenance — SB3 version(s) in the model files: {sb3_versions}")
    if ok:
        s = training_stack(model_stub(args.models_dir, ok[0]) + ".zip")
        print("  sample model trained on:", ", ".join(f"{k}={s[k]}" for k in STACK_KEYS if k in s))
    if train_skew:
        print("  ⚠ clip_range/lr_schedule did NOT deserialize (SB3 version skew vs local).\n"
              "    Harmless for EVALUATION/inference; to RESUME training, match the model's SB3 version.")

    # ---- LEVEL 2: faithful round-trip for one run -------------------------------------------------
    if args.roundtrip:
        roundtrip(args)


def roundtrip(args):
    """Re-evaluate one reloaded model locally and diff the trajectory against the fetched CSV."""
    import shutil

    rid = args.roundtrip
    zp = model_stub(args.models_dir, rid) + ".zip"
    fetched = os.path.join("evaluation", args.dataset, f"{rid}_{args.dataset}.csv")
    if not os.path.isfile(zp):
        sys.exit(f"\n--roundtrip: no model for {rid} under {args.models_dir}/")
    if not os.path.isfile(fetched):
        sys.exit(f"\n--roundtrip: no fetched eval CSV to compare against: {fetched}")

    print(f"\nLEVEL 2  faithful round-trip for {rid} on {args.dataset} (bcap={args.bcap})")
    import application  # noqa: E402 - heavy; only needed for the round-trip
    import config       # noqa: E402

    bess = dict(config.bess_properties)
    bess["energy_capacity_puh"] = args.bcap
    journal = f"evaluation_journal_{args.dataset}_model.csv"
    journal_before = pd.read_csv(journal) if os.path.isfile(journal) else None
    bak = fetched + ".bunya.bak"
    shutil.move(fetched, bak)   # move aside: the env APPENDS to csv_path, so it must start absent
    try:
        application.evaluate_model(rid, args.dataset, bess)   # reload + re-evaluate -> writes `fetched`
        a, b = pd.read_csv(bak), pd.read_csv(fetched)
        if len(a) != len(b):
            print(f"  row mismatch: bunya={len(a)} local={len(b)} — cannot compare")
            return
        mx = 0.0
        for c in CMP_COLS:
            if c in a.columns and c in b.columns:
                d = np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
                mx = max(mx, float(d.max()))
                print(f"  {c:24s} max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}")
        print(f"  => overall max|Δ|={mx:.3e} (tol {args.tol:g})  -> "
              f"{'FAITHFUL ✓' if mx <= args.tol else 'DIVERGES ✗'}")
    finally:
        shutil.move(bak, fetched)                 # restore the fetched Bunya CSV
        if journal_before is not None:
            journal_before.to_csv(journal, index=False)   # undo the row evaluate_model appended
        elif os.path.isfile(journal):
            os.remove(journal)
        print("  (restored fetched CSV + journal — no side effects)")


if __name__ == "__main__":
    main()
