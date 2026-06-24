#!/usr/bin/env python3
"""
build_case.py — load ANDES's bundled, validated IEEE 39-bus case, solve the power flow,
and sanity-check it. This is the foundation for the chapter (Option B): rather than
import the PSS/E .raw (whose transformer taps the ANDES parser mishandled, collapsing the
voltage profile), we build on ANDES's own ieee39 case, which is verified against
commercial software. Our modifications (~44% PV-BESS Config-B build, machine retirement,
slack-as-interconnector) are applied on top in later steps. The system keeps its NATIVE
60 Hz base throughout (decided); `--freq 50` is an opt-in relabel only for NEM-standard
frequency plots, not the default.

The original PSS/E case is retained only as an optional EXTERNAL validation reference for
a single snapshot (cases/ieee39_pf_reference.csv), not as the working base.

Run (inside the ANDES venv):
    python src/build_case.py                 # load bundled ieee39 (native 60 Hz), solve PF
    python src/build_case.py --freq 50       # opt-in: relabel base frequency to 50 Hz
    python src/build_case.py --list-cases    # show candidate bundled case names

Exit code 0 = converged and sane; 1 = problem.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Bundled ANDES cases to try, in order (full = power flow + dynamics).
CANDIDATE_CASES = ["ieee39/ieee39_full.xlsx", "ieee39/ieee39.xlsx"]


def resolve_case(name: str | None):
    """Return an absolute path to a bundled ANDES case (or the user-supplied one)."""
    import andes

    candidates = [name] if name else CANDIDATE_CASES
    for cand in candidates:
        try:
            path = Path(andes.get_case(cand))
        except Exception:
            continue
        if path.exists():
            return path
    sys.exit(f"Could not locate a bundled ANDES ieee39 case. Tried: {candidates}. "
             f"Run with --list-cases or check your ANDES install.")


def build_system(case: Path, freq_hz: float | None):
    """Load the ANDES case and solve the power flow. Returns the System object."""
    try:
        import andes
    except ImportError:
        sys.exit("ANDES not installed. In the venv:  pip install andes  (then andes selftest)")

    andes.config_logger(stream_level=30)  # WARNING
    ss = andes.load(str(case), setup=False, no_output=True)
    if freq_hz is not None:
        ss.config.freq = float(freq_hz)   # base frequency for the swing equation (dynamics)
    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        sys.exit("ERROR: ANDES power flow did NOT converge.")
    return ss


def report_and_check(ss, freq_hz: float | None) -> bool:
    """Print the case scale and run basic sanity checks; return pass/fail."""
    import numpy as np

    v = np.asarray(ss.Bus.v.v, float)
    n_bus = len(v)
    pg = (float(np.sum(ss.PV.p.v)) + float(np.sum(ss.Slack.p.v))) * C.SBASE_MVA
    pl = float(np.sum(ss.PQ.p0.v)) * C.SBASE_MVA
    n_gen = ss.PV.n + ss.Slack.n
    freq = ss.config.freq

    print(f"  case            : {n_bus} buses, {n_gen} generators, {ss.PQ.n} loads")
    print(f"  base frequency  : {freq:g} Hz")
    print(f"  Sigma Pg        : {pg:.1f} MW")
    print(f"  Sigma Pload     : {pl:.1f} MW   (losses ~= {pg - pl:.1f} MW)")
    print(f"  voltage range   : {v.min():.4f} .. {v.max():.4f} pu")

    # write a bus snapshot for downstream reference
    out = C.RESULTS / "base_pf.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bus", "V_pu", "angle_deg"])
        a_deg = np.degrees(np.asarray(ss.Bus.a.v, float))
        for i, b in enumerate(ss.Bus.idx.v):
            w.writerow([int(b), f"{v[i]:.5f}", f"{a_deg[i]:.4f}"])
    print(f"  base PF snapshot : {out}")

    # sanity checks
    checks = {
        "PF converged": bool(ss.PFlow.converged),
        "39 buses": n_bus == 39,
        "10 generators": n_gen == 10,
        # generator buses in the standard 39-bus are scheduled up to ~1.07 pu; the
        # case's own bus Vmax limit is 1.10, so that is the meaningful band.
        "voltages within 0.94-1.10 pu": bool(v.min() > 0.94 and v.max() < 1.10),
        "losses 0-5% of load": bool(0.0 < (pg - pl) < 0.05 * pl),
    }
    print("  sanity checks:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def save_base_case(ss, path: Path) -> None:
    """Freeze the loaded base case to xlsx so downstream cases derive from one artifact."""
    import andes

    path.parent.mkdir(exist_ok=True)
    andes.io.xlsx.write(ss, str(path), overwrite=True)
    print(f"  saved base case : {path}")
    print("  (note: base frequency is applied at load time via config.FREQ_HZ, not stored)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Load ANDES bundled ieee39, solve PF, sanity-check.")
    p.add_argument("--case", default=None,
                   help="bundled case name (default: auto-detect ieee39_full / ieee39)")
    p.add_argument("--freq", type=float, default=None,
                   help="override system base frequency in Hz (e.g. 50 for the study case); "
                        "default leaves the case's native value")
    p.add_argument("--list-cases", action="store_true",
                   help="print candidate bundled case names and exit")
    p.add_argument("--save", nargs="?", type=Path, const=C.BASE_CASE, default=None,
                   help=f"freeze the loaded base case to xlsx (default {C.BASE_CASE.name})")
    args = p.parse_args(argv)

    if args.list_cases:
        print("Candidate bundled cases (resolve via andes.get_case):")
        for c in CANDIDATE_CASES:
            print("   ", c)
        return 0

    case = resolve_case(args.case)
    print(f"Loading ANDES bundled case: {case.name}"
          f"{f' (freq -> {args.freq:g} Hz)' if args.freq else ''} ...")
    ss = build_system(case, args.freq)
    ok = report_and_check(ss, args.freq)
    if args.save is not None:
        if ok:
            save_base_case(ss, args.save)
        else:
            print("  NOT saving — sanity checks failed; fix before freezing the base case.")
    print("  RESULT          :", "PASS — base case ready" if ok else "FAIL — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# ----------------------------------------------------------------------------------
# Downstream of this base case (now implemented elsewhere):
#   - `dfc pvbess` (add_pv_bess.py): install the 9 PV-BESS plants at config.PV_BUSES,
#     scaled to config.PV_NAMEPLATE_MW, with config.SLACK_BUS as the interconnector proxy.
#   - `dfc qsts` (study_a_qsts.py): time-varying merit-order dispatch over the day.
#   - Study B dynamics (REGCA1/REECA1/REPCA1 + frequency) is WIP in attach_dynamics.py.
# ----------------------------------------------------------------------------------
