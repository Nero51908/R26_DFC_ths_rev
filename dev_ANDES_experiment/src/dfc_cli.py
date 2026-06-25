#!/usr/bin/env python3
"""
dfc_cli.py — single entrypoint for the grid-level DFC experiment.

Instead of remembering which of ~16 scripts does what, run everything through one command
with grouped sub-commands:

    dfc <command> [options]          # options are passed straight to the underlying module
    dfc                              # list all commands, grouped
    dfc <command> -h                 # the module's own --help

It is a thin dispatcher: each sub-command just runs the matching src/<module>.py as if you
had called it directly (same flags, same behaviour), so every module still works standalone
and stays importable. Adding a module = one line in COMMANDS below.

Examples
    dfc data --list clear                         # PV data summary / common days
    dfc qsts --scenario a3 --date 2025-01-29      # one QSTS day (Study A)
    dfc mpc --all-days                            # full-period MPC trajectories (A4)
    dfc availability --all-days                   # fleet firm-capacity-at-reliability
    dfc by-plant --plant BANN1                    # per-plant agent-vs-MPC head-to-head
    dfc forecast --plot                           # POE50 forecast accuracy + Plotly HTML
    dfc explore --prefix agent --plant BANN1      # interactive trajectory viewer
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent

# command -> (module filename in src/, one-line description)
COMMANDS: dict[str, tuple[str, str]] = {
    # --- run: build cases & produce scenario trajectories -------------------------
    "case":         ("build_case.py",          "load ANDES base case, solve PF, sanity-check (--save base)"),
    "pvbess":       ("add_pv_bess.py",         "install the 9 PV-BESS plants -> high-penetration case"),
    "qsts":         ("study_a_qsts.py",        "Study A QSTS power-flow loop (scenarios a1-a4)"),
    "mpc":          ("benchmark_mpc.py",       "A4 receding-horizon MPC firming trajectories"),
    "rule":         ("make_firm_trajectory.py","A2 rule-based (forecast-tracking) firming"),
    "export-agent": ("export_trajectory.py",   "A3: convert a trained-agent eval CSV -> canonical"),
    "dynamics":     ("attach_dynamics.py",     "Study B: attach dynamics for time-domain sim (WIP)"),
    # --- analyse: score the outputs ----------------------------------------------
    "data":         ("pv_data.py",             "PV data summary: common days, day classes"),
    "availability": ("availability.py",        "fleet DFC firm-capacity-at-reliability + dependability"),
    "by-plant":     ("availability_by_plant.py","per-plant availability + agent-vs-MPC significance"),
    "forecast":     ("forecast_quality.py",    "per-plant POE50 forecast accuracy (RMSE/bias/skill)"),
    "injection":    ("injection_drivers.py",   "Study A: fleet net-injection firmness drivers (a1-a4)"),
    "transmission": ("transmission_metrics.py","network value: congestion, flow variability, voltage"),
    "tracker":      ("tracker.py",             "rebuild experiment_tracker.csv from QSTS outputs"),
    # --- explore & test ----------------------------------------------------------
    "viz":          ("viz.py",                 "static matplotlib thesis figures"),
    "explore":      ("explore.py",             "interactive Plotly HTML trajectory viewer"),
    "test":         ("run_tests.py",           "pipeline self-test (sections A-H)"),
}

GROUPS: list[tuple[str, list[str]]] = [
    ("Run experiments", ["case", "pvbess", "qsts", "mpc", "rule", "export-agent", "dynamics"]),
    ("Analyse output",  ["data", "availability", "by-plant", "forecast", "injection", "transmission", "tracker"]),
    ("Explore & test",  ["viz", "explore", "test"]),
]


def _usage() -> str:
    w = max(len(c) for c in COMMANDS)
    out = [__doc__.strip().split("\n\n")[0], "",
           "usage: dfc <command> [options]    (dfc <command> -h for a command's own help)", ""]
    for title, cmds in GROUPS:
        out.append(f"{title}:")
        for c in cmds:
            out.append(f"  {c:<{w}}  {COMMANDS[c][1]}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"dfc: unknown command '{cmd}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    module = SRC / COMMANDS[cmd][0]
    # run the target module exactly as `python src/<module>.py rest...`
    sys.argv = [f"dfc {cmd}", *rest]
    try:
        runpy.run_path(str(module), run_name="__main__")
        return 0
    except SystemExit as e:                       # modules end with raise SystemExit(main())
        return int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
