# dev_ANDES_experiment

Software for the new thesis chapter: **grid-level value of DFC on a native 60 Hz IEEE
39-bus system at ~44% PV penetration**. Primary simulation tool: **ANDES** (open-source,
Python). One representative case is validated against **PSS/E** for credibility.

## Quick start — the `dfc` CLI

Everything runs through one entrypoint (`./dfc <command>`, or `python src/dfc_cli.py`):

```bash
./dfc                                        # list all commands, grouped
./dfc <command> -h                           # a command's own options
./dfc data                                   # PV data summary / common days
./dfc qsts --scenario a3 --date 2025-01-29   # one Study-A QSTS day
./dfc mpc --all-days                         # full-period MPC (A4) trajectories
./dfc availability --all-days                # fleet firm-capacity-at-reliability
./dfc by-plant --plant BANN1                 # per-plant agent-vs-MPC head-to-head
./dfc forecast --plot                        # POE50 forecast accuracy + Plotly HTML
./dfc explore --prefix agent --plant BANN1   # interactive trajectory viewer
```

The dispatcher just runs the matching `src/<module>.py` with your flags, so every module
also still works standalone and stays importable. HPC batch jobs live in `slurm/`.

This is a standalone dev workspace at `~/dev_py_playground/Rsrch_Thesis_rev`, kept
**outside the OneDrive thesis tree** to avoid sync churn from the venv, caches, and run
outputs. Back it up with **git + GitHub** (no cloud-sync backup here).

## Related locations (in the separately-mounted OneDrive thesis tree `3.1_revision/`)

- Master design doc & decisions: `dev_PSSE_experiment/experiment_design.md`
- Ch. 5 DRL codebase (trained agents, data, DFC trajectories): `dev_DRL_experiment/`
- Original PSS/E base network: `ieee39-bus_by_Claude/ieee39.raw`, `ieee39.dyr`

The base `.raw`/`.dyr` have been **copied locally into `cases/`** so this folder is
self-contained; ANDES imports them directly.

## Layout

```
dev_ANDES_experiment/
├── src/                     # source (tracked) — see "Source map" below
├── cases/                   # base + generated ANDES/PSS-E case files
│   ├── ieee39.raw|.dyr      # PSS/E originals (external reference only)
│   ├── ieee39_base.xlsx     # frozen, validated ANDES base case (canonical start)
│   └── ieee39_pvbess.xlsx   # high-penetration Config-B case
├── pv_data/                 # RUGBYR1 / BANN1 / EDENVSF1 per-unit profiles (periods 1 & 2)
├── results/                 # run outputs: CSV, figures, trajectories (GIT-IGNORED)
├── slurm/                   # UQ Bunya batch jobs (MPC, QSTS array) + HPC README
├── Ch5_DFC_Code/            # DRL agents (dfc_gymnasium env + PPO) — revised firm-capacity reward
├── experiment_tracker.csv   # experiment journal (tracked; same schema as DRL chapters)
├── dfc                      # CLI launcher (./dfc <command>)
└── .gitignore
```

## Source map (`src/`)

Data → case → engine → scenarios → reporting. Each module has one job. The `dfc` command
in front of each shows how to invoke it through the dispatcher.

| Module | `dfc` cmd | Role |
|---|---|---|
| `dfc_cli.py` | — | **single entrypoint**: grouped sub-commands dispatched to the modules below |
| `config.py` | — | single source of truth for every decision (native 60 Hz, ~44% pen, BESS anchor, dispatch, paths) |
| `pv_data.py` | `data` | load the 3 plant profiles, find common days, classify clear/mixed/overcast, build injection matrices |
| `build_case.py` | `case` | load ANDES `ieee39_full`, solve PF, sanity-check, `--save` the frozen base case |
| `add_pv_bess.py` | `pvbess` | install the 9 PV-BESS plants + derate the fleet → `ieee39_pvbess.xlsx` |
| `dfc_plant.py` | — | **shared** faithful Ch.5 DeepComp battery+plant model (A2/A3/A4 all use it) |
| `make_firm_trajectory.py` | `rule` | **A2** rule-based firming baseline → `results/trajectories/rule_*.csv` |
| `benchmark_mpc.py` | `mpc` | **A4** receding-horizon MPC firming → `mpc_*.csv` (needs cvxpy/osqp) |
| `export_trajectory.py` | `export-agent` | **A3** convert a trained-agent eval CSV → `agent_*.csv` |
| `trajectory.py` | — | adapt any canonical trajectory CSV → QSTS injection matrix |
| `study_a_qsts.py` | `qsts` | the QSTS engine: AEMO-style merit-order dispatch + power-flow loop (scenarios a1–a4); logs `qsts_net_*.npz` (line loadings, bus V) |
| `attach_dynamics.py` | `dynamics` | **Study B (WIP)** attach dynamic models for time-domain frequency sim |
| `netobs.py` | — | transmission observables from a solved PF — per-line MVA/%loading (pi-model + tap), bus voltages |
| `availability.py` | `availability` | **DFC-as-availability**: fleet firm-capacity-at-reliability (capacity credit) + commitment dependability per method |
| `availability_by_plant.py` | `by-plant` | per-plant version + head-to-head significance (firm-MW day-bootstrap, paired daily dependability/shortfall, breach depth) |
| `forecast_quality.py` | `forecast` | per-plant POE50 forecast accuracy (RMSE/MAE/bias/skill); drives A2 P_dfc quality |
| `transmission_metrics.py` | `transmission` | transmission value: congestion frequency, lines stressed, flow variability, voltage band, from `qsts_net_*.npz` |
| `tracker.py` | `tracker` | append/rebuild `experiment_tracker.csv` from the QSTS run outputs |
| `viz.py` | `viz` | static matplotlib figures for the thesis (dispatch stack, penetration, voltage band, balancing bar, POI) |
| `explore.py` | `explore` | **interactive Plotly** HTML viewer for inspection — zoom/pan/hover, written next to each CSV (needs plotly) |
| `run_tests.py` | `test` | pipeline self-test → `results/test_report.txt` (sections A–H; E/F need cvxpy/ANDES) |

Scenarios: **a1** raw PV · **a2** rule firming · **a3** DRL agent · **a4** MPC — all
share `dfc_plant`'s battery and the canonical trajectory schema
(`idx, datetime, P_forecast, P_actual, P_net, P_dfc, P_bess_cmd, P_curtail`). The QSTS
injects **`P_net`** = actual PV+BESS delivery; **`P_dfc`** is the dispatch target/forecast.

## Environment

Create the venv here locally (it's git-ignored and outside OneDrive, so it's safe):

```bash
cd ~/dev_py_playground/Rsrch_Thesis_rev/dev_ANDES_experiment
python3 -m venv .venv
source .venv/bin/activate
pip install andes numpy pandas cvxpy osqp matplotlib plotly
andes selftest                         # verify install (macOS: numpy/scipy/kvxopt wheels OK)
```

## Status

Base case = **ANDES's bundled, validated `ieee39_full`** (Option B). The PSS/E `.raw`
import was rejected: ANDES's parser mishandled the transformer taps and the network
voltage profile collapsed (gen buses exact, network buses −0.05..−0.11 pu). The PSS/E
`.raw`/`.dyr` in `cases/` are kept only as an optional one-snapshot external reference.

The system runs at its **native 60 Hz** throughout (decided): Study A power flow is
frequency-agnostic, and Study B reports base-independent Δf/RoCoF. A 50 Hz relabel is an
opt-in only for NEM-standard frequency plots (`dfc dynamics --rebase 50`).

Study A (A1–A4) is built and analysed: `dfc case` → `dfc pvbess` → produce trajectories
(`dfc rule` / `dfc mpc` / `dfc export-agent`) → `dfc qsts` → analyse (`dfc availability`,
`dfc by-plant`, `dfc forecast`, `dfc transmission`). Study B (`dfc dynamics`) is WIP.
