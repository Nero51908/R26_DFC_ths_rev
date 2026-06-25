# CLAUDE.md — dev_ANDES_experiment (grid-level DFC, ANDES Study A)

Context for an agent running the **ANDES-dependent** work of the new thesis chapter
(grid-level / transmission-network value of Dispatchable Firm Capacity from PV–BESS).
This half **requires `import andes`**, so it runs where ANDES is installed — the local
`.venv` or Bunya — NOT in the Cowork sandbox. Git repo `Rsrch_Thesis_rev` is shared state
with the ANDES-free analysis done in Cowork.

## Division of labour
- **Here (local venv / Bunya):** anything importing `andes` — the QSTS network run
  (`study_a_qsts`), `netobs`/power-flow debugging, the Study B dynamics
  (`attach_dynamics.py`, WIP), and any local batch mode.
- **Cowork:** the ANDES-free layer — `dfc availability` / `by-plant` / `forecast` /
  `injection`, Plotly views, Obsidian notes (`3.1_revision/thesis_notes/`), SLURM script prep.

## One entrypoint: the `dfc` CLI
`./dfc` lists everything; `./dfc <cmd> -h` for a command's own flags. ANDES commands:
`dfc case` (build base), `dfc pvbess` (high-pen case), `dfc qsts` (Study-A QSTS),
`dfc transmission` (network metrics from the QSTS `.npz`). Verify env first:
`python -c "import andes; print(andes.__version__)"`.

## Study A network run — current readiness (verified 2026-06-25)
- Case `cases/ieee39_pvbess.xlsx` present (`config.PVBESS_CASE`).
- Trajectories present for all 3 plants × {rule, agent, mpc}; the day sets ALIGN — **110
  common days** in period p2, runnable for a1–a4 (0 days missing for any scenario).
- `study_a_qsts.py` runs ONE `(scenario, day)` per invocation and writes
  `results/qsts_net_<scen>_<date>.npz` (+ `qsts_<scen>_<date>.csv`). a1 = raw PV (no
  trajectory); a2/a3/a4 read `results/trajectories/<rule|agent|mpc>_*.csv`.
- ⚠️ Mixed agent vintage: BANN1 & RUGBYR1 agents are Ch.5 settings, **EDENVSF1 is still
  Ch.4**. Fine for a prototype; refresh once new EDENVSF1 (and RUGBYR1) agents arrive.

### Run locally (serial; per-run case reload dominates, ~30–60 min for 440 runs)
```bash
source .venv/bin/activate
DAYS=$(python -c "import sys;sys.path.insert(0,'src');import pv_data as P;\
print(' '.join(str(d.date()) for d in P.common_complete_days(P.load_all('p2'))))")
for S in a1 a2 a3 a4; do for D in $DAYS; do
  python src/study_a_qsts.py --scenario $S --date $D --period p2; done; done
./dfc transmission
```
(Optional speed-up worth building & TESTING here: a batch mode that loads the ANDES case
ONCE and loops days×scenarios — much faster than 440 reloads. Keep it behind a flag.)

### Run on Bunya (recommended; parallel array)
Get code + trajectories onto scratch via **sftp** (Bunya recommends it; no data-mover node),
then:
```bash
# project already uploaded to scratch (sftp); build the venv once:
cd /scratch/user/neroliu/R26/R26_DFC_ths_rev/dev_ANDES_experiment
python -m venv .venv && source .venv/bin/activate && pip install andes numpy pandas
sbatch --array=0-439%10 slurm/qsts_array.slurm      # full run: 4 scenarios × 110 days = 440 tasks, max 10 at once
squeue --me
```
QSTS needs only `andes numpy pandas` (cvxpy/osqp are NOT needed — MPC trajectories already
exist). Account `a_yang_du`, partition `general`, qos `normal` (see slurm/README.md).

### Fetch results — SFTP (Bunya has no data-mover node)
Bunya recommends **sftp**; reuse the established pattern (feed commands on STDIN, not `sftp -b`,
so DUO MFA still prompts; login node; MFA once). Template: `Ch5_DFC_Code/slurm/fetch_results_sftp.sh`.
ANDES twin:
```bash
BUNYA_USER=neroliu ./slurm/fetch_qsts_sftp.sh       # pulls results/qsts_net_*.npz + qsts_*.csv
./dfc transmission                                  # build the network metrics table
```

## Guardrails
- `results/`, `.venv/`, `cases/*_out.*`, `*.npz` are git-ignored (regenerable). Don't commit them.
- Keep the venv OUT of OneDrive; this repo (Rsrch_Thesis_rev) is the dev tree, separate from
  the OneDrive thesis folder. Never commit secrets (WANDB keys etc.).
- Keep the `dfc` CLI working and `python src/run_tests.py` green after edits.

Related notes (OneDrive Obsidian vault `3.1_revision/thesis_notes/`):
"Study A — injection firmness and the dispatchability framing",
"DFC availability metrics — firm@R and dependability".
