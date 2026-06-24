# CLAUDE.md — Ch5_DFC_Code (DRL DFC agents)

Context for an agent picking up the **training/selection** phase of the thesis chapter on
grid-level value of Dispatchable Firm Capacity (DFC). Runs in the user's **local venv**
(torch/gymnasium/stable-baselines3 present) — unlike the Cowork sandbox, training executes
here. Git repo `Rsrch_Thesis_rev` is the shared state with the analysis/ANDES side.

## Goal of this phase
Train PPO DFC agents under the **revised firm-capacity reward** and select the best per
plant and per BESS size. The reward redesign targets the known failure of the old agents:
deep commitment **shortfalls** (worst on BANN1, whose POE50 forecast is biased high/noisy).

## What the reward now does (helper_fns.dfc_reward — single source of truth)
`reward = met·(1−actual_c) − k_short·shortfall² − k_surplus·surplus`, where
`shortfall=max(Pdfc−Pnet,0)`, `surplus=max(Pnet−Pdfc,0)`, `met = shortfall ≤ atol`.
Asymmetric (only under-delivery punished hard), convex on shortfall (tail-risk pressure),
absolute tolerance. Tunables in `config.reward_params` (atol=0.02, k_short=15, k_surplus=0.5).
Both envs call this one function — do not re-inline a per-env reward.

## Run
```bash
export DFC_DATA_ROOT="$PWD/../data"     # data lives in dev_ANDES_experiment/data (or ../pv_data)
python tests/test_reward.py             # numpy-only, must stay green (11 checks)
# one agent (≈10 min): train on period-1 *_old, evaluate on period-2 *_new
python application.py --bcap 0.30 --train --datat BANN1_old --evaluate --datae BANN1_new --visualize
```
Env is the **curtailment** variant (`config.sb3_config["env_id"] = .../UtilityScalePVBESS-v0`)
— the agent sets both `P_dfc` and curtailment `c`. Outputs: model under `models/<run_id>/`,
eval trajectory `evaluation/<dataset>/<run_id>_<dataset>.csv`, journal row in
`evaluation_journal_*.csv`. HPC: `slurm/train_agents.slurm` (array over plant×seed; see header).

## Experiment design: bcap × plant × seed
- Train on period 1 (`*_old`), evaluate on period 2 (`*_new`). Period 2 is NOT further split
  (user decision) — it is the single unseen evaluation set.
- Sweep `--bcap` over the Study-C grid (e.g. 0.15, 0.30, 0.50, 1.00); 0.30 is the anchor.
- Train **multiple seeds** per (plant, bcap) — PPO is high-variance; never trust one run.
- **Seeds are recorded.** `application.py --seed N` seeds PPO/env and appends a row to
  `models/run_manifest.csv` (run_id, dataset, bcap, seed, env_id, total_steps, policy, ts).
  The SLURM array sets `--seed = SLURM_ARRAY_TASK_ID` (unique per agent). If `--seed` is
  omitted a random seed is drawn AND recorded, so every run is reproducible.
- **Selection without splitting period 2**: to avoid tuning on the test set, prefer selecting
  each plant's best seed by its PERIOD-1 performance, then report on period 2; OR select on
  period 2 but report the across-seed distribution (median/IQR), not just the best. Decide per
  the thesis methodology — the selection script (below) supports both.
- Selection criterion = the DFC claim metrics (shortfall energy ↓, firm-MW@R ↑, dependability ↑),
  not training reward or RMSE.

## Agent selection (script to add when candidates exist — TODO)
A `select_agents.py` will: read `models/run_manifest.csv` + each run's eval CSV
(`evaluation/<dataset>/<run_id>_<dataset>.csv`), compute the DFC metrics per (plant, bcap)
using the same definitions as `../src/availability_by_plant.py` (firm-MW@R, dependability,
shortfall energy, breach depth — per-unit, single plant), rank seeds, pick the winner per
(plant, bcap), write `models/selection.csv`, and emit the `export_trajectory` command(s) to
promote the chosen agents to `../results/trajectories/agent_<PLANT>.csv`. Keep numpy/pandas
only (no torch) so it runs anywhere.

## Guardrails
- **Never commit a WANDB API key.** Use `wandb login` / `WANDB_MODE=offline`; the key stays
  in the environment only (the SLURM script deliberately does not contain it).
- `data/`, `models/`, `evaluation/`, `wandb/`, `logs/` are git-ignored (large/regenerable).
- Keep `tests/test_reward.py` green after any reward edit; keep helper_fns importable without
  torch/plotly (lazy imports) so analysis/tests run on a minimal install.
- No hardcoded magic numbers: tolerances/paths live in `config.py` (`reward_params`,
  `perfect_plan_tol_pu`, `DATA_ROOT`).

## Handoff back to the analysis side (Cowork / ANDES)
Per plant, pick the selected run id, then convert its eval CSV to the ANDES canonical schema:
```bash
python ../src/export_trajectory.py --in evaluation/<EVAL>/<run_id>_<EVAL>.csv \
                                   --out ../results/trajectories/agent_<PLANT>.csv
```
Judge whether the new reward beat MPC with `../src/availability_by_plant.py --plant <PLANT>`
(firm-MW bootstrap CI, paired daily dependability/shortfall, breach depth).
