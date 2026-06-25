# Ch5_DFC_Code — DRL DFC agents (revised reward)

Working copy of the Chapter-5 DRL code (env package `dfc_gymnasium`, training driver
`application.py`), revised for the new chapter. Canonical env history lives at
`github.com/Nero51908/dfc_gymnasium`; here it is flattened into the thesis dev repo so the
whole tree is version-controlled and portable.

## What changed (vs the original Ch.5 code)

**Reward redesign — `helper_fns.dfc_reward()` (single source of truth).** The old
`reward fn n3` penalised the tracking error `|Pnet − Pdfc|` *symmetrically* and *linearly*,
so over- and under-delivery cost the same and a rare deep shortfall was cheaply amortised
by many near-hits — the cause of the agent's deep commitment breaches (worst on BANN1).
The new reward treats `P_dfc` as a firm floor:

```
shortfall = max(Pdfc − Pnet, 0)      # under-delivery: the only grid-harmful miss
surplus   = max(Pnet − Pdfc, 0)      # over-delivery: benign
met       = shortfall <= atol        # ABSOLUTE tolerance (grid-consistent), not relative
reward    = met·(1 − actual_c) − k_short·shortfall² − k_surplus·surplus
```

- **asymmetric**: only shortfall is punished hard; surplus is a light nudge;
- **convex** (`shortfall²`): deep breaches hurt super-linearly → tail-risk pressure;
- **absolute `atol`**: replaces the old `rtol=0.05` band that widened with commitment and
  quietly rewarded over-commitment.

Tunables in `config.reward_params` (`atol=0.02`, `k_short=15`, `k_surplus=0.5`) — sweep
these first if a plant's shortfall energy stays high.

**Selectable reward for ablation (`DFC_REWARD_FN`).** The old symmetric reward is now ported
verbatim as `helper_fns.dfc_reward_n3()`, so the redesign can be A/B-tested against it on identical
data and seeds. `helper_fns.env_step_reward()` dispatches on the `DFC_REWARD_FN` env var — `firm`
(default; the redesign above) or `n3` (the old baseline) — and the choice is recorded in
`models/run_manifest.csv`, so every agent is attributable. Both envs call `env_step_reward`, so the
A/B switch is a one-line env var, not a code change.

**Other changes:** `config.sb3_config["env_id"]` now selects the **curtailment** env
(`UtilityScalePVBESS-v0`; agent sets both `P_dfc` and curtailment `c`, the headroom lever);
both envs call the shared `env_step_reward` (no more copy-pasted per-env reward that could drift);
`data_register` gains `BANN1_*`/`EDENVSF1_*`; `helper_fns` imports torch/plotly lazily so
the reward/losses import on a minimal install; `perfect_plan_rate` tolerance documented. Two
numpy/pandas-only reducers — `analyze_seeds.py` and `merge_journals.py` — were added to turn
multi-seed SLURM-array output into a paired firm-vs-n3 verdict (see *Evaluate across seeds*).

## Test

```bash
python tests/test_reward.py        # numpy-only; asserts the reward properties (16 checks: firm + n3)
```

## Train

Local (≈10 min/agent on an M1):
```bash
pip install "stable-baselines3>=2.0" gymnasium torch wandb pandas numpy
export WANDB_MODE=offline                         # or `wandb login`
python application.py --bcap 0.30 --train --datat BANN1_old --evaluate --datae BANN1_new --visualize
```

HPC (SLURM array), a pool of candidate agents per plant for selection:
```bash
sbatch --array=0-23%12 slurm/train_agents.slurm   # 8 agents x 3 plants; see the script header
```

Reward A/B ablation on one plant (firm vs n3, **paired** seeds):
```bash
# tasks 0..N_SEEDS-1 → firm, N_SEEDS..2N-1 → n3; seed = task % N_SEEDS (paired across rewards).
sbatch --export=ALL,N_SEEDS=10 --array=0-19%6 slurm/train_reward_ablation.slurm
```
Each array task seeds with `--seed = SLURM_ARRAY_TASK_ID % N_SEEDS` and writes per-task output
**shards** (`DFC_OUTPUT_TAG=<jobid>_<task>`) so concurrent tasks never race on the shared manifest /
journal. The HPC deploy + fetch scripts and a step-by-step runbook live in `slurm/` (kept locally,
gitignored — they carry cluster-specific account/paths).

**Data prep:** `application.py` reads `<DATA_ROOT>/<PLANT>/<period>/`, where `DATA_ROOT`
defaults to `data/` but can point anywhere via the `DFC_DATA_ROOT` env var — so you need not
duplicate the CSVs. They are the same NEM format as the ANDES `pv_data/`
(`DATETIME, FORECAST_POE50, SCADAVALUE`); the layout is:
```
<DATA_ROOT>/<PLANT>/202309_202408/   # *_old  -> training (period 1)
<DATA_ROOT>/<PLANT>/202408_202505/   # *_new  -> evaluation (period 2, unseen)
```
e.g. reuse the copy at `dev_ANDES_experiment/data/` or the canonical `pv_data/`:
```bash
export DFC_DATA_ROOT="$PWD/../data"      # or .../dev_ANDES_experiment/pv_data
```
Data dirs are git-ignored (large). Tolerances/paths are configurable in `config.py`
(`DATA_ROOT`, `perfect_plan_tol_pu`, `reward_params`) — no magic numbers in the code.

## Evaluate across seeds

PPO is high-variance — never trust one run. Train several seeds per (plant, bcap) and compare the
*distribution*. After pulling the per-task outputs back, two numpy/pandas-only reducers (no torch,
so they run anywhere) turn the shards into a verdict:

```bash
# manifest shards + each run's eval CSV -> per-reward comparison, with a PAIRED firm-vs-n3 by-seed
# contrast and a two-sided Wilcoxon signed-rank p-value (scipy optional; add --ttest for paired-t):
python analyze_seeds.py --merge-shards --dataset BANN1_new --bcap 0.30

# per-task evaluation-journal shards -> evaluation_journal_<dataset>_model.csv (de-duped by run_id,
# tagged with reward_fn/seed). --latest-per-seed also writes a 1-run-per-seed *_latest.csv view:
python merge_journals.py
```

`analyze_seeds.py` collapses to the most recent run per (reward_fn, seed), then reports median [IQR]
per reward and — for each deep-tail metric (shortfall energy, deep breaches, max breach, p99,
dependability, firm@95, perfect-rate) — how many seeds firm wins, the median gap, and the Wilcoxon p.
A gap that holds across seeds is real; one that flips seed-to-seed is PPO noise (n≥6 paired seeds are
needed to reach p<0.05). `merge_journals.py` is the journal twin — idempotent and self-healing. On
BANN1 (bcap 0.30) the redesign significantly cuts deep-tail shortfalls vs n3, confirmed across 10
paired seeds (firm wins 10/10 on every tail metric, Wilcoxon p<0.01).

## Selection → Study A

Each run writes `evaluation/<dataset>/<run_id>_<dataset>.csv`. Pick the best run id per plant
(lowest shortfall / highest firm MW — use `../src/availability_by_plant.py` on the candidates),
then convert to the ANDES canonical trajectory:
```bash
python ../src/export_trajectory.py --in evaluation/<EVAL>/<run_id>_<EVAL>.csv \
                                   --out ../results/trajectories/agent_<PLANT>.csv
```

> Note: `dfc_gymnasium` was flattened from its own git repo into this tree (all its commits
> were already on `origin/dev`). To resync with the standalone package, re-clone it from GitHub.
