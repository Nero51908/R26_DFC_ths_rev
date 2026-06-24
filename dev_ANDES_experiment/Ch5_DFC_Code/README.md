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

**Other changes:** `config.sb3_config["env_id"]` now selects the **curtailment** env
(`UtilityScalePVBESS-v0`; agent sets both `P_dfc` and curtailment `c`, the headroom lever);
both envs call the shared `dfc_reward` (no more copy-pasted reward that could drift);
`data_register` gains `BANN1_*`/`EDENVSF1_*`; `helper_fns` imports torch/plotly lazily so
the reward/losses import on a minimal install; `perfect_plan_rate` tolerance documented.

## Test

```bash
python tests/test_reward.py        # numpy-only; asserts the reward properties (11 checks)
```

## Train

Local (≈10 min/agent on an M1):
```bash
pip install "stable-baselines3>=2.0" gymnasium torch wandb pandas numpy
export WANDB_MODE=offline                         # or `wandb login`
python application.py --bcap 0.30 --train --datat BANN1_old --evaluate --datae BANN1_new --visualize
```

HPC (Bunya), a pool of candidate agents per plant for selection:
```bash
sbatch --array=0-23%12 slurm/train_agents.slurm   # 8 agents x 3 plants; see the script header
```

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
