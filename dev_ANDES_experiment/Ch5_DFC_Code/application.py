import time
import os
import sys
import csv
import argparse

import pandas as pd
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
from stable_baselines3 import PPO

# local modules
# sys.path will list pv-bess-dfc, where this file is located, as the first dir to search for local modules.
# once dfc_gymnasium/__init__.py is run, the env should be registered and therefore it can be found by env_id.
import dfc_gymnasium
import helper_fns as hlp
from data_buffer import DataBuffer
import config

def record_run(run_id: str, dataset: str, bess_properties: dict, seed: int):
  """Append a reproducibility row to models/run_manifest.csv (one line per trained agent).

  This is the authoritative seed record: the agent-selection step joins evaluation results
  to this manifest by run_id to recover (seed, bcap, dataset, env_id, total_steps, reward_fn).
  The manifest lives under models/ (git-ignored). It is rewritten with a stable schema each
  run (read-modify-write): older rows are backfilled when the schema grows, so firm- and
  n3-reward agents stay distinguishable by the reward_fn column."""
  # Per-task shard when DFC_OUTPUT_TAG is set (e.g. a SLURM array): each task writes its OWN
  # manifest file so concurrent tasks never clobber a shared one. Merge shards on retrieval
  # (analyze_seeds.py --merge-shards). Unset tag -> the canonical single manifest (local runs).
  tag = os.environ.get("DFC_OUTPUT_TAG", "")
  manifest = os.path.join(config.dir_names["training_output"],
                          f"run_manifest{('_' + tag) if tag else ''}.csv")
  os.makedirs(config.dir_names["training_output"], exist_ok=True)
  row = {
    "run_id": run_id, "dataset": dataset,
    "bcap": bess_properties["energy_capacity_puh"], "seed": seed,
    "env_id": config.sb3_config["env_id"], "total_steps": config.sb3_config["total_steps"],
    "policy": config.sb3_config["policy_type"],
    "reward_fn": getattr(config, "reward_fn", "firm"),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  }
  fieldnames = list(row)
  rows = []
  if os.path.isfile(manifest):
    with open(manifest, newline="") as f:
      for r in csv.DictReader(f):
        r.setdefault("reward_fn", "firm")   # backfill: every run predating this column used the firm reward
        rows.append(r)
  rows.append(row)
  with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, "") for k in fieldnames})
  print(f"record_run(): {run_id} seed={seed} -> {manifest}")


# train model for dfc using PPO
def dfc_ppo(dataset: str, bess_properties: dict, seed: int = None):
  start_time = time.time()
  # Resolve and RECORD the seed: if none given, draw one and keep it so the run is
  # reproducible. PPO(seed=...) seeds torch/numpy and the env's action/obs sampling.
  if seed is None:
    seed = int.from_bytes(os.urandom(4), "little")
  seed = int(seed)

  data_path = config.data_register[dataset]['path']
  data_feature = config.data_register[dataset]['feature']
  db = DataBuffer()
  db.read_data_from(data_path, *data_feature['file_structure'])
  training_set = db.prepare_data()

  # Create the environment
  env = gym.make(config.sb3_config['env_id'],
    forecast_scada_timeseries = training_set,
    bess_properties = bess_properties,
    sec_per_step = data_feature['temporal_resolution'],
    soc_levels = config.number_of_soc_levels_for_training,
    render_mode = None,
    )

  info_dict = {
    "dataset": dataset,
    "BESS_capacity" : bess_properties["energy_capacity_puh"],
    "BESS_charg_lim": bess_properties["power_boundary_Erate"].limsup,
    "BESS_disch_lim": bess_properties["power_boundary_Erate"].liminf,
    "seed": seed,                       # recorded in the WandB run config too
  }

  info_dict.update(config.sb3_config)

  # Track with WandB
  run = wandb.init(
    project='R2024_2_DFC',
    config=info_dict,
    sync_tensorboard=True,  # upload sb3's training metrics
    tags=[f"reward_{config.reward_fn}"],
  )

  # use WandB run id as the unique identifier of each model trained
  model_save_path = os.path.join(config.dir_names["training_output"], run.id)
  tensorboard_log_path = os.path.join("logs", run.id)
  record_run(run.id, dataset, bess_properties, seed)   # authoritative seed <-> run_id record

  # Train with SB3 PPO (seeded for reproducibility)
  policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
  model = PPO(config.sb3_config["policy_type"], env, verbose=1, seed=seed,
              tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs)
  model.learn(
    total_timesteps=config.sb3_config["total_steps"],
    callback=WandbCallback(
      model_save_path=model_save_path,
      verbose=1,
    ),
  )

  run.finish()
  end_time = time.time()
  simulation_time = end_time - start_time
  print(f"train_ppo(): {config.sb3_config['total_steps']} steps finished in {simulation_time: .2f} seconds")
  return run.id

def journalize_evaluation(csv_path: str, dummy: bool, fixed_c, run_id: str, dataset: str, bess_properties: dict):
  # Per-task shard when DFC_OUTPUT_TAG is set (SLURM array) so concurrent evals don't interleave
  # rows in one shared journal; merge shards on retrieval. Unset tag -> the canonical journal.
  tag = os.environ.get("DFC_OUTPUT_TAG", "")
  journal_filename = f"evaluation_journal_{dataset}{'_dummy' if dummy else '_model'}{('_' + tag) if tag else ''}.csv"
  if config.sb3_config['env_id'].endswith("nocurtailment"):
    actor_name = "baseline" if dummy else run_id
  else:
    actor_name = "downscaled" if dummy else run_id
  info_dict   = {
    "dataset": dataset,
    "actor"  : actor_name,
    "fixed_c": fixed_c,
    "BESS_capacity": bess_properties["energy_capacity_puh"],
    "BESS_charg_lim": bess_properties["power_boundary_Erate"].limsup,
    "BESS_disch_lim": bess_properties["power_boundary_Erate"].liminf,
  }
  info_dict.update(analyze_evaluation(csv_path))
  info_df = pd.DataFrame([info_dict])
  print("journalize_evaluation():\n", info_df)
  info_df.to_csv(journal_filename, mode='a', header= not os.path.exists(journal_filename), index=False)

def evaluate_model(run_id: str, dataset: str, bess_properties: dict):
  start_time = time.time()
  # Load model to be evaluated
  model_path = os.path.join(config.dir_names['training_output'], run_id, 'model')
  model = PPO.load(model_path)

  # prepare data for evaluation
  data_path = config.data_register[dataset]['path']
  data_feature = config.data_register[dataset]['feature']
  db = DataBuffer()
  db.read_data_from(data_path, *data_feature["file_structure"])
  evaluation_set = db.prepare_data()
  # prepare evaluation output directory
  csv_path = os.path.join(config.dir_names['evaluation_output'], f"{dataset}", f"{run_id}_{dataset}.csv")
  if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path))
    
  # create environment for evaluation
  env = gym.make(config.sb3_config["env_id"],
    forecast_scada_timeseries = evaluation_set,
    bess_properties = bess_properties,
    sec_per_step = data_feature["temporal_resolution"],
    soc_levels = 1,
    render_mode = "evaluation",
    csv_path = csv_path,
    )

  # run simulation for evaluation
  print(f"Simulation is using model {run_id}")
  obs, info = env.reset()
  for i in range(info["max_steps"]): 
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, _, info = env.step(action)
    if terminated and not info["last_cluster"]:
      obs, info = env.reset()

  # close the environment 
  env.close()
  end_time = time.time()
  simulation_time = end_time - start_time
  print(f"evaluate_model(): {info['max_steps']} steps finished in {simulation_time: .2f} seconds")

  journalize_evaluation(csv_path, False, None, run_id, dataset, bess_properties)
 
  return csv_path


def evaluate_dummy_c(c, dataset: str, bess_properties: dict):
  if config.sb3_config['env_id'].endswith("nocurtailment"):
    actor_name = "baseline" 
  else:
    actor_name = "downscaled" 
  run_id = f"{actor_name}_b{bess_properties['energy_capacity_puh']}c{c:.2f}"
  start_time = time.time()
  # prepare data for evaluation
  data_path = config.data_register[dataset]['path']
  data_feature = config.data_register[dataset]['feature']
  db = DataBuffer()
  db.read_data_from(data_path, *data_feature["file_structure"])
  evaluation_set = db.prepare_data()
  # evaluation output directory
  csv_path = os.path.join(config.dir_names['evaluation_output'], f"{run_id}_{dataset}.csv")

  # create environment for evaluation
  env = gym.make(config.sb3_config["env_id"],
    forecast_scada_timeseries = evaluation_set,
    bess_properties = bess_properties,
    sec_per_step = data_feature["temporal_resolution"],
    soc_levels = 1,
    render_mode = "evaluation",
    csv_path = csv_path,
    )

  # run simulation for evaluation
  print(f"Simulation is using curtailment ratio {c=}.")
  obs, info = env.reset()
  for i in range(info["max_steps"]): 
    action = np.append(obs["pv_forecast"] * (1-c), [c])
    obs, _, terminated, _, info = env.step(action)
    if terminated and not info["last_cluster"]:
      obs, info = env.reset()

  # close the environment 
  env.close()
  end_time = time.time()
  simulation_time = end_time - start_time
  print(f"evaluate_dummy_c(): {info['max_steps']} steps finished in {simulation_time: .2f} seconds")
  
  journalize_evaluation(csv_path, True, c, run_id, dataset, bess_properties)
  
  return csv_path

def analyze_evaluation(filepath: str, spotlight: bool = False):
  df = pd.read_csv(filepath)
  if spotlight:
    df = df[df['Pnet'] != df['Pdfc']]
  Pm   = df['env_state_pv_potential'].to_numpy()
  Pnet = df['Pnet'].to_numpy()
  Pdfc = df['Pdfc'].to_numpy()
  actual_c = df['actual_cr'].to_numpy()

  mean_curtailment_ratio = hlp.mean_curtailment_ratio(actual_c)
  mean_curtailment = hlp.mean_curtailment(Pm, actual_c)
  perfect_rate = hlp.perfect_plan_rate(Pnet, Pdfc, atol=config.perfect_plan_tol_pu)
  rmse = hlp.rmse_loss_np(Pnet, Pdfc)
  mae = hlp.mae_loss_np(Pnet, Pdfc)
  try:
      mape = hlp.mape_loss_np(Pnet, Pdfc)
  except:
      mape = None
          
  metric_dict = {
    "filename": os.path.basename(filepath),
    "mean_curtailment_ratio": mean_curtailment_ratio,
    "mean_curtailment": mean_curtailment,
    "perfect_rate": perfect_rate,
    "rmse": rmse,
    "mae": mae,
#    "mape": mape, # warning: adding this column will corrupt existing evaluation_journal.csv as it did not journalize mape.
    "spotlight": spotlight,
  }
  return metric_dict

def analyze(dirname: str, spotlight: bool = False):
  results = []
  for dirpath, _, filenames in os.walk(dirname):
    for filename in filenames:
      if filename.startswith('.'):
        continue 
      filepath = os.path.join(dirpath, filename)
      metric_dict = analyze_evaluation(filepath, spotlight)
      results.append(metric_dict)
  results_df = pd.DataFrame(results).sort_values(by='mean_curtailment', ascending=True)
  results_df.to_csv(dirname+f"_spotlight{spotlight}_report.csv", index=False)
  print(results_df)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="CLI of pv-bess-dfc")
  parser.add_argument("--bcap",     type=float, default=None, help="BESS capacity in pu relative to PV peak power")
  parser.add_argument("--dummy",    action="store_true", help="Run evaluate_dummy_c() instead of training")
  parser.add_argument("--c",        type=float, default=0.1, help="(0,1) float that means curtailment ratio")
  parser.add_argument("--train",    action="store_true", help="Run training script")
  parser.add_argument("--evaluate", action="store_true", help="Run evaluation script")
  parser.add_argument("--model",    type=str, help="Specify wandb run id of trained model")
  parser.add_argument("--visualize",action="store_true", help="Run visualization script")
  parser.add_argument("--show",     action="store_true", help="Use to show Plotly output in browser")
  parser.add_argument("--datetime", action="store_true", help="True to use datetime as x-axis in the visualization")
  parser.add_argument("--csvpath",  type=str, help="Explicitly tell the visualization script where the csv file is")
  parser.add_argument("--analyze",  type=str, help="Run analyze() to show perfect rate, rmse of evaluation csv files")
  parser.add_argument("--spotlight",action="store_true", help="Calculate metrix such as rmse using only Pnet != Pdfc")
  parser.add_argument("--algo",     type=str, help="DRL algorithm name")
  parser.add_argument("--datat",    type=str, help="Data for training")
  parser.add_argument("--datae",    type=str, help="Data for evaluation")
  parser.add_argument("--seed",     type=int, default=None, help="PPO/env seed (recorded in models/run_manifest.csv; random if omitted)")

  args, remaining_argv = parser.parse_known_args()
  
  bess_properties = config.bess_properties
  if args.bcap != None:
    # adjust BESS properties as CLI args specified
    bess_properties["energy_capacity_puh"] = args.bcap
  else:
    pass

  if args.train:
    # train
    run_id = dfc_ppo(args.datat, bess_properties, seed=args.seed)

  if args.evaluate:
    run_id = run_id if args.train else args.model
    # evaluate
    if args.dummy:
      csv_path = evaluate_dummy_c(args.c, args.datae, bess_properties)
    else:
      csv_path = evaluate_model(run_id, args.datae, bess_properties)

  if args.visualize:
    # visualize evaluation
    csv_path     = csv_path if args.evaluate else args.csvpath
    assert os.path.isfile(csv_path), "csv_path must point to a file."
    target_dir   = os.path.dirname(csv_path)
    figure_title = f"{csv_path}"
    hlp.visualize_pnet_pdfc(csv_path, target_dir, figure_title, save=True, show=args.show, show_datetime=args.datetime)

  if args.analyze:
    analyze(args.analyze, spotlight=args.spotlight)

  # Visualize action vs observation
  #   csv_path = os.path.join('evaluation_csv','csv_filename')
  #   hlp.visualize_pdfc_vs_c(csv_path)
  #   hlp.visualize_pdfc_vs_pf(csv_path)
