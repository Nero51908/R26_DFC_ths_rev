import os
# local Python modules
from helper_fns import StateSpaceLimits

# # WandB tracking
# os.environ['WANDB_MODE'] = 'online'
# os.environ['WANDB_API_KEY']= 'your wandb api key here if using wandb.ai'
# wandb_tags = ["dev"]

# directory names (root is .. relative to main.py)
dir_names = {
  'training_output'   : 'models',
  'evaluation_output' : 'evaluation',
  'visualization'     : 'evaluation',
}

#############################
# Input Data Specifications
#############################
# just a note of supported input data formats readable by the DataBuffer.read_data_from()
supported_data_formats = ['NEM_csv', 'Elia_xls', 'DeepComp_csv']
# interesting_col_* are specifying columns names in the dataset correcponding to datetime, forecast, and measured.
# file_structure of dataFeature_* is a tuple of:
# header_row_number: int, points_per_day: int, column_names_to_read: list[str], data_format: str (one of the supported_data_formats)
# temporal_resolution of dataFeature_* used when creating an environment instance.

# data features specifies the structure for read_data_from() and time per step for the environment
# data from NEM (AEMO, 5 min ahead forecast, 5 min dispatch interval, 288 points per 24 hours)
interesting_col_NEM = ['DATETIME', 'FORECAST_POE50', 'SCADAVALUE']
dataFeature_NEM  = {'file_structure': (0,288,interesting_col_NEM,'NEM_csv'),
                    'temporal_resolution': 5*60., # 5 minutes expressed in seconds
                  }

# data from Elia (day-ahead forecast, 15 min dispatch interval, 96 points per 24 hours)
interesting_col_Elia = ['DateTime', 'Day-Ahead forecast [MW]', 'Corrected Upscaled Measurement [MW]']
dataFeature_Elia_dah = {'file_structure': (3,96,interesting_col_Elia,'Elia_xls'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      } 

# data from Elia (hour-ahead forecast, 15 min dispatch interval, 96 points per 24 hours)
dataFeature_Elia_hah = {'file_structure': (3,96,interesting_col_Elia,'Elia_xls'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      } 

# data from DeepComp is Elia data
interesting_col_DeepComp = ['Time', 'Forecast(MW)', 'Power(MW)']
dataFeature_DeepComp = {'file_structure': (0,96,interesting_col_DeepComp, 'DeepComp_csv'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      }


# Root directory holding the per-plant data folders. Defaults to 'data' (relative to the
# working dir, the original behaviour), but can point anywhere via the DFC_DATA_ROOT env var
# so the data need not be duplicated — e.g. reuse the canonical ANDES copy:
#   export DFC_DATA_ROOT=/abs/path/to/dev_ANDES_experiment/pv_data
DATA_ROOT = os.environ.get("DFC_DATA_ROOT", "data")

# data register (from which directory data files will be found)
data_register = {
  'Elia_2023'      : {'path': os.path.join(DATA_ROOT, 'Elia_2023'),              'feature': dataFeature_Elia_hah},
  'Elia_2022'      : {'path': os.path.join(DATA_ROOT, 'Elia_2022'),              'feature': dataFeature_Elia_hah},
  'DeepComp_2018'  : {'path': os.path.join(DATA_ROOT, 'DeepComp_2018'),          'feature': dataFeature_DeepComp},
  'DeepComp_2019'  : {'path': os.path.join(DATA_ROOT, 'DeepComp_2019'),          'feature': dataFeature_DeepComp},
  'RUGBYR1_old'    : {'path': os.path.join(DATA_ROOT, 'RUGBYR1','202309_202408'),'feature': dataFeature_NEM},
  'RUGBYR1_new'    : {'path': os.path.join(DATA_ROOT, 'RUGBYR1','202408_202505'),'feature': dataFeature_NEM},
# _even and _odd are subsets of _old. 
  'RUGBYR1_even'   : {'path': os.path.join(DATA_ROOT, 'RUGBYR1','even'),         'feature': dataFeature_NEM},
  'RUGBYR1_odd'    : {'path': os.path.join(DATA_ROOT, 'RUGBYR1', 'odd'),         'feature': dataFeature_NEM},
# BANN1 / EDENVSF1 use the same NEM 5-min format; populate data/<PLANT>/<period>/ from the
# ANDES pv_data/ CSVs (identical columns DATETIME/FORECAST_POE50/SCADAVALUE) before training.
  'BANN1_old'      : {'path': os.path.join(DATA_ROOT, 'BANN1','202309_202408'),  'feature': dataFeature_NEM},
  'BANN1_new'      : {'path': os.path.join(DATA_ROOT, 'BANN1','202408_202505'),  'feature': dataFeature_NEM},
  'EDENVSF1_old'   : {'path': os.path.join(DATA_ROOT, 'EDENVSF1','202309_202408'),'feature': dataFeature_NEM},
  'EDENVSF1_new'   : {'path': os.path.join(DATA_ROOT, 'EDENVSF1','202408_202505'),'feature': dataFeature_NEM},
#  'BNGSF1_even'    : {'path': os.path.join(DATA_ROOT, 'NEM_BNGSF1', 'even'),'feature': dataFeature_NEM},
#  'BNGSF1_odd'     : {'path': os.path.join(DATA_ROOT, 'NEM_BNGSF1', 'odd'),'feature': dataFeature_NEM},
#  'exp_evaluation': {'path': os.path.join(DATA_ROOT, 'exp_evaluation'),      'feature': dataFeature_NEM},
#  'exp_training'  : {'path': os.path.join(DATA_ROOT, 'exp_training'),        'feature': dataFeature_NEM},
#  'NEM_eval'      : {'path': os.path.join(DATA_ROOT, 'NEM_eval'),            'feature': dataFeature_NEM},
#  'unittest_data' : {'path': os.path.join('pv-bess-dfc', 'tests', 'data_AEMO'),   'feature': dataFeature_NEM},
}

#############################
# Environment Settings
#############################
number_of_soc_levels_for_training = 10 # number of levels for the state space of the battery SoC
initial_soc_for_evaluation = 50 # percent

#############################
# Training Settings (SB3)
#############################
sb3_config = {
"policy_type": "MultiInputPolicy",
"total_steps": 500_000,
# curtailment env (agent sets BOTH P_dfc and curtailment c -> headroom to firm the output);
# this is the variant used for the thesis chapter. The nocurtailment (BESS-only) variant is
# kept for the baseline comparison.
"env_id": "dfc_gymnasium/UtilityScalePVBESS-v0",
# "env_id": "dfc_gymnasium/UtilityScalePVBESS-v0-nocurtailment",
}

#############################
# Reward shaping (helper_fns.dfc_reward)
#############################
# Firm-capacity reward tunables (per-unit of nameplate). The reward honours the commitment
# P_dfc as a floor: SHORTFALL (Pnet<Pdfc) is penalised CONVEXLY (k_short * shortfall^2) to
# suppress deep breaches/tail risk; SURPLUS (Pnet>Pdfc) is only lightly nudged (k_surplus);
# meeting the commitment within an ABSOLUTE band (atol) earns +1 scaled by spilled PV.
# Sweep these first if a plant's shortfall energy stays high (e.g. BANN1).
reward_params = {
  "atol":      0.02,   # commitment "met" if shortfall <= 2% of nameplate (abs, grid-consistent)
  "k_short":   15.0,   # convex shortfall weight (tail-risk pressure)
  "k_surplus": 0.5,    # light over-delivery nudge to keep net tracking the commitment
  # linear curtailment penalty (mirrors the old n3 reward's -actual_c): punishes massive curtailment
  # / daytime zeroing so the agent predicts the MAX available net power, not a safe-low commitment.
  # SWEPT via DFC_K_CURTAIL; raise toward ~1.0 (n3's weight) if zeroing persists (worst on RUGBYR1/EDENVSF1).
  "k_curtail": float(os.environ.get("DFC_K_CURTAIL", "0.3")),
}

# Which per-step reward the envs use during TRAINING. Default "firm" = the redesigned
# firm-capacity reward (helper_fns.dfc_reward, tunables above). Set DFC_REWARD_FN=n3 to train
# the OLD symmetric baseline reward (helper_fns.dfc_reward_n3) for the new-vs-old-reward
# comparison. Evaluation/metrics are reward-agnostic, so only the trained policy changes.
reward_fn = os.environ.get("DFC_REWARD_FN", "firm")

#############################
# Evaluation-metric tolerances
#############################
# Tolerances are expressed in PER-UNIT OF NAMEPLATE. Because every plant's series is
# normalised to its own rating, a pu tolerance is plant-agnostic and extends to any new
# dataset without change (for a physical MW band, multiply by that plant's MW rating).
# Single source of truth: helper_fns takes the tolerance as an argument; callers pass these.
#   perfect_plan_tol_pu  - |Pnet - Pdfc| <= this counts as a "perfect plan" in
#                          perfect_plan_rate(). 0.092 pu (~9% of nameplate) reproduces the
#                          legacy KPI (historically 6 MW on RUGBYR1's 65 MW). Lower = stricter.
perfect_plan_tol_pu = 0.092

#############################
# BESS Properties
#############################
bess_capacity_puh = 0.5 # should be overwritten by application.py when --bcap is specified
soc_upper_limit_percent = 90.0 # percent
soc_lower_limit_percent = 10.0 # percent

# charging and discharging power limit for the battery is specified based on the time in sec taken to full charge or empty cahrge
charging_power_Erate = 0.28 #2/3
discharg_power_Erate = 0.28 #2/3
charging_efficiency = 0.9 # factor to multiply the input power to the battery for the actual power reflected in the SoC
discharg_efficiency = 0.9 # factor to multiply the output power to the battery for the actualy power experienced by the load

bess_properties = {
  "soc_boundary_percent": StateSpaceLimits(soc_upper_limit_percent, soc_lower_limit_percent),
  "power_boundary_Erate": StateSpaceLimits(charging_power_Erate, (-1)*discharg_power_Erate),
  "energy_capacity_puh":  bess_capacity_puh,
  "charging_efficiency":  charging_efficiency,
  "discharging_efficiency": discharg_efficiency,
}
