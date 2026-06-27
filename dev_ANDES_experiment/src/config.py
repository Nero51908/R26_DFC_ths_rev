"""
Central configuration for the grid-level DFC experiment (native 60 Hz IEEE 39-bus, ~44% PV).

All design decisions agreed during planning live here so the study scripts share a
single source of truth. See ../dev_PSSE_experiment/experiment_design.md (OneDrive tree)
for the rationale behind each value.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT      = Path(__file__).resolve().parent.parent      # dev_ANDES_experiment/
CASES     = ROOT / "cases"
RESULTS   = ROOT / "results"

# Base case (Option B): ANDES's bundled, commercially-validated IEEE 39-bus case,
# resolved at runtime via andes.get_case() (see build_case.CANDIDATE_CASES).
# The PSS/E .raw/.dyr below are retained ONLY as an optional external-validation
# reference for a single snapshot — they are NOT the working base (the ANDES PSS/E
# parser mishandled the transformer taps, collapsing the voltage profile).
BASE_RAW  = CASES / "ieee39.raw"                          # PSS/E v33 (external ref only)
BASE_DYR  = CASES / "ieee39.dyr"                          # PSS/E dynamics (external ref only)
PF_REFERENCE = CASES / "ieee39_pf_reference.csv"          # PSS/E solved V/angle (ext ref only)

# Frozen, validated base case (ANDES ieee39_full, our canonical starting point).
# build_case.py --save writes this; add_pv_bess and the study scripts load it.
# Frequency is NOT stored in the xlsx data — always (re)apply FREQ_HZ at load time.
BASE_CASE = CASES / "ieee39_base.xlsx"
PVBESS_CASE = CASES / "ieee39_pvbess.xlsx"               # high-penetration Config-B case

# ---------------------------------------------------------------- system
SBASE_MVA  = 100.0
# Transmission reinforcement factor on the (legacy) IEEE39 line MVA ratings. The bundled IEEE39 was
# rated for its original ~6 GW load, NOT a ~5.9 GW PV-BESS overlay — at x1.0 the worst line hits ~243%
# (STRUCTURAL congestion that saturates every scenario and masks the dispatch effect). x3.0 reinforces
# the network enough to host the fleet (worst line <80%), so Study A shows OPERATION, not a capacity
# shortfall. Applied in transmission_metrics (the .npz keep physical legacy-rating loadings); override
# with DFC_RATE_SCALE (=1.0 recovers the legacy view as a deliberate foil). NOTE: congestion is not what
# DFC targets — this just removes a structural artefact so the dispatch comparison is meaningful.
LINE_RATE_SCALE = float(os.environ.get("DFC_RATE_SCALE", "3.0"))
# Use the benchmark's NATIVE 60 Hz throughout (decided). Study A power flow is
# frequency-agnostic; Study B reports base-independent Δf/RoCoF. A 50 Hz relabel is an
# opt-in (attach_dynamics --rebase 50) only for NEM-standard frequency plots.
FREQ_HZ    = 60.0

# Base-case scale from the ANDES bundled ieee39_full power flow (build_case base_pf):
#   Sigma Pg = 5893.5 MW, Sigma Pload = 5856.4 MW, losses ~37 MW (0.6%), V 1.002-1.073 pu.
GEN_DISPATCH_MW   = 5893.5
TOTAL_LOAD_MW     = 5856.4     # ANDES ieee39_full base; treated as system PEAK
INSTALLED_PMAX_MW = 9280.0     # (PSS/E figure; ANDES gen Pmax may differ — refine if used)

# ---------------------------------------------------------------- penetration target
# Annual-average PV energy share ~= 44% (NEM 2025), on an annual-average demand basis.
RENEWABLE_SHARE_TARGET = 0.44
PV_CAPACITY_FACTOR     = 0.276    # measured from RUGBYR1 dataset (mean 18.02 / 65.25 MW peak)
NEM_LOAD_FACTOR        = 0.61     # AEMO: 178 TWh/yr (avg ~20.3 GW) vs max op. demand 33.26 GW
AVG_DEMAND_MW          = TOTAL_LOAD_MW * NEM_LOAD_FACTOR              # ~= 3750 MW
PV_NAMEPLATE_MW        = RENEWABLE_SHARE_TARGET * AVG_DEMAND_MW / PV_CAPACITY_FACTOR  # ~= 5.9 GW

# ---------------------------------------------------------------- PV plants (Config B)
# Three measured Australian profiles, each replicated across multiple buses with a small
# time offset to emulate geographic decorrelation. ~650 MW per site keeps each below the
# 900 MVA step-up transformers (no uprating needed).
PV_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38]   # nine non-slack generator buses
SLACK_BUS = 39                                     # kept as NEM-interconnector proxy (export)

# Real AEMO-registered AC ratings used to set the per-site MW scaling base.
PLANT_PROFILES = {
    "RUGBYR1":  {"site": "Rugby Run, QLD",  "rating_mw_ac": 65.45},
    "BANN1":    {"site": "Bannerton, VIC",  "rating_mw_ac": 88.0},
    "EDENVSF1": {"site": "Edenvale, QLD",   "rating_mw_ac": 150.0},  # ~150 AC / 204 MWp DC
}

# ---------------------------------------------------------------- BESS anchor case
# Anchor for the DFC export (scenario A3) and the centre of the BESS sizing sweep. Both knobs are
# env-overridable so the sweep can be driven without editing config (see case_tag() below).
# Thesis notation: E_cap (energy capacity, pu*h) and E-rate (power rating, P_c,max = P_d,max).
BESS_ENERGY_PU   = float(os.environ.get("DFC_BCAP",  "0.30"))  # E_cap: energy capacity (pu*h)
BESS_ERATE       = float(os.environ.get("DFC_ERATE", "0.28"))  # E-rate: charge/discharge power limit
BESS_ETA_CHG     = 0.90
BESS_ETA_DIS     = 0.90
SOC_MIN_PCT      = 10.0
SOC_MAX_PCT      = 90.0

# BESS sizing sweep — aligned to thesis Ch.3: E_cap {0.05,0.1,0.2,0.5,1.0}; E-rate {0.28 (avg 2023
# fleet), 0.67 (Victorian Big Battery, 300 MW / 450 MWh)}. (0.30 anchor sits between 0.20 and 0.50.)
SWEEP_BESS_ENERGY_PU   = [0.05, 0.10, 0.20, 0.50, 1.00]
SWEEP_BESS_ERATE       = [0.28, 0.67]
SWEEP_CURTAILMENT_FRAC = [0.0, 0.05, 0.10, 0.20]


# Per-case output namespacing for the sizing sweep. The ANCHOR (0.30 / 0.28) keeps the legacy FLAT
# layout (results/trajectories/, results/qsts/) so existing results are untouched; every other
# (bcap, erate) writes under a per-case subfolder "b<bcap>_e<erate>". The study scripts call these.
def case_tag(bcap: float = None, erate: float = None) -> str:
    b = BESS_ENERGY_PU if bcap is None else bcap
    e = BESS_ERATE if erate is None else erate
    return "" if (abs(b - 0.30) < 1e-9 and abs(e - 0.28) < 1e-9) else f"b{b:g}_e{e:g}"


def traj_dir(bcap: float = None, erate: float = None) -> Path:
    t = case_tag(bcap, erate)
    return RESULTS / "trajectories" / t if t else RESULTS / "trajectories"


def qsts_dir(bcap: float = None, erate: float = None) -> Path:
    t = case_tag(bcap, erate)
    return RESULTS / "qsts" / t if t else RESULTS / "qsts"

# ---------------------------------------------------------------- synchronous dispatch
# QSTS re-dispatch mimics AEMO's 5-min central dispatch: PV (zero marginal cost,
# semi-scheduled) is taken first, then synchronous units fill the residual in MERIT
# ORDER subject to RAMP-RATE limits and min/max. The cost/ramp numbers are
# representative (the IEEE 39-bus carries no real offer data) and are easy to adjust.
GEN_RAMP_FRAC_PER_MIN = 0.03     # ramp limit as fraction of Pmax per minute (~3%/min)
GEN_MIN_FRAC          = 0.0      # must-run floor as fraction of Pmax (0 = full backdown)
# Representative merit order ($/MWh), keyed by generator bus; cheapest dispatched first.
SYNC_MERIT_COST = {38: 18, 31: 21, 30: 24, 33: 27, 35: 30, 32: 33, 37: 36, 34: 39, 36: 45}

# ---------------------------------------------------------------- validation
PF_VOLTAGE_TOL_PU  = 1e-3   # |V_andes - V_pss/e| pass threshold
PF_ANGLE_TOL_DEG   = 0.1    # |angle_andes - angle_pss/e| pass threshold

# ---------------------------------------------------------------- PV time-series data
# (placed last: depends on PV_NAMEPLATE_MW and PV_BUSES defined above)
PV_DATA_DIR = ROOT / "pv_data"
INTERVALS_PER_DAY = 288            # 5-min resolution
SEC_PER_STEP = 86400 // INTERVALS_PER_DAY   # 300 s (matches the DRL env sec_per_step)
# Per-unit normalised SCADAVALUE + FORECAST_POE50. Two periods; the profile keys are
# identical across periods, so only the file paths differ.
#   P1 = Sep 2023-Aug 2024 (DRL training window)
#   P2 = Sep 2024-May 2025 (UNSEEN evaluation set) — default for the A1-A4 comparison,
#        matching the window the DRL agents are evaluated on.
PLANT_FILES_P1 = {
    "RUGBYR1":  "RUGBYR1/202309_202408/RUGBYR1_clean.csv",
    "BANN1":    "BANN1/202309_202408/BANN1.csv",
    "EDENVSF1": "EDENVSF1/202309_202408/EDENVSF1.csv",
}
PLANT_FILES_P2 = {
    "RUGBYR1":  "RUGBYR1/202408_202505/RUGBYR1.csv",
    "BANN1":    "BANN1/202408_202505/BANN1.csv",
    "EDENVSF1": "EDENVSF1/202408_202505/EDENVSF1.csv",
}
PERIODS = {"p1": PLANT_FILES_P1, "p2": PLANT_FILES_P2}
DEFAULT_PERIOD = "p2"
PROFILES = list(PLANT_FILES_P1)                          # the three profile names (period-agnostic)


def plant_files(period: str | None = None) -> dict:
    """File-path mapping for the requested period (default DEFAULT_PERIOD)."""
    return PERIODS[period or DEFAULT_PERIOD]


NAMEPLATE_PER_SITE_MW = PV_NAMEPLATE_MW / len(PV_BUSES)   # ~633 MW
