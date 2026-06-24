"""
Central configuration for the grid-level DFC experiment (native 60 Hz IEEE 39-bus, ~44% PV).

All design decisions agreed during planning live here so the study scripts share a
single source of truth. See ../dev_PSSE_experiment/experiment_design.md (OneDrive tree)
for the rationale behind each value.
"""
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
# Anchor for the DFC export (scenario A3) and the centre of the Study C sweep.
BESS_ENERGY_PU   = 0.30    # energy capacity relative to PV peak power (pu*h)
BESS_ERATE       = 0.28    # charge/discharge power limit (E-rate)
BESS_ETA_CHG     = 0.90
BESS_ETA_DIS     = 0.90
SOC_MIN_PCT      = 10.0
SOC_MAX_PCT      = 90.0

# Study C sweep grid (centred on the anchor)
SWEEP_BESS_ENERGY_PU   = [0.15, 0.30, 0.50, 1.00]   # 0.30 = anchor
SWEEP_CURTAILMENT_FRAC = [0.0, 0.05, 0.10, 0.20]

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
