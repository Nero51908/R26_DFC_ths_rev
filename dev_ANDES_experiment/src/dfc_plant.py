#!/usr/bin/env python3
"""
dfc_plant.py — faithful, dependency-free port of the Ch.5 PV-BESS plant model.

Ports the DeepComp battery dynamics and the curtailment-aware step EXACTLY from
dev_DRL_experiment/dfc_gymnasium/envs/utility_scale_pv_bess.py (methods _bess_dynamics
and _pvbess_dynamics). This lets the MPC benchmark (scenario A4) run in the ANDES
environment without importing gymnasium / torch / SB3, while guaranteeing the SAME
battery physics the DRL agent (A3) was evaluated on — the basis of a fair comparison.

All powers per-unit of PV peak; energy in pu*s; SoC in percent; dt in seconds.
Cross-check against the real env once available: feed an identical (Pdfc, c, SoC0)
sequence to both and confirm identical (Pnet, SoC) — see tests in run_tests.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


class DFCPlant:
    """PV-BESS plant with the Ch.5 DeepComp battery. Stateless except for SoC, which the
    caller passes in and receives back (mirrors how the env carries state["initial_soc"])."""

    def __init__(self, energy_pu_h=None, erate=None, eta_c=None, eta_d=None,
                 soc_min_pct=None, soc_max_pct=None, sec_per_step=None):
        cap = C.BESS_ENERGY_PU if energy_pu_h is None else energy_pu_h
        self.erate = C.BESS_ERATE if erate is None else erate
        self.c_eff = C.BESS_ETA_CHG if eta_c is None else eta_c
        self.d_eff = C.BESS_ETA_DIS if eta_d is None else eta_d
        soc_min = C.SOC_MIN_PCT if soc_min_pct is None else soc_min_pct
        soc_max = C.SOC_MAX_PCT if soc_max_pct is None else soc_max_pct
        self.dt = C.SEC_PER_STEP if sec_per_step is None else sec_per_step

        # mirror helper_fns: Eb_max = puh_to_pus(cap) = cap * 3600 ; boundaries from SoC %
        self.Eb_max = cap * 3600.0                      # pu*s
        self.Eb_sup = (soc_max / 100.0) * self.Eb_max   # pu*s
        self.Eb_inf = (soc_min / 100.0) * self.Eb_max   # pu*s
        # mirror calc_Pb_boundary: Pb limit = Erate * cap  (pu)
        self.Pc_max = self.erate * cap                  # pu (charge)
        self.Pd_max = self.erate * cap                  # pu (discharge)

    # ---- DeepComp Eq.(2a,2b,3a,3b,4) — identical to env._bess_dynamics ----
    def bess_dynamics(self, Ppv: float, Pdfc: float, soc_pct: float):
        Eb = (soc_pct / 100.0) * self.Eb_max
        P_c_lim = min(self.Pc_max, (1.0 / self.c_eff) * (self.Eb_sup - Eb) / self.dt)
        P_d_lim = min(self.Pd_max, self.d_eff * (Eb - self.Eb_inf) / self.dt)
        P_c = min(max(Ppv - Pdfc, 0.0), P_c_lim)
        P_d = min(max(Pdfc - Ppv, 0.0), P_d_lim)
        next_Eb = Eb + self.c_eff * P_c * self.dt - (1.0 / self.d_eff) * P_d * self.dt
        Pb = P_c - P_d
        next_soc = (next_Eb / self.Eb_max) * 100.0
        return Pb, next_soc

    # ---- curtailment-aware step — identical to env._pvbess_dynamics ----
    def step(self, Pm: float, Pdfc: float, c: float, soc_pct: float):
        """Execute one interval. Returns (next_soc, Ppv, Pb, Pnet, actual_c)."""
        PV_deficit = Pdfc > Pm * (1.0 - c)
        if PV_deficit:
            Ppv = Pdfc if Pm > Pdfc else Pm
            Pb, next_soc = self.bess_dynamics(Ppv, Pdfc, soc_pct)
        else:
            Pb, next_soc = self.bess_dynamics(Pm, Pdfc, soc_pct)
            Ppv = Pdfc + Pb
        Pnet = Ppv - Pb
        actual_c = 1.0 - Ppv / Pm if Pm > 1e-12 else 0.0
        return next_soc, Ppv, Pb, Pnet, actual_c
