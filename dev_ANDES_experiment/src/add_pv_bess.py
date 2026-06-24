#!/usr/bin/env python3
"""
add_pv_bess.py — build the high-PV-penetration IEEE 39-bus case (Config B).

Loads the frozen base case (config.BASE_CASE), installs nine PV-BESS plants across the
non-slack generator buses, derates the synchronous fleet to make room, keeps bus 39 as
the export-capable interconnector slack, then re-solves and sanity-checks.

Modelling level (this script): POWER FLOW. Each PV-BESS plant is one net-injection
generator at its bus; the BESS energy/SoC is tracked externally by the QSTS engine
(see experiment_design.md §7.1). The REGCA1/REECA1/REPCA1 inverter dynamics needed for
Study B are attached separately in attach_dynamics.py (kept apart so that wiring is
validated on its own).

Run (in the ANDES venv):
    python src/add_pv_bess.py                       # 44% instantaneous snapshot, save case
    python src/add_pv_bess.py --pen 0.6 --no-save   # explore a higher-PV snapshot
    python src/add_pv_bess.py --dry-run             # print the dispatch plan only (no ANDES)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# ----------------------------------------------------------------- dispatch planning
@dataclass
class PlantPlan:
    bus: int
    profile: str          # RUGBYR1 / BANN1 / EDENVSF1 (sets the time series later)
    nameplate_mw: float
    p_mw: float           # net injection at this snapshot


def plan_dispatch(load_mw: float, syn_bus_p: dict[int, float], pen: float,
                  pv_buses: list[int], pv_nameplate_total: float,
                  profiles: list[str]) -> dict:
    """Pure-Python dispatch plan (no ANDES) so the arithmetic is unit-testable.

    - Installs `pv_nameplate_total` split equally across `pv_buses`.
    - Sets each plant's snapshot output so total PV injection = pen * load.
    - Derates the synchronous machines (buses in syn_bus_p) proportionally so
      Sigma(syn) = load - PV; surplus (if PV exceeds the fleet) exports via the slack.
    """
    n = len(pv_buses)
    nameplate_each = pv_nameplate_total / n
    pv_total = pen * load_mw
    pv_each = pv_total / n

    plants = [PlantPlan(bus=b, profile=profiles[i % len(profiles)],
                        nameplate_mw=nameplate_each,
                        p_mw=min(pv_each, nameplate_each))   # cannot exceed nameplate
              for i, b in enumerate(pv_buses)]

    syn_base_total = sum(syn_bus_p.values())
    syn_target_total = max(0.0, load_mw - sum(p.p_mw for p in plants))
    scale = (syn_target_total / syn_base_total) if syn_base_total > 0 else 0.0
    syn_new = {b: p * scale for b, p in syn_bus_p.items()}

    return {
        "load_mw": load_mw,
        "plants": plants,
        "pv_total_mw": sum(p.p_mw for p in plants),
        "syn_new": syn_new,
        "syn_base_total": syn_base_total,
        "syn_target_total": syn_target_total,
        "scale": scale,
        "penetration": sum(p.p_mw for p in plants) / load_mw,
    }


# ----------------------------------------------------------------- ANDES application
def load_base(freq_hz: float):
    import andes
    if not C.BASE_CASE.exists():
        sys.exit(f"Base case not found: {C.BASE_CASE}. Run build_case.py --save first.")
    andes.config_logger(stream_level=30)
    ss = andes.load(str(C.BASE_CASE), setup=False, no_output=True)
    ss.config.freq = float(freq_hz)
    return ss


def gather_syn(ss) -> dict[int, float]:
    """Base synchronous real-power dispatch at non-slack generator buses {bus: P_MW}."""
    import numpy as np
    out = {}
    buses = list(ss.PV.bus.v)
    p = np.asarray(ss.PV.p0.v, float) * C.SBASE_MVA
    for i, b in enumerate(buses):
        out[int(b)] = float(p[i])
    return out


def apply_plan(ss, plan: dict) -> None:
    """Derate synchronous machines and add the PV-BESS net-injection generators."""
    import numpy as np

    # 1) derate synchronous units in place (p0 is per-unit on SBASE)
    buses = list(ss.PV.bus.v)
    p0 = ss.PV.p0.v
    for i, b in enumerate(buses):
        b = int(b)
        if b in plan["syn_new"]:
            p0[i] = plan["syn_new"][b] / C.SBASE_MVA

    # 2) add each PV-BESS plant as a fixed unity-PF injection (negative PQ load).
    #    Modelling it as PQ (not a voltage-regulating PV gen) avoids two regulators
    #    fighting at the same bus — the synchronous machine holds the bus voltage while
    #    the inverter injects real power at unity PF (semi-scheduled assumption). The
    #    REGCA1/REECA1/REPCA1 dynamics that add reactive/voltage support come later in
    #    attach_dynamics.py for Study B.
    for p in plan["plants"]:
        ss.add("PQ", dict(
            idx=f"PVBESS_{p.bus}",
            name=f"PVBESS_{p.bus}_{p.profile}",
            bus=p.bus,
            p0=-p.p_mw / C.SBASE_MVA,       # negative load == injection
            q0=0.0,                          # unity power factor
        ))


def report(ss, plan: dict) -> bool:
    import numpy as np
    v = np.asarray(ss.Bus.v.v, float)
    pv_inj = float(np.sum([pp.p_mw for pp in plan["plants"]]))
    slack_p = float(np.sum(ss.Slack.p.v)) * C.SBASE_MVA
    load = plan["load_mw"]   # true load (ss.PQ now also holds negative PV injections)

    print(f"  installed PV nameplate : {sum(pp.nameplate_mw for pp in plan['plants']):.0f} MW "
          f"over {len(plan['plants'])} buses")
    print(f"  PV injection (snapshot): {pv_inj:.0f} MW  ->  {pv_inj/load*100:.1f}% of load")
    print(f"  synchronous fleet      : {plan['syn_target_total']:.0f} MW "
          f"(was {plan['syn_base_total']:.0f} MW, scale {plan['scale']:.2f})")
    print(f"  slack (interconnector) : {slack_p:+.0f} MW  (negative = export)")
    print(f"  voltage range          : {v.min():.4f} .. {v.max():.4f} pu")

    checks = {
        "PF converged": bool(ss.PFlow.converged),
        "voltages 0.94-1.10 pu": bool(v.min() > 0.94 and v.max() < 1.10),
        "penetration within 2% of target": abs(pv_inj/load - plan["penetration"]) < 0.02,
    }
    print("  sanity checks:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the high-PV-penetration Config-B case.")
    p.add_argument("--pen", type=float, default=C.RENEWABLE_SHARE_TARGET,
                   help="target instantaneous PV share of load for this snapshot")
    p.add_argument("--freq", type=float, default=C.FREQ_HZ)
    p.add_argument("--out", type=Path, default=C.CASES / "ieee39_pvbess.xlsx")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and print the dispatch plan without ANDES")
    args = p.parse_args(argv)

    profiles = list(C.PLANT_PROFILES.keys())

    if args.dry_run:
        # use config scale figures so the plan can be checked without ANDES
        syn_mock = {b: C.GEN_DISPATCH_MW / 9 for b in C.PV_BUSES}  # rough placeholder
        plan = plan_dispatch(C.TOTAL_LOAD_MW, syn_mock, args.pen,
                             C.PV_BUSES, C.PV_NAMEPLATE_MW, profiles)
        print(f"DRY RUN  load={C.TOTAL_LOAD_MW:.0f} MW  pen={args.pen:.2f}")
        print(f"  PV total injection : {plan['pv_total_mw']:.0f} MW "
              f"({plan['penetration']*100:.1f}% of load)")
        print(f"  per plant          : {plan['plants'][0].p_mw:.0f} MW "
              f"(nameplate {plan['plants'][0].nameplate_mw:.0f} MW)")
        print(f"  synchronous derate : {plan['syn_base_total']:.0f} -> "
              f"{plan['syn_target_total']:.0f} MW (scale {plan['scale']:.2f})")
        for pl in plan["plants"]:
            print(f"    bus {pl.bus}: {pl.profile:8s} P={pl.p_mw:.0f} MW")
        return 0

    ss = load_base(args.freq)
    plan = plan_dispatch(
        load_mw=float(__import__("numpy").sum(ss.PQ.p0.v)) * C.SBASE_MVA,
        syn_bus_p=gather_syn(ss), pen=args.pen,
        pv_buses=C.PV_BUSES, pv_nameplate_total=C.PV_NAMEPLATE_MW, profiles=profiles,
    )
    print(f"Building Config-B case (freq {args.freq:g} Hz, target pen {args.pen:.0%}) ...")
    apply_plan(ss, plan)
    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        sys.exit("ERROR: high-penetration power flow did NOT converge.")
    ok = report(ss, plan)

    if ok and not args.no_save:
        import andes
        args.out.parent.mkdir(exist_ok=True)
        andes.io.xlsx.write(ss, str(args.out), overwrite=True)
        print(f"  saved case             : {args.out}")
    elif not ok:
        print("  NOT saving — sanity checks failed.")
    print("  RESULT                 :", "PASS — Config-B case ready" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
