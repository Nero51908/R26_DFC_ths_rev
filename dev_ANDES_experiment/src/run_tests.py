#!/usr/bin/env python3
"""
run_tests.py — pipeline self-test. Run this in your ANDES venv; it writes
results/test_report.txt (and prints the same) so the expected behaviours can be reviewed.

Sections:
  A config & import integrity          (always)
  B PV data loader                     (always)
  C economic dispatch unit tests       (always)
  D trajectory adapter                 (always)
  E0 ported DeepComp battery           (always)
  G rule firming baseline (A2)         (always)
  E MPC QP solver availability         (needs cvxpy/osqp; SKIP otherwise)
  F ANDES QSTS smoke (3 intervals)     (needs andes + ieee39_pvbess.xlsx; SKIP otherwise)

A FAIL never aborts the run — every section is attempted so the report is complete.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

LOG: list[str] = []


def rec(tag, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {tag}" + (f"  — {detail}" if detail else "")
    print(line); LOG.append(line)
    return ok


def skip(tag, why):
    line = f"[SKIP] {tag}  — {why}"
    print(line); LOG.append(line)


def section(title):
    line = f"\n=== {title} ==="
    print(line); LOG.append(line)


# ----------------------------------------------------------------- A
def test_config():
    section("A. config & import integrity")
    import pv_data, trajectory  # noqa: F401
    rec("A1 imports (config, pv_data, trajectory)", True)
    rec("A2 PV nameplate ~5.7 GW", 5000 < C.PV_NAMEPLATE_MW < 6500,
        f"{C.PV_NAMEPLATE_MW:.0f} MW")
    rec("A3 nine PV buses", len(C.PV_BUSES) == 9, str(C.PV_BUSES))
    rec("A4 per-site nameplate consistent",
        abs(C.NAMEPLATE_PER_SITE_MW * 9 - C.PV_NAMEPLATE_MW) < 1)
    rec("A5 BESS anchor 0.3pu / 0.28 Erate",
        C.BESS_ENERGY_PU == 0.30 and C.BESS_ERATE == 0.28)


# ----------------------------------------------------------------- B
def test_loader():
    section("B. PV data loader")
    import pv_data as P
    plants = P.load_all()
    days = P.common_complete_days(plants)
    rec("B1 common complete days > 100", len(days) > 100, f"{len(days)} days")
    tbl = P.day_class_table(plants, days)
    counts = tbl["klass"].value_counts().to_dict()
    rec("B2 all 3 day-classes present",
        all(k in counts for k in ("clear", "mixed", "overcast")), str(counts))
    M = P.day_injection_matrix(plants, days[len(days)//2])
    rec("B3 injection matrix shape (288, 9)", M.shape == (288, 9), str(M.shape))
    rec("B4 peak injection <= nameplate",
        M.sum(axis=1).max() <= C.PV_NAMEPLATE_MW * 1.001,
        f"peak {M.sum(axis=1).max():.0f} MW")


# ----------------------------------------------------------------- C
def test_dispatch():
    section("C. economic dispatch unit tests")
    import study_a_qsts as SA
    n = 9
    pmax = np.full(n, 1000.0); pmin = np.zeros(n)
    cost = np.arange(n, dtype=float)            # unit 0 cheapest
    prev = np.full(n, 200.0)

    d = SA.econ_dispatch(prev, 1800.0, pmin, pmax, np.full(n, 1e6), cost)
    rec("C1 residual fully met (loose ramp)", abs(d.sum() - 1800.0) < 1e-3,
        f"sum={d.sum():.1f}")
    rec("C2 merit order (cheapest loaded first)", d[0] >= d[-1],
        f"d0={d[0]:.0f} d8={d[-1]:.0f}")

    d2 = SA.econ_dispatch(prev, 1800.0, pmin, pmax, np.full(n, 50.0), cost)
    rec("C3 ramp limit respected", bool(np.all(d2 <= prev + 50 + 1e-6)))

    d3 = SA.econ_dispatch(prev, 1e6, pmin, pmax, np.full(n, 1e6), cost)
    rec("C4 capped at Pmax when residual huge", bool(np.all(d3 <= pmax + 1e-6)),
        f"max={d3.max():.0f}")


# ----------------------------------------------------------------- D
def test_trajectory():
    section("D. trajectory adapter")
    import pandas as pd
    import trajectory as T
    tmp = C.RESULTS / "trajectories"
    tmp.mkdir(parents=True, exist_ok=True)
    # synthesize a one-day bell-curve trajectory for each profile
    t = np.arange(288)
    bell = np.clip(np.sin((t - 72) / 144 * np.pi), 0, 1)
    for name in C.PROFILES:
        pd.DataFrame({"idx": t, "datetime": "", "P_forecast": bell, "P_actual": bell,
                      "P_net": bell * 0.9, "P_dfc": bell, "P_bess_cmd": 0.0,
                      "P_curtail": bell * 0.1}).to_csv(tmp / f"selftest_{name}.csv", index=False)
    traj = T.load_profile_trajectories(tmp, "selftest")
    M = T.injection_from_trajectories(traj, date=None, column="P_net")
    rec("D1 trajectory injection shape (288, 9)", M.shape == (288, 9), str(M.shape))
    rec("D2 firm peak ~ 0.9 x nameplate sum",
        abs(M.sum(axis=1).max() - 0.9 * C.PV_NAMEPLATE_MW) < 50,
        f"peak {M.sum(axis=1).max():.0f} MW")
    for name in C.PROFILES:                       # remove transient fixtures (best-effort)
        try:
            (tmp / f"selftest_{name}.csv").unlink(missing_ok=True)
        except OSError:
            pass


# ----------------------------------------------------------------- E0 (ported battery)
def test_dfc_plant():
    section("E0. ported DeepComp battery (dfc_plant)")
    from dfc_plant import DFCPlant
    p = DFCPlant()
    rec("E0a Eb_max = cap*3600", abs(p.Eb_max - C.BESS_ENERGY_PU * 3600) < 1e-6,
        f"{p.Eb_max:.0f} pu*s")
    rec("E0b Pc_max = Erate*cap", abs(p.Pc_max - C.BESS_ERATE * C.BESS_ENERGY_PU) < 1e-9,
        f"{p.Pc_max:.4f} pu")
    # charge from surplus PV: SoC must rise and stay <= 90%
    soc, Ppv, Pb, Pnet, ac = p.step(Pm=0.8, Pdfc=0.4, c=0.0, soc_pct=50.0)
    rec("E0c surplus -> charge (Pb>0, SoC up)", Pb > 0 and soc > 50.0,
        f"Pb={Pb:.3f} soc={soc:.1f}")
    # deficit: discharge to lift delivery toward Pdfc; SoC falls, stays >= 10%
    soc2, *_ , Pnet2, _ = p.step(Pm=0.2, Pdfc=0.5, c=0.0, soc_pct=50.0)
    rec("E0d deficit -> discharge (SoC down, >=10%)", soc2 < 50.0 and soc2 >= 10.0,
        f"soc={soc2:.1f}")
    # SoC never leaves [10,90] even with extreme commands
    socs = []
    s = 50.0
    for _ in range(300):
        s, *_ = p.step(Pm=1.0, Pdfc=0.0, c=0.0, soc_pct=s)   # force max charge
        socs.append(s)
    rec("E0e SoC respects 90% ceiling under sustained charge", max(socs) <= 90.0 + 1e-6,
        f"max soc={max(socs):.2f}")


# ----------------------------------------------------------------- G (rule firming A2)
def test_rule_firming():
    section("G. rule firming baseline (A2, make_firm_trajectory)")
    import pandas as pd
    import make_firm_trajectory as MF
    # one synthetic volatile day -> firm delivery must be smoother and stay in [0,1]
    t = np.arange(C.INTERVALS_PER_DAY)
    bell = np.clip(np.sin((t - 72) / 144 * np.pi), 0, 1)
    noisy = np.clip(bell + 0.15 * np.sin(t / 3.0), 0, 1)
    df = pd.DataFrame({"forecast": bell, "scada": noisy},
                      index=pd.date_range("2024-01-01", periods=len(t), freq="5min"))
    traj = MF.firm_profile(df)
    rec("G1 canonical schema + length", list(traj.columns) ==
        ["idx", "datetime", "P_forecast", "P_actual", "P_net", "P_dfc", "P_bess_cmd", "P_curtail"]
        and len(traj) == C.INTERVALS_PER_DAY)
    rec("G2 P_net within [0,1]", traj.P_net.min() >= -1e-9 and traj.P_net.max() <= 1 + 1e-6,
        f"{traj.P_net.min():.3f}..{traj.P_net.max():.3f}")
    rfirm = float(np.std(np.diff(traj.P_net))); rraw = float(np.std(np.diff(traj.P_actual)))
    rec("G3 delivery smoother than raw PV", rfirm < rraw, f"ramp {rraw:.4f} -> {rfirm:.4f}")


# ----------------------------------------------------------------- H (env equivalence)
def test_env_equivalence():
    section("H. dfc_plant bit-equivalence vs Ch.5 env")
    from dfc_plant import DFCPlant
    cap, er, ce, de, socu, socl, dt = 0.30, 0.28, 0.9, 0.9, 90.0, 10.0, 300.0
    Eb_max = cap * 3600.0
    Pb_sup, Pb_inf = er * cap, -er * cap            # env calc_Pb_boundary (uses pu*h cap)
    Eb_sup, Eb_inf = (socu / 100) * Eb_max, (socl / 100) * Eb_max

    def env_bess(Ppv, Pdfc, soc):                   # transcription of env._bess_dynamics
        Eb = (soc / 100) * Eb_max
        Pc = min(max(Ppv - Pdfc, 0), min(Pb_sup, (1 / ce) * (Eb_sup - Eb) / dt))
        Pd = min(max(Pdfc - Ppv, 0), min(-Pb_inf, de * (Eb - Eb_inf) / dt))
        nEb = Eb + ce * Pc * dt - (1 / de) * Pd * dt
        return Pc - Pd, (nEb / Eb_max) * 100

    def env_step(Pm, Pdfc, c, soc):                 # transcription of env._pvbess_dynamics
        if Pdfc > Pm * (1 - c):
            Ppv = Pdfc if Pm > Pdfc else Pm
            Pb, ns = env_bess(Ppv, Pdfc, soc)
        else:
            Pb, ns = env_bess(Pm, Pdfc, soc); Ppv = Pdfc + Pb
        Pnet = Ppv - Pb
        return ns, Ppv, Pb, Pnet, (1 - Ppv / Pm if Pm > 0 else 0.0)

    p = DFCPlant()
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(50000):
        Pm, Pdfc = rng.uniform(0.01, 1.0), rng.uniform(0, 1.0)
        c, soc = rng.uniform(0, 0.3), rng.uniform(10, 90)
        maxerr = max(maxerr, max(abs(a - b) for a, b in zip(env_step(Pm, Pdfc, c, soc),
                                                            p.step(Pm, Pdfc, c, soc))))
    rec("H1 dfc_plant == env (DeepComp battery, 0.28 E-rate) bit-exact", maxerr < 1e-12,
        f"max|Δ|={maxerr:.1e}")


# ----------------------------------------------------------------- E (MPC)
def test_mpc_solver():
    section("E. MPC QP (self-contained, cvxpy)")
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        skip("E MPC QP", "cvxpy not installed (pip install cvxpy osqp)")
        return
    import pandas as pd
    import pv_data as P
    import benchmark_mpc as M
    plants = P.load_all(); days = P.common_complete_days(plants)
    day = days[len(days) // 2]
    traj = M.run_profile("RUGBYR1", day, plants["RUGBYR1"], horizon=12, mode="forecast")
    rec("E1 MPC produced a full day", len(traj) == C.INTERVALS_PER_DAY, f"{len(traj)} rows")
    rec("E2 P_net within [0,1]", float(traj.P_net.min()) >= -1e-6 and float(traj.P_net.max()) <= 1.0 + 1e-3,
        f"{traj.P_net.min():.3f}..{traj.P_net.max():.3f}")
    # firming: the delivered P_net should be SMOOTHER than the raw PV it tracks
    ramp_net = float(np.mean(np.abs(np.diff(traj.P_net))))
    ramp_pv = float(np.mean(np.abs(np.diff(traj.P_actual))))
    rec("E3 MPC delivery smoother than raw PV", ramp_net <= ramp_pv,
        f"ramp net={ramp_net:.4f} vs pv={ramp_pv:.4f}")


# ----------------------------------------------------------------- F
def test_andes_smoke():
    section("F. ANDES QSTS smoke (3 intervals)")
    try:
        import andes  # noqa: F401
    except ImportError:
        skip("F ANDES smoke", "andes not installed")
        return
    if not C.PVBESS_CASE.exists():
        skip("F ANDES smoke", f"{C.PVBESS_CASE.name} not found (run add_pv_bess.py)")
        return
    import pv_data as P
    import study_a_qsts as SA
    plants = P.load_all(); days = P.common_complete_days(plants)
    inj = P.day_injection_matrix(plants, days[len(days)//2])[:3]   # 3 intervals
    ss = SA.load_case()
    _, real_load = SA.split_pq(ss)
    rows, _net = SA.run_day(ss, inj, real_load, record_net=False)
    rec("F1 all 3 intervals converged", all(r["converged"] for r in rows),
        f"{sum(r['converged'] for r in rows)}/3")
    rec("F2 voltages in band",
        all(0.9 < r["vmin"] and r["vmax"] < 1.12 for r in rows),
        f"V {min(r['vmin'] for r in rows):.3f}..{max(r['vmax'] for r in rows):.3f}")


def main():
    for fn in (test_config, test_loader, test_dispatch, test_trajectory,
               test_dfc_plant, test_env_equivalence, test_rule_firming,
               test_mpc_solver, test_andes_smoke):
        try:
            fn()
        except Exception:
            rec(f"{fn.__name__} crashed", False, "see traceback below")
            LOG.append(traceback.format_exc())

    npass = sum(l.startswith("[PASS]") for l in LOG)
    nfail = sum(l.startswith("[FAIL]") for l in LOG)
    nskip = sum(l.startswith("[SKIP]") for l in LOG)
    summary = f"\nSUMMARY: {npass} pass, {nfail} fail, {nskip} skip"
    print(summary); LOG.append(summary)

    C.RESULTS.mkdir(exist_ok=True)
    (C.RESULTS / "test_report.txt").write_text("\n".join(LOG) + "\n")
    print(f"report -> {C.RESULTS / 'test_report.txt'}")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
