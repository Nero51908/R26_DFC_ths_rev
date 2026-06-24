#!/usr/bin/env python3
"""
benchmark_mpc.py — receding-horizon MPC firming benchmark (scenario A4), self-contained.

Runs in the ANDES environment: depends only on numpy, pandas, cvxpy and the local
pv_data + dfc_plant modules — NOT on the DRL stack. It commits a firm target (P_dfc, c)
each 5-min interval from a convex QP, executes it through the ported DeepComp battery
(dfc_plant.DFCPlant, identical physics to the Ch.5 env), and writes a canonical
trajectory CSV per plant profile that study_a_qsts consumes as scenario A4.

Fairness: same battery, same per-unit data, same firm-delivery objective as the DRL
agent (A3); only the decision rule (QP vs learned policy) differs (experiment_design §7).

    python src/benchmark_mpc.py --klass clear --mode forecast --bcap 0.3
    python src/benchmark_mpc.py --date 2024-04-26 --mode oracle    # A4b upper bound
Requires: pip install cvxpy osqp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import pv_data as P
from dfc_plant import DFCPlant


class FirmingMPC:
    """Convex receding-horizon firming QP over the ported battery's own parameters."""

    def __init__(self, plant: DFCPlant, horizon: int,
                 w_ramp=5.0, w_track=1.0, w_cur=0.5, w_soc=1e-2, ref_window=6):
        self.p = plant
        self.H = int(horizon)
        self.w_ramp, self.w_track, self.w_cur, self.w_soc = w_ramp, w_track, w_cur, w_soc
        self.ref_window = ref_window
        # Energy is scaled to pu*HOURS inside the QP (values ~0.03-0.27, comparable to the
        # per-unit powers) for good OSQP conditioning. This is just a rescaling of the env's
        # pu*s convention (divide by 3600) and does not change the physics; the EXECUTION
        # model (dfc_plant) keeps the env's exact pu*s units, so A3/A4 fairness is unaffected.
        self.dt_h = plant.dt / 3600.0                    # 5 min = 1/12 h
        self.E_sup = plant.Eb_sup / 3600.0               # pu*h
        self.E_inf = plant.Eb_inf / 3600.0               # pu*h
        self.E_target = 0.5 * (self.E_sup + self.E_inf)

    def act(self, pv_hat: np.ndarray, E0_pus: float, ppoi_prev: float):
        import cvxpy as cp
        p, H = self.p, min(self.H, len(pv_hat))
        E0 = E0_pus / 3600.0                              # pu*s -> pu*h
        Pc = cp.Variable(H, nonneg=True)
        Pd = cp.Variable(H, nonneg=True)
        Pcur = cp.Variable(H, nonneg=True)
        E = cp.Variable(H + 1)
        Ppoi = pv_hat[:H] - Pcur - Pc + Pd

        ref = np.array([pv_hat[max(0, k - self.ref_window + 1): k + 1].mean()
                        for k in range(H)])
        cons = [E[0] == E0]
        for k in range(H):
            cons += [
                Pcur[k] <= pv_hat[k],
                Pc[k] <= p.Pc_max, Pd[k] <= p.Pd_max,
                E[k + 1] == E[k] + p.c_eff * Pc[k] * self.dt_h
                                 - (1.0 / p.d_eff) * Pd[k] * self.dt_h,
                E[k + 1] <= self.E_sup, E[k + 1] >= self.E_inf,
                Ppoi[k] >= 0,
            ]
        ramp = cp.square(Ppoi[0] - ppoi_prev) + cp.sum_squares(Ppoi[1:] - Ppoi[:-1])
        J = (self.w_ramp * ramp + self.w_track * cp.sum_squares(Ppoi - ref)
             + self.w_cur * cp.sum(Pcur) + self.w_soc * cp.sum_squares(E[1:] - self.E_target))
        prob = cp.Problem(cp.Minimize(J), cons)
        prob.solve(solver=cp.OSQP, warm_start=True,
                   max_iter=20000, eps_abs=1e-6, eps_rel=1e-6, polish=True)

        if Ppoi.value is None:                           # solver failed outright
            return float(np.clip(pv_hat[0], 0.0, 1.0)), 0.0
        p_dfc = float(np.clip(Ppoi.value[0], 0.0, 1.0))
        c = float(np.clip(Pcur.value[0] / max(pv_hat[0], 1e-9), 0.0, 1.0))
        return p_dfc, c


def window(series: np.ndarray, i: int, H: int, mode: str) -> np.ndarray:
    if mode == "persistence":
        return np.full(H, series[i])
    w = series[i: i + H]
    if len(w) < H:
        w = np.concatenate([w, np.full(H - len(w), w[-1] if len(w) else 0.0)])
    return w


def run_profile(profile: str, day, df: pd.DataFrame, horizon: int, mode: str,
                init_soc=50.0) -> pd.DataFrame:
    """One plant profile, one day: receding-horizon MPC -> canonical trajectory rows."""
    day_df = df.loc[str(pd.Timestamp(day).date())].iloc[:C.INTERVALS_PER_DAY]
    Pf = day_df["forecast"].to_numpy()
    Pm = day_df["scada"].to_numpy()
    stamps = day_df.index.strftime("%Y-%m-%d %H:%M:%S")
    horizon_series = Pm if mode == "oracle" else Pf

    plant = DFCPlant()
    mpc = FirmingMPC(plant, horizon)
    soc, ppoi_prev = init_soc, 0.0
    rows = []
    for t in range(len(Pm)):
        pv_hat = window(horizon_series, t, horizon, mode)
        E0 = (soc / 100.0) * plant.Eb_max
        p_dfc, c = mpc.act(pv_hat, E0, ppoi_prev)
        soc, Ppv, Pb, Pnet, actual_c = plant.step(float(Pm[t]), p_dfc, c, soc)
        rows.append(dict(idx=t, datetime=stamps[t], P_forecast=float(Pf[t]),
                         P_actual=float(Pm[t]), P_net=Pnet, P_dfc=p_dfc, P_bess_cmd=Pb,
                         P_curtail=float(Pm[t]) * actual_c))
        ppoi_prev = Pnet
    return pd.DataFrame(rows)


def write_accumulate(traj_new: pd.DataFrame, out: Path) -> None:
    """Append this day to a multi-day MPC trajectory file, replacing that day if present.

    MPC is solved one day at a time; accumulating keeps every generated day in the file so
    `--compare`/QSTS can slice any of them by date (a single-day file would zero-fill the
    others). De-dupes by calendar day and keeps a running idx.
    """
    days_new = set(pd.to_datetime(traj_new["datetime"]).dt.date)
    if out.exists():
        old = pd.read_csv(out)
        if list(old.columns) == list(traj_new.columns):
            keep = ~pd.to_datetime(old["datetime"], errors="coerce").dt.date.isin(days_new)
            traj_new = pd.concat([old[keep], traj_new], ignore_index=True)
    comb = traj_new.sort_values("datetime").reset_index(drop=True)
    comb["idx"] = range(len(comb))
    comb.to_csv(out, index=False)


def main(argv=None) -> int:
    import time
    ap = argparse.ArgumentParser(description="MPC firming benchmark (A4), self-contained.")
    ap.add_argument("--klass", choices=["clear", "mixed", "overcast"], default="clear")
    ap.add_argument("--date", default=None, help="explicit YYYY-MM-DD")
    ap.add_argument("--days", nargs="+", default=None, help="explicit list of YYYY-MM-DD")
    ap.add_argument("--all-days", action="store_true",
                    help="every complete period day (full availability comparison; ~25-50 min)")
    ap.add_argument("--horizon", type=int, default=24, help="MPC horizon in 5-min steps")
    ap.add_argument("--mode", choices=["persistence", "forecast", "oracle"], default="forecast")
    ap.add_argument("--bcap", type=float, default=None, help="override BESS energy (pu*h)")
    ap.add_argument("--prefix", default="mpc")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    args = ap.parse_args(argv)
    if args.bcap is not None:
        C.BESS_ENERGY_PU = args.bcap

    plants = P.load_all(args.period)
    common = P.common_complete_days(plants)
    if args.all_days:
        day_list = [pd.Timestamp(d) for d in common]
    elif args.days:
        day_list = [pd.Timestamp(d) for d in args.days]
    elif args.date:
        day_list = [pd.Timestamp(args.date)]
    else:
        tbl = P.day_class_table(plants, common)
        sel = tbl[tbl["klass"] == args.klass]
        day_list = [pd.Timestamp(sel.iloc[len(sel) // 2]["date"])]

    out_dir = C.RESULTS / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    multi = len(day_list) > 1
    print(f"MPC A4  {len(day_list)} day(s)  mode={args.mode}  horizon={args.horizon}  "
          f"bcap={C.BESS_ENERGY_PU}")
    per_plant = {name: [] for name in C.PROFILES}
    bar = None
    if multi:                                        # progress bar over days (low overhead)
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(day_list), unit="day", desc="MPC")
        except ImportError:
            pass
    t0 = time.time()
    for i, day in enumerate(day_list):
        for name in C.PROFILES:
            per_plant[name].append(run_profile(name, day, plants[name], args.horizon, args.mode))
        if bar is not None:
            bar.update(1)
        elif multi and ((i + 1) % 10 == 0 or i + 1 == len(day_list)):
            print(f"  {i + 1}/{len(day_list)} days  ({time.time() - t0:.0f}s)", flush=True)
    if bar is not None:
        bar.close()

    for name in C.PROFILES:
        out = out_dir / f"{args.prefix}_{name}.csv"
        if multi:                                    # write the full multi-day file at once
            comb = pd.concat(per_plant[name], ignore_index=True)
            comb = comb.sort_values("datetime").reset_index(drop=True)
            comb["idx"] = range(len(comb))
            comb.to_csv(out, index=False)
        else:                                        # single day: accumulate onto existing
            write_accumulate(per_plant[name][0], out)
        print(f"  {name:9s}: {len(day_list)} day(s) -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
