#!/usr/bin/env python3
"""
forecast_quality.py — compare AEMO POE50 forecast accuracy across the three plants.

Why it matters here: in scenario A2 the committed DFC is the forecast itself (P_dfc =
FORECAST_POE50) and the BESS only has to cover the forecast ERROR. So a plant whose
forecast is more accurate hands the firming layer an easier job — its raw forecast is
already a higher-quality DFC signal. This module quantifies that per plant.

All series are per-unit of plant nameplate, so every error metric below is a FRACTION OF
CAPACITY; multiply by 100 to read it as "% of nameplate". Night is excluded: metrics are
computed on GENERATING intervals (actual > gen_thresh of capacity) because the long run of
shared night-time zeros otherwise flatters every forecast equally and hides the difference.

Metrics (per plant, generating intervals):
    cap_factor   mean actual output (pu)                      — context, not skill
    MBE          mean(forecast - actual)  (pu, signed)        — bias: +over-, -under-forecast
    MAE          mean|forecast - actual|  (pu)
    RMSE         sqrt(mean(err^2))        (pu)                — the headline P_dfc-error metric
    nRMSE        RMSE / mean(actual)      (%)                 — error relative to typical output
    corr         Pearson(forecast, actual)
    skill_vs_persist  1 - RMSE_forecast / RMSE_persistence    — beats a 5-min "no change" guess?
    P(act>=fc)   exceedance rate (%)      — POE50 should sit near 50% if well-calibrated

    python src/forecast_quality.py                 # period 2 (unseen eval set), both views
    python src/forecast_quality.py --period p1
    python src/forecast_quality.py --plot          # also write an interactive Plotly HTML
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

GEN_THRESH = 0.02            # actual > 2% of capacity => "generating" (daylight) interval


def _metrics(fc: np.ndarray, act: np.ndarray) -> dict:
    err = fc - act
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # 5-min persistence baseline: forecast_t = actual_{t-1}
    persist_err = act[1:] - act[:-1]
    rmse_persist = float(np.sqrt(np.mean(persist_err ** 2))) if len(persist_err) else np.nan
    mean_act = float(np.mean(act))
    return {
        "n": int(len(act)),
        "cap_factor": round(mean_act, 4),
        "MBE": round(float(np.mean(err)), 4),
        "MAE": round(float(np.mean(np.abs(err))), 4),
        "RMSE": round(rmse, 4),
        "nRMSE_%": round(100 * rmse / mean_act, 1) if mean_act > 1e-9 else np.nan,
        "corr": round(float(np.corrcoef(fc, act)[0, 1]), 4) if len(act) > 2 else np.nan,
        "skill_vs_persist": round(1 - rmse / rmse_persist, 4) if rmse_persist and rmse_persist > 1e-9 else np.nan,
        "P(act>=fc)_%": round(100 * float(np.mean(act >= fc)), 1),
    }


def plant_table(period: str, gen_only: bool) -> pd.DataFrame:
    rows = []
    for name in C.PROFILES:
        df = P.load_plant(name, period).dropna()
        fc, act = df["forecast"].to_numpy(), df["scada"].to_numpy()
        if gen_only:
            m = act > GEN_THRESH
            fc, act = fc[m], act[m]
        rows.append({"plant": name, **_metrics(fc, act)})
    return pd.DataFrame(rows)


def per_day_rmse(period: str) -> pd.DataFrame:
    """Daily generating-interval RMSE per plant on the COMMON complete days (apples-to-apples
    with the experiment) — gives a distribution, not just a pooled number."""
    plants = P.load_all(period)
    days = P.common_complete_days(plants)
    rows = []
    for name in C.PROFILES:
        df = plants[name]
        for d in days:
            day = df.loc[str(d.date())].iloc[: C.INTERVALS_PER_DAY]
            fc, act = day["forecast"].to_numpy(), day["scada"].to_numpy()
            m = act > GEN_THRESH
            if m.sum() < 3:
                continue
            rmse = float(np.sqrt(np.mean((fc[m] - act[m]) ** 2)))
            rows.append({"plant": name, "date": str(d.date()), "rmse": rmse})
    return pd.DataFrame(rows), days


def make_plot(period: str, daily: pd.DataFrame, out: Path) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  (plotly not installed: skipping HTML)")
        return None
    colors = {"RUGBYR1": "#1f77b4", "BANN1": "#ff7f0e", "EDENVSF1": "#2ca02c"}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=("Daily forecast RMSE (generating intervals)",
                                        "RMSE distribution"))
    for name in C.PROFILES:
        sub = daily[daily.plant == name]
        fig.add_trace(go.Scatter(x=sub.date, y=sub.rmse * 100, name=name, mode="lines",
                                 line=dict(color=colors[name], width=1)), row=1, col=1)
        fig.add_trace(go.Box(y=sub.rmse * 100, name=name, marker_color=colors[name],
                            boxmean=True, showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="RMSE (% of capacity)", row=1, col=1)
    fig.update_yaxes(title_text="RMSE (% of capacity)", row=1, col=2)
    fig.update_layout(title=f"POE50 forecast quality by plant — period {period} "
                            f"({daily.date.nunique()} common days)",
                      template="plotly_white", height=480, width=1100,
                      legend=dict(orientation="h", y=1.12))
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PV forecast-accuracy comparison across plants.")
    ap.add_argument("--period", choices=["p1", "p2"], default=C.DEFAULT_PERIOD)
    ap.add_argument("--plot", action="store_true", help="also write interactive Plotly HTML")
    args = ap.parse_args(argv)

    gen = plant_table(args.period, gen_only=True)
    allv = plant_table(args.period, gen_only=False)
    daily, days = per_day_rmse(args.period)

    pd.set_option("display.width", 140, "display.max_columns", 20)
    period_lab = "p2 (unseen eval set)" if args.period == "p2" else "p1 (training-era)"
    print(f"\nPV forecast quality — period {period_lab}\n"
          f"per-unit of nameplate, so RMSE/MAE are fractions of capacity (x100 = % cap)\n")
    print("GENERATING intervals only (actual > {:.0%} cap) — the honest comparison:".format(GEN_THRESH))
    print(gen.to_string(index=False))
    print("\nALL intervals (night zeros included; flatters everyone, shown for reference):")
    print(allv[["plant", "RMSE", "MAE", "nRMSE_%"]].to_string(index=False))

    # pooled daily RMSE on the common days, for an apples-to-apples ranking
    piv = (daily.groupby("plant")["rmse"].agg(["mean", "median", "std"]) * 100).round(2)
    piv = piv.reindex(C.PROFILES)
    print(f"\nDaily RMSE on the {len(days)} COMMON complete days (% of capacity):")
    print(piv.to_string())

    best = gen.sort_values("RMSE").iloc[0]["plant"]
    print(f"\n  Lowest generating-interval RMSE (best raw P_dfc): {best}")

    outdir = C.RESULTS
    outdir.mkdir(exist_ok=True)
    gen.assign(view="generating").to_csv(outdir / f"forecast_quality_{args.period}.csv", index=False)
    print(f"  -> {outdir / f'forecast_quality_{args.period}.csv'}")
    if args.plot:
        html = make_plot(args.period, daily, outdir / f"forecast_quality_{args.period}.html")
        if html:
            print(f"  -> {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
