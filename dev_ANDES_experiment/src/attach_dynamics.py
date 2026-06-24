#!/usr/bin/env python3
"""
attach_dynamics.py — Study B foundation: dynamic base case + flat-line check.

Study A (QSTS) is steady-state and frequency-blind. Study B is the time-domain
(electromechanical) study that produces the actual frequency behaviour behind claim C2
(firm PV-BESS capacity substitutes for synchronous reserve and keeps the transmission
system frequency-secure). This first step validates the dynamic base before any PV-BESS
inverter models or disturbances are added:

  1. load the PRISTINE bundled ieee39_full (GENROU + TGOV1N governors + IEEEX1 exciters +
     IEEEST PSS + BusFreq meters) — NOT the round-tripped cases/ieee39_base.xlsx, whose
     xlsx re-dump loses dynamic fidelity and diverges the TDS,
  2. keep the benchmark's NATIVE 60 Hz (decided): rebasing a 60 Hz-tuned feeder to 50 Hz
     is a relabeling, not a re-derivation, and Study B reports base-independent Δf/RoCoF
     (the optional --rebase HZ relabels every device fn, e.g. 50 for NEM-standard plots),
  3. run a no-disturbance ANDES TDS to tf<12 s (the stock Toggle trips GENROU_1 at t=12 s,
     so it stays inert) and confirm the COI frequency holds flat at the nominal.

Hard-won init lessons (mirror andes.run exactly): do NOT flip model `u` flags or set
ss.config.freq AFTER setup — both corrupt the TDS initialization and diverge it.

Next steps (not here): add REGCA1/REECA1(/REPCA1) PV + BESS inverters at config.PV_BUSES
(BESS P-limit = the verified 0.084 pu E-rate cap), then the B1 cloud-ramp and B2
generator-trip events.

Run (in the ANDES venv):
    python src/attach_dynamics.py                 # native 60 Hz flat-line, tf=10 s
    python src/attach_dynamics.py --rebase 50     # optional 50 Hz relabel
Requires: andes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


BUNDLED_CASES = ["ieee39/ieee39_full.xlsx", "ieee39/ieee39.xlsx"]


def _resolve_dynamic_case():
    """Pristine ANDES-bundled ieee39_full (validated for TDS). We deliberately do NOT use
    the round-tripped cases/ieee39_base.xlsx here: the xlsx re-dump loses dynamic fidelity
    (empty GENROU coi reference, etc.) and the TDS diverges. The operating point is the
    same — ieee39_base was dumped from ieee39_full — so Study A and B stay consistent."""
    import andes
    for cand in BUNDLED_CASES:
        try:
            p = Path(andes.get_case(cand))
            if p.exists():
                return p
        except Exception:
            continue
    return C.BASE_CASE


def load_dynamic(rebase_hz: float | None = None):
    """Load the pristine dynamic case and solve PF — mirroring `andes.run` (no pre-setup
    model tampering, which we found corrupts the TDS init).

    rebase_hz=None (default) keeps the benchmark's NATIVE nominal frequency (60 Hz) and
    reports base-independent metrics. Pass rebase_hz=50 to RELABEL every dynamic device's
    fn (a relabeling, not a re-derivation). The stock Toggle (GENROU_1 trip at t=12 s) is
    left intact — keep tf < 12 s for the flat-line; Study-B event scripts handle it later.
    """
    import andes
    case = _resolve_dynamic_case()
    andes.config_logger(stream_level=30)

    if rebase_hz is None:
        # exactly like the working `andes.run` control: setup at load, DON'T touch
        # ss.config.freq (overriding it post-setup desyncs the DAE and diverges the TDS).
        ss = andes.load(str(case), setup=True, no_output=True)
        fn = float(ss.GENROU.fn.v[0]) if ss.GENROU.n else 60.0
        print(f"  dynamic case        : {case.name}")
        print(f"  native nominal frequency: {fn:g} Hz (no rebase)")
    else:                                                            # fn must change pre-setup
        ss = andes.load(str(case), setup=False, no_output=True)
        native = float(ss.GENROU.fn.v[0]) if ss.GENROU.n else 60.0
        fn = float(rebase_hz)
        for name, mdl in ss.models.items():
            if mdl.n and "fn" in mdl.params:
                mdl.fn.v[:] = [fn] * mdl.n
        ss.config.freq = fn                                          # set BEFORE setup
        ss.setup()
        print(f"  dynamic case        : {case.name}")
        print(f"  RELABELED dynamics {native:g} -> {fn:g} Hz")

    ss.PFlow.run()
    if not ss.PFlow.converged:
        sys.exit("ERROR: dynamic-base power flow did not converge.")
    print(f"  machines={ss.GENROU.n}  governors={getattr(ss,'TGOV1N',_Z()).n}  "
          f"freq-meters={getattr(ss,'BusFreq',_Z()).n}")
    return ss, fn


class _Z:        # tiny shim so getattr(...).n is 0 when a model is absent
    n = 0


def coi_frequency(ss, freq_hz: float):
    """Return (t, f_coi_Hz): inertia-weighted centre-of-inertia frequency over the TDS."""
    t = np.asarray(ss.dae.ts.t, dtype=float)
    omega = np.asarray(ss.dae.ts.y[:, ss.GENROU.omega.a], dtype=float)   # (nt, ngen), pu
    M = np.asarray(ss.GENROU.M.v, dtype=float)                           # 2H, inertia weight
    f_coi = (omega * M).sum(axis=1) / M.sum() * freq_hz
    return t, f_coi


def run_flatline(ss, tf: float, fn: float):
    ss.TDS.config.tf = float(tf)
    ss.TDS.config.no_tqdm = True
    ss.TDS.run()
    converged = bool(getattr(ss.TDS, "converged", True)) and float(ss.dae.t) >= tf - 1e-6
    t, f = coi_frequency(ss, fn)            # scale by machine nominal, not ss.config.freq
    return t, f, converged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Study B: dynamic base + flat-line check.")
    ap.add_argument("--rebase", type=float, default=None, metavar="HZ",
                    help="relabel nominal frequency (e.g. 50); default keeps native 60 Hz")
    ap.add_argument("--tf", type=float, default=10.0, help="simulation end time (s)")
    ap.add_argument("--tol", type=float, default=0.02, help="flatness tolerance (Hz)")
    args = ap.parse_args(argv)

    print(f"Study B flat-line  base={C.BASE_CASE.name}  tf={args.tf:g}s"
          f"{'' if args.rebase is None else f'  rebase->{args.rebase:g}Hz'}")
    ss, fn = load_dynamic(args.rebase)
    t, f, converged = run_flatline(ss, args.tf, fn)

    if not converged:
        print(f"  TDS did NOT converge (reached t={float(ss.dae.t):.3f}s of {args.tf:g}s)")
        print("  RESULT              : FAIL — integration diverged before tf")
        return 1

    dev = float(np.max(np.abs(f - fn)))
    out = C.RESULTS / "dyn_flatline.csv"
    out.parent.mkdir(exist_ok=True)
    np.savetxt(out, np.column_stack([t, f]), delimiter=",",
               header="t_s,f_coi_hz", comments="")
    print(f"  COI frequency range : {f.min():.4f} .. {f.max():.4f} Hz  (nominal {fn:g})")
    print(f"  max |f - {fn:g}|       : {dev:.4f} Hz  (tol {args.tol})")
    print(f"  series              : {out}")
    ok = dev <= args.tol
    print("  RESULT              :", "PASS — flat at nominal, dynamic base sound" if ok else
          "FAIL — converged but not flat (init equilibrium off)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
