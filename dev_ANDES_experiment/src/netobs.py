#!/usr/bin/env python3
"""
netobs.py — transmission-network observables from a solved ANDES power flow.

Computes per-line apparent power (MVA) and % of rating, and per-bus voltage, from the
solved bus complex voltages and the line pi-model (series y, total charging b, complex
tap T = tap·e^{jφ}). MATPOWER convention, verified for sign/loss/charging:

    I_f = (y + jb/2)/|T|²·V_f − y/conj(T)·V_t
    I_t = −y/T·V_f + (y + jb/2)·V_t
    S_end = V_end · conj(I_end);   loading% = max(|S_f|,|S_t|)·Sbase / rate_a

Used by study_a_qsts to log line loadings / bus voltages each interval for the
transmission-network analysis (transmission_metrics.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_RAW_RATE_CACHE: dict = {}


def _rate_from_raw(ss):
    """ANDES' bundled ieee39 ships with Line.rate_a = 0, so %loading is undefined (all-NaN). Pull the
    real per-line MVA ratings from the PSS/E reference (config.BASE_RAW), matched by UNORDERED bus pair
    (IEEE39 has no parallels, so the pair is unique). Cached per line-set so the .raw loads once."""
    key = tuple((int(a), int(b)) for a, b in zip(ss.Line.bus1.v, ss.Line.bus2.v))
    if key in _RAW_RATE_CACHE:
        return _RAW_RATE_CACHE[key]
    from collections import defaultdict, deque
    try:
        import andes
        raw = andes.load(str(C.BASE_RAW), setup=True, no_output=True)
    except Exception:
        _RAW_RATE_CACHE[key] = None
        return None
    pool = defaultdict(deque)
    for b1, b2, ra in zip(raw.Line.bus1.v, raw.Line.bus2.v, np.asarray(raw.Line.rate_a.v, float)):
        pool[frozenset((int(b1), int(b2)))].append(float(ra))
    out = np.array([(pool[frozenset(p)].popleft() if pool[frozenset(p)] else 0.0) for p in key], float)
    _RAW_RATE_CACHE[key] = out
    return out


def line_meta(ss):
    """Static per-line metadata (computed once): from/to bus, rating, and a precomputed
    index map. Returns a dict reused across intervals."""
    busidx = [int(b) for b in ss.Bus.idx.v]
    b2i = {b: i for i, b in enumerate(busidx)}
    f = np.array([b2i[int(b)] for b in ss.Line.bus1.v])
    t = np.array([b2i[int(b)] for b in ss.Line.bus2.v])
    r = np.asarray(ss.Line.r.v, float)
    x = np.asarray(ss.Line.x.v, float)
    bsh = np.asarray(ss.Line.b.v, float)
    tap = np.asarray(getattr(ss.Line, "tap").v, float) if hasattr(ss.Line, "tap") else np.ones(len(f))
    phi = np.asarray(getattr(ss.Line, "phi").v, float) if hasattr(ss.Line, "phi") else np.zeros(len(f))
    rate = np.asarray(ss.Line.rate_a.v, float)
    if (rate <= 0).all():                        # ANDES ieee39 ships UNRATED -> %loading would be all
        r2 = _rate_from_raw(ss)                   # NaN; inject real MVA ratings from the PSS/E reference
        if r2 is not None and (r2 > 0).any():     # (config.BASE_RAW), matched by bus pair.
            rate = r2
    ys = 1.0 / (r + 1j * x)
    T = tap * np.exp(1j * phi)
    return dict(f=f, t=t, ys=ys, bsh=bsh, T=T, rate=rate,
                line_idx=list(ss.Line.idx.v), busidx=busidx)


def line_flows(ss, meta=None):
    """Return (smax_mva, loading_pct) arrays over all lines for the current solved state."""
    if meta is None:
        meta = line_meta(ss)
    V = np.asarray(ss.Bus.v.v, float) * np.exp(1j * np.asarray(ss.Bus.a.v, float))
    Vf, Vt = V[meta["f"]], V[meta["t"]]
    ys, T, bsh = meta["ys"], meta["T"], meta["bsh"]
    If = (ys + 1j * bsh / 2) / np.abs(T) ** 2 * Vf - ys / np.conj(T) * Vt
    It = -ys / T * Vf + (ys + 1j * bsh / 2) * Vt
    smax = np.maximum(np.abs(Vf * np.conj(If)), np.abs(Vt * np.conj(It))) * C.SBASE_MVA
    rate = meta["rate"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(rate > 0, 100.0 * smax / rate, np.nan)
    return smax, pct


def bus_voltages(ss):
    return np.asarray(ss.Bus.v.v, float)
