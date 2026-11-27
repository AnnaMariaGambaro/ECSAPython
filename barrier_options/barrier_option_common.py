"""
Shared components of the barrier-option application (Gaussian, NIG, Heston):
sensitivities of a discretely-monitored up-and-out call with respect to the
barrier level B.

Contents
--------
1. `BarrierParams`: contract and market parameters.
2. Deterministic CONV/FFT benchmark of Lord, Fang, Bervoets & Oosterlee (2008)
   for an arbitrary per-step characteristic function, its grid-Richardson
   extrapolation and the central-FD benchmark derivatives (Gaussian, NIG).
3. Analytic-strike path proxy for models whose log-price is conditionally
   Gaussian given the path of a variance/subordinator process (NIG, Heston),
   with the c-independent cache shared by all ECSA evaluation points.
4. ECSA estimators of d^n C / dB^n with the barrier level itself as the
   differentiation parameter, theta = B (Section 4.3 of main-v13.tex): the
   proxy is evaluated at vartheta = B + h e^{i angle} and combined with the
   formula of Table tab:algorithm; no differentiation in ln B and no chain rule.
5. Generic replication drivers for the tables and the vary-h figures, and the
   table/figure export (standardized CSV, h-sweep plot).

The model-specific parts (simulation, GLR, crude price) live in
barrier_option_{gaussian,nig,heston}_model.py; each of those modules exposes the
same interface used by the drivers below:

    simulate(N, l, seed)                 -> tuple of primitives
    ecsa(prims, l, B, h, n, M)           -> ECSA estimate of d^n C / dB^n
    glr_d1(prims, l, B)                  -> GLR estimate of dC/dB
    fd_d1(prims, l, B, delta)            -> central FD of the crude price
    glr_fd_d2(prims, l, B, delta)        -> central FD of the GLR estimator
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.ecsa_utils import (angle_key, mobius_terms, summarize, richardson_estimate,
                        ecsa_phi, ecsa_prefactor, method_row, write_csv, table_columns,
                        richardson_columns, hsweep_columns, table_path, fig_path,
                        l_tag, print_table)

# Candidate Richardson pairs (delta, delta/2), finest first, and their distinct
# deltas in the evaluation order of the original scripts (coarsest first).
RICH_PAIRS = [(0.5, 0.25), (1.0, 0.5), (2.0, 1.0), (4.0, 2.0)]
RICH_DELTAS = [4.0, 2.0, 1.0, 0.5, 0.25]

# ECSA (n, M) configurations, in the order used by the tables and figures.
ECSA_CONFIGS = [(1, 1), (1, 2), (2, 1), (2, 2)]


# ============================================================================
# 1. Parameters
# ============================================================================


@dataclass(frozen=True)
class BarrierParams:
    """Up-and-out discretely-monitored call option and market parameters."""
    S0: float = 100.0
    K: float = 100.0
    B: float = 130.0
    r: float = 0.05
    q: float = 0.0
    T: float = 1.0
    l: int = 10  # number of monitoring dates

    def __post_init__(self):
        if self.B <= self.K:
            raise ValueError("Barrier B must exceed strike K (up-and-out call, B>K).")
        if self.l < 1:
            raise ValueError("Need at least one monitoring date.")

    @property
    def dt(self) -> float:
        return self.T / self.l


# ============================================================================
# 2. Deterministic CONV/FFT benchmark
# ============================================================================


def _frft(x: np.ndarray, a: float) -> np.ndarray:
    """Fractional FFT (Bailey-Swarztrauber / Chourdakis)."""
    m = len(x)
    idx = np.arange(m)
    cfw = np.exp(np.pi * 1j * a * idx ** 2)
    cbw = np.exp(-np.pi * 1j * a * idx ** 2)
    ct = np.exp(np.pi * 1j * a * (np.arange(m, 2 * m) - 2 * m) ** 2)
    blk = np.fft.ifft(np.fft.fft(np.concatenate([x * cbw, np.zeros(m)])) *
                      np.fft.fft(np.concatenate([cfw, ct])))
    return cbw * blk[:m]


def conv_price_uoc(B: float, l: int, T: float, S0: float, K: float, r: float, q: float,
                   char_fn: Callable[[np.ndarray, float], np.ndarray],
                   N_grid: int = 2 ** 16, ymax: float = 7.0, umax0: float = 50.0) -> float:
    """Deterministic price of the discretely-monitored up-and-out call via the
    CONV method of Lord, Fang, Bervoets & Oosterlee (2008), for an arbitrary
    per-step characteristic function `char_fn(u, dt)` of the log-return.
    Independent of the Monte Carlo machinery (no shared random numbers)."""
    dt = T / l
    N = N_grid
    ymin = -ymax
    dy = (ymax - ymin) / N
    ylower = np.log(B) + np.fix((ymin - np.log(B)) / dy) * dy
    y = ylower + np.arange(N) * dy

    umax = umax0
    while abs(char_fn(np.array([umax]), dt)[0]) >= 1e-5:
        umax += 20.0
    du = 2 * umax / N
    u = (np.arange(N) - 0.5 * N) * du
    alive = (np.exp(y) < B).astype(float)

    def ft(p):
        w = np.ones(N)
        w[0] = 0.5
        w[-1] = 0.5
        g = _frft(np.exp(1j * u[0] * y) * p * w, -du * dy / (2 * np.pi))
        return g * np.exp(1j * (u - u[0]) * y[0]) * dy

    def inv(cf, p, x=None):
        if x is None:
            x = y
        g = _frft(np.exp(-1j * u * x[0]) * cf * ft(p), du * dy / (2 * np.pi))
        return g * np.exp(-1j * u[0] * (x - x[0])) * du / (2 * np.pi)

    cf = char_fn(-u, dt)
    pt = np.maximum(np.exp(y) - K, 0) * alive
    qt = inv(cf, pt)
    for _ in range(l - 2):
        pt = qt * alive
        qt = inv(cf, pt)
    pt = qt * alive

    xl = np.log(S0) + np.fix((ymin - np.log(S0)) / dy) * dy
    xp = xl + np.arange(N) * dy
    qp = np.exp(-r * T) * inv(cf, pt, xp)
    return float(qp[int(np.fix((np.log(S0) - xl) / dy))].real)


def richardson_conv_price(B: float, l: int, T: float, S0: float, K: float, r: float,
                          q: float, char_fn) -> float:
    """Grid-Richardson extrapolated CONV price 2 P(2^16) - P(2^15)."""
    p1 = conv_price_uoc(B, l, T, S0, K, r, q, char_fn, N_grid=2 ** 15)
    p2 = conv_price_uoc(B, l, T, S0, K, r, q, char_fn, N_grid=2 ** 16)
    return 2 * p2 - p1


def fd_derivatives_from_pricer(price_fn: Callable[[float], float], B0: float,
                               delta: float) -> tuple[float, float]:
    """Central finite-difference first and second derivative of a deterministic
    pricer `price_fn(B)` at B0 (benchmark only)."""
    pu, p0, pd = price_fn(B0 + delta), price_fn(B0), price_fn(B0 - delta)
    d1 = (pu - pd) / (2 * delta)
    d2 = (pu - 2 * p0 + pd) / delta ** 2
    return d1, d2


def conv_benchmark(char_fn, l: int, B0: float, T: float, S0: float, K: float,
                   r: float, q: float = 0.0, delta: float = 0.10) -> tuple[float, float]:
    """Benchmark (dC/dB, d2C/dB2) at B0: central FD with step `delta` of the
    Richardson-extrapolated CONV price."""
    return fd_derivatives_from_pricer(
        lambda B: richardson_conv_price(B, l, T, S0, K, r, q, char_fn), B0, delta=delta)


# ============================================================================
# 3. Analytic-strike chi-path proxy for conditionally Gaussian log-prices
# ============================================================================


def conditional_gaussian_chi_cache(S: np.ndarray, m: np.ndarray, sd: np.ndarray,
                                   log_b: float, log_k: float, disc: float, K: float,
                                   S0_factor: float | None = None) -> dict:
    """Every part of the chi-path proxy that does not depend on the complex step c.

    S  : (N, l) centred log-price path (log-price minus its conditional mean m);
    m  : (N, l) conditional mean path;
    sd : (N, l) conditional per-step standard deviations;
    log_b, log_k : barrier and strike in the log coordinates of S + m;
    S0_factor : spot multiplier for the terminal price when m excludes ln S0
                (NIG), or None when m already includes ln S0 (Heston).
    """
    Bb = log_b - m
    k0t = log_k - m[:, -1]
    surv = np.all(S < Bb, axis=1).astype(float)
    strike = (S[:, -1] > k0t).astype(float)
    Bbl = Bb[:, -1]
    dS = np.hstack([S[:, 0:1], np.diff(S, axis=1)])
    return dict(
        surv=surv, strike=strike, Bbl=Bbl, k0t=k0t,
        S_penultimate=S[:, -2], S_last=S[:, -1],
        dS_first=dS[:, 0], sd_first=sd[:, 0],
        Zr_first=dS[:, 0] / sd[:, 0],
        sd_last=sd[:, -1], Zr_last=dS[:, -1] / sd[:, -1],
        m_last=m[:, -1], disc=disc, K=K, S0_factor=S0_factor,
    )


def conditional_gaussian_chi(cache: dict, c: complex) -> np.ndarray:
    """Per-path realizations f_k(vartheta) of eq. (fkBarrier), with
    c = ln(vartheta) - ln(B) (see `log_barrier_shift`): first coordinate of the
    path shifted by c/sigma_1, last coordinate rescaled by the Jacobian
    R = (ln vartheta - ln K)/(ln B - ln K). Only the first and last monitoring
    dates' increments change; the middle columns cancel exactly in the
    density-ratio exponent."""
    k0t, Bbl = cache["k0t"], cache["Bbl"]
    R = (Bbl + c - k0t) / (Bbl - k0t)
    Scx_lm1 = cache["S_penultimate"] + c
    Scx_l = k0t + R * (cache["S_last"] - k0t)

    Zcx_first = (cache["dS_first"] + c) / cache["sd_first"]
    Zcx_last = (Scx_l - Scx_lm1) / cache["sd_last"]
    exponent = (Zcx_first ** 2 - cache["Zr_first"] ** 2
                + Zcx_last ** 2 - cache["Zr_last"] ** 2)
    DR = np.exp(-0.5 * exponent)

    ST_cx = np.exp(cache["m_last"] + Scx_l)
    if cache["S0_factor"] is not None:
        ST_cx = cache["S0_factor"] * ST_cx
    return cache["disc"] * (ST_cx - cache["K"]) * cache["strike"] * cache["surv"] * DR * R


# ============================================================================
# 4. ECSA derivatives with respect to the barrier level, theta = B
# ============================================================================
#
# Section 4.3 of main-v13.tex: theta = B and the evaluation points of Table
# tab:algorithm are vartheta = B + h e^{i(2 pi j/(k n) + phi)}. The complex path
# gamma_vartheta and the Jacobian (ln vartheta - ln K)/(ln B - ln K) depend on
# vartheta only through ln vartheta, so every proxy below is a function of
# c = ln(vartheta) - ln(B), obtained from vartheta by `log_barrier_shift`.
# The n-th derivative in B is then read off directly with the prefactor
# n!/(h^n sin(n phi)).

# Path used for the complex evaluation points:
#   "exact": gamma_vartheta of Section 4.3, c = ln(vartheta/B) (default);
#   "jet2" : expanded path gamma^[2] (Section subsec:expansion, Proposition
#            prop:jet), ln vartheta replaced by its second-order Taylor
#            polynomial at B, c = eps/B - eps^2/(2 B^2), eps = vartheta - B.
ECSA_PATHS = ("exact", "jet2")
ECSA_PATH = "exact"


def set_ecsa_path(path: str) -> None:
    global ECSA_PATH
    if path not in ECSA_PATHS:
        raise ValueError(f"unknown ECSA path {path!r}; choose from {ECSA_PATHS}")
    ECSA_PATH = path


def log_barrier_shift(vartheta: complex, B: float, path: str | None = None) -> complex:
    """c = ln(vartheta) - ln(B) for a complex barrier level vartheta near B.

    "exact": principal complex logarithm of vartheta/B = 1 + x, computed as
    ln|1+x| = log1p(2 Re x + |x|^2)/2 and arg(1+x) = atan2(Im x, 1 + Re x),
    which keeps full relative accuracy when |x| = h/B is tiny.
    "jet2": second-order Taylor polynomial x - x^2/2.
    """
    path = path or ECSA_PATH
    x = (vartheta - B) / B
    if path == "jet2":
        return x - 0.5 * x * x
    return complex(0.5 * np.log1p(2.0 * x.real + abs(x) ** 2), np.arctan2(x.imag, 1.0 + x.real))


def ecsa_points(B: float, h: float, angles) -> dict:
    """{rounded angle: c} for the evaluation points vartheta = B + h e^{i angle}."""
    return {a: log_barrier_shift(B + h * np.exp(1j * a), B)
            for a in sorted({angle_key(a) for a in angles})}


def ecsa_from_proxy(chi: Callable[[complex], np.ndarray], B: float, h: float,
                    n: int, M: int, phi: float) -> float:
    """ECSA estimate of d^n C / dB^n (Table tab:algorithm with theta = B):
        n!/(h^n sin(n phi)) sum_k mu(k) (1/(k n)) sum_j Im fhat(B + h e^{i(2 pi j/(k n) + phi)}),
    fhat(vartheta) = sample mean of chi(c(vartheta)). Points shared by several
    Moebius terms are evaluated once."""
    terms = mobius_terms(n, M, phi)
    pts = ecsa_points(B, h, (a for _, _, angs in terms for a in angs))
    vals = {a: np.imag(chi(c).mean()) for a, c in pts.items()}
    out = sum(mk * sum(vals[angle_key(a)] for a in angs) / (kk * n) for kk, mk, angs in terms)
    return ecsa_prefactor(n, h, phi) * out


# ============================================================================
# 5a. Table driver
# ============================================================================


def run_table(model, l: int, B0: float, N: int, reps: int, h1: float, h2: float,
              deltas=RICH_DELTAS, with_fd: bool = True) -> dict:
    """Replication loop of the barrier tables, for one monitoring frequency l.

    Within each replication the primitives are simulated once (seed s) and
    shared by all methods (common random numbers); the simulation is not
    billed to any method. Each method's time is the wall-clock time of its
    own evaluation, summed over replications.
    """
    keys = ["d1m1", "d1m2", "glr", "d2m1", "d2m2"]
    vals = {k: [] for k in keys}
    tt = {k: 0.0 for k in keys}
    fd_vals = {g: [] for g in deltas}
    t_fd = {g: 0.0 for g in deltas}
    glrfd_vals = {g: [] for g in deltas}
    t_glrfd = {g: 0.0 for g in deltas}

    for s in range(reps):
        P = model.simulate(N, l, seed=s)
        steps = [("d1m1", lambda: model.ecsa(P, l, B0, h1, 1, 1)),
                 ("d1m2", lambda: model.ecsa(P, l, B0, h1, 1, 2)),
                 ("glr", lambda: model.glr_d1(P, l, B0)),
                 ("d2m1", lambda: model.ecsa(P, l, B0, h2, 2, 1)),
                 ("d2m2", lambda: model.ecsa(P, l, B0, h2, 2, 2))]
        for k, fn in steps:
            t0 = time.perf_counter()
            vals[k].append(fn())
            tt[k] += time.perf_counter() - t0

        for g in deltas:
            if with_fd:
                t0 = time.perf_counter()
                fd_vals[g].append(model.fd_d1(P, l, B0, g))
                t_fd[g] += time.perf_counter() - t0

            t0 = time.perf_counter()
            glrfd_vals[g].append(model.glr_fd_d2(P, l, B0, g))
            t_glrfd[g] += time.perf_counter() - t0

    out = {k: summarize(vals[k]) + (tt[k],) for k in keys}
    out["raw"] = vals
    if with_fd:
        m, se, t, pair, stats = richardson_estimate(fd_vals, t_fd, RICH_PAIRS)
        out["fd_rich"], out["fd_rich_pair"], out["fd_all_pairs"] = (m, se, t), pair, stats
    m, se, t, pair, stats = richardson_estimate(glrfd_vals, t_glrfd, RICH_PAIRS)
    out["glrfd_rich"], out["glrfd_rich_pair"], out["glrfd_all_pairs"] = (m, se, t), pair, stats
    return out


def table_rows(res: dict, model_name: str, l: int, N: int, reps: int, h1: float, h2: float,
               bench: tuple | None) -> list[dict]:
    """Standardized rows (one per derivative order and method) for one l."""
    d1b, d2b = bench if bench is not None else (None, None)
    common = dict(application="barrier_option", model=model_name, l=l)
    mc = dict(N=N, replications=reps)
    rows = []
    if bench is not None:
        rows.append(method_row(1, "Benchmark", d1b))
    for key, M in (("d1m1", 1), ("d1m2", 2)):
        m, se, t = res[key]
        rows.append(method_row(1, f"ECSA M={M}", m, se, t, d1b, ecsa_M=M, h=h1, **mc))
    m, se, t = res["glr"]
    rows.append(method_row(1, "GLR", m, se, t, d1b, **mc))
    if "fd_rich" in res:
        m, se, t = res["fd_rich"]
        rows.append(method_row(1, "FD (Rich.)", m, se, t, d1b, fd_pair=res["fd_rich_pair"], **mc))
    if bench is not None:
        rows.append(method_row(2, "Benchmark", d2b))
    for key, M in (("d2m1", 1), ("d2m2", 2)):
        m, se, t = res[key]
        rows.append(method_row(2, f"ECSA M={M}", m, se, t, d2b, ecsa_M=M, h=h2, **mc))
    m, se, t = res["glrfd_rich"]
    rows.append(method_row(2, "GLR+FD (Rich.)", m, se, t, d2b, fd_pair=res["glrfd_rich_pair"], **mc))
    return [dict(common, **r) for r in rows]


def richardson_rows(res: dict, model_name: str, l: int) -> list[dict]:
    """All four candidate Richardson pairs of each FD-type method, with the selected one flagged."""
    rows = []
    for order, method, key in ((1, "FD (Rich.)", "fd"), (2, "GLR+FD (Rich.)", "glrfd")):
        if f"{key}_all_pairs" not in res:
            continue
        for m, se, pair in res[f"{key}_all_pairs"]:
            rows.append(dict(application="barrier_option", model=model_name, l=l,
                             derivative_order=order, method=method,
                             fd_delta=pair[0], fd_delta_half=pair[1], estimate=m, se=se,
                             selected=pair == res[f"{key}_rich_pair"]))
    return rows


def export_table(prefix: str, model_name: str, l: int, res: dict, N: int, reps: int,
                 h1: float, h2: float, bench: tuple | None, title: str) -> list[dict]:
    """Write <prefix>_l<l>_table.csv and <prefix>_l<l>_table_richardson_pairs.csv."""
    rows = table_rows(res, model_name, l, N, reps, h1, h2, bench)
    write_csv(table_path(f"{prefix}_{l_tag(l)}_table.csv"), table_columns(["l"]), rows)
    write_csv(table_path(f"{prefix}_{l_tag(l)}_table_richardson_pairs.csv"),
              richardson_columns(["l"]), richardson_rows(res, model_name, l))
    print_table(title, rows, ["l"])
    return rows


# ============================================================================
# 5b. Vary-h figure driver
# ============================================================================


def run_hsweep(model, l: int, B0: float, N: int, nrep: int, h_vals,
               reuse_paths_across_h: bool) -> dict:
    """ECSA estimates (mean, SE over `nrep` replications) for every (n, M) in
    ECSA_CONFIGS and every h in `h_vals`.

    reuse_paths_across_h=False reproduces the Gaussian/NIG figure scripts:
    the primitives of seed s are re-simulated for every (n, M, h).
    reuse_paths_across_h=True reproduces the Heston figure script: each seed is
    simulated once and reused for all h and (n, M). Both give identical numbers
    (same seed, same draws); only the run time differs.
    """
    vals = {(cfg, ih): [] for cfg in ECSA_CONFIGS for ih in range(len(h_vals))}
    if reuse_paths_across_h:
        for s in range(nrep):
            P = model.simulate(N, l, seed=s)
            for ih, h in enumerate(h_vals):
                for (n, M) in ECSA_CONFIGS:
                    vals[((n, M), ih)].append(model.ecsa(P, l, B0, h, n, M))
    else:
        for (n, M) in ECSA_CONFIGS:
            for ih, h in enumerate(h_vals):
                for s in range(nrep):
                    P = model.simulate(N, l, seed=s)
                    vals[((n, M), ih)].append(model.ecsa(P, l, B0, h, n, M))
    out = {}
    for cfg in ECSA_CONFIGS:
        stats = [summarize(vals[(cfg, ih)]) for ih in range(len(h_vals))]
        out[cfg] = (np.array([s[0] for s in stats]), np.array([s[1] for s in stats]))
    return out


BARRIER_RC = {
    "mathtext.fontset": "cm", "font.family": "serif", "font.size": 12,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "legend.fontsize": 10,
}
COLORS = {"exact": "k", "e1": "#d62728", "e2": "#2ca02c"}
KSE = 3.0


def plot_hsweep(powers, sweep: dict, n: int, benchmark: float | None, title: str, out) -> None:
    """Vary-h figure of one derivative order: benchmark (if any), ECSA M=1
    (red circles), ECSA M=2 (green squares), dashed +-3 SE bands."""
    matplotlib.rcParams.update(BARRIER_RC)
    m1, s1 = sweep[(n, 1)]
    m2, s2 = sweep[(n, 2)]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if benchmark is not None:
        ax.axhline(benchmark, color=COLORS["exact"], ls="-", lw=1.2, label="Benchmark")
    ax.plot(powers, m2, color=COLORS["e2"], ls="-", lw=1.5, marker="s", ms=5, label=r"ECSA $M=2$")
    ax.plot(powers, m2 - KSE * s2, color=COLORS["e2"], ls="--", lw=0.9)
    ax.plot(powers, m2 + KSE * s2, color=COLORS["e2"], ls="--", lw=0.9)
    ax.plot(powers, m1, color=COLORS["e1"], ls="-", lw=1.5, marker="o", ms=4, label=r"ECSA $M=1$")
    ax.plot(powers, m1 - KSE * s1, color=COLORS["e1"], ls="--", lw=0.9)
    ax.plot(powers, m1 + KSE * s1, color=COLORS["e1"], ls="--", lw=0.9)
    ax.set_xlabel(r"$\log_{10} h$"); ax.set_ylabel(rf"$C_B^{{({n})}}$")
    ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(ncol=2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def export_hsweep(prefix: str, model_name: str, l: int, powers, h_vals, sweep: dict,
                  bench: tuple | None, N: int, nrep: int, titles: tuple[str, str]) -> None:
    """Write the two figures <prefix>_l<l>_hsweep_der{1,2}.pdf and the plotted
    data <prefix>_l<l>_hsweep_data.csv."""
    for n in (1, 2):
        plot_hsweep(powers, sweep, n, None if bench is None else bench[n - 1], titles[n - 1],
                    fig_path(f"{prefix}_{l_tag(l)}_hsweep_der{n}.pdf"))
    rows = []
    for n in (1, 2):
        if bench is not None:
            rows.append(dict(application="barrier_option", model=model_name, l=l,
                             derivative_order=n, method="Benchmark", estimate=bench[n - 1]))
        for M in (1, 2):
            means, ses = sweep[(n, M)]
            for p, h, mval, sval in zip(powers, h_vals, means, ses):
                rows.append(dict(application="barrier_option", model=model_name, l=l,
                                 derivative_order=n, method=f"ECSA M={M}", ecsa_M=M,
                                 ecsa_phi=ecsa_phi(n, M), log10_h=p, h=h,
                                 estimate=mval, se=sval, N=N, replications=nrep))
    write_csv(fig_path(f"{prefix}_{l_tag(l)}_hsweep_data.csv"), hsweep_columns(["l"]), rows)


# ============================================================================
# 6. Command-line driver shared by the three barrier main scripts
# ============================================================================


@dataclass(frozen=True)
class TableConfig:
    N: int
    reps: int
    h1: float          # ECSA step, first derivative
    h2: float          # ECSA step, second derivative


@dataclass(frozen=True)
class FigureConfig:
    N: int
    nrep: int
    powers: tuple      # log10 h grid
    reuse_paths_across_h: bool


def run_barrier_main(model, prefix: str, table_cfg: TableConfig, figure_cfg: FigureConfig,
                     titles, description: str, argv=None, extra_args=None) -> None:
    """Parse `{table,figure,all} [--l ...]` and run the requested experiments.

    titles(l) -> (first-derivative title, second-derivative title) of the figures.
    extra_args(parser) may add model-specific options; it must return a callback
    applied to the parsed arguments before running (or None).
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("experiment", choices=["table", "figure", "all"],
                        help="table: Table of the manuscript; figure: vary-h figures; all: both")
    parser.add_argument("--l", type=int, nargs="+", default=[10, 20, 30],
                        help="monitoring frequencies (default: 10 20 30)")
    parser.add_argument("--path", choices=ECSA_PATHS, default="exact",
                        help="ECSA complex path: exact gamma_vartheta (default) or its "
                             "second-order jet gamma^[2]; non-default paths add '_jet2' "
                             "to the output names")
    apply_extra = extra_args(parser) if extra_args else None
    args = parser.parse_args(argv)
    if apply_extra:
        prefix = apply_extra(args, prefix)
    set_ecsa_path(args.path)
    if args.path != "exact":
        prefix = f"{prefix}_{args.path}"

    has_bench = hasattr(model, "benchmark")
    t_start = time.time()
    if args.experiment in ("table", "all"):
        tc = table_cfg
        for l in args.l:
            t_l0 = time.perf_counter()
            bench = model.benchmark(l) if has_bench else None
            res = run_table(model, l, model.B0, tc.N, tc.reps, tc.h1, tc.h2)
            print(f"l={l} table done in {time.perf_counter() - t_l0:.1f}s  "
                  f"fd_rich_pair={res['fd_rich_pair']}  glrfd_rich_pair={res['glrfd_rich_pair']}",
                  flush=True)
            export_table(prefix, model.MODEL_NAME, l, res, tc.N, tc.reps, tc.h1, tc.h2, bench,
                         f"{prefix}: B={model.B0}, l={l}, N={tc.N}, {tc.reps} reps, "
                         f"theta=B, ECSA path={ECSA_PATH}, h1={tc.h1}, h2={tc.h2} (units of B), "
                         f"Richardson pairs={RICH_PAIRS}")
    if args.experiment in ("figure", "all"):
        fc = figure_cfg
        powers = np.array(fc.powers)
        h_vals = 10.0 ** powers
        for l in args.l:
            bench = model.benchmark(l) if has_bench else None
            sweep = run_hsweep(model, l, model.B0, fc.N, fc.nrep, h_vals, fc.reuse_paths_across_h)
            print(f"l={l} h-sweep done [{time.time() - t_start:.0f}s]", flush=True)
            export_hsweep(prefix, model.MODEL_NAME, l, powers, h_vals, sweep, bench,
                          fc.N, fc.nrep, titles(l))
