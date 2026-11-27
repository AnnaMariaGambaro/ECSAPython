"""
Gaussian (Black-Scholes / GBM) model for the barrier-option application.

Up-and-out discretely-monitored call, S0 = K = 100, B = 110, r = 0.05,
sigma = 0.1, T = 1 (Peng, Fu & Hu 2018, Example 3). Supporting module of
barrier_option_gaussian_main.py; formerly vg_barrier_ecsa.py plus the helper
functions of gaussian_table1_v2.py.

ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex): the analytic-strike chi-path proxy f_k(vartheta)
of eq. (fkBarrier) is evaluated at vartheta = B + h e^{i angle} and combined
pathwise with the formula of Table tab:algorithm.
"""
from __future__ import annotations

import numpy as np

from utils.ecsa_utils import (ecsa_phi, mobius_terms, required_angles, angle_key, ecsa_prefactor)
from barrier_option_common import BarrierParams, conv_benchmark, ecsa_points

MODEL_NAME = "gaussian"
S0, K, B0, R_RATE, SIGMA, T = 100.0, 100.0, 110.0, 0.05, 0.10, 1.0


def bp_for(l: int, B: float = B0) -> BarrierParams:
    return BarrierParams(S0=S0, K=K, B=B, r=R_RATE, q=0.0, T=T, l=l)


def characteristic_function(u: np.ndarray, dt: float, r: float, q: float,
                            sigma: float = SIGMA) -> np.ndarray:
    """Characteristic function of the GBM log-return over a step dt."""
    return np.exp(dt * (1j * (r - q - 0.5 * sigma ** 2) * u - 0.5 * sigma ** 2 * u ** 2))


def benchmark(l: int) -> tuple[float, float]:
    """Deterministic CONV benchmark (dC/dB, d2C/dB2) at B0."""
    bp = bp_for(l)
    cf = lambda u, dt: characteristic_function(u, dt, bp.r, bp.q)
    return conv_benchmark(cf, bp.l, bp.B, bp.T, bp.S0, bp.K, bp.r, bp.q)


# ============================================================================
# Simulation and real paths
# ============================================================================


def simulate(N: int, l: int, seed: int) -> tuple[np.ndarray]:
    """Primitive standard normal increments Z of shape (N, l)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N, l)),)


def real_paths(Z: np.ndarray, bp: BarrierParams, sigma: float = SIGMA) -> dict:
    """Real GBM log-price path from primitives Z (centred part, mean, terminal)."""
    dt = bp.dt
    l = Z.shape[1]
    vs = sigma * np.sqrt(dt)
    S_cent = np.cumsum(vs * Z, axis=1)
    j = np.arange(1, l + 1)
    m = np.broadcast_to((bp.r - bp.q - 0.5 * sigma ** 2) * dt * j, Z.shape).copy()
    logST = np.log(bp.S0) + m[:, -1] + S_cent[:, -1]
    ST = np.exp(logST)
    return {"S_cent": S_cent, "m": m, "logST": logST, "ST": ST}


def raw_survival_bound(B: float, S0: float, l: int) -> np.ndarray:
    """Log-price survival boundary ln(B/S0) at each of the l monitoring dates."""
    return np.full(l, np.log(B / S0))


# ============================================================================
# ECSA (chi-path proxy, pathwise combination)
# ============================================================================


def chi_cache(Z: np.ndarray, bp: BarrierParams, sigma: float = SIGMA) -> dict:
    """Every part of the chi-path proxy that does not depend on the complex step c."""
    l = Z.shape[1]
    dt = bp.dt
    vs = sigma * np.sqrt(dt)
    mu = bp.r - bp.q - 0.5 * sigma ** 2
    S = np.cumsum(vs * Z, axis=1)
    j = np.arange(1, l + 1)
    m = mu * dt * j  # (l,), same for every path
    m_full = np.broadcast_to(m, Z.shape)

    b_raw = raw_survival_bound(bp.B, bp.S0, l)
    Bb = b_raw[None, :] - m_full
    k0t = np.log(bp.K / bp.S0) - m[-1]

    surv = np.all(S < Bb, axis=1)
    strike = S[:, -1] > k0t
    alive = (surv & strike).astype(float)

    Bbl = Bb[:, -1]
    denom = Bbl - k0t

    dS_real = np.hstack([S[:, 0:1], np.diff(S, axis=1)])
    Z_real_first = dS_real[:, 0] / vs
    Z_real_last = dS_real[:, -1] / vs if l >= 2 else None

    return dict(
        l=l, vs=vs, alive=alive, Bbl=Bbl, denom=denom, k0t=k0t,
        dS_first=dS_real[:, 0], Z_real_first=Z_real_first,
        S_penultimate=S[:, -2] if l >= 2 else None,
        S_last=S[:, -1], Z_real_last=Z_real_last,
        m_last=m[-1], S0=bp.S0, K=bp.K, r=bp.r, T=bp.T,
        N=Z.shape[0],
    )


def chi_from_cache(cache: dict, c: complex) -> np.ndarray:
    """Per-path realizations f_k(vartheta) of eq. (fkBarrier), with
    c = ln(vartheta) - ln(B) (barrier_option_common.log_barrier_shift): the path
    gamma_vartheta shifts z_1 by c/sigma_1 and rescales the last coordinate by
    the Jacobian R = (ln vartheta - ln K)/(ln B - ln K); only the first and last
    increments change."""
    l, vs, k0t = cache["l"], cache["vs"], cache["k0t"]
    Bbl, denom, alive = cache["Bbl"], cache["denom"], cache["alive"]

    R = (Bbl + c - k0t) / denom
    S_cx_l = k0t + R * (cache["S_last"] - k0t)

    Z_cx_first = (cache["dS_first"] + c) / vs
    exponent = Z_cx_first ** 2 - cache["Z_real_first"] ** 2
    if l >= 2:
        S_cx_last_before_l = cache["S_penultimate"] + c
        dS_cx_last = S_cx_l - S_cx_last_before_l
        Z_cx_last = dS_cx_last / vs
        exponent = exponent + Z_cx_last ** 2 - cache["Z_real_last"] ** 2
    density_ratio = np.exp(-0.5 * exponent)

    logST_cx = np.log(cache["S0"]) + cache["m_last"] + S_cx_l
    ST_cx = np.exp(logST_cx)
    disc = np.exp(-cache["r"] * cache["T"])

    return disc * (ST_cx - cache["K"]) * alive * density_ratio * R


def ecsa_pathwise_derivative(f_values: dict, n: int, M: int, h: float,
                             phi: float) -> np.ndarray:
    """Pathwise ECSA n-th derivative in theta = B from the per-path complex
    evaluations `f_values` (dict: rounded angle -> f_k(B + h e^{i angle}))."""
    N = next(iter(f_values.values())).shape[0]
    total = np.zeros(N, dtype=float)
    for k, mk, angs in mobius_terms(n, M, phi):
        s = np.zeros(N, dtype=float)
        for ang in angs:
            s += np.imag(f_values[angle_key(ang)])
        total += mk * s / (k * n)
    return ecsa_prefactor(n, h, phi) * total


def ecsa_derivative(Z: np.ndarray, bp: BarrierParams, n: int, M: int, h: float,
                    phi: float | None = None) -> tuple[float, float, np.ndarray]:
    """ECSA estimate of d^n C / dB^n from primitives Z, with theta = B and the
    evaluation points vartheta = B + h e^{i(2 pi j/(k n) + phi)} (h in units of B).

    Returns (estimate, pathwise standard error, pathwise contributions).
    """
    if phi is None:
        phi = ecsa_phi(n, M)
    cache = chi_cache(Z, bp)
    f_all = {a: chi_from_cache(cache, c)
             for a, c in ecsa_points(bp.B, h, required_angles(n, M, phi)).items()}
    Y_B = ecsa_pathwise_derivative(f_all, n, M, h, phi)
    N = Y_B.shape[0]
    return float(np.mean(Y_B)), float(np.std(Y_B, ddof=1) / np.sqrt(N)), Y_B


# ============================================================================
# GLR and crude Monte Carlo
# ============================================================================


def glr_raw(Z: np.ndarray, bp: BarrierParams, sigma: float = SIGMA) -> np.ndarray:
    """Pathwise GLR contributions to dC/dB:
    disc * 1{surv, S_T > K} * (S_T - (Z_1/vs)(S_T - K)) / B."""
    real = real_paths(Z, bp, sigma)
    l = Z.shape[1]
    vs = sigma * np.sqrt(bp.dt)
    b_raw = raw_survival_bound(bp.B, bp.S0, l)
    W = real["S_cent"] + real["m"]  # cumulative log-return path, W_j = ln(S_tj/S0)
    surv = np.all(W < b_raw[None, :], axis=1)
    strike = real["ST"] > bp.K
    alive = (surv & strike).astype(float)
    ST = real["ST"]
    disc = np.exp(-bp.r * bp.T)
    g_theta = disc * alive * (ST - (Z[:, 0] / vs) * (ST - bp.K))
    return g_theta / bp.B  # 1/B = d ln(vartheta/B)/d vartheta, the GLR shift of z_1 per unit of B


def crude_raw(Z: np.ndarray, bp: BarrierParams, sigma: float = SIGMA) -> np.ndarray:
    """Pathwise discounted payoff of the up-and-out call (crude Monte Carlo)."""
    real = real_paths(Z, bp, sigma)
    l = Z.shape[1]
    b_raw = raw_survival_bound(bp.B, bp.S0, l)
    surv = np.all(real["S_cent"] < (b_raw[None, :] - real["m"]), axis=1)
    return np.exp(-bp.r * bp.T) * np.maximum(real["ST"] - bp.K, 0.0) * surv


# ============================================================================
# Uniform interface used by barrier_option_common.run_table / run_hsweep
# ============================================================================


def ecsa(P, l: int, B: float, h: float, n: int, M: int) -> float:
    return ecsa_derivative(P[0], bp_for(l, B), n, M, h)[0]


def glr_d1(P, l: int, B: float) -> float:
    return glr_raw(P[0], bp_for(l, B)).mean()


def fd_d1(P, l: int, B: float, delta: float) -> float:
    up = crude_raw(P[0], bp_for(l, B + delta))
    dn = crude_raw(P[0], bp_for(l, B - delta))
    return ((up - dn) / (2 * delta)).mean()


def glr_fd_d2(P, l: int, B: float, delta: float) -> float:
    upg = glr_raw(P[0], bp_for(l, B + delta))
    dng = glr_raw(P[0], bp_for(l, B - delta))
    return ((upg - dng) / (2 * delta)).mean()
