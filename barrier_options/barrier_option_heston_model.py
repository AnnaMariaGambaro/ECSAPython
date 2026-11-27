"""
Heston stochastic-volatility model for the barrier-option application.

kappa = 2, theta_v = 0.04, sigma_v = 0.3, rho = -0.7, v0 = 0.04 (Feller:
2 kappa theta_v - sigma_v^2 = 0.07 > 0); S0 = K = 100, B = 120, r = 0.05, T = 1.
Variance path: Andersen (2008) exact scheme (noncentral chi-square), with the
log-price exactly Gaussian conditional on the variance path, so the
analytic-strike chi-path proxy of barrier_option_common applies with the
conditional per-step standard deviation sqrt(K3 v_j + K4 v_{j+1}).
No deterministic benchmark is available for this model.
ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex), via barrier_option_common.ecsa_from_proxy.

Supporting module of barrier_option_heston_main.py; 
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ncx2

from utils.ecsa_utils import ecsa_phi
from barrier_option_common import (conditional_gaussian_chi_cache, conditional_gaussian_chi,
                                   ecsa_from_proxy)

MODEL_NAME = "heston"
KAPPA, THETA_V, SIGMA_V, RHO, V0 = 2.0, 0.04, 0.3, -0.7, 0.04
S0 = K = 100.0
B0 = 120.0
R_RATE = 0.05
T = 1.0
GAMMA1 = GAMMA2 = 0.5
logK = np.log(K)
disc = np.exp(-R_RATE * T)


def setup(l: int):
    """Andersen (2008) exact-scheme constants for dt = T/l."""
    dt = T / l
    K0 = -RHO * KAPPA * THETA_V * dt / SIGMA_V
    K1 = GAMMA1 * dt * (KAPPA * RHO / SIGMA_V - 0.5) - RHO / SIGMA_V
    K2 = GAMMA2 * dt * (KAPPA * RHO / SIGMA_V - 0.5) + RHO / SIGMA_V
    K3 = GAMMA1 * dt * (1 - RHO ** 2)
    K4 = GAMMA2 * dt * (1 - RHO ** 2)
    ncx2_scale = SIGMA_V ** 2 * (1 - np.exp(-KAPPA * dt)) / (4 * KAPPA)
    ncx2_df = 4 * KAPPA * THETA_V / SIGMA_V ** 2
    ncx2_base_lam = 4 * KAPPA * np.exp(-KAPPA * dt) / (SIGMA_V ** 2 * (1 - np.exp(-KAPPA * dt)))
    return dt, K0, K1, K2, K3, K4, ncx2_scale, ncx2_df, ncx2_base_lam


def simulate(N: int, l: int, seed: int):
    """Andersen exact-scheme variance path and log-price path.
    Returns (lg, Zp, v): log-price (N, l+1), orthogonal normals (N, l), variance (N, l+1)."""
    dt, K0, K1, K2, K3, K4, ncx2_scale, ncx2_df, ncx2_base_lam = setup(l)
    rng = np.random.default_rng(seed)
    v = np.empty((N, l + 1))
    v[:, 0] = V0
    lg = np.empty((N, l + 1))
    lg[:, 0] = np.log(S0)
    Zp = rng.standard_normal((N, l))
    for j in range(l):
        lam = ncx2_base_lam * v[:, j]
        v[:, j + 1] = ncx2_scale * ncx2.rvs(df=ncx2_df, nc=lam, size=N, random_state=rng)
        sq = np.sqrt(np.maximum(K3 * v[:, j] + K4 * v[:, j + 1], 0))
        lg[:, j + 1] = lg[:, j] + R_RATE * dt + K0 + K1 * v[:, j] + K2 * v[:, j + 1] + sq * Zp[:, j]
    return lg, Zp, v


def crude_price(lg: np.ndarray, B: float) -> float:
    """Crude Monte Carlo up-and-out call price at barrier B on simulated paths."""
    surv = (lg[:, 1:] < np.log(B)).all(axis=1)
    ST = np.exp(lg[:, -1])
    return float((disc * np.maximum(ST - K, 0.0) * surv).mean())


def _conditional_moments(v: np.ndarray, l: int):
    """Conditional per-step sd and cumulative conditional mean m (incl. ln S0)."""
    dt, K0, K1, K2, K3, K4, *_ = setup(l)
    sd = np.sqrt(np.maximum(K3 * v[:, :-1] + K4 * v[:, 1:], 0))
    mc = R_RATE * dt + K0 + K1 * v[:, :-1] + K2 * v[:, 1:]
    m = np.log(S0) + np.cumsum(mc, axis=1)
    return sd, m


def chi_cache(lg, Zp, v, l, B) -> dict:
    """c-independent part of the chi-path proxy at barrier B."""
    sd, m = _conditional_moments(v, l)
    S = lg[:, 1:] - m
    return conditional_gaussian_chi_cache(S, m, sd, np.log(B), logK, disc, K, S0_factor=None)


def glr_raw(lg, Zp, v, l, B) -> np.ndarray:
    """Pathwise GLR contributions to dC/dB, with the first-step conditional
    volatility sqrt(K3 v0 + K4 v1) in place of the Gaussian step volatility."""
    sd, m = _conditional_moments(v, l)
    S = lg[:, 1:] - m
    bB = np.log(B) - m
    k0t = logK - m[:, -1]
    surv = np.all(S < bB, axis=1)
    strike = S[:, -1] > k0t
    alive = (surv & strike).astype(float)
    ST = np.exp(lg[:, -1])
    vs1 = sd[:, 0]
    g_theta = disc * alive * (ST - (Zp[:, 0] / vs1) * (ST - K))
    return g_theta / B  # 1/B = d ln(vartheta/B)/d vartheta, the GLR shift of z_1 per unit of B


def ecsa_d1(lg, Zp, v, l, B, h, phi, M):
    cache = chi_cache(lg, Zp, v, l, B)
    return ecsa_from_proxy(lambda c: conditional_gaussian_chi(cache, c), B, h, 1, M, phi)


def ecsa_d2(lg, Zp, v, l, B, h, phi, M):
    cache = chi_cache(lg, Zp, v, l, B)
    return ecsa_from_proxy(lambda c: conditional_gaussian_chi(cache, c), B, h, 2, M, phi)


# ============================================================================
# Uniform interface used by barrier_option_common.run_table / run_hsweep
# ============================================================================


def ecsa(P, l: int, B: float, h: float, n: int, M: int) -> float:
    fn = ecsa_d1 if n == 1 else ecsa_d2
    return fn(*P, l, B, h, ecsa_phi(n, M), M)


def glr_d1(P, l: int, B: float) -> float:
    return glr_raw(*P, l, B).mean()


def fd_d1(P, l: int, B: float, delta: float) -> float:
    """Central FD of the crude price (difference of the two sample means)."""
    up = crude_price(P[0], B + delta)
    dn = crude_price(P[0], B - delta)
    return (up - dn) / (2 * delta)


def glr_fd_d2(P, l: int, B: float, delta: float) -> float:
    upg = glr_raw(*P, l, B + delta)
    dng = glr_raw(*P, l, B - delta)
    return ((upg - dng) / (2 * delta)).mean()
