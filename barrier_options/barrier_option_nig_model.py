"""
Normal Inverse Gaussian (NIG) model for the barrier-option application.

Up-and-out discretely-monitored call, S0 = K = 100, B = 130, r = 0.05, T = 1.
NIG shape parameters (alpha, beta, delta) of the AMZN calibration of
Kirkby, Aglieri Rinella & Aguilar (2024), Frontiers of Mathematical Finance
3(3), 400-439 (used in the manuscript); the Schoutens (2003) calibration is
also available. Supporting module of barrier_option_nig_main.py; formerly the
model code duplicated in nig_table8_amzn.py and nig_barrier_h_sweep.py.

The NIG log-return over a step is Gaussian conditional on the inverse-Gaussian
subordinator increment V, so the analytic-strike chi-path proxy of
barrier_option_common applies with conditional standard deviation sqrt(V).
ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex), via barrier_option_common.ecsa_from_proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import invgauss

from utils.ecsa_utils import ecsa_phi
from barrier_option_common import (conv_benchmark, conditional_gaussian_chi_cache,
                                   conditional_gaussian_chi, ecsa_from_proxy)

MODEL_NAME = "nig_amzn"   # "nig_<calibration>", updated by set_calibration
S0 = K = 100.0
B0 = 130.0
R_RATE = 0.05
T = 1.0
DISC = np.exp(-R_RATE * T)
LOGK = np.log(K / S0)


@dataclass(frozen=True)
class NIGCalibration:
    name: str    # filesystem-safe tag used in output filenames
    label: str   # label used in figure titles
    ALPHA: float
    BETA: float
    DELTA: float

    @property
    def gamma(self) -> float:
        return np.sqrt(self.ALPHA ** 2 - self.BETA ** 2)

    @property
    def mu(self) -> float:
        """Risk-neutral drift of the log-price."""
        gp = np.sqrt(self.ALPHA ** 2 - (self.BETA + 1) ** 2)
        return R_RATE + self.DELTA * (gp - self.gamma)


AMZN = NIGCalibration("amzn", "AMZN", 9.873, -4.902, 0.853)
SCHOUTENS = NIGCalibration("schoutens", "Schoutens", 6.1882, -3.8941, 0.1622)
CALIBRATIONS = {c.name: c for c in (AMZN, SCHOUTENS)}

# Active calibration (set by barrier_option_nig_main.py via set_calibration).
CALIB = AMZN


def set_calibration(name: str) -> NIGCalibration:
    global CALIB, MODEL_NAME
    CALIB = CALIBRATIONS[name]
    MODEL_NAME = f"nig_{CALIB.name}"
    return CALIB


def setup(l: int, c: NIGCalibration | None = None):
    """Step size and inverse-Gaussian parameters (mean mu_IG, shape lam) for l dates."""
    c = c or CALIB
    dt = T / l
    mu_IG = c.DELTA * dt / c.gamma
    lam = (c.DELTA * dt) ** 2
    return dt, mu_IG, lam


def characteristic_function(u, dt, c: NIGCalibration | None = None):
    """Characteristic function of the NIG log-return over a step dt."""
    c = c or CALIB
    inner = c.ALPHA ** 2 - (c.BETA + 1j * u) ** 2
    return np.exp(1j * u * c.mu * dt) * np.exp(c.DELTA * dt * (c.gamma - np.sqrt(inner)))


def benchmark(l: int) -> tuple[float, float]:
    """Deterministic CONV benchmark (dC/dB, d2C/dB2) at B0."""
    c = CALIB
    return conv_benchmark(lambda u, dt: characteristic_function(u, dt, c), l, B0, T, S0, K, R_RATE, 0.0)


# ============================================================================
# Simulation, ECSA, GLR, crude Monte Carlo
# ============================================================================


def simulate(N: int, l: int, seed: int):
    """Inverse-Gaussian increments V and standard normals Z, both (N, l)."""
    dt, mu_IG, lam = setup(l)
    rng = np.random.default_rng(seed)
    V = invgauss.rvs(mu_IG / lam, scale=lam, size=(N, l), random_state=rng)
    Z = rng.standard_normal((N, l))
    return V, Z


def chi_cache(V: np.ndarray, Z: np.ndarray, l: int, B: float) -> dict:
    """c-independent part of the chi-path proxy at barrier B."""
    c = CALIB
    dt, _, _ = setup(l)
    MU = c.mu
    sqV = np.sqrt(V)
    Y = c.BETA * V + sqV * Z
    W = np.cumsum(MU * dt + Y, axis=1)
    m = np.cumsum(MU * dt + c.BETA * V, axis=1)
    S = W - m
    return conditional_gaussian_chi_cache(S, m, sqV, np.log(B / S0), LOGK, DISC, K, S0_factor=S0)


def ecsa_d1(V, Z, h, phi, M, l, B):
    cache = chi_cache(V, Z, l, B)
    return ecsa_from_proxy(lambda c: conditional_gaussian_chi(cache, c), B, h, 1, M, phi)


def ecsa_d2(V, Z, h, phi, M, l, B):
    cache = chi_cache(V, Z, l, B)
    return ecsa_from_proxy(lambda c: conditional_gaussian_chi(cache, c), B, h, 2, M, phi)


def _log_price_path(V, Z, l):
    dt, _, _ = setup(l)
    sqV = np.sqrt(V)
    Y = CALIB.BETA * V + sqV * Z
    W = np.cumsum(CALIB.mu * dt + Y, axis=1)
    return sqV, W


def glr_raw(V, Z, l, B):
    """Pathwise GLR contributions to dC/dB (first-step conditional sd sqrt(V_1))."""
    sqV, W = _log_price_path(V, Z, l)
    b = np.full(l, np.log(B / S0))
    surv = np.all(W < b[None, :], axis=1)
    ST = S0 * np.exp(W[:, -1])
    strike = ST > K
    alive = (surv & strike).astype(float)
    vs1 = sqV[:, 0]
    g_theta = DISC * alive * (ST - (Z[:, 0] / vs1) * (ST - K))
    return g_theta / B  # 1/B = d ln(vartheta/B)/d vartheta, the GLR shift of z_1 per unit of B


def crude_raw(V, Z, l, B):
    """Pathwise discounted payoff of the up-and-out call (crude Monte Carlo)."""
    _, W = _log_price_path(V, Z, l)
    b = np.full(l, np.log(B / S0))
    surv = np.all(W < b[None, :], axis=1)
    ST = S0 * np.exp(W[:, -1])
    return DISC * np.maximum(ST - K, 0.0) * surv


# ============================================================================
# Uniform interface used by barrier_option_common.run_table / run_hsweep
# ============================================================================


def ecsa(P, l: int, B: float, h: float, n: int, M: int) -> float:
    fn = ecsa_d1 if n == 1 else ecsa_d2
    return fn(P[0], P[1], h, ecsa_phi(n, M), M, l, B)


def glr_d1(P, l: int, B: float) -> float:
    return glr_raw(P[0], P[1], l, B).mean()


def fd_d1(P, l: int, B: float, delta: float) -> float:
    up = crude_raw(P[0], P[1], l, B + delta)
    dn = crude_raw(P[0], P[1], l, B - delta)
    return ((up - dn) / (2 * delta)).mean()


def glr_fd_d2(P, l: int, B: float, delta: float) -> float:
    upg = glr_raw(P[0], P[1], l, B + delta)
    dng = glr_raw(P[0], P[1], l, B - delta)
    return ((upg - dng) / (2 * delta)).mean()
