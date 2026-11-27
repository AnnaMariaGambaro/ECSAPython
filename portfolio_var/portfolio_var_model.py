"""
Portfolio value-at-risk sensitivities (Hong 2009, Sec. 7.1): model and estimators.

R = mu + C Z, Z ~ N(0, I_3), C C' = Sigma; portfolio return R_p = w'R with
w = (0.2, 0.3, 0.5); parameter theta = w_3; q_alpha is the lower alpha-quantile
of R_p. Supporting module of portfolio_var_main.py; formerly the model code of
var_table_v2.py and var_fig4_chunk.py.

Estimators of dq/dw3 and d2q/dw3^2
----------------------------------
* ECSA via the implicit-function route: q' = -F_theta/F_v and
  q'' = -(F_thth F_v^2 - 2 F_vth F_th F_v + F_vv F_th^2)/F_v^3, with F_theta,
  F_thetatheta estimated by ECSA on the complex-path CDF integrand and the
  density terms F_v, F_vv, F_vth in closed form.
  - `ecsa_reference`: literal implementation (used by the vary-h figure);
  - `ecsa_d1_fast`/`ecsa_d2_fast`: algebraically identical, faster evaluation
    (used by the table; see the comment block before `ecsa_cache`).
* Hong (2009) window estimator; GLR of Peng, Fu, Hu & Heidergott (2018).
* FD (Rich.), Hong+FD (Rich.), GLR+FD (Rich.): central differences at
  w_3 +/- d on common random numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from utils.ecsa_utils import mobius, ecsa_phi, ecsa_angles, richardson_deltas

MODEL_NAME = "hong2009"
mu = np.array([0.06, 0.15, 0.25]); s = np.array([0.02, 0.10, 0.22])
Corr = np.array([[1, -0.3, -0.2], [-0.3, 1, 0.2], [-0.2, 0.2, 1]])
Sigma = np.diag(s) @ Corr @ np.diag(s); Q = np.array([0.2, 0.3, 0.5]); jj = 2
C = np.linalg.cholesky(Sigma); mu_p = mu @ Q; Sig2_p = Q @ Sigma @ Q
Sinv = np.linalg.inv(Sigma); w = C.T @ Q; sig_p = np.sqrt(Sig2_p)
S33 = Sinv[jj, jj]

# Richardson grid: barrier grid rescaled to theta = w_3 = 0.5 by 1/200, finest first.
RICH_PAIRS = [(0.0025, 0.00125), (0.005, 0.0025), (0.01, 0.005), (0.02, 0.01)]
RICH_DELTAS = richardson_deltas(RICH_PAIRS)


def exact(alpha):
    """Exact (q_alpha, dq/dw3, d2q/dw3^2) for the Gaussian portfolio."""
    za = norm.ppf(alpha); VaR = mu_p + za * np.sqrt(Sig2_p)
    d1 = (mu + za * Sigma @ Q / np.sqrt(Sig2_p))[jj]
    Hm = za * (Sigma / np.sqrt(Sig2_p) - np.outer(Sigma @ Q, Sigma @ Q) / Sig2_p ** 1.5)
    return VaR, d1, Hm[jj, jj]


def sample(alpha, M, seed, use_is):
    """Returns R (3, M), R_p = Q'R, likelihood ratios LR (ones under plain MC).
    IS: mean shift d = (q_alpha - mu_p)/sigma_p^2 C'w using the exact q_alpha."""
    rng = np.random.default_rng(seed)
    if use_is:
        za = norm.ppf(alpha); vg = mu_p + za * np.sqrt(Sig2_p); dvec = (vg - mu_p) / Sig2_p * w
        Z = rng.standard_normal((3, M)) + dvec[:, None]; LR = np.exp(-dvec @ Z + 0.5 * dvec @ dvec)
    else:
        Z = rng.standard_normal((3, M)); LR = np.ones(M)
    R = mu[:, None] + C @ Z; Rp = Q @ R
    return R, Rp, LR


def quantile(Rp, LR, alpha, use_is):
    """Empirical alpha-quantile (LR-weighted under IS)."""
    if use_is:
        o = np.argsort(Rp); cw = np.cumsum(LR[o]) / np.sum(LR); return Rp[o][np.searchsorted(cw, alpha)]
    return np.quantile(Rp, alpha)


def density_terms(VaR):
    """Closed-form Gaussian F_v, F_vv, F_v,theta at the estimated quantile."""
    Fy = norm.pdf(VaR, mu_p, np.sqrt(Sig2_p))
    Fyy = -Fy * (VaR - mu_p) / Sig2_p
    grad = Fy * ((VaR - mu_p) * mu / Sig2_p + ((VaR - mu_p) ** 2 - Sig2_p) * Sigma @ Q / Sig2_p ** 2)
    return Fy, Fyy, grad[jj]


def _second_order(Fy, Fyy, Fyx, Fx, Fxx):
    return -(Fxx * Fy ** 2 - 2 * Fyx * Fx * Fy + Fyy * Fx ** 2) / Fy ** 3


# ============================================================================
# ECSA -- reference (literal) implementation
# ============================================================================


def lr(Rcx, R):
    """Gaussian density ratio p(Rcx)/p(R) of complex and real samples."""
    dcx = Rcx - mu[:, None]; dr = R - mu[:, None]
    return np.exp(-0.5 * np.einsum('ik,ij,jk->k', dcx, Sinv, dcx)
                  + 0.5 * np.einsum('ik,ij,jk->k', dr.astype(complex), Sinv, dr.astype(complex)))


def ccdf(a, R, h, phi):
    """Per-sample complex-path CDF integrand f_l(vartheta), vartheta = w_3 + h e^{i phi}:
    1{R_3 < b~(theta)} * p(gamma_vartheta(R)) / p(R), gamma = translation of R_3 by
    b~(vartheta) - b~(theta), b~ = (a - w_1 R_1 - w_2 R_2)/w_3, Jacobian 1."""
    Qj = Q[jj]; Qu = Qj + h * np.exp(1j * phi); mask = np.ones(3, bool); mask[jj] = False
    base = a - Q[mask] @ R[mask, :]; a_t = base / Qj; a_c = base / Qu; Rj = R[jj, :]
    ind = (Rj < a_t).astype(float); Rcx = R.astype(complex).copy()
    Rcx[jj, :] = Rj + (np.real(a_c) - a_t) + 1j * np.imag(a_c); return ind * lr(Rcx, R)


def F_theta(VaR, R, LR, h, Mo):
    """ECSA n=1: 1/(h sin phi) sum_k mu(k) (1/k) sum_{j<k} Im f(theta + h e^{i(2 pi j/k + phi)})."""
    if Mo == 1:
        return np.mean(np.imag(ccdf(VaR, R, h, np.pi / 2)) * LR) / h
    phi = ecsa_phi(1, Mo); out = 0.0
    for kk in range(1, Mo + 1):
        mk = mobius(kk)
        if mk == 0: continue
        ssum = sum(np.imag(ccdf(VaR, R, h, a)) * LR for a in ecsa_angles(1, kk, phi))
        out += mk * np.mean(ssum) / kk
    return out / (h * np.sin(phi))


def F_thetatheta(VaR, R, LR, h, Mo):
    """ECSA n=2: 2/(h^2 sin 2phi) sum_k mu(k) (1/2k) sum_{j<2k} Im f(theta + h e^{i(2 pi j/(2k) + phi)})."""
    phi = ecsa_phi(2, Mo); out = 0.0
    for kk in range(1, Mo + 1):
        mk = mobius(kk)
        if mk == 0: continue
        ssum = sum(np.imag(ccdf(VaR, R, h, a)) * LR for a in ecsa_angles(2, kk, phi))
        out += mk * np.mean(ssum) / (2 * kk)
    return out * 2.0 / (h ** 2 * np.sin(2 * phi))


def ecsa_reference(R, LR, VaR, h, Mo, want2):
    """Literal ECSA VaR sensitivities at the estimated quantile VaR:
    returns (dq/dw3, d2q/dw3^2 or None)."""
    Fy = norm.pdf(VaR, mu_p, np.sqrt(Sig2_p))
    Fx = F_theta(VaR, R, LR, h, Mo)
    d1 = -Fx / Fy
    if not want2:
        return d1, None
    Fxx = F_thetatheta(VaR, R, LR, h, Mo)
    _, Fyy, Fyx = density_terms(VaR)
    return d1, _second_order(Fy, Fyy, Fyx, Fx, Fxx)


# ============================================================================
# ECSA -- computationally optimized evaluation of the SAME estimator.
#
# Same per-sample integrand f_l(vartheta) = 1{R_3 < b~(theta)} p(gamma(R))/p(R),
# same angles, M, h and normalizations; only the arithmetic is organized
# differently:
#  (1) exact algebraic density ratio: only R_3 is shifted by Delta, so
#      p(gamma(R))/p(R) = exp(-Delta s_3 - Delta^2 (Sigma^-1)_33 / 2),
#      s_3 = (Sigma^-1 (R - mu))_3;
#  (2) only samples with R_3 < b~(theta) are processed (f_l = 0 otherwise),
#      their sum being divided by the full N;
#  (3) angle-independent quantities are computed once per (sample, VaR);
#  (4) evaluation points shared by several Moebius terms are evaluated once;
#  (5) Im f is computed in real arithmetic: with Delta = x + i y,
#      Re z = -(x s_3 + S_33 (x^2 - y^2)/2), Im z = -(y s_3 + S_33 x y),
#      Im e^z = e^{Re z} sin(Im z).
# The table uses this version; `verify_ecsa` compares it with the reference.
# ============================================================================


def ecsa_cache(VaR, R, LR):
    """Angle-independent quantities for one (sample, VaR); same indicator and b~ as ccdf."""
    Qj = Q[jj]; mask = np.ones(3, bool); mask[jj] = False
    base = VaR - Q[mask] @ R[mask, :]
    a_t = base / Qj
    idx = R[jj, :] < a_t                                  # indicator of ccdf
    Rs = R[:, idx]
    return {'base': base[idx], 'a_t': a_t[idx], 'LR': LR[idx],
            's3': Sinv[jj] @ (Rs - mu[:, None]), 'N': R.shape[1], 'ev': {}}


def ecsa_im_mean(c, h, phi):
    """(1/N) sum_l Im f_l(w_3 + h e^{i phi}) * LR_l  (== np.mean(np.imag(ccdf(...)) * LR))."""
    key = (h, phi)
    if key in c['ev']:
        return c['ev'][key]
    Qu = Q[jj] + h * np.exp(1j * phi)
    a_c = c['base'] / Qu
    x = np.real(a_c) - c['a_t']; y = np.imag(a_c)         # Delta = b~(vartheta) - b~(theta)
    s3 = c['s3']
    zr = -(x * s3 + 0.5 * S33 * (x * x - y * y))
    zi = -(y * s3 + S33 * x * y)
    val = np.sum(np.exp(zr) * np.sin(zi) * c['LR']) / c['N']
    c['ev'][key] = val
    return val


def F_theta_fast(c, h, Mo):
    if Mo == 1:
        return ecsa_im_mean(c, h, np.pi / 2) / h
    phi = ecsa_phi(1, Mo); out = 0.0
    for kk in range(1, Mo + 1):
        mk = mobius(kk)
        if mk == 0: continue
        ssum = sum(ecsa_im_mean(c, h, a) for a in ecsa_angles(1, kk, phi))
        out += mk * ssum / kk
    return out / (h * np.sin(phi))


def F_thetatheta_fast(c, h, Mo):
    phi = ecsa_phi(2, Mo); out = 0.0
    for kk in range(1, Mo + 1):
        mk = mobius(kk)
        if mk == 0: continue
        ssum = sum(ecsa_im_mean(c, h, a) for a in ecsa_angles(2, kk, phi))
        out += mk * ssum / (2 * kk)
    return out * 2.0 / (h ** 2 * np.sin(2 * phi))


def ecsa_d1_fast(R, Rp, LR, alpha, use_is, Mo, h):
    VaR = quantile(Rp, LR, alpha, use_is)
    Fy = norm.pdf(VaR, mu_p, np.sqrt(Sig2_p))
    c = ecsa_cache(VaR, R, LR)
    Fx = F_theta_fast(c, h, Mo)
    return -Fx / Fy


def ecsa_d2_fast(R, Rp, LR, alpha, use_is, Mo, h):
    VaR = quantile(Rp, LR, alpha, use_is)
    Fy, Fyy, Fyx = density_terms(VaR)
    c = ecsa_cache(VaR, R, LR)
    Fx = F_theta_fast(c, h, Mo)
    Fxx = F_thetatheta_fast(c, h, Mo)
    return _second_order(Fy, Fyy, Fyx, Fx, Fxx)


def verify_ecsa(alphas, seeds, N, h):
    """Largest relative difference between the reference and the fast ECSA on identical samples."""
    worst = 0.0
    for alpha in alphas:
        for use_is in (False, True):
            for sd in seeds:
                R, Rp, LR = sample(alpha, N, sd, use_is)
                VaR = quantile(Rp, LR, alpha, use_is)
                for Mo in (1, 2):
                    ref1, ref2 = ecsa_reference(R, LR, VaR, h, Mo, True)
                    for a, b in ((ref1, ecsa_d1_fast(R, Rp, LR, alpha, use_is, Mo, h)),
                                 (ref2, ecsa_d2_fast(R, Rp, LR, alpha, use_is, Mo, h))):
                        worst = max(worst, abs(a - b) / abs(a))
    return worst


# ============================================================================
# Hong (2009) window estimator, GLR (Peng et al. 2018), FD-type estimators
# ============================================================================


def hong_at(wv, R, LR, alpha, use_is):
    Rp = wv @ R
    q = quantile(Rp, LR, alpha, use_is)
    delta = np.sqrt(wv @ Sigma @ wv) * R.shape[1] ** (-1 / 3)
    win = np.abs(Rp - q) <= delta
    return np.sum(R[jj, win] * LR[win]) / np.sum(LR[win])


def hong_base(R, Rp, LR, alpha, use_is):
    q = quantile(Rp, LR, alpha, use_is)
    delta = sig_p * R.shape[1] ** (-1 / 3)
    win = np.abs(Rp - q) <= delta
    return np.sum(R[jj, win] * LR[win]) / np.sum(LR[win])


def w_pert(d):
    wv = Q.copy(); wv[jj] += d; return wv


def fd_quantile(R, LR, alpha, use_is, d):
    qp = quantile(w_pert(+d) @ R, LR, alpha, use_is)
    qm = quantile(w_pert(-d) @ R, LR, alpha, use_is)
    return (qp - qm) / (2 * d)


def hongfd(R, LR, alpha, use_is, d):
    return (hong_at(w_pert(+d), R, LR, alpha, use_is) - hong_at(w_pert(-d), R, LR, alpha, use_is)) / (2 * d)


def glr_omega(wv, R):
    """GLR weight omega = -(1/w_3) [1 - R_3 (Sigma^-1 (R - mu))_3]."""
    s3 = Sinv[jj] @ (R - mu[:, None])
    return -(1.0 / wv[jj]) * (1.0 - R[jj] * s3)


def glr_from(wv, R, Rp, LR, alpha, use_is):
    """GLR VaR sensitivity -F_theta(q)/F_v(q) at weights wv."""
    q = quantile(Rp, LR, alpha, use_is)
    Fv = norm.pdf(q, mu @ wv, np.sqrt(wv @ Sigma @ wv))
    Ft = np.mean((Rp < q) * glr_omega(wv, R) * LR)
    return -Ft / Fv


def glr_base(R, Rp, LR, alpha, use_is):
    return glr_from(Q, R, Rp, LR, alpha, use_is)


def glr_at(wv, R, LR, alpha, use_is):
    return glr_from(wv, R, wv @ R, LR, alpha, use_is)


def glrfd(R, LR, alpha, use_is, d):
    return (glr_at(w_pert(+d), R, LR, alpha, use_is) - glr_at(w_pert(-d), R, LR, alpha, use_is)) / (2 * d)
