"""
Shewhart control chart with an exponential change point: model, exact ARL
derivatives and Monte Carlo estimators of ARL'(theta2), ARL''(theta2).

Supporting module of control_chart_main.py; formerly the code duplicated in
control_chart_table_main.py and control_chart_plot_main.py.

Model: X_i ~ N(mu0, sigma) before the change point z ~ Exp(mean Zbar) and
N(mu1, sigma) after it; run length L = first i with X_i outside [theta1, theta2].

Estimators
----------
* ECSA, literal formula of the manuscript (Section sec:shewhart):
  ARL_k(vartheta) = sum_{l=1}^{L-1} K^l exp(-[(K^2-1) S1(l) + 2(K-1) S2(l)] / (2 sigma^2)),
  K = (vartheta - theta1)/(theta2 - theta1), evaluated at vartheta = theta2 + h e^{i angle}
  and combined with the Moebius weights of Table tab:algorithm.
  Implemented with a "wavefront" loop over time steps that advances all still
  active replications at once (`wavefront_accumulate`); the per-path reference
  implementation `run_ecsa_reference` is kept for verification.
* GLR (Peng et al. 2018) first-derivative estimator.
* FD (Rich.) / GLR+FD (Rich.) with common random numbers: one long buffer of
  observations per replication, re-applying the theta2 +/- delta exit rule.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from utils.ecsa_utils import mobius, ecsa_angles, ecsa_prefactor, richardson_estimate, richardson_deltas

THETA1, THETA2 = -2.81, 2.81
MU0, MU1, SIGMA, ZBAR = 0.0, 1.0, 1.0, 20.0
MAXSTEP = 3000   # safety cap on the wavefront steps (99.99th pct of L is < 300)

# Richardson grid: four (delta, delta/2) pairs, finest first.
RICH_PAIRS = [(0.025, 0.0125), (0.05, 0.025), (0.1, 0.05), (0.2, 0.1)]
RICH_DELTAS = richardson_deltas(RICH_PAIRS)


# ============================================================================
# Exact closed-form benchmark
# ============================================================================


def arl_and_derivatives(theta1=THETA1, theta2=THETA2, mu0=MU0, mu1=MU1, sigma=SIGMA, Zbar=ZBAR):
    """Exact ARL(theta2) and its first and second derivatives in theta2."""
    rho = np.exp(-1.0 / Zbar)
    p0 = norm.cdf((theta2 - mu0) / sigma) - norm.cdf((theta1 - mu0) / sigma)
    p1 = norm.cdf((theta2 - mu1) / sigma) - norm.cdf((theta1 - mu1) / sigma)
    p0p = norm.pdf((theta2 - mu0) / sigma) / sigma
    p1p = norm.pdf((theta2 - mu1) / sigma) / sigma
    p0pp = -(theta2 - mu0) * norm.pdf((theta2 - mu0) / sigma) / sigma**3
    p1pp = -(theta2 - mu1) * norm.pdf((theta2 - mu1) / sigma) / sigma**3
    q0 = 1.0 - p0
    q1 = 1.0 - p1
    C = (1.0 - rho) / (1.0 - rho * p0)
    D = p1 / q1 - p0 / q0
    Cp = (1.0 - rho) * rho * p0p / (1.0 - rho * p0)**2
    Cpp = ((1.0 - rho) * rho * p0pp / (1.0 - rho * p0)**2
           + 2.0 * (1.0 - rho) * rho**2 * p0p**2 / (1.0 - rho * p0)**3)
    Dp = p1p / q1**2 - p0p / q0**2
    Dpp = (p1pp / q1**2 + 2.0 * p1p**2 / q1**3 - p0pp / q0**2 - 2.0 * p0p**2 / q0**3)
    arl = 1.0 / q0 + C * D
    darl = p0p / q0**2 + Cp * D + C * Dp
    ddarl = (p0pp / q0**2 + 2.0 * p0p**2 / q0**3 + Cpp * D + 2.0 * Cp * Dp + C * Dpp)
    return arl, darl, ddarl


# ============================================================================
# ECSA
# ============================================================================


def ecsa_eval_points(n: int, M: int, phi: float, h: float) -> list[tuple[complex, float]]:
    """(K, weight) for every ECSA evaluation point vartheta = theta2 + h e^{i angle}
    (points shared by several Moebius terms are kept, as in the literal formula)."""
    C = ecsa_prefactor(n, h, phi)
    pts = []
    for k in range(1, M + 1):
        mu_k = mobius(k)
        if mu_k == 0:
            continue
        for angle in ecsa_angles(n, k, phi):
            vartheta = THETA2 + h * np.exp(1j * angle)
            K = (vartheta - THETA1) / (THETA2 - THETA1)
            weight = C * mu_k * (1.0 / (k * n))
            pts.append((K, weight))
    return pts


def wavefront_accumulate(rng, batch_size: int, Ks: np.ndarray, maxstep: int = MAXSTEP):
    """Simulate one batch of run lengths and accumulate, for every replication
    and every evaluation point K, the sum over l = 1..L-1 of
    K^l exp(-[(K^2-1) S1(l) + 2(K-1) S2(l)] / (2 sigma^2)).

    Returns (acc, n_active): the (batch_size, len(Ks)) complex accumulator and
    the number of replications still active at maxstep (expected 0)."""
    z_k = rng.exponential(ZBAR, size=batch_size)
    S1 = np.zeros(batch_size)
    S2 = np.zeros(batch_size)
    acc = np.zeros((batch_size, len(Ks)), dtype=complex)
    active = np.ones(batch_size, dtype=bool)

    step = 0
    while step < maxstep:
        step += 1
        idx_act = np.where(active)[0]
        if idx_act.size == 0:
            break
        muzi = np.where(step <= z_k[idx_act], MU0, MU1)
        Xi = rng.normal(muzi, SIGMA)
        exit_now = (Xi < THETA1) | (Xi > THETA2)
        stay = ~exit_now

        idx_stay = idx_act[stay]
        if idx_stay.size > 0:
            Xi_s = Xi[stay]
            mu_s = muzi[stay]
            S1[idx_stay] += (Xi_s - THETA1) ** 2
            S2[idx_stay] += (Xi_s - THETA1) * (THETA1 - mu_s)
            expo = (Ks[None, :] ** 2 - 1) * S1[idx_stay, None] \
                + 2 * (Ks[None, :] - 1) * S2[idx_stay, None]
            term = (Ks[None, :] ** step) * np.exp(-expo / (2 * SIGMA ** 2))
            acc[idx_stay, :] += term

        idx_exit = idx_act[exit_now]
        active[idx_exit] = False
    return acc, active.sum()


def run_ecsa(n: int, M: int, phi: float, total_sims: int, batch_size: int, h: float,
             seed: int, maxstep: int = MAXSTEP):
    """Wavefront ECSA estimate of ARL^(n)(theta2): (mean, SE, time, truncated fraction)."""
    eval_points = ecsa_eval_points(n, M, phi, h)
    Ks = np.array([K for K, _ in eval_points])
    weights = np.array([weight for _, weight in eval_points])

    rng = np.random.default_rng(seed)
    num_batches = total_sims // batch_size
    vals = np.zeros(total_sims)
    n_truncated = 0
    t0 = time.perf_counter()
    for b in range(num_batches):
        b0 = b * batch_size
        acc, n_active = wavefront_accumulate(rng, batch_size, Ks, maxstep)
        n_truncated += n_active
        vals[b0:b0 + batch_size] = (weights[None, :] * np.imag(acc)).sum(axis=1)
    elapsed = time.perf_counter() - t0
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(total_sims)
    return m, se, elapsed, n_truncated / total_sims


def run_ecsa_reference(n: int, M: int, phi: float, total_sims: int, batch_size: int,
                       h: float, seed: int):
    """Per-path reference implementation of the same literal ECSA formula (slow;
    uses the legacy RandomState stream, so it agrees with `run_ecsa` only
    statistically). Used by `verify_wavefront`."""
    eval_points = ecsa_eval_points(n, M, phi, h)
    vals = np.zeros(total_sims)
    rng_state = np.random.RandomState(seed)
    num_batches = total_sims // batch_size
    t0 = time.perf_counter()
    for b in range(num_batches):
        b0 = b * batch_size
        for s in range(batch_size):
            idx = b0 + s
            z_k = rng_state.exponential(ZBAR)
            path, means = [], []
            i = 1
            while True:
                mu_ik = MU0 if i <= z_k else MU1
                X_i = rng_state.normal(mu_ik, SIGMA)
                path.append(X_i)
                means.append(mu_ik)
                if X_i < THETA1 or X_i > THETA2:
                    break
                i += 1
            X_path = np.array(path)
            mu_path = np.array(means)
            L = len(X_path)
            est = 0.0
            if L >= 2:
                d1 = (X_path - THETA1) ** 2
                d2 = (X_path - THETA1) * (THETA1 - mu_path)
                S1 = np.cumsum(d1)[:L - 1]
                S2 = np.cumsum(d2)[:L - 1]
                l_vals = np.arange(1, L)
                for K, weight in eval_points:
                    expo = (K ** 2 - 1) * S1 + 2 * (K - 1) * S2
                    terms = (K ** l_vals) * np.exp(-expo / (2 * SIGMA ** 2))
                    est += weight * np.imag(np.sum(terms))
            vals[idx] = est
    elapsed = time.perf_counter() - t0
    return vals.mean(), vals.std(ddof=1) / np.sqrt(total_sims), elapsed


def verify_wavefront(n_check: int = 100_000, seed: int = 42, h: float = 1e-5) -> bool:
    """Statistical agreement (|diff| < 3 combined SE) of the reference and the
    wavefront ECSA implementations for the four (n, M) configurations."""
    print(f"=== Reference vs wavefront ECSA, N={n_check:,} ===", flush=True)
    configs = [("1st M=1", 1, 1, np.pi / 2), ("1st M=2", 1, 2, np.pi / 3),
               ("2nd M=1", 2, 1, np.pi / 4), ("2nd M=2", 2, 2, np.pi / 6)]
    all_ok = True
    for name, n, M, phi in configs:
        m_old, se_old, t_old = run_ecsa_reference(n, M, phi, n_check, n_check, h, seed)
        m_new, se_new, t_new, trunc = run_ecsa(n, M, phi, n_check, n_check, h, seed)
        diff = abs(m_old - m_new)
        combined_se = np.sqrt(se_old ** 2 + se_new ** 2)
        ratio = diff / combined_se if combined_se > 0 else float('nan')
        ok = ratio < 3.0
        all_ok &= ok
        print(f"  {name:<8} ref={m_old:.4f}({se_old:.4f}, {t_old:.2f}s)  "
              f"wavefront={m_new:.4f}({se_new:.4f}, {t_new:.2f}s)  "
              f"|diff|={diff:.4f} ({ratio:.2f} combined SE)  speedup={t_old / t_new:.1f}x  "
              f"trunc={trunc:.1e}  [{'OK' if ok else 'FLAG'}]", flush=True)
    print(f"=== {'PASSED' if all_ok else 'FLAGGED -- inspect above'} ===\n", flush=True)
    return all_ok


# ============================================================================
# GLR
# ============================================================================


def run_glr(total_sims: int, batch_size: int, seed: int, maxstep: int = 4000):
    """GLR estimate of ARL'(theta2) (wavefront): (mean, SE, time)."""
    rng = np.random.default_rng(seed)
    acc_sum = 0.0
    acc_sq = 0.0
    done = 0
    t0 = time.perf_counter()
    while done < total_sims:
        bs = min(batch_size, total_sims - done)
        Zcp = rng.exponential(ZBAR, bs)
        Np = np.zeros(bs)
        gs = np.zeros(bs)
        ex = np.zeros(bs, bool)
        step = 0
        while step < maxstep:
            step += 1
            act = ~ex
            if not act.any():
                break
            idx = np.where(act)[0]
            muzi = np.where(step <= Zcp[idx], MU0, MU1)
            xi = rng.normal(muzi, SIGMA)
            gi = (xi - THETA1) / (THETA2 - THETA1)
            gs[idx] += gi * (xi - muzi)
            hit = idx[(xi <= THETA1) | (xi >= THETA2)]
            Np[hit] = step
            ex[hit] = True
        Np[Np == 0] = maxstep
        w = Np * (Np / (THETA2 - THETA1) - gs)
        acc_sum += w.sum()
        acc_sq += (w * w).sum()
        done += bs
    elapsed = time.perf_counter() - t0
    mean = acc_sum / total_sims
    var = acc_sq / total_sims - mean ** 2
    se = np.sqrt(var / total_sims)
    return mean, se, elapsed


# ============================================================================
# FD (Rich.) / GLR+FD (Rich.) with a common long buffer
# ============================================================================


def _stopping_time_and_glr(X_seq, mu_seq, theta2_val):
    """First-exit time and GLR score under theta2_val on the SAME buffer."""
    exceed = (X_seq < THETA1) | (X_seq > theta2_val)
    hit_any = exceed.any(axis=1)
    L = np.where(hit_any, np.argmax(exceed, axis=1) + 1, X_seq.shape[1])
    L = L.astype(float)
    col_idx = np.arange(X_seq.shape[1])[None, :]
    within = col_idx < L[:, None]
    g_i = (X_seq - THETA1) / (theta2_val - THETA1)
    score = np.where(within, g_i * (X_seq - mu_seq) / SIGMA ** 2, 0.0).sum(axis=1)
    Gscore = L * (L / (theta2_val - THETA1) - score)
    trunc_frac = 1.0 - hit_any.mean()
    return L, Gscore, trunc_frac


def run_fd_richardson(total_sims: int, batch_size: int, seed: int, max_len: int, want: str):
    """want='fd': FD (Rich.) of the run length (first derivative);
    want='glrfd': GLR+FD (Rich.) of the GLR score (second derivative).
    Returns (mean, SE, full-search time, selected pair, max truncated fraction, all pair stats)."""
    rng = np.random.default_rng(seed + 1000)  # offset seed: independent stream from ECSA/GLR
    num_batches = total_sims // batch_size
    valsdict = {d: [] for d in RICH_DELTAS}
    tdict = {d: 0.0 for d in RICH_DELTAS}
    max_trunc = 0.0
    for b in range(num_batches):
        z_k = rng.exponential(ZBAR, batch_size)
        idx = np.arange(1, max_len + 1)
        mu_seq = np.where(idx[None, :] <= z_k[:, None], MU0, MU1)
        X_seq = rng.normal(mu_seq, SIGMA)
        for d in RICH_DELTAS:
            t0 = time.perf_counter()
            L_p, G_p, tr_p = _stopping_time_and_glr(X_seq, mu_seq, THETA2 + d)
            L_m, G_m, tr_m = _stopping_time_and_glr(X_seq, mu_seq, THETA2 - d)
            if want == 'fd':
                per_rep = (L_p - L_m) / (2 * d)
            else:
                per_rep = (G_p - G_m) / (2 * d)
            valsdict[d].append(per_rep)
            tdict[d] += time.perf_counter() - t0
            max_trunc = max(max_trunc, tr_p, tr_m)
    for d in RICH_DELTAS:
        valsdict[d] = np.concatenate(valsdict[d])
    m, se, full_time, pair, stats = richardson_estimate(valsdict, tdict, RICH_PAIRS)
    return m, se, full_time, pair, max_trunc, stats


# ============================================================================
# Vary-h sweep (one shared simulation for all h and all (n, M))
# ============================================================================


def run_ecsa_hsweep(h_vals, configs: dict, total_sims: int, batch_size: int, seed: int):
    """ECSA mean and SE for every config and every h from ONE wavefront
    simulation per batch (the paths and partial sums do not depend on h).
    The per-replication combined value of each (config, h) is formed before
    squaring, so the SE accounts for the cancellation between angle points."""
    point_K, point_weight, point_index = [], [], []
    for h_idx, h in enumerate(h_vals):
        for cfg_name, cfg in configs.items():
            for K, weight in ecsa_eval_points(cfg['n'], cfg['M'], cfg['phi'], h):
                point_K.append(K)
                point_weight.append(weight)
                point_index.append((cfg_name, h_idx))
    Ks = np.array(point_K)
    weights = np.array(point_weight)
    group_cols = {}
    for p, key in enumerate(point_index):
        group_cols.setdefault(key, []).append(p)

    sums = {c: np.zeros(len(h_vals)) for c in configs}
    sq_sums = {c: np.zeros(len(h_vals)) for c in configs}
    num_batches = total_sims // batch_size
    print(f"Starting {total_sims:,} simulations, {len(Ks)} shared angle-points "
          f"across {len(h_vals)} h-values x {len(configs)} configs, "
          f"in {num_batches} batches (wavefront ECSA, seed={seed})...", flush=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for batch_idx in range(num_batches):
        acc, _ = wavefront_accumulate(rng, batch_size, Ks)
        per_point_all = weights[None, :] * np.imag(acc)
        for (cfg_name, h_idx), cols in group_cols.items():
            combined = per_point_all[:, cols].sum(axis=1)
            sums[cfg_name][h_idx] += combined.sum()
            sq_sums[cfg_name][h_idx] += np.sum(combined ** 2)
        print(f"  Batch {batch_idx + 1}/{num_batches} completed | Elapsed: {time.time() - t0:.1f}s",
              flush=True)

    means, ses = {}, {}
    for cfg_name in configs:
        mean_val = sums[cfg_name] / total_sims
        var_val = (sq_sums[cfg_name] - total_sims * (mean_val ** 2)) / (total_sims - 1)
        means[cfg_name] = mean_val
        ses[cfg_name] = np.sqrt(np.maximum(0.0, var_val)) / np.sqrt(total_sims)
    return means, ses
