"""
Shared utilities for the ECSA numerical applications (control charts, barrier
options, portfolio VaR sensitivities).

This module holds the single authoritative implementation of the components
that were previously duplicated across the scripts of ``Python021026``:

  * the Moebius function and the ECSA angle-selection rule
    (Corollary th:corollary_phi of the manuscript);
  * the ECSA evaluation angles, Moebius terms and prefactor
    n! / (h^n sin(n phi)) of Table tab:algorithm;
  * replication summaries (mean, standard error);
  * the two-point Richardson extrapolation of finite-difference estimators and
    the benchmark-free, finest-first, 3x-SE back-off pair-selection rule;
  * output-folder helpers and the standardized CSV export of numerical tables.

Every function reproduces, operation by operation, the arithmetic of the
original scripts, so that the refactored code yields bit-identical numbers.
"""
from __future__ import annotations

import csv
import sys
import math
import time
from pathlib import Path

import numpy as np

# ============================================================================
# 1. Moebius function and ECSA angles
# ============================================================================


def mobius(n: int) -> int:
    """Moebius function mu(n): 1 if n=1, 0 if n has a squared prime factor,
    (-1)^k if n is a product of k distinct primes."""
    if n == 1:
        return 1
    d, m, factors = n, 2, 0
    while m * m <= d:
        if d % m == 0:
            d //= m
            factors += 1
            if d % m == 0:
                return 0  # squared factor
        m += 1
    if d > 1:
        factors += 1
    return -1 if factors % 2 else 1


# Optimal ECSA angle phi of Corollary th:corollary_phi, keyed by (n, M).
ECSA_PHI = {(1, 1): np.pi / 2, (1, 2): np.pi / 3, (2, 1): np.pi / 4, (2, 2): np.pi / 6}


def ecsa_phi(n: int, M: int) -> float:
    """ECSA angle phi for derivative order n and truncation level M."""
    if (n, M) not in ECSA_PHI:
        raise ValueError(f"No ECSA angle tabulated for (n={n}, M={M}).")
    return ECSA_PHI[(n, M)]


def ecsa_angles(n: int, k: int, phi: float) -> list[float]:
    """The k*n angles phi + 2 pi j / (k n), j = 0..k n - 1, of the k-th Moebius term."""
    return [phi + 2 * np.pi * j / (k * n) for j in range(k * n)]


def mobius_terms(n: int, M: int, phi: float) -> list[tuple[int, int, list[float]]]:
    """(k, mu(k), angles of term k) for k = 1..M with mu(k) != 0."""
    terms = []
    for k in range(1, M + 1):
        mk = mobius(k)
        if mk == 0:
            continue
        terms.append((k, mk, ecsa_angles(n, k, phi)))
    return terms


def required_angles(n: int, M: int, phi: float) -> list[float]:
    """Sorted union over k = 1..M of the angles needed by the (n, M, phi) estimator."""
    angles = set()
    for _, _, angs in mobius_terms(n, M, phi):
        angles.update(angs)
    return sorted(angles)


def angle_key(angle: float) -> float:
    """Angle rounded to 12 decimals, used as a dictionary key for evaluated angles."""
    return round(angle, 12)


def ecsa_prefactor(n: int, h: float, phi: float) -> float:
    """ECSA normalization n! / (h^n sin(n phi))."""
    return math.factorial(n) / (h ** n * np.sin(n * phi))


# ============================================================================
# 2. Replication summaries and Richardson extrapolation
# ============================================================================


def summarize(vals) -> tuple[float, float]:
    """Mean and standard error (sample SD with ddof=1, divided by sqrt(n))."""
    v = np.array(vals)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def richardson_stat(valsdict: dict, d: float, dh: float):
    """Two-point Richardson extrapolation (4 D(d/2) - D(d)) / 3 of paired (CRN)
    per-replication finite-difference values; returns (mean, SE, (d, dh))."""
    a = np.array(valsdict[d])
    b = np.array(valsdict[dh])
    dtil = (4.0 * b - a) / 3.0
    m, se = summarize(dtil)
    return m, se, (d, dh)


def select_richardson(stats: list):
    """Benchmark-free pair selection. `stats` lists (mean, SE, pair) from the
    finest pair to the coarsest; keep the finest pair unless its SE exceeds 3x
    the SE of the next coarser pair, in which case move to the coarser one."""
    for i in range(len(stats) - 1):
        if stats[i][1] <= 3.0 * stats[i + 1][1]:
            return stats[i]
    return stats[-1]


def richardson_estimate(valsdict: dict, tdict: dict, pairs: list):
    """Richardson FD estimate over the candidate `pairs` (finest first).

    Returns (mean, SE, full_time, selected_pair, all_pair_stats). The reported
    time charges every distinct delta touched by any candidate pair (the
    "full search" cost), not only the two deltas of the selected pair.
    """
    stats = [richardson_stat(valsdict, d, dh) for d, dh in pairs]
    full_time = sum(tdict[d] for d in richardson_deltas(pairs))
    m, se, pair = select_richardson(stats)
    return m, se, full_time, pair, stats


def richardson_deltas(pairs: list) -> list[float]:
    """Sorted distinct finite-difference steps appearing in the candidate pairs."""
    return sorted({d for pair in pairs for d in pair})


def timed(fn, *args):
    """Evaluate fn(*args) and return (value, elapsed wall-clock seconds)."""
    t0 = time.perf_counter()
    v = fn(*args)
    return v, time.perf_counter() - t0


# ============================================================================
# 3. Output folders
# ============================================================================

# Outputs go to <output dir>/fig and <output dir>/tables. By default the output
# dir is the folder of the main script being run (e.g. barrier_options/), so
# each application keeps its own results; `set_output_dir` overrides it.
OUTPUT_DIR: Path | None = None


def set_output_dir(path) -> None:
    """Write figures and tables under `path`/fig and `path`/tables."""
    global OUTPUT_DIR
    OUTPUT_DIR = Path(path).resolve()


def output_dir() -> Path:
    """Current output folder: the one set by `set_output_dir`, else the folder
    of the running main script, else the current working directory."""
    if OUTPUT_DIR is not None:
        return OUTPUT_DIR
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(main_file).resolve().parent if main_file else Path.cwd()


def fig_path(name: str) -> Path:
    """Path of a figure file inside <output dir>/fig (folder created on demand)."""
    d = output_dir() / "fig"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def table_path(name: str) -> Path:
    """Path of a table/data file inside <output dir>/tables (folder created on demand)."""
    d = output_dir() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def l_tag(l: int) -> str:
    """Filesystem-safe tag of the number of monitoring dates, e.g. 'l10'."""
    return f"l{l}"


# ============================================================================
# 4. Standardized CSV export
# ============================================================================
#
# Every main table is exported in "long" form: one row per (configuration,
# derivative order, method), mirroring the rows of Tables 3-7 of main-v13.tex
# (Method | Est. | SE | AE | Time), with the table's column blocks (l, alpha,
# sampling scheme) stored as explicit configuration columns. Column order:
#
#   application, model, <application-specific configuration columns>,
#   derivative_order, method, ecsa_M, ecsa_phi, h,
#   estimate, se, abs_error, time_s, fd_delta, fd_delta_half, N, replications
#
# Missing values (e.g. the SE of a deterministic benchmark, the AE when no
# benchmark exists, the ECSA angle of a non-ECSA method) are empty fields.
# Floats are written with repr(), i.e. the shortest string that round-trips to
# the identical double, so no precision is lost.

TABLE_LEADING = ["application", "model"]
TABLE_TRAILING = ["derivative_order", "method", "ecsa_M", "ecsa_phi", "h",
                  "estimate", "se", "abs_error", "time_s",
                  "fd_delta", "fd_delta_half", "N", "replications"]
RICHARDSON_TRAILING = ["derivative_order", "method", "fd_delta", "fd_delta_half",
                       "estimate", "se", "selected"]
HSWEEP_TRAILING = ["derivative_order", "method", "ecsa_M", "ecsa_phi",
                   "log10_h", "h", "estimate", "se", "N", "replications"]


def table_columns(config_columns: list[str]) -> list[str]:
    return TABLE_LEADING + list(config_columns) + TABLE_TRAILING


def richardson_columns(config_columns: list[str]) -> list[str]:
    return TABLE_LEADING + list(config_columns) + RICHARDSON_TRAILING


def hsweep_columns(config_columns: list[str]) -> list[str]:
    return TABLE_LEADING + list(config_columns) + HSWEEP_TRAILING


def _csv_field(x):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return x


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    """Write `rows` (dicts) to `path` with the given column order; keys missing
    from a row are written as empty fields."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(columns)
        for row in rows:
            unknown = set(row) - set(columns)
            if unknown:
                raise KeyError(f"Unknown CSV columns {sorted(unknown)} for {path.name}")
            wr.writerow([_csv_field(row.get(c)) for c in columns])
    print(f"wrote {path}", flush=True)
    return path


def method_row(derivative_order: int, method: str, estimate, se=None, time_s=None,
               benchmark=None, ecsa_M=None, h=None, fd_pair=None, **extra) -> dict:
    """One standardized table row. AE = |estimate - benchmark| is computed from
    the full-precision values when a benchmark is supplied."""
    row = dict(derivative_order=derivative_order, method=method,
               estimate=estimate, se=se, time_s=time_s)
    if benchmark is not None and method not in ("Benchmark", "Exact"):
        row["abs_error"] = abs(estimate - benchmark)
    if ecsa_M is not None:
        row["ecsa_M"] = ecsa_M
        row["ecsa_phi"] = ecsa_phi(derivative_order, ecsa_M)
        row["h"] = h
    if fd_pair is not None:
        row["fd_delta"], row["fd_delta_half"] = fd_pair
    row.update(extra)
    return row


def print_table(title: str, rows: list[dict], config_columns: list[str]) -> None:
    """Console rendering of a standardized table (Method, Est., SE, AE, Time)."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    last = None
    for r in rows:
        key = tuple(r.get(c) for c in config_columns) + (r["derivative_order"],)
        if key != last:
            cfg = ", ".join(f"{c}={r.get(c)}" for c in config_columns)
            print(f"\n--- {cfg + ', ' if cfg else ''}derivative order {r['derivative_order']} ---")
            last = key

        def f(x, fmt):
            return "--" if x is None else format(x, fmt)
        pair = f"  pair=({r['fd_delta']}, {r['fd_delta_half']})" if "fd_delta" in r else ""
        print(f"  {r['method']:<16} est={f(r['estimate'], '.6f')}  se={f(r.get('se'), '.6f')}"
              f"  AE={f(r.get('abs_error'), '.6f')}  time={f(r.get('time_s'), '.3f')}s{pair}")
