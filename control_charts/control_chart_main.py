"""
Control charts (Shewhart chart, exponential change point) -- main script.

    python control_chart_main.py table    # Table tab:shewhart_full (Table 3)
    python control_chart_main.py figure   # Figure fig:shewhart_full (Figure 1)
    python control_chart_main.py all      # both
    python control_chart_main.py verify   # reference vs wavefront ECSA check (optional)

Table: Exact (closed form), ECSA M=1/M=2, GLR, FD (Rich.) for ARL'(theta2),
N = 1e6; Exact, ECSA M=1/M=2, GLR+FD (Rich.) for ARL''(theta2), N = 1e7;
h = 1e-5, seed 42 (FD-type methods: seed 42 + 1000). SEs are pathwise.
Times: wall-clock seconds of each method.

Figure: ECSA M=1/M=2 versus log10 h in {-7,-6,-5,-4,-3,-2.5,-2,-1.8},
N = 1e6, seed 10, one shared simulation for all h and configurations,
+-3 SE bands, exact benchmark line.

Outputs (in tables/ and fig/):
    control_chart_table.csv
    control_chart_table_richardson_pairs.csv
    control_chart_hsweep.png
    control_chart_hsweep_data.csv

Formerly control_chart_table_main.py (table) and control_chart_plot_main.py (figure).
"""
import sys
from pathlib import Path

# Project root on the import path, so that the shared utils/ package is found
# whatever the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import control_chart_model as cc
from utils.ecsa_utils import (ecsa_phi, method_row, write_csv, table_columns, richardson_columns,
                        hsweep_columns, table_path, fig_path, print_table)

PREFIX = "control_chart"
MODEL_NAME = "shewhart"

# Table settings
H = 1e-5
N_FIRST = 1_000_000     # ECSA M=1, ECSA M=2, GLR, FD (Rich.)
N_SECOND = 10_000_000   # ECSA M=1, ECSA M=2, GLR+FD (Rich.)
BATCH_ECSA = 200_000
BATCH_GLR = 200_000
BATCH_FD = 50_000
MAX_LEN = 3000          # long-buffer length for the CRN FD / GLR+FD construction
SEED_TABLE = 42

# Figure settings
SEED_FIGURE = 10
N_FIGURE = 1_000_000
BATCH_FIGURE = 200_000
LOG10_H = np.array([-7.0, -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.8])


def run_table():
    exact_arl, exact_d1, exact_d2 = cc.arl_and_derivatives()
    print(f"Exact ARL={exact_arl:.6f}  ARL^(1)={exact_d1:.6f}  ARL^(2)={exact_d2:.6f}\n", flush=True)
    res = {}

    print(f"--- First-order block (N={N_FIRST:,}) ---", flush=True)
    for M in (1, 2):
        m, se, t, trunc = cc.run_ecsa(1, M, ecsa_phi(1, M), N_FIRST, BATCH_ECSA, H, SEED_TABLE)
        res[(1, M)] = (m, se, t)
        print(f"ECSA M={M} (1st): est={m:.6f} se={se:.6f} time={t:.1f}s trunc={trunc:.1e}", flush=True)
    res["glr"] = cc.run_glr(N_FIRST, BATCH_GLR, SEED_TABLE)
    print("GLR:            est={:.6f} se={:.6f} time={:.1f}s".format(*res["glr"]), flush=True)
    m, se, t, pair, trunc, stats_fd = cc.run_fd_richardson(N_FIRST, BATCH_FD, SEED_TABLE, MAX_LEN, 'fd')
    res["fd"] = (m, se, t, pair)
    print(f"FD (Rich.):     est={m:.6f} se={se:.6f} time={t:.1f}s pair={pair} max_trunc_frac={trunc:.2e}",
          flush=True)

    print(f"\n--- Second-order block (N={N_SECOND:,}) ---", flush=True)
    for M in (1, 2):
        m, se, t, trunc = cc.run_ecsa(2, M, ecsa_phi(2, M), N_SECOND, BATCH_ECSA, H, SEED_TABLE)
        res[(2, M)] = (m, se, t)
        print(f"ECSA M={M} (2nd): est={m:.6f} se={se:.6f} time={t:.1f}s trunc={trunc:.1e}", flush=True)
    m, se, t, pair, trunc, stats_glrfd = cc.run_fd_richardson(N_SECOND, BATCH_FD, SEED_TABLE, MAX_LEN, 'glrfd')
    res["glrfd"] = (m, se, t, pair)
    print(f"GLR+FD (Rich.): est={m:.6f} se={se:.6f} time={t:.1f}s pair={pair} max_trunc_frac={trunc:.2e}",
          flush=True)

    common = dict(application="control_chart", model=MODEL_NAME)
    rows = [method_row(1, "Exact", exact_d1)]
    for M in (1, 2):
        rows.append(method_row(1, f"ECSA M={M}", *res[(1, M)], exact_d1, ecsa_M=M, h=H, N=N_FIRST))
    rows.append(method_row(1, "GLR", *res["glr"], exact_d1, N=N_FIRST))
    rows.append(method_row(1, "FD (Rich.)", *res["fd"][:3], exact_d1, fd_pair=res["fd"][3], N=N_FIRST))
    rows.append(method_row(2, "Exact", exact_d2))
    for M in (1, 2):
        rows.append(method_row(2, f"ECSA M={M}", *res[(2, M)], exact_d2, ecsa_M=M, h=H, N=N_SECOND))
    rows.append(method_row(2, "GLR+FD (Rich.)", *res["glrfd"][:3], exact_d2,
                           fd_pair=res["glrfd"][3], N=N_SECOND))
    rows = [dict(common, **r) for r in rows]
    write_csv(table_path(f"{PREFIX}_table.csv"), table_columns([]), rows)

    rich_rows = []
    for order, method, stats, sel in ((1, "FD (Rich.)", stats_fd, res["fd"][3]),
                                      (2, "GLR+FD (Rich.)", stats_glrfd, res["glrfd"][3])):
        for m, se, pair in stats:
            rich_rows.append(dict(common, derivative_order=order, method=method,
                                  fd_delta=pair[0], fd_delta_half=pair[1], estimate=m, se=se,
                                  selected=pair == sel))
    write_csv(table_path(f"{PREFIX}_table_richardson_pairs.csv"), richardson_columns([]), rich_rows)
    print_table(f"Control chart (Shewhart): theta1={cc.THETA1}, theta2={cc.THETA2}, h={H}", rows, [])
    return rows


def run_figure():
    h_vals = 10.0 ** LOG10_H
    configs = {
        '1st_order_M=1': {'n': 1, 'M': 1, 'phi': ecsa_phi(1, 1)},
        '1st_order_M=2': {'n': 1, 'M': 2, 'phi': ecsa_phi(1, 2)},
        '2nd_order_M=1': {'n': 2, 'M': 1, 'phi': ecsa_phi(2, 1)},
        '2nd_order_M=2': {'n': 2, 'M': 2, 'phi': ecsa_phi(2, 2)},
    }
    means, ses = cc.run_ecsa_hsweep(h_vals, configs, N_FIGURE, BATCH_FIGURE, SEED_FIGURE)
    exact_arl, exact_darl, exact_ddarl = cc.arl_and_derivatives()
    print(f"Exact: dARL/dtheta2 = {exact_darl:.6f}, d2ARL/dtheta2^2 = {exact_ddarl:.6f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    def plot_series_with_se_bands(ax, x_vals, mean, se, color, marker, label):
        ax.plot(x_vals, mean, color=color, marker=marker, linestyle='-', linewidth=1.5, label=label)
        ax.plot(x_vals, mean + 3 * se, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
        ax.plot(x_vals, mean - 3 * se, color=color, linestyle='--', alpha=0.7, linewidth=1.2)

    for ax, n, exact, title in ((axes[0], 1, exact_darl, 'First derivative'),
                                (axes[1], 2, exact_ddarl, 'Second derivative')):
        order = '1st' if n == 1 else '2nd'
        ax.axhline(exact, color='black', linestyle='-', linewidth=1.5, label='Exact')
        plot_series_with_se_bands(ax, LOG10_H, means[f'{order}_order_M=1'], ses[f'{order}_order_M=1'],
                                  color='red', marker='o', label='ECSA $M = 1$')
        plot_series_with_se_bands(ax, LOG10_H, means[f'{order}_order_M=2'], ses[f'{order}_order_M=2'],
                                  color='green', marker='s', label='ECSA $M = 2$')
        ax.set_xlabel(r'$\log_{10} h$', fontsize=12)
        ax.set_ylabel(rf'$ARL^{{({n})}}(\theta_2)$', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    out = fig_path(f"{PREFIX}_hsweep.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    rows = []
    for n, exact in ((1, exact_darl), (2, exact_ddarl)):
        order = '1st' if n == 1 else '2nd'
        rows.append(dict(application="control_chart", model=MODEL_NAME, derivative_order=n,
                         method="Exact", estimate=exact))
        for M in (1, 2):
            key = f'{order}_order_M={M}'
            for p, h, mval, sval in zip(LOG10_H, h_vals, means[key], ses[key]):
                rows.append(dict(application="control_chart", model=MODEL_NAME, derivative_order=n,
                                 method=f"ECSA M={M}", ecsa_M=M, ecsa_phi=ecsa_phi(n, M),
                                 log10_h=p, h=h, estimate=mval, se=sval, N=N_FIGURE))
    write_csv(fig_path(f"{PREFIX}_hsweep_data.csv"), hsweep_columns([]), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("experiment", choices=["table", "figure", "all", "verify"])
    args = parser.parse_args()
    if args.experiment == "verify":
        cc.verify_wavefront()
    if args.experiment in ("table", "all"):
        run_table()
    if args.experiment in ("figure", "all"):
        run_figure()
