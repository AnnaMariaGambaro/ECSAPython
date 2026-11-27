"""
Portfolio value-at-risk sensitivities (theta = w_3) -- main script.

    python portfolio_var_main.py table    # Table tab:VaR (Table 7)
    python portfolio_var_main.py figure   # Figure fig:VaR (Figure 5)
    python portfolio_var_main.py all      # both
    python portfolio_var_main.py verify   # reference vs fast ECSA check (optional)
    (optional for table: --alpha 0.01 0.05 0.1)

Table: Exact, ECSA M=1/M=2, Hong, GLR, FD (Rich.) for dq/dw3 and Exact,
ECSA M=1/M=2, Hong+FD (Rich.), GLR+FD (Rich.) for d2q/dw3^2, for
alpha in {0.01, 0.05, 0.1}, under plain Monte Carlo (MC) and importance
sampling (IS). N = 1e6 per replication, 20 replications (seeds 0..19), h = 1e-5;
one sample per (alpha, seed, MC/IS) shared by all methods (CRN). SE = sample
SD of the 20 replication estimates / sqrt(20). Times: time.perf_counter summed
over replications; the simulation of the returns is shared and not billed.
Run the table alone on the machine, since it measures wall-clock times.

Figure: ECSA M=1/M=2 (MC and IS) versus log10 h in {-7,-5,-4,-3,-2.5},
N = 3e5, 4 replications (seeds 0..3), +-3 SE bands, exact line; literal
(reference) ECSA implementation.

Outputs (in tables/ and fig/):
    portfolio_var_table.csv
    portfolio_var_table_richardson_pairs.csv
    portfolio_var_table_results.json      (full results incl. raw replication values)
    portfolio_var_hsweep.pdf
    portfolio_var_hsweep_preview.png      (only if pypdfium2 is installed)
    portfolio_var_hsweep_data.csv

Formerly var_table_v2.py + make_var_table_tex.py (table) and
var_fig4_chunk.py + make_var_fig4.py (figure).
"""
import sys
from pathlib import Path

# Project root on the import path, so that the shared utils/ package is found
# whatever the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import portfolio_var_model as pv
from utils.ecsa_utils import (summarize, richardson_stat, richardson_estimate, timed, ecsa_phi,
                        method_row, write_csv, table_columns, richardson_columns,
                        hsweep_columns, table_path, fig_path, print_table)

PREFIX = "portfolio_var"
ALPHAS = [0.01, 0.05, 0.1]
SAMPLINGS = [('mc', False), ('is', True)]

# Table settings
NS = 20              # replications (seeds 0..19)
N = 1_000_000        # samples per replication
H = 1e-5             # ECSA radius

# Figure settings
FIG_POWERS = np.array([-8,-7, -5, -4, -3])
FIG_N = 300_000
FIG_NREP = 4


# ============================================================================
# Table
# ============================================================================


def run_block(alpha, use_is):
    """All methods for one (alpha, MC/IS) block over NS replications."""
    keys = ['ecsa1_d1', 'ecsa2_d1', 'hong_d1', 'glr_d1', 'ecsa1_d2', 'ecsa2_d2']
    vals = {k: [] for k in keys}; tt = {k: 0.0 for k in keys}
    fd_v = {d: [] for d in pv.RICH_DELTAS}; fd_t = {d: 0.0 for d in pv.RICH_DELTAS}
    hf_v = {d: [] for d in pv.RICH_DELTAS}; hf_t = {d: 0.0 for d in pv.RICH_DELTAS}
    gf_v = {d: [] for d in pv.RICH_DELTAS}; gf_t = {d: 0.0 for d in pv.RICH_DELTAS}
    t_sim = 0.0
    for sd in range(NS):
        (R, Rp, LR), t = timed(pv.sample, alpha, N, sd, use_is); t_sim += t
        for k, fn, Mo in [('ecsa1_d1', pv.ecsa_d1_fast, 1), ('ecsa2_d1', pv.ecsa_d1_fast, 2),
                          ('ecsa1_d2', pv.ecsa_d2_fast, 1), ('ecsa2_d2', pv.ecsa_d2_fast, 2)]:
            v, t = timed(fn, R, Rp, LR, alpha, use_is, Mo, H); vals[k].append(v); tt[k] += t
        v, t = timed(pv.hong_base, R, Rp, LR, alpha, use_is); vals['hong_d1'].append(v); tt['hong_d1'] += t
        v, t = timed(pv.glr_base, R, Rp, LR, alpha, use_is); vals['glr_d1'].append(v); tt['glr_d1'] += t
        for d in pv.RICH_DELTAS:
            v, t = timed(pv.fd_quantile, R, LR, alpha, use_is, d); fd_v[d].append(v); fd_t[d] += t
            v, t = timed(pv.hongfd, R, LR, alpha, use_is, d); hf_v[d].append(v); hf_t[d] += t
            v, t = timed(pv.glrfd, R, LR, alpha, use_is, d); gf_v[d].append(v); gf_t[d] += t
    out = {k: (*summarize(vals[k]), tt[k]) for k in keys}
    out['raw'] = {k: list(map(float, vals[k])) for k in keys}
    for name, pair_name, v_, t_ in (('fdrich_d1', 'fdrich_pair', fd_v, fd_t),
                                    ('hongfdrich_d2', 'hongfdrich_pair', hf_v, hf_t),
                                    ('glrfdrich_d2', 'glrfdrich_pair', gf_v, gf_t)):
        m, se, t, pair, _ = richardson_estimate(v_, t_, pv.RICH_PAIRS)
        out[name] = (m, se, t); out[pair_name] = pair
    out['fd_raw'] = {str(d): list(map(float, fd_v[d])) for d in pv.RICH_DELTAS}
    out['hongfd_raw'] = {str(d): list(map(float, hf_v[d])) for d in pv.RICH_DELTAS}
    out['glrfd_raw'] = {str(d): list(map(float, gf_v[d])) for d in pv.RICH_DELTAS}
    out['rich_pairs_fd'] = [richardson_stat(fd_v, d, dh)[:2] for d, dh in pv.RICH_PAIRS]
    out['rich_pairs_hongfd'] = [richardson_stat(hf_v, d, dh)[:2] for d, dh in pv.RICH_PAIRS]
    out['rich_pairs_glrfd'] = [richardson_stat(gf_v, d, dh)[:2] for d, dh in pv.RICH_PAIRS]
    out['t_sim'] = t_sim
    return out


# (derivative order, method label, result key, ECSA M, Richardson pair key, raw-pairs key)
TABLE_METHODS = [
    (1, "ECSA M=1", 'ecsa1_d1', 1, None, None),
    (1, "ECSA M=2", 'ecsa2_d1', 2, None, None),
    (1, "Hong", 'hong_d1', None, None, None),
    (1, "GLR", 'glr_d1', None, None, None),
    (1, "FD (Rich.)", 'fdrich_d1', None, 'fdrich_pair', 'rich_pairs_fd'),
    (2, "ECSA M=1", 'ecsa1_d2', 1, None, None),
    (2, "ECSA M=2", 'ecsa2_d2', 2, None, None),
    (2, "Hong+FD (Rich.)", 'hongfdrich_d2', None, 'hongfdrich_pair', 'rich_pairs_hongfd'),
    (2, "GLR+FD (Rich.)", 'glrfdrich_d2', None, 'glrfdrich_pair', 'rich_pairs_glrfd'),
]


def table_rows(res):
    """Standardized rows ordered as Table tab:VaR: derivative order, then
    Exact, then MC methods, then IS methods; one row per alpha."""
    rows, rich_rows = [], []
    for order in (1, 2):
        for a in res:
            r = res[a]
            rows.append(dict(application="portfolio_var", model=pv.MODEL_NAME, alpha=float(a),
                             **method_row(order, "Exact", r['exact'][order], time_s=r['t_exact'])))
        for tag, _ in SAMPLINGS:
            for (o, label, key, M, pair_key, pairs_key) in TABLE_METHODS:
                if o != order:
                    continue
                for a in res:
                    r = res[a]; b = r[tag]
                    m, se, t = b[key]
                    pair = b[pair_key] if pair_key else None
                    common = dict(application="portfolio_var", model=pv.MODEL_NAME,
                                  alpha=float(a), sampling=tag.upper())
                    rows.append(dict(common, **method_row(
                        order, label, m, se, t, r['exact'][order], ecsa_M=M,
                        h=H if M else None, fd_pair=pair, N=N, replications=NS)))
                    if pairs_key:
                        for (pm, pse), pp in zip(b[pairs_key], pv.RICH_PAIRS):
                            rich_rows.append(dict(common, derivative_order=order, method=label,
                                                  fd_delta=pp[0], fd_delta_half=pp[1],
                                                  estimate=pm, se=pse, selected=tuple(pp) == tuple(pair)))
    return rows, rich_rows


def run_table(alphas):
    res = {}
    json_path = table_path(f"{PREFIX}_table_results.json")
    for alpha in alphas:
        t0 = time.perf_counter(); Ve, d1e, d2e = pv.exact(alpha); t_ex = time.perf_counter() - t0
        res[str(alpha)] = {'exact': (Ve, d1e, d2e), 't_exact': t_ex}
        for tag, is_ in SAMPLINGS:
            r = run_block(alpha, is_); res[str(alpha)][tag] = r
            print(f"alpha={alpha} {tag.upper()}  sim(not billed)={r['t_sim']:.1f}s", flush=True)
            print(f"  d1 exact={d1e:.5f}", flush=True)
            for k, lab in [('ecsa1_d1', 'ECSA M=1'), ('ecsa2_d1', 'ECSA M=2'), ('hong_d1', 'Hong'),
                           ('glr_d1', 'GLR'), ('fdrich_d1', 'FD (Rich.)')]:
                m, se, t = r[k]
                print(f"    {lab:<18} {m:.5f} ({se:.5f})  AE={abs(m-d1e):.5f}  time={t:.1f}s", flush=True)
            print(f"    FD (Rich.) pair={r['fdrich_pair']}", flush=True)
            print(f"  d2 exact={d2e:.5f}", flush=True)
            for k, lab in [('ecsa1_d2', 'ECSA M=1'), ('ecsa2_d2', 'ECSA M=2'),
                           ('hongfdrich_d2', 'Hong+FD (Rich.)'), ('glrfdrich_d2', 'GLR+FD (Rich.)')]:
                m, se, t = r[k]
                print(f"    {lab:<18} {m:.5f} ({se:.5f})  AE={abs(m-d2e):.5f}  time={t:.1f}s", flush=True)
            print(f"    Hong+FD (Rich.) pair={r['hongfdrich_pair']}  GLR+FD (Rich.) pair={r['glrfdrich_pair']}",
                  flush=True)
        with open(json_path, 'w') as f:
            json.dump(res, f)
    print(f"wrote {json_path}", flush=True)
    rows, rich_rows = table_rows(res)
    write_csv(table_path(f"{PREFIX}_table.csv"), table_columns(["alpha", "sampling"]), rows)
    write_csv(table_path(f"{PREFIX}_table_richardson_pairs.csv"),
              richardson_columns(["alpha", "sampling"]), rich_rows)
    print_table(f"Portfolio VaR (Hong 2009), theta=w3, N={N}, {NS} reps, h={H}",
                sorted(rows, key=lambda r: (r['alpha'], r.get('sampling') or '', r['derivative_order'])),
                ["alpha", "sampling"])
    return res


# ============================================================================
# Figure
# ============================================================================


def run_figure_data():
    """Means/SEs of the literal ECSA estimator over the h grid, for every
    (order, alpha, MC/IS, M); returns {(order, alpha, tag, M): (means, ses)}."""
    data = {}
    for order in ('d1', 'd2'):
        for alpha in ALPHAS:
            for tag, is_ in SAMPLINGS:
                for Mo in (1, 2):
                    ms, ss = [], []
                    for p in FIG_POWERS:
                        vals = []
                        for sd in range(FIG_NREP):
                            R, Rp, LR = pv.sample(alpha, FIG_N, sd, is_)
                            VaR = pv.quantile(Rp, LR, alpha, is_)
                            a, b = pv.ecsa_reference(R, LR, VaR, 10.0 ** p, Mo, order == 'd2')
                            vals.append(b if order == 'd2' else a)
                        vals = np.array(vals); ms.append(vals.mean()); ss.append(vals.std(ddof=1) / np.sqrt(FIG_NREP))
                    data[(order, alpha, tag, Mo)] = (ms, ss)
        print(f"{order} done", flush=True)
    return data


def plot_figure(data, out):
    matplotlib.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.size': 11,
                                'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                                'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
                                'legend.fontsize': 7.5})
    P = FIG_POWERS; KSE = 3.0
    Cexact = 'k'; Cmc = '#1f77b4'; Cis = '#d62728'
    # MC blue, IS red; M=1 circle solid line, M=2 square dashed line (open marker)
    styles = {('mc', 1): (Cmc, 'o', '-', 'MC, ECSA $M=1$'),
              ('mc', 2): (Cmc, 's', '--', 'MC, ECSA $M=2$'),
              ('is', 1): (Cis, 'o', '-', 'IS, ECSA $M=1$'),
              ('is', 2): (Cis, 's', '--', 'IS, ECSA $M=2$')}
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for col, alpha in enumerate(ALPHAS):
        _, d1e, d2e = pv.exact(alpha)
        for row, (od, exval, lab) in enumerate([('d1', d1e, r'$\mathrm{d}q_\alpha/\mathrm{d}w_3$'),
                                                ('d2', d2e, r'$\mathrm{d}^2 q_\alpha/\mathrm{d}w_3^2$')]):
            a = ax[row, col]
            a.axhline(exval, color=Cexact, ls='-', lw=1.3, label='Exact', zorder=1)
            for (tag, Mo), (col_, mk, ls, nm) in styles.items():
                m = np.array(data[(od, alpha, tag, Mo)][0]); se = np.array(data[(od, alpha, tag, Mo)][1])
                a.plot(P, m, color=col_, ls=ls, lw=1.2, marker=mk, ms=4, label=nm,
                       markerfacecolor='none' if Mo == 2 else col_)
                a.plot(P, m - KSE * se, color=col_, ls=':', lw=0.6); a.plot(P, m + KSE * se, color=col_, ls=':', lw=0.6)
            a.set_xlabel(r'$\log_{10} h$'); a.set_ylabel(lab)
            a.set_title(rf'{"First" if row == 0 else "Second"} derivative  $\alpha={alpha}$')
            a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
            if row == 0 and col == 0: a.legend(loc='best', ncol=1)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"saved {out}", flush=True)
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("pypdfium2 not installed: PNG preview skipped", flush=True)
        return
    png = out.with_name(out.stem + "_preview.png")
    pdfium.PdfDocument(str(out))[0].render(scale=1.0).to_pil().save(png)
    print(f"saved {png}", flush=True)


def run_figure():
    data = run_figure_data()
    rows = []
    for order in ('d1', 'd2'):
        n = 1 if order == 'd1' else 2
        for alpha in ALPHAS:
            rows.append(dict(application="portfolio_var", model=pv.MODEL_NAME, alpha=alpha,
                             derivative_order=n, method="Exact", estimate=pv.exact(alpha)[n]))
            for tag, _ in SAMPLINGS:
                for Mo in (1, 2):
                    ms, ss = data[(order, alpha, tag, Mo)]
                    for p, mval, sval in zip(FIG_POWERS, ms, ss):
                        rows.append(dict(application="portfolio_var", model=pv.MODEL_NAME, alpha=alpha,
                                         sampling=tag.upper(), derivative_order=n, method=f"ECSA M={Mo}",
                                         ecsa_M=Mo, ecsa_phi=ecsa_phi(n, Mo), log10_h=p, h=10.0 ** p,
                                         estimate=mval, se=sval, N=FIG_N, replications=FIG_NREP))
    write_csv(fig_path(f"{PREFIX}_hsweep_data.csv"), hsweep_columns(["alpha", "sampling"]), rows)
    plot_figure(data, fig_path(f"{PREFIX}_hsweep.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("experiment", choices=["table", "figure", "all", "verify"])
    parser.add_argument("--alpha", type=float, nargs="+", default=ALPHAS,
                        help="quantile levels of the table (default: 0.01 0.05 0.1)")
    args = parser.parse_args()
    if args.experiment == "verify":
        worst = pv.verify_ecsa(ALPHAS, range(NS), N, H)
        print(f"max relative difference reference vs fast ECSA: {worst:.3e}")
    if args.experiment in ("table", "all"):
        run_table(args.alpha)
    if args.experiment in ("figure", "all"):
        run_figure()
