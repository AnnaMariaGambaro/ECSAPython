# ECSA numerical applications

Python code for the numerical results of the paper
*Higher-order sensitivity estimation for discontinuous performance functions:
a new complex-step method*. It reproduces Tables 3–7 and Figures 1–5, the
three applications of the Extended Complex Step Approximation (ECSA) to
first- and second-order sensitivities of expectations of discontinuous
performance functions.

| Application | Folder | Paper | Sensitivity |
|---|---|---|---|
| Control charts (Shewhart chart, exponential change point) | `control_charts/` | Table 3, Figure 1 | ARL derivatives w.r.t. the upper control limit $\theta_2$ |
| Barrier options (Gaussian, NIG, Heston) | `barrier_options/` | Tables 4–6, Figures 2–4 | up-and-out call derivatives w.r.t. the barrier level $B$ |
| Portfolio value-at-risk | `portfolio_var/` | Table 7, Figure 5 | VaR derivatives w.r.t. a portfolio weight $w_3$ |

In every application ECSA ($M=1,2$) is compared with exact values or a
deterministic benchmark (where available), GLR, and Richardson-extrapolated
finite differences (FD (Rich.), GLR+FD (Rich.); Hong's estimator for VaR).
Details are in each application's README.

## Repository structure

```
CodePython-final/
├── README.md                      this file
├── requirements.txt               Python dependencies
├── .gitignore
├── barrier_options/
│   ├── README_barrier.md
│   ├── barrier_option_gaussian_main.py    Table 4, Figure 2
│   ├── barrier_option_nig_main.py         Table 5, Figure 3
│   ├── barrier_option_heston_main.py      Table 6, Figure 4
│   ├── barrier_option_{gaussian,nig,heston}_model.py
│   ├── barrier_option_common.py   shared by the three barrier models
│   ├── fig/                       figures + plotted data (CSV)
│   └── tables/                    tables (CSV)
├── control_charts/
│   ├── README_control_chart.md
│   ├── control_chart_main.py      Table 3, Figure 1
│   ├── control_chart_model.py
│   ├── fig/
│   └── tables/
├── portfolio_var/
│   ├── README_VaR.md
│   ├── portfolio_var_main.py      Table 7, Figure 5
│   ├── portfolio_var_model.py
│   ├── fig/
│   └── tables/                    tables (CSV) + full results (JSON)
└── utils/
    ├── __init__.py
    └── ecsa_utils.py              shared by all applications
```

`*_main.py` files are the executable scripts; `*_model.py` and
`barrier_option_common.py` are supporting modules of their application;
`utils/ecsa_utils.py` holds what all applications share (Möbius function,
ECSA angles and prefactor, Richardson extrapolation and step-pair selection,
replication summaries, output folders, CSV export).

## Installation

Python 3.9 or later (results produced with Python 3.13.3 on Windows 11).

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on Linux/macOS), then:

```bash
pip install -r requirements.txt
```

`pypdfium2` and `Pillow` are optional (listed, commented out, in
`requirements.txt`): they only create the PNG preview of the VaR figure.

## Running the experiments

Every main script takes `table`, `figure` or `all`, and can be run from any
working directory. From the repository root:

```bash
python control_charts/control_chart_main.py table
python control_charts/control_chart_main.py figure
python barrier_options/barrier_option_gaussian_main.py table
python barrier_options/barrier_option_gaussian_main.py figure
python barrier_options/barrier_option_nig_main.py table
python barrier_options/barrier_option_nig_main.py figure
python barrier_options/barrier_option_heston_main.py table
python barrier_options/barrier_option_heston_main.py figure
python portfolio_var/portfolio_var_main.py table
python portfolio_var/portfolio_var_main.py figure
```

Further options (`--help` on any script): `--l` (monitoring frequencies),
`--path` (ECSA complex path) and `--calibration` (NIG) for the barrier
scripts; `--alpha` for the VaR table; `verify` for the control-chart and VaR
consistency checks.

Approximate run times on the reference machine: the control-chart table about
1 hour ($N=10^7$ for the second derivative), the NIG figure about 40 minutes,
the other experiments from a few minutes to about 20 minutes.

## Outputs

Each script writes into its own application folder: figures and the data they
plot in `fig/`, tables in `tables/`, with names starting with the script name
(e.g. `barrier_options/tables/barrier_option_nig_amzn_l10_table.csv`).
Re-running a script overwrites its previous outputs.

All tables share one CSV structure, one row per method and configuration:
`application, model, <configuration>, derivative_order, method, ecsa_M,
ecsa_phi, h, estimate, se, abs_error, time_s, fd_delta, fd_delta_half, N,
replications` (configuration = `l` for barrier options, `alpha, sampling` for
VaR, none for control charts). Empty fields mean "not applicable"; numbers are
stored at full precision.

## Reproducibility

- Random numbers come from `numpy.random.default_rng` with fixed seeds, so
  estimates and standard errors are reproduced exactly with the same library
  versions; within a replication all methods share the same samples (common
  random numbers).
- Reported computational times are wall-clock measurements and vary with the
  machine and its load. Run the `table` commands alone on the machine.
- Sample sizes, seeds, model parameters and ECSA settings of each experiment
  are listed in the application READMEs.
