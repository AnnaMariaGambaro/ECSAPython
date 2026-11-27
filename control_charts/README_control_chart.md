# README — Shewhart Control-Chart Application

Reproducibility notes for section `Control Charts` of
`Higher-order sensitivity estimation for discontinuous performance functions: a new complex-step method`: 
**Table 3** and **Figure 1**.

The application estimates the first and second derivatives of the average run
length $\operatorname{ARL}(\theta_2)$ of a Shewhart chart with an exponential
change point, with respect to the upper control limit $\theta_2$. ECSA
($M=1,2$, literal formula of the manuscript) is compared with the exact
closed-form derivatives, GLR (first derivative), FD (Rich.) and GLR+FD (Rich.).

---

## 1. Files

| File | Type | Role |
|---|---|---|
| `control_chart_main.py` | main | `table` → Table 3; `figure` → Figure 1; `verify` → optional reference-vs-wavefront ECSA check. Formerly `control_chart_table_main.py` + `control_chart_plot_main.py`. |
| `control_chart_model.py` | supporting | Model parameters; exact ARL derivatives (`arl_and_derivatives`); ECSA evaluation points; the wavefront kernel `wavefront_accumulate`, shared by the table (`run_ecsa`) and the figure (`run_ecsa_hsweep`); per-path reference ECSA (`run_ecsa_reference`, `verify_wavefront`); GLR (`run_glr`); FD (Rich.) / GLR+FD (Rich.) with a common long buffer (`run_fd_richardson`). |
| `../utils/ecsa_utils.py` | supporting (shared) | Shared by all applications: Möbius function, ECSA angles and prefactor, Richardson extrapolation and pair selection, output paths, CSV writer. |

Previously, the exact formula, the Möbius function and the wavefront loop were
duplicated in the table and plot scripts; they now have one implementation.

## 2. Dependencies

Python 3.13, NumPy, SciPy (`scipy.stats.norm`), Matplotlib.

## 3. How to run

Run from this folder as shown, or from the repository root as
`python control_charts/<script> ...`; the scripts work from any working directory
and write their outputs to `control_charts/fig` and `control_charts/tables`.

```bash
python control_chart_main.py table    # Table 3 (about 1 hour; N=1e7 for the second derivative)
python control_chart_main.py figure   # Figure 1 (a few minutes)
```

`all` runs both. `verify` re-runs the statistical check between the slow
per-path implementation of the literal ECSA formula and the wavefront
implementation ($N=10^5$, four $(n,M)$ configurations). The table measures
wall-clock times: run it alone on the machine.

## 4. Outputs

| Output | Content |
|---|---|
| `tables/control_chart_table.csv` | Table 3 (standardized structure: `application, model, derivative_order, method, ecsa_M, ecsa_phi, h, estimate, se, abs_error, time_s, fd_delta, fd_delta_half, N, replications`) |
| `tables/control_chart_table_richardson_pairs.csv` | The four candidate Richardson pairs of FD (Rich.) and GLR+FD (Rich.), `selected = 1` for the chosen one |
| `fig/control_chart_hsweep.png` | Figure 1 (formerly `fig/app2_ecsa_vs_h.png`) |
| `fig/control_chart_hsweep_data.csv` | Data of Figure 1 (exact values, ECSA means and SEs for every $h$) |

Rows of the table CSV follow the manuscript: `Exact`, `ECSA M=1`, `ECSA M=2`,
`GLR`, `FD (Rich.)` (first derivative, $N=10^6$) and `Exact`, `ECSA M=1`,
`ECSA M=2`, `GLR+FD (Rich.)` (second derivative, $N=10^7$). SEs are pathwise
(single long run), so `replications` is empty; `abs_error` = |estimate − exact|
at full precision; `time_s` in seconds. Empty field = not applicable.

## 5. Parameters and simulation settings

| Parameter | Value |
|---|---|
| $\theta_1,\theta_2$ | $-2.81$, $2.81$ |
| $\mu_0,\mu_1,\sigma$ | $0$, $1$, $1$ |
| $\bar R$ (`ZBAR`) | $20$ (mean of the exponential change point) |
| ECSA | $h=10^{-5}$ (table); $\phi=\pi/2,\pi/3$ ($n=1$), $\pi/4,\pi/6$ ($n=2$) for $M=1,2$ |
| $N$ (table) | $10^6$ (first derivative, all methods), $10^7$ (second derivative, all methods); batches of $2\times10^5$ (ECSA, GLR) and $5\times10^4$ (FD) |
| Richardson pairs | $\{(0.025,0.0125),(0.05,0.025),(0.1,0.05),(0.2,0.1)\}$, finest first, 3×-SE back-off; time = cost of all five deltas |
| CRN buffer | 3000 observations per replication, exit rule re-applied for $\theta_2\pm\delta$ |
| Figure | $\log_{10}h\in\{-7,-6,-5,-4,-3,-2.5,-2,-1.8\}$, $N=10^6$, one shared wavefront simulation for all $h$ and configurations, ±3 SE bands |

Seeds: table ECSA and GLR use `default_rng(42)` (each method its own stream
from seed 42); FD-type methods use `default_rng(42 + 1000)`; the figure uses
`default_rng(10)`. Points shared by the $k=1$ and $k=2$ Möbius terms are
evaluated twice, following the literal formula.


