# README — Portfolio Value-at-Risk Sensitivities Application

Reproducibility notes for section `Portfolio Value-at-Risk sensitivities` of 
`Higher-order sensitivity estimation for discontinuous performance functions: a new complex-step method`: 
**Table 7** and **Figure 5**.

The application estimates the first and second derivatives of the lower
$\alpha$-quantile $q_\alpha$ of a three-asset Gaussian portfolio return with
respect to the weight $\theta=w_3$ (Hong 2009, Sec. 7.1), under plain Monte
Carlo (MC) and importance sampling (IS). ECSA ($M=1,2$) is compared with the
exact values, Hong's window estimator, GLR (Peng, Fu, Hu & Heidergott 2018),
FD (Rich.), Hong+FD (Rich.) and GLR+FD (Rich.).


---

## 1. Files

| File | Type | Role |
|---|---|---|
| `portfolio_var_main.py` | main | `table` → Table 7 (CSV + full JSON); `figure` → Figure 5 (data + plot).  |
| `portfolio_var_model.py` | supporting | Model and sampling (MC/IS), exact derivatives, empirical quantile, closed-form density terms; ECSA reference implementation (`ccdf`, `F_theta`, `F_thetatheta`, `ecsa_reference`, used by the figure) and optimized implementation (`ecsa_d1_fast`, `ecsa_d2_fast`, used by the table); Hong, GLR and the FD-type estimators. |
| `../utils/ecsa_utils.py` | supporting (shared) | Shared by all applications: Möbius function, ECSA angles, Richardson extrapolation and pair selection, summaries, timing helper, output paths, CSV writer. |

## 2. Dependencies

Python 3.13, NumPy, SciPy, Matplotlib. Optional: `pypdfium2` (PNG preview of
the figure; skipped with a message if not installed).

## 3. How to run

Run from this folder as shown, or from the repository root as
`python portfolio_var/<script> ...`; the scripts work from any working directory
and write their outputs to `portfolio_var/fig` and `portfolio_var/tables`.

```bash
python portfolio_var_main.py table    # Table 7 (about 10 minutes)
python portfolio_var_main.py figure   # Figure 5 (a few minutes)
```

`all` runs both; `table --alpha 0.05` restricts the quantile levels.
The table measures wall-clock times: run it alone on the machine. 

## 4. Outputs

| Output | Content |
|---|---|
| `tables/portfolio_var_table.csv` | Table 7 (standardized structure, below) |
| `tables/portfolio_var_table_richardson_pairs.csv` | Four candidate Richardson pairs of FD, Hong+FD and GLR+FD for every (α, MC/IS), `selected = 1` for the chosen pair |
| `tables/portfolio_var_table_results.json` | Full results, same structure as the former `var_table_v2_results.json` (raw replication values, raw FD values per δ, all pairs, simulation time) |
| `fig/portfolio_var_hsweep.pdf` | Figure 5 (formerly `fig/Fig8-AppVaR.pdf`; `Fig8-AppVaR_v2.pdf` in the manuscript) |
| `fig/portfolio_var_hsweep_preview.png` | PNG preview (formerly `fig/var_fig4_prev.png`) |
| `fig/portfolio_var_hsweep_data.csv` | Data of Figure 5 (formerly `var_fig4.json`) |

Table CSV columns: `application, model, alpha, sampling, derivative_order,
method, ecsa_M, ecsa_phi, h, estimate, se, abs_error, time_s, fd_delta,
fd_delta_half, N, replications`. Rows: for each derivative order, the `Exact`
row of each α (with the time of the closed-form formulas; `sampling` empty),
then the methods under `MC` and under `IS` in the order of Table 7, one row
per α. `abs_error` = |estimate − exact| at full precision; `time_s` = seconds
summed over the 20 replications.

## 5. Model and simulation settings 

$R=\mu+CZ$, $Z\sim N(0,I_3)$, $CC^\top=\Sigma$, $\mu=(0.06,0.15,0.25)$,
volatilities $(0.02,0.10,0.22)$, $\rho_{12}=-0.3$, $\rho_{13}=-0.2$,
$\rho_{23}=0.2$, $w=(0.2,0.3,0.5)$, $\alpha\in\{0.01,0.05,0.10\}$.

| Setting | Table | Figure |
|---|---|---|
| $N$ per replication | $10^6$ | $3\times10^5$ |
| Replications / seeds | 20, `default_rng(seed)`, seeds 0–19 | 4, seeds 0–3 |
| ECSA $h$ | $10^{-5}$ | $\log_{10}h\in\{-7,-5,-4,-3,-2.5\}$ |
| ECSA implementation | optimized (`*_fast`) | literal reference |
| ECSA angles | $\phi=\pi/2,\pi/3$ ($n=1$), $\pi/4,\pi/6$ ($n=2$) for $M=1,2$ | same |

- One sample per (α, seed, MC/IS) is shared by all methods (CRN); its
  simulation is timed separately and not billed to any method.
- IS: mean shift $d=\frac{q_\alpha-\mu_p}{\sigma_p^2}C^\top w$ with the exact
  $q_\alpha$; LR-weighted empirical quantile. Plain MC uses `np.quantile`.
- GLR window and Hong window: $\sigma_p N^{-1/3}$.
- Richardson pairs $\{(0.0025,0.00125),(0.005,0.0025),(0.01,0.005),(0.02,0.01)\}$,
  finest first, 3×-SE back-off; time = cost of all five deltas.
- SE (all MC methods) = sample SD (ddof=1) of the 20 replication estimates / √20.



