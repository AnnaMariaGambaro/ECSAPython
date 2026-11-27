# README — Barrier-Option Application (Gaussian, NIG, Heston), ECSA with θ = B

Reproducibility notes for Section `Barrier Options` of 
`Higher-order sensitivity estimation for discontinuous performance functions: a new complex-step method`: 
**Tables 4–6** and **Figures 2–4**.

The application estimates the first and second derivatives of a discretely
monitored up-and-out call, with $l$ monitoring dates, with respect to the
**barrier level $B$**, under three models. ECSA ($M=1,2$) is compared with GLR,
FD (Rich.) and GLR+FD (Rich.) and, for the Gaussian and NIG models, with a
deterministic transform-based benchmark.

**The ECSA differentiation parameter is $\theta=B$** The
derivatives $\partial C/\partial B$ and $\partial^2 C/\partial B^2$ are
estimated directly, with complex evaluation points
$\vartheta=B+h e^{i(2\pi j/(kn)+\phi)}$.

---

## 1. How the derivatives with respect to $B$ are computed

Section 4.3 writes $S_{t_j}=S_{t_{j-1}}e^{m_j+\sigma_j Z_j}$ (Black–Scholes,
NIG and Heston differ only in $m_j,\sigma_j$) and uses, for $\vartheta\in\mathbb C$,
the complex path

$$\gamma_{\vartheta,1}(z_1)=z_1+\frac{\ln(\vartheta/B)}{\sigma_1},\qquad
\gamma_{\vartheta,j}=z_j\ (1<j<l),\qquad
\gamma_{\vartheta,l}=\frac{\ln\vartheta-\ln K}{\ln B-\ln K}\big(z_l-\tilde a_l(\mathbf z_{l-1},B)\big)
+\tilde a_l(\gamma_\vartheta(\mathbf z_{l-1}),\vartheta),$$

with Jacobian $(\ln\vartheta-\ln K)/(\ln B-\ln K)$, and the per-path
realizations $f_k(\vartheta)$ of eq. (`eq:fkBarrier`): survival/strike
indicator at the real $B$ × payoff on $\gamma_\vartheta$ × Jacobian × normal
density ratio. The code implements exactly these steps:

1. **Real-path quantities** at the real barrier $B$ (indicators,
   $\tilde a_l$, increments): `chi_cache` (Gaussian) and
   `conditional_gaussian_chi_cache` (NIG, Heston), computed once per
   replication.
2. **Evaluation points.** For derivative order $n$, truncation level $M$ and
   angle $\phi$ (Corollary `th:corollary_phi`: $\pi/2,\pi/3$ for $n=1$;
   $\pi/4,\pi/6$ for $n=2$; $M=1,2$), the set
   $\mathcal S=\{B+he^{i(2\pi j/(kn)+\phi)}\}$ of Table `tab:algorithm`;
   points shared by several Möbius terms are evaluated once
   (`barrier_option_common.ecsa_points`).
3. **Complex path.** $\gamma_\vartheta$ depends on $\vartheta$ only through
   $\ln\vartheta$, so the proxies take $c=\ln(\vartheta/B)$, computed from
   $\vartheta$ by `log_barrier_shift` (principal complex logarithm, evaluated
   as $\tfrac12\operatorname{log1p}(2\Re x+|x|^2)+i\,\operatorname{atan2}(\Im x,1+\Re x)$,
   $x=(\vartheta-B)/B$, accurate for tiny $h$).
4. **Proxy** $f_k(\vartheta)$: `chi_from_cache` (Gaussian) and
   `conditional_gaussian_chi` (NIG, Heston).
5. **ECSA formula** of Table `tab:algorithm` with $\theta=B$:
   $\widehat{C^{(n)}_M}(B)=\frac{n!}{h^n\sin(n\phi)}\sum_{k=1}^M\mu(k)\frac1{kn}
   \sum_{j=0}^{kn-1}\Im\hat f(B+he^{i(2\pi j/(kn)+\phi)})$ —
   `ecsa_from_proxy` (NIG, Heston; sample mean first) and
   `ecsa_pathwise_derivative` (Gaussian; pathwise, then averaged). The result
   is $\partial^n C/\partial B^n$.


**Optional expanded path.** The remark of Section 4.3 allows replacing $\ln\vartheta$ by
its second-order Taylor polynomial $\ln B+\varepsilon/B-\varepsilon^2/(2B^2)$,
$\varepsilon=\vartheta-B$ (path $\gamma^{[2]}_\vartheta$). It is available with
`--path jet2`; outputs then carry the suffix `_jet2`. The default, used for
all delivered outputs, is the exact path `--path exact`. 

**Comparison methods.** GLR (Peng et al. 2018) shifts $z_1$ by
$\ln(\vartheta/B)/\sigma_1$ and differentiates at $\vartheta=B$; the factor
$1/B$ in `glr_raw` is $\mathrm d\ln(\vartheta/B)/\mathrm d\vartheta$ at
$\vartheta=B$, i.e. part of the GLR estimator of $\partial C/\partial B$, not
a conversion of an ECSA result. FD (Rich.) and GLR+FD (Rich.) are central
differences in $B$ (steps $\delta$ in units of $B$); the CONV benchmark is a
central difference in $B$ (step 0.10) of a deterministic price.

## 2. Files

| File | Type | Role |
|---|---|---|
| `barrier_option_gaussian_main.py` | main | Table 4 (`table`) and Figure 2 (`figure`) |
| `barrier_option_nig_main.py` | main | Table 5 and Figure 3 (AMZN calibration; `--calibration schoutens` available) |
| `barrier_option_heston_main.py` | main | Table 6 and Figure 4 |
| `barrier_option_gaussian_model.py` | supporting | Gaussian model: simulation, $\chi$-path proxy and pathwise direct-B ECSA (`ecsa_derivative`, `ecsa_pathwise_derivative`), GLR, crude MC, CONV benchmark |
| `barrier_option_nig_model.py` | supporting | NIG model: inverse-Gaussian subordinator simulation, calibrations, proxy cache, ECSA (`ecsa_d1`, `ecsa_d2` → `ecsa_from_proxy`), GLR, crude MC, CONV benchmark |
| `barrier_option_heston_model.py` | supporting | Heston model: Andersen (2008) exact-variance scheme, proxy cache, ECSA, GLR, crude MC price |
| `barrier_option_common.py` | supporting | `BarrierParams`; CONV/FFT pricer and benchmark; NIG/Heston proxy; **direct-B ECSA** (`log_barrier_shift`, `ecsa_points`, `ecsa_from_proxy`, `set_ecsa_path`); table and $h$-sweep drivers, CSV export, plot, command line |
| `../utils/ecsa_utils.py` | supporting (shared) | Möbius function, ECSA angles and prefactor $n!/(h^n\sin n\phi)$, Richardson extrapolation and pair selection, output paths, CSV writer (unchanged) |

Each model module exposes `simulate`, `ecsa(P, l, B, h, n, M)`, `glr_d1`,
`fd_d1`, `glr_fd_d2` (and `benchmark` for Gaussian/NIG), used by
`barrier_option_common.run_table` / `run_hsweep`.

## 3. Dependencies

Python 3.13, NumPy, SciPy (`scipy.stats.invgauss`, `scipy.stats.ncx2`),
Matplotlib. Used here: Windows 11, Python 3.13.3, NumPy 2.4.2, SciPy 1.17.0,
Matplotlib 3.10.8.

## 4. How to run

Run from this folder as shown, or from the repository root as
`python barrier_options/<script> ...`; the scripts work from any working directory
and write their outputs to `barrier_options/fig` and `barrier_options/tables`.

```bash
python barrier_option_gaussian_main.py table     # Table 4
python barrier_option_gaussian_main.py figure    # Figure 2
python barrier_option_nig_main.py table          # Table 5
python barrier_option_nig_main.py figure         # Figure 3 (longest, ~40 min)
python barrier_option_heston_main.py table       # Table 6
python barrier_option_heston_main.py figure      # Figure 4
```

Options: `all` (table and figure), `--l 10 20 30` (default all three),
`--path exact|jet2` (default `exact`). Run the `table` commands alone on the
machine: they measure wall-clock times.

## 5. Outputs

| Output | Content |
|---|---|
| `tables/barrier_option_<model>_l<l>_table.csv` | Tables 4/5/6, one file per $l$ |
| `tables/barrier_option_<model>_l<l>_table_richardson_pairs.csv` | All four candidate Richardson pairs, `selected = 1` for the chosen one |
| `fig/barrier_option_<model>_l<l>_hsweep_der1.pdf`, `..._der2.pdf` | Figure panels (derivative vs $\log_{10}h$) |
| `fig/barrier_option_<model>_l<l>_hsweep_data.csv` | Plotted data |

`<model>` ∈ {`gaussian`, `nig_amzn`, `heston`}, `<l>` ∈ {10, 20, 30}; with
`--path jet2`, `<model>` becomes e.g. `gaussian_jet2`. File names, CSV columns
(`application, model, l, derivative_order, method, ecsa_M, ecsa_phi, h,
estimate, se, abs_error, time_s, fd_delta, fd_delta_half, N, replications`),
precision (shortest round-trip repr), missing-value convention (empty field)
and plot layout are the same as in `Python151026`. In the CSV, `h` of the
ECSA rows is now a step in $B$.

## 6. Model and simulation settings (unchanged)

| Model | Parameters |
|---|---|
| Gaussian | $S_0=K=100$, $B=110$, $r=0.05$, $\sigma=0.1$, $T=1$ |
| NIG (AMZN) | $S_0=K=100$, $B=130$, $r=0.05$, $T=1$; $(\alpha,\beta,\delta)=(9.873,-4.902,0.853)$ |
| Heston | $\kappa=2$, $\theta_v=0.04$, $\sigma_v=0.3$, $\rho=-0.7$, $v_0=0.04$; $S_0=K=100$, $B=120$, $r=0.05$, $T=1$ |

| Setting | Value |
|---|---|
| Monitoring frequencies | $l\in\{10,20,30\}$ |
| ECSA | $M=1,2$; $\phi$ as in §1; tables: $h=10^{-5}$ (Gaussian, NIG), Heston $h=10^{-5}$ ($n=1$), $10^{-4}$ ($n=2$); $h$ in units of $B$ |
| Benchmark (Gaussian, NIG) | central FD in $B$, step 0.10, of the CONV price $2P(2^{16})-P(2^{15})$ |
| Richardson pairs | $\{(0.5,0.25),(1,0.5),(2,1),(4,2)\}$, finest first, 3×-SE back-off; time = all five deltas |
| Tables | Gaussian/NIG $N=2\times10^6$, 10 reps (seeds 0–9); Heston $N=10^6$, 8 reps (seeds 0–7) |
| Figures | $N=5\times10^5$, 20 reps (seeds 0–19); $\log_{10}h\in\{-9,\dots,-3,-2.5,-2\}$ (Heston $\{-9,\dots,-3\}$) |

Random numbers: `np.random.default_rng(seed)`; within a replication all methods
share the same primitives (common random numbers); the simulation is not
billed to any method.
