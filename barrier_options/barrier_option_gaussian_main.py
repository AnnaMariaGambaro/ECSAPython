"""
Barrier option, Gaussian (Black-Scholes) model -- main script.

    python barrier_option_gaussian_main.py table    # Table tab:barrierB (Table 4)
    python barrier_option_gaussian_main.py figure   # Figure fig:barrierB (Figure 2)
    python barrier_option_gaussian_main.py all      # both
    (optional: --path exact|jet2, default exact; --l 10 20 30 to select the monitoring frequencies)

ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex): evaluation points vartheta = B + h e^{i angle},
h in units of B, no ln B parametrization and no chain rule.

Table: Benchmark (CONV/FFT pricer + central FD), ECSA M=1/M=2, GLR, FD (Rich.)
for the first derivative; Benchmark, ECSA M=1/M=2, GLR+FD (Rich.) for the
second derivative. N = 2e6 per replication, 10 replications (seeds 0..9),
h = 1e-5. Times: wall-clock seconds summed over replications; the shared
simulation of the primitives is not billed to any method.

Figure: ECSA M=1/M=2 versus log10 h in {-9,...,-3,-2.5,-2}, N = 5e5,
20 replications (seeds 0..19), +-3 SE bands, benchmark line.

Outputs (in tables/ and fig/):
    barrier_option_gaussian_l{l}_table.csv
    barrier_option_gaussian_l{l}_table_richardson_pairs.csv
    barrier_option_gaussian_l{l}_hsweep_der{1,2}.pdf
    barrier_option_gaussian_l{l}_hsweep_data.csv

"""
import sys
from pathlib import Path

# Project root on the import path, so that the shared utils/ package is found
# whatever the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import barrier_option_gaussian_model as model
from barrier_option_common import TableConfig, FigureConfig, run_barrier_main

PREFIX = "barrier_option_gaussian"
TABLE = TableConfig(N=2_000_000, reps=10, h1=1e-5, h2=1e-5)
FIGURE = FigureConfig(N=500_000, nrep=20, powers=(-9, -8, -7, -6, -5, -4, -3, -2.5, -2),
                      reuse_paths_across_h=False)


def titles(l):
    return rf"First derivative $l={l}$", rf"Second derivative $l={l}$"


if __name__ == "__main__":
    run_barrier_main(model, PREFIX, TABLE, FIGURE, titles, __doc__.splitlines()[1])
