"""
Barrier option, Heston model -- main script.

    python barrier_option_heston_main.py table    # Table tab:barrierB-heston (Table 6)
    python barrier_option_heston_main.py figure   # Figure fig:barrierB-heston (Figure 4)
    python barrier_option_heston_main.py all      # both
    (optional: --path exact|jet2, default exact; --l 10 20 30)

ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex): evaluation points vartheta = B + h e^{i angle},
h in units of B, no ln B parametrization and no chain rule.

Table: ECSA M=1/M=2, GLR, FD (Rich.) for the first derivative; ECSA M=1/M=2,
GLR+FD (Rich.) for the second derivative. No deterministic benchmark exists
for this model, so no AE is reported. N = 1e6 per replication, 8 replications
(seeds 0..7), h = 1e-5 (first derivative) and h = 1e-4 (second derivative).

Figure: ECSA M=1/M=2 versus log10 h in {-9,...,-3}, N = 5e5, 20 replications
(seeds 0..19), +-3 SE bands, no benchmark line. Each seed is simulated once
and reused for every h (as in the original figure script).

Outputs (in tables/ and fig/):
    barrier_option_heston_l{l}_table.csv
    barrier_option_heston_l{l}_table_richardson_pairs.csv
    barrier_option_heston_l{l}_hsweep_der{1,2}.pdf
    barrier_option_heston_l{l}_hsweep_data.csv

Formerly heston_table_driver.py + heston_fd_rich_d1.py (table; the FD (Rich.)
row is now computed in the same replication loop, on the same simulated
paths) and heston_barrier_h_sweep.py (figure).
"""
import sys
from pathlib import Path

# Project root on the import path, so that the shared utils/ package is found
# whatever the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import barrier_option_heston_model as model
from barrier_option_common import TableConfig, FigureConfig, run_barrier_main

PREFIX = "barrier_option_heston"
TABLE = TableConfig(N=1_000_000, reps=8, h1=1e-5, h2=1e-4)
FIGURE = FigureConfig(N=500_000, nrep=20, powers=(-9, -8, -7, -6, -5, -4, -3),
                      reuse_paths_across_h=True)


def titles(l):
    return rf"Heston first derivative $l={l}$", rf"Heston second derivative $l={l}$"


if __name__ == "__main__":
    run_barrier_main(model, PREFIX, TABLE, FIGURE, titles, __doc__.splitlines()[1])
