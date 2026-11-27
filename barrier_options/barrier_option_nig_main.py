"""
Barrier option, NIG model (AMZN calibration) -- main script.

    python barrier_option_nig_main.py table    # Table tab:barrierB-NIG (Table 5)
    python barrier_option_nig_main.py figure   # Figure fig:barrierB-NIG (Figure 3)
    python barrier_option_nig_main.py all      # both
    (optional: --path exact|jet2, default exact; --l 10 20 30; --calibration amzn|schoutens, default amzn)

ECSA differentiates directly with respect to the barrier level, theta = B
(Section 4.3 of main-v13.tex): evaluation points vartheta = B + h e^{i angle},
h in units of B, no ln B parametrization and no chain rule.

Table: Benchmark (CONV/FFT pricer with the NIG characteristic function +
central FD), ECSA M=1/M=2, GLR, FD (Rich.) for the first derivative;
Benchmark, ECSA M=1/M=2, GLR+FD (Rich.) for the second derivative.
N = 2e6 per replication, 10 replications (seeds 0..9), h = 1e-5.

Figure: ECSA M=1/M=2 versus log10 h in {-9,...,-3,-2.5,-2}, N = 5e5,
20 replications (seeds 0..19), +-3 SE bands, benchmark line.

Outputs (in tables/ and fig/), <cal> = calibration tag (amzn):
    barrier_option_nig_<cal>_l{l}_table.csv
    barrier_option_nig_<cal>_l{l}_table_richardson_pairs.csv
    barrier_option_nig_<cal>_l{l}_hsweep_der{1,2}.pdf
    barrier_option_nig_<cal>_l{l}_hsweep_data.csv

Formerly nig_table8_amzn.py (table) and nig_barrier_h_sweep.py (figure).
"""
import sys
from pathlib import Path

# Project root on the import path, so that the shared utils/ package is found
# whatever the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import barrier_option_nig_model as model
from barrier_option_common import TableConfig, FigureConfig, run_barrier_main

PREFIX = "barrier_option_nig"
TABLE = TableConfig(N=2_000_000, reps=10, h1=1e-5, h2=1e-5)
FIGURE = FigureConfig(N=500_000, nrep=20, powers=(-9, -8, -7, -6, -5, -4, -3, -2.5, -2),
                      reuse_paths_across_h=False)


def titles(l):
    lab = model.CALIB.label
    return rf"NIG ({lab}) first derivative $l={l}$", rf"NIG ({lab}) second derivative $l={l}$"


def calibration_option(parser):
    parser.add_argument("--calibration", choices=sorted(model.CALIBRATIONS), default="amzn",
                        help="NIG parameter set (default: amzn, as in the manuscript)")

    def apply(args, prefix):
        calib = model.set_calibration(args.calibration)
        return f"{prefix}_{calib.name}"
    return apply


if __name__ == "__main__":
    run_barrier_main(model, PREFIX, TABLE, FIGURE, titles, __doc__.splitlines()[1],
                     extra_args=calibration_option)
