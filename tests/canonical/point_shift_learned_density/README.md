# A point additive shift with a learned density

This directory will hold the reporting study `point-shift-learned-density` (X121 part b2). The
declaration is committed before any run. No declared run has been made, and the study is not yet
in the evidence register. The design, the sizing rule and the reading declared in advance are in
`tests/studies/point_shift_learned_density.py`.

| part | declaration |
| --- | --- |
| law | `shift_dgp()` at its defaults (the law of `make_shift_dose`), n = 3,000 |
| parameters | three contrasts against the natural course (+0.5 capped at 5, +0.5, +1.0) and the four policy means; truths fixed by the law |
| arms | `in_sample_learned` (the interventions page's fit: degree-2 least squares, regularized booster density, 40 core bins, in sample); `crossfit_learned` (the same, three folds, outcome range (-7.89, 35.91) from the law's 1e-10 tails); `in_sample_oracle` (the exact density on the fit's grid, diagnostic) |
| primary | 1,200 draws, all three arms on each draw: bias leg, coverage floor 0.90, SE sanity [0.8, 1.2] |
| `interval_calibration` | the cross-fitted arm's three contrasts with the shrunken-SE (0.70) and noise controls; count from the pilot by the per-cell predictive rule at `max(C_FIT, pilot c)`, capped, never above 19,600 |
| comparator | none |

## Running

Stage 1, the sizing pilot (3,200 cross-fitted fits, splittable by range):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m tests.studies.point_shift_learned_density_pilot --raw <scratch>/pilot \
        --jobs <cores> --first <r0> --last <r1>
    python -m tests.studies.point_shift_learned_density_pilot --raw <scratch>/pilot --summarize

The summary writes `tests/studies/point_shift_learned_density_pilot.csv`. The design test then
names the calibration count, which is set before stage 2.

Stage 2, the declared run (primaries, then the calibration family; resumable):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m tests.canonical.point_shift_learned_density.regenerate --output <empty scratch dir>
