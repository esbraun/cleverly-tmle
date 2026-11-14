"""Regenerate the ``known-mechanism-sample-weights`` evidence (Fix-16) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_known_sample_weights`,
:mod:`tests.studies.known_sample_weights_properties` and
:mod:`tests.studies.known_sample_weights_law`, fixed before the run and recomputed by
``tests/unit/test_known_sample_weights_design.py`` from the committed pilot
(``tests/studies/known_sample_weights_pilot.csv``).  Before the run, the declared smoke gate
``python -m tests.studies.known_sample_weights_properties smoke --draws 2`` must exit 0.

Red-cell reading (declared before any run; Fix-16 plan section 7 as amended by rev 2.1 item 1
and the design review, in the terms of the plans guide's "Red cells").  Every one of the 26
property cells is gated.  The designed chance that some gated row is red when every cell
behaves as its pilot did is 1 - 0.956 = 0.044 (predictive) and 1 - 0.988 = 0.012 (plug-in).

* A red ``interval_calibration`` positive, at any size and on either law, is a finite-sample red
  only when the 99% interval of its SE ratio contains 1, its coverage interval contains 0.95 and
  the diagnosis finds no defect.  It then goes to ``reporting`` with an owner in the roadmap's
  Red-cell owners table, and the support row carries the measured numbers.  Any other red
  calibration positive is a defect suspect: it blocks the merge and goes to remediation, and a
  re-run happens only under a fresh declaration that records what changed and why.  A
  shortfall at n = 1,000 is read by this rule too; it is never answered by a run at a larger
  size or with more replications.
* A derived control (``__shrunken_se_control``, ``__noise_control``) that does not leave the
  band is a red of the study, not of the estimator.  The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* A red ``sample_bias`` positive (the 99% interval of the standardized bias not inside 0.25) is
  a defect suspect in the ``"sample"`` route and blocks the merge.
* A red ``sample_misspecified`` positive is a defect suspect in the tilt route and blocks the
  merge.  This family is the study's direct witness of the tilt: with the outcome regression
  wrong, only a correct tilted mechanism removes the bias, and the control's paired difference
  is its explanation.  Its SE ratio and coverage are published, ungated.
* A red ``tilt_misspecified`` positive is a defect suspect in the ``"sample"`` route's use of the
  outcome regression (the double robustness), and blocks the merge.
* A control (``target_reading_control``, ``both_misspecified``) that fails to fail is a design
  bug or a defect suspect.  The exact-law biases are -0.0330 on law SW (-1.0, -2.0 and -4.0
  pilot SDs at n = 1,000, 4,000 and 16,000), +0.4550 on law LW (+6.8 SDs) and -0.0348 for the
  both-wrong cell (-2.1 SDs).  A control inside the margin means the fit did not divide by
  what it should or did not use the declared learner.  It is diagnosed with the exact-law limit
  and fixed under a fresh declaration.
* A red primary row reads as a calibration positive: finite-sample only on the three conditions
  above, else a defect suspect.
* Each control's ``exact_bias`` and ``paired_difference_over_sd``, and the SE ratio and coverage
  of the bias cells, are published and turn no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Pilot.  ``tests/studies/known_sample_weights_pilot.py``, 8,000 draws of each block from its own
seed stream at the declared n, learners and fits, every cell of a size on the same draws; the
design test reads the committed summary.  SE ratio, coverage and standardized bias of the
calibration cells: law SW 1.012, 0.9489, 0.001 at n = 1,000; 0.995, 0.9470, 0.003 at 4,000;
0.995, 0.9490, -0.010 at 16,000; law LW (five folds, declared outcome bounds) 0.998, 0.9496,
-0.010 at 4,000.  Per-draw SE-ratio spread 0.710 to 0.748 (0.707 for normal estimates with an
exact SE).  Kish effective sample size 0.65 n (SW) and 0.63 n (LW).  The outcome-wrong
positives read 0.004, 0.014, -0.008 SD (SW) and -0.051 SD (LW); the tilt-wrong positive -0.009
SD.  The controls read -1.00, -1.98, -3.99 SD (SW), +6.76 SD (LW) and -2.05 SD (both wrong), and
each partner removes all of it on the same draws.

Sizes.  Every property cell runs 4,000 replications and the primary 2,000.  4,000 is the
smallest common multiple of 400 at which the predictive family-wise pass probability of the 26
gated cells is at least 0.95 (0.956; plug-in 0.988), with the SE ratio's interval scaled on the
log scale; the n = 1,000 calibration cell binds (0.970 predictive).  The bias cells pass with
probability 1.0 to four places, and the derived controls above 0.9999.

Budget.  The pilot's mean seconds per fit give 0.65 CPU-hours of fits for the declared run on
the 4-vCPU cloud machine, about 1.3 CPU-hours with the harness (the bootstrap intervals, the
derived controls and the accounting): about 30 to 45 minutes of wall time at four jobs.
"""

import sys
from pathlib import Path

# Run as ``python tests/canonical/known_sample_weights/regenerate.py``, Python puts this
# directory on the path rather than the repository root, so ``tests`` would not import.
_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.canonical.declared_run import run  # noqa: E402
from tests.studies import (  # noqa: E402
    canonical_known_sample_weights,
    known_sample_weights_properties,
)

if __name__ == "__main__":
    run(
        canonical_known_sample_weights,
        known_sample_weights_properties,
        here=Path(__file__).resolve().parent,
    )
