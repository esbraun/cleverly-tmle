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

Red-cell reading (declared before any run; Fix-16 plan section 7 as amended by rev 2.1 item 1,
in the terms of the plans guide's "Red cells").  The designed chance that some gated row is red
when every cell behaves as its pilot did is 1 - 0.962 = 0.038 (predictive) and 1 - 0.991 =
0.009 (plug-in).

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
* A ``target_reading_control`` that fails to fail is a design bug or a defect suspect: its
  exact-law bias is -0.0330, which is -1.0, -2.0 and -4.0 pilot SDs at n = 1,000, 4,000 and
  16,000, so a control inside the margin means the fit did not divide by the declaration or did
  not use the declared outcome regression.  It is diagnosed with the exact-law limit and fixed
  under a fresh declaration.
* A red primary row reads as a calibration positive: finite-sample only on the three conditions
  above, else a defect suspect.
* Every ``sample_misspecified`` row, and each control's ``exact_bias`` and
  ``paired_difference_over_sd``, is published and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Pilot.  ``tests/studies/known_sample_weights_pilot.py``, 8,000 draws of each block from its own
seed stream at the declared n, learners and fits; the design test reads the committed summary.
SE ratio, coverage and standardized bias of the gated cells: law SW 1.012, 0.9489, 0.001 at
n = 1,000; 0.995, 0.9470, 0.003 at 4,000; 0.995, 0.9490, -0.010 at 16,000; law LW 1.002,
0.9471, -0.012 at 4,000.  Kurtosis of the estimates 3.00 to 3.10, SE coefficient of variation
0.012 to 0.096, per-draw SE-ratio spread 0.710 to 0.726 (0.707 for normal estimates with an exact
SE).  Kish effective sample size 0.65 n (SW) and 0.63 n (LW).  The control reads -1.00, -1.98
and -3.99 SDs, and the reporting cell removes all of it on the same draws.

Sizes.  Every property cell runs 4,000 replications and the primary 2,000.  4,000 is the
smallest common multiple of 400 at which the predictive family-wise pass probability of the 19
gated cells is at least 0.95 (0.962; plug-in 0.991); the n = 1,000 calibration cell binds.
The controls pass with probability 1.0 to six places, and the derived controls above 0.9999.

Budget.  The pilot's mean seconds per fit give 0.43 CPU-hours of fits for the declared run on
the 4-vCPU cloud machine, about 1 CPU-hour with the harness (the bootstrap intervals, the
derived controls and the accounting): about 20 to 30 minutes of wall time at four jobs.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_known_sample_weights, known_sample_weights_properties

if __name__ == "__main__":
    run(
        canonical_known_sample_weights,
        known_sample_weights_properties,
        here=Path(__file__).resolve().parent,
    )
