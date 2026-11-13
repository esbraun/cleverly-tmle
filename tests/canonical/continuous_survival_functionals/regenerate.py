"""Regenerate the continuous-survival-functionals evidence (X6 part (c)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.continuous_survival_functionals` and its
properties module, declared from X6 plan rev 2, section 5.4, and checked by
``tests/unit/test_continuous_survival_functionals_design.py``: 3,200 primary replicates of
``functionals_cf_1k``, 3,200 calibration replicates of the RMST difference and 1,000 risk-only
fits for the targeting-necessity reading, all at n = 1,000.  The pilot (800 draws on its own
stream) runs first; the shared rule then confirms the 3,200 or raises it, up to the registered
study's 7,600, before any declared draw.

Scope.  The record gates 25 targets of the default cross-fitted fit with
``restricted_horizons=(2.32,)``: each arm's risk of each cause and each risk difference at the
median and upper quartile (1.12 and 2.32), each cause's risk ratio there on the log scale, and at
tau = 2.32 the RMTL of each arm and cause, the RMST of each arm and their differences.  Every row
reads the shipped interval: range-logit for a risk on (0, 1) and for an RMTL or RMST on
(0, tau), log for a ratio, identity for a difference until Fix-3 PR 3.  The joint band row is a
second phase that waits for the band thread's PR 2.

Budget.  One fit at n = 1,000 with the restricted horizon took 6.4 to 6.7 s on one core (two
draws, 4 and 5 walk steps), against 5.8 to 5.9 s without it.  At 6.5 s: primary 3,200 fits
(5.8 CPU-h), calibration 3,200 (5.8), necessity 1,000 (1.8) and pilot 800 (1.4), about 15 CPU-h
and 22 with a 1.5 margin, so about 5.5 h on 4 cores.  Cloud.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner row
X6, the run is repeated under the new commit with the same deterministic seeds, and the claim the
cell carries stays experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is
seen.

* A red bias row is a defect suspect of the walk or the restricted rows: every nuisance library
  holds the truth.  Diagnose it against ``tests/unit/test_continuous_survival_functionals.py``
  (the Gateaux-derivative witnesses past the last target time and their mutation controls).
* A red coverage or SE-ratio row on a risk or risk difference at the median, with its SE ratio's
  99% interval below 1 and its bias equivalent, reads as the registered study's early-time
  standard-error finding reaching the median once the walk carries the RMTL targets, and takes
  the route above with the measured ratio stated.  Any other red coverage row is a variance
  defect suspect.
* A red risk-ratio row with a green row for each of its two risks is a defect suspect of the
  log-scale curve.
* The calibration cell red on its SE ratio while its coverage is green reads as a finite-sample
  variance finding of the RMST difference at n = 1,000, and takes the route above with the
  measured ratio stated.  Red on coverage is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The targeting-necessity reading carries no verdict.  If the plug-in RMTL's bias is inside the
  0.2 SD margin, the reading says this law does not separate the plug-in from the TMLE, and the
  claim rests on the score test of the fast tier.
* A replication that raises is never redrawn.  A ``DataError`` (an initial hazard fit whose
  survival reaches zero before tau, or whose combined jump reaches one) is a recordable fit
  failure, bounded by the failure bound of 0.01.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_functionals,
    continuous_survival_functionals_properties,
)

if __name__ == "__main__":
    run(
        continuous_survival_functionals,
        continuous_survival_functionals_properties,
        here=Path(__file__).resolve().parent,
    )
