"""Regenerate the ``hazard-msm`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_msm` and its properties
module, fixed before any run and checked by ``tests/unit/test_hazard_msm_study_design.py``.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with an owner row
and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the coefficient walk, the projection or its
  influence curve: the truths are exact projections and every learner is correctly specified.
  Diagnose it against ``tests/unit/test_hazard_msm.py``, fix a defect, and re-run under a fresh
  declaration.
* A red positive ``double_robustness`` cell is a defect suspect: its exact-law limit is the
  truth (``test_every_positive_limit_is_the_truth``), including the ``competing`` death
  coefficient under the wrong hazard learner.  A ``both_wrong`` control that does not clear the
  margin is a failed control: its exact-law gap is -2.10 efficiency SEs
  (``test_the_both_wrong_control_is_discriminated``), so the family's verdict is void and the
  study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the wrong learner's exact plug-in limit
  (``test_the_wrong_hazard_learner_rule``).
* A red calibration row reads as a finite-sample finding only when all three conditions of
  the plans guide (``guide/studies.md``) and plan X70 section 5 hold: the 99% interval of the SE ratio contains
  1; coverage is consistent with 0.95; the diagnosis finds no defect.  It then takes the route
  above with the measured ratio stated.  Any other red calibration row is a defect suspect and
  blocks the pull request.
* The smallest ``root_n_and_efficiency`` rung, n = 1,000, is a rate rung that carries no claim
  at 800 replicates (``clause-binding-reasons.csv``); the harness gives it the ladder's
  smallest-rung role, and it cannot be red on a claim.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The ``iterated_mean_difference`` rows read the ``overlap`` law, state no verdict and cannot be
  red.  Their summary reports each coefficient's paired-difference ratio, each fit's coverage and
  SE ratio side by side, and the share of decreasing iterated-mean curves with its 99% interval.
  The pilot floor of 0.05 on the largest ratio (plan X70 rev 2.4) is checked by the design test
  before the run, together with the same floor on ``a_W1``; the share carries no floor.  On that
  law the two efficient fits differ by about 3 to 8% of an efficiency SE: both are efficient, so
  where both nuisance fits are correct they cannot differ by more than o_P(n^-1/2).  The
  ``mechanism_correct_difference`` rows read the ``survival`` law's ``mechanism_correct`` draws,
  where both regressions are misspecified the same way and both fits stay consistent through the
  correct mechanism; they report each coefficient's paired ratio and carry no verdict (the
  200-draw pilot reads 0.04 to 0.07, so they do not separate the fits at first order either).  A Wald SE shortfall on the
  ``overlap`` law is shared by the two fits and small (the 200-draw pilot reads 0.92 for the
  intercept in both fits, a planning run of 800 draws read 0.94 and 0.95 for the arm
  coefficient); it is reported, not read as a defect.  A run whose ratios fall below 0.05 is
  reported with the ratios stated, not re-sized.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_hazard_msm, hazard_msm_properties

if __name__ == "__main__":
    run(
        canonical_hazard_msm,
        hazard_msm_properties,
        here=Path(__file__).resolve().parent,
    )
