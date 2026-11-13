"""Regenerate the ``hazard-mtp`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_mtp` and its properties
module, fixed before any run and checked by ``tests/unit/test_hazard_mtp_study_design.py``.

Order of the work (plan X70 rev 2.4, design review d): the calibration pilot, one
``calibration_<fit>`` label per run of :mod:`tests.studies.hazard_mtp_pilot`, committed after
each; then the replicate counts set from it; then a declaration commit in which the design test
skips nothing; then the declared run.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``, except as the calibration bullet below narrows it
and except for the readings below that name a red as a defect suspect or a failed control: the
record moves to ``reporting`` with an owner row and the reading below, the run is repeated under
the new commit, which refits the same deterministic seeds and so reproduces the same rows, and
the claim the cell carries stays experimental.  A defect suspect never takes that route.  No cell
is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the weighted walk, the shifted-dose plug-in or its
  influence curve: the truths are exact sums or quadratures to 1e-10 and every learner is
  correctly specified.  Diagnose it against ``tests/unit/test_hazard_mtp.py``, fix a defect,
  and re-run under a fresh declaration.
* The density route is gated on the ``truncated`` law only (the dose truncated to [0, 5.5], the
  shift capped at 5.5), where every gated route's influence-curve gap meets
  ``hazard_mtp_properties.IC_GAP_BOUND`` and falls with n (``TestTheInfluenceCurveGap``).  The
  ``untruncated_density`` rows are reported with no verdict and cannot be red: on the
  untruncated law the shipped route's quantile tail bins are wider than the shift, and its gap
  does not fall (0.26 SE at n = 20,000 in the committed ``ic_gap`` pilot).  That is the shipped
  route's limitation, disclosed on the evidence page and owned by its own row, not this study's.
* A red calibration row, in the words of the plans guide (``guide/studies.md``, Red cells): "A
  red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. It then
  goes to `reporting` with an owner in the roadmap's Red-cell owners table. Any other red row
  is a defect suspect. An unfixed defect never ends as an owner row."  A defect suspect blocks
  the pull request.
* A red positive ``double_robustness`` cell is a defect suspect: the ``levels`` limits are the
  truth on the exact sample (``test_every_positive_limit_is_the_truth``), and the four
  continuous ``mechanism_correct`` cells (``gaussian_density``, ``gaussian_classifier``,
  ``gaussian_crossfit``, ``vector``) are consistent through the correct mechanism, whose
  influence curves meet the declared gap bound.  A ``both_wrong`` control
  that does not clear the margin is a failed control: its exact-law gap is +3.44 efficiency SEs
  (``test_the_both_wrong_control_is_discriminated``), so the family's verdict is void and the
  study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the wrong learner's exact plug-in limit,
  +3.47 efficiency SEs (``test_the_wrong_hazard_learner_rule``).  An ``untargeted`` control
  that does not clear the margin is a failed control and blocks the pull request.
* The smallest ``root_n_and_efficiency`` rung, n = 1,000, is a rate rung that carries no claim
  at 800 replicates (``clause-binding-reasons.csv``); the harness gives it the ladder's
  smallest-rung role, and it cannot be red on a claim.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The ``iterated_mean_difference`` rows state no verdict and cannot be red.  Their summary
  reports each estimate's paired-difference ratio (against the iterated-mean fit without and
  with ``isotonic=True``), each fit's coverage and SE ratio side by side, and the share of
  decreasing iterated-mean curves with its 99% interval.  The floors of plan X70 section 5 are
  quoted beside the ratios by
  :func:`tests.studies.hazard_mtp_properties.fits_differ_reading` and read by no test or
  verdict; a decision on the requirement is pending with the project owner, and if the owner
  keeps it the family is redesigned.  A run whose ratios fall below the floor is reported with
  the ratios stated, not re-sized.
* Every replicate row records ``converged`` and ``unexpected_warnings`` for the fits of its key,
  and the summary counts ``nonconverged_fits`` and ``unexpected_warning_fits`` per cell.  No
  verdict reads them; a cell with a nonconverged fit is reported with the count stated.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_hazard_mtp, hazard_mtp_properties

if __name__ == "__main__":
    run(
        canonical_hazard_mtp,
        hazard_mtp_properties,
        here=Path(__file__).resolve().parent,
    )
