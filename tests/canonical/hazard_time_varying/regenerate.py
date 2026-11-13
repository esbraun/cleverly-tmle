"""Regenerate the ``hazard-time-varying`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_time_varying` and its
properties module, fixed before any run and checked by
``tests/unit/test_hazard_time_varying_study_design.py``.

Order of the work (plan X70 rev 2.4 and rev 2.7): the calibration pilots
(``calibration_<fit>`` of :mod:`tests.studies.hazard_time_varying_pilot`, one label per run,
committed after each) and the 800-draw wrong-learner pilot; then the replicate counts set from
them by the pre-committed rules of the design test; then a declaration commit in which the
design test skips nothing; then the declared run.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``, except as the calibration bullet below narrows it
and except for the readings below that name a red as a defect suspect or a failed control: the
record moves to ``reporting`` with an owner row and the reading below, the run is repeated under
the new commit, which refits the same deterministic seeds and so reproduces the same rows, and
the claim the cell carries stays experimental.  A defect suspect never takes that route.  No cell
is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the node panel, the backward pass or the influence
  curve: the truths are exact sums over the law's support and every learner is correctly
  specified (the continuation is the cell mean over ``(W, A, L_k, s)``, the true regression of
  the future on these Markov laws).  Diagnose it against ``tests/unit/test_hazard_time_varying.py``,
  fix a defect, and re-run under a fresh declaration.
* A red calibration row, in the words of the plans guide (``guide/studies.md``, Red cells): "A
  red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. It then
  goes to `reporting` with an owner in the roadmap's Red-cell owners table. Any other red row
  is a defect suspect. An unfixed defect never ends as an owner row."  A defect suspect blocks
  the pull request.  The ``crossfit`` rows read the same reading.
* A red positive ``double_robustness`` cell is a defect suspect: each positive limit is the
  truth on the exact sample (``test_every_positive_limit_is_the_truth``).  A ``both_wrong``
  control that does not clear the margin is a failed control: its exact-law gap is +4.22
  efficiency SEs (``test_the_both_wrong_control_is_discriminated``), so the family's verdict is
  void and the study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the exact limit of the untargeted backward
  plug-in whose hazard and continuation omit ``L``, +4.22 efficiency SEs at n = 2,000
  (``test_the_wrong_learner_rule``).  An ``untargeted`` control that does not clear the margin is
  a failed control and blocks the pull request.  A red ``targeted`` arm is a defect suspect of
  the covariate term: the exact-law tests detect each mutation of it at the declared bar.
* The smallest ``root_n_and_efficiency`` rung, n = 1,000, is a rate rung that carries no claim
  at 800 replicates (``clause-binding-reasons.csv``); the harness gives it the ladder's
  smallest-rung role, and it cannot be red on a claim.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The ``iterated_mean_difference`` rows state no verdict and cannot be red.  Their summary
  reports each estimate's paired-difference ratio (against the iterated-mean fit without and
  with ``isotonic=True``), each fit's coverage and SE ratio side by side, the share of
  decreasing iterated-mean curves with its 99% interval
  (:func:`tests.studies.hazard_time_varying_properties.fits_differ_reading`), and, on the
  two-cause law, each fit's largest sum of cumulative incidences minus one.  No reading carries
  a floor (plan X70 rev 2.6, open question 11): both fits are efficient, so where both nuisance
  fits are correct they cannot differ by more than o_P(n^-1/2).
* Every replicate row records ``converged``, the fit's final relative score (``final_score``)
  and ``unexpected_warnings`` for the fits of its key.  A fit that does not converge and whose
  final relative score is above 1e-6 (``hazard_time_varying_properties.STALLED_SCORE``) is a
  recordable failure: the harness records the key as a lost replicate (``stalled_fit``), charged
  to the cell's failure bound of 0.01.  A non-converged fit at or below 1e-6 keeps its rows.  A
  cell whose non-converged fits, whatever their scores, exceed 1% of its requested replicates
  is a defect suspect (``convergence_defect_suspect``), does not pass, and blocks the pull
  request.
* ``SparseInfluenceWarning`` is expected on the first nodes' risks, which are near 0.02.  One
  that names a gated estimate (a contrast or a risk at the last node) is counted in
  ``unexpected_warnings`` and reported; no verdict reads that count.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_hazard_time_varying, hazard_time_varying_properties

if __name__ == "__main__":
    run(
        canonical_hazard_time_varying,
        hazard_time_varying_properties,
        here=Path(__file__).resolve().parent,
    )
