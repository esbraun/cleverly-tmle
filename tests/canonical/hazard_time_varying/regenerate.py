"""Regenerate the ``hazard-time-varying`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_time_varying` and its
properties module, fixed before any run and checked by
``tests/unit/test_hazard_time_varying_study_design.py``.

Order of the work (plan X70 rev 2.4 and rev 2.7): the calibration pilots
(``calibration_<fit>`` of :mod:`tests.studies.hazard_time_varying_pilot`, one label per run,
committed after each; ``one_cause`` and ``competing`` banded) and the four 800-draw
wrong-learner pilots; then the replicate counts set from them by the pre-committed rules of the
design test; then a declaration commit in which the design test skips nothing; then the declared
run.

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
  fix a defect, and re-run under a fresh declaration.  A node-1 risk reads few events (in the
  calibration pilot a median smallest Kish count of 19 and a median Edgeworth loss of 0.011,
  recorded per row in ``largest_edgeworth_loss``); the
  primary coverage floor of 0.90 absorbs that loss, so a red node-1 coverage row is read as any
  other.
* A red calibration row, in the words of the plans guide (``guide/studies.md``, Red cells): "A
  red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. It then
  goes to `reporting` with an owner in the roadmap's Red-cell owners table. Any other red row
  is a defect suspect. An unfixed defect never ends as an owner row."  A defect suspect blocks
  the pull request.  The ``crossfit`` rows read the same reading.
* A red ``simultaneous_coverage`` band is read as a calibration row, at its pilot: the band
  passed the pre-committed rule of the design test (probability at least 0.90 at the pilot's
  joint coverage plus and minus one standard error).  A red band whose pointwise control also
  under-covers, and whose miss the node-1 risks' Edgeworth losses explain, is a finite-sample
  red under the calibration route; any other red band is a defect suspect.  The band is never
  widened after a reading.
* A red positive ``double_robustness`` cell is a defect suspect: each positive limit is the
  truth on the exact sample (``test_every_positive_limit_is_the_truth``).  The arms read
  different code.  ``mechanism_correct`` (hazard and continuation without ``L``) witnesses the
  hazard fluctuation, whose weights read ``L_k``: its covariate term is zero before targeting.
  ``continuation_wrong`` (hazard correct, continuation without ``L``; plug-in +1.37 efficiency
  SEs) witnesses the covariate term of the walk alone, so a red ``continuation_wrong`` arm is a
  defect suspect of the continuation fluctuation.  ``competing_contrast__mechanism_correct``
  (+3.20 SEs) witnesses the two-cause soft-label panel and
  ``three_arm_contrast__mechanism_correct`` (+3.63 SEs) the policy ratio times the retention
  that reads ``L_k``.  A ``both_wrong`` control that does not clear the margin is a failed
  control: its exact-law gap is +4.22 efficiency SEs
  (``test_the_both_wrong_control_is_discriminated``), so the family's verdict is void and the
  study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the exact limit of the untargeted backward
  plug-in whose hazard and continuation omit ``L``, +4.22 efficiency SEs at n = 2,000
  (``test_the_wrong_learner_rule``).  That is the same bias the ``both_wrong`` control reads
  (+4.22): with every factor a function of ``(W, A)`` the walk barely moves, so the two controls
  are not independent witnesses.  An ``untargeted`` control that does not clear the margin is a
  failed control and blocks the pull request.  A red ``targeted`` arm is a defect suspect of the
  hazard fluctuation with weights that read ``L_k``; the covariate term is witnessed by the
  ``continuation_wrong`` arm, not here.
* A red claimed ``root_n_and_efficiency`` rung (n = 2,000 or 8,000) is read as a calibration
  row at that rung: a finite-sample red only under the calibration bullet's conditions, any
  other red a defect suspect.  A red ``root_n_rate`` slope outside the root-n band is a defect
  suspect of the influence curve or the walk.  The smallest rung, n = 1,000, is a rate rung that
  carries no claim at 800 replicates (``clause-binding-reasons.csv``); the harness gives it the
  ladder's smallest-rung role, and it cannot be red on a claim.
* A red ``power`` cell is a defect suspect: its exact-law power at n = 2,000 on the calibration
  fits' count is at least 0.80 with a cell pass probability of 0.99
  (``test_the_power_cell_has_power``), so a miss means the estimator or its interval is off.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The ``iterated_mean_difference`` rows state no verdict and cannot be red.  Their summary
  reports each estimate's paired-difference ratio (against the iterated-mean fit without and
  with ``isotonic=True``), each fit's coverage and SE ratio side by side, the share of
  decreasing iterated-mean curves with its 99% interval
  (:func:`tests.studies.hazard_time_varying_properties.fits_differ_reading`), and, on the
  two-cause law, each fit's largest sum of cumulative incidences minus one.  No reading carries
  a floor (plan X70 rev 2.6, open question 11): both fits are efficient, so where both nuisance
  fits are correct they cannot differ by more than o_P(n^-1/2).  The decrease and coherence rows
  record a reference of zero, not a truth (their true values are negative), so only their share
  above zero and their largest value are read.
* Every replicate row records ``converged``, the fit's final relative score (``final_score``,
  the largest node walk's own relative score) and ``unexpected_warnings`` for the fits of its
  key.  A fit that does not converge and whose final relative score is above 1e-6
  (``hazard_time_varying_properties.STALLED_SCORE``) is a recordable failure: the harness
  records the key as a lost replicate (``stalled_fit``), charged to the cell's failure bound of
  0.01.  A non-converged fit at or below 1e-6 keeps its rows.  A cell whose non-converged fits,
  whatever their scores, exceed 1% of its requested replicates is a defect suspect
  (``convergence_defect_suspect``), does not pass, and blocks the pull request.
* ``SparseInfluenceWarning`` is expected on the node-1 risks, which are near 0.02.  Those
  estimates do carry verdicts: every primary row and the two gated bands read them.  The warning
  text names only five estimates, so it is not parsed; each row records its fit's smallest Kish
  count, largest relative SD and largest Edgeworth loss, which the readings above use.
* A draw whose fit holds a node-hazard cell (no unit of one treatment level at a node has an
  event of a cause) reports estimates without an interval; the fit raises ``HeldCellError``, a
  recordable ``DataError``, and the draw is a lost replicate on every path, charged to the cell's
  failure bound of 0.01.  The design test puts the expected count of such draws over the study
  and its pilot at 0.05.  A cell past its bound on held draws takes the route above with the
  count stated.
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
