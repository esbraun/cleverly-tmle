"""Regenerate the ``hazard-multinode`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_multinode` and its
properties module, fixed before any run and checked by
``tests/unit/test_hazard_multinode_study_design.py``.

Order of the work (plan X70 rev 2.4, design review d): the calibration pilot, one
``calibration_<fit>`` label per run of :mod:`tests.studies.hazard_multinode_pilot`, and the
``wrong_learner_*``, ``both_wrong_base`` and ``fits_differ_*`` labels, each committed after it
runs; then the
replicate counts set from them; then a declaration commit in which the design test skips
nothing; then the declared run.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``, except as the calibration bullet below narrows it
and except for the readings below that name a red as a defect suspect or a failed control: the
record moves to ``reporting`` with an owner row and the reading below, the run is repeated under
the new commit, which refits the same deterministic seeds and so reproduces the same rows, and
the claim the cell carries stays experimental.  A defect suspect never takes that route.  No cell
is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the path slots, the shared prefix update, the path
  plug-in or its influence curve: the truths are exact sums over the paths and every learner
  is correctly specified.  Diagnose it against ``tests/unit/test_hazard_multinode.py``, fix a
  defect, and re-run under a fresh declaration.
* A red calibration row, in the words of the plans guide (``guide/studies.md``, Red cells): "A
  red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect. It then
  goes to `reporting` with an owner in the roadmap's Red-cell owners table. Any other red row
  is a defect suspect. An unfixed defect never ends as an owner row."  A defect suspect blocks
  the pull request.  This holds for the ``stay`` arm as for every other: a red ``stay`` row is
  not read as the collinearity of its targets with ``always`` and ``never``.
* A red positive ``double_robustness`` cell is a defect suspect: every positive limit is the
  truth on the exact sample (``test_every_positive_limit_is_the_truth``), the node-only
  ``four_contrast`` cell among them.  A green ``mechanism_correct`` cell whose wrong learner's
  plug-in lies within about one SE of the truth (``base_rule``, ``base_policy``,
  ``base_stay``, ``four_contrast``, ``four_policy``) is a consistency check and is not read as
  evidence that the targeting moved the estimate; the node-only cell (+2.20 SEs) is the
  witness on ``four``.  A ``both_wrong`` control that does not clear the margin is a failed control: its
  exact-law gap is +3.07 efficiency SEs (``test_the_both_wrong_control_is_discriminated``), so
  the family's verdict is void and the study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the wrong learner's exact plug-in limit,
  +3.54 efficiency SEs (``test_the_wrong_hazard_learner_rule``).  An ``untargeted`` control
  that does not clear the margin is a failed control and blocks the pull request.
* The smallest ``root_n_and_efficiency`` rung, n = 1,600, is a rate rung that carries no claim
  at 800 replicates (``clause-binding-reasons.csv``); the harness gives it the ladder's
  smallest-rung role, and it cannot be red on a claim.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* A red ``simultaneous_coverage`` band is read as a calibration row is: a finite-sample red
  only when its joint coverage interval is consistent with 0.95, the pointwise cells of the
  same fits are green, and the diagnosis finds no defect; any other red band is a defect
  suspect of the shared walk's joint influence curves.  The calibration pilot reads the band
  before the run (``test_the_joint_band_pilot_meets_the_band``).  A pointwise control that
  does not fall short of the band is a failed control and voids the family's verdict.
* A red ``power`` cell is a defect suspect: the design power at the exact contrast is above
  0.9999 (``test_the_power_cell_has_power``), so a miss means the contrast's estimate or its
  interval is wrong, and it is diagnosed with the calibration rows of the same fits.
* A fit with a held prefix cell (no event, or only events, among the units at risk with one
  treatment prefix) reports no interval for the regimens that read it.  The fit raises
  ``HeldCellError``, a ``DataError``, before any row reads it, on every path, so the draw is a
  recorded lost replicate under the bound of 0.01
  (``canonical_hazard_multinode.refuse_held_cells``).  The design test expects about 6.6 such
  draws over the study and its pilot, all on ``base``
  (``test_the_expected_held_fits_stay_within_each_allowance``); a cell past its bound from held
  draws takes the bound route below, and the held count is stated beside it.
* The ``one_step_walk`` rows state no verdict and cannot be red.  Their summary reports, for
  the contrast and the policy's risk on ``base``, the one-step fit's coverage and SE ratio, its
  convergence, and its paired difference from the per-horizon fit of the same draw.
* The ``iterated_mean_difference`` rows state no verdict and cannot be red.  Their summary
  reports each estimate's paired-difference ratio (against the iterated-mean fit without and
  with ``isotonic=True``), each fit's coverage and SE ratio side by side, and the share of
  decreasing iterated-mean curves with its 99% interval
  (:func:`tests.studies.hazard_multinode_properties.fits_differ_reading`).  No reading carries a
  floor: X70 reports the agreement of the two fits instead of requiring a gap (the owner's
  decision of 2026-11-13).  The iterated-mean regressions are saturated cell means, so with the
  correct mechanism both fits are asymptotically efficient and differ by o_P(n^-1/2); at
  n = 2,000 the sparse cells of the ``four`` policy add finite-sample noise to the reading.
* Every replicate row records ``converged``, the fit's final relative score (``final_score``)
  and ``unexpected_warnings`` for the fits of its key.  A fit that does not converge and whose
  final relative score is above 1e-6 (``hazard_multinode_properties.STALLED_SCORE``) is a
  recordable failure: the harness records the key as a lost replicate (``stalled_fit``),
  charged to the cell's failure bound of 0.01, and the summary refuses a kept row that breaks
  the rule.  A non-converged fit at or below 1e-6 is the walk's reporting floor and keeps its
  rows.  A cell whose non-converged fits, whatever their scores, exceed 1% of its requested
  replicates is a defect suspect (``convergence_defect_suspect``; the count adds the cell's
  lost replicates, so it can only over-state), does not pass, and blocks the pull request.
  The summary also counts ``nonconverged_fits`` and ``unexpected_warning_fits`` per cell.
* ``SparseInfluenceWarning`` is expected on the ``base``, ``four`` and ``onestep`` regimen fits, where it
  fires on every draw at n = 2,000: a static arm's risk by the last node reads its influence
  curve from few weighted units (``canonical_hazard_multinode.EXPECTED_WARNINGS``).  A red
  calibration row there is still read by the calibration bullet above, never excused by the
  warning.  On ``msm`` it is unexpected.  Every unexpected warning is counted in
  ``unexpected_warnings`` and reported; no verdict reads that count.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_hazard_multinode, hazard_multinode_properties

if __name__ == "__main__":
    run(
        canonical_hazard_multinode,
        hazard_multinode_properties,
        here=Path(__file__).resolve().parent,
    )
