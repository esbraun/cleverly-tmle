"""Regenerate the ``hazard-time-varying-multinode`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_hazard_time_varying_multinode`
and its properties module, fixed before any run and checked by
``tests/unit/test_hazard_time_varying_multinode_study_design.py``.

Order of the work (plan X70 section 5, rev 3.6, and rev 3.8): the smoke gate
(``canonical_hazard_time_varying_multinode.SMOKE_GATE``), which writes ``smoke.json`` and whose
times re-declare the cost; the calibration pilots (``calibration_multinode``, banded, and
``calibration_crossfit`` of :mod:`tests.studies.hazard_time_varying_multinode_pilot`, one label
per run, committed after each), the four 800-draw wrong-learner pilots, and the 800-draw
``untargeted_outcome_correct_n8000`` pilot of the untargeted arms at n = 8,000; then the replicate
counts set from them by the pre-committed rules of the design test; then a declaration commit in
which the design test skips nothing; then the declared run.  The record is registered, with its
support rows ``G63.multinode.in_sample`` and ``G63.multinode.cross_fitted``, only after the run
passes.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``, except as the calibration bullet below narrows it
and except for the readings below that name a red as a defect suspect or a failed control.  A
defect suspect never takes that route.  No cell is re-sized, re-drawn or downgraded after a
verdict is seen.

* A red primary row is a defect suspect of the multi-node slot rule, the node panel, the backward
  pass or the influence curve: the truths are exact sums over the law's support and every learner
  is correctly specified.  Diagnose it against ``tests/unit/test_hazard_time_varying_multinode.py``,
  fix a defect, and re-run under a fresh declaration.
* A red calibration row follows the plans guide (``guide/studies.md``, Red cells) with one
  change declared here before any run (plan X70 rev 3.8 addendum): it is a finite-sample red
  only when the 99% interval of its SE ratio contains that arm's expected ratio, not 1
  (``hazard_time_varying_multinode_properties.finite_sample_red``: the arm's
  expected mean of the influence-curve SD over the bound at n = 2,000, from an exact simulation
  of the law: 0.978 on the policy arms and 0.987 to 0.994 on the rest), its coverage is
  consistent with 0.95, and the diagnosis finds no defect.  It then goes to `reporting` with an
  owner in the roadmap's Red-cell owners table.  Any other red row is a defect suspect, and an
  unfixed defect never ends as an owner row.  The heavy-tailed influence curves of law (M) put
  each arm's expected ratio below one, so at 15,600 replicates the policy arms' intervals
  exclude 1 by design.  The ``crossfit`` rows read the same reading.  A derived control
  (``shrunken_se_control``, ``noise_control``) that does not leave the band is an instrument
  defect, and the run is void.
* A red ``simultaneous_coverage`` band over the 21 estimates is read as a calibration row, at its
  pilot: the band passed the pre-committed rule of the design test.  Any red band whose pointwise
  control covers is a defect suspect.  The band is never widened after a reading.
* A red positive ``double_robustness`` cell is a defect suspect: each positive limit is the truth
  on the exact sample (``test_every_positive_limit_is_the_truth``).  A red policy arm is a defect
  suspect of the policy's slots (the policy read at the observed earlier treatment, the policy
  mixture at every policy node), which change the target under every configuration.  A red
  untargeted ``outcome_correct`` arm (``__outcome_correct_untargeted``) is a defect suspect of the
  slot hazards (the current-level prediction, the held slot), which the walk repairs at a static
  or rule node; its limit is the truth (``test_the_untargeted_positive_limits_are_the_truth``).
  A red ``continuation_wrong`` arm is a defect suspect of the continuation fluctuation.  A
  ``both_wrong`` control that does not clear the margin is a failed control: the family's verdict
  is void and the study is a defect suspect that blocks the pull request.
* A red ``targeting_necessity`` pair is read against the exact limit of the untargeted backward
  plug-in of the ``mechanism_correct`` configuration (``test_the_untargeted_gaps``).  An
  ``untargeted`` control that does not clear the margin is a failed control and blocks the pull
  request.  A red ``targeted`` arm is a defect suspect of the hazard fluctuation.
* A red claimed ``root_n_and_efficiency`` rung (n = 2,000 or 8,000) is read as a calibration row
  at that rung.  A red ``root_n_rate`` slope outside the root-n band is a defect suspect.  The
  n = 1,000 rung carries no claim.
* A red ``power`` cell is a defect suspect.
* The ``iterated_mean_difference`` rows state no verdict and cannot be red.  Their summary reports
  each estimate's paired-difference ratio against the iterated-mean fit without and with
  ``isotonic=True``, each fit's coverage and SE ratio side by side, and the share of decreasing
  iterated-mean curves with its 99% interval; no reading carries a floor (plan X70 rev 2.6).
* Every replicate row records ``converged``, ``final_score`` and ``unexpected_warnings``.  A fit
  that does not converge and whose final relative score is above 1e-6 is a recordable failure
  charged to the cell's failure bound of 0.01.  A cell whose non-converged fits exceed 1% of its
  requested replicates is a defect suspect and blocks the pull request.
* A draw whose fit holds a cell raises ``HeldCellError``, a recordable ``DataError``, and is a
  lost replicate on every path, charged to the cell's failure bound of 0.01.  A cell past its
  bound takes the route above with the count stated.
* A replication that raises is never redrawn.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    canonical_hazard_time_varying_multinode,
    hazard_time_varying_multinode_properties,
)

if __name__ == "__main__":
    run(
        canonical_hazard_time_varying_multinode,
        hazard_time_varying_multinode_properties,
        here=Path(__file__).resolve().parent,
    )
