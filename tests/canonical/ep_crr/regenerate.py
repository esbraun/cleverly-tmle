"""Regenerate the ep-crr evidence (P1 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The hte3 comparator is declared pending (:data:`tests.studies.ep_crr.COMPARATOR`), so the run
starts no container.  Laws, learners, schedule, margins, controls and replicate counts are
those of :mod:`tests.studies.ep_crr`, fixed before the run from the committed pilot.

Red-cell reading (declared before any run; P1 plan sections 5.0 and 5.3, and the PR-b design
review's B1, B2, S2 and S3).

* Reading (a). A growth row (ensemble or rescue) red while its 1k mean log ratio is below 0
  and the 4k row of the same estimator is green reads as EP approaching the oracle from
  below, as Theorem 5 predicts: it is reported, not a defect suspect, and the 4k row's verdict
  stands. :func:`tests.studies.ep_crr.design_numbers` computes its probability (``assurance``,
  ``*_reading_a``) before the run.
* Any other red Thm 5 or growth row on an ensemble law is a defect suspect: diagnose it against the
  run's own nuisance-error rows, fix a defect if one is found, and re-run under a fresh
  declaration.  When no defect is found and the product of the nuisance L2 errors does not
  fall faster than ``n^{-1/2}`` on that law, C2 is not met at these ``n``: the schedule row
  stays experimental with that reason, and no margin moves.
* A red rescue row (``ep_misspecified`` or ``ep_offset`` on ``d3_complex_limited_4k``) with its
  T-learner control green is a defect suspect of the log-CRR debiasing. With the control red,
  whether the rescue row is red or green, the misspecification is too weak for the cells to see
  the debiasing: that learner's rescue claim is withheld until the control is repaired under a
  fresh declaration. The intercept-learner rescue states that the debiasing corrects an
  intercept-only ``mu`` at this law; it does not witness the ``1 / pi`` weight, which the offset
  rescue and the unit test witness.
* A gated row red only because its group excluded more than ``floor(0.01 R)`` replicates for a
  non-positive-definite oracle Hessian is a finding about the oracle, not about EP: the
  ``oracle_hessian`` rows say where, and the comparison is re-declared with a convex
  comparator before any claim is made.
* A row red only because a replicate failed ("failures not accountable": every gated oracle
  row reads its whole group, so one lost fit turns it red at any bound) is read from the
  recorded exception. A learner or numerical refusal (a SuperLearner candidate, a Newton step,
  a singular design) is a finding about the learner or the law, reported with the exception,
  and the claim it blocks stays withheld; an exception in the EP code is a defect suspect. The
  pilots lost no fit, so a failure-only red is an accepted risk of the declaration.
* The CV-size, T-learner, Hessian and nuisance-error rows are diagnostic and state no verdict,
  and so is the offset learner's growth row (plan amendment rev 2.5): its pilot reads the EP
  ratio rising from 0.16 at 1k to 0.25 at 4k, and with ``mu`` wrong outside the sieve no
  theorem bounds that growth. A rising offset ratio with the 4k rescue row green is reported.
  The ``oracle_comparison`` rows publish their reading in the ``ratio_*`` columns; the shared
  bias, coverage and rejection columns are blank there, because each row is a log ratio
  carried with a unit standard error.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ep_crr, ep_crr_properties

if __name__ == "__main__":
    run(ep_crr, ep_crr_properties, here=Path(__file__).resolve().parent)
