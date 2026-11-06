"""Regenerate the ep-cate evidence (P1 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The hte3 comparator is
declared pending (:data:`tests.studies.ep_cate.COMPARATOR`), so the run starts no container. The
laws, groups, learners, schedule, margins, controls and replicate counts are those of
:mod:`tests.studies.ep_cate`, fixed before the run from the committed pilot.

Red-cell reading (declared before any run; P1 plan sections 5.0 and 5.2, and the PR-b design
review's B2 and S2).

* Reading (a). A growth row (ensemble or rescue) red while its 1k mean log ratio is below 0
  and the 4k row of the same estimator is green reads as EP approaching the oracle from
  below, as Theorem 5 predicts: it is reported, not a defect suspect, and the 4k row's verdict
  stands. :func:`tests.studies.ep_cate.design_numbers` computes its probability (``assurance``,
  ``*_reading_a``) before the run.
* Any other red Thm 5 or growth row on an ensemble law is a defect suspect: diagnose it against the
  run's own nuisance-error rows, fix a defect if one is found, and re-run under a fresh
  declaration. When the diagnosis finds no defect and the product of the nuisance L2 errors does
  not fall faster than ``n^{-1/2}`` on that law, C2 is not met at these ``n`` and the row is a
  finding about the learner, not the estimator: the schedule row of the plan's 2.6 stays
  experimental with that reason, and no margin moves.
* A red rescue row (misspecified learner, the ``_noise2`` laws) with its T-learner control
  green is a defect suspect of the debiasing. With the control red, whether the rescue row is
  red or green, the misspecification is too weak for the cells to see the debiasing: the
  rescue claim is withheld until the control is repaired under a fresh declaration.  The code-law misspecified rows on ``d3_complex_*_4k`` are diagnostic:
  the pilot reads them near ``log 4.3``, and they state no verdict (plan amendment rev 2.4).
* A red ``ep_fixed_obs_4k`` bias row is a defect suspect of the fixed-size EP fit; its paired
  EP-minus-TMLE diagnostic rows say whether the projection TMLE on the same draws moved too.
* The CV-size, greedy final learner, T-learner, DR-learner and nuisance-error rows are
  diagnostic and state no verdict. The ``oracle_comparison`` rows publish their reading in the
  ``ratio_*`` columns; the shared bias, coverage and rejection columns are blank there, because
  each row is a log ratio carried with a unit standard error.
* A row red only because a replicate failed ("failures not accountable": every gated
  oracle row reads its whole group, so one lost fit turns it red at any bound) is read from the
  recorded exception. A learner or numerical refusal (a SuperLearner candidate, a Newton step,
  a singular design) is a finding about the learner or the law, reported with the exception,
  and the claim it blocks stays withheld; an exception in the EP code is a defect suspect. The
  pilots lost no fit in about 850 replicates, so a failure-only red is an accepted risk of
  the declaration, not a planned case.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ep_cate, ep_cate_properties

if __name__ == "__main__":
    run(ep_cate, ep_cate_properties, here=Path(__file__).resolve().parent)
