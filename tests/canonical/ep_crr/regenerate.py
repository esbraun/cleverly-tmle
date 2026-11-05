"""Regenerate the ep-crr evidence (P1 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The hte3 comparator is declared pending (:data:`tests.studies.ep_crr.COMPARATOR`), so the run
starts no container.  Laws, learners, schedule, margins, controls and replicate counts are
those of :mod:`tests.studies.ep_crr`, fixed before the run from the committed pilot.

Red-cell reading (declared before any run; P1 plan sections 5.0 and 5.3).

* A red Thm 5 or growth row on an ensemble law is a defect suspect: diagnose it against the
  run's own nuisance-error rows, fix a defect if one is found, and re-run under a fresh
  declaration.  When no defect is found and the product of the nuisance L2 errors does not
  fall faster than ``n^{-1/2}`` on that law, C2 is not met at these ``n``: the schedule row
  stays experimental with that reason, and no margin moves.
* A red rescue row (intercept learner, ``d3_complex_limited_4k``) with its T-learner control
  green is a defect suspect of the log-CRR debiasing; with the control red as well, the cells
  cannot see the debiasing, and the rescue claim is withheld until the control is repaired
  under a fresh declaration.
* A gated row red only because its group excluded more than ``floor(0.01 R)`` replicates for a
  non-positive-definite oracle Hessian is a finding about the oracle, not about EP: the
  ``oracle_hessian`` rows say where, and the comparison is re-declared with a convex
  comparator before any claim is made.
* The CV-size, T-learner, Hessian and nuisance-error rows are diagnostic and state no verdict.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ep_crr, ep_crr_properties

if __name__ == "__main__":
    run(ep_crr, ep_crr_properties, here=Path(__file__).resolve().parent)
