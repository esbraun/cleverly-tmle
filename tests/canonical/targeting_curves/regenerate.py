"""Regenerate the targeting-curves evidence (X92 PR-5) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator yet (grf is pending), so the run starts no container. The laws, groups, sizes,
learners, controls and design numbers are those of :mod:`tests.studies.targeting_curves`,
fixed before the run; the pilot (:mod:`tests.studies.targeting_curves_pilot`) runs first and
its CSV is committed before the declaration is final.

Red-cell reading (declared before any run; X92 PR-5 plan, section 10).

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval is consistent with 0.95 and the diagnosis finds no defect. It then goes
  to ``reporting`` with an owner. Every other red is a defect suspect, and the diagnosis
  takes D1 to D4 first: the fixed-q expansion and its quantile term (D1), the band's stack
  (D2), the tie rule (D3) and the targeted scores the draws read (D4).
* A red joint-coverage row of cell ``H`` with green pointwise rows is a defect of the band's
  critical value or of the draws' joint law, not finite-sample noise in one quantity.
* A red type-I row of ``U`` or ``N`` is a defect suspect whatever its pointwise rows read.
* A red row of cell ``S`` alone, with ``H`` green on the same law, points at the split: the
  refit, the evaluation rows or the per-replicate truth (whose Sobol error the design test
  bounds).
* A control that does not detect is a red of the study, not of the estimator: the cell it
  guards cannot see the component it is named for, and the claim that cell supports is
  withheld until the control is repaired under a fresh declaration.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import targeting_curves, targeting_curves_properties

if __name__ == "__main__":
    run(
        targeting_curves,
        targeting_curves_properties,
        here=Path(__file__).resolve().parent,
    )
