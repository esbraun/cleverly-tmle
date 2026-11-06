"""Regenerate the conditional-projection evidence (P1 part (a)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container. The groups, sizes, learners, controls and design
numbers are those of :mod:`tests.studies.conditional_projection`, fixed before the run.

Red-cell reading (declared before any run; P1 plan section 5.0, and the design review of the
declaration).

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with an owner. Every other red is a defect suspect: diagnose it, fix the
  defect, and re-run under a fresh declaration.
* A red ``bias`` row of ``slow_1k`` alone, with ``slow_4k`` and ``slow_16k`` green, is a
  finite-sample red when its 99% interval contains the pilot's standardized bias 0.161: the
  declared offset is of order ``n^{-1/3}`` and the ladder's claim is the trend.
* A control that does not detect is a red of the study, not of the estimator: the cell it
  guards cannot see the component it is named for, and the claim that cell supports is
  withheld until the control is repaired under a fresh declaration.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import conditional_projection, conditional_projection_properties

if __name__ == "__main__":
    run(
        conditional_projection,
        conditional_projection_properties,
        here=Path(__file__).resolve().parent,
    )
