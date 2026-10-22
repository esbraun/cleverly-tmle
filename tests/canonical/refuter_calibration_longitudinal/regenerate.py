"""Regenerate the longitudinal refutation evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container.

Red-cell reading (declared before any run, X82 plan 8.2 and ``AGENT-BRIEF.md``).

* A red size cell is a defect suspect until diagnosed. A defect is fixed and the study re-runs
  under a fresh declaration. A red with no defect goes to ``reporting`` with an owner, and the
  rule's registry row stays descriptive.
* A red power cell with no defect means the defect was sized too small for that configuration.
  The rule stays descriptive for that configuration and the cell goes to reporting.
* A red I1 subset cell has a pre-identified cause: IPSI's iterated regression is
  misspecified on law E, and its influence-curve SE can be conservative. Diagnosis tests that
  cause first.
* No threshold, trial count, law, learner, fold count, defect size or seed moves after a
  verdict is seen. No rerun buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    refuter_calibration_longitudinal,
    refuter_calibration_longitudinal_properties,
)

if __name__ == "__main__":
    run(
        refuter_calibration_longitudinal,
        refuter_calibration_longitudinal_properties,
        here=Path(__file__).resolve().parent,
    )
