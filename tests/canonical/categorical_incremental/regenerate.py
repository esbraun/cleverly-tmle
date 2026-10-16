"""Regenerate the categorical incremental policy evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory.  The study has no
comparator, so the run starts no container.  A killed run resumes when the same command runs
again at the same commit with a new empty ``--output`` and the same ``--cache`` scratch.

Red-cell reading (declared before any run, ``X43-plan.md`` 5.3 and ``AGENT-BRIEF.md``,
"Reading a red calibration row").

* A red primary truth row blocks the merge and gets a diagnosis.
* A red calibration row is a finite-sample red only when three things hold: the 99% interval of
  its SE ratio contains 1, its coverage is consistent with 0.95 and the diagnosis finds no
  defect.  It then moves to ``reporting`` with an owner row.  Any other red row is a defect
  suspect: it is diagnosed, a defect is fixed and the study re-runs under a fresh declaration.
* A red ``mechanism_targeting`` positive cell is a defect suspect in the softmax update.  A
  control that does not fail is a defect suspect in the instrument.
* No margin, sample size, replicate count, law, learner, fold count or seed moves after a
  verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_categorical_incremental, categorical_incremental_properties

if __name__ == "__main__":
    run(
        canonical_categorical_incremental,
        categorical_incremental_properties,
        here=Path(__file__).resolve().parent,
    )
