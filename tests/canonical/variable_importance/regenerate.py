"""Regenerate the step-down max-T evidence (``variable-importance``) under the declared run form.

Declared, not run. The study is not in the register yet, and
:func:`tests.studies.variable_importance_properties.requested` refuses the property phase
until the declaration commit sets ``DECLARED_REPLICATES`` from the committed pilot
(``tests/studies/variable_importance_pilot.csv``). The order is in
:mod:`tests.studies.variable_importance`.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard (a clean tree, ``HEAD`` equal to its upstream, every thread variable 1), the
run log, the checkpoints under ``~/.cache/cleverly/runs/variable-importance/<key>`` that let a
stopped run resume, and the copy into this directory. No reference implementation runs:
``STUDY.reference`` is ``None`` (the design's comparator survey rejected every candidate).

**Red-row reading**, declared before any run (design rev 2, sections 7 and 10). A red
``fwer_calibration`` or ``family_wise_error`` positive row, or a control that does not fail,
is a defect suspect and gets a diagnosis before any merge. A red ``power`` or
``interval_calibration`` row is a finite-sample red only when the diagnosis finds no defect and
its interval contains the planning value. No margin, sample size, replicate count, law,
learner, fold count or seed moves after a verdict is seen, and the Benjamini-Hochberg and
Benjamini-Yekutieli cells stay diagnostic whatever they read.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import variable_importance, variable_importance_properties

if __name__ == "__main__":
    run(
        variable_importance,
        variable_importance_properties,
        here=Path(__file__).resolve().parent,
    )
