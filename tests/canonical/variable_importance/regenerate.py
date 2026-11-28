"""Regenerate the step-down max-T evidence (``variable-importance``) under the declared run form.

Declared at R = 12,000, not run. The study is not in the register yet. The declaration set
``DECLARED_REPLICATES`` from the committed pilot and its extension
(``tests/studies/variable_importance_pilot.csv``,
``tests/studies/variable_importance_pilot_extension.csv``). The order is in
:mod:`tests.studies.variable_importance`.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard (a clean tree, ``HEAD`` equal to its upstream, every thread variable 1), the
run log, the checkpoints under ``~/.cache/cleverly/runs/variable-importance/<key>`` that let a
stopped run resume, and the copy into this directory. No reference implementation runs:
``STUDY.reference`` is ``None`` (the design's comparator survey rejected every candidate).

**Red-row reading**, declared before any run (design rev 3). A red ``fwer_calibration`` or
``family_wise_error`` positive row, a red ``power`` row, a red primary row, or a control that
does not fail, is a defect suspect and gets a diagnosis before any merge. A red
``interval_calibration`` row is a finite-sample red only when the 99% interval of its SE ratio
contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect; it then
goes to ``reporting`` with an owner. Any other red calibration row is a defect suspect. The
pilot extension measured the cross-fitted X5 SE ratio at 1.034 and the in-sample X1 ratio at
0.984 on 8,000 draws; a red on either starts its diagnosis from those readings and asks
whether the variance estimate of the per-candidate TMLE is at fault. No margin, sample size,
replicate count, law, learner, fold count or seed moves after a verdict is seen, and the
Benjamini-Hochberg and Benjamini-Yekutieli cells stay diagnostic whatever they read.
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
