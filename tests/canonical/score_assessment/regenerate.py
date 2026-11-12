"""Regenerate the score-assessment evidence (X97) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator (EconML's ``RScorer`` and ``DRTester`` score one model with no interval and
no paired risk), so the run starts no container. The groups, sizes, learners, controls and
design numbers are those of :mod:`tests.studies.score_assessment`, fixed before the run.

Red-cell reading (declared before any run; X97 plan section 5.1 and ``guide/studies.md``).

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with an owner. Every other red calibration row is a defect suspect: diagnose
  it, fix the defect, and re-run under a fresh declaration.
* A red ``type_i_error`` or ``power`` cell, or a red positive ``bias`` row, is a defect
  suspect: the power and every truth are exact-law numbers.
* A control that does not fail (an ``__inflated_se_control`` or ``__term_dropped_control``
  row inside or across the band, or an ``untargeted`` row not wholly outside +-0.25 of its own
  SD) is a red of the study, not of the estimator. The claim of the cell it guards is withheld
  until the control is repaired under a fresh declaration.
* ``comparison.csv`` (the one-step at the initial fit, and the TMLE coverage of every row,
  the bias groups' included) is published ungated and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import score_assessment, score_assessment_properties

if __name__ == "__main__":
    run(score_assessment, score_assessment_properties, here=Path(__file__).resolve().parent)
