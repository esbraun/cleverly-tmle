"""Regenerate the score-assessment evidence (X97) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator (EconML's ``RScorer`` and ``DRTester`` score one model with no interval and
no paired risk), so the run starts no container. The groups, sizes, learners, controls and
design numbers are those of :mod:`tests.studies.score_assessment`, fixed before the run.

Changes since the first declaration (``3c00484``), all before any run (plan rev 2.2, from the
design review ``rows/X97/design-review.md``):

* The log-ratio fluctuation's reduced-span retry (``8fb90bd``): a fit whose alternation cycles
  at a collinear calibration slope reruns without its thinnest span direction. Its acceptance
  now also requires every original numerator's score equation to meet the ``scaled_var``
  threshold (S2), the span record carries ``reduced``, and such a fit reads ``experimental``.
  No pilot fit changed.
* Every calibration group's pilot runs 2,400 draws on ``se_ratio_pilot`` (S3), not 800.
* The three single-score log-ratio risk-difference rows form their own sizing pool (B2), and
  the committed oracle reference (``tests/studies/score_assessment_oracle.py``, first-principles
  one-step at the law's nuisances, 40,000 draws) is their declared finite-sample prediction.
* The pre-run n rule (B2): at 2,000 rows the log-ratio pilot (pooled ``S`` ratio 1.047) agreed
  with the oracle (1.026) within the 99% interval, but no ``R`` up to 50,000 met the calibration
  rule, so ``single_crr_2k`` and ``paired_crr_2k`` became ``single_crr_4k`` and
  ``paired_crr_4k`` (``n = 4,000``, new seed streams by label), with a fresh pilot and oracle.
* ``R_CAL`` = 15,200 (family-wise pass 0.952; each log-ratio risk-difference row 0.978).
* Every positive bias row is priced at the larger of its second-order theory bias and its
  oracle bias (B1, and the re-review's nit R2: at the oracle's 0.128 the first redeclaration's
  1,600 read 0.9895); ``R_BIAS`` = 2,000, the smallest multiple of 400 at which each row passes
  with probability at least 0.99.
* The finite-sample red-cell route below (S4), and the reduced-span counts in
  ``comparison.csv`` (S2).

Red-cell reading (declared before any run; X97 plan section 5.1 and ``guide/studies.md``).

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with an owner. Every other red calibration row is a defect suspect, but for the
  exception below: diagnose it, fix the defect, and re-run under a fresh declaration.
* A red ``type_i_error`` or ``power`` cell, or a red positive ``bias`` row, is a defect
  suspect, but for the exception below: the power and every truth are exact-law numbers.
* One declared exception to "defect suspect" (plan rev 2.2, S4), for the single-score
  risk-difference rows only (``S_risk_difference`` and ``SB_risk_difference``, either scale),
  which carry the best-constant term of plan section 2.6. A red bias row, or a red calibration
  row whose SE-ratio interval excludes 1, is finite-sample when all three hold: (a) the excess
  has the predicted sign (bias above 0, SE ratio above 1); (b) its 99% interval contains the
  declared prediction, the analytic bias (``score_assessment.theory_bias``) or the oracle SE
  ratio of ``tests/studies/score_assessment_oracle.csv`` at the row's group; (c) the diagnosis
  finds no defect. It then goes to ``reporting`` with an owner in the Red-cell owners table. Any
  other red, the paired rows and the calibration coefficients included, is a defect suspect.
  On the log-ratio risk-difference rows the pilot's SE ratio exceeds the oracle's by about
  0.024 at 2,000 rows and 0.021 at 4,000; the oracle's own excess halves with n and this gap
  does not, so it is not the best-constant term. A calibration red there needs a point above
  about 1.055, whose interval excludes the oracle's 1.013: it reads as a defect suspect, not
  through this route.
* A control that does not fail (an ``__inflated_se_control`` or ``__term_dropped_control``
  row inside or across the band, or an ``untargeted`` row not wholly outside +-0.25 of its own
  SD) is a red of the study, not of the estimator. The claim of the cell it guards is withheld
  until the control is repaired under a fresh declaration.
* ``comparison.csv`` (the one-step at the initial fit, the TMLE coverage of every row, the
  bias groups' included, and each group's count of reduced-span fits with their coverage) is
  published ungated and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import score_assessment, score_assessment_properties

if __name__ == "__main__":
    run(score_assessment, score_assessment_properties, here=Path(__file__).resolve().parent)
