"""Regenerate the conditional-projection evidence (P1 part (a)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container. The groups, sizes, learners, controls and design
numbers are those of :mod:`tests.studies.conditional_projection`, fixed before the run.

Fresh declaration for X97 (plan section 5.2, rev 2.2), before any run. ``assess_score_2k``
now measures the TMLE score assessment of X97, which replaces the one-step that its three cells
measured before (plan section 2.4: one pooled fluctuation along the span of the score's
features, whose substitution estimates respect the parameter space and share one targeted fit
with the paired comparison). That is a result-determining change, so the whole study is rerun
under this declaration. The three ``assess_score_2k`` pilot rows of
``tests/studies/conditional_projection_pilot.csv`` were re-read on the TMLE (2,400 draws on
``se_ratio_pilot``: SE ratios 1.0085, 1.0061 and 1.0049, against the one-step's 1.0083, 1.0059
and 1.0046); every other pilot row is unchanged. At ``R_GLM = 4,800`` the pooled family-wise
pass probability is 0.9617 and the row-by-row one 0.7792, so ``R_GLM`` stays at 4,800, and the
study's budget is about 51 CPU-hours.

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
