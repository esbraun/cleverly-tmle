"""Regenerate the continuous-time survival refutation evidence (X101 PR 2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator, so the run starts no container. The design is that of
:mod:`tests.studies.refuter_calibration_continuous_survival` and its properties module, declared
from X101 plan rev 2, section 5, and checked by
``tests/unit/test_refuter_calibration_continuous_survival_design.py``.

Pending. While ``DECLARATION_PENDING`` is set the declared run is refused (a smoke run, which
passes ``--replicates``, is not): the 57-trial pilot
(``tests/studies/refuter_calibration_continuous_survival_pilot.csv.gz``) leaves the predictive
family pass of the size cells at 0.12, short of the 0.90 the plan requires before declaration,
and lost 3 of 60 size trials to the "initial survival reaches zero" DataError on a subset
refit, against the failure bound of 0.01. X6 PR A removes that DataError and makes the fit's
default SE the one the subset reading reads; the pilot reruns after it.

Scope. Eight size cells: the placebo ``t`` (permute, R = 20) and the subset ``rho`` (fraction
0.7, frozen outer folds, R = 20) on the risk difference of each of the two causes at the median
and upper quartile of the observed event times, 200 trials each on the base law of
``continuous-time-survival``, all eight readings of a trial from the same 41 fits; pass at no
more than 17 rejections. Twelve power cells: the stale-row placebo (share 0.35) on 50 trials of
the power law (beta 2.0, cause-2 arm coefficient -1.0, every difference at least 4.3 efficiency
SE), and the subset ``rho`` of size trials 0 to 49 at the reported SE halved and doubled; pass at
no fewer than 38 detections of 50. Two reporting cells, 100 trials each, placebo only, no
verdict: dependent censoring and 25% treated. The fit is the default ``ContinuousSurvivalTMLE``,
cross-fitted with 10 outer and 5 inner folds, at n = 1,000.

Budget. The pilot measured 146 s per size trial (41 fits, 3.6 s per fit, the full fit about
4.5 s and a 700-row subset refit less) and 89 s per power or reporting trial (21 fits) on one
core of four busy ones. Size 200 trials (8.1 CPU-h), power 50 (1.2), reporting 200 (4.9), the
primary phase 200 fits (0.3): 14.5 CPU-h, about 19 with a 1.3 margin, so about 5 h on 4 cores.
The plan's estimate was about 20 CPU-h. Cloud.

Red-cell readings (declared before any run, X101 plan rev 2 section 5, as the X82 studies).

* A red size cell is a defect suspect until diagnosed. A defect is fixed and the study re-runs
  under a fresh declaration. A red with no defect goes to ``reporting`` with an owner, and the
  rule's registry row stays descriptive for that estimand.
* A red subset cell at the median (1.12) has a pre-identified cause: the registered study's
  influence-curve SE runs below the sampling spread at early times (roadmap X6 (a)), and the
  pilot's mean ``rho`` there is 1.03 to 1.07. Diagnosis tests that cause first.
* A red power cell with no defect means the defect was sized too small for that estimand. The
  rule stays descriptive there and the cell goes to reporting.
* The reporting cells carry no verdict, whatever they read.
* No threshold, trial count, law, learner, fold count, defect size or seed moves after a
  verdict is seen. No rerun buys a pass. A trial that raises is never redrawn; a recordable
  fit failure counts against the failure bound of 0.01.
* The subset cells rerun if X6 PR A makes its standard error the fit's default after this run.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    refuter_calibration_continuous_survival,
    refuter_calibration_continuous_survival_properties,
)

if __name__ == "__main__":
    smoke = any(argument.startswith("--replicates") for argument in sys.argv[1:])
    if refuter_calibration_continuous_survival.DECLARATION_PENDING and not smoke:
        raise SystemExit(
            "refused: the declaration is pending: "
            f"{refuter_calibration_continuous_survival.DECLARATION_PENDING}"
        )
    run(
        refuter_calibration_continuous_survival,
        refuter_calibration_continuous_survival_properties,
        here=Path(__file__).resolve().parent,
    )
