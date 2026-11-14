"""Regenerate the ``known-score-att`` evidence (X35) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.known_score_att`,
:mod:`tests.studies.known_score_att_properties` and :mod:`tests.studies.known_score_att_law`,
fixed before the run and recomputed by ``tests/unit/test_known_score_att_design.py`` from the
committed pilot (``tests/studies/known_score_att_pilot.csv``).

Red-cell reading (declared before any run; X35 plan section 5, rev 2.1).  The design test
applies it mechanically to the committed tables (``gating_findings`` over the 36 gated cells
and the primary truth rows).  The designed chance that at least one gated cell reads red with
the estimator correct is 0.025 (the design test's ``designed_pass``), within the study-wide
0.05 of :mod:`tests.studies.repeated_sizing`.

* A red ``interval_calibration`` positive is a finite-sample red only when the 99% interval of
  its SE ratio contains 1, its coverage interval contains 0.95 and the diagnosis finds no
  defect.  It then goes to ``reporting`` with an owner row in the roadmap's Red-cell owners
  table, and its support row reads the measured numbers.  Every other red calibration
  positive (an SE ratio or efficiency interval clear of 1, coverage clear of 0.95) reads as a
  variance or plug-in defect of the known-score curve and blocks the row: diagnose it, fix the
  defect, and re-run under a fresh declaration.  The arm-mean labels read the shipped form; a
  red there is a defect suspect of the shipped ATT and ATC, routed the same way.
* A derived control (``__shrunken_se_control``, ``__noise_control``) that does not leave the
  band is a red of the study, not of the estimator.  The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* A red ``efficiency_gain`` positive (the 99% upper bound of SD(known score) / SD(arm mean)
  at or above 0.98) reads as a theory or implementation defect: the exact ratios are 0.9014
  (ATT) and 0.9081 (ATC), and the gate passes at the declared size with probability above
  0.9999.  A ``zero_gain`` control that reads red (upper bound below 0.98 on Law S0, where the
  exact ratio is 1) is a dead control or a defect of the plug-in: the study is re-designed,
  not re-run.
* A red ``known_score_accuracy`` positive (bias outside the margin, coverage below the floor,
  or an SE ratio outside the sanity band) reads as an implementation defect of the known-score
  path with a wrong outcome regression, where the remainder is zero for any regression limit.
  The ``s_att__declared_constant`` control must place its bias interval beyond the margin (its
  exact bias is -0.663, about 10.8 SDs); a control that fails to fail is a dead control or a
  fit that did not use the declared constant, and the study is re-designed, not re-run.
* Scope of the claim.  Efficiency and interval calibration are measured on the two-arm
  continuous Law S only.  On Law B the variance gain is about 1% of the curve's variance, so
  the Law B cells license bias and coverage of the known-score plug-in and nothing about its
  curve; the support rows' limit sentence says so.
* A red ``root_n_and_efficiency`` rung or ``root_n_rate`` slope is a defect suspect.
* No replicate count, law, learner, margin, boundary or seed moves after a verdict is seen.
  No re-run buys a pass.  Before any re-run the designed chance of a red row is stated, a
  cheap targeted diagnosis (paired arms on fresh seeds) runs, and at most the affected cell
  re-runs, only when the diagnosis points at a code defect.

Sizes.  Each sized instrument passes with at least its share of the study-wide 0.95
(``repeated_sizing.SIZED_SHARE`` = 0.9837 for calibration; ``family_share(2)`` = 0.9918 for each
of the two property families).  The six calibration labels and the zero-gain pairs run 4,800
replications: 3,200 at target (six cells, a product), raised to 4,800 by the arm-pooled
predictive rule at the 8,000-draw pilot (SE ratios 0.985 to 0.997).  The accuracy family runs
1,400, the root-n rungs 1,600, the primary phase 2,400.

Budget.  About 0.2 CPU-hours of fits (about 20 ms per fit, measured on the 4-vCPU cloud
machine with one thread) and under 0.1 CPU-hours of summaries and failure accounting: about
10 minutes wall with ``--jobs 4``.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import known_score_att, known_score_att_properties

if __name__ == "__main__":
    run(
        known_score_att,
        known_score_att_properties,
        here=Path(__file__).resolve().parent,
    )
