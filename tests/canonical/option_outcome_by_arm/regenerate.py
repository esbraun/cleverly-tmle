"""Regenerate the ``option-outcome-by-arm`` evidence (X76 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_outcome_by_arm`,
:mod:`tests.studies.option_outcome_by_arm_properties` and :mod:`tests.studies._option_by_arm_law`,
fixed before the run and recomputed by ``tests/unit/test_option_outcome_by_arm_design.py``
from the committed pilot (``tests/studies/option_outcome_by_arm_pilot.csv``).

Red-cell reading (declared before any run; Fix-8 plan section 5.0).  The design test applies it
mechanically to the committed tables (``gating_findings`` over the 14 gated cells).

* A red ``interval_calibration`` positive is finite-sample only when the 99% interval of its SE
  ratio contains 1, its coverage interval contains 0.95 and the diagnosis finds no defect.  It
  then gets an owner row in the roadmap's Red-cell owners table, and its ``T53`` row reads
  experimental with the measured numbers.  Every other red calibration positive is a defect
  suspect: diagnose it, fix the defect, and re-run under a fresh declaration.
* A derived control (``__shrunken_se_control``, ``__noise_control``) that does not leave the
  band is a red of the study, not of the estimator.  The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* A red ``double_robustness/treatment_correct`` (bias outside the margin, or an SE ratio
  outside the union-model band) is a defect suspect in the per-arm path.
* A ``double_robustness/both_wrong`` control that fails to fail is a design bug or a defect
  suspect: its exact-law limit is 3.43 SDs, so a control inside the margin means the fit did
  not use the declared learners.  It is diagnosed with the exact-law limit and fixed under a
  fresh declaration.
* Every ``pooled_outcome`` row is published beside its exact limit and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Law.  Plan 5.3 wrote the mechanism as ``expit(0.4 W1 - 0.4 W2)``.  Under that sign the
confounding of the two arm means cancels, and the ``both_wrong`` control's exact limit is 0.16
SD, below the 0.45 SD floor: the control could not fail, and the positive could not tell an
estimator that ignores the confounding from one that removes it.  The declared mechanism is
``expit(0.4 W1 + 0.4 W2)``, with the same overlap.

Sizes.  The four calibration cells run 6,400 replications: plan 5.0's product floor is 2,800,
and the arm-pooled predictive rule at the 8,000-draw pilot (draws 0 to 7,999 of each
calibration block's stream) raises it, pricing the spread between the binary SE ratios (1.006,
1.009) and the continuous and bounded ones (0.990, 0.991).  Each ``pooled_outcome`` cell runs
on its calibration cell's draws.  The double-robustness pair runs 1,600 and the primary phase
2,800.

Budget.  About 0.5 CPU-hours of fits on the 4-vCPU cloud machine (the pilot fitted 8,000 draws
of the four binary fits in 7 minutes on one core), about 1 CPU-hour with the harness; plan 5.9
allows 2.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_option_outcome_by_arm, option_outcome_by_arm_properties

if __name__ == "__main__":
    run(
        canonical_option_outcome_by_arm,
        option_outcome_by_arm_properties,
        here=Path(__file__).resolve().parent,
    )
