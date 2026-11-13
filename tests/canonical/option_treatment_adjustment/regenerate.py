"""Regenerate the ``option-treatment-adjustment`` evidence (X76 part (d)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_treatment_adjustment`,
:mod:`tests.studies.option_treatment_adjustment_properties` and
:mod:`tests.studies._option_adjustment_laws`, fixed before the run and recomputed by
``tests/unit/test_option_treatment_adjustment_design.py`` from the committed pilot
(``tests/studies/option_treatment_adjustment_pilot.csv``).

Red-cell reading (declared before any run; Fix-8 plan section 5.0).

* A red ``interval_calibration`` positive is finite-sample only when the 99% interval of its SE
  ratio contains 1, its coverage interval contains 0.95 and the diagnosis finds no defect.  It
  then gets an owner row in the roadmap's Red-cell owners table, and its ``T54`` row reads
  experimental with the measured numbers.  Every other red calibration positive is a defect
  suspect: diagnose it, fix the defect, and re-run under a fresh declaration.
* A derived control (``__shrunken_se_control``, ``__noise_control``) that does not leave the
  band is a red of the study, not of the estimator.  The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* A red ``double_robustness/treatment_correct`` (bias outside the margin, or an SE ratio
  outside the union-model band) is a defect suspect in the restricted mechanism path.
* A ``double_robustness/both_wrong`` or ``instrument_slow_outcome/n16000`` control that fails
  to fail is a design bug or a defect suspect: the exact-law limits are 1.19 and 1.91 SDs, so
  a control inside the margin means the fit did not use the declared learners or the
  declaration did not reach the mechanism.  It is diagnosed with the exact-law limits and
  fixed under a fresh declaration.
* Every reported row (the ladder at n = 1,000 and 4,000, the slope of its log absolute bias,
  the full-mechanism references and ``instrument_parametric``) is published beside its exact
  limit and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Sizes.  The two calibration cells run 6,800 replications: plan 5.0's product floor is 2,400,
and the arm-pooled predictive rule at the 8,000-draw pilot raises it.  The bias cells and
every ladder cell run 1,600, ``instrument_parametric`` 2,400 and the primary phase 2,400.

Budget.  About 0.5 CPU-hours of fits on the 4-vCPU cloud machine (in-sample point fits 0.02
to 0.03 s, stacked 0.07 s, the n = 16,000 fits up to 0.16 s), about 1.2 CPU-hours with the
harness; plan 5.9 allows 2.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    canonical_option_treatment_adjustment,
    option_treatment_adjustment_properties,
)

if __name__ == "__main__":
    run(
        canonical_option_treatment_adjustment,
        option_treatment_adjustment_properties,
        here=Path(__file__).resolve().parent,
    )
