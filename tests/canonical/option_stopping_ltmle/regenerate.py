"""Regenerate the ``option-stopping-ltmle`` evidence (X76 part (a)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator, so the run starts no container. The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_stopping_ltmle` and
:mod:`tests.studies.option_stopping_ltmle_properties`, fixed before the run and recomputed by
``tests/unit/test_option_stopping_ltmle_design.py`` from the committed pilot
(``tests/studies/option_stopping_pilot.csv``).

Sizes. Every property cell runs 6,800 replications: plan section 5.0's product floor is 2,800,
and the arm-pooled predictive rule at the 2,000-draw pilot raises it, because the cells of one
law read the same draws and the pilot holds one SE-ratio reading per law. The pilot's four gated
calibration cells read widened coverage 0.9445 (L3) to 0.9530 (L4 ``scaled_var``). The primary
phase runs 2,800 replications. Measured on the 4-vCPU cloud machine: about 7.3 CPU-hours of
fits (6.4 for the property cells, 0.9 for the primary phase), about 15 with the harness.

On L3 the per-node Newton walk never binds: its pilot residual is about 7.6e-11 SE under either
rule, identical in coverage and SE ratio to the relative fit. The L3 cells license only that
Newton walk, not a one-step end-of-study walk under a rule.

Red-cell reading (declared before any run; Fix-8 plan section 5.0 and ``guide/studies.md``).

* A red ``interval_calibration`` positive is finite-sample only when the 99% interval of its SE
  ratio contains 1, its coverage interval contains 0.95 and the diagnosis finds no defect. It
  then gets an owner row, and its ``G55`` row reads experimental with the measured numbers.
  Every other red calibration positive is a defect suspect: diagnose it, fix the defect, and
  re-run under a fresh declaration.
* A derived control that does not leave the band, or the N control
  (``residual_necessity/l4_mc_risk__plain``) whose plain-interval coverage upper limit is not
  below 0.92, is a red of the study, not of the estimator. The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* The N positive (``residual_necessity/l4_mc_risk__widened``) is a coverage-floor reading: its
  lower limit is at least 0.90, because the widened interval is conservative by construction.
  A red N positive is a defect suspect.
* A red W cell (L4 ``scaled_var``, share of draws whose residual is at least 0.01 SE, lower
  limit below 0.5) is a red of the study's witness; the K cells keep their own verdicts.
* A red primary row (a ``cleverly`` truth row) reads as a red calibration positive does: finite
  sample only under the same three conditions, with an owner row, and otherwise a defect suspect.
* No red control, W cell or N cell takes an owner row.
  ``tests/unit/test_option_stopping_ltmle_design.py`` applies this reading through
  ``tests.studies.mtp_study_tools.gating_findings`` to the gated cells and the truth rows.
* Every ``diagnostic`` row (plain intervals, the arm-1 risk, relative fits, the N fit's
  contrast, the ``mechanism_correct`` fit at threshold times 1, rule-met shares and residual
  distributions) is published and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_option_stopping_ltmle, option_stopping_ltmle_properties

if __name__ == "__main__":
    run(
        canonical_option_stopping_ltmle,
        option_stopping_ltmle_properties,
        here=Path(__file__).resolve().parent,
    )
