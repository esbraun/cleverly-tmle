"""Regenerate the ``option-markov-order`` evidence (X76 part (f)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_markov_order`,
:mod:`tests.studies.option_markov_order_properties` and :mod:`tests.studies._option_markov_law`,
fixed before the run and recomputed by ``tests/unit/test_option_markov_order_design.py`` from
the committed pilot (``tests/studies/option_markov_order_pilot.csv``).

Red-cell reading (declared before any run; Fix-8 plan section 5.0).  The design test applies it
mechanically to the committed tables (``gating_findings`` over the 11 gated cells).

* A red ``interval_calibration`` positive is finite-sample only when the 99% interval of its SE
  ratio contains 1, its coverage interval contains 0.95 and the diagnosis finds no defect.  It
  then gets an owner row in the roadmap's Red-cell owners table, and its ``G57`` row reads
  experimental with the measured numbers.  The diagnosis reads the full-history fit on the same
  draws (``markov_reported/full_history_*``): a shortfall the full-history fit shares is a
  property of the law and the sample size, not of the window.  Every other red calibration
  positive is a defect suspect: diagnose it, fix the defect, and re-run under a fresh
  declaration.
* A derived control (``__shrunken_se_control``, ``__noise_control``) that does not leave the
  band is a red of the study, not of the estimator.  The claim of the cell it guards is
  withheld until the control is repaired under a fresh declaration.
* An ``order_necessity/ate`` control that fails to fail is a design bug or a defect suspect:
  its exact-law limit is 1.37 SDs, so a control inside the margin means the window was not
  applied.  It is diagnosed with the exact-law limit and fixed under a fresh declaration.
* A red ``order_double_robustness/ate`` (bias not inside the margin) is a defect suspect in the
  window of the treatment and censoring mechanisms: its exact-law limit is 0.
* Every ``markov_reported`` row is published beside its exact limit and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Law.  The rev 2 law ``LF1`` had the treatment mechanism
``expit(-0.6 + 0.4 W - 0.5 L_t + 0.3 A_{t-1} + 1.5 L_{t-1})``.  Its pilot covered 0.919 for
``EY(always)`` at n = 2,000, and the full-history fit on the same draws covered 0.919 too: the
shortfall came from weak support for the always-treated path, not from the window.  Under the
plan review's rule (a law change only when the option-free fit shows the same shortfall) the law
was replaced by the declared one, with intercept -1.0 and carry-over 2.0 on ``A_{t-1}``, which
keeps a treated unit treated.  The discarded law is recorded in the study's configuration.

Sizes.  The three calibration cells run 4,000 replications on one draw stream: plan 5.0's
product floor for three cells is 2,800, and the arm-pooled predictive rule at the 8,000-draw
pilot (draws 0 to 7,999 of the calibration stream) raises it, pricing pilot SE ratios of 0.986,
0.992 and 0.989.  The full-history fit runs on the same draws.  The necessity control and the double-robustness positive run 1,600 each, and the
primary phase 2,800.

Budget.  About 0.3 CPU-hours of fits on the 4-vCPU cloud machine (the run fits 18,000 samples,
and the pilot fitted 24,000 in 22 CPU-minutes), under an hour with the harness; plan 5.9 allows
2.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_option_markov_order, option_markov_order_properties

if __name__ == "__main__":
    run(
        canonical_option_markov_order,
        option_markov_order_properties,
        here=Path(__file__).resolve().parent,
    )
