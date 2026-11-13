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
  its exact-law limit is 1.37 SDs (of the limit fit's plug-in curve SD at n = 2,000; the pilot
  read 1.32 empirical SDs), so a control inside the margin means the window was not
  applied.  It is diagnosed with the exact-law limit and fixed under a fresh declaration.
* A red ``order_double_robustness/ey_never`` (bias not inside the margin) is a defect suspect in
  the window of the treatment and censoring mechanisms: its exact-law limit is 0, and the
  shipped mutation that drops ``A_{t-1}`` from that window moves the limit to -0.73 plug-in SDs.
* An ``order_double_robustness/ey_never_blind_mechanisms`` control that fails to fail is a
  design bug or a defect suspect: with intercept-only mechanisms its exact-law limit is 0.54
  plug-in SDs (the pilot read 1.23 empirical SDs, because the blind fit's plug-in SE is about
  2.3 times its spread), so a control inside the margin means the order-0 outcome window was
  not applied or the mechanisms were not ignored.
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

``LF2``'s outcome (review of the declaration, M1).  ``LF2`` first shared ``LF1``'s outcome, on
which the order-0 fit with intercept-only mechanisms sat inside the margin (-0.17 plug-in SDs on
the ATE, +0.21 on ``EY(never)``), so the double-robustness positive could not show that the
mechanisms do the rescue.  ``LF2`` now has its own outcome, ``0.05 + 0.04 W + 0.9 L_3``, which
leans on the ``L_3`` the order-0 window drops, and the blind fit is a gated control on its own
stream.  The pair reads ``EY(never)``: inside the linear-probability range the ATE control
reaches only 0.42 plug-in SDs, under the 0.45 floor.

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
