"""Regenerate the learned-rule EP rule-learner evidence (P1 PR-c) under the RM30 run form.

Red-cell readings, declared before any run and before the sizing pilot's summary was read
(P1 plan section 5.5).  A red cell is diagnosed before a route is chosen; no cell is re-sized,
re-drawn or downgraded after the run.  Plan rev 2.6 moved the calibration cells from n = 4,000 to
n = 16,000 after both pilots were read and before any declared draw; it changed the size each
reading names and the sizing reading's pilot, and no reading's text otherwise.  Plan rev 2.7,
also before any declared draw, added the limit-rule paired fit to every calibration replicate
(the ``limit_*`` columns), the reading that reads it, and the optimal cell to the first reading.

* A ``<law>__declared`` or ``optimal_known_g__declared`` cell whose SE-ratio interval lies below
  the band while its oracle-SE ratio (the fixed-rule SE of the realized rules over the error
  SD) sits below one by about the same amount reads as the rule learner's fold covariance:
  each fold's curve is fitted on the other folds' validation rows, the second-order excess the
  learned-rule-learners pilot measured at n = 1,000 and 2,000.  It is a finding about the size,
  not a defect: the cell is published red under ``reporting`` with the owner row P1, its
  ``T49`` (or ``T49.optimal``) row is not added, and the page states the measured deficit.
* A ``<law>__declared`` or ``optimal_known_g__declared`` cell whose SE-ratio interval lies above
  the band while its limit-rule fit's SE ratio (``limit_std_error`` over the SD of
  ``limit_estimate - limit_truth`` on the same rows) sits above one by about the same amount
  reads as the fixed-rule CV-TMLE's own conservative variance, which a wrong outcome regression
  gives.  It is not a rule-learner defect: the cell is published red under ``reporting`` with
  the owner row P1, and that law's ``T49`` (or ``T49.optimal``) row is not added.
* An SE-ratio cell red while neither ratio of the two readings above explains it, below or
  above the band, is a variance defect suspect.
* A ``<law>__declared`` cell whose coverage is red while its SE ratio is green reads as a bias of
  the realized-rule target: a fold-locality or harness defect suspect.  Diagnose it against the
  unit witnesses W-c1 to W-c3, the published ``rule_rows_checked`` and the standardized bias.
* ``optimal_known_g__declared`` red on its coverage while both learned-value cells are green is
  not explained by the EP limit rule, whose regret is 0.005 SE at n = 16,000: it is a defect
  suspect.  With no defect found it is published red under ``reporting`` with the owner P1, and
  ``T49.optimal`` is not added.  It is the only cell that sees the learned rule's quality: the
  other cells' target is the learned rule's own value.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band
  is an instrument defect, and the run is void.
* ``non_exceptional__validation_rule_control`` whose coverage interval does not lie below 0.92
  means the instrument does not resolve a curve fitted on its own validation rows at n = 2,000.
  The run's fold-locality evidence then rests on the unit witnesses W-c2 and W-c3 alone; the
  ``T49`` rows are not added until a stronger control is declared under a fresh run.
* The primary phase (n = 2,000) red on its coverage floor 0.90 or its SE sanity screen while the
  n = 16,000 cells are green reads as the finite-sample rule-learner deficit: reported.  Red
  with a red n = 16,000 cell reads with that cell.  Whatever the primary phase shows, the
  ``T49`` and ``T49.optimal`` limits carry
  :data:`tests.studies.learned_rule_ep_properties.T49_LIMITS`: calibrated at n = 16,000 at
  these laws, with the n = 4,000 pilot's deficit.
* The pilot's sizing reading.  When the ``non_exceptional`` pilot's SE ratio minus one pilot SE
  at the calibration size (the n = 16,000 cell of the size pilot, rev 2.6) is below 0.96,
  R = 7,200 and the ``T49`` rows stay experimental whatever the run shows, with the reason "the
  calibration cells are not powered at the measured SE deficit" (plan 5.5).

The run predicts some false reds under a correct method: the primary floor's assurance is about
0.77, and the chance that some gated n = 16,000 cell is red is about 16% on the unpaired reading
and 6% on the pooled limit-rule reading (``tests/unit/test_learned_rule_ep_design.py``).  The
readings above are written for them.

The reporting cells state no verdict.  ``small_sample_diagnostic`` publishes the SE ratio, the
coverage and the bias at n = 1,000.  ``exceptional_law_report`` publishes the coverage at the
RM30 ``exceptional`` law and the declared reading of
:func:`tests.studies.learned_rule_ep_properties.exceptional_reading`.
"""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_ep, learned_rule_ep_properties

if __name__ == "__main__":
    run(learned_rule_ep, learned_rule_ep_properties, here=Path(__file__).resolve().parent)
