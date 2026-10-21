r"""Property families of ``long-horizon-categorical-ltmle`` (S2): ``LTMLE``, three arms.

Every cell draws from a law of :mod:`tests.discrete_law_longitudinal_horizon` at the n the
design declares (:mod:`tests.long_horizon_design`), with ``g_bounds=(1e-8, 1)`` unless a row
says otherwise.  :mod:`tests.studies.long_horizon_properties` builds the fit sets and the
verdicts; this module declares the budgets.  "Policy" is ``mix``, "ref" is ``low``.

=======================================  =======================================================
family                                   cells
=======================================  =======================================================
``interval_calibration``                 per K in 3, 4, 5: ``respond_vs_ref``, ``policy``,
                                         ``policy_vs_ref``, ``policy_vs_policylast``,
                                         ``static_contrast``; and the five-fold
                                         ``policy_vs_ref__k5_crossfit`` and
                                         ``policy_vs_policylast__k5_crossfit``; each
                                         ``__known_mechanism`` with its shrunken and noise
                                         controls, and each paired policy contrast with its
                                         ``inflated_se_control``
``root_n_and_efficiency``, rate rows     ``respond_vs_ref__k4__n_{4000, 12000, 48000}`` on
                                         ``L_R``; 4,000 is the control rung and carries no
                                         claim
``double_robustness``                    ``respond_vs_ref__k5__{both_correct, outcome_correct,
                                         mechanism_correct}`` at 48,000 and ``both_wrong``
                                         at 52,000
``targeting_necessity``                  ``{policy, respond}__k{K}__{targeted, untargeted}``
``censoring_necessity``                  ``policy__k5__{declared, ignored}`` on (0.75, 0.05,
                                         3.5) with ``gamma_C = 2``: 60,000 and 128,000
``categorical_probability_necessity``    ``third_arm__k5__{assigned_probability,
                                         binary_complement}`` on (0.25, 0.05, 5.5): 4,000 and
                                         296,000 (:data:`~tests.long_horizon_design.NMAX_Q14`)
``rule_necessity``                       ``respond__k3__{declared_rule, reversed_rule}`` on
                                         (0.75, 0.1, 3.5): 4,000 and 112,000
``policy_necessity``                     ``policy__k5__{declared_policy, mix_node1_control}``
                                         on (1, 0.05, 3.5): 96,000 each
``simultaneous_coverage``                the band over the six K = 5 plans on (0.25, 0, 3.5)
                                         at 64,000, and its pointwise control
``crossfit_overfitting``                 ``policy_vs_ref__k5__{cross_fitted,
                                         in_sample_control}``: grown trees on ``L_R`` at 16,000
``support_loss`` (reporting)             ``{static, policy}__k5__n4000__{default_bounds,
                                         no_bounds}``, ``static__k5__n{16000, 64000}__no_bounds``
``learned_mechanism`` (reporting)        ``respond_vs_ref__k5__learned`` at 48,000
=======================================  =======================================================

**Budgets.**  The replicate counts below are provisional.  The declaration commit sets them from
the pilots.

**Declared red-cell reading** (plan section 5.6, before any run).  The study publishes under
``reporting``.  Its gated truth rows (``GATED_ROWS``) and gated property cells (``GATED_CELLS``)
are listed in :mod:`tests.long_horizon_design`.  ``tests/unit/test_long_horizon_design.py`` fails
on any red member of either list once the study commits its summaries.  A row applies to the
study that declares its family.

=====================================  =========================================================
red cell                               reading
=====================================  =========================================================
a primary truth row                    Only a gated row blocks: a row gates in its gating
                                       scenario alone, and the same quantity in another
                                       scenario reports.  Diagnose.  A defect is fixed and the
                                       run repeats under a fresh declaration.
a calibration positive                 Finite-sample only when its 99 percent SE-ratio interval
(``interval_calibration``)             contains 1, its coverage interval contains 0.95, and the
                                       diagnosis finds no defect; it then goes to an owner row.
                                       Any other red calibration row is a defect suspect:
                                       diagnose, fix, re-run under a fresh declaration.  An
                                       unfixed defect never ends as an owner row.
a bias-family positive                 Its 99 percent standardized-bias interval does not lie
(the necessity families,               inside the margin.  Defect suspect: diagnose the
``sequential_double_robustness``,      estimator, then the law and the learner rung of the
``double_robustness``)                 design test.  Finite-sample only when the diagnosis finds
                                       no defect and the interval is wider than the margin
                                       through R alone; it then goes to an owner row.  The run
                                       never adds replicates to a red positive.
a control that passes                  A design defect, because its exact displacement is at
                                       least 3 sampling SDs.  Diagnose the estimator and the
                                       design test first.  Fix the control or the law under a
                                       fresh declaration, never by buying replications or
                                       lowering its floor.
the simultaneous band                  Its 99 percent joint-coverage interval leaves (0.92,
(``simultaneous_coverage``)            0.98).  Defect suspect on either side: below, the critical
                                       value or the tail regime is wrong; above, the band is
                                       conservative and the diagnosis names why.  The pointwise
                                       control must stay below nominal.
the overfitting positive               Its SE-ratio interval leaves [0.85, 1.20], or the coverage
(``crossfit_overfitting``)             gain over the in-sample control falls short.  Defect
                                       suspect: own-row reuse across the K nodes.  The in-sample
                                       control must fall below the SE-ratio ceiling.
a root-n rung or slope row             A positive rung (12,000 and 48,000) is red when its
(``root_n_and_efficiency``,            coverage lower limit is below 0.90, its SE-ratio interval
``root_n_rate``)                       leaves (0.80, 1.20), or its bias is not established
                                       inside the margin.  A slope row is red when its interval
                                       is not within 0.125 of -0.5 or reaches -0.25.  Defect
                                       suspect, with the rung pilots read first.  The 4,000 rung
                                       carries no claim at R = 1,000, because its Clopper-
                                       Pearson interval is narrower than the 0.05 gap between
                                       the coverage floor and nominal; the page does not list it
                                       as a control.
type I error and power                 Size is red when the upper limit of its rejection
(``type_i_error``, ``power``)          interval exceeds 0.10 or its coverage lower limit falls
                                       below 0.90.  Power is red when the lower limit of its
                                       rejection rate is below 0.80.  Defect suspect; the sharp
                                       null has truth 0 exactly.
a ``support_loss`` or                  Reporting; no verdict.  The page states the measured
``learned_mechanism`` cell             values, with the measured binding share per node, beside
                                       the exact-law prediction.  The two-sided tripwire of the
                                       plan makes a cell a defect suspect.
a recorded fitting failure             Accounted by RM39b2's ``require_accounted`` under
                                       ``FAILURE_BOUND`` = 0.01 and the module's declared clause
                                       tree.  A cell over the bound is red and diagnosed.
=====================================  =========================================================

No budget, margin, law, learner, fold count or seed moves after a verdict is seen, except under
the diagnosed-defect route with a fresh declaration commit.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import long_horizon_properties as shared
from tests.studies.canonical_long_horizon_categorical_ltmle import STUDY as STUDY
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyCell
from tests.studies.evidence.replication import fit_properties

TAG = "S2"

#: The calibration pilot sets each study's calibration R (pilot merge at 67ac67df).
CALIBRATION_REPLICATES = 14_000
#: The bias pilots read R 1,000 on the ladder 1,000 to 4,000; the study keeps 2,000.
BIAS_REPLICATES = 2_000
ROOT_N_REPLICATES = 1_000
BAND_REPLICATES = 2_000
OVERFIT_REPLICATES = 2_000
REPORTING_REPLICATES = 2_000

#: The replicate count of each family.
REPLICATES = {
    "interval_calibration": CALIBRATION_REPLICATES,
    "root_n_and_efficiency": ROOT_N_REPLICATES,
    "double_robustness": BIAS_REPLICATES,
    "targeting_necessity": BIAS_REPLICATES,
    "censoring_necessity": BIAS_REPLICATES,
    "categorical_probability_necessity": BIAS_REPLICATES,
    "rule_necessity": BIAS_REPLICATES,
    "policy_necessity": BIAS_REPLICATES,
    "simultaneous_coverage": BAND_REPLICATES,
    "crossfit_overfitting": OVERFIT_REPLICATES,
    "support_loss": REPORTING_REPLICATES,
    "learned_mechanism": REPORTING_REPLICATES,
}

#: The share of a cell's requested replications that a run may lose .
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: Each calibration positive to its derived controls.
DERIVED_CELLS = shared.derived_cells(TAG)

#: The bias families' rule; every other family reads its shared rule.
PROPERTY_RULES = (shared.bias_rule(),)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, n and replicate count."""
    return shared.declared_cells(TAG, REPLICATES)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Every property row: one ``fit_properties`` loop per family, then the derived controls."""
    return shared.run_loops(TAG, REPLICATES, fit_properties, n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the bias-family verdicts."""
    return shared.summarize(TAG, rows)
