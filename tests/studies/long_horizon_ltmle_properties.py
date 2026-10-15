r"""Property families of ``long-horizon-ltmle`` (S1): ordinary ``LTMLE``, binary nodes.

Every cell draws from a law of :mod:`tests.discrete_law_longitudinal_horizon` at the n the
design declares (:mod:`tests.long_horizon_design`), with ``g_bounds=(1e-8, 1)`` unless a row
says otherwise.  :mod:`tests.studies.long_horizon_properties` builds the fit sets and the
verdicts; this module declares the budgets.  "Policy" is ``coin``, "ref" is ``never``.

=====================================  =========================================================
family                                 cells
=====================================  =========================================================
``interval_calibration``               per K in 3, 4, 5: ``respond_vs_ref``, ``policy``,
                                       ``policy_vs_ref``, ``policy_vs_policylast``,
                                       ``static_contrast``; and ``msm_duration__k5``; each
                                       ``__known_mechanism`` with its shrunken and noise
                                       controls, and ``policy_vs_policylast__k{K}`` with its
                                       ``inflated_se_control``
``root_n_and_efficiency``, rate rows   ``static_contrast__k4__n_{4000, 12000, 48000}`` on
                                       ``L_R``; 4,000 is the control rung
``double_robustness``                  ``respond_vs_ref__k5__{both_correct, outcome_correct,
                                       mechanism_correct}`` at 40,000 and ``both_wrong``
                                       at 40,000
``targeting_necessity``                ``{policy, respond}__k{K}__{targeted, untargeted}``
``censoring_necessity``                ``policy__k5__{declared, ignored}`` on ``L_R`` with
                                       ``gamma_C = 2``: 8,000 and 36,000
``rule_necessity``                     ``respond__k5__{declared_rule, reversed_rule}`` on
                                       (0.5, 0, 3.5): 4,000 and 116,000
``policy_necessity``                   ``policy__k5__{declared_policy, mix_reversed_control}``:
                                       4,000 and 16,000
``type_i_error``, ``power``            ``policy_vs_policylast__k5__{sharp_null, alternative}``
                                       at :data:`~tests.long_horizon_design.POWER_N`
``simultaneous_coverage``              ``k5__simultaneous_band`` and its pointwise control over
                                       the six K = 5 plans on ``L_R`` at 76,000
``support_loss`` (reporting)           ``{static, policy}__k5__n4000__{default_bounds,
                                       no_bounds}``
``learned_mechanism`` (reporting)      ``respond_vs_ref__k5__learned`` at 40,000
=====================================  =========================================================

**Budgets.**  The replicate counts below are provisional.  The declaration commit sets them from
the pilots.

**Declared red-cell reading.**  A red calibration positive is a
finite-sample red only when its 99 percent SE-ratio interval contains 1, its coverage interval
contains 0.95, and the diagnosis finds no defect; it then goes to an owner row.  Any other red
calibration row is a defect suspect: diagnose, fix, and re-run under a fresh declaration.  A
control that passes is a design defect, because its exact displacement is at least 3 sampling
SDs.  A reporting cell has no verdict; the page prints it beside the exact-law prediction.  A
recorded fitting failure is accounted under :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import long_horizon_properties as shared
from tests.studies.canonical_long_horizon_ltmle import STUDY as STUDY
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyCell
from tests.studies.evidence.replication import fit_properties

TAG = "S1"

#: Provisional: the calibration pilot sets each study's calibration R.
CALIBRATION_REPLICATES = 6_000
#: Provisional: the bias pilots pick R from 1,000, 1,200, 2,000, 3,000 and 4,000.
BIAS_REPLICATES = 2_000
ROOT_N_REPLICATES = 1_000
TESTING_REPLICATES = 2_000
BAND_REPLICATES = 2_000
REPORTING_REPLICATES = 2_000

#: The replicate count of each family.
REPLICATES = {
    "interval_calibration": CALIBRATION_REPLICATES,
    "root_n_and_efficiency": ROOT_N_REPLICATES,
    "double_robustness": BIAS_REPLICATES,
    "targeting_necessity": BIAS_REPLICATES,
    "censoring_necessity": BIAS_REPLICATES,
    "rule_necessity": BIAS_REPLICATES,
    "policy_necessity": BIAS_REPLICATES,
    "type_i_error": TESTING_REPLICATES,
    "power": TESTING_REPLICATES,
    "simultaneous_coverage": BAND_REPLICATES,
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
