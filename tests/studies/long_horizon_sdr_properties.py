r"""Property families of ``long-horizon-sdr`` (S3): ``SDR``, binary and three-level nodes.

Every cell draws from a law of :mod:`tests.discrete_law_longitudinal_horizon` at the n the
design declares (:mod:`tests.long_horizon_design`), with ``g_bounds=(1e-8, 1)`` unless a row
says otherwise.  :mod:`tests.studies.long_horizon_properties` builds the fit sets and the
verdicts; this module declares the budgets.  A cell of a family that runs at both arm counts
carries the prefix ``binary_`` or ``categorical_``.

=======================================  =======================================================
family                                   cells
=======================================  =======================================================
``interval_calibration``                 per arm count and K in 3, 4, 5: ``respond_vs_ref``,
                                         ``policy``, ``policy_vs_ref``,
                                         ``policy_vs_policylast``, ``static_contrast``; and the
                                         five-fold ``categorical_policy_vs_ref__k5_crossfit``
                                         and ``categorical_policy_vs_policylast__k5_crossfit``;
                                         each ``__known_mechanism`` with its shrunken and noise
                                         controls, and each paired policy contrast with its
                                         ``inflated_se_control``
``root_n_and_efficiency``, rate rows     ``respond_vs_ref__k4__n_{4000, 12000, 48000}``, three
                                         arms, on ``L_R``; 4,000 is the control rung
``sequential_double_robustness``         ``P_odd`` and ``P_even`` at 56,000 (gated bias) and at
                                         4,000 (reporting); ``odd_nodes_both_wrong_control``
                                         at 56,000 and ``ltmle_P_odd_control`` at 72,000
``censoring_necessity``                  ``{binary, categorical}_policy__k5__{declared,
                                         ignored}``: 4,000 and 16,000 on ``L_R``; 96,000 and
                                         116,000 on (1, 0.05, 3.5)
``categorical_probability_necessity``    ``third_arm__k5__{assigned_probability,
                                         binary_complement}`` on (1, 0.05, 3.5) at 96,000
``policy_necessity``                     ``binary_policy__k5__{declared_policy,
                                         mix_reversed_control}`` (4,000 and 16,000) and
                                         ``categorical_policy__k5__{declared_policy,
                                         mix_node1_control}`` (96,000 each)
``crossfit_overfitting``                 ``policy_vs_ref__k5__{cross_fitted,
                                         in_sample_control}``: grown trees on ``L_R`` at 16,000
``support_loss`` (reporting)             per arm count ``{static, policy}__k5__n4000__
                                         {default_bounds, no_bounds}``; three arms also
                                         ``static__k5__n{16000, 64000}__no_bounds``
``learned_mechanism`` (reporting)        ``{binary, categorical}_respond_vs_ref__k5__learned``
=======================================  =======================================================

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
from tests.studies.canonical_long_horizon_sdr import STUDY as STUDY
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyCell
from tests.studies.evidence.replication import fit_properties

TAG = "S3"

#: Provisional: the calibration pilot sets each study's calibration R.
CALIBRATION_REPLICATES = 6_000
#: Provisional: the bias pilots pick R from 1,000, 1,200, 2,000, 3,000 and 4,000.
BIAS_REPLICATES = 2_000
ROOT_N_REPLICATES = 1_000
OVERFIT_REPLICATES = 2_000
REPORTING_REPLICATES = 2_000

#: The replicate count of each family.
REPLICATES = {
    "interval_calibration": CALIBRATION_REPLICATES,
    "root_n_and_efficiency": ROOT_N_REPLICATES,
    "sequential_double_robustness": BIAS_REPLICATES,
    "censoring_necessity": BIAS_REPLICATES,
    "categorical_probability_necessity": BIAS_REPLICATES,
    "policy_necessity": BIAS_REPLICATES,
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
