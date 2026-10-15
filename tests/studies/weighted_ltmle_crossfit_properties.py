"""Properties for cross-fitted weighted end-of-study longitudinal TMLE."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import weighted_longitudinal_properties_common as shared
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY
from tests.studies.evidence import property_verdicts

DERIVED_CELLS = shared.DERIVED_CELLS
EFFICIENCY_RATIO_BAND = shared.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = shared.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared.TARGETING_DISPLACEMENT
WEIGHT_DISPLACEMENT = shared.WEIGHT_DISPLACEMENT
LEARNER_WEIGHT_DISPLACEMENT = shared.LEARNER_WEIGHT_DISPLACEMENT


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return shared.generate_property_rows(STUDY, cross_fit=True, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return shared.summarize_properties(rows, STUDY)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(*shared.NECESSITIES),
)
