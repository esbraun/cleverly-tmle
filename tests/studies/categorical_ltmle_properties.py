"""Ordinary categorical longitudinal TMLE property adapter."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_properties as shared
from tests.studies.canonical_categorical_ltmle import STUDY
from tests.studies.evidence import property_verdicts

DERIVED_CELLS = shared.DERIVED_CELLS
EFFICIENCY_RATIO_BAND = shared.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = shared.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared.TARGETING_DISPLACEMENT
NECESSITY_DISPLACEMENT = shared.NECESSITY_DISPLACEMENT
CATEGORICAL_PROBABILITY_DISPLACEMENT = shared.CATEGORICAL_PROBABILITY_DISPLACEMENT


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return shared.generate_property_rows(STUDY, cross_fit=False, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return shared.summarize_properties(rows, STUDY, cross_fit=False)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(*shared.NECESSITIES),
)
