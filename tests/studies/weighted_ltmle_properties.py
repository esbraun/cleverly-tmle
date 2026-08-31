"""Properties for ordinary weighted end-of-study longitudinal TMLE."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import weighted_longitudinal_properties_common as shared
from tests.studies.canonical_weighted_ltmle import STUDY

EFFICIENCY_RATIO_BAND = shared.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = shared.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared.TARGETING_DISPLACEMENT
WEIGHT_DISPLACEMENT = shared.WEIGHT_DISPLACEMENT
LEARNER_WEIGHT_DISPLACEMENT = shared.LEARNER_WEIGHT_DISPLACEMENT


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return shared.generate_property_rows(STUDY, cross_fit=False, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return shared.summarize_properties(rows, STUDY)
