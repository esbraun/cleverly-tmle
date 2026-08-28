"""Cross-fitted categorical longitudinal TMLE property adapter."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_properties as shared
from tests.studies.canonical_categorical_ltmle_crossfit import STUDY

EFFICIENCY_RATIO_BAND = shared.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = shared.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared.TARGETING_DISPLACEMENT
NECESSITY_DISPLACEMENT = shared.NECESSITY_DISPLACEMENT
CATEGORICAL_PROBABILITY_DISPLACEMENT = shared.CATEGORICAL_PROBABILITY_DISPLACEMENT


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return shared.generate_property_rows(STUDY, cross_fit=True, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return shared.summarize_properties(rows, STUDY, cross_fit=True)
