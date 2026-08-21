"""Property adapter for stacked CV-TMLE."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import STUDY
from tests.studies.cvtmle_properties import decision_rule as _decision_rule
from tests.studies.cvtmle_properties import generate, summarize
from tests.studies.evidence.registry import StudyRecord


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate(STUDY, "stacked", n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(rows, STUDY, "stacked")


def decision_rule(record: StudyRecord, row: pd.Series) -> str:
    return _decision_rule(record, "stacked", row)
