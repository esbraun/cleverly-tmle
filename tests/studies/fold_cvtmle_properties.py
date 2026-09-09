"""Property adapter for fold-evaluated CV-TMLE."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.cvtmle_properties import generate, summarize
from tests.studies.fold_evaluated_cvtmle import STUDY


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate("fold_evaluated", n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(rows, STUDY, "fold_evaluated")
