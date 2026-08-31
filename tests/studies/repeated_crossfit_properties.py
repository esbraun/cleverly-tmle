"""Independent properties for repeated point-treatment cross-fitting."""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.cvtmle_properties import generate, summarize
from tests.studies.evidence.property_verdicts import finish
from tests.studies.repeated_crossfit import N_FOLDS, REPEATS, STUDY


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate(
        "repeated",
        repeats=REPEATS,
        n_folds=N_FOLDS,
        include_overfitting=False,
        n_jobs=n_jobs,
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = summarize(
        rows,
        STUDY,
        "repeated",
        include_overfitting=False,
        return_parts=True,
    )
    return finish(summary, rates)
