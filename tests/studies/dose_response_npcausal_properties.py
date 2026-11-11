"""The property adapter of ``dose-response-npcausal``, which declares no property cell.

The study pairs ``DoseResponseTMLE`` with npcausal ``ctseff`` on its primary scenario; the
``dose-response`` study (X93 S1) holds every property cell of the estimator.  The regeneration
driver still writes both property artefacts, so this module returns a schema-bearing empty
table and an empty summary with the verdict columns the driver reads.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS

#: The share of a cell's requested replications the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """No property cell, so no row."""
    del n_jobs
    return pd.DataFrame(columns=list(REPLICATE_COLUMNS))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """No property cell, so no verdict."""
    if not rows.empty:
        raise ValueError("the comparator study declares no property cell, and it received rows")
    return pd.DataFrame(
        {
            "property": pd.Series(dtype=str),
            "cell": pd.Series(dtype=str),
            "role": pd.Series(dtype=str),
            "passed": pd.Series(dtype=bool),
            "property_passed": pd.Series(dtype=bool),
        }
    )
