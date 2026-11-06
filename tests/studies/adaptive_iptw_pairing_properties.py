"""The property adapter of ``adaptive-iptw-pairing``, which declares no property cell.

The study pairs ``AdaptiveIPTW`` with R ``drtmle`` on its primary scenarios; the gate study
``adaptive-iptw`` holds every property cell.  The regeneration driver still writes both
property artefacts, so this module returns a schema-bearing empty table and an empty summary
with the verdict columns the driver reads.
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
        raise ValueError("the pairing study declares no property cell, and it received rows")
    return pd.DataFrame(
        {
            "property": pd.Series(dtype=str),
            "cell": pd.Series(dtype=str),
            "role": pd.Series(dtype=str),
            "passed": pd.Series(dtype=bool),
            "property_passed": pd.Series(dtype=bool),
        }
    )
