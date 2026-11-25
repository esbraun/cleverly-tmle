"""The property module of ``continuous-survival-variance``: it declares no property family.

The record's readings (the ship rule, its controls and the floors) are computed from the primary
rows by :mod:`tests.studies.continuous_survival_variance_readings`, so its property phase writes
the schema-bearing empty tables and nothing else.
"""

from __future__ import annotations

import pandas as pd

from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS as PROPERTY_COLUMNS

#: No property fit runs, so no property failure can be recorded against this bound.
FAILURE_BOUND = 0.01
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = 1) -> pd.DataFrame:
    """No property rows."""
    del n_jobs
    return pd.DataFrame(columns=list(PROPERTY_COLUMNS))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """No property cell, so no verdict: an empty table with the columns the evidence reads."""
    if not rows.empty:
        raise ValueError("continuous-survival-variance declares no property cell, and got rows")
    return property_verdicts.empty_property_table()
