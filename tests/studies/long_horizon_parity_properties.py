"""The property module of the two X59 parity records: they declare no property family.

A parity record reads only its exact gates (``parity.csv``), so its property phase writes the
schema-bearing empty tables and nothing else.
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
    """No property verdicts."""
    del rows
    return pd.DataFrame(columns=["property", "cell", "passed"])
