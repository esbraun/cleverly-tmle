"""The property adapter of the learned-rule boundary study, which declares no property cell.

The RM30 declaration gives the reporting study the gated study's primary verdicts and no
property cell.  The regeneration driver still writes both property artefacts, so this module
returns a schema-bearing empty table and an empty summary with the verdict columns the driver
reads.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies._learned_rule_law import HARNESS_COLUMNS
from tests.studies.evidence.properties import REPLICATE_COLUMNS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """No property cell, so no row."""
    return pd.DataFrame(columns=[*REPLICATE_COLUMNS, *HARNESS_COLUMNS])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """No property cell, so no verdict."""
    if not rows.empty:
        raise ValueError("the boundary study declares no property cell, and it received rows")
    return pd.DataFrame(
        {
            "property": pd.Series(dtype=str),
            "cell": pd.Series(dtype=str),
            "role": pd.Series(dtype=str),
            "passed": pd.Series(dtype=bool),
            "property_passed": pd.Series(dtype=bool),
        }
    )
