"""Property adapter for fold-evaluated CV-TMLE.

Every family this row runs is one of the shared cross-fitted families in
:mod:`tests.studies.cvtmle_properties`, so the cells are that module's and only the
overfitting cell's label is this study's.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.cvtmle_properties import cells, generate, summarize
from tests.studies.evidence.properties import PropertyCell
from tests.studies.fold_evaluated_cvtmle import STUDY

#: The variant name the shared module labels this row's overfitting cell with.
VARIANT = "fold_evaluated"


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, with the bounded law each one samples from.

    Returns
    -------
    tuple of PropertyCell
        The cross-fitted families from :func:`tests.studies.cvtmle_properties.cells`.
    """
    return cells(VARIANT)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate(VARIANT, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(rows, STUDY, VARIANT)
