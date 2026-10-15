"""Property adapter for fold-evaluated CV-TMLE.

Every family this row runs is one of the shared cross-fitted families in
:mod:`tests.studies.cvtmle_properties`, so the cells are that module's and only the
overfitting cell's label is this study's.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.cvtmle_properties import cells, generate, sampling_batch, summarize
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyBatch, PropertyCell
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


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate(VARIANT, n_jobs=n_jobs)


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The one actual shared sampling call."""
    return (sampling_batch(VARIANT),)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(rows, STUDY, VARIANT)
