"""Property adapter for stacked CV-TMLE.

The shared cross-fitted families come from :mod:`tests.studies.cvtmle_properties`.  What
this row adds is the fold-policy diagnostic, and it is declared here rather than in the
shared module because it belongs to this study alone: the other three cross-fitted rows
report the same construction under a different aggregation, and running the comparison
four times would publish one experiment four times.
"""

from __future__ import annotations

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import bounded_cv_laws
from tests.studies.canonical_cvtmle import G_BOUNDS as CV_G_BOUNDS
from tests.studies.canonical_cvtmle import STUDY
from tests.studies.cvtmle_properties import cells, generate, summarize
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import finish, fold_policy_diagnostics

#: The variant name the shared module labels this row's overfitting cell with.
VARIANT = "stacked"


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, shared and study-specific alike.

    Returns
    -------
    tuple of PropertyCell
        The cross-fitted families from :func:`tests.studies.cvtmle_properties.cells`,
        followed by the three fold-policy arms.
    """
    return (*cells(VARIANT), *bounded_cv_laws.fold_policy_cells())


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The shared cross-fitted cells, then the three fold-policy arms.

    Two calls rather than one, because the fold-policy arms reach two splits the package
    refuses and are fitted through :class:`~tests.studies.bounded_cv_laws.FoldPolicyTMLE`
    rather than configured.  They sample from a binary law and pass no ``q_bounds``, which
    is the one thing every other cell here does pass.
    """
    return pd.concat(
        [
            generate(VARIANT, n_jobs=n_jobs),
            run_cells(
                bounded_cv_laws.fold_policy_cells(),
                bounded_cv_laws.fold_policy_estimator(g_bounds=CV_G_BOUNDS),
                n_jobs=n_jobs,
            ),
        ],
        ignore_index=True,
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Every shared verdict, then the fold-policy numbers the study reports rather than gates."""
    summary, rates = summarize(rows, STUDY, VARIANT, return_parts=True)
    fold_policy_diagnostics(summary, rows, STUDY)
    return finish(summary, rates)
