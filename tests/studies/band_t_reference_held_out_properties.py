"""The held-out ``band-t-reference`` cell's rows, through the shared property engine.

:mod:`tests.studies.band_t_reference_properties` builds every row; this module binds it to
:data:`tests.studies.band_t_reference_held_out.DECLARATION`.
``python -m tests.studies.band_t_reference_held_out_properties --smoke`` fits the cell at a small
R on this study's pilot stream and writes nothing; ``--pilot`` runs its declared pilot.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import band_t_reference_properties as engine
from tests.studies.band_t_reference_held_out import DECLARATION
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyCell

FAILURE_BOUND = engine.FAILURE_BOUND
PROPERTY_WITNESS = engine.PROPERTY_WITNESS


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every declared replicate; the rows, and the per-replicate diagnostics by file."""
    return engine.property_artifacts(DECLARATION, n_jobs=n_jobs)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every declared replicate of the held-out cell."""
    return generate_property_artifacts(n_jobs=n_jobs)[0]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary: the joint rule gated, the paired gain reported."""
    return engine.summarize(DECLARATION, rows)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every declared cell of the held-out study."""
    return engine.property_cells_of(DECLARATION)


#: The failure accounting's declared rules: the joint rule, without the reported reference gain.
PROPERTY_RULES = (property_verdicts.simultaneous_rule(),)


def main(argv: Sequence[str] | None = None) -> None:
    """``--smoke`` or ``--pilot`` for the held-out study."""
    engine.main(argv, DECLARATION)


if __name__ == "__main__":
    main()
