"""Interval calibration of the omitted-variable bound's standard error.

One family, ``interval_calibration``, with a positive cell for each end of each bound and an
``inflated_se_control`` for each end of the ATT bound.  Every cell reads the same draws: one
fit per replication gives all eight rows, so the control and its positive arm differ only in
the curve their standard error comes from.

The control reads the curve of ``nu^2`` without the conditioning-share term, which is the
curve the package reported before RM22.  On this law it is too wide, so its SE-ratio interval
must rise above the calibration band.  The ATC has no control, because the RM22 plan's probe
put its ratios without the term, 1.098 and 1.111, too close to the band's upper edge of 1.07 to
predict a failure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat
from tests.studies.evidence.seeds import stream_seed
from tests.studies.omitted_variable_bound import (
    ENDS,
    FITTED,
    PRIMARY_N,
    PROPERTY_CELLS,
    STUDY,
    TWO_SIDED,
    bound_standard_errors,
    draw_from_seed,
    fit,
)

#: Replications per property cell.  Every cell reads the same draws.
PROPERTY_REPLICATES = 10_000

#: The bounds whose ATT control the study declares.
CONTROLLED = ("att",)


def _row(
    cell: str, role: str, replicate: int, truth: float, estimate: float, std_error: float
) -> dict[str, Any]:
    half = TWO_SIDED * std_error
    return {
        "property": "interval_calibration",
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": PRIMARY_N,
        "requested_replicates": PROPERTY_REPLICATES,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": estimate,
        "std_error": std_error,
        "covered": int(estimate - half <= truth <= estimate + half),
        "rejected": int(abs(estimate / std_error) > TWO_SIDED),
    }


def _replicate(replicate: int) -> list[dict[str, Any]]:
    seed = stream_seed(STUDY, "property_sample", "interval_calibration", "linear", replicate)
    frame, truth = draw_from_seed("linear", PRIMARY_N, seed)
    result = fit(frame)
    rows: list[dict[str, Any]] = []
    for name in FITTED:
        values = bound_standard_errors(result, name)
        for end in ENDS:
            bound = f"{name}_{end}"
            estimate, std_error = values[f"reported_{end}"]
            rows.append(
                _row(
                    f"{bound}__correctly_specified",
                    "positive",
                    replicate,
                    float(truth[bound]),
                    float(estimate),
                    float(std_error),
                )
            )
            if name in CONTROLLED:
                estimate, std_error = values[f"without_share_term_{end}"]
                rows.append(
                    _row(
                        f"{bound}__inflated_se_control",
                        "control",
                        replicate,
                        float(truth[bound]),
                        float(estimate),
                        float(std_error),
                    )
                )
    return rows


def _cells_of(key: tuple[int]) -> list[str]:
    """Every cell reads the same fit, so a key fills all of them."""
    return [f"interval_calibration/{cell}" for cell in PROPERTY_CELLS["interval_calibration"]]


def _cell_of(key: tuple[int]) -> str:
    return "interval_calibration/linear"


def _replicate_of(key: tuple[int]) -> int:
    return key[0]


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = PROPERTY_REPLICATES
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication."""
    outcomes = fit_properties(
        [(index,) for index in range(replicates)],
        splat(_replicate),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if replicates != PROPERTY_REPLICATES:
        rows["requested_replicates"] = replicates
    cells = set(rows["cell"])
    declared = set(PROPERTY_CELLS["interval_calibration"])
    if cells != declared:
        raise RuntimeError(f"cells {sorted(cells ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration verdict of each cell, each kind against its own rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    return finish(summary, rates)
