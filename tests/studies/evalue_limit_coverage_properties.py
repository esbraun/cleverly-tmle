"""Interval calibration of the risk-ratio intervals that the part G E-values map.

One family, ``interval_calibration``, with a positive cell for each risk ratio and an
``inflated_se_control`` for each: the same log-scale estimate with its standard error times
1.5.  That control must rise above the SE-ratio band (``X82-SG-plan.md`` rev 2, section 5).
One fit per draw gives every row of its scenario.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evalue_limit_coverage import (
    CELLS_FILE,
    PRIMARY_N,
    PROPERTY_CELLS,
    STUDY,
    cell_configurations,
    draw_from_seed,
    estimates,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

#: Replications per scenario.  Every cell of a scenario reads the same draws.
PROPERTY_REPLICATES = 4_000

#: The standard-error factor of the ``inflated_se_control`` arm.
INFLATED_SE_FACTOR = 1.5

#: The two-sided critical value of the calibration interval.
CRITICAL = 1.959963984540054


def _row(
    cell: str, role: str, replicate: int, truth: float, estimate: float, se: float
) -> dict[str, Any]:
    return control_row(
        property_name="interval_calibration",
        cell=cell,
        role=role,
        replicate=replicate,
        n=PRIMARY_N,
        requested=PROPERTY_REPLICATES,
        truth=truth,
        estimate=estimate,
        standard_error=se,
        critical=CRITICAL,
    )


def _draw(scenario: str, replicate: int) -> Any:
    seed = stream_seed(STUDY, "property_sample", "interval_calibration", scenario, replicate)
    return draw_from_seed(scenario, PRIMARY_N, seed)


def _replicate(scenario: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    frame, truth = drawn
    rows: list[dict[str, Any]] = []
    for name, estimate in estimates(scenario, frame).items():
        if estimate.log_psi is None:  # pragma: no cover - estimator contract guard
            raise AssertionError(f"{scenario} {name}: a risk ratio needs a log-scale estimate")
        std_error, _, _ = reported_inference(estimate)
        target = math.log(float(truth[name]))
        log_estimate = float(estimate.log_psi)
        row = f"{scenario}_{name}"
        rows.append(
            _row(
                f"{row}__correctly_specified",
                "positive",
                replicate,
                target,
                log_estimate,
                std_error,
            )
        )
        rows.append(
            _row(
                f"{row}__inflated_se_control",
                "control",
                replicate,
                target,
                log_estimate,
                INFLATED_SE_FACTOR * std_error,
            )
        )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    """Every cell of the key's scenario reads the key's fit."""
    prefix = f"{key[0]}_"
    return [
        f"interval_calibration/{cell}"
        for cell in PROPERTY_CELLS["interval_calibration"]
        if cell.startswith(prefix)
    ]


def _cell_of(key: tuple[str, int]) -> str:
    return f"interval_calibration/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = PROPERTY_REPLICATES
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication."""
    outcomes = fit_properties(
        [(scenario, index) for scenario in STUDY.scenarios for index in range(replicates)],
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
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


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, replicates: int = PROPERTY_REPLICATES
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every property replication, and record each cell's fit configuration.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    replicates : int
        Replications per scenario.

    Returns
    -------
    tuple
        The property rows, and the configuration table by file name.
    """
    rows = generate_property_rows(n_jobs=n_jobs, replicates=replicates)
    return rows, {CELLS_FILE: cell_configurations()}


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration verdict of each cell, each kind against its own rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    return finish(summary, rates)


#: The failure accounting's declared rules.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)
