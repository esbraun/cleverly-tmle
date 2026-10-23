"""Interval calibration of each end of the marginal sensitivity model bounds (X82 SH).

One family, ``interval_calibration``, with three kinds for each end of the ATE bound:

=========================  ==================================================  ===============
kind                       row                                                 passes when
=========================  ==================================================  ===============
``correctly_specified``    the end at ``Gamma = 1.5`` against the sharp end     SE ratio and coverage inside the calibration bands
``inflated_se_control``    the same estimate, its standard error times 1.5     SE ratio above the band
``wrong_gamma_control``    the end at ``Gamma = 1.25`` against the 1.5 end      coverage below the band
=========================  ==================================================  ===============

The wrong-gamma control shows that the study can see a bound for the wrong ``Gamma``. One fit
per draw gives every row of its scenario.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed
from tests.studies.marginal_bound_coverage import (
    CELLS_FILE,
    CONTROL_GAMMA,
    GAMMA,
    PRIMARY_N,
    PROPERTY_CELLS,
    STUDY,
    bounds,
    cell_configurations,
    draw_from_seed,
    end_values,
)

#: Replications per scenario.  Every cell of a scenario reads the same draws.  6,800 is the
#: smallest multiple of 400 whose arm-pooled predictive family-wise pass probability, from the
#: pilots of ``X82-H-plan.md`` section 8, reaches 0.95 (it is 0.959).
PROPERTY_REPLICATES = 6_800

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
    result = bounds(scenario, frame)
    declared = end_values(result, GAMMA)
    control = end_values(result, CONTROL_GAMMA)
    rows: list[dict[str, Any]] = []
    for name, (estimate, se) in declared.items():
        prefix = f"{scenario}_{name}"
        target = float(truth[name])
        rows.append(
            _row(f"{prefix}__correctly_specified", "positive", replicate, target, estimate, se)
        )
        rows.append(
            _row(
                f"{prefix}__inflated_se_control",
                "control",
                replicate,
                target,
                estimate,
                INFLATED_SE_FACTOR * se,
            )
        )
        rows.append(
            _row(f"{prefix}__wrong_gamma_control", "control", replicate, target, *control[name])
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
