"""Interval calibration of the part G omitted-variable limits, and the O1 response mask.

Two families read the same draws.  One fit per draw gives every row of its scenario.

``interval_calibration`` has a positive cell for each end of each bound, read as the primary
rows read it, and an ``inflated_se_control`` for each: the same estimate with its standard
error times 1.5.  That control must rise above the SE-ratio band (``X82-SG-plan.md`` rev 2,
section 5, note R2).  It over-covers, so it cannot fail the coverage band.

``response_mask_necessity`` pairs each O1 bound end against a control that reads ``sigma^2``
as the mean squared residual of the observed rows, not as the masked mean over all rows
(``E[Delta (Y - Qbar)^2]``).  The control's bound misses its truth, and the family requires a
displacement of at least 0.25 positive-arm empirical SDs.  The pilot measured about 1.5
(section 5.1).

The MSM Gram term has no control here.  Removing it moves the limit SE by at most 4%, up to
``n = 40,000``, so no SE-ratio reading can see it (section 5.1).  The Gateaux unit test of the
MSM bound is its witness.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from cleverly.sensitivity import omitted_variable as omitted_variable_module
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed
from tests.studies.ovb_limit_coverage import (
    CELLS,
    CELLS_FILE,
    ENDS,
    PRIMARY_N,
    PROPERTY_CELLS,
    SCENARIOS,
    STUDY,
    TWO_SIDED,
    bound_values,
    cell_configurations,
    draw_from_seed,
    fit,
    parameter_result,
)

#: Replications per scenario.  Every cell of a scenario reads the same draws.
PROPERTY_REPLICATES = 4_000

#: The standard-error factor of the ``inflated_se_control`` arm.
INFLATED_SE_FACTOR = 1.5

#: The declared minimum displacement of the O1 control, in positive-arm empirical SDs.
RESPONSE_MASK_DISPLACEMENT = 0.25

#: The ``response_mask_necessity`` family's declaration, read by its verdict and its rule.
RESPONSE_MASK_NECESSITY = property_verdicts.Necessity(
    family="response_mask_necessity",
    labels=tuple(f"O1_{row}" for row in SCENARIOS["O1"]),
    arms=("masked", "unmasked_sigma2"),
    column="response_mask_displacement",
    threshold=RESPONSE_MASK_DISPLACEMENT,
)


def unmasked_bounds(fitted: Any, label: str, frame: pd.DataFrame) -> dict[str, float]:
    """The O1 bound ends with ``sigma^2`` read over the observed rows only.

    The package's ``sigma^2`` is the masked mean ``E[Delta (Y - Qbar)^2]``.  Dividing it by
    the observed share gives the observed rows' mean, which is the control's defect.
    """
    spec = CELLS["O1"]
    result, name = parameter_result("O1", fitted, label)
    elements = omitted_variable_module.sensitivity_elements(result, name)
    share = float((frame["D"] == 1.0).mean())
    half = spec.strength * math.sqrt(elements.sigma2 / share * elements.nu2)
    psi = float(result[name].psi)
    return {"lower": psi - half, "upper": psi + half}


def _row(
    family: str, cell: str, role: str, replicate: int, truth: float, estimate: float, se: float
) -> dict[str, Any]:
    return control_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=PRIMARY_N,
        requested=PROPERTY_REPLICATES,
        truth=float(truth),
        estimate=float(estimate),
        standard_error=float(se),
        critical=TWO_SIDED,
    )


def _draw(scenario: str, replicate: int) -> Any:
    seed = stream_seed(STUDY, "property_sample", "interval_calibration", scenario, replicate)
    return draw_from_seed(scenario, PRIMARY_N, seed)


def _replicate(scenario: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    frame, truth = drawn
    fitted = fit(scenario, frame)
    rows: list[dict[str, Any]] = []
    for label in CELLS[scenario].parameters:
        values = bound_values(scenario, fitted, label)
        unmasked = unmasked_bounds(fitted, label, frame) if scenario == "O1" else None
        for end in ENDS:
            row = f"{scenario}_{label}_{end}"
            target = truth[f"{label}_{end}"]
            estimate, se = values[end]
            rows.append(
                _row(
                    "interval_calibration",
                    f"{row}__correctly_specified",
                    "positive",
                    replicate,
                    target,
                    estimate,
                    se,
                )
            )
            rows.append(
                _row(
                    "interval_calibration",
                    f"{row}__inflated_se_control",
                    "control",
                    replicate,
                    target,
                    estimate,
                    INFLATED_SE_FACTOR * se,
                )
            )
            if unmasked is not None:
                family = RESPONSE_MASK_NECESSITY.family
                rows.append(
                    _row(family, f"{row}__masked", "positive", replicate, target, estimate, se)
                )
                rows.append(
                    _row(
                        family,
                        f"{row}__unmasked_sigma2",
                        "control",
                        replicate,
                        target,
                        unmasked[end],
                        se,
                    )
                )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    """Every cell of the key's scenario reads the key's fit."""
    prefix = f"{key[0]}_"
    return [
        f"{family}/{cell}"
        for family, cells in PROPERTY_CELLS.items()
        for cell in cells
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
    declared = {cell for family in PROPERTY_CELLS.values() for cell in family}
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
    """The verdict of each cell, each kind against its own rule."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), extra_columns=(RESPONSE_MASK_NECESSITY.column,)
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    necessity_verdicts(summary, rows, RESPONSE_MASK_NECESSITY)
    return finish(summary, rates)


#: The failure accounting's declared rules.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(RESPONSE_MASK_NECESSITY),
)
