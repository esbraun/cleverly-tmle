"""Interval calibration of the plug-in omitted-variable bound's standard error (X26).

One family, ``interval_calibration``.  Each scenario has a positive cell for each row
(``nu2``, ``lower``, ``upper``) of each parameter: 5 scenarios x 3 parameters x 3 rows = 45
cells.  Every cell of a scenario reads the same draws, and one fit per draw gives all of its
rows, so a control and its positive cell differ only in the curve their standard error
comes from.

The controls are on the ATT and the ATC.  At a correct model the ATE's mechanism term
reflects its second-moment term and keeps the variance (plan section 3.4), so an ATE control
could not fail.

* ``inflated_se_control`` (24 cells, the four estimated-mechanism scenarios): the package
  curve minus the mechanism term this study rebuilt.  The predicted ratios are about 2.6 for
  ``nu2`` and 1.17 to 1.24 for the bounds, so the interval must rise above the band.
* ``shrunken_se_control`` (6 cells, ``binary_known``): the package curve plus the mechanism
  term of a study-side logistic fit.  That is the curve of the estimated mechanism, which is
  narrower than the known mechanism's (predicted ratios 0.38 and 0.85), so the interval must
  fall below the band.

The replicate count is the smallest multiple of 400 whose family-wise pass probability under
perfect calibration reaches 0.95 (``tests/unit/test_omitted_variable_plugin_design.py``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed
from tests.studies.omitted_variable_bound_plugin import (
    CONTROLLED,
    PARAMETERS,
    PRIMARY_N,
    PROPERTY_CELLS,
    ROWS,
    STUDY,
    TWO_SIDED,
    cell_label,
    control_kind,
    draw_from_seed,
    estimand_key,
    fit,
    parameter_rows,
)

#: Replications per property cell.  Every cell of a scenario reads the same draws.  The design
#: test pins this as the smallest multiple of 400 that reaches a family-wise pass probability
#: of 0.95 under perfect calibration.
PROPERTY_REPLICATES = 10_000


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


def _draw(scenario: str, replicate: int) -> Any:
    seed = stream_seed(STUDY, "property_sample", "interval_calibration", scenario, replicate)
    return draw_from_seed(scenario, PRIMARY_N, seed)


def _replicate(scenario: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    frame, truth = drawn
    result = fit(frame, scenario)
    rows: list[dict[str, Any]] = []
    for token in PARAMETERS[scenario]:
        values = parameter_rows(result, scenario, token)
        for row in ROWS:
            label = cell_label(scenario, token, row)
            reference = float(truth[estimand_key(token, row)])
            estimate, std_error, control = values[row]
            rows.append(
                _row(
                    f"{label}__correctly_specified",
                    "positive",
                    replicate,
                    reference,
                    estimate,
                    std_error,
                )
            )
            if token in CONTROLLED:
                assert control is not None
                rows.append(
                    _row(
                        f"{label}__{control_kind(scenario)}",
                        "control",
                        replicate,
                        reference,
                        estimate,
                        control,
                    )
                )
    return rows


def scenario_cells(scenario: str) -> list[str]:
    """The cells one scenario's fits fill."""
    labels = {cell_label(scenario, token, row) for token in PARAMETERS[scenario] for row in ROWS}
    return [
        cell for cell in PROPERTY_CELLS["interval_calibration"] if cell.split("__")[0] in labels
    ]


def _cells_of(key: tuple[str, int]) -> list[str]:
    """Every cell of a scenario reads the same fit, so a key fills all of them."""
    return [f"interval_calibration/{cell}" for cell in scenario_cells(key[0])]


def _cell_of(key: tuple[str, int]) -> str:
    return f"interval_calibration/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


#: The share of a cell's requested replications that the accounting lets a run lose (RM39b2
#: rev 3 section 6).  Up to ``floor(0.01 R)`` lost replicates are judged by the worst case of
#: each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = PROPERTY_REPLICATES
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication."""
    outcomes = fit_properties(
        [(scenario, index) for scenario in PARAMETERS for index in range(replicates)],
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


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration verdict of each cell, each kind against its own rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    return finish(summary, rates)
