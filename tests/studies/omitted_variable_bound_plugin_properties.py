"""Interval calibration of the plug-in omitted-variable bound's standard error (X26).

One family, ``interval_calibration``.  Each scenario has a positive cell for the lower and the
upper bound of each parameter: 8 scenarios x 3 parameters x 2 rows = 48 cells.  The corrected
``nu2`` rows are measured, not claimed, so they have no cell (``extra_artifacts`` of
``tests.studies.omitted_variable_bound_plugin`` publishes them unjudged).  A fit gives every
row of one parameter of one draw, so a control and its positive cell differ only in the curve
their standard error comes from.

A replicate is one parameter of one draw.  The three parameters of a scenario read one sample
(the draw is keyed on the scenario and the replicate) and fit it three times, because a fit
costs a few hundredths of a second against the package's bound at 0.05 to 1 second.  A refusal
of one parameter fails that parameter's cells for that draw and no other parameter's.

The controls are on the ATT and the ATC.  At a correct model the ATE's mechanism term
reflects its second-moment term and keeps the variance (plan section 3.4), so an ATE control
could not fail.

* ``inflated_se_control`` (28 cells, the seven estimated-mechanism scenarios): the package
  curve minus the mechanism term this study rebuilt, with the package's variance correction.
  The predicted ratios are about 1.17 to 1.3 for the bounds, so the interval must rise above
  the band.
* ``shrunken_se_control`` (4 cells, ``binary_known``): the package curve plus the mechanism
  term of a study-side logistic fit.  That is the curve of the estimated mechanism, which is
  narrower than the known mechanism's (predicted ratio 0.85), so the interval must fall below
  the band.

The replicate count is 10,000 per cell.  The design test
(``tests/unit/test_omitted_variable_plugin_design.py``) gives the family-wise pass probability
under perfect calibration and the smallest multiple of 400 that reaches 0.95.
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
    CLAIM_ROWS,
    CONTROLLED,
    PARAMETERS,
    PRIMARY_N,
    PROPERTY_CELLS,
    STUDY,
    TWO_SIDED,
    cell_label,
    control_kind,
    draw_from_seed,
    estimand_key,
    fit,
    parameter_rows,
)

#: Replications per property cell.  The three parameters of a scenario read the same draws.
#: The design test checks that this reaches a family-wise pass probability of 0.95 under
#: perfect calibration, and that it is at least the smallest multiple of 400 that does.
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


def _draw(scenario: str, token: str, replicate: int) -> Any:
    """The draw of a scenario, shared by its parameters: ``token`` does not enter the seed."""
    del token
    seed = stream_seed(STUDY, "property_sample", "interval_calibration", scenario, replicate)
    return draw_from_seed(scenario, PRIMARY_N, seed)


def _replicate(scenario: str, token: str, replicate: int, drawn: Any) -> list[dict[str, Any]]:
    frame, truth = drawn
    result = fit(frame, scenario)
    values = parameter_rows(result, scenario, token)
    rows: list[dict[str, Any]] = []
    for row in CLAIM_ROWS:
        label = cell_label(scenario, token, row)
        reference = float(truth[estimand_key(token, row)])
        bound = values.bounds[row]
        rows.append(
            _row(
                f"{label}__correctly_specified",
                "positive",
                replicate,
                reference,
                bound.estimate,
                bound.std_error,
            )
        )
        if token in CONTROLLED:
            assert bound.control is not None
            rows.append(
                _row(
                    f"{label}__{control_kind(scenario)}",
                    "control",
                    replicate,
                    reference,
                    bound.estimate,
                    bound.control,
                )
            )
    return rows


def parameter_cells(scenario: str, token: str) -> list[str]:
    """The cells one parameter of one scenario fills."""
    labels = {cell_label(scenario, token, row) for row in CLAIM_ROWS}
    return [
        cell for cell in PROPERTY_CELLS["interval_calibration"] if cell.split("__")[0] in labels
    ]


def scenario_cells(scenario: str) -> list[str]:
    """The cells one scenario's fits fill."""
    return [cell for token in PARAMETERS[scenario] for cell in parameter_cells(scenario, token)]


def _cells_of(key: tuple[str, str, int]) -> list[str]:
    """Every cell of a parameter reads the same fit, so a key fills all of them."""
    return [f"interval_calibration/{cell}" for cell in parameter_cells(key[0], key[1])]


def _cell_of(key: tuple[str, str, int]) -> str:
    return f"interval_calibration/{key[0]}_{key[1]}"


def _replicate_of(key: tuple[str, str, int]) -> int:
    return key[2]


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
        [
            (scenario, token, index)
            for scenario in PARAMETERS
            for token in PARAMETERS[scenario]
            for index in range(replicates)
        ],
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
