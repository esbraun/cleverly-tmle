"""The registry rows that the longitudinal refutation study licenses (X82 SL).

Each row's scope must lie inside passing size and power cells of the committed run, read from
``cells.csv`` and ``properties.csv``. A row licenses a verdict on a fit inside that scope and
leaves a fit outside it descriptive.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

import pandas as pd
import pytest

from cleverly.learners import GLM
from cleverly.longitudinal import SequentialIPW
from cleverly.validation import refute
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    CalibrationCell,
    FitConfiguration,
    VerdictEvidence,
    check_scope_containment,
)
from tests.studies.refuter_calibration_longitudinal import (
    CELLS,
    COLUMNS,
    REFERENCE,
    REGIMENS,
    STUDY,
    draw_end_of_study,
)

ARTIFACTS = STUDY.artifacts

#: The refit count each instrument resolves its reading at. ``refute`` resolves a
#: negative-control reading at one refit, so its cells carry that count.
REFITS = {"placebo": 20, "subset": 20, "negative_control": 1}


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.study[:1] == (STUDY.slug,)]


def _cells() -> dict[str, CalibrationCell]:
    """Every property cell of the run, keyed ``family/cell``, with its recorded configuration."""
    configurations = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False).set_index("cell")
    properties = pd.read_csv(ARTIFACTS / "properties.csv")
    cells: dict[str, CalibrationCell] = {}
    for row in properties.itertuples():
        name, reading = str(row.cell).split("__", 1)
        recorded = configurations.loc[name]
        instrument = next(key for key in REFITS if reading.startswith(key))
        events = recorded["min_event_count_min"]
        configuration = FitConfiguration(
            method=str(recorded["method"]),
            estimand_kind=str(recorded["estimand_kind"]),
            # The control of law E is Gaussian, and ``refute`` scopes a control reading by it.
            outcome_family=(
                "gaussian" if instrument == "negative_control" else str(recorded["outcome_family"])
            ),
            treatment_kind=str(recorded["treatment_kind"]),
            learners=tuple(str(recorded["learners"]).split(" | ")),
            cross_fitting=str(recorded["cross_fitting"]),
            inference=str(recorded["inference"]),
            known_mechanism=str(recorded["known_mechanism"]) == "True",
            n=int(recorded["n"]),
            covariates_per_row=float(recorded["covariates_per_row"]),
            min_event_count=(
                None if instrument == "negative_control" or events == "" else int(events)
            ),
            clusters=None,
            arms=int(recorded["arms"]),
            replicates=REFITS[instrument],
            nodes=int(recorded["nodes"]),
        )
        kind: Literal["size", "power"] = "size" if row.property == "refuter_size" else "power"
        cells[f"{row.property}/{row.cell}"] = CalibrationCell(
            kind=kind, passed=bool(row.property_passed), configuration=configuration
        )
    return cells


def test_the_study_registers_one_row_per_size_reading() -> None:
    expected = {
        (instrument, cell)
        for cell, declared in CELLS.items()
        for reading in declared.readings
        for instrument in [
            {
                "placebo": "placebo",
                "subset": "subset",
                "negative_control": "negative_control_outcome",
            }[reading]
        ]
    }
    found = {(row.instrument, row.study[1].split("/")[1].split("__")[0]) for row in _rows()}
    assert found == expected
    assert len(_rows()) == 17


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.study[1]}")
def test_each_row_lies_inside_its_passing_cells(row: VerdictEvidence) -> None:
    assert check_scope_containment(row, _cells()) == []


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.study[1]}")
def test_each_row_names_cells_the_run_recorded(row: VerdictEvidence) -> None:
    cells = _cells()
    assert all(cell in cells for cell in row.study[1:])
    kinds = {cells[cell].kind for cell in row.study[1:]}
    assert kinds == {"size", "power"}


def test_a_failed_size_cell_would_withdraw_the_row() -> None:
    # Deliberate-mutation control: the containment check reads the recorded verdicts.
    row = _rows()[0]
    cells = _cells()
    size = row.study[1]
    cells[size] = replace(cells[size], passed=False)
    assert check_scope_containment(row, cells) == [
        f"{row.instrument}/{row.rule}: no passing size cell"
    ]


def test_a_wider_row_fails_containment() -> None:
    # A row that claimed n = 500 or a second learner would reach fits no cell ran.
    row = _rows()[0]
    wider = replace(row, scope=replace(row.scope, n_min=500))
    assert check_scope_containment(wider, _cells())
    assert row.scope.learners is not None
    learners = replace(row, scope=replace(row.scope, learners=row.scope.learners | {"Lasso()"}))
    assert check_scope_containment(learners, _cells())


def _ipw_placebo(n: int) -> bool | None:
    frame = draw_end_of_study(n, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SequentialIPW(
            REGIMENS, reference=REFERENCE, treatment_learner=GLM(), simultaneous=False
        ).fit(frame, **COLUMNS["end_of_study"])
        report = refute(
            result,
            estimand="ate_regimen[always vs never]",
            tests=("placebo",),
            n_replicates=20,
            random_state=11,
        )
    return report["placebo"].passed


def test_a_covered_fit_gets_a_verdict_and_a_smaller_one_does_not() -> None:
    # B2's configuration at the calibrated n prints a verdict. Below n_min it is descriptive.
    assert _ipw_placebo(1000) is not None
    assert _ipw_placebo(600) is None
