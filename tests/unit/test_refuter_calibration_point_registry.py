"""The registry rows that the point-treatment refutation study licenses (X82 SP).

Each row's scope must lie inside passing size and power cells of the committed run, read from
``cells.csv`` and ``properties.csv``. A row licenses a verdict on a fit inside that scope and
leaves a fit outside it descriptive.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from cleverly.validation import refute
from cleverly.validation._interpretation import (
    _POINT_CELLS,
    VERDICT_EVIDENCE,
    CalibrationCell,
    FitConfiguration,
    VerdictEvidence,
    check_scope_containment,
    method_axis,
)
from tests.studies import refuter_calibration_point as study
from tests.studies import refuter_calibration_point_properties as properties

ARTIFACTS = study.STUDY.artifacts

#: The refit count each instrument resolves its reading at. ``refute`` resolves a
#: negative-control reading at one refit.
REFITS = {"placebo": 20, "subset": 20, "negative_control": 1}

INSTRUMENTS = {
    "placebo": "placebo",
    "subset": "subset",
    "negative_control": "negative_control_outcome",
}


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.study[:1] == (study.STUDY.slug,)]


@functools.cache
def _control_events(name: str) -> int:
    """The smallest arm event count of the binary negative control over the cell's trials.

    ``refute`` scopes a negative-control reading by the control's own event count, which
    ``cells.csv`` does not record. The draws are deterministic, so the count is recomputed.
    """
    counts = []
    for trial in range(study.SIZE_TRIALS):
        frame = properties._draw(name, trial)
        for arm in (0.0, 1.0):
            control = frame.loc[frame["A"] == arm, "NC"].to_numpy()
            events = int(np.count_nonzero(control > 0.5))
            counts += [events, control.size - events]
    return min(counts)


def _method(name: str, recorded: str) -> str:
    """The method axis of a cell. The run recorded a ``CTMLE`` by its class name, before the
    axis named the strategy; the declared estimator gives the variant it ran."""
    axis = method_axis(study.estimator(study.CELLS[name], seed=0))
    assert axis.split("[")[0] == recorded
    return axis


def _cells() -> dict[str, CalibrationCell]:
    """Every property cell of the run, keyed ``family/cell``, with its recorded configuration."""
    configurations = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False).set_index("cell")
    table = pd.read_csv(ARTIFACTS / "properties.csv")
    cells: dict[str, CalibrationCell] = {}
    for row in table.itertuples():
        name, reading = str(row.cell).split("__", 1)
        recorded = configurations.loc[name]
        instrument = next(key for key in REFITS if reading.startswith(key))
        events = recorded["min_event_count_min"]
        if events == "":
            count = None
        elif instrument == "negative_control":
            count = _control_events(name)
        else:
            count = int(events)
        configuration = FitConfiguration(
            method=_method(name, str(recorded["method"])),
            estimand_kind=str(recorded["estimand_kind"]),
            outcome_family=str(recorded["outcome_family"]),
            treatment_kind=str(recorded["treatment_kind"]),
            learners=tuple(str(recorded["learners"]).split(" | ")),
            cross_fitting=str(recorded["cross_fitting"]),
            inference=str(recorded["inference"]),
            known_mechanism=str(recorded["known_mechanism"]) == "True",
            n=int(recorded["n"]),
            covariates_per_row=float(recorded["covariates_per_row"]),
            min_event_count=count,
            clusters=None,
            arms=int(recorded["arms"]),
            replicates=REFITS[instrument],
            nodes=None,
        )
        kind: Literal["size", "power"] = "size" if row.property == "refuter_size" else "power"
        cells[f"{row.property}/{row.cell}"] = CalibrationCell(
            kind=kind, passed=bool(row.property_passed), configuration=configuration
        )
    return cells


def test_the_study_registers_one_row_per_size_reading() -> None:
    expected = {
        (INSTRUMENTS[reading], name)
        for name, cell in study.CELLS.items()
        for reading in cell.readings
    }
    found = {(row.instrument, row.study[1].split("/")[1].split("__")[0]) for row in _rows()}
    assert found == expected
    assert len(_rows()) == 12


def test_the_negative_control_floors_are_the_recomputed_counts() -> None:
    for name, values in _POINT_CELLS.items():
        control_events = values[-1]
        if "negative_control" in study.CELLS[name].readings and values[1] == "binomial":
            assert control_events == _control_events(name)


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.study[1]}")
def test_each_row_lies_inside_its_passing_cells(row: VerdictEvidence) -> None:
    assert check_scope_containment(row, _cells()) == []


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.study[1]}")
def test_each_row_names_cells_the_run_recorded(row: VerdictEvidence) -> None:
    cells = _cells()
    assert all(cell in cells for cell in row.study[1:])
    assert {cells[cell].kind for cell in row.study[1:]} == {"size", "power"}


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
    # A row that claimed n = 1,000, another C-TMLE strategy or a lower event floor would
    # reach fits no cell ran.
    row = _rows()[0]
    assert check_scope_containment(replace(row, scope=replace(row.scope, n_min=1000)), _cells())
    greedy = next(row for row in _rows() if row.scope.methods == frozenset({"CTMLE[greedy]"}))
    oat = replace(greedy, scope=replace(greedy.scope, methods=frozenset({"CTMLE[oat:shared]"})))
    assert check_scope_containment(oat, _cells())
    control = next(
        row
        for row in _rows()
        if row.instrument == "negative_control_outcome" and row.scope.min_event_count_min
    )
    assert control.scope.min_event_count_min is not None
    lower = replace(
        control,
        scope=replace(control.scope, min_event_count_min=control.scope.min_event_count_min - 1),
    )
    assert check_scope_containment(lower, _cells())


def _t2_placebo(n: int) -> bool | None:
    frame = study.draw_law("binomial", n, seed=7)
    result = study.fit_cell("T2", frame, seed=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(result, tests=("placebo",), n_replicates=20, random_state=11)
    return report["placebo"].passed


def test_a_covered_fit_gets_a_verdict_and_a_smaller_one_does_not() -> None:
    # T2's configuration at the calibrated n prints a verdict. Below n_min it is descriptive.
    assert _t2_placebo(2000) is not None
    assert _t2_placebo(1500) is None
