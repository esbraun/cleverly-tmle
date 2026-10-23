"""The registry rows that the band-rank placebo study licenses (X82 SJ).

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

from cleverly.validation import refute
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    CalibrationCell,
    FitConfiguration,
    VerdictEvidence,
    check_scope_containment,
)
from tests.studies import refuter_calibration_band as study

ARTIFACTS = study.STUDY.artifacts


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.study[:1] == (study.STUDY.slug,)]


def _cells() -> dict[str, CalibrationCell]:
    """Every property cell of the run, keyed ``family/cell``, with its recorded configuration."""
    configurations = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False).set_index("cell")
    table = pd.read_csv(ARTIFACTS / "properties.csv")
    cells: dict[str, CalibrationCell] = {}
    for row in table.itertuples():
        name = str(row.cell).split("__", 1)[0]
        recorded = configurations.loc[name]
        events = recorded["min_event_count_min"]
        nodes = recorded["nodes"]
        configuration = FitConfiguration(
            method=str(recorded["method"]),
            estimand_kind=str(recorded["estimand_kind"]),
            outcome_family=str(recorded["outcome_family"]),
            treatment_kind=str(recorded["treatment_kind"]),
            learners=tuple(str(recorded["learners"]).split(" | ")),
            cross_fitting=str(recorded["cross_fitting"]),
            inference=str(recorded["inference"]),
            known_mechanism=str(recorded["known_mechanism"]) == "True",
            n=int(recorded["n"]),
            covariates_per_row=float(recorded["covariates_per_row"]),
            min_event_count=None if events == "" else int(events),
            clusters=None,
            arms=int(recorded["arms"]),
            replicates=int(recorded["replicates"]),
            nodes=None if nodes == "" else int(float(nodes)),
            placebo_draw=str(recorded["placebo_draw"]),
        )
        kind: Literal["size", "power"] = "size" if row.property == "refuter_size" else "power"
        cells[f"{row.property}/{row.cell}"] = CalibrationCell(
            kind=kind, passed=bool(row.property_passed), configuration=configuration
        )
    return cells


def test_the_study_registers_one_placebo_row_per_cell() -> None:
    found = {(row.instrument, row.study[1].split("/")[1].split("__")[0]) for row in _rows()}
    assert found == {("placebo", name) for name in study.CELLS}
    assert len(_rows()) == 2


def test_each_cell_recorded_its_declared_draw() -> None:
    configurations = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False).set_index("cell")
    for name, cell in study.CELLS.items():
        assert configurations.loc[name, "placebo_draw"] == cell.placebo_draw


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.study[1])
def test_each_row_lies_inside_its_passing_cells(row: VerdictEvidence) -> None:
    assert check_scope_containment(row, _cells()) == []


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.study[1])
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


@pytest.mark.parametrize(
    "axis, widened",
    [
        ("placebo_draws", frozenset({"band_permute", "band_markov", "permute"})),
        ("nodes", frozenset({None, 2, 3})),
        ("treatment_kinds", frozenset({"continuous", "discrete"})),
    ],
)
def test_a_wider_row_fails_containment(axis: str, widened: frozenset[object]) -> None:
    # A row that claimed the unbanded draw, another node count or a discrete treatment would
    # reach fits no cell ran.
    for row in _rows():
        assert check_scope_containment(
            replace(row, scope=replace(row.scope, **{axis: widened})), _cells()
        )
    for row in _rows():
        smaller = replace(row, scope=replace(row.scope, n_min=1000))
        assert check_scope_containment(smaller, _cells())


def _pj1_placebo(n: int) -> bool | None:
    frame = study.draw_point(n, seed=7)
    result = study.fit_cell("PJ1", frame, seed=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(
            result,
            estimand=study.CELLS["PJ1"].estimand,
            tests=("placebo",),
            n_replicates=study.REFITS,
            random_state=11,
        )
    return report["placebo"].passed


def test_a_covered_fit_gets_a_verdict_and_a_smaller_one_does_not() -> None:
    # PJ1's configuration at the calibrated n prints a verdict. Below n_min it is descriptive.
    assert _pj1_placebo(2000) is not None
    assert _pj1_placebo(1500) is None
