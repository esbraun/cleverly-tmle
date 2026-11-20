"""The registry rows that the continuous-time survival refutation study licenses (X101).

Each row's scope must lie inside passing size and power cells of the committed run, read from
``cells.csv`` and ``properties.csv``. A row licenses a verdict on a fit inside that scope and
leaves a fit outside it descriptive. The run fitted the default switched variance, so a fit with
the influence-curve variance stays descriptive.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    CalibrationCell,
    FitConfiguration,
    VerdictEvidence,
    check_scope_containment,
    continuous_survival_configuration,
    resolve,
)
from tests.studies import refuter_calibration_continuous_survival as study

ARTIFACTS = study.STUDY.artifacts
CELLS = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False)


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.study[:1] == (study.STUDY.slug,)]


def _row(instrument: str) -> VerdictEvidence:
    return next(row for row in _rows() if row.instrument == instrument)


def _configuration(recorded: pd.Series, *, variance: str = "switched") -> FitConfiguration:
    """A cell's recorded configuration. The run fitted the default variance (design test)."""
    return FitConfiguration(
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
        min_event_count=int(recorded["min_event_count_min"]),
        clusters=None,
        arms=int(recorded["arms"]),
        replicates=int(recorded["replicates"]),
        nodes=None,
        event_share=float(recorded["event_share_min"]),
        causes=int(recorded["causes"]),
        variance=variance,
    )


def _cells(instrument: str) -> dict[str, CalibrationCell]:
    """Every property cell of the run, keyed ``family/cell``, with its recorded configuration."""
    table = pd.read_csv(ARTIFACTS / "properties.csv")
    cells: dict[str, CalibrationCell] = {}
    for row in table.itertuples():
        key = str(row.cell).split("__", 1)[0]
        law = "size" if row.property == "refuter_size" else "power"
        recorded = CELLS.loc[(CELLS["cell"] == key) & (CELLS["law"] == law)].iloc[0]
        configuration = _configuration(recorded)
        if instrument == "placebo":
            configuration = replace(configuration, placebo_draw=study.PLACEBO_DRAW)
        kind: Literal["size", "power"] = "size" if row.property == "refuter_size" else "power"
        cells[f"{row.property}/{row.cell}"] = CalibrationCell(
            kind=kind, passed=bool(row.property_passed), configuration=configuration
        )
    return cells


def test_the_study_registers_one_placebo_and_one_subset_row() -> None:
    assert sorted((row.instrument, row.rule) for row in _rows()) == [
        ("placebo", "t"),
        ("subset", "rho"),
    ]
    for row in _rows():
        assert row.rung == "L2" and not row.exact


@pytest.mark.parametrize("instrument", ["placebo", "subset"])
def test_each_row_names_every_estimand_cell_of_its_rule(instrument: str) -> None:
    row = _row(instrument)
    size = {f"refuter_size/{key}__{instrument}" for key in study.ESTIMANDS}
    defects = ("placebo_stale_rows",) if instrument == "placebo" else study.SUBSET_DEFECTS
    power = {f"refuter_power/{key}__{d}" for key in study.ESTIMANDS for d in defects}
    assert set(row.study[1:]) == size | power
    assert all(cell in _cells(instrument) for cell in row.study[1:])


@pytest.mark.parametrize("instrument", ["placebo", "subset"])
def test_each_row_lies_inside_its_passing_cells(instrument: str) -> None:
    assert check_scope_containment(_row(instrument), _cells(instrument)) == []


@pytest.mark.parametrize("instrument", ["placebo", "subset"])
def test_failed_cells_would_withdraw_the_row(instrument: str) -> None:
    # Deliberate-mutation control: every size cell, then every power cell, turned red.
    row = _row(instrument)
    for kind in ("size", "power"):
        cells = _cells(instrument)
        for name in row.study[1:]:
            if cells[name].kind == kind:
                cells[name] = replace(cells[name], passed=False)
        assert check_scope_containment(row, cells) == [
            f"{row.instrument}/{row.rule}: no passing {kind} cell"
        ]


@pytest.mark.parametrize("instrument", ["placebo", "subset"])
def test_the_scalar_axes_match_the_size_cells(instrument: str) -> None:
    scope = _row(instrument).scope
    size = CELLS.loc[CELLS["law"] == "size"]
    assert scope.min_event_count_min == int(size["min_event_count_min"].min())
    assert scope.covariates_per_row_max == float(size["covariates_per_row"].max())
    assert scope.n_min == int(size["n"].min()) == study.N
    assert scope.causes == frozenset(int(c) for c in size["causes"])
    assert scope.event_share_range is not None
    low, high = scope.event_share_range
    # The covered range lies inside the size trials' measured shares, rounded inward.
    assert float(size["event_share_min"].min()) <= low < float(size["event_share_min"].min()) + 1e-4
    assert (
        float(size["event_share_max"].max()) - 1e-4 < high <= float(size["event_share_max"].max())
    )
    assert scope.variances == frozenset({study.estimator(0).variance}) == frozenset({"switched"})


@pytest.mark.parametrize(
    "axis, widened",
    [
        ("min_event_count_min", 16),
        ("covariates_per_row_max", 0.008),
        ("n_min", 500),
        ("estimand_kinds", frozenset({"continuous:ate_regimen", "continuous:rr_regimen"})),
    ],
)
def test_a_wider_row_fails_containment(axis: str, widened: Any) -> None:
    for instrument in ("placebo", "subset"):
        row = _row(instrument)
        wider = replace(row, scope=replace(row.scope, **{axis: widened}))
        assert check_scope_containment(wider, _cells(instrument))


def test_only_the_placebo_row_restricts_the_draw() -> None:
    assert _row("placebo").scope.placebo_draws == frozenset({"permute"})
    assert _row("subset").scope.placebo_draws is None


@pytest.mark.parametrize("instrument, rule", [("placebo", "t"), ("subset", "rho")])
def test_the_recorded_configuration_is_licensed_and_its_neighbours_are_not(
    instrument: str, rule: str
) -> None:
    # The size cell's own configuration resolves to the row. Each changed axis leaves it
    # descriptive: the influence-curve variance (design review M2), one cause, an event share
    # outside the measured range, and too few events.
    recorded = CELLS.loc[(CELLS["cell"] == "c1_t1.12") & (CELLS["law"] == "size")].iloc[0]
    # The cell records its smallest trial share, which sits just below the inward-rounded range.
    fit = replace(_configuration(recorded), event_share=0.5)
    if instrument == "placebo":
        fit = replace(fit, placebo_draw="permute")
    assert resolve(instrument, rule, fit).licensed
    outside = {
        "variance": replace(fit, variance="influence_curve"),
        "causes": replace(fit, causes=1),
        "share": replace(fit, event_share=0.30),
        "events": replace(fit, min_event_count=16),
    }
    for name, other in outside.items():
        assert not resolve(instrument, rule, other).licensed, name
    sentence = resolve(instrument, rule, outside["variance"]).sentence
    assert "variance rule 'influence_curve'" in sentence


def test_the_subset_row_states_its_measured_size() -> None:
    assert "0.080 to 0.095" in _row("subset").sentence
    assert "0.045 to 0.055" in _row("placebo").sentence


def _small(variance: str) -> Any:
    rng = np.random.default_rng(3)
    n = 200
    w = rng.normal(size=n)
    a = rng.binomial(1, 0.5, n)
    h1, h2 = np.exp(-1.0 + 0.3 * w - 0.5 * a), np.exp(-1.5 - 0.3 * w)
    t = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    c = rng.exponential(5.0, n)
    frame = pd.DataFrame(
        {"time": np.minimum(t, c), "event": np.where(t <= c, cause, 0), "A": a, "W": w}
    )
    estimator = ContinuousSurvivalTMLE(
        [1.0],
        cross_fit=False,
        hazard_library=[CoxHazard("main")],
        censoring_library=[CoxHazard("main")],
        variance=variance,  # type: ignore[arg-type]
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, time="time", event="event", treatment="A", baseline=["W"])


@pytest.mark.parametrize("variance", ["switched", "influence_curve"])
def test_the_configuration_reads_the_fits_variance_rule(variance: str) -> None:
    # A nonzero witness of the axis: the configuration reads the estimator's setting, so the
    # two rules give two values and the scope can tell them apart.
    result = _small(variance)
    name = next(name for name in result if name.startswith("ate_regimen"))
    assert continuous_survival_configuration(result, name).variance == variance
