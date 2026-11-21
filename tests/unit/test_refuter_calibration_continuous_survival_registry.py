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
    continuous_survival_learners,
    resolve,
)
from tests.studies import refuter_calibration_continuous_survival as study

ARTIFACTS = study.STUDY.artifacts
CELLS = pd.read_csv(ARTIFACTS / "cells.csv", keep_default_na=False)

#: The learners every trial ran, named whole as the configuration reader names them now. The
#: committed ``cells.csv`` names them member by member, as the reader did when the run wrote
#: it; ``test_the_committed_learners_are_the_study_estimators`` ties the two together.
LEARNERS = continuous_survival_learners(study.estimator(0), study.N)


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
        learners=LEARNERS,
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
        key, reading = str(row.cell).split("__", 1)
        # A subset power cell rescales the statistic of one of size trials 0 to 49, so it
        # reads the size law; only the stale-row placebo cells run the power law.
        law = "power" if reading == "placebo_stale_rows" else "size"
        recorded = CELLS.loc[(CELLS["cell"] == key) & (CELLS["law"] == law)].iloc[0]
        configuration = _configuration(recorded)
        if instrument == "placebo":
            configuration = replace(configuration, placebo_draw=study.PLACEBO_DRAW)
        kind: Literal["size", "power"] = "size" if row.property == "refuter_size" else "power"
        shares = (float(recorded["event_share_min"]), float(recorded["event_share_max"]))
        cells[f"{row.property}/{row.cell}"] = CalibrationCell(
            kind=kind,
            passed=bool(row.property_passed),
            configuration=configuration,
            event_shares=shares,
        )
    return cells


def test_the_committed_learners_are_the_study_estimators() -> None:
    # cells.csv records each library member in turn; the same estimator, named whole, is what
    # the registry rows cover, so the committed column and LEARNERS describe the same fits.
    estimator = study.estimator(0)
    members = [repr(m) for m in (*estimator.hazard_library, *estimator.censoring_library)]
    assert set(CELLS["learners"]) == {" | ".join([*members, "default"])}
    assert (
        "hazard=[" + ", ".join(repr(m) for m in estimator.hazard_library) + "]",
        "censoring=[" + ", ".join(repr(m) for m in estimator.censoring_library) + "]",
        "treatment=default",
    ) == LEARNERS
    for row in _rows():
        assert row.scope.learners == frozenset(LEARNERS)


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
    # The covered range holds every size trial's measured share, rounded outward to 1e-4.
    assert float(size["event_share_min"].min()) - 1e-4 < low <= float(size["event_share_min"].min())
    assert (
        float(size["event_share_max"].max()) <= high < float(size["event_share_max"].max()) + 1e-4
    )
    assert scope.variances == frozenset({study.estimator(0).variance}) == frozenset({"switched"})


@pytest.mark.parametrize(
    "axis, widened",
    [
        ("min_event_count_min", 16),
        ("covariates_per_row_max", 0.008),
        ("n_min", 500),
        ("estimand_kinds", frozenset({"continuous:ate_regimen", "continuous:rr_regimen"})),
        ("variances", frozenset({"switched", "influence_curve"})),
        ("variances", None),
        ("causes", frozenset({1, 2})),
        ("causes", None),
        ("event_share_range", (0.40, 0.7946)),
        ("event_share_range", (0.4336, 0.85)),
        ("event_share_range", None),
        ("learners", frozenset({*LEARNERS, "hazard=[CoxHazard('main')]"})),
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
    # outside the measured range, too few events, and a reduced hazard or censoring library.
    recorded = CELLS.loc[(CELLS["cell"] == "c1_t1.12") & (CELLS["law"] == "size")].iloc[0]
    # The cell records its smallest trial share, the edge of the covered range.
    fit = _configuration(recorded)
    if instrument == "placebo":
        fit = replace(fit, placebo_draw="permute")
    assert resolve(instrument, rule, fit).licensed
    outside = {
        "variance": replace(fit, variance="influence_curve"),
        "causes": replace(fit, causes=1),
        "share": replace(fit, event_share=0.30),
        "events": replace(fit, min_event_count=16),
        # A subset of a default library is a different fit (implementation review 1).
        "hazard library": replace(fit, learners=("hazard=[CoxHazard('main')]", *fit.learners[1:])),
        "censoring library": replace(
            fit, learners=(fit.learners[0], "censoring=[CoxHazard('main')]", fit.learners[2])
        ),
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


def test_a_fit_with_a_reduced_library_is_not_licensed() -> None:
    # The fit's own reader names a one-member library whole, so the scope, which names the
    # three-member defaults, leaves it descriptive. Before the libraries were named whole,
    # CoxHazard('main') was a member of the default and this fit's learners were covered.
    result = _small("switched")
    name = next(name for name in result if name.startswith("ate_regimen"))
    reduced = continuous_survival_configuration(result, name).learners
    assert reduced[:2] == ("hazard=[CoxHazard('main')]", "censoring=[CoxHazard('main')]")
    recorded = CELLS.loc[(CELLS["cell"] == "c1_t1.12") & (CELLS["law"] == "size")].iloc[0]
    fit = _configuration(recorded)
    assert resolve("subset", "rho", fit).licensed
    assert not resolve("subset", "rho", replace(fit, learners=reduced)).licensed
