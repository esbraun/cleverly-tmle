"""X82 part F, point law (e): simulated confounding on a fit with missing observations.

The outcome law acts on recorded outcomes only, and every observation indicator stays as
fitted.  The latent is independent of the data, so outcome-axis cells keep missing at random
whenever the fit had it.  A cell that moves the treatment keeps it only when no observation
mechanism reads the treatment, which the caller declares with ``observation_reads=``.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_missing_outcome
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity.simulated_confounding import _draw_latent, _latent_child_seed
from tests.unit._confounding_support import forbid_draw_and_refit

GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.3))
OUTCOME_ONLY = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.3))


@pytest.fixture(scope="module")
def mar() -> Any:
    """An ordinary in-sample TMLE fit with a missing Gaussian outcome."""
    frame, _ = make_missing_outcome(n=300, seed=23)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), missingness="Delta"
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=23,
        )


@pytest.fixture(scope="module")
def composite() -> Any:
    """A three-arm fit with a missing outcome and a declared missing treatment."""
    from tests.unit.test_composite_missing_data import _fitted_learners, _study_frame

    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W",),
        missingness="R",
        treatment_missingness="RA",
    )
    effect = CausalStudy(_study_frame(), design=design).identify(ATE())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(
            **_fitted_learners(), cross_fit=False, simultaneous=False, random_state=0
        )


def _capture(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[Any]:
    seen: list[Any] = []
    original = result.estimator.refit

    def refit(data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(result.estimator, "refit", refit)
    return seen


def _latent(result: Any, seed: int) -> np.ndarray:
    return _draw_latent(result.data.n, _latent_child_seed(seed), None, "row", None)


def test_outcome_axis_cells_move_recorded_outcomes_only(
    mar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture(monkeypatch, mar)
    surface = simulated_confounding(mar, "ate", grid=OUTCOME_ONLY, random_state=1)
    assert surface.complete
    assert surface.missing_observations == "outcome"
    assert surface.observation_reads is None
    (cell,) = seen
    observed = mar.data.observed
    assert np.array_equal(cell.observed, observed)
    assert np.array_equal(cell.treatment, mar.data.treatment)
    expected = mar.data.outcome - 0.3 * _latent(mar, surface.root_seed)
    assert np.allclose(cell.outcome[observed], expected[observed], rtol=0.0, atol=1e-12)
    assert np.array_equal(cell.outcome[~observed], mar.data.outcome[~observed])


def test_a_moved_treatment_needs_the_observation_declaration(
    mar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_draw_and_refit(monkeypatch, mar.estimator)
    with pytest.raises(CapabilityError, match="observation_reads") as caught:
        simulated_confounding(mar, "ate", grid=GRID)
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert caught.value.refusal.roadmap == "F12"


def test_a_declared_mechanism_runs_the_treatment_axis_and_says_so(
    mar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture(monkeypatch, mar)
    surface = simulated_confounding(
        mar, "ate", grid=GRID, random_state=1, observation_reads=("W1", "W2")
    )
    assert surface.complete
    assert surface.observation_reads == ("W1", "W2")
    lines = surface.population_lines()
    assert "the treatment-axis cells assume the observation mechanisms read only W1, W2" in lines
    moved = [data for data in seen if not np.array_equal(data.treatment, mar.data.treatment)]
    assert moved
    for data in seen:
        assert np.array_equal(data.observed, mar.data.observed)
    tail = _latent(mar, surface.root_seed) >= NormalDist().inv_cdf(0.8)
    assert np.array_equal(
        moved[0].treatment, np.where(tail, 1.0 - mar.data.treatment, mar.data.treatment)
    )


@pytest.mark.parametrize(
    ("reads", "error", "fragment"),
    [
        (("A",), ValueError, "which is the treatment"),
        (("Y",), ValueError, "which is the outcome"),
        (("Q",), ValueError, "not an adjustment covariate"),
        (("W1", "W1"), ValueError, "names a covariate twice"),
        ("W1", TypeError, "sequence of covariate names"),
    ],
)
def test_the_declaration_names_adjustment_covariates_only(
    mar: Any, monkeypatch: pytest.MonkeyPatch, reads: Any, error: type, fragment: str
) -> None:
    forbid_draw_and_refit(monkeypatch, mar.estimator)
    with pytest.raises(error, match=fragment):
        simulated_confounding(mar, "ate", grid=GRID, observation_reads=reads)


def test_a_complete_fit_takes_no_observation_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = make_missing_outcome(n=200, seed=4)
    frame = frame.assign(Delta=1)
    study = CausalStudy(
        frame.dropna(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1",))
    )
    result = study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        simultaneous=False,
        random_state=2,
    )
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(ValueError, match="observes every outcome and treatment"):
        simulated_confounding(result, "ate", grid=GRID, observation_reads=("W1",))


def test_calibration_reads_the_recorded_rows_of_each_target(mar: Any) -> None:
    surface = simulated_confounding(
        mar, "ate", grid=OUTCOME_ONLY, random_state=1, benchmark_covariates=("W1",)
    )
    outcome_row = next(row for row in surface.calibrations if row.role == "outcome")
    observed = mar.data.observed
    column = mar.data.covariates[observed, mar.data.covariate_names.index("W1")]
    target = mar.data.outcome[observed]
    expected = np.corrcoef(column, target)[0, 1] * target.std()
    assert outcome_row.strength == pytest.approx(expected, rel=1e-10)
    assert surface.calibration_population == "recorded_rows_of_each_target"


def test_a_missing_treatment_stays_missing_and_the_swap_reads_recorded_rows(
    composite: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = composite.data
    assert data.has_missing_treatment and data.has_missing_outcome
    alias = next(name for name in composite.estimates if name.startswith("ate["))
    with pytest.raises(CapabilityError, match="observation_reads"):
        simulated_confounding(composite, alias, grid=GRID)
    seen = _capture(monkeypatch, composite)
    surface = simulated_confounding(
        composite, alias, grid=GRID, random_state=1, observation_reads=("W",)
    )
    assert surface.complete
    assert surface.missing_observations == "outcome and treatment"
    recorded = data.treatment_recorded
    for replacement in seen:
        assert np.all(np.isnan(replacement.treatment[~recorded]))
        assert np.all(np.isfinite(replacement.treatment[recorded]))
        assert np.array_equal(replacement.treatment_recorded, recorded)
    assert all(cell.induced_treatment_association is not None for cell in surface.cells)
