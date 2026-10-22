"""X82 part F, point laws (f) and (g): the intermediate axis and the bounded outcome laws.

(f) A controlled-direct-effect fit holds its treatment.  Each cell flips the binary
intermediate in the latent tail and moves the outcome, so the latent confounds the
intermediate and the outcome.  A treatment perturbation stays refused (roadmap F15).

(g) A Gaussian fit whose estimator declares ``q_bounds`` moves its outcome on the logit scale
of the bounds, or on the normal-score scale, so every perturbed outcome stays inside them.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, ControlledDirectEffect, PointTreatment
from cleverly.datasets import make_cde
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity.simulated_confounding import (
    _draw_latent,
    _gaussian_outcome,
    _latent_child_seed,
    _logit_outcome,
    _normal_score_outcome,
)
from tests.unit._confounding_support import forbid_draw_and_refit


def _latent(result: Any, seed: int) -> np.ndarray:
    return _draw_latent(result.data.n, _latent_child_seed(seed), None, "row", None)


def _capture(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[tuple[Any, Any]]:
    seen: list[tuple[Any, Any]] = []
    original = result.estimator.refit

    def refit(data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((data, kwargs.get("intermediate_value")))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(result.estimator, "refit", refit)
    return seen


# ------------------------------------------------------------------ (f) intermediate


@pytest.fixture(scope="module")
def cde() -> Any:
    frame, _ = make_cde(n=400, seed=3)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), intermediate="Z"
        ),
    )
    return study.identify(ControlledDirectEffect(intermediate=1.0)).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=2,
        simultaneous=False,
    )


INTERMEDIATE_GRID = ConfounderStrengthGrid(
    treatment=(0.0,), intermediate=(0.0, 0.25), outcome=(0.0, 0.3)
)


def test_intermediate_cells_hold_the_treatment_and_flip_the_intermediate(
    cde: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture(monkeypatch, cde)
    surface = simulated_confounding(cde, "ate", grid=INTERMEDIATE_GRID, random_state=1)
    assert surface.complete
    assert surface.intermediate_name == "Z"
    assert surface.intermediate_value == 1.0
    assert [cell.intermediate_strength for cell in surface.cells] == [0.0, 0.0, 0.25, 0.25]
    latent = _latent(cde, surface.root_seed)
    tail = latent >= NormalDist().inv_cdf(0.75)
    flipped = np.where(tail, 1.0 - cde.data.intermediate, cde.data.intermediate)
    assert len(seen) == 3
    for data, level in seen:
        # Witness: the treatment never moves on a controlled-direct-effect fit.
        assert np.array_equal(data.treatment, cde.data.treatment)
        assert level == 1.0
    assert np.array_equal(seen[0][0].intermediate, cde.data.intermediate)
    assert np.array_equal(seen[1][0].intermediate, flipped)
    assert np.array_equal(seen[2][0].intermediate, flipped)
    moved = surface.cells[2]
    expected = np.corrcoef(latent, flipped)[0, 1]
    assert moved.induced_intermediate_association == pytest.approx(expected, abs=1e-12)
    assert moved.estimate != pytest.approx(surface.original_estimate, abs=1e-9)


def test_the_zero_intermediate_cell_reproduces_the_estimate(cde: Any) -> None:
    surface = simulated_confounding(
        cde,
        "ate",
        grid=ConfounderStrengthGrid(treatment=(0.0,), intermediate=(0.0,), outcome=(0.0,)),
        random_state=1,
    )
    (anchor,) = surface.cells
    assert anchor.estimate == surface.original_estimate
    assert anchor.intermediate_strength == 0.0


def test_a_moved_treatment_on_a_controlled_direct_effect_is_refused(
    cde: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_draw_and_refit(monkeypatch, cde.estimator)
    with pytest.raises(CapabilityError, match="holds the treatment") as caught:
        simulated_confounding(
            cde, "ate", grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
        )
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert caught.value.refusal.roadmap == "F15"


def test_an_intermediate_axis_needs_an_intermediate(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = make_cde(n=200, seed=4)
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    result = study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=2,
        simultaneous=False,
    )
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(ValueError, match="declares no intermediate"):
        simulated_confounding(result, "ate", grid=INTERMEDIATE_GRID)


@pytest.mark.parametrize(
    ("values", "fragment"), [((0.1,), "must contain zero"), ((0.0, 0.6), "between 0 and 0.5")]
)
def test_intermediate_strengths_are_flip_probabilities(values: Any, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,), intermediate=values)


def test_the_intermediate_columns_appear_on_an_intermediate_fit_only(cde: Any) -> None:
    surface = simulated_confounding(cde, "ate", grid=INTERMEDIATE_GRID, random_state=1)
    columns = list(surface.to_frame().columns)
    assert "intermediate_strength" in columns
    assert "induced_intermediate_association" in columns
    assert "intermediate strength" in surface.summary()
    assert any(line.startswith("intermediate: Z held at 1") for line in surface.population_lines())


# ------------------------------------------------------------------ (g) bounded outcome


@pytest.fixture(scope="module")
def bounded() -> Any:
    rng = np.random.default_rng(3)
    n = 400
    w = rng.normal(size=n)
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
    y = np.clip(0.3 + 0.2 * a + 0.1 * w + 0.15 * rng.normal(size=n), 0.0, 1.0)
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=1,
            simultaneous=False,
            q_bounds=(0.0, 1.0),
        )


BOUNDED_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 2.0))


@pytest.mark.parametrize("law", ["logit", "normal_score"])
def test_a_bounded_law_keeps_every_outcome_inside_the_bounds(
    bounded: Any, monkeypatch: pytest.MonkeyPatch, law: str
) -> None:
    seen = _capture(monkeypatch, bounded)
    surface = simulated_confounding(
        bounded,
        "ate",
        grid=BOUNDED_GRID,
        random_state=1,
        bounded_outcome_law=law,  # type: ignore[arg-type]
    )
    assert surface.complete
    assert surface.bounded_outcome_law == law
    assert surface.outcome_bounds == (0.0, 1.0)
    for data, _ in seen:
        assert np.all((data.outcome >= 0.0) & (data.outcome <= 1.0))
    # Mutation control: the linear shift at the same strength leaves the bounds.
    latent = _latent(bounded, surface.root_seed)
    linear = _gaussian_outcome(bounded.data.outcome, latent, 2.0)
    assert np.any((linear < 0.0) | (linear > 1.0))
    assert all(cell.induced_outcome_association is not None for cell in surface.cells)


def test_the_logit_law_is_the_default_and_keeps_rows_at_a_bound(
    bounded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture(monkeypatch, bounded)
    surface = simulated_confounding(bounded, "ate", grid=BOUNDED_GRID, random_state=1)
    assert surface.bounded_outcome_law == "logit"
    y = bounded.data.outcome
    at_bound = (y <= 0.0) | (y >= 1.0)
    assert at_bound.any()
    assert surface.bounded_share == pytest.approx(float(at_bound.mean()), abs=1e-12)
    moved = [data for data, _ in seen if not np.array_equal(data.outcome, y)]
    assert moved
    for data in moved:
        assert np.array_equal(data.outcome[at_bound], y[at_bound])
    lines = surface.population_lines()
    assert any("not outcome SD units" in line for line in lines)


def test_both_bounded_laws_are_the_identity_at_zero_strength() -> None:
    rng = np.random.default_rng(1)
    y = np.round(rng.random(50), 2)
    y[:3] = (0.0, 1.0, 0.5)
    observed = np.ones(50, dtype=bool)
    observed[5] = False
    latent = rng.normal(size=50)
    assert np.array_equal(_logit_outcome(y, observed, latent, 0.0, (0.0, 1.0)), y)
    assert np.array_equal(_normal_score_outcome(y, observed, latent, 0.0, (0.0, 1.0)), y)
    # Without the shortcut, the round trip is the identity up to float error.
    assert np.allclose(
        _normal_score_outcome(y, observed, latent, 1e-300, (0.0, 1.0)), y, rtol=0, atol=1e-12
    )
    # An unrecorded row keeps its placeholder under both laws.
    assert _logit_outcome(y, observed, latent, 3.0, (0.0, 1.0))[5] == y[5]
    assert _normal_score_outcome(y, observed, latent, 3.0, (0.0, 1.0))[5] == y[5]


def test_the_logit_law_matches_its_formula() -> None:
    rng = np.random.default_rng(2)
    y = 2.0 + 3.0 * rng.random(40)
    latent = rng.normal(size=40)
    observed = np.ones(40, dtype=bool)
    t = (y - 2.0) / 3.0
    expected = 2.0 + 3.0 / (1.0 + np.exp(-(np.log(t / (1.0 - t)) - 0.7 * latent)))
    got = _logit_outcome(y, observed, latent, 0.7, (2.0, 5.0))
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_the_normal_score_law_matches_its_formula_at_ties() -> None:
    from scipy.stats import norm

    y = np.array([0.2, 0.5, 0.5, 0.9, 0.1])
    latent = np.array([0.4, -0.3, -0.3, 1.1, -0.8])
    observed = np.ones(5, dtype=bool)
    k = 0.6
    got = _normal_score_outcome(y, observed, latent, k, (0.0, 1.0))
    # Hand-computed: average ranks (2, 3.5, 3.5, 5, 1) over m + 1 = 6, then the shifted
    # score maps back through the sorted outcomes at the positions 1/6, ..., 5/6.
    ranks = np.array([2.0, 3.5, 3.5, 5.0, 1.0])
    shifted = norm.ppf(ranks / 6.0) - k * latent
    expected = np.interp(norm.cdf(shifted), np.arange(1, 6) / 6.0, np.sort(y))
    assert np.allclose(got, expected, rtol=0, atol=1e-12)
    # Tied outcomes with equal latent values move together.
    assert got[1] == got[2]
    # Y' is monotone in the shifted score.
    order = np.argsort(shifted)
    assert np.all(np.diff(got[order]) >= 0.0)


def test_the_normal_score_law_clips_to_the_declared_bounds() -> None:
    # A recorded outcome can lie beyond q_bounds, where the fit clips it.  The empirical
    # quantile would carry it back out, so the law clips every moved row to the bounds.
    y = np.array([-0.2, 0.1, 0.4, 0.7, 1.3])
    latent = np.array([3.0, 0.0, 0.0, 0.0, -3.0])
    observed = np.ones(5, dtype=bool)
    got = _normal_score_outcome(y, observed, latent, 2.0, (0.0, 1.0))
    assert np.all((got >= 0.0) & (got <= 1.0))
    unclipped = np.interp(
        NormalDist().cdf(NormalDist().inv_cdf(1 / 6) - 6.0), np.arange(1, 6) / 6.0, np.sort(y)
    )
    assert unclipped < 0.0  # mutation control: without the clip the row leaves the bounds


def test_a_bounded_law_needs_declared_bounds(cde: Any, bounded: Any) -> None:
    with pytest.raises(ValueError, match="declares none"):
        simulated_confounding(cde, "ate", grid=INTERMEDIATE_GRID, bounded_outcome_law="logit")
    with pytest.raises(ValueError, match="must be one of"):
        simulated_confounding(
            bounded,
            "ate",
            grid=BOUNDED_GRID,
            bounded_outcome_law="probit",  # type: ignore[arg-type]
        )
