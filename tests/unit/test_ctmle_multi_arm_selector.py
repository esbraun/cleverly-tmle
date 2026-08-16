"""Nonzero evidence for the selector-based multi-arm C-TMLE path."""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import CTMLE
from cleverly.estimators.ctmle import _penalty_of, _weighted_partial_correlation
from cleverly.estimators.serialize import dumps, loads
from tests import discrete_law_multi as law
from tests.conftest import FAST_KWARGS


def _fit(stem: str = "ate", **overrides):  # type: ignore[no-untyped-def]
    settings = {
        **FAST_KWARGS,
        "strategy": "discrete",
        "candidates": ((), ("W",)),
        "selection_folds": 2,
        "selection_inner_folds": 2,
        "estimands": ("ey", "ate", "rr", "or"),
        "ctmle_estimand": stem,
        "reference": "low",
        **overrides,
    }
    return CTMLE(**settings).fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.mark.parametrize(
    ("stem", "names"),
    [
        ("ey", ("ey[high]", "ey[low]", "ey[mid]")),
        ("ate", ("ate[high vs low]", "ate[mid vs low]")),
        ("rr", ("rr[high vs low]", "rr[mid vs low]")),
        ("or", ("or[high vs low]", "or[mid vs low]")),
    ],
)
def test_selector_jointly_names_every_nonredundant_component(
    stem: str, names: tuple[str, ...]
) -> None:
    result = _fit(stem)
    selection = result.extra["ctmle"]
    assert selection.target_names == names
    assert result.nuisance.propensity.values.shape == (law.N, law.K)
    np.testing.assert_allclose(result.nuisance.propensity.values.sum(axis=1), 1.0, atol=1e-12)


def test_vector_penalty_is_trace_plus_squared_mean_norm() -> None:
    curve = np.array([[1.0, -2.0], [2.0, 1.0], [4.0, 3.0], [-1.0, 2.0]])
    expected = np.var(curve[:, 0], ddof=1) + np.var(curve[:, 1], ddof=1)
    expected += curve.shape[0] * np.sum(np.mean(curve, axis=0) ** 2)
    assert _penalty_of(curve) == pytest.approx(float(expected))
    assert _penalty_of(curve[:, 0]) == pytest.approx(
        float(np.var(curve[:, 0], ddof=1) + curve.shape[0] * np.mean(curve[:, 0]) ** 2)
    )


def test_scoring_only_one_contrast_is_a_load_bearing_mutation() -> None:
    curve = np.column_stack(
        [np.linspace(-1.0, 1.0, 31), 8.0 + np.linspace(-3.0, 3.0, 31)]
    )
    assert abs(_penalty_of(curve) - _penalty_of(curve[:, 0])) > 100.0


def test_treatment_risk_is_categorical_and_arm_alignment_is_load_bearing() -> None:
    result = _fit("ate")
    selection = result.extra["ctmle"]
    propensity = result.nuisance.propensity
    columns = np.array(
        [propensity.column_for(float(arm)) for arm in result.data.treatment], dtype=int
    )
    observed = propensity.values[np.arange(result.data.n), columns]
    expected = -np.sum(result.data.weights * np.log(np.clip(observed, 1e-12, 1.0)))
    assert selection.treatment_risk_selected == pytest.approx(float(expected))

    permuted = propensity.values[:, [2, 1, 0]]
    wrong = -np.sum(
        result.data.weights
        * np.log(np.clip(permuted[np.arange(result.data.n), columns], 1e-12, 1.0))
    )
    assert abs(float(wrong) - float(expected)) > 1.0


def test_partial_correlation_conditions_on_the_categorical_span() -> None:
    rng = np.random.default_rng(4)
    treatment = np.tile(np.arange(3), 30)
    conditional = np.column_stack([treatment == 1, treatment == 2]).astype(float)
    left = rng.normal(size=treatment.size) + 2.0 * (treatment == 2)
    right = rng.normal(size=treatment.size) - 3.0 * (treatment == 1)
    weights = rng.uniform(0.5, 2.0, size=treatment.size)
    score = _weighted_partial_correlation(left, right, conditional, weights)
    relabelled_basis = np.column_stack([treatment == 0, treatment == 2]).astype(float)
    assert score == pytest.approx(
        _weighted_partial_correlation(left, right, relabelled_basis, weights), abs=1e-12
    )


def test_declared_reference_changes_the_joint_contrast_labels() -> None:
    selection = _fit("ate", reference="high").extra["ctmle"]
    assert selection.target_names == ("ate[low vs high]", "ate[mid vs high]")


def test_joint_selection_diagnostics_survive_serialization() -> None:
    before = _fit("ate").extra["ctmle"]
    after = loads(dumps(_fit("ate"))).extra["ctmle"]
    assert after.target_names == before.target_names
    assert after.path == before.path
    assert after.selected == before.selected
    np.testing.assert_array_equal(after.cv_risk, before.cv_risk)


@pytest.mark.parametrize("strategy", ["greedy", "ordered"])
def test_constructed_paths_fit_one_categorical_mechanism(strategy: str) -> None:
    overrides = {"strategy": strategy, "candidates": None}
    if strategy == "ordered":
        overrides["ordering"] = ("W",)
    result = _fit("ate", **overrides)
    selection = result.extra["ctmle"]
    assert selection.path[0] == ()
    assert selection.path[-1] == ("W",)
    assert result.nuisance.arms == (0.0, 1.0, 2.0)
