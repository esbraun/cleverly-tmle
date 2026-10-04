"""Longitudinal modified treatment policies, against an independent g-formula.

The oracle is :mod:`tests.discrete_law_longitudinal_mtp`: the parameter under each plan of
modified treatment policies, written as ratios of cell masses, and its complex-step Gateaux
derivative.  A saturated learner on a sample that realises the law exactly makes every
comparison exact, so the point estimate and the influence curve must equal the oracle to
rounding, whichever reading of the doses the fit takes: a continuous node with a binned
density or a stacked classifier, a categorical node with numeric levels, or a vector node.
"""

from __future__ import annotations

import itertools
import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import longitudinal_mtp as mtp

CONTINUOUS = ("A1", "A2")
NO_TRUNCATION = (1e-8, 1.0)
ROWS = law.first_row_of()
#: How far a mutation must move the estimate or the curve to count as caught.
CAUGHT = 1e-4

UNIT_PLANS = (
    "natural",
    "up",
    "history",
    "decreasing",
    "random",
    "up at 2",
    "up then history",
)


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    return settings


def fit(
    labels: tuple[str, ...] = UNIT_PLANS,
    *,
    reading: str = "continuous",
    regimens: Any = None,
    frame: Any = None,
    **overrides: Any,
) -> Any:
    """A saturated fit of ``labels`` on the exact law, the first as reference."""
    columns: dict[str, Any] = {
        "outcome": "Y",
        "baseline": ["W"],
        "time_varying": [[], ["L2"]],
    }
    if reading == "vector":
        columns["treatment"] = [["A1a", "A1b"], ["A2a", "A2b"]]
        data = law.frame(vector=True) if frame is None else frame
    else:
        columns["treatment"] = ["A1", "A2"]
        data = law.frame() if frame is None else frame
        if reading == "continuous":
            columns["continuous_treatment"] = list(CONTINUOUS)
    plans = mtp.regimens(labels) if regimens is None else regimens
    settings = _settings(reference=labels[0], **overrides)
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return LTMLE(plans, **settings).fit(data, **columns)


def names(labels: tuple[str, ...]) -> tuple[str, ...]:
    reference = labels[0]
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {reference}]" for label in labels if label != reference
    )


def deviation(result: Any, labels: tuple[str, ...]) -> float:
    """The largest error of any reported estimate or curve against the oracle."""
    worst = 0.0
    for name in names(labels):
        worst = max(worst, abs(result.psi(name) - law.truth(name)))
        worst = max(
            worst, float(np.max(np.abs(result.influence_curves[name][ROWS] - law.eif(name))))
        )
    return worst


@cache
def _fit(reading: str, ratio: str = "density", n_folds: int = 1) -> Any:
    return fit(reading=reading, ratio=ratio, n_folds=n_folds)


@pytest.mark.parametrize(
    ("reading", "ratio"),
    [("continuous", "density"), ("continuous", "classifier"), ("categorical", "density")],
)
@pytest.mark.parametrize("name", names(UNIT_PLANS))
def test_point_estimate_is_the_g_formula(reading: str, ratio: str, name: str) -> None:
    """T2: every plan, every reading of the doses, both ratio routes."""
    assert _fit(reading, ratio).psi(name) == pytest.approx(law.truth(name), abs=1e-12)


@pytest.mark.parametrize(
    ("reading", "ratio"),
    [("continuous", "density"), ("continuous", "classifier"), ("categorical", "density")],
)
@pytest.mark.parametrize("name", names(UNIT_PLANS))
def test_influence_curve_is_the_gateaux_derivative(reading: str, ratio: str, name: str) -> None:
    """T2: the influence curve at every support point."""
    np.testing.assert_allclose(
        _fit(reading, ratio).influence_curves[name][ROWS], law.eif(name), atol=1e-10, rtol=0
    )


def test_a_vector_node_is_its_product_levels() -> None:
    """Part (c): two binary components at each node, read as one four-level node."""
    labels = ("natural", "vector")
    result = fit(labels, reading="vector")
    assert deviation(result, labels) < 1e-10
    assert result.data.treatment_levels[0] == ((0, 0), (0, 1), (1, 0), (1, 1))


def test_a_scale_at_node_two_reads_its_jacobian() -> None:
    """``Scale(2, cap=8)`` on bins that double in width equals the law's index map."""
    labels = ("natural", "up at 2")
    frame = law.frame(law.UNIT_GRID, law.SCALED_GRID)
    result = fit(labels, regimens=mtp.scaled_regimens(), frame=frame)
    assert deviation(result, labels) < 1e-10


# ------------------------------------------------------------------ nonzero witnesses


def test_the_witnesses_are_not_degenerate() -> None:
    natural = law.truth("ey_regimen[natural]")
    statics = law.static_means()
    labels = (*UNIT_PLANS[1:], "vector")
    means = {label: law.truth(f"ey_regimen[{label}]") for label in labels}
    for label, value in means.items():
        assert abs(value - natural) > 0.01, label
        assert min(abs(value - other) for other in statics.values()) > 0.005, label
    ordered = sorted(means.values())
    assert min(b - a for a, b in itertools.pairwise(ordered)) > 0.002
    # Node 2 reads the intervened earlier dose: reading the natural one moves the parameter.
    plan = law.PLANS["up then history"]
    natural_read = float(np.real(law.functional_mtp(law.PROBS, plan, intervened_history=False)))
    assert abs(law.truth("ey_regimen[up then history]") - natural_read) > 0.01


def test_the_ratios_differ_from_one_at_each_node() -> None:
    result = _fit("continuous")
    plan = result.fits["up"]
    numerator = plan.cumulative_numerator
    assert numerator is not None
    first = numerator[:, 0]
    second = numerator[:, 1] / np.where(first > 0, first, 1.0)
    assert np.max(np.abs(first - 1.0)) > 0.5
    assert np.max(np.abs(second - 1.0)) > 0.5
