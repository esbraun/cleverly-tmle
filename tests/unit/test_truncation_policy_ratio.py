"""A truncation curve of a policy fit sweeps only a bound its clever covariate reads (X124 B6).

``g_bounds`` reaches a policy's ratio only through a categorical factor the policy changes:
the fitted probabilities of a categorical treatment, or the categorical factor of a vector
treatment on the density route (``PolicySet.design_at``).  A dose's density ratio, a
classifier ratio, or a factor that cancels leaves every point of the curve equal to the
fit, so the curve refuses those fits.  The moving cases are the nonzero witnesses: the rule
must not refuse a fit whose curve moves.

The categorical and flag fits use the M1 law of ``test_vector_density_route``, where
``P(F = 1 | W = 0) = 1 / 4``, so the bound 0.3 binds.  The dose fits use the mixed law of
``test_point_vector_treatment``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sklearn.linear_model import LogisticRegression

import cleverly.sensitivity.positivity as positivity
from cleverly._refusal import RefusalReason
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import ModifiedPolicy
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from . import test_point_vector_treatment as point_tests
from . import test_vector_density_route as density_tests

NAME = "ey_policy[set flag]"
SETTINGS: dict[str, Any] = {
    "outcome_learner": LogisticRegression(),
    "cross_fit": False,
    "simultaneous": False,
    "g_bounds": density_tests.NO_TRUNCATION,
}


def _keep(a: Any, h: Any) -> Any:
    return a


def _fit(case: str) -> Any:
    frame = density_tests._m1_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = CausalData.from_frame(
            frame, outcome="Y", treatment=["F", "D"], covariates=["W"], continuous_treatment=["D"]
        )
        flag = CausalData.from_frame(frame, outcome="Y", treatment="F", covariates=["W", "D"])
        set_flag = ModifiedPolicy("set flag", apply=density_tests.set_flag, policy_kind="known")
        keep = ModifiedPolicy("keep", apply=_keep, policy_kind="known")
        cells = {"treatment_learner": CellProbabilities()}
        fits = {
            "categorical density": lambda: TMLE(policies=[set_flag], **cells, **SETTINGS).fit(flag),
            # The identity map on the density route: the bounded factor cancels.
            "categorical identity": lambda: TMLE(policies=[keep], **cells, **SETTINGS).fit(flag),
            "categorical classifier": lambda: TMLE(
                policies=[set_flag], ratio="classifier", **SETTINGS
            ).fit(flag),
            "vector flag density": lambda: TMLE(
                policies=[density_tests.SET_FLAG_ONLY], ratio="density", **cells, **SETTINGS
            ).fit(vector),
            "vector flag classifier": lambda: TMLE(
                policies=[density_tests.SET_FLAG_ONLY], ratio="classifier", **SETTINGS
            ).fit(vector),
        }
        return fits[case]().single()


def _dose_fits() -> dict[str, Any]:
    """A vector and a scalar fit of the mixed law whose policy moves only the dose."""
    vector, scalar = point_tests._sample_fits("density")
    return {"vector dose density": vector, "scalar dose density": scalar}


MOVING = ("categorical density", "vector flag density")
FLAT = (
    "categorical identity",
    "categorical classifier",
    "vector flag classifier",
    "vector dose density",
    "scalar dose density",
)


@pytest.fixture(scope="module")
def fits(exact_grid_law_module: Any) -> dict[str, Any]:
    """Every case, fitted once.  Both laws are exact grids, so the atom refusal is off."""
    m1 = (*MOVING, "categorical identity", "categorical classifier", "vector flag classifier")
    return {**{case: _fit(case) for case in m1}, **_dose_fits()}


def _unruled_curve(result: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The curve with the density-ratio rule dropped: the deliberate mutation."""
    rules = tuple(rule for rule in positivity._TRUNCATION_RULES if rule[0] != "density_ratio")
    monkeypatch.setattr(positivity, "_TRUNCATION_RULES", rules)
    return positivity.truncation_curve(result, [0.3, 0.1])


@pytest.mark.parametrize("case", MOVING)
def test_a_policy_that_changes_a_categorical_value_moves_under_the_bound(
    fits: dict[str, Any], case: str
) -> None:
    result = fits[case]
    assert result.nuisance.policies.reads_bounded_probability
    assert result.diagnostics.capability("truncation_curve").available
    curve = result.diagnostics.truncation_curve([0.3, 0.1])
    row = curve[(curve["bound"] == 0.3) & (curve["estimand"] == NAME)]
    assert abs(float(row["delta_from_fitted"].iloc[0])) > 1e-4


@pytest.mark.parametrize("case", FLAT)
def test_a_ratio_that_reads_no_bound_refuses_the_curve(fits: dict[str, Any], case: str) -> None:
    result = fits[case]
    assert not result.nuisance.policies.reads_bounded_probability
    refusal = positivity.truncation_refusal(result)
    assert refusal is not None
    assert refusal.reason is RefusalReason.INVALID_DECLARATION
    row = result.diagnostics.capability("truncation_curve")
    assert not row.available
    with pytest.raises(CapabilityError, match="every point of the curve would equal the fit"):
        positivity.truncation_curve(result, [0.3])


@pytest.mark.parametrize("case", FLAT)
def test_without_the_rule_the_curve_repeats_the_fit(
    fits: dict[str, Any], case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation: dropping the rule returns a curve that never leaves the fit."""
    curve = _unruled_curve(fits[case], monkeypatch)
    assert (curve["delta_from_fitted"].abs() == 0.0).all()


def test_the_rule_is_on_the_propensity_axis_only(fits: dict[str, Any]) -> None:
    assert positivity._refuse_unbounded_ratio(fits["vector dose density"], True) is None


@pytest.mark.parametrize(("ratio", "noted"), [("density", True), ("classifier", False)])
def test_the_capped_weights_note_appears_only_where_the_bound_reaches_the_ratio(
    exact_grid_law_module: Any, ratio: str, noted: bool
) -> None:
    """The note sends the reader to truncation_curve(), so it must not appear where that
    curve refuses: a classifier ratio reads no bound even though the bound clips rows."""
    frame = density_tests._m1_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flag = CausalData.from_frame(frame, outcome="Y", treatment="F", covariates=["W", "D"])
        set_flag = ModifiedPolicy("set flag", apply=density_tests.set_flag, policy_kind="known")
        learner = {"treatment_learner": CellProbabilities()} if ratio == "density" else {}
        settings = {**SETTINGS, "g_bounds": (0.3, 0.7)}
        result = TMLE(policies=[set_flag], ratio=ratio, **learner, **settings).fit(flag).single()
    assert ("weights_capped" in result.summary().note_codes) is noted
