"""Plan plumbing, refusals, diagnostics and replay for known longitudinal policies.

The estimator itself is checked against the exact law in
``tests/unit/test_influence_gateaux_longitudinal_policy.py``.  This module checks what a
policy node does around the estimator: how a plan is read, printed and fingerprinted,
which inputs are refused before any learner runs and with what words, what the support
report says at a policy node, and that the truncation replay reproduces a policy fit.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor

from cleverly.exceptions import CapabilityError, DataError, LongitudinalError
from cleverly.interventions import Stochastic
from cleverly.interventions.base import _ESTIMATED_DENSITY, _UNDECLARED_DENSITY
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData, Regimen
from cleverly.longitudinal.estimator import longitudinal_truncation_curve
from cleverly.longitudinal.regimen import (
    declares_policy,
    describe_plan,
    resolve_plans,
    resolve_regimens,
)

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_multivalue as law
from .. import discrete_law_longitudinal_policy as policy
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import QuasiBinomialGLM

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
ARMS = list(policy.ARM_LABELS)


class RaisesIfFitted(BaseEstimator):
    """A learner whose use is the failure: every refusal below precedes any learner."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RaisesIfFitted:
        raise AssertionError("a learner ran before the refusal")

    def predict(self, X: Any) -> np.ndarray:  # pragma: no cover - never fitted
        raise AssertionError("a learner ran before the refusal")

    def predict_proba(self, X: Any) -> np.ndarray:  # pragma: no cover - never fitted
        raise AssertionError("a learner ran before the refusal")


def _raising(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": RaisesIfFitted(),
        "pseudo_learner": RaisesIfFitted(),
        "treatment_learner": RaisesIfFitted(),
        "n_folds": 1,
        "simultaneous": False,
    }
    settings.update(overrides)
    return settings


def _exact(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": law.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    return settings


def _uniform(frame: Any) -> np.ndarray:
    return np.full((len(frame), 3), 1.0 / 3.0)


def _plan(first: Any, second: Any = "low", label: str = "p") -> DynamicRegimen:
    return DynamicRegimen(label, (Stochastic(first, "q1", density_kind="known"), second))


def _fit_refused(plan: Any, frame: Any = None, **overrides: Any) -> None:
    LTMLE({"p": plan}, **_raising(**overrides)).fit(
        law.frame() if frame is None else frame, **COLUMNS
    )


# ------------------------------------------------------------------ T12 refusals


def test_r2_a_density_of_the_wrong_shape_names_the_level_order() -> None:
    with pytest.raises(DataError) as caught:
        _fit_refused(_plan(lambda frame: np.full((len(frame), 2), 0.5)))
    assert str(caught.value) == (
        "the policy at time 1 of regimen 'p' returned shape (512, 2); expected (512, 3). "
        "One column per level of treatment column 'A1', in the order "
        "('high', 'low', 'standard')."
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            [np.nan, 0.5, 0.5],
            "the policy at time 1 of regimen 'p' contains a non-finite probability",
        ),
        ([-0.5, 1.0, 0.5], "the policy at time 1 of regimen 'p' returned a negative probability"),
        (
            [0.5, 0.5, 0.5],
            "the policy at time 1 of regimen 'p' has rows summing to as far as 0.5 from one; "
            "a regime is a distribution over the arms, so its rows must be normalised",
        ),
    ],
)
def test_r3_a_density_that_is_not_a_simplex_is_refused(values: list[float], message: str) -> None:
    with pytest.raises(DataError) as caught:
        _fit_refused(_plan(lambda frame: np.tile(values, (len(frame), 1))))
    assert str(caught.value) == message


def test_r3_reads_only_the_rows_still_in_the_study() -> None:
    """A censored row's density is never read, so it cannot be refused."""
    frame = law.frame().assign(C1=1.0, C2=1.0)
    frame.loc[:9, "C1"] = 0.0
    frame.loc[:9, ["L2", "A2", "C2", "Y"]] = np.nan
    data = LongitudinalData.from_frame(frame, censoring=["C1", "C2"], **COLUMNS)
    reachable = data.uncensored_through(1)

    def second(history: Any) -> np.ndarray:
        values = np.full((len(history), 3), 1.0 / 3.0)
        values[~reachable] = np.nan
        return values

    plan = DynamicRegimen("p", ("low", Stochastic(second, "q2", density_kind="known")))
    (resolved,) = resolve_plans(resolve_regimens({"p": plan}, 2), data)
    assert np.all(resolved.policy[1][~reachable] == 0.0)


def test_r4_a_density_that_raises_names_what_it_was_handed() -> None:
    def reads_later(history: Any) -> Any:
        return history["L2"]

    with pytest.raises(DataError) as caught:
        _fit_refused(_plan(reads_later))
    assert str(caught.value) == (
        "the policy at time 1 of regimen 'p' raised KeyError: 'L2'. It is handed "
        "[W, L_1, ..., L_t] and the earlier treatments, which at time 1 are ['W']. A policy "
        "reading a covariate measured later can only be used from that node on, so pass a "
        "plan with one entry per node."
    )


def test_r5_a_frame_with_other_columns_is_refused() -> None:
    def misnamed(history: Any) -> pd.DataFrame:
        return pd.DataFrame(_uniform(history), columns=["high", "low", "medium"])

    with pytest.raises(DataError) as caught:
        _fit_refused(_plan(misnamed))
    assert str(caught.value) == (
        "the policy at time 1 of regimen 'p' returned columns ['high', 'low', 'medium']. A "
        "frame must have exactly the levels ['high', 'low', 'standard'] of treatment column "
        "'A1', or return an array in that column order."
    )


def test_r6_a_policy_no_observed_history_supports_is_refused_before_any_learner() -> None:
    """``W = 0`` received ``high`` only and ``W = 1`` ``low`` or ``standard``; the policy avoids both."""
    frame = law.frame()
    keep = ((frame["W"] == 0) & (frame["A1"] == "high")) | (
        (frame["W"] == 1) & (frame["A1"] != "high")
    )
    frame = frame[keep].reset_index(drop=True)

    def avoids(history: Any) -> np.ndarray:
        w = np.asarray(history["W"], dtype=int)
        # Sorted order (high, low, standard).
        return np.where(w[:, None] == 0, [[0.0, 0.5, 0.5]], [[1.0, 0.0, 0.0]])

    with pytest.raises(LongitudinalError) as caught:
        _fit_refused(_plan(avoids, second="low"), frame=frame)
    assert str(caught.value) == (
        "no unit's observed treatment history through time 1 has positive probability under "
        "regimen 'p' while remaining in the study, so the sequential regression there has "
        "nothing to fit. The policy is not supported by this sample."
    )


def test_r7_an_undeclared_policy_names_the_node_case() -> None:
    node = Stochastic(_uniform, "q1", density_kind="known")
    plan = DynamicRegimen("p", (node, "low"))
    object.__setattr__(node, "density_kind", None)
    with pytest.raises(CapabilityError) as caught:
        _fit_refused(plan)
    assert "q_t(a | H_t) at a longitudinal node" in str(caught.value)
    assert _UNDECLARED_DENSITY in str(caught.value)


def test_r8_an_estimated_policy_names_x19() -> None:
    node = Stochastic(_uniform, "q1", density_kind="known")
    plan = DynamicRegimen("p", (node, "low"))
    object.__setattr__(node, "density_kind", "estimated")
    with pytest.raises(CapabilityError) as caught:
        _fit_refused(plan)
    assert _ESTIMATED_DENSITY in str(caught.value)
    assert str(caught.value).endswith(
        "A tilt of the mechanism at each node of a longitudinal fit is not fitted yet; "
        "docs/roadmap.md X19 tracks it."
    )


def test_r9_a_static_regimen_cannot_hold_a_policy() -> None:
    with pytest.raises(DataError) as caught:
        Regimen("p", (Stochastic(_uniform, "q1", density_kind="known"), "low"))
    assert str(caught.value) == (
        "regimen 'p' holds a stochastic policy node. Write a plan with a policy node as "
        "DynamicRegimen(label, plan), or pass it in regimens= as a mapping value"
    )


def test_r10_interventions_points_to_a_policy_node() -> None:
    with pytest.raises(TypeError) as caught:
        LTMLE({"low": "low"}, interventions=[])
    assert str(caught.value).endswith(
        "A node that draws its arm from a known distribution is a Stochastic node in that "
        "plan: DynamicRegimen('mix', (Stochastic(q1, 'q1', density_kind='known'), 'low'))"
    )


def _r11_frame(extra_rows: int) -> pd.DataFrame:
    """``A2 = standard`` only after ``A1 = high``, plus ``extra_rows`` after ``A1 = low``."""
    frame = law.frame()
    drop = (frame["A1"] != "high") & (frame["A2"] == "standard")
    kept = frame[~drop]
    extra = frame[(frame["A1"] == "low") & (frame["A2"] == "standard")].head(extra_rows)
    return pd.concat([kept, extra], ignore_index=True)


def _r11_plan() -> DynamicRegimen:
    def avoid_high(history: Any) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.5, 0.0, 0.5], (len(history), 1)), columns=["standard", "high", "low"]
        )

    return DynamicRegimen(
        "p",
        (
            Stochastic(avoid_high, "q1", density_kind="known"),
            Stochastic(_uniform, "q2", density_kind="known"),
        ),
    )


def test_r11_a_level_with_policy_mass_and_no_fitted_row_is_refused_in_sample() -> None:
    with pytest.raises(LongitudinalError) as caught:
        _fit_refused(_r11_plan(), frame=_r11_frame(0))
    assert str(caught.value) == (
        "regimen 'p' puts probability on 'standard' at time 2, but no unit that remains on "
        "the policy through time 2 received 'standard' there. The regression at that node "
        "would predict 'standard' with no row to fit it on, so the policy is not supported "
        "by this sample."
    )


def test_r11_is_checked_in_each_outer_training_fold() -> None:
    frame = _r11_frame(1)
    with pytest.raises(LongitudinalError) as caught:
        _fit_refused(_r11_plan(), frame=frame, n_folds=2, random_state=0)
    message = str(caught.value)
    assert message.startswith(
        "regimen 'p' puts probability on 'standard' at time 2, but no unit in outer training fold "
    )
    assert message.endswith(
        "Fit in sample (CrossFitting(enabled=False), or n_folds=1 on the engine), or choose "
        "a policy this sample supports."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LTMLE({"p": _r11_plan()}, **_exact()).fit(frame, **COLUMNS)


# ------------------------------------------------------------------ T13 plumbing


def test_a_policy_node_prints_its_name() -> None:
    assert describe_plan(policies.regimen("taper")) == "high/q:taper2"


def test_a_stochastic_mapping_value_is_one_policy_at_every_node() -> None:
    node = Stochastic(_uniform, "u", density_kind="known")
    (resolved,) = resolve_regimens({"u": node}, 2)
    assert isinstance(resolved, DynamicRegimen)
    assert resolved.plan == (node, node)
    assert declares_policy({"u": node}) and not declares_policy({"low": "low"})


def test_the_policy_frame_holds_the_history_and_the_earlier_labels() -> None:
    """A third node, so a unit censored after the first node has no second arm to read."""
    frame = law.frame().assign(A3=lambda f: f["A2"], C1=1.0, C2=1.0, C3=1.0)
    frame.loc[:4, "C1"] = 0.0
    frame.loc[:4, ["L2", "A2", "C2", "A3", "C3", "Y"]] = np.nan
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], []],
        censoring=["C1", "C2", "C3"],
    )
    handed = data.policy_frame(3)
    assert list(handed.columns) == [*data.history_names(3), "A1", "A2"]
    assert set(handed["A1"]) == set(data.treatment_levels[0])
    absent = np.isnan(data.treatment[:, 1])
    assert absent.sum() == 5
    assert set(handed["A2"][absent]) == {data.treatment_levels[1][0]}
    present = ~absent
    labels = np.asarray(data.treatment_levels[1], dtype=object)
    codes = data.treatment[present, 1].astype(int)
    assert list(handed["A2"][present]) == list(labels[codes])


def test_two_policies_with_one_support_have_different_fingerprints() -> None:
    def fit(plans: dict[str, Any]) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE(plans, **_exact()).fit(law.frame(), **COLUMNS)

    def leaning(table: list[float]) -> Any:
        return lambda frame: np.tile(table, (len(frame), 1))

    first = DynamicRegimen(
        "a", (Stochastic(leaning([0.5, 0.25, 0.25]), "a", density_kind="known"), "low")
    )
    second = DynamicRegimen(
        "b", (Stochastic(leaning([0.25, 0.5, 0.25]), "b", density_kind="known"), "low")
    )
    moved = DynamicRegimen(
        "c", ("low", Stochastic(leaning([0.5, 0.25, 0.25]), "c", density_kind="known"))
    )
    digests = dict(fit({"a": first, "b": second, "c": moved}).config.plan_fingerprints)
    assert len(set(digests.values())) == 3


def test_continuing_the_drawn_arm_is_a_one_hot_policy() -> None:
    """The route for a rule that must read an earlier policy arm (review finding 12)."""
    levels = ("high", "low", "standard")

    def first(history: Any) -> np.ndarray:
        chosen = np.where(np.asarray(history["W"]) == 0, "high", "low")
        return np.column_stack([(chosen == level).astype(float) for level in levels])

    def keep(history: Any) -> np.ndarray:
        drawn = np.asarray(history["A1"])
        return np.column_stack([(drawn == level).astype(float) for level in levels])

    continued = DynamicRegimen(
        "plan",
        (
            Stochastic(first, "first", density_kind="known"),
            Stochastic(keep, "keep", density_kind="known"),
        ),
    )

    def recomputed(history: Any) -> Any:
        return np.where(np.asarray(history["W"]) == 0, "high", "low")

    rule = DynamicRegimen("plan", (recomputed, recomputed), rule_kind="known")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        left = LTMLE({"plan": continued}, **_exact()).fit(law.frame(), **COLUMNS)
        right = LTMLE({"plan": rule}, **_exact()).fit(law.frame(), **COLUMNS)
    assert left.config.policy_point_mass_nodes == (("plan", (1, 2)),)
    name = "ey_regimen[plan]"
    assert left.psi(name) == pytest.approx(right.psi(name), abs=1e-12)
    np.testing.assert_allclose(
        left.influence_curves[name], right.influence_curves[name], atol=1e-12
    )


# ------------------------------------------------------------------ T14 diagnostics


@pytest.fixture(scope="module")
def mix_fit() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(policies.regimens(("low", "mix")), **_exact(reference="low")).fit(
            law.frame(), **COLUMNS
        )


def test_the_support_report_shows_what_the_policy_draws(mix_fit: Any) -> None:
    frame = mix_fit.diagnostics.support().to_frame()
    assert "node_kind" in frame.columns
    rows = frame.set_index(["regimen", "time"])
    assert rows.loc[("low", 1), "node_kind"] == "label"
    assert rows.loc[("mix", 2), "node_kind"] == "policy"
    fit = mix_fit.fits["mix"]
    step = fit.steps[1]
    means = fit.policy[1][step.at_risk].mean(axis=0)
    expected = ", ".join(
        f"{level}={mean:.3g}"
        for level, mean in zip(mix_fit.data.treatment_levels[1], means, strict=True)
    )
    assert rows.loc[("mix", 2), "assigned_shares"] == expected


def test_the_support_report_of_a_deterministic_fit_has_no_node_kind() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE({"low": "low", "high": "high"}, **_exact()).fit(law.frame(), **COLUMNS)
    assert "node_kind" not in result.diagnostics.support().to_frame().columns


def test_the_settings_report_lists_the_policies(mix_fit: Any) -> None:
    lines = mix_fit.config.describe(contrast=True)
    assert any(line.startswith("  a 'q' is a known policy") for line in lines)


# ------------------------------------------------------------------ T15 truncation replay


def test_the_truncation_replay_reproduces_a_policy_fit_and_moves_under_a_bound() -> None:
    """A misspecified initial fit, so the bound moves the fluctuation and the estimate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            policies.regimens(("low", "mix")),
            **_exact(
                reference="low",
                outcome_learner=QuasiBinomialGLM(),
                pseudo_learner=DummyRegressor(),
            ),
        ).fit(law.frame(), **COLUMNS)
    lower = 0.1
    curve = longitudinal_truncation_curve(result, [NO_TRUNCATION, lower])
    curve = curve[curve["estimand"] == "ey_regimen[mix]"].set_index("is_fitted_bound")
    assert curve.loc[True, "delta_from_fitted"] == 0.0
    assert abs(curve.loc[False, "delta_from_fitted"]) > 1e-4
    assert curve.loc[False, "truncated_score_cells"] > 0


def test_a_bounded_denominator_can_leave_a_ratio_below_one(mix_fit: Any) -> None:
    """Review finding 13: ``g_bounds`` bounds the denominator, so a truncated cell can be light."""
    fit = mix_fit.fits["mix"]
    step = fit.steps[1]
    lower = 0.1
    rows = step.trained_on & (fit.cumulative_unbounded[:, 1] < lower)
    ratio = fit.cumulative_numerator[:, 1] / lower
    assert rows.any()
    assert np.any(ratio[rows] < 1.0)
