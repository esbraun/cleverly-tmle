"""The typed learned-rule contrasts: ``LearnedRuleContrast`` and ``OptimalRuleContrast``.

``CausalStudy.identify(LearnedRuleContrast(against=(...)))`` fits the engine's learned rule
beside the comparators (``TMLE(learned_rule=..., interventions=[...])``) and keys every reported
name.  The witnesses:

- W-d1 (route): the typed route equals the engine route bit for bit, for a fixed and a random
  plan, a random plan and the natural course, the natural course alone, and the optimal
  contrast against a level.
- W-d2 (keys): the key and estimate guard passes with one to three comparators, with strata
  and for the optimal contrast.
- W-d3 (reduction): a one-hot ``Stochastic`` equals the known ``Rule`` of the same plan bit for
  bit at a nonzero coefficient.
- W-d4 (stochastic column): ``Stochastic(0.5 1{W1 > 0})`` at unequal folds with a misspecified
  regression has the longhand point, curve and variance of :math:`D_L - D_K` under each report,
  its score is solved, and the fluctuation of the learned column alone leaves it unsolved.
  Rounding the plan to one-hot moves the point.
- W-d7 (sign): with a stub regression every fold rule is ``1{W1 > 0}``, and each known
  contrast equals the regime fit of that rule minus the comparator, on the same folds.
- W-d8 (optimal): the optimal contrast reports ``ey_optimal_rule``, each comparator's value and
  ``ate_optimal_rule``, bit for bit with the learned-rule arithmetic.
- W-d9 (label): a level, a ``Static`` and a known ``Rule`` read ``T44``; a ``Stochastic``, its
  subclass, a subclass of ``Static`` and a user-written density read ``T58.stochastic`` with
  ``known_stochastic_regimes``.  The mutation that whitelists by class name lets a user class
  called ``Static`` through, and the whitelist test catches it.
- W-d10 (refusals): each typed refusal, before any learner.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import (
    CausalStudy,
    CrossFitting,
    LearnedRuleContrast,
    LearnedRuleValue,
    OptimalRuleContrast,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    TMLEMethod,
)
from cleverly._support import stochastic_comparators
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import (
    Incremental,
    LearnedRule,
    NaturalCourse,
    Rule,
    Static,
    Stochastic,
)
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_learned_rule_in_sample import SignStub
from tests.unit.test_learned_rule_strata import _strata_frame

FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
DESIGN = PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES)
LEARNED = support.NAME

#: The declared floors of W-d4, each below what the fixed draw gives.
W_D4_UNSOLVED = 1e-2  # observed 0.071 under the learned-column-only fluctuation
W_D4_ONE_HOT = 1e-3  # observed 0.068 between the offer and its one-hot rounding


def _offer(name: str = "offer") -> Stochastic:
    def density(w: pd.DataFrame) -> np.ndarray:
        p = 0.5 * (np.asarray(w["W1"]) > 0)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, name=name, density_kind="known")


def _one_hot(name: str = "w1") -> Stochastic:
    def density(w: pd.DataFrame) -> np.ndarray:
        p = (np.asarray(w["W1"]) > 0).astype(float)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, name=name, density_kind="known")


def _w1_rule(name: str = "w1") -> Rule:
    return Rule(lambda w: (w["W1"] > 0).astype(float), name=name, rule_kind="known")


class _Mixture:
    """A user-written known regime: arm 1 with probability 0.3 everywhere."""

    name = "mixture"
    density_kind = "known"

    def density(self, data: Any) -> np.ndarray:
        return np.tile([0.7, 0.3], (data.n, 1))


def _half(self: Any, data: Any) -> np.ndarray:
    return np.tile([0.5, 0.5], (data.n, 1))


#: A user subclass of ``Static`` that is also named ``Static``, with a density that is not
#: one-hot: the class a whitelist by class name would let through.
_SubStatic = type(
    "Static", (Static,), {"density": _half, "density_kind": "known", "__module__": __name__}
)


def _engine(cv_evaluation: bool, **settings: Any) -> Any:
    return support.fit(
        support.law_frame(203, 1),
        outcome_learner=support.misspecified_learner(),
        cv_evaluation=cv_evaluation,
        **settings,
    )


def _typed(estimand: Any, cv_evaluation: bool, frame: Any = None, design: Any = DESIGN) -> Any:
    study = CausalStudy(support.law_frame(203, 1) if frame is None else frame, design=design)
    method = TMLEMethod(cross_fitting=CrossFitting(n_folds=5, fold_evaluation=cv_evaluation))
    return study.identify(estimand).estimate(
        method,
        outcome_learner=support.misspecified_learner(),
        treatment_learner=support.treatment_learner(),
        simultaneous=False,
        random_state=0,
    )


# ------------------------------------------------------------------------------- W-d1


ROUTES: dict[str, tuple[Any, dict[str, Any]]] = {
    "static_offer": (lambda: (Static(0), _offer()), {}),
    "offer_natural": (lambda: (_offer(), NaturalCourse()), {}),
    "natural": (lambda: (NaturalCourse(),), {}),
}


@pytest.mark.parametrize("cv_evaluation", [True, False])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_w_d1_the_typed_route_is_the_engine_route(route: str, cv_evaluation: bool) -> None:
    against, _ = ROUTES[route]
    typed = _typed(LearnedRuleContrast(against=against()), cv_evaluation)
    engine = _engine(cv_evaluation, interventions=list(against()))
    assert list(typed.estimates) == list(engine.estimates)
    for name, estimate in engine.estimates.items():
        other = typed.estimates[name]
        assert other.psi == estimate.psi, name
        assert other.std_error == estimate.std_error, name
        np.testing.assert_array_equal(other.influence_curve, estimate.influence_curve)
    assert set(typed.parameter_keys) == set(typed.estimates)


def test_w_d1_the_natural_course_alone_reports_three_rows() -> None:
    typed = _typed(LearnedRuleContrast(against=(NaturalCourse(),)), True)
    assert list(typed.estimates) == [
        LEARNED,
        "ey_natural_course[natural course]",
        "ate_learned_rule[learned rule vs natural course]",
    ]


def test_w_d1_the_optimal_contrast_is_the_engine_route() -> None:
    typed = _typed(OptimalRuleContrast(against=(Static(0),)), True)
    engine = _engine(True, interventions=[Static(0)], estimands=["ey_optimal_rule"])
    assert list(typed.estimates) == list(engine.estimates)
    for name, estimate in engine.estimates.items():
        assert typed.estimates[name].psi == estimate.psi
        np.testing.assert_array_equal(
            typed.estimates[name].influence_curve, estimate.influence_curve
        )


# ------------------------------------------------------------------------------- W-d2


@pytest.mark.parametrize(
    "estimand",
    [
        LearnedRuleContrast(against=(0,)),
        LearnedRuleContrast(against=(0, Static(1))),
        LearnedRuleContrast(against=(0, _offer(), NaturalCourse())),
        OptimalRuleContrast(against=(0, NaturalCourse())),
    ],
    ids=["one", "two", "three", "optimal"],
)
def test_w_d2_every_reported_name_is_keyed(estimand: Any) -> None:
    result = _typed(estimand, False)
    assert set(result.parameter_keys) == set(result.estimates)
    for name, key in result.parameter_keys.items():
        assert key.alias == name
        assert key.axis == "learned_rule"


def test_w_d2_stratum_names_are_keyed() -> None:
    frame = _strata_frame(203, 1)
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=(*support.COVARIATES, "S"), strata=("S",)
    )
    result = _typed(
        LearnedRuleContrast(against=(0, NaturalCourse())), False, frame=frame, design=design
    )
    assert set(result.parameter_keys) == set(result.estimates)
    assert any(key.stratum is not None for key in result.parameter_keys.values())


# ------------------------------------------------------------------------------- W-d3


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_w_d3_a_one_hot_density_is_the_known_rule(cv_evaluation: bool) -> None:
    density = _engine(cv_evaluation, interventions=[_one_hot()])
    rule = _engine(cv_evaluation, interventions=[_w1_rule()])
    assert float(np.min(np.abs(density.fluctuations["regime"].epsilon))) > 1e-3
    assert list(density.estimates) == list(rule.estimates)
    for name, estimate in rule.estimates.items():
        assert density.estimates[name].psi == estimate.psi, name
        np.testing.assert_array_equal(
            density.estimates[name].influence_curve, estimate.influence_curve
        )


# ------------------------------------------------------------------------------- W-d4


def _offer_curves(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The plug-ins and residual terms of the learned column and the offer, longhand."""
    pieces = support.validation_pieces(result)
    data = result.data
    lower, upper = result.config.g_bounds
    g1 = np.clip(np.asarray(result.nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    treatment = np.asarray(data.treatment, dtype=float)
    star = 0.5 * (np.asarray(data.covariates[:, 0]) > 0)
    targeted = result.fluctuations["regime"].targeted
    q1 = np.asarray(targeted.arms[1.0], dtype=float)
    q0 = np.asarray(targeted.arms[0.0], dtype=float)
    h_known = np.where(treatment == 1.0, star / g1, (1.0 - star) / (1.0 - g1))
    residual = pieces["y"] - pieces["q_star_observed"]
    return (
        pieces["q_star_rule"],
        pieces["h"] * residual,
        star * q1 + (1.0 - star) * q0,
        h_known * residual,
    )


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_w_d4_the_stochastic_contrast_is_the_longhand_difference(cv_evaluation: bool) -> None:
    result = _engine(cv_evaluation, interventions=[_offer()])
    name = "ate_learned_rule[learned rule vs offer]"
    plug_l, res_l, plug_k, res_k = _offer_curves(result)
    tests = [np.asarray(test) for _, test in result.nuisance.folds]
    n, v = plug_l.size, len(tests)
    estimate = result.estimates[name]
    curve = np.empty(n)
    if cv_evaluation:
        assert len({t.size for t in tests}) > 1
        point = np.mean([np.mean(plug_l[t] - plug_k[t]) for t in tests])
        for t in tests:
            raw = res_l[t] - res_k[t] + plug_l[t] - plug_k[t] - np.mean(plug_l[t] - plug_k[t])
            curve[t] = raw * n / (v * t.size)
        variance = sum(np.sum((curve[t] * v * t.size / n) ** 2) / t.size**2 for t in tests) / v**2
    else:
        point = np.mean(plug_l - plug_k)
        curve = res_l - res_k + plug_l - plug_k - point
        variance = np.sum(curve**2) / (n * (n - 1))
    assert estimate.psi == pytest.approx(point, abs=1e-12)
    np.testing.assert_allclose(estimate.influence_curve, curve, atol=1e-12, rtol=0)
    assert estimate.std_error**2 == pytest.approx(variance, rel=1e-9)


def test_w_d4_the_offer_score_is_solved_and_the_learned_column_alone_leaves_it() -> None:
    result = _engine(True, interventions=[_offer()])
    _, _, _, res_k = _offer_curves(result)
    # Solved to the stopping rule's tolerance: observed 8e-5, far below the unsolved control.
    assert abs(float(np.mean(res_k))) < 1e-3
    alone = _engine(True)
    pieces = support.validation_pieces(result)
    data = result.data
    lower, upper = result.config.g_bounds
    g1 = np.clip(np.asarray(result.nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    treatment = np.asarray(data.treatment, dtype=float)
    star = 0.5 * (np.asarray(data.covariates[:, 0]) > 0)
    h_known = np.where(treatment == 1.0, star / g1, (1.0 - star) / (1.0 - g1))
    unsolved = h_known * (pieces["y"] - np.asarray(alone.fluctuations["regime"].targeted.observed))
    assert abs(float(np.mean(unsolved))) > W_D4_UNSOLVED


def test_w_d4_control_rounding_the_offer_to_one_hot_moves_the_point() -> None:
    offer = _engine(True, interventions=[_offer("plan")])
    rounded = _engine(True, interventions=[_one_hot("plan")])
    name = "ate_learned_rule[learned rule vs plan]"
    assert abs(offer.psi(name) - rounded.psi(name)) > W_D4_ONE_HOT


# ------------------------------------------------------------------------------- W-d7


@cache
def _sign() -> tuple[Any, Any]:
    frame = support.law_frame(400, 0)
    common: dict[str, Any] = {
        "outcome_learner": SignStub(),
        "treatment_learner": support.treatment_learner(),
        "cross_fit": True,
        "cv_evaluation": True,
        "n_folds": 5,
        "simultaneous": False,
        "random_state": 0,
    }
    learned = (
        TMLE(learned_rule=LearnedRule(), interventions=[Static(0), _offer()], **common)
        .fit(frame, **FIT)
        .single()
    )
    regime = (
        TMLE(interventions=[_w1_rule(), Static(0), _offer()], reference="w1", **common)
        .fit(frame, **FIT)
        .single()
    )
    return learned, regime


@pytest.mark.parametrize("comparator", ["always 0", "offer"])
def test_w_d7_each_known_contrast_is_the_rule_minus_the_comparator(comparator: str) -> None:
    learned, regime = _sign()
    contrast = learned.psi(f"ate_learned_rule[learned rule vs {comparator}]")
    expected = regime.psi("ey_regime[w1]") - regime.psi(f"ey_regime[{comparator}]")
    assert contrast == pytest.approx(expected, abs=1e-14)
    assert contrast > 0.0


# ------------------------------------------------------------------------------- W-d8


def test_w_d8_the_optimal_contrast_reads_the_learned_arithmetic() -> None:
    learned = _typed(LearnedRuleContrast(against=(Static(0), NaturalCourse())), True)
    optimal = _typed(OptimalRuleContrast(against=(Static(0), NaturalCourse())), True)
    pairs = {
        "ey_optimal_rule[learned rule]": LEARNED,
        "ey_known_regime[always 0]": "ey_known_regime[always 0]",
        "ey_natural_course[natural course]": "ey_natural_course[natural course]",
        "ate_optimal_rule[learned rule vs always 0]": "ate_learned_rule[learned rule vs always 0]",
        "ate_optimal_rule[learned rule vs natural course]": (
            "ate_learned_rule[learned rule vs natural course]"
        ),
    }
    assert list(optimal.estimates) == list(pairs)
    for name, other in pairs.items():
        assert optimal.estimates[name].psi == learned.estimates[other].psi
        np.testing.assert_array_equal(
            optimal.estimates[name].influence_curve, learned.estimates[other].influence_curve
        )


# ------------------------------------------------------------------------------- W-d9


@pytest.mark.parametrize(
    "comparators",
    [(0,), (Static(1),), (_w1_rule(),)],
    ids=["level", "static", "rule"],
)
def test_w_d9_a_one_hot_comparator_reads_t44(comparators: tuple[Any, ...]) -> None:
    result = _engine(True, interventions=list(comparators))
    assert "known_stochastic_regimes" not in result.config.support_key.qualifiers
    rows = {e.support.row for e in result.estimates.values()}
    assert rows == {"T44.fold_evaluated"}


@pytest.mark.parametrize(
    "comparators",
    [(_offer(),), (Static(1), _offer()), (_Mixture(),), (_SubStatic(1, name="half"),)],
    ids=["stochastic", "beside_static", "user_density", "static_subclass"],
)
def test_w_d9_a_random_plan_reads_t58(comparators: tuple[Any, ...]) -> None:
    """A density that is not one-hot keys ``known_stochastic_regimes``, which only the T58 rows
    declare, so it never reads T44."""
    result = _engine(True, interventions=list(comparators))
    assert "known_stochastic_regimes" in result.config.support_key.qualifiers
    rows = {e.support.row for e in result.estimates.values()}
    assert rows == {"T58.stochastic.fold_evaluated"}


def test_w_d9_control_a_class_name_whitelist_lets_a_user_static_through() -> None:
    """The mutation reads ``type(item).__name__``; the shipped whitelist reads the type."""

    def by_name(items: Any) -> bool:
        return not all(type(item).__name__ in ("Static", "Rule") for item in items)

    user = _SubStatic(1, name="half")
    assert type(user).__name__ == "Static"
    assert not by_name([user])
    assert stochastic_comparators([user])
    assert not stochastic_comparators([Static(1), _w1_rule()])


# ------------------------------------------------------------------------------- W-d10


def test_r_d2_an_empty_against_refuses() -> None:
    with pytest.raises(DataError) as caught:
        LearnedRuleContrast(against=())
    assert str(caught.value) == (
        "LearnedRuleContrast(against=...) needs at least one comparator: a treatment level, a "
        "known regime or NaturalCourse(). For the value of the learned rule alone, request "
        "LearnedRuleValue()."
    )


def test_r_d5_a_second_learned_rule_refuses() -> None:
    with pytest.raises(CapabilityError) as caught:
        OptimalRuleContrast(against=(0, LearnedRule(name="other")))
    refusal = caught.value.refusal
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == "X11"
    assert str(refusal.condition).startswith(
        "OptimalRuleContrast.against holds a second learned rule"
    )
    assert str(refusal.alternative).startswith("Fit LearnedRuleValue(rule=...) for each rule")


@pytest.mark.parametrize("estimand", [RegimeMean, RegimeContrast])
def test_r_d1_a_learned_rule_among_regimes_redirects(estimand: Any) -> None:
    study = CausalStudy(support.law_frame(203, 1), design=DESIGN)
    with pytest.raises(CapabilityError) as caught:
        study.identify(estimand(regimens=(0, LearnedRule())))
    refusal = caught.value.refusal
    assert refusal.reason is RefusalReason.API_REDIRECTION
    assert str(refusal.alternative).startswith("LearnedRuleContrast(against=(...)) compares")


@pytest.mark.parametrize("estimand", [RegimeMean, RegimeContrast])
def test_r_d8a_a_natural_course_among_regimes_refuses(estimand: Any) -> None:
    study = CausalStudy(support.law_frame(203, 1), design=DESIGN)
    with pytest.raises(CapabilityError) as caught:
        study.identify(estimand(regimens=(0, NaturalCourse())))
    refusal = caught.value.refusal
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == "X11"
    assert f"{estimand.__name__}.regimens holds NaturalCourse()" in refusal.condition


@pytest.mark.parametrize(
    ("against", "error"),
    [({"a": 0}, DataError), (iter([0]), DataError), (0, DataError)],
    ids=["mapping", "iterator", "scalar"],
)
def test_r_d3_a_set_that_is_not_a_sequence_refuses(against: Any, error: type) -> None:
    study = CausalStudy(support.law_frame(203, 1), design=DESIGN)
    with pytest.raises(error, match=r"LearnedRuleContrast\.against must be a sequence"):
        study.identify(LearnedRuleContrast(against=against))


def test_r_d4_a_tilt_among_the_comparators_refuses() -> None:
    study = CausalStudy(support.law_frame(203, 1), design=DESIGN)
    with pytest.raises(CapabilityError):
        study.identify(LearnedRuleContrast(against=(0, Incremental(2.0))))


def test_r_d6_a_comparator_named_like_the_rule_refuses_before_any_learner() -> None:
    study = CausalStudy(support.law_frame(203, 1), design=DESIGN)
    identified = study.identify(LearnedRuleContrast(against=(NaturalCourse("learned rule"),)))
    method = TMLEMethod(cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True))
    with pytest.raises(CapabilityError) as caught:
        identified.estimate(method, **never_fit_learners())
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert NeverFit.calls == 0


def test_r_d7_a_natural_course_with_a_held_intermediate_refuses() -> None:
    frame = support.law_frame(203, 1).assign(Z=lambda f: (f["W2"] > 0).astype(int))
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=support.COVARIATES, intermediate="Z"
    )
    study = CausalStudy(frame, design=design)
    with pytest.raises(CapabilityError) as caught:
        study.identify(LearnedRuleContrast(against=(NaturalCourse(),), intermediate=1.0))
    refusal = caught.value.refusal
    assert refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
    assert refusal.condition == (
        "LearnedRuleContrast(intermediate=1) holds Z at 1 in the learned rule's value, and "
        "NaturalCourse() reads the observed course, in which Z is not held. Their difference "
        "mixes a controlled value with an uncontrolled one, which is not a contrast of two "
        "interventions."
    )


def test_a_known_regime_with_a_held_intermediate_is_admitted() -> None:
    frame = support.law_frame(203, 1).assign(Z=lambda f: (f["W2"] > 0).astype(int))
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=support.COVARIATES, intermediate="Z"
    )
    identified = CausalStudy(frame, design=design).identify(
        LearnedRuleContrast(against=(Static(0),), intermediate=1.0)
    )
    assert identified.functional.target == "ate_learned_rule"


def test_the_value_estimand_is_still_the_rule_alone() -> None:
    result = _typed(LearnedRuleValue(), True)
    assert list(result.estimates) == [LEARNED]
