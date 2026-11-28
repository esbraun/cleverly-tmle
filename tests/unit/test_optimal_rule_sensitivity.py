r"""The sensitivity analyses of the value of the optimal rule (X11 (f3), plan rev 2, PR 3).

**The omitted-variable bound.**  The strengths ``(rho, cf_y, cf_d)`` are read as limits that
hold for every regime measurable in ``W``, stochastic included.  The lower side is the
learned-rule point bound of the stochastic regime :math:`\bar d` that averages the fold rules,
:math:`\theta \ge \Psi_{\bar d} \ge \tilde\Psi_{\bar d} - B(\bar d)`, and the upper side adds
the bound over every rule, :math:`\theta \le \theta_s + B_{\sup}` with
:math:`\nu^2_{\sup} = E[\max_a 1/g(a \mid W)]`.

* :class:`TestTheBoundOnExactLaws` is W-ovb1 and W-ovb2: two exact discrete laws with an
  omitted binary ``U`` and fold rules that differ from the short-optimal rule at one level.
  The asymmetric bound holds on both.  On the first, the long-optimal rule differs from the
  short one and the symmetric bound with the learned-rule scale misses :math:`\theta`; on the
  second, a lower side centred at :math:`\theta_s` misses it.  One law cannot carry both
  mutations: the first needs :math:`\theta > \theta_s` and the second :math:`\theta < \theta_s`.
* :class:`TestTheLowerSideIsTheLearnedRuleBound` is W-ovb3, bit for bit.
* :class:`TestEachSideHasItsRobustnessValue` is W-ovb4.
* :class:`TestTheUpperScaleReadsTheFitsMechanism` checks each component of
  :math:`\hat\nu^2_{\sup}` (weights, truncation, missingness) against a wrong variant.
* :class:`TestTheBoundRefusals` and :class:`TestTheBoundReports` pin the refusals, the
  withheld limits, the summaries and the plots.

**The relearning replay.**  Every cell of ``simulated_confounding`` refits the stored
estimator, which relearns the rule, on the fit's folds and repeat seeds, with the known mechanism
of the flipped treatment.  :class:`TestTheRelearningReplay` is W-sc1 and W-sc2.
"""

from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import cleverly.sensitivity.omitted_variable as ov
from cleverly import CausalStudy, OptimalRuleContrast, OptimalRuleValue, PointTreatment
from cleverly.estimators import TMLE
from cleverly.exceptions import (
    CapabilityError,
    InSampleLearnerWarning,
    PositivityWarning,
    RefusalReason,
)
from cleverly.interventions import LearnedRule, NaturalCourse, Rule, Static, Stochastic
from cleverly.interventions.learned import (
    OPTIMAL_RULE_BENCHMARK_REFUSAL,
    OPTIMAL_RULE_DOUBLY_ROBUST_NU2,
    OPTIMAL_RULE_LIMITS,
    OPTIMAL_RULE_TARGET,
    fold_rules,
    freeze_learned_rule,
    frozen_rule_replay,
)
from cleverly.plotting import confounding, sensitivity
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_request as request_module
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.simulated_confounding import flipped_mechanism
from cleverly.utils.bounds import g_bounds_for
from tests.unit import _learned_rule_support as support
from tests.unit._confounding_support import forbid_draw_and_refit
from tests.unit._confounding_support import replacement as _replacement
from tests.unit.test_learned_rule_bound import REPLICATES, ParityRule
from tests.unit.test_learned_rule_optimal import _pairs

matplotlib.use("Agg")

pytestmark = pytest.mark.xdist_group("optimal_rule_sensitivity")

OPTIMAL = "ey_optimal_rule[learned rule]"


# ------------------------------------------------------------------------ the exact laws


@dataclass(frozen=True)
class Law:
    """A discrete law of ``(W, U, A, Y)`` with ``W`` in three levels and an omitted binary ``U``.

    ``g_long[w, u]`` is ``g_L(1 | w, u)`` and ``q_long[a, w, u]`` is ``Qbar_L(a, w, u)``.
    """

    p_w: tuple[float, ...]
    p_u: tuple[float, ...]
    g_long: tuple[tuple[float, ...], ...]
    q_long: tuple[tuple[tuple[float, ...], ...], ...]

    def joint(self) -> np.ndarray:
        """``P(W = w, U = u)``, ``(3, 2)``."""
        p_u = np.asarray(self.p_u)
        return np.asarray(self.p_w)[:, None] * np.column_stack([1.0 - p_u, p_u])

    def g_l(self, a: int) -> np.ndarray:
        g = np.asarray(self.g_long)
        return g if a == 1 else 1.0 - g

    def g_s(self, a: int) -> np.ndarray:
        """``g_S(a | w)``, ``(3,)``."""
        return np.sum(self.joint() * self.g_l(a), axis=1) / np.asarray(self.p_w)

    def q_l(self, a: int) -> np.ndarray:
        return np.asarray(self.q_long)[a]

    def q_s(self, a: int) -> np.ndarray:
        """``Qbar_S(a, w)``, ``(3,)``."""
        mass = self.joint() * self.g_l(a)
        return np.sum(mass * self.q_l(a), axis=1) / np.sum(mass, axis=1)

    def long_value(self, regime: dict[int, np.ndarray]) -> float:
        return sum(float(np.sum(self.joint() * regime[a][:, None] * self.q_l(a))) for a in (0, 1))

    def short_value(self, regime: dict[int, np.ndarray]) -> float:
        p_w = np.asarray(self.p_w)
        return sum(float(np.sum(p_w * regime[a] * self.q_s(a))) for a in (0, 1))

    def moments(self, regime: dict[int, np.ndarray]) -> dict[str, float]:
        """``E[dQ dalpha]``, ``E[dQ^2]``, ``E[alpha_L^2]`` and ``E[alpha_S^2]`` of a regime."""
        out = {"product": 0.0, "dq2": 0.0, "long2": 0.0, "short2": 0.0}
        joint = self.joint()
        for a, w, u in itertools.product(range(2), range(3), range(2)):
            mass = joint[w, u] * self.g_l(a)[w, u]
            dq = self.q_l(a)[w, u] - self.q_s(a)[w]
            alpha_long = regime[a][w] / self.g_l(a)[w, u]
            alpha_short = regime[a][w] / self.g_s(a)[w]
            out["product"] += mass * dq * (alpha_long - alpha_short)
            out["dq2"] += mass * dq**2
            out["long2"] += mass * alpha_long**2
            out["short2"] += mass * alpha_short**2
        return out

    def sigma2(self) -> float:
        """``E[(Y - Qbar_S(A, W))^2]``, the short residual variance."""
        residual = sum(
            float(np.sum(self.joint() * self.g_l(a) * self.q_l(a) * (1.0 - self.q_l(a))))
            for a in (0, 1)
        )
        return residual + self.moments(_vertex((0, 0, 0)))["dq2"]

    def nu2_sup(self) -> float:
        """``E[max_a 1/g_S(a | W)]`` by enumeration."""
        return float(np.sum(np.asarray(self.p_w) * np.maximum(1 / self.g_s(0), 1 / self.g_s(1))))


def _vertex(rule: tuple[int, ...]) -> dict[int, np.ndarray]:
    """The deterministic regime ``d(w)`` as a density over the two arms."""
    d = np.asarray(rule, dtype=float)
    return {1: d, 0: 1.0 - d}


#: Every deterministic rule of ``W``, the vertices of the regime class.
VERTICES = tuple(itertools.product((0, 1), repeat=3))

#: W-ovb1, upper mutation: the long-optimal rule (never treat) differs from the short-optimal
#: rule (treat at ``W = 0`` only), and :math:`\theta` exceeds the symmetric bound.
UPPER_LAW = Law(
    p_w=(0.6, 0.2, 0.2),
    p_u=(0.8, 0.9, 0.8),
    g_long=((0.4, 0.9), (0.2, 0.3), (0.2, 0.2)),
    q_long=(
        ((0.16, 0.86), (0.47, 0.65), (0.13, 0.91)),
        ((0.58, 0.45), (0.87, 0.32), (0.38, 0.82)),
    ),
)

#: W-ovb1, lower mutation: :math:`\theta` lies below :math:`\theta_s` by more than the
#: learned-rule bound of the fold-average regime.
LOWER_LAW = Law(
    p_w=(0.2, 0.7, 0.1),
    p_u=(0.4, 0.4, 0.8),
    g_long=((0.7, 0.9), (0.1, 0.7), (0.6, 0.7)),
    q_long=(
        ((0.51, 0.7), (0.24, 0.19), (0.74, 0.61)),
        ((0.42, 0.8), (0.45, 0.91), (0.37, 0.67)),
    ),
)

LAWS = {"upper": UPPER_LAW, "lower": LOWER_LAW}


class LawTreatment(ClassifierMixin, BaseEstimator):
    """The short treatment mechanism ``g_S(1 | W)`` of one law, whatever the rows."""

    def __init__(self, law: str = "upper") -> None:
        self.law = law

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LawTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        p = LAWS[self.law].g_s(1)[w]
        return np.column_stack([1.0 - p, p])


def _law_frame(law: Law) -> pd.DataFrame:
    """One weighted row per replicate of each support point of the short law of ``(W, A, Y)``.

    As in ``tests/unit/test_learned_rule_bound.py``, one point has a replicate fewer, so the
    two folds have 23 and 24 rows and :class:`ParityRule` learns rules that differ at
    ``W = 1``.
    """
    rows = []
    for w, a, y in itertools.product(range(3), range(2), range(2)):
        q = law.q_s(a)[w]
        p = law.p_w[w] * law.g_s(a)[w] * (q if y else 1.0 - q)
        copies = REPLICATES - 1 if (w, a, y) == (0, 0, 0) else REPLICATES
        rows.extend(
            {"W": float(w), "A": float(a), "Y": float(y), "p": p / copies} for _ in range(copies)
        )
    return pd.DataFrame(rows)


@cache
def _law_fit(name: str) -> Any:
    return (
        TMLE(
            learned_rule=LearnedRule(),
            estimands=[OPTIMAL_RULE_TARGET],
            outcome_learner=ParityRule(),
            treatment_learner=LawTreatment(name),
            n_folds=2,
            simultaneous=False,
            random_state=0,
            g_bounds=(0.01, 0.99),
            **support.FOLD_EVALUATED,
        )
        .fit(_law_frame(LAWS[name]), outcome="Y", treatment="A", covariates=["W"], weights="p")
        .single()
    )


def _rule_table(result: Any) -> np.ndarray:
    """``d_v(w)`` as a ``(3, V)`` table, one value per level of ``W``."""
    rules = fold_rules(result.repeats[0].nuisance)
    assert rules is not None
    w = np.rint(result.data.covariates[:, 0]).astype(int)
    table = np.array([rules[w == level][0] for level in range(3)])
    for level in range(3):
        assert np.all(rules[w == level] == table[level])
    return table.astype(float)


def _fold_average(result: Any) -> dict[int, np.ndarray]:
    """The stochastic regime :math:`\\bar d` that averages the fold rules."""
    star = np.mean(_rule_table(result), axis=1)
    return {1: star, 0: 1.0 - star}


@dataclass(frozen=True)
class _Reading:
    """The enumerated targets, the uniform strengths and the scales of one law."""

    theta: float
    theta_s: float
    psi_bar: float
    rule_long: tuple[int, ...]
    rule_short: tuple[int, ...]
    strength: float
    sigma: float
    nu_bar: float
    nu_short_rule: float
    nu_sup: float


def _reading(name: str) -> _Reading:
    """Enumerate one law.  The uniform strengths are maxima over the regime class.

    The class holds the eight deterministic rules and the fold-average regime, so each of
    its regimes meets its own bound at the strengths.  ``nu`` of the two sides is read off
    the fit, which W-ovb2 checks against the law.
    """
    law, result = LAWS[name], _law_fit(name)
    bar = _fold_average(result)
    regimes = [_vertex(rule) for rule in VERTICES] + [bar]
    moments = [law.moments(regime) for regime in regimes]
    sigma2 = law.sigma2()
    cf_y = moments[0]["dq2"] / sigma2
    cf_d = max(1.0 - m["short2"] / m["long2"] for m in moments)
    rho = max(abs(m["product"]) / np.sqrt(m["dq2"] * (m["long2"] - m["short2"])) for m in moments)
    assert 0.0 < cf_y < 1.0 and 0.0 < cf_d < 1.0 and 0.0 < rho < 1.0
    longs = [law.long_value(_vertex(rule)) for rule in VERTICES]
    shorts = [law.short_value(_vertex(rule)) for rule in VERTICES]
    short_rule = VERTICES[int(np.argmax(shorts))]
    elements = ov.sensitivity_elements(result, OPTIMAL, nu2_estimator="plugin")
    assert elements.nu2_upper is not None
    return _Reading(
        theta=max(longs),
        theta_s=max(shorts),
        psi_bar=law.short_value(bar),
        rule_long=VERTICES[int(np.argmax(longs))],
        rule_short=short_rule,
        strength=ov._confounding_strength(cf_y, cf_d, float(rho)),
        sigma=float(np.sqrt(sigma2)),
        nu_bar=float(np.sqrt(elements.nu2)),
        nu_short_rule=float(np.sqrt(law.moments(_vertex(short_rule))["short2"])),
        nu_sup=float(np.sqrt(elements.nu2_upper)),
    )


class TestTheBoundOnExactLaws:
    """W-ovb1 and W-ovb2 on two exact laws."""

    @pytest.mark.parametrize("name", list(LAWS))
    def test_the_fold_rules_differ_from_the_short_optimal_rule_at_one_level(
        self, name: str
    ) -> None:
        table = _rule_table(_law_fit(name))
        short = np.asarray(_reading(name).rule_short, dtype=float)
        assert short[0] == 1.0 and short[2] == 0.0
        np.testing.assert_array_equal(table[0], [1.0, 1.0])
        np.testing.assert_array_equal(table[2], [0.0, 0.0])
        assert sorted(table[1]) == [0.0, 1.0]
        # One fold's rule disagrees with the short-optimal rule at W = 1, the other agrees.
        assert np.sum(table[1] != short[1]) == 1

    @pytest.mark.parametrize("name", list(LAWS))
    def test_nu2_sup_is_the_mean_of_the_largest_inverse_probability(self, name: str) -> None:
        """W-ovb2, with its mutation: the value at the fitted rules is far below the maximum."""
        law, result = LAWS[name], _law_fit(name)
        for estimator in ("auto", "plugin"):
            elements = ov.sensitivity_elements(result, OPTIMAL, nu2_estimator=estimator)
            assert elements.nu2_upper == pytest.approx(law.nu2_sup(), rel=1e-12)
            assert elements.max_bias_upper == pytest.approx(
                np.sqrt(elements.sigma2 * law.nu2_sup()), rel=1e-12
            )
        table = _rule_table(result)
        p_w = np.asarray(law.p_w)
        at_fitted = np.mean(
            [
                np.sum(p_w * np.where(table[:, v] == 1.0, 1 / law.g_s(1), 1 / law.g_s(0)))
                for v in range(table.shape[1])
            ]
        )
        assert law.nu2_sup() - at_fitted > 0.5

    @pytest.mark.parametrize("name", list(LAWS))
    def test_the_optimal_value_lies_in_the_asymmetric_bound(self, name: str) -> None:
        r = _reading(name)
        lower = r.psi_bar - r.strength * r.sigma * r.nu_bar
        upper = r.theta_s + r.strength * r.sigma * r.nu_sup
        assert lower <= r.theta <= upper
        # The omitted U moves the optimal value: theta and theta_s differ.
        assert abs(r.theta - r.theta_s) > 0.1

    def test_a_symmetric_bound_with_the_learned_rule_scale_misses_the_optimal_value(self) -> None:
        """Upper mutation: the long-optimal rule is not the short one, and its gain is larger."""
        r = _reading("upper")
        assert r.rule_long != r.rule_short
        symmetric = r.theta_s + r.strength * r.sigma * max(r.nu_bar, r.nu_short_rule)
        assert r.theta - symmetric > 0.04

    def test_a_lower_side_centred_at_the_short_optimum_misses_the_optimal_value(self) -> None:
        """Lower mutation: the fit estimates the fold-average value, not theta_s."""
        r = _reading("lower")
        assert r.psi_bar < r.theta_s - 0.2
        assert (r.theta_s - r.strength * r.sigma * r.nu_bar) - r.theta > 0.025

    @pytest.mark.parametrize("name", list(LAWS))
    def test_the_fit_s_bound_reads_the_two_scales(self, name: str) -> None:
        result = _law_fit(name)
        bounds = ov.omitted_variable_bounds(result, OPTIMAL, cf_y=0.04, cf_d=0.03, rho=0.8)
        elements = ov.sensitivity_elements(result, OPTIMAL)
        strength = bounds.confounding_strength
        assert bounds.psi - bounds.lower == pytest.approx(strength * elements.max_bias, rel=1e-12)
        assert bounds.upper - bounds.psi == pytest.approx(
            strength * elements.max_bias_upper, rel=1e-12
        )
        assert bounds.upper - bounds.psi > bounds.psi - bounds.lower


# --------------------------------------------------------------------------- W-ovb3


def _with_known_g(frame: pd.DataFrame) -> dict[str, Any]:
    g1 = expit(0.3 * frame["W1"].to_numpy() - 0.2 * frame["W2"].to_numpy())
    return {"treatment_probabilities": np.column_stack([1.0 - g1, g1])}


#: Each pair: the frame, the fit keywords, and the estimator keywords.
PAIRS: dict[str, tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]] = {
    "rows": (support.law_frame(), {}, {}),
    "known_g": (support.law_frame(), _with_known_g(support.law_frame()), {}),
    "weights": (
        support.law_frame().assign(w=lambda f: 1.0 + 1.5 * (f["W1"] > 0)),
        {"weights": "w"},
        {},
    ),
    "in_sample": (support.law_frame(), {}, {"cross_fit": False, "cv_evaluation": False}),
}


@cache
def _pair(name: str) -> tuple[Any, Any]:
    frame, fit, settings = PAIRS[name]
    out = []
    for estimands in (None, [OPTIMAL_RULE_TARGET]):
        tmle = support.learned_tmle(
            **settings, **({} if estimands is None else {"estimands": estimands})
        )
        with warnings.catch_warnings():
            # The in-sample fit warns about its adaptive learner (condition C5a).
            warnings.simplefilter("ignore", InSampleLearnerWarning)
            fitted = tmle.fit(
                frame, outcome="Y", treatment="A", covariates=support.COVARIATES, **fit
            )
        out.append(fitted.single())
    return out[0], out[1]


class TestTheLowerSideIsTheLearnedRuleBound:
    """W-ovb3: the lower side, its scale and its robustness value are the learned rule's."""

    @pytest.mark.parametrize("name", list(PAIRS))
    @pytest.mark.parametrize("estimator", ["auto", "plugin"])
    def test_bit_for_bit(self, name: str, estimator: str) -> None:
        learned, optimal = _pair(name)
        assert learned[support.NAME].psi == optimal[OPTIMAL].psi
        a = ov.sensitivity_elements(learned, support.NAME, nu2_estimator=estimator)
        b = ov.sensitivity_elements(optimal, OPTIMAL, nu2_estimator=estimator)
        assert (a.sigma2, a.nu2, a.max_bias) == (b.sigma2, b.nu2, b.max_bias)
        np.testing.assert_array_equal(a.riesz_representer, b.riesz_representer)
        assert a.features == b.features == ("learned_rule",)
        assert a.nu2_upper is None and b.nu2_upper is not None
        kwargs = {"cf_y": 0.05, "cf_d": 0.04, "rho": 0.9, "nu2_estimator": estimator}
        x = ov.omitted_variable_bounds(learned, support.NAME, **kwargs)
        y = ov.omitted_variable_bounds(optimal, OPTIMAL, **kwargs)
        assert x.lower == y.lower
        assert x.upper < y.upper
        below = float(optimal[OPTIMAL].psi) - 0.05
        assert (
            ov.robustness_value(
                learned, support.NAME, null_hypothesis=below, nu2_estimator=estimator
            )["rv"]
            == ov.robustness_value(
                optimal, OPTIMAL, null_hypothesis=below, nu2_estimator=estimator
            )["rv"]
        )

    def test_the_in_sample_optimal_bound_withholds_the_limits_the_learned_bound_publishes(
        self,
    ) -> None:
        learned, optimal = _pair("in_sample")
        assert np.isfinite(ov.omitted_variable_bounds(learned, support.NAME).ci_lower)
        bounds = ov.omitted_variable_bounds(optimal, OPTIMAL)
        with pytest.raises(CapabilityError, match=re.escape("nu^2_sup = E[max_a 1/g(a | W)]")):
            _ = bounds.ci_lower


# --------------------------------------------------------------------------- W-ovb4


def _direct_nu2_sup(result: Any, *, weighted: bool = True, bounded: bool = True) -> float:
    r""":math:`E_n[\max_a 1/g(a \mid W)]` from the fit's propensity, or a wrong variant."""
    nuisance = result.repeats[0].nuisance
    config = result.config
    bounds = g_bounds_for("regime", config.g_bounds, config.g_bounds_conditional)
    g = nuisance.bounded_propensity(bounds if bounded else (1e-12, 1.0 - 1e-12))
    largest = (1.0 / g).max(axis=1)
    return float(np.average(largest, weights=result.data.weights if weighted else None))


def _missing_outcome_fit() -> Any:
    """An in-sample known-rule fit with missing outcomes, whose regime submodel reads ``pR``.

    Every learned-rule fit refuses missing outcomes (F27), so the missingness factor of
    :func:`ov._sup_rule_nu2` is checked on the regime fit whose submodel it shares.
    """
    frame = support.law_frame(400, 2)
    draw = np.random.default_rng(5).uniform(size=len(frame))
    missing = draw < expit(-1.2 + 0.8 * frame["W1"] + 0.5 * frame["A"])
    frame.loc[missing, "Y"] = np.nan
    frame["R"] = (~missing).astype(float)
    rule = Rule(lambda w: (w["W1"] > 0).astype(int), name="w1", rule_kind="known")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InSampleLearnerWarning)
        return (
            TMLE(
                interventions=[rule],
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                cross_fit=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, delta="R")
            .single()
        )


class TestTheUpperScaleReadsTheFitsMechanism:
    """``nu2_upper`` is the weighted mean of ``max_a 1/(g pR)`` under the bounded mechanism.

    Each witness has a nonzero gap from the variant that drops one component.
    """

    def test_a_weighted_fit_takes_the_weighted_mean(self) -> None:
        result = _pair("weights")[1]
        elements = ov.sensitivity_elements(result, OPTIMAL)
        direct = _direct_nu2_sup(result)
        assert elements.nu2_upper == pytest.approx(direct, rel=1e-12)
        assert abs(_direct_nu2_sup(result, weighted=False) - direct) > 0.01

    def test_a_binding_truncation_bounds_the_mechanism(self) -> None:
        with pytest.warns(PositivityWarning, match="outside the truncation bounds"):
            result = support.fit(estimands=[OPTIMAL_RULE_TARGET], g_bounds=(0.42, 0.58))
        elements = ov.sensitivity_elements(result, OPTIMAL)
        direct = _direct_nu2_sup(result)
        assert elements.nu2_upper == pytest.approx(direct, rel=1e-12)
        assert abs(_direct_nu2_sup(result, bounded=False) - direct) > 0.01

    def test_missing_outcomes_divide_by_the_recording_probability(self) -> None:
        result = _missing_outcome_fit()
        nuisance, config = result.repeats[0].nuisance, result.config
        bounds = g_bounds_for("regime", config.g_bounds, config.g_bounds_conditional)
        value = ov._sup_rule_nu2(
            result.data,
            nuisance,
            "regime",
            bounds,
            None,
            config.reference_arm,
            result.data.weights,
            config.missingness_bound,
        )
        g = nuisance.bounded_propensity(bounds)
        recorded = nuisance.bounded_missingness(config.missingness_bound)
        assert recorded is not None
        direct = float(np.average((1.0 / (g * recorded)).max(axis=1), weights=result.data.weights))
        assert value == pytest.approx(direct, rel=1e-12)
        without = float(np.average((1.0 / g).max(axis=1), weights=result.data.weights))
        assert direct - without > 0.5


class TestEachSideHasItsRobustnessValue:
    """W-ovb4: each side reaches the null at its own strength, and the two differ."""

    def test_each_value_moves_its_side_to_the_null(self) -> None:
        _, result = _pair("known_g")
        psi = float(result[OPTIMAL].psi)
        distance = 0.05
        down = ov.robustness_value(result, OPTIMAL, null_hypothesis=psi - distance)["rv"]
        up = ov.robustness_value(result, OPTIMAL, null_hypothesis=psi + distance)["rv"]
        assert down is not None and up is not None
        # nu_sup > nu_bar on a nonconstant g, so the upper side reaches the null sooner.
        elements = ov.sensitivity_elements(result, OPTIMAL)
        assert elements.max_bias_upper > elements.max_bias * 1.05
        assert down - up > 0.005
        at_down = ov.omitted_variable_bounds(
            result, OPTIMAL, cf_y=down, cf_d=down, null_hypothesis=psi - distance
        )
        at_up = ov.omitted_variable_bounds(
            result, OPTIMAL, cf_y=up, cf_d=up, null_hypothesis=psi + distance
        )
        assert at_down.lower == pytest.approx(psi - distance, abs=1e-9)
        assert at_up.upper == pytest.approx(psi + distance, abs=1e-9)
        assert at_down.robustness_value == down and at_up.robustness_value == up
        # The mutation: the lower side's value on the upper side stops short of the null.
        assert at_down.upper > psi + distance + 1e-3


# ------------------------------------------------------------------- refusals and reports


class TestTheBoundRefusals:
    """The refusals of the bound of the optimal value, each pinned by its text and roadmap."""

    def test_the_doubly_robust_nu2_refuses(self) -> None:
        _, result = _pair("rows")
        calls = (
            lambda: ov.sensitivity_elements(result, OPTIMAL, nu2_estimator="doubly_robust"),
            lambda: ov.omitted_variable_bounds(result, OPTIMAL, nu2_estimator="doubly_robust"),
            lambda: ov.robustness_value(result, OPTIMAL, nu2_estimator="doubly_robust"),
            lambda: ov.contour_data(result, OPTIMAL, nu2_estimator="doubly_robust"),
        )
        for call in calls:
            with pytest.raises(CapabilityError) as caught:
                call()
            assert caught.value.refusal.reason is RefusalReason.INFERENCE_GAP
            assert OPTIMAL_RULE_DOUBLY_ROBUST_NU2.condition in str(caught.value)
            assert "X11 (n2) in docs/roadmap.md" in str(caught.value)

    @pytest.mark.parametrize("name", list(PAIRS))
    def test_every_limit_refuses(self, name: str) -> None:
        _, result = _pair(name)
        bounds = ov.omitted_variable_bounds(result, OPTIMAL)
        assert bounds.limit_refusal is not None and bounds.limit_refusal.key == "optimal_rule"
        assert bounds.limit_refusal.text == OPTIMAL_RULE_LIMITS
        for accessor in ("ci_lower", "ci_upper", "robustness_value_ci", "plugin_interval_lower"):
            with pytest.raises(CapabilityError, match="nu\\^2_sup") as caught:
                getattr(bounds, accessor)
            assert caught.value.refusal.roadmap == "X11"
        assert "rva" not in ov.robustness_value(result, OPTIMAL)

    def test_benchmark_refuses(self) -> None:
        _, result = _pair("rows")
        assert ov.benchmark_refusal(result, ["W2"]) is OPTIMAL_RULE_BENCHMARK_REFUSAL
        row = result.sensitivity.capability("benchmark")
        assert not row.available
        assert "benchmark of the value of the optimal rule is not built" in (row.reason or "")

    #: The (n2) settings that keep a refusal, with the phrase it names.
    SETTINGS: ClassVar[dict[str, str]] = {
        "repeats": "the value of the optimal rule over 3 splits (repeats=3) is not built",
        "three_arms": "the value of the optimal rule over 3 arms is not built",
        "stacked": "under the stacked report (cv_evaluation=False) is not built",
    }

    @pytest.mark.parametrize("setting", list(SETTINGS))
    def test_the_learned_rule_n2_settings_keep_their_reason(self, setting: str) -> None:
        learned, optimal = _pairs(setting)
        refusal = ov.fit_wide_bound_refusal(optimal)
        inner = ov.fit_wide_bound_refusal(learned)
        assert refusal is not None and inner is not None
        assert self.SETTINGS[setting] in refusal.condition
        assert "its upper limit adds the bound over every rule" in refusal.condition
        # The learned-rule reason follows, capitalized as its own sentence.
        assert inner.condition[1:] in refusal.condition
        assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP and refusal.roadmap == "X11"
        for operation in ("omitted_confounding", "robustness_value", "contour", "elements"):
            assert not optimal.sensitivity.capability(operation).available, operation


class TestTheBoundReports:
    """Every report of the bound states its asymmetric reading, and the plots draw it."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return _pair("known_g")[1]

    def test_the_summaries_name_the_asymmetric_reading(self, result: Any) -> None:
        bounds = ov.omitted_variable_bounds(result, OPTIMAL)
        codes = [n.code for n in bounds.summary().notes]
        assert "optimal_rule_bound" in codes and "learned_rule_bound" not in codes
        withheld = next(n for n in bounds.summary().notes if n.code == "limits_withheld")
        assert "best rule is a point bound" in withheld.text
        settings = bounds.summary(detail="full").section("Settings").lines
        assert any("nu^2_sup = E[max_a 1/g(a | W)]" in line for line in settings)
        value = ov.robustness_value(result, OPTIMAL)
        assert "optimal_rule_bound" in [n.code for n in value.summary().notes]
        # The default null lies below the estimate, so the value is solved on the lower side.
        assert value["side"] == "lower" and value["max_bias"] == bounds.max_bias
        elements = ov.sensitivity_elements(result, OPTIMAL)
        labels = [row[0].text for row in elements.summary().tables[0].rows]
        assert "bias scale of the upper side" in labels
        assert bounds.to_dict()["max_bias_upper"] == bounds.max_bias_upper
        assert bounds.bias == bounds.confounding_strength * bounds.max_bias_upper

    @pytest.mark.parametrize("bound", ["lower", "upper"])
    def test_the_contour_reads_the_side_s_scale_and_draws(self, result: Any, bound: str) -> None:
        contour = ov.contour_data(result, OPTIMAL, bound=bound, grid_size=4)
        elements = ov.sensitivity_elements(result, OPTIMAL)
        scale = elements.max_bias if bound == "lower" else elements.max_bias_upper
        assert contour.max_bias == scale
        frame = contour.to_frame()
        last = frame.iloc[-1]
        strength = ov._confounding_strength(float(last["cf_y"]), float(last["cf_d"]), 1.0)
        sign = -1.0 if bound == "lower" else 1.0
        expected = float(result[OPTIMAL].psi) + sign * strength * scale
        assert float(last["value"]) == pytest.approx(expected, rel=1e-12)
        # A level has no no-effect value, so the bold contour reads a declared null on the
        # side the bound moves toward.
        figure = confounding(contour, null=0.5 if bound == "lower" else 0.7)
        assert figure.axes

    def test_the_upper_robustness_value_draws_on_the_upper_contour(self, result: Any) -> None:
        """A null above the estimate solves the upper side, with that side's scale and labels."""
        elements = ov.sensitivity_elements(result, OPTIMAL)
        assert elements.max_bias_upper is not None
        assert elements.max_bias_upper > elements.max_bias + 0.01
        null = float(result[OPTIMAL].psi) + 0.05
        value = ov.robustness_value(result, OPTIMAL, null_hypothesis=null)
        assert value["side"] == "upper"
        assert value["max_bias"] == elements.max_bias_upper
        assert value["nu2_estimator"] == "plugin" and value["nu2_correction"] is None
        settings = value.summary(detail="full").section("Settings").lines
        assert any(line.startswith("Bias scale of the upper side") for line in settings)
        assert not any(line.startswith("Bias scale sqrt(sigma^2 nu^2):") for line in settings)
        upper = ov.contour_data(result, OPTIMAL, bound="upper", grid_size=4)
        assert confounding(upper, robustness=value, null=null).axes
        # The mutation: the lower contour, whose scale the value was not solved on, refuses.
        lower = ov.contour_data(result, OPTIMAL, bound="lower", grid_size=4)
        with pytest.raises(ValueError, match=re.escape("bound='upper'")):
            confounding(lower, robustness=value, null=null)

    def test_sensitivity_refuses_the_bound_with_a_reason(self, result: Any) -> None:
        with pytest.raises(TypeError):
            sensitivity(ov.omitted_variable_bounds(result, OPTIMAL))


# ------------------------------------------------------------------ the relearning replay

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2))


def _study(frame: pd.DataFrame, **design: Any) -> CausalStudy:
    g1 = expit(0.3 * frame["W1"].to_numpy() - 0.2 * frame["W2"].to_numpy())
    frame = frame.assign(p0=1.0 - g1, p1=g1)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=support.COVARIATES,
            treatment_probabilities={0: "p0", 1: "p1"},
            **design,
        ),
    )


@cache
def _replay_fit(repeats: int = 1) -> Any:
    return (
        _study(support.law_frame(n=403))
        .identify(OptimalRuleValue())
        .estimate(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            n_folds=5,
            cv_evaluation=True,
            simultaneous=False,
            random_state=0,
            repeats=repeats,
        )
    )


@cache
def _surface(repeats: int = 1) -> Any:
    return simulated_confounding(_replay_fit(repeats), estimand=OPTIMAL, grid=_GRID)


def _perturbed(result: Any, surface: Any, cell: Any, *, rebuild: bool = True) -> Any:
    """The cell's table, with the flipped known mechanism or, as a mutation, the original."""
    data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
    if data.known_treatment is not None:
        # A cell that leaves the treatment keeps its declared mechanism.
        assert cell.treatment_strength == 0.0
        return data
    declared = result.data.known_treatment.values
    strength = cell.treatment_strength if rebuild else 0.0
    return data.with_known_mechanism(flipped_mechanism(declared, strength))


def _relearned(result: Any, data: Any) -> Any:
    folds = [repeat.nuisance.folds for repeat in result.repeats]
    return result.estimator.refit(data, random_state=0, frozen_folds=folds)


class TestTheRelearningReplay:
    """W-sc1 and W-sc2, and the replay's refusals."""

    def test_the_anchor_reproduces_the_fit_bit_for_bit(self) -> None:
        """W-sc1, first half."""
        result, surface = _replay_fit(), _surface()
        assert result.sensitivity.capability("simulated_confounding").available
        assert surface.complete and surface.refit_seed == 0
        assert surface.known_mechanism_carried
        assert surface.cells[0].estimate == result[OPTIMAL].psi
        assert _relearned(result, result.data)[OPTIMAL].psi == result[OPTIMAL].psi

    def test_a_perturbed_cell_relearns_the_rules(self) -> None:
        """W-sc1, second half: the cell's fold rules differ from the fit's on a share of entries."""
        result, surface = _replay_fit(), _surface()
        cell = surface.cells[-1]
        refit = _relearned(result, _perturbed(result, surface, cell))
        before = fold_rules(result.repeats[0].nuisance)
        after = fold_rules(refit.repeats[0].nuisance)
        assert before is not None and after is not None
        assert float(np.mean(before != after)) > 0.02
        assert cell.estimate == pytest.approx(refit[OPTIMAL].psi, abs=1e-12)

    @pytest.mark.parametrize("repeats", [1, 2])
    def test_every_cell_is_the_relearning_refit_of_its_table(self, repeats: int) -> None:
        """W-sc2, with the mutation: the original mechanism moves a flipped cell."""
        result, surface = _replay_fit(repeats), _surface(repeats)
        assert result.n_repeats == repeats
        for cell in surface.cells[1:]:
            manual = _relearned(result, _perturbed(result, surface, cell))
            assert cell.estimate == pytest.approx(manual[OPTIMAL].psi, abs=1e-12)
        flipped = next(cell for cell in surface.cells if cell.treatment_strength > 0.0)
        stale = _relearned(result, _perturbed(result, surface, flipped, rebuild=False))
        assert abs(stale[OPTIMAL].psi - flipped.estimate) > 5e-4
        assert max(abs(cell.displacement) for cell in surface.cells) > 5e-3

    def test_an_unseeded_stochastic_fit_replays_with_its_own_seeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cell's refit, run on the fit's own table, reproduces the stored estimate exactly.

        The surface reports the stored estimate at its zero cell without a refit, so the
        witness reruns a cell's refit, with the keywords the surface passed, on the unperturbed
        table.  An unseeded random forest draws new trees unless the replay keeps the fit's
        repeat seeds.  The mutation, the same refit without them, moves the estimate.
        """
        result = (
            _study(support.law_frame(n=403))
            .identify(OptimalRuleValue())
            .estimate(
                outcome_learner=RandomForestClassifier(n_estimators=20, min_samples_leaf=20),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                cv_evaluation=True,
                simultaneous=False,
            )
        )
        assert result.estimator.random_state is None
        seeds = [repeat.seed for repeat in result.repeats]
        assert all(seed is not None for seed in seeds)
        refit = type(result.estimator).refit
        calls: list[dict[str, Any]] = []

        def spy(estimator: Any, data: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return refit(estimator, data, **kwargs)

        monkeypatch.setattr(type(result.estimator), "refit", spy)
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        surface = simulated_confounding(result, estimand=OPTIMAL, grid=grid)
        assert surface.complete and len(calls) == 1
        keywords = calls[0]
        assert keywords["frozen_seeds"] == seeds
        replay = refit(result.estimator, result.data, **keywords)
        assert replay[OPTIMAL].psi == result[OPTIMAL].psi
        stale = refit(
            result.estimator,
            result.data,
            **{key: value for key, value in keywords.items() if key != "frozen_seeds"},
        )
        # Fresh trees move the estimate by an amount that varies run to run (as little as
        # 5e-5 on CI), so the witness asks only that the stale refit does not reproduce it.
        assert stale[OPTIMAL].psi != result[OPTIMAL].psi

    def test_the_summary_states_the_relearning_reading(self) -> None:
        surface = _surface()
        assert "optimal_rule_reading" in [n.code for n in surface.summary().notes]
        assert "learned_rule_reading" not in [n.code for n in surface.summary().notes]
        assert any(line.startswith("optimal-rule reading") for line in surface.population_lines())

    def test_the_surface_draws(self) -> None:
        assert confounding(_surface()).axes
        with pytest.raises(TypeError):
            sensitivity(_surface())


# ------------------------------------------------- the replay with strata and comparators


def _offer(w: Any) -> np.ndarray:
    """A known random plan: offer the treatment to half the rows with ``W1 > 0``."""
    p = 0.5 * (np.asarray(w["W1"]) > 0)
    return np.column_stack([1.0 - p, p])


#: The fits of 4.1 and 4.2 of the plan, each study-identified with the known mechanism.
_COMPOSITIONS: dict[str, tuple[Any, tuple[str, ...] | None]] = {
    "strata": (OptimalRuleValue(), ("W2",)),
    "known": (OptimalRuleContrast(against=(Static(1), Static(0))), None),
    "plan_natural": (
        OptimalRuleContrast(
            against=(Stochastic(_offer, name="offer", density_kind="known"), NaturalCourse())
        ),
        None,
    ),
    "known_strata": (OptimalRuleContrast(against=(Static(1), Static(0))), ("W2",)),
}

#: The stratum cells of one refit differ by 0.032, 0.064, 0.055 and 0.023 on the four
#: cells of the strata fit (probe 3 and this fixture alike); the margin is half the smallest.
W_F4S3_STRATUM_GAP = 0.01


@cache
def _composition(name: str, repeats: int = 1) -> Any:
    estimand, strata = _COMPOSITIONS[name]
    design = {} if strata is None else {"strata": strata}
    return (
        _study(support.law_frame(n=403), **design)
        .identify(estimand)
        .estimate(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            n_folds=5,
            cv_evaluation=True,
            simultaneous=False,
            random_state=0,
            repeats=repeats,
        )
    )


def _replayed(result: Any) -> list[str]:
    """Every alias of the fit but the natural-course value, which the replay refuses."""
    return [alias for alias in result.estimates if not alias.startswith("ey_natural_course[")]


@cache
def _composition_surfaces(name: str, repeats: int = 1) -> dict[str, Any]:
    result = _composition(name, repeats)
    return {
        alias: simulated_confounding(result, estimand=alias, grid=_GRID)
        for alias in _replayed(result)
    }


def _check_cells_are_relearning_refits(result: Any, surfaces: dict[str, Any]) -> None:
    """Every nonzero cell of every alias is that alias of the manual relearning refit."""
    first = next(iter(surfaces.values()))
    for index, cell in enumerate(first.cells):
        if index == 0:
            continue
        manual = _relearned(result, _perturbed(result, first, cell))
        for alias, surface in surfaces.items():
            assert surface.cells[index].estimate == pytest.approx(manual[alias].psi, abs=1e-12), (
                alias
            )


class TestTheRelearningReplayWithStrata:
    """W-f4s1 to W-f4s5 of the plan (X11 (f4a)), with the mutation controls."""

    def test_the_anchor_and_the_manual_refit_reproduce_every_alias(self) -> None:
        """W-f4s1."""
        result, surfaces = _composition("strata"), _composition_surfaces("strata")
        assert len(surfaces) == 3
        manual = _relearned(result, result.data)
        for alias, surface in surfaces.items():
            assert surface.complete and surface.rule_replay == "relearned"
            assert surface.cells[0].estimate == result[alias].psi
            assert manual[alias].psi == result[alias].psi

    @pytest.mark.parametrize("repeats", [1, 2])
    def test_every_cell_of_every_alias_is_the_relearning_refit(self, repeats: int) -> None:
        """W-f4s2."""
        result = _composition("strata", repeats)
        assert result.n_repeats == repeats
        _check_cells_are_relearning_refits(result, _composition_surfaces("strata", repeats))

    def test_the_marginal_cell_is_the_share_mixture_of_the_stratum_cells(self) -> None:
        """W-f4s3: the stratum mask is read, so a replay that read the marginal fails."""
        result, surfaces = _composition("strata"), _composition_surfaces("strata")
        marginal, *strata = surfaces
        shares = [float(np.mean(result.data.strata == code)) for code in range(2)]
        for index in range(len(_GRID.treatment_points) * len(_GRID.outcome)):
            cells = [surfaces[alias].cells[index].estimate for alias in strata]
            mixture = sum(share * value for share, value in zip(shares, cells, strict=True))
            assert surfaces[marginal].cells[index].estimate == pytest.approx(mixture, abs=1e-12)
            assert abs(cells[0] - cells[1]) > W_F4S3_STRATUM_GAP

    def test_one_rule_per_fold_is_relearned_and_shared_by_the_strata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-f4s4: the cell refits of both stratum aliases learn one set of fold rules."""
        result = _composition("strata")
        refit = type(result.estimator).refit
        rules: list[np.ndarray] = []

        def spy(estimator: Any, data: Any, **kwargs: Any) -> Any:
            out = refit(estimator, data, **kwargs)
            rules.append(fold_rules(out.repeats[0].nuisance))
            return out

        monkeypatch.setattr(type(result.estimator), "refit", spy)
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        for alias in list(result.estimates)[1:]:
            simulated_confounding(result, estimand=alias, grid=grid)
        monkeypatch.undo()
        assert len(rules) == 2 and np.array_equal(rules[0], rules[1])
        surface = _composition_surfaces("strata")[list(result.estimates)[1]]
        cell = next(
            c for c in surface.cells if c.treatment_strength > 0 and c.outcome_strength == 0
        )
        manual = fold_rules(
            _relearned(result, _perturbed(result, surface, cell)).repeats[0].nuisance
        )
        assert np.array_equal(rules[0], manual)
        assert float(np.mean(rules[0] != fold_rules(result.repeats[0].nuisance))) > 0.02

    def test_a_stratum_cell_reads_its_association_on_the_stratum_rows(self) -> None:
        """W-f4s5: the association is the stratum's own; the population fraction stays 1."""
        result, surfaces = _composition("strata"), _composition_surfaces("strata")
        marginal, *strata = surfaces
        cell_index = next(
            i for i, c in enumerate(surfaces[marginal].cells) if c.treatment_strength > 0
        )
        cell = surfaces[marginal].cells[cell_index]
        treatment = _perturbed(result, surfaces[marginal], cell).treatment
        latent = np.random.default_rng(surfaces[marginal].latent_seed).normal(size=result.data.n)
        for code, alias in enumerate(strata):
            mask = result.data.strata == code
            own = surfaces[alias].cells[cell_index]
            expected = float(np.corrcoef(latent[mask], treatment[mask])[0, 1])
            assert own.induced_treatment_association == pytest.approx(expected, abs=1e-12)
            assert (
                abs(own.induced_treatment_association - cell.induced_treatment_association) > 0.01
            )
            assert own.target_population_fraction == 1.0

    @pytest.mark.parametrize("mutation", [None, "freezes_the_rules", "redraws_the_folds"])
    def test_a_replay_that_freezes_the_rules_or_redraws_the_folds_fails(
        self, monkeypatch: pytest.MonkeyPatch, mutation: str | None
    ) -> None:
        """Mutation controls of W-f4s2: each moves a flipped stratum cell by more than 5e-4.

        The surface runs under a seed other than the fit's, so a replay without the fit's
        folds draws other folds.  The control, with no mutation, still equals the manual
        refit, because the folds are frozen and the parametric learners draw no seed.
        """
        result = _composition("strata")
        validate = request_module.validate_optimal_rule_replay

        def mutated(fit: Any, estimand: str, key: Any) -> Any:
            replay, folds, kind = validate(fit, estimand, key)
            if mutation == "freezes_the_rules":
                return frozen_rule_replay(replay, freeze_learned_rule(fit)), folds, kind
            if mutation == "redraws_the_folds":
                return replay, None, kind
            return replay, folds, kind

        monkeypatch.setattr(request_module, "validate_optimal_rule_replay", mutated)
        alias = list(result.estimates)[2]
        surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=3)
        assert surface.complete
        cell = next(c for c in surface.cells if c.treatment_strength > 0)
        manual = _relearned(result, _perturbed(result, surface, cell))
        gap = abs(cell.estimate - manual[alias].psi)
        assert gap < 1e-12 if mutation is None else gap > 5e-4


#: The rule cell and each comparator cell of the "known" fit differ by 0.0069 to 0.129 over
#: the four cells and two comparators; the margin is half the smallest observed gap.
W_F4R3_COLUMN_GAP = 3e-3


class TestTheRelearningReplayBesideComparators:
    """W-f4r1 to W-f4r8 of the plan (X11 (f4b)), with the mutation controls."""

    @pytest.mark.parametrize("name", ["known", "plan_natural", "known_strata"])
    def test_the_anchor_and_the_manual_refit_reproduce_every_alias(self, name: str) -> None:
        """W-f4r1."""
        result, surfaces = _composition(name), _composition_surfaces(name)
        manual = _relearned(result, result.data)
        for alias, surface in surfaces.items():
            assert surface.complete and surface.rule_replay == "relearned"
            assert surface.cells[0].estimate == result[alias].psi
            assert manual[alias].psi == result[alias].psi

    @pytest.mark.parametrize(
        ("name", "repeats"),
        [("known", 1), ("known", 2), ("plan_natural", 1), ("known_strata", 1)],
    )
    def test_every_cell_of_every_alias_is_the_relearning_refit(
        self, name: str, repeats: int
    ) -> None:
        """W-f4r2: the manual refit evaluates the declared comparators again."""
        _check_cells_are_relearning_refits(
            _composition(name, repeats), _composition_surfaces(name, repeats)
        )

    def test_each_contrast_is_the_rule_cell_minus_the_comparator_cell(self) -> None:
        """W-f4r3: the sign and the columns of every contrast."""
        result, surfaces = _composition("known"), _composition_surfaces("known")
        rule = surfaces["ey_optimal_rule[learned rule]"]
        for comparator in ("always 1", "always 0"):
            known = surfaces[f"ey_known_regime[{comparator}]"]
            contrast = surfaces[f"ate_optimal_rule[learned rule vs {comparator}]"]
            for index, cell in enumerate(contrast.cells):
                rule_value = rule.cells[index].estimate
                known_value = known.cells[index].estimate
                assert cell.estimate == pytest.approx(rule_value - known_value, abs=1e-12)
                assert abs(rule_value - known_value) > W_F4R3_COLUMN_GAP
        assert result.identified_effect.functional.reference is None

    def test_the_comparators_are_frozen_at_the_stored_densities(self) -> None:
        """W-f4r4, first half: the replay's comparator columns are the stored ones."""
        result = _composition("plan_natural")
        key = result.parameter_keys["ey_known_regime[offer]"]
        replay, folds, kind = request_module.validate_optimal_rule_replay(
            result, "ey_known_regime[offer]", key
        )
        assert kind == "relearned" and len(folds) == 1
        assert replay.natural_course is result.estimator.natural_course
        assert replay.learned_rule is result.estimator.learned_rule
        (frozen,) = replay.interventions
        stored = np.asarray(result.nuisance.regimes.values)[:, :, 1]
        assert np.array_equal(frozen.density(result.data), stored)

    def test_a_comparator_whose_callback_changes_is_refused_before_any_draw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-f4r4, the mutation control of the freeze: R5 names the stored draw."""
        state = {"moved": False}

        def stateful(w: Any) -> np.ndarray:
            p = (0.6 if state["moved"] else 0.5) * (np.asarray(w["W1"]) > 0)
            return np.column_stack([1.0 - p, p])

        plan = Stochastic(stateful, name="offer", density_kind="known")
        result = (
            _study(support.law_frame(n=403))
            .identify(OptimalRuleContrast(against=(plan,)))
            .estimate(
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                cv_evaluation=True,
                simultaneous=False,
                random_state=0,
            )
        )
        alias = "ate_optimal_rule[learned rule vs offer]"
        assert simulated_confounding_refusal(result, alias) is None
        state["moved"] = True
        refusal = simulated_confounding_refusal(result, alias)
        assert refusal is not None and refusal.reason is RefusalReason.INVALID_DECLARATION
        text = "declared regime densities disagree with a stored cross-fitting draw"
        assert text in refusal.condition
        forbid_draw_and_refit(monkeypatch, result.estimator)
        with pytest.raises(CapabilityError, match=text):
            simulated_confounding(result, estimand=alias, grid=_GRID)

    def test_the_rule_is_relearned_beside_the_frozen_comparators(self) -> None:
        """W-f4r5: a frozen-rule replay moves the rule and contrast cells.

        The comparator cell equals the manual relearning refit's.  It is not free of the
        rule: one fluctuation targets every column of the regime set together (one clever
        covariate per column, ``targeting.build_submodel``), so the comparator's targeted
        value moves by about 1e-4 when the rule column changes, while its initial outcome
        regression is the same in both refits.  ``simultaneous`` names the bands, not the
        targeting.
        """
        result, surfaces = _composition("known"), _composition_surfaces("known")
        rule = surfaces["ey_optimal_rule[learned rule]"]
        index, cell = next(
            (i, c)
            for i, c in enumerate(rule.cells)
            if c.treatment_strength > 0 and c.outcome_strength > 0
        )
        data = _perturbed(result, rule, cell)
        manual = _relearned(result, data)
        # The learned column's fold rules, every fold's rule at every row, as in W-sc1.
        learned = fold_rules(manual.repeats[0].nuisance)
        assert float(np.mean(learned != fold_rules(result.repeats[0].nuisance))) > 0.02
        folds = [repeat.nuisance.folds for repeat in result.repeats]
        frozen = frozen_rule_replay(result.estimator, freeze_learned_rule(result)).refit(
            data, random_state=0, frozen_folds=folds
        )
        for alias in (
            "ey_optimal_rule[learned rule]",
            "ate_optimal_rule[learned rule vs always 0]",
        ):
            assert abs(frozen[alias].psi - surfaces[alias].cells[index].estimate) > 5e-4
        comparator = "ey_known_regime[always 0]"
        assert surfaces[comparator].cells[index].estimate == pytest.approx(
            manual[comparator].psi, abs=1e-12
        )
        for arm in (0.0, 1.0):
            assert np.array_equal(
                frozen.nuisance.outcome.arms[arm], manual.nuisance.outcome.arms[arm]
            )

    def test_the_contrast_with_the_natural_course_reads_the_perturbed_outcome(self) -> None:
        """W-f4r6: rule value minus the fold-weighted mean of the cell's perturbed outcome."""
        result, surfaces = _composition("plan_natural"), _composition_surfaces("plan_natural")
        rule = surfaces["ey_optimal_rule[learned rule]"]
        contrast = surfaces["ate_optimal_rule[learned rule vs natural course]"]
        folds = result.nuisance.folds
        for index, cell in enumerate(contrast.cells[1:], start=1):
            outcome = _perturbed(result, contrast, cell).outcome
            natural = float(np.mean([outcome[test].mean() for _, test in folds]))
            assert cell.estimate == pytest.approx(rule.cells[index].estimate - natural, abs=1e-12)
        moved = next(c for c in contrast.cells if c.treatment_strength == 0 and c.outcome_strength)
        # Observed 0.042; the margin is about a tenth of it.
        assert abs(moved.displacement) > 5e-3

    def test_the_natural_course_value_is_refused_before_any_draw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R2, with the capability row's text."""
        result = _composition("plan_natural")
        alias = "ey_natural_course[natural course]"
        refusal = simulated_confounding_refusal(result, alias)
        assert refusal is not None and refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "it is E[Y], with no counterfactual treatment dependence" in refusal.condition
        assert "'ate_optimal_rule[learned rule vs natural course]'" in str(refusal.alternative)
        assert alias not in request_module._eligible_binary_parameter_names(result)
        forbid_draw_and_refit(monkeypatch, result.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(result, estimand=alias, grid=_GRID)
        assert caught.value.refusal == refusal

    def test_a_fit_beside_the_natural_course_alone_replays_its_contrast(self) -> None:
        """The defect of probe 2: this fit raised "inconsistent optimal-rule metadata"."""
        result = _learned_contrast(OptimalRuleContrast(against=(NaturalCourse(),)))
        alias = "ate_optimal_rule[learned rule vs natural course]"
        surface = simulated_confounding(result, estimand=alias, grid=_GRID)
        assert surface.complete and surface.cells[0].estimate == result[alias].psi

    @pytest.mark.parametrize(
        "alias", ["ate_optimal_rule[learned rule vs always 1]", "ey_known_regime[always 1]"]
    )
    def test_every_alias_of_an_unseeded_fit_replays_with_its_own_seeds(
        self, monkeypatch: pytest.MonkeyPatch, alias: str
    ) -> None:
        """W-f4r7: the refit of a comparator or contrast cell carries the repeat seeds."""
        result = (
            _study(support.law_frame(n=403))
            .identify(OptimalRuleContrast(against=(Static(1), Static(0))))
            .estimate(
                outcome_learner=RandomForestClassifier(n_estimators=20, min_samples_leaf=20),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                cv_evaluation=True,
                simultaneous=False,
            )
        )
        assert result.estimator.random_state is None
        seeds = [repeat.seed for repeat in result.repeats]
        refit = type(result.estimator).refit
        calls: list[dict[str, Any]] = []

        def spy(estimator: Any, data: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return refit(estimator, data, **kwargs)

        monkeypatch.setattr(type(result.estimator), "refit", spy)
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        surface = simulated_confounding(result, estimand=alias, grid=grid)
        monkeypatch.undo()
        assert surface.complete and len(calls) == 1
        keywords = calls[0]
        assert keywords["frozen_seeds"] == seeds
        replay = request_module.validate_optimal_rule_replay(
            result, alias, result.parameter_keys[alias]
        )[0]
        assert refit(replay, result.data, **keywords)[alias].psi == result[alias].psi
        # The mutation: the 415fca9 condition passed seeds for ey_optimal_rule alone.
        stale = refit(
            replay, result.data, **{k: v for k, v in keywords.items() if k != "frozen_seeds"}
        )
        assert (
            stale["ey_optimal_rule[learned rule]"].psi
            != result["ey_optimal_rule[learned rule]"].psi
        )

    @pytest.mark.parametrize("name", ["known", "strata", "known_strata"])
    def test_every_alias_carries_the_relearning_reading(self, name: str) -> None:
        """W-f4r8: the reading follows the fit, not the alias's prefix."""
        for surface in _composition_surfaces(name).values():
            _check_relearning_reading(surface)

    def test_a_reading_keyed_on_the_wrong_replay_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-f4r8, the mutation: a comparator alias routed as frozen loses its reading."""
        result = _composition("known")
        validate = request_module.validate_optimal_rule_replay

        def mutated(fit: Any, estimand: str, key: Any) -> Any:
            replay, folds, _ = validate(fit, estimand, key)
            return replay, folds, "frozen"

        monkeypatch.setattr(request_module, "validate_optimal_rule_replay", mutated)
        grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        surface = simulated_confounding(result, estimand="ey_known_regime[always 1]", grid=grid)
        assert surface.rule_replay == "frozen"
        with pytest.raises(AssertionError):
            _check_relearning_reading(surface)

    def test_the_surfaces_draw(self) -> None:
        for surface in _composition_surfaces("known_strata").values():
            assert confounding(surface).axes
            with pytest.raises(TypeError):
                sensitivity(surface)


def _check_relearning_reading(surface: Any) -> None:
    codes = [n.code for n in surface.summary().notes]
    assert surface.rule_replay == "relearned"
    assert "optimal_rule_reading" in codes and "learned_rule_reading" not in codes
    lines = surface.population_lines()
    assert any(line.startswith("optimal-rule reading") for line in lines)
    assert not any(line.startswith("learned-rule reading") for line in lines)


def _learned_contrast(estimand: Any) -> Any:
    return (
        _study(support.law_frame(n=403))
        .identify(estimand)
        .estimate(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            n_folds=5,
            cv_evaluation=True,
            simultaneous=False,
            random_state=0,
        )
    )


# -------------------------------------------------------- the relearning replay at a level
#
# X11 (j4b).  Each cell holds the treatment, flips the intermediate and moves the outcome,
# and relearns the rule at the fit's level on the fit's folds and repeat seeds.

#: W-j4o3: at every nonzero cell of the three fits the frozen-rule replay misses the
#: relearned cell by 0.0065 to 0.089 (observed); the margin is half the smallest.
W_J4O3_GAP = 3e-3


@cache
def _optimal_level(name: str) -> Any:
    from tests.unit.test_simulated_confounding_learned_rule import _level_options, _level_study

    estimand = {
        "0": OptimalRuleValue(intermediate=0.0),
        "1": OptimalRuleValue(intermediate=1.0),
        "contrast": OptimalRuleContrast(against=(Static(1),), intermediate=1.0),
    }[name]
    return _level_study().identify(estimand).estimate(**_level_options())


@cache
def _optimal_level_surfaces(name: str) -> dict[str, Any]:
    from tests.unit.test_simulated_confounding_learned_rule import _LEVEL_GRID

    result = _optimal_level(name)
    return {
        alias: simulated_confounding(result, estimand=alias, grid=_LEVEL_GRID)
        for alias in result.estimates
    }


def _relearned_level(result: Any, data: Any) -> Any:
    z = result.intermediate_value
    folds = [repeat.nuisance.folds for repeat in result.repeats]
    return result.estimator.refit(data, intermediate_value=z, random_state=0, frozen_folds=folds)


@pytest.mark.parametrize("name", ["0", "1", "contrast"])
class TestTheRelearningReplayAtALevel:
    """W-j4o1 to W-j4o4 of the plan, and the reading of every level alias (W-f4r8)."""

    def test_the_anchor_and_the_manual_refit_reproduce_every_alias(self, name: str) -> None:
        """W-j4o1."""
        result, surfaces = _optimal_level(name), _optimal_level_surfaces(name)
        manual = _relearned_level(result, result.data)
        for alias, surface in surfaces.items():
            assert surface.complete and surface.intermediate_value == result.intermediate_value
            assert surface.cells[0].estimate == result[alias].psi
            assert manual[alias].psi == result[alias].psi
            _check_relearning_reading(surface)

    def test_every_cell_is_the_relearning_refit_at_the_level(self, name: str) -> None:
        """W-j4o2."""
        from tests.unit.test_simulated_confounding_learned_rule import level_table

        result, surfaces = _optimal_level(name), _optimal_level_surfaces(name)
        first = next(iter(surfaces.values()))
        for index, cell in enumerate(first.cells[1:], start=1):
            manual = _relearned_level(result, level_table(result, first, cell))
            for alias, surface in surfaces.items():
                assert surface.cells[index].estimate == pytest.approx(
                    manual[alias].psi, abs=1e-12
                ), alias

    def test_a_flipped_cell_relearns_the_level_rules(self, name: str) -> None:
        """W-j4o3, nonzero: the frozen-rule replay of the same cell misses it."""
        from tests.unit.test_simulated_confounding_learned_rule import level_table

        result, surfaces = _optimal_level(name), _optimal_level_surfaces(name)
        alias = "ey_optimal_rule[learned rule]"
        surface = surfaces[alias]
        z = result.intermediate_value
        for cell in surface.cells[1:]:
            table = level_table(result, surface, cell)
            manual = _relearned_level(result, table)
            changed = fold_rules(manual.repeats[0].nuisance) != fold_rules(
                result.repeats[0].nuisance
            )
            assert float(np.mean(changed)) > 0.02
            frozen = frozen_rule_replay(result.estimator, freeze_learned_rule(result)).refit(
                table, intermediate_value=z, random_state=0, frozen_folds=[result.nuisance.folds]
            )
            assert abs(frozen[alias].psi - cell.estimate) > W_J4O3_GAP

    def test_a_treatment_strength_is_refused_before_any_draw(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-j4o4: R1."""
        result = _optimal_level(name)
        forbid_draw_and_refit(monkeypatch, result.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(result, estimand="ey_optimal_rule[learned rule]", grid=_GRID)
        assert caught.value.refusal.roadmap == "F15"
