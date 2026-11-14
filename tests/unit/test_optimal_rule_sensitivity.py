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
* :class:`TestTheBoundRefusals` and :class:`TestTheBoundReports` pin the refusals, the
  withheld limits, the summaries and the plots.

**The relearning replay.**  Every cell of ``simulated_confounding`` refits the stored
estimator, which relearns the rule, on the fit's folds and seed, with the known mechanism of
the flipped treatment.  :class:`TestTheRelearningReplay` is W-sc1 and W-sc2.
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

import cleverly.sensitivity.omitted_variable as ov
from cleverly import CausalStudy, OptimalRuleValue, PointTreatment
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, InSampleLearnerWarning, RefusalReason
from cleverly.interventions import LearnedRule
from cleverly.interventions.learned import (
    OPTIMAL_RULE_BENCHMARK_REFUSAL,
    OPTIMAL_RULE_DOUBLY_ROBUST_NU2,
    OPTIMAL_RULE_LIMITS,
    OPTIMAL_RULE_REPLAY_STRATA_REFUSAL,
    OPTIMAL_RULE_TARGET,
    fold_rules,
)
from cleverly.plotting import confounding, sensitivity
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity.simulated_confounding import flipped_mechanism
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
        assert value["max_bias_upper"] == bounds.max_bias_upper
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

    def test_the_summary_states_the_relearning_reading(self) -> None:
        surface = _surface()
        assert "optimal_rule_reading" in [n.code for n in surface.summary().notes]
        assert "learned_rule_reading" not in [n.code for n in surface.summary().notes]
        assert any(line.startswith("optimal-rule reading") for line in surface.population_lines())

    def test_the_surface_draws(self) -> None:
        assert confounding(_surface()).axes
        with pytest.raises(TypeError):
            sensitivity(_surface())

    def test_strata_refuse_before_any_draw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = (
            _study(support.law_frame(n=403), strata=("W2",))
            .identify(OptimalRuleValue())
            .estimate(
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                cv_evaluation=True,
                simultaneous=False,
                random_state=0,
            )
        )
        row = result.sensitivity.capability("simulated_confounding")
        assert not row.available
        forbid_draw_and_refit(monkeypatch, result.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(result, estimand=OPTIMAL, grid=_GRID)
        assert OPTIMAL_RULE_REPLAY_STRATA_REFUSAL.condition in str(caught.value)
        assert "X11 (f4) in docs/roadmap.md" in str(caught.value)
