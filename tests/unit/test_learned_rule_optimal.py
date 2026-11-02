"""The value of the optimal rule: ``ey_optimal_rule`` and ``OptimalRuleValue`` (X11 (f)).

The target is the fixed parameter :math:`\\Psi_{d_0}(P_0) = E\\,\\bar Q_0(d_0(W), W)`, with
:math:`d_0(w) = \\arg\\max_a \\bar Q_0(a, w)`.  The estimator is the cross-validated TMLE of the
learned-rule value, so the arithmetic is unchanged and only the target and its conditions
differ (van der Laan and Luedtke 2015, Section 7.2, last paragraph; ``X11-wave2-plan.md``
rev 2.1, section 6).  The witnesses below are the plan's:

- W-f1 (identity): the point, the variance and the curve equal the learned-rule fit's bit for
  bit on the same draw and seeds, under each admitted composition, at a nonzero fluctuation.
- W-f2 (target gap, nonzero): on the wrong-limit law with the even blip ``3 (W1^2 - 0.5)``, the
  limit of the declared main-terms outcome learner treats nobody, and
  :math:`\\Psi_{d_0} - \\Psi_{d_1}` by quadrature is 0.045, more than three influence-curve
  standard errors at n = 2,500.  The same quadrature gives a gap of 0 on the ``non_exceptional``
  law, where the learner is correct, and an amplitude of 1 needs n of about 16,400, which the
  rule "n at most 5,000" refuses.
- W-f3 (known g): with ``treatment_probabilities=`` the fit solves its score at the supplied
  probabilities, and the mutation that substitutes an estimated g moves the point.

The rest pins the refusals of X11 (f3) with remedies that run, the support family ``P12``, whose
one row ``T46`` (the ``learned-rule-optimal-value`` study) covers the fold-evaluated report on
rows with a known treatment mechanism, while every learned-rule key and row is unchanged, the
record and summary line, the fold-rule disagreement diagnostic, and the sensitivity analyses that
refuse.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import cache
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LearnedRuleValue,
    LongitudinalTreatment,
    OptimalRuleValue,
    PointTreatment,
    TMLEMethod,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import LearnedRule, Static
from cleverly.interventions.learned import (
    OPTIMAL_REPLAY_TARGET_KIND,
    OPTIMAL_RULE_SENSITIVITY_REFUSAL,
    OPTIMAL_RULE_TARGET,
    OPTIMAL_TARGET_KIND,
    TARGET_KIND,
    fold_rules,
    freeze_learned_rule,
    frozen_rule_replay,
    optimal_rule_requested,
)
from cleverly.sensitivity import evalue
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.omitted_variable import (
    benchmark_refusal,
    fit_wide_bound_refusal,
)
from cleverly.study import _continuous_dose_refusal
from cleverly.targets import TARGETS
from cleverly.validation.refute import refute_refusal
from tests import discrete_law as exact
from tests.conftest import OracleOutcome, OracleTreatment
from tests.studies import _learned_rule_arms_law as arms_law
from tests.studies import _learned_rule_law as law
from tests.studies import _learned_rule_optimal_law as optimal_law
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The reported names of the two targets of the default rule.
LEARNED = support.NAME
OPTIMAL = "ey_optimal_rule[learned rule]"
FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}

#: W-f2's declared amplitude of the even blip ``c (W1^2 - 0.5)``, the rows at which the gap
#: must be three standard errors, and the floor of the gap.  The quadrature gives a gap of
#: 0.0452 and needs n = 2,085 for three standard errors, so n = 2,500 leaves 3.3.
W_F2_AMPLITUDE = 3.0
W_F2_N = 2_500
W_F2_FLOOR = 0.04
#: The largest n the plan admits for the control (``X11-wave2-plan.md`` section 11, finding 6).
W_F2_MAX_N = 5_000

#: W-f3's declared move of the point, below the 0.016 the fixed draw gives.
W_F3_POINT_MOVE = 0.01


def _flat(text: Any) -> str:
    """``text`` with its line wrapping undone, so a sentence is found across lines."""
    return " ".join(str(text).split())


def _pair(frame: pd.DataFrame | None = None, **overrides: Any) -> tuple[Any, Any]:
    """The learned-rule fit and the optimal-value fit of one declaration, on one draw."""
    return (
        support.fit(frame, **overrides),
        support.fit(frame, estimands=[OPTIMAL_RULE_TARGET], **overrides),
    )


# ------------------------------------------------------------------------------- W-f1


def _clustered() -> tuple[Any, Any]:
    frame = support.law_frame(400, 3).assign(cluster=np.repeat(np.arange(100), 4))
    return (
        support.learned_tmle().fit(frame, id="cluster", **FIT).single(),
        support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET])
        .fit(frame, id="cluster", **FIT)
        .single(),
    )


def _known_g() -> tuple[Any, Any]:
    frame = support.law_frame(400, 4)
    g1 = expit(0.3 * frame["W1"].to_numpy() - 0.2 * frame["W2"].to_numpy())
    known = {"treatment_probabilities": np.column_stack([1.0 - g1, g1])}
    return tuple(  # type: ignore[return-value]
        support.learned_tmle(outcome_learner=support.misspecified_learner(), **extra)
        .fit(frame, **known, **FIT)
        .single()
        for extra in ({}, {"estimands": [OPTIMAL_RULE_TARGET]})
    )


#: The compositions the plan admits, each a pair of fits on one draw.  The planted draw has
#: unequal folds and a misspecified outcome regression, so the fluctuation moves.
PAIRS = {
    "fold_evaluated": lambda: _pair(
        support.law_frame(203, 1), outcome_learner=support.misspecified_learner()
    ),
    "stacked": lambda: _pair(
        support.law_frame(203, 1),
        outcome_learner=support.misspecified_learner(),
        cv_evaluation=False,
    ),
    "repeats": lambda: _pair(support.law_frame(203, 1), repeats=3),
    "fold_targeting": lambda: _pair(support.law_frame(203, 1), targeting_scheme="fold"),
    "rule_learner": lambda: _pair(
        support.law_frame(300, 2), learned_rule=LearnedRule(loss="blip", learner=LinearRegression())
    ),
    "clusters": _clustered,
    "known_g": _known_g,
    "three_arms": lambda: _pair(arms_law.draw(300, 0), outcome_learner=arms_law.outcome_learner()),
}


@cache
def _pairs(name: str) -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PAIRS[name]()  # type: ignore[no-untyped-call]


class TestWf1Identity:
    """W-f1: the optimal-value fit is the learned-rule fit under another name."""

    @pytest.mark.parametrize("name", list(PAIRS))
    def test_point_variance_and_curve_are_bit_for_bit(self, name: str) -> None:
        learned, optimal = _pairs(name)
        assert list(learned.estimates) == [LEARNED]
        assert list(optimal.estimates) == [OPTIMAL]
        a, b = learned.estimates[LEARNED], optimal.estimates[OPTIMAL]
        assert a.psi == b.psi
        assert a.variance == b.variance
        assert a.ci == b.ci
        assert np.array_equal(a.influence_curve, b.influence_curve)
        record_a, record_b = learned.extra["learned_rule"], optimal.extra["learned_rule"]
        assert record_b == replace(record_a, estimand=OPTIMAL_RULE_TARGET)

    def test_the_fluctuation_moves(self) -> None:
        """The nonzero witness: the identity is not that of an untargeted plug-in."""
        _, optimal = _pairs("fold_evaluated")
        assert abs(float(optimal.fluctuations["regime"].epsilon[0])) > 1e-2

    def test_the_study_estimates_the_learned_rule_estimate(self) -> None:
        frame = support.law_frame(400, 0)
        study = CausalStudy(frame, design=PointTreatment(**_design()))
        method = TMLEMethod(cross_fitting=CrossFitting(fold_evaluation=True))
        options = {
            "outcome_learner": support.outcome_learner(),
            "treatment_learner": support.treatment_learner(),
            "simultaneous": False,
            "random_state": 0,
        }
        learned = study.identify(LearnedRuleValue()).estimate(method, **options)
        optimal = study.identify(OptimalRuleValue()).estimate(method, **options)
        assert list(optimal.estimates) == [OPTIMAL]
        assert optimal.estimates[OPTIMAL].psi == learned.estimates[LEARNED].psi
        assert optimal.estimates[OPTIMAL].variance == learned.estimates[LEARNED].variance
        assert optimal.parameter_keys[OPTIMAL].estimand == OPTIMAL_RULE_TARGET
        assert optimal.parameter_keys[OPTIMAL].axis == "learned_rule"


def _design(**extra: Any) -> dict[str, Any]:
    return {"outcome": "Y", "treatment": "A", "adjustment": support.COVARIATES, **extra}


# ---------------------------------------------------------------------- the exact law

#: The variant of the discrete law whose true blips, ``+0.3``, ``-0.3`` and ``+0.2``, have the
#: signs of :data:`exact.LEARNED_RULE`, so that rule is the optimal rule ``d0``.  An oracle
#: outcome regression on :data:`WRONG_Q`, with the same blip signs, learns ``d0`` in every
#: fold, and its score is not zero, so the pooled update moves.
EXACT_COUNTS = exact.cell_counts(q=exact.LEARNED_RULE_Q)
EXACT_PROBS = EXACT_COUNTS / exact.N
WRONG_Q = np.array([[0.45, 0.60], [0.55, 0.25], [0.55, 0.85]])
EXACT_NAME = f"ey_optimal_rule[{exact.LEARNED_RULE_LABEL}]"


def _exact_law(q: np.ndarray) -> np.ndarray:
    """The cell probabilities of the discrete law's ``W`` and ``A`` with the regression ``q``."""
    arm = np.stack([1.0 - exact.G, exact.G], axis=1)
    outcome = np.stack([1.0 - q, q], axis=2)
    return exact.P_W[:, None, None] * arm[:, :, None] * outcome


@cache
def _exact_fit() -> Any:
    return (
        TMLE(
            learned_rule=LearnedRule(),
            estimands=[OPTIMAL_RULE_TARGET],
            outcome_learner=OracleOutcome(exact.DiscreteLaw(_exact_law(WRONG_Q))),
            treatment_learner=OracleTreatment(exact.DiscreteLaw(EXACT_PROBS)),
            cross_fit=True,
            cv_evaluation=True,
            n_folds=4,
            simultaneous=False,
            random_state=3,
        )
        .fit(exact.frame(EXACT_COUNTS), outcome="Y", treatment="A")
        .single()
    )


def _exact_fold_curves() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each fold: the support point of each row, the fold's unscaled curve, the fold law."""
    result = _exact_fit()
    n = result.data.n
    cells = np.repeat(
        np.arange(len(exact.SUPPORT)), [EXACT_COUNTS[w, a, y] for w, a, y in exact.SUPPORT]
    )
    w = np.rint(result.data.covariates[:, 0]).astype(int)
    targeted = result.fluctuations["regime"].targeted
    q_star = np.array(
        [
            [float(targeted.arms[float(a)][np.flatnonzero(w == cell)[0]]) for a in (0, 1)]
            for cell in range(3)
        ]
    )
    g = EXACT_PROBS.sum(axis=2)[:, 1] / EXACT_PROBS.sum(axis=(1, 2))
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q_star, q_star], axis=2)
    curve = np.asarray(result.estimates[EXACT_NAME].influence_curve, dtype=float)
    folds = [np.asarray(test) for _, test in result.nuisance.folds]
    pieces = []
    for test in folds:
        share = np.bincount(w[test], minlength=3) / test.size
        probs = share[:, None, None] * arm[:, :, None] * outcome
        pieces.append((cells[test], curve[test] * len(folds) * test.size / n, probs))
    return pieces


class TestTheExactLaw:
    """The oracle branch of ``ey_optimal_rule`` and the fold curves of a fit at oracle nuisances."""

    def test_the_oracle_is_the_best_arm_of_each_cell(self) -> None:
        """At the base law every blip is positive, so the optimal value is ``E[Y^1]``."""
        assert exact.TRUTH[EXACT_NAME] == pytest.approx(exact.TRUTH["ey1"], abs=1e-15)
        assert np.all(exact.Q[:, 1] > exact.Q[:, 0])

    def test_every_fold_learns_the_optimal_rule(self) -> None:
        result = _exact_fit()
        w = np.rint(result.data.covariates[:, 0]).astype(int)
        np.testing.assert_array_equal(
            result.nuisance.regimes.values[:, :, 0], exact.LEARNED_RULE[w]
        )
        assert float(np.max(np.abs(result.fluctuations["regime"].epsilon))) > 1e-2

    def test_each_fold_curve_is_the_gateaux_derivative_of_the_optimal_value(self) -> None:
        for cells, raw, probs in _exact_fold_curves():
            for point in np.unique(cells):
                expected = exact.gateaux(EXACT_NAME, int(point), probs=probs)
                np.testing.assert_allclose(raw[cells == point], expected, atol=1e-10, rtol=0)

    def test_control_the_worst_rule_is_caught(self) -> None:
        """The mutation of the functional: the min over arms in place of the max."""

        def worst(p: Any) -> Any:
            q = p[:, :, 1] / p.sum(axis=2)
            low = np.argmin(np.real(q), axis=1)
            return (p.sum(axis=(1, 2)) * q[np.arange(3), low]).sum()

        for cells, raw, probs in _exact_fold_curves():
            derivative = exact.gateaux_eif(worst, probs)
            gaps = [
                float(np.max(np.abs(raw[cells == point] - derivative[int(point)])))
                for point in np.unique(cells)
            ]
            assert max(gaps) > 1e-2

    def test_a_tie_makes_the_value_nonregular(self) -> None:
        """At an exceptional law the two one-sided derivatives differ: no influence curve.

        The second cell's blip is 0.  Moving mass towards its treated successes makes the
        treated arm best, and moving it away makes control best, so the slope of the
        optimal value jumps.  At the base law the same two slopes agree.
        """

        def slopes(probs: np.ndarray) -> tuple[float, float]:
            direction = np.zeros_like(probs)
            direction[1, 1, 1] = 1.0
            step = 1e-7
            base = float(np.real(exact.functional(probs, EXACT_NAME)))
            out = []
            for sign in (1.0, -1.0):
                moved = (1.0 - sign * step) * probs + sign * step * direction
                out.append((float(np.real(exact.functional(moved, EXACT_NAME))) - base) / step)
            return out[0], -out[1]

        tied = _exact_law(np.array([[0.40, 0.70], [0.50, 0.50], [0.60, 0.80]]))
        right, left = slopes(tied)
        assert abs(right - left) > 0.5, (right, left)
        right, left = slopes(exact.PROBS)
        assert abs(right - left) < 1e-5, (right, left)


# ------------------------------------------------------------------------------- W-f2


def _wrong_limit_q(a: Any, w1: np.ndarray, w2: float, amplitude: float) -> np.ndarray:
    """``P(Y = 1 | A = a, W)`` of the wrong-limit law, the RM30 base with an even blip."""
    return optimal_law.wrong_limit_qbar(a, w1, w2, amplitude)


def _gap(amplitude: float | None) -> tuple[float, float, float]:
    """The gap ``Psi_d0 - Psi_d1``, the variance of ``d1``'s curve and ``d1``'s treated share.

    The study law module's quadrature, which the declared ``optimal_target_control`` reads too.
    ``amplitude=None`` reads the ``non_exceptional`` law, the mutation of the law.
    """
    return optimal_law.target_gap(amplitude)


class TestWf2TargetGap:
    """W-f2: the optimal value and the value of the learner's limit rule differ, by quadrature."""

    def test_the_limit_rule_treats_nobody(self) -> None:
        _, _, share = _gap(W_F2_AMPLITUDE)
        assert share == 0.0

    def test_the_gap_exceeds_the_floor_and_three_standard_errors(self) -> None:
        gap, variance, _ = _gap(W_F2_AMPLITUDE)
        assert gap > W_F2_FLOOR, gap
        assert gap > 3.0 * np.sqrt(variance / W_F2_N), (gap, np.sqrt(variance / W_F2_N))
        assert 9.0 * variance / gap**2 < W_F2_N

    def test_a_correct_learner_has_no_gap(self) -> None:
        """The mutation of the law: on ``non_exceptional`` the limit rule is the optimal rule."""
        gap, _, share = _gap(None)
        assert abs(gap) < 1e-4
        assert 0.0 < share < 1.0

    def test_a_weaker_blip_cannot_be_powered(self) -> None:
        """The mutation of the amplitude: ``c = 1`` needs more rows than the plan admits."""
        gap, variance, _ = _gap(1.0)
        assert 0.0 < gap < W_F2_FLOOR
        assert 9.0 * variance / gap**2 > W_F2_MAX_N

    def test_the_fit_learns_the_limit_rule(self) -> None:
        """At n = 2,000 the declared learner's fold rules treat nobody, as its limit does."""
        rng = np.random.default_rng(5)
        n = 2_000
        w1 = rng.uniform(-1.0, 1.0, n)
        w2 = rng.binomial(1, 0.5, n).astype(float)
        a = rng.binomial(1, law.g0(w1, w2)).astype(float)
        y = rng.binomial(1, _wrong_limit_q(a, w1, w2, W_F2_AMPLITUDE)).astype(float)
        frame = pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})
        result = support.fit(
            frame,
            estimands=[OPTIMAL_RULE_TARGET],
            outcome_learner=law.outcome_learner(),
            treatment_learner=law.treatment_learner(),
        )
        (split,) = result.extra["learned_rule"].splits
        assert max(shares[1] for shares in split.arm_shares) < 0.02


# ------------------------------------------------------------------------------- W-f3


class TestWf3KnownProbabilitiesEnterTheCleverCovariate:
    """W-f3: under known g the supplied probabilities are the ``g`` of ``H``."""

    @staticmethod
    @cache
    def _fits() -> tuple[Any, Any, np.ndarray]:
        frame = law.draw("non_exceptional", 1000, 11)
        g1 = law.g0(frame["W1"].to_numpy(), frame["W2"].to_numpy())
        settings: dict[str, Any] = {"outcome_learner": support.misspecified_learner()}
        known = (
            support.learned_tmle(
                estimands=[OPTIMAL_RULE_TARGET],
                treatment_learner=law.treatment_learner(),
                **settings,
            )
            .fit(frame, treatment_probabilities=np.column_stack([1.0 - g1, g1]), **FIT)
            .single()
        )
        # The mutation: the same fit with an estimated g in place of the supplied one.
        estimated = support.fit(
            frame,
            estimands=[OPTIMAL_RULE_TARGET],
            treatment_learner=DummyClassifier(strategy="prior"),
            **settings,
        )
        return known, estimated, g1

    def test_the_score_is_solved_at_the_supplied_probabilities(self) -> None:
        known, _, g1 = self._fits()
        pieces = support.validation_pieces(known)
        treatment = np.asarray(known.data.treatment, dtype=float)
        h = (treatment == pieces["rule"]) / np.where(treatment == 1.0, g1, 1.0 - g1)
        np.testing.assert_allclose(pieces["h"], h, rtol=0, atol=1e-12)
        residual = pieces["y"] - pieces["q_star_observed"]
        assert abs(float(np.mean(h * residual))) < 1e-9
        assert abs(float(known.fluctuations["regime"].epsilon[0])) > 1e-2

    def test_the_mutation_moves_the_point(self) -> None:
        known, estimated, _ = self._fits()
        assert np.array_equal(
            np.asarray(known.nuisance.regimes.values[:, 1, 0]),
            np.asarray(estimated.nuisance.regimes.values[:, 1, 0]),
        )
        move = known.estimates[OPTIMAL].psi - estimated.estimates[OPTIMAL].psi
        assert abs(move) > W_F3_POINT_MOVE, move

    def test_a_known_mechanism_keys_known(self) -> None:
        known, _, _ = self._fits()
        assert "known_mechanism" in known.config.support_key.qualifiers
        assert known.estimates[OPTIMAL].support.label == "supported"
        assert known.estimates[OPTIMAL].support.row == "T46"


# ---------------------------------------------------------------------------- the target


class TestTheTarget:
    """The registry entry, the request and the estimand list."""

    def test_the_registry_entry(self) -> None:
        target = TARGETS[OPTIMAL_RULE_TARGET]
        assert target.parameter_axis == "learned_rule"
        assert target.group == TARGETS["ey_learned_rule"].group
        assert not target.in_default_set
        assumptions = " ".join(target.identification.assumptions)
        for phrase in (
            "fixed parameter",
            "d1 = d0",
            "non-exceptional law",
            "nothing is guaranteed",
            "regret",
            "For the plug-in rule these conditions read only the outcome regression",
            "For a pseudo-outcome rule learner the blip estimate is built with g_n",
            "the gap between the top two arms",
            "read with clusters as units",
            "estimated g",
            "Section 5.1",
        ):
            assert phrase in assumptions, phrase

    def test_the_default_and_all_report_the_learned_rule_value(self) -> None:
        assert list(support.fit().estimates) == [LEARNED]
        assert list(support.fit(estimands="all").estimates) == [LEARNED]
        assert not optimal_rule_requested(support.learned_tmle())
        assert not optimal_rule_requested(support.learned_tmle(estimands="all"))
        assert optimal_rule_requested(support.learned_tmle(estimands=OPTIMAL_RULE_TARGET))

    def test_a_string_request_names_the_one_target(self) -> None:
        result = support.fit(estimands=OPTIMAL_RULE_TARGET)
        assert list(result.estimates) == [OPTIMAL]

    def test_without_learned_rule_the_name_belongs_to_another_axis(self) -> None:
        with pytest.raises(CapabilityError, match="learned_rule="):
            TMLE(estimands=[OPTIMAL_RULE_TARGET]).fit(support.law_frame(), **FIT)

    def test_a_rule_that_is_not_a_learned_rule(self) -> None:
        with pytest.raises(Exception, match=r"takes a cleverly\.interventions\.LearnedRule"):
            OptimalRuleValue(rule=Static(1))  # type: ignore[arg-type]

    def test_the_functional_names_the_optimal_rule(self) -> None:
        study = CausalStudy(support.law_frame(), design=PointTreatment(**_design()))
        effect = study.identify(OptimalRuleValue())
        assert effect.functional.target == OPTIMAL_RULE_TARGET
        assert effect.functional.axis == "learned_rule"
        text = _flat(effect.summary(detail="full"))
        assert "d0(W) is the arm" in text
        assert "outer fold" not in text

    def test_a_longitudinal_design_names_the_estimand(self) -> None:
        rng = np.random.default_rng(8080)
        frame = pd.DataFrame(
            {
                "L0": rng.normal(size=60),
                "A0": rng.binomial(1, 0.5, 60).astype(float),
                "L1": rng.normal(size=60),
                "A1": rng.binomial(1, 0.5, 60).astype(float),
                "Y": rng.binomial(1, 0.5, 60).astype(float),
            }
        )
        design = LongitudinalTreatment(
            outcome="Y", treatment=("A0", "A1"), baseline=("L0",), time_varying=((), ("L1",))
        )
        study = CausalStudy(frame, design=design)
        with pytest.raises(CapabilityError, match="OptimalRuleValue is a point-treatment target"):
            study.identify(OptimalRuleValue())

    def test_the_continuous_dose_refusal_names_the_estimand(self) -> None:
        """The axis branch names the estimand's own type; the learned-rule rows refuse first."""
        refusal = _continuous_dose_refusal(OptimalRuleValue(), "learned_rule")
        assert refusal.condition.startswith("OptimalRuleValue over a continuous treatment")
        assert _continuous_dose_refusal(LearnedRuleValue(), "learned_rule").condition.startswith(
            "LearnedRuleValue"
        )


# --------------------------------------------------------------------------- the refusals


def _refused(call: Any, item: str, reason: RefusalReason | None = None) -> CapabilityError:
    NeverFit.calls = 0
    with pytest.raises(CapabilityError) as caught:
        call()
    assert item in str(caught.value), str(caught.value)
    if reason is not None:
        assert caught.value.refusal.reason is reason
    assert NeverFit.calls == 0
    return caught.value


def _never(**overrides: Any) -> TMLE:
    settings = {"estimands": [OPTIMAL_RULE_TARGET], **never_fit_learners(), **overrides}
    return support.learned_tmle(**settings)


def _weighted_frame() -> pd.DataFrame:
    return support.law_frame(400, 0).assign(w=np.linspace(0.5, 1.5, 400))


class TestTheRefusals:
    """Each X11 (f3) row refuses before any learner, and its remedy runs as written."""

    #: The configuration rows: what the estimator declares, and the item the refusal names.
    CONFIGURATION: ClassVar[dict[str, tuple[dict[str, Any], str, RefusalReason]]] = {
        "known_regime": (
            {"interventions": [Static(1, name="treat")]},
            "X11 (f3)",
            RefusalReason.IMPLEMENTATION_GAP,
        ),
        "reference": ({"reference": "learned rule"}, "X11 (f3)", RefusalReason.IMPLEMENTATION_GAP),
        "contrast_target": (
            {"estimands": [OPTIMAL_RULE_TARGET, "ate_learned_rule"]},
            "X11 (f3)",
            RefusalReason.IMPLEMENTATION_GAP,
        ),
        "learned_value": (
            {"estimands": [OPTIMAL_RULE_TARGET, "ey_learned_rule"]},
            "One fit reports one of the two targets",
            RefusalReason.INVALID_DECLARATION,
        ),
    }

    @pytest.mark.parametrize("row", list(CONFIGURATION))
    def test_the_configuration_rows_refuse_at_construction(self, row: str) -> None:
        overrides, item, reason = self.CONFIGURATION[row]
        _refused(lambda: _never(**overrides), item, reason)

    def test_the_contrast_remedies_fit(self) -> None:
        caught = _refused(lambda: _never(interventions=[Static(1, name="treat")]), "X11 (f3)")
        assert "TMLE(learned_rule=LearnedRule(), estimands=['ey_optimal_rule'])" in str(caught)
        assert list(support.fit(estimands=[OPTIMAL_RULE_TARGET]).estimates) == [OPTIMAL]
        contrast = support.fit(interventions=[Static(1, name="treat")])
        assert "ate_learned_rule[learned rule vs treat]" in contrast.estimates

    #: The data rows: the fit options, and the remedy that fits.
    DATA: ClassVar[dict[str, dict[str, Any]]] = {
        "weights": {"weights": "w"},
        "strata": {"strata": ["W2"]},
    }

    @pytest.mark.parametrize("row", list(DATA))
    def test_the_data_rows_refuse_before_any_learner(self, row: str) -> None:
        frame = _weighted_frame()
        caught = _refused(
            lambda: _never().fit(frame, **FIT, **self.DATA[row]),
            "X11 (f3)",
            RefusalReason.IMPLEMENTATION_GAP,
        )
        assert "estimands=None (ey_learned_rule)" in str(caught)

    @pytest.mark.parametrize("row", list(DATA))
    def test_the_data_remedies_fit(self, row: str) -> None:
        frame = _weighted_frame()
        alone = support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET]).fit(frame, **FIT).single()
        assert list(alone.estimates) == [OPTIMAL]
        learned = support.learned_tmle().fit(frame, **FIT, **self.DATA[row]).single()
        assert LEARNED in learned.estimates

    def test_the_in_sample_row_refuses_and_its_remedies_fit(self) -> None:
        frame = support.law_frame()
        caught = _refused(
            lambda: _never(cross_fit=False, cv_evaluation=False).fit(frame, **FIT),
            "X11 (f3)",
            RefusalReason.IMPLEMENTATION_GAP,
        )
        assert "a different target" in str(caught)
        assert list(support.fit(frame, estimands=[OPTIMAL_RULE_TARGET]).estimates) == [OPTIMAL]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            in_sample = support.fit(frame, cross_fit=False, cv_evaluation=False)
        assert in_sample.extra["learned_rule"].scheme == "in_sample"

    def test_an_earlier_row_comes_first(self) -> None:
        """Estimated weights (row 7) and strata with clusters (row 8) refuse before (f3)."""
        frame = _weighted_frame().assign(cluster=np.repeat(np.arange(100), 4))
        _refused(
            lambda: _never().fit(frame, weights="w", weights_estimated=True, **FIT),
            "X11 (k2)",
        )
        _refused(lambda: _never().fit(frame, strata=["W2"], id="cluster", **FIT), "X28")

    def test_the_study_refuses_the_data_rows_and_admits_them_for_the_learned_value(self) -> None:
        frame = _weighted_frame()
        for extra in ({"weights": "w"}, {"strata": ("W2",)}):
            study = CausalStudy(frame, design=PointTreatment(**_design(**extra)))
            _refused(lambda study=study: study.identify(OptimalRuleValue()), "X11 (f3)")
            assert study.identify(LearnedRuleValue()).functional.target == "ey_learned_rule"

    def test_the_study_refuses_the_in_sample_method(self) -> None:
        study = CausalStudy(support.law_frame(), design=PointTreatment(**_design()))
        effect = study.identify(OptimalRuleValue())
        _refused(
            lambda: effect.estimate(
                TMLEMethod(cross_fitting=CrossFitting(enabled=False)),
                **never_fit_learners(),
                simultaneous=False,
            ),
            "X11 (f3)",
        )


# ------------------------------------------------------------------------------ the label


class TestTheSupportKey:
    """The optimal value keys family ``P12``, which only ``T46`` declares; learned-rule keys stay."""

    #: Learned-rule configurations with a registered row, and that row.
    ROWS: ClassVar[dict[str, tuple[dict[str, Any], str]]] = {
        "fold_evaluated": ({}, "T24"),
        "stacked": ({"cv_evaluation": False}, "T42.rows"),
    }

    @pytest.mark.parametrize("name", list(ROWS))
    def test_only_the_family_differs(self, name: str) -> None:
        overrides, row = self.ROWS[name]
        learned, optimal = _pair(**overrides)
        key_a, key_b = learned.config.support_key, optimal.config.support_key
        assert key_a.family == frozenset({"P11"})
        assert key_b.family == frozenset({"P12"})
        assert replace(key_b, family=key_a.family) == key_a
        assert learned.estimates[LEARNED].support.label == "supported"
        assert learned.estimates[LEARNED].support.row == row
        support_b = optimal.estimates[OPTIMAL].support
        assert support_b.label == "experimental"
        assert support_b.reason.startswith("No registered study covers this configuration.")
        assert support_b.reason.endswith("differing on family.")

    def test_a_known_g_fold_evaluated_fit_on_rows_reads_t46(self) -> None:
        """The learned-rule-optimal-value study registers this composition (X11 PR I)."""
        _, optimal = _pairs("known_g")
        key = optimal.config.support_key
        assert key.family == frozenset({"P12"})
        assert key.splitting == ("fold_evaluated",)
        assert key.qualifiers == frozenset({"known_mechanism"})
        assert optimal.estimates[OPTIMAL].support.label == "supported"
        assert optimal.estimates[OPTIMAL].support.row == "T46"

    def test_every_other_admitted_composition_reads_experimental(self) -> None:
        """Without a known g, or with any other admitted composition, no row covers the fit."""
        for name in PAIRS:
            if name == "known_g":
                continue
            _, optimal = _pairs(name)
            assert optimal.estimates[OPTIMAL].support.label == "experimental", name
            assert optimal.estimates[OPTIMAL].support.row is None, name
            assert optimal.config.support_key.family == frozenset({"P12"}), name


# ----------------------------------------------------------------------------- the record


class TestTheRecord:
    """The record names the target, and the summary states its conditions."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit(estimands=[OPTIMAL_RULE_TARGET])

    def test_the_record_names_the_target(self, result: Any) -> None:
        record = result.extra["learned_rule"]
        assert record.estimand == OPTIMAL_RULE_TARGET
        assert record.target == OPTIMAL_TARGET_KIND
        assert record.scheme == "fold_evaluated"
        assert support.fit().extra["learned_rule"].target == TARGET_KIND["fold_evaluated"]

    def test_the_summary_states_the_target_and_its_conditions(self, result: Any) -> None:
        line = result.extra["learned_rule"].describe()
        for phrase in (
            "Target: the value of the optimal rule",
            "a fixed parameter",
            "converge to the optimal rule",
            "the effect is zero for almost no unit",
            "can under-cover",
            "Section 7.2",
            "The fold rules assign different arms to",
        ):
            assert phrase in line, phrase
        summary = result.summary(detail="full")
        assert line in summary.section("Method details").lines
        assert summary.facts["Question"] == (
            "value of the best treatment rule, from rules learned on the data"
        )
        learned = support.fit().summary(detail="full").facts["Question"]
        assert learned == "value of a treatment rule learned from the data"

    def test_the_stacked_and_repeated_lines(self) -> None:
        _, stacked = _pairs("stacked")
        assert "(the stacked report)" in stacked.extra["learned_rule"].describe()
        _, repeated = _pairs("repeats")
        assert "averaged over 3 random splits" in repeated.extra["learned_rule"].describe()

    def test_the_three_arm_line_names_the_top_two_tie(self) -> None:
        _, arms = _pairs("three_arms")
        line = arms.extra["learned_rule"].describe()
        assert "the best arm ties with another for almost no unit" in line
        assert "the effect is zero" not in line


class TestTheRuleDisagreement:
    """The record's share of rows on which the fold rules disagree."""

    def test_it_is_read_from_every_fold_rule(self) -> None:
        result = support.fit(estimands=[OPTIMAL_RULE_TARGET])
        (split,) = result.extra["learned_rule"].splits
        rules = fold_rules(result.nuisance)
        assert rules is not None
        longhand = float(np.mean([len(set(row)) > 1 for row in rules.tolist()]))
        assert split.rule_disagreement == longhand
        # The nonzero witness, and the deliberate mutation that compares two fold rules only.
        assert longhand > 0.05
        two_folds = float(np.mean(rules[:, 0] != rules[:, 1]))
        assert two_folds != split.rule_disagreement

    def test_an_exceptional_law_shows_more_disagreement(self) -> None:
        """Descriptive, on fixed draws at n = 4,000: the boundary blip ``0`` keeps rules apart."""
        values = {}
        for name in ("non_exceptional", "exceptional"):
            result = support.fit(
                law.draw(name, 4_000, 0),
                estimands=[OPTIMAL_RULE_TARGET],
                outcome_learner=law.outcome_learner(),
                treatment_learner=law.treatment_learner(),
            )
            values[name] = result.extra["learned_rule"].splits[0].rule_disagreement
        assert values["non_exceptional"] < 0.1 < 0.2 < values["exceptional"], values

    def test_in_sample_and_frozen_fits(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            in_sample = support.fit(cross_fit=False, cv_evaluation=False)
        assert in_sample.extra["learned_rule"].splits[0].rule_disagreement == 0.0
        result = support.fit()
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        refit = replay.fit(support.law_frame(), **FIT).single()
        assert np.isnan(refit.extra["learned_rule"].splits[0].rule_disagreement)
        optimal = support.fit(estimands=[OPTIMAL_RULE_TARGET])
        replay = frozen_rule_replay(optimal.estimator, freeze_learned_rule(optimal))
        record = replay.fit(support.law_frame(), **FIT).single().extra["learned_rule"]
        assert record.target == OPTIMAL_REPLAY_TARGET_KIND
        line = record.describe()
        assert line.startswith("Target: a replay of the frozen fold rules") and "C3" not in line
        assert "converge to the optimal rule" not in line


# ------------------------------------------------------------------------ the assessments


class TestTheAssessments:
    """The analyses that hold the fold rules fixed refuse; the refutations run."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit(estimands=[OPTIMAL_RULE_TARGET])

    def test_the_bound_refuses(self, result: Any) -> None:
        assert fit_wide_bound_refusal(result) is OPTIMAL_RULE_SENSITIVITY_REFUSAL
        assert benchmark_refusal(result, ["W2"]) is OPTIMAL_RULE_SENSITIVITY_REFUSAL
        for operation in ("omitted_confounding", "robustness_value", "benchmark", "contour"):
            row = result.sensitivity.capability(operation)
            assert not row.available, operation
            assert "X11 (f3) in docs/roadmap.md" in (row.reason or ""), operation
        assert fit_wide_bound_refusal(support.fit()) is None

    def test_simulated_confounding_refuses(self, result: Any) -> None:
        refusal = simulated_confounding_refusal(result, OPTIMAL)
        assert refusal is OPTIMAL_RULE_SENSITIVITY_REFUSAL
        assert simulated_confounding_refusal(support.fit(), LEARNED) is not refusal
        # At repeats above 1 the optimal refusal comes first: the splits refusal's remedy,
        # a refit at repeats=1, would be refused by the optimal rule.
        _, repeated = _pairs("repeats")
        assert simulated_confounding_refusal(repeated, OPTIMAL) is refusal

    def test_the_remedy_runs(self, result: Any) -> None:
        learned = support.fit()
        assert fit_wide_bound_refusal(learned) is None
        assert learned.sensitivity.capability("omitted_confounding").available

    def test_the_evalue_keeps_its_row(self, result: Any) -> None:
        with pytest.raises(CapabilityError, match="F27"):
            evalue(result, OPTIMAL)

    def test_the_refutations_run(self, result: Any) -> None:
        tests = ("placebo", "random_common_cause", "subset", "bootstrap_stability")
        assert refute_refusal(result, estimand=OPTIMAL, tests=tests) is None
