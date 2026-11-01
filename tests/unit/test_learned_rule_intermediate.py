"""The learned rule at the treatment with the intermediate held at each level (X11 (j)).

``TMLE(learned_rule=LearnedRule()).fit(..., intermediate="Z")`` reports one result per level
``z`` of the binary intermediate, as every controlled-direct-effect fit does.  Each level learns
its own fold rules: the plug-in rule of :math:`\\bar Q_{nv}(1, z, w) - \\bar Q_{nv}(0, z, w)`,
the best treatment given :math:`Z = z`, not the best pair.  The clever covariate is
:math:`1\\{A = d^z_i\\}1\\{Z = z\\}/(g(A \\mid W) q(z \\mid A, W))`, and the target per level is
:math:`V^{-1}\\sum_v E_0 \\bar Q_0(d^z_{nv}(W), z, W)`.

The witnesses are the plan's (``X11-wave2-plan.md`` rev 2.1, section 7):

- W-j1 (reduction): with a stub outcome regression whose blip has the sign of ``W1`` at both
  levels, every fold rule is ``1{W1 > 0}``, and each level's fit equals the controlled-direct-
  effect fit of ``Rule(1{W1 > 0}, known)`` at that level, bit for bit, at a nonzero epsilon.
- W-j2 (nonzero): with a stub whose blip at ``Z = 1`` has the sign of ``W1 - 0.5``, the rules
  at the two levels differ on a declared share of rows, and the level-1 fit equals the known
  rule ``1{W1 > 0.5}`` while a fit that ignores ``Z`` (the level-0 rule at level 1) does not.
- W-j3 (mutation): the targeted fit solves the score of the clever covariate with
  :math:`1\\{Z = z\\}/q`, and the covariate without that factor leaves a declared nonzero score,
  at two arms and over three, where ``g`` and ``q`` are read at each row's own arm.

Rule L4 runs per level: a longhand refit of each fold's outcome regression gives the stored
fold rules at each level, pooled, per arm and over three arms.  A Gateaux check at an exact
law shows the in-sample curve of each level is the influence function of the value of its
rule.  Every composition the contract names has a test here: admitted and experimental, or
refused with a remedy.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import CausalStudy, LearnedRuleValue, OptimalRuleValue, PointTreatment, load
from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, InSampleLearnerWarning, RefusalReason
from cleverly.interventions import LearnedRule, Rule, Static
from cleverly.interventions.learned import (
    freeze_learned_rule,
    frozen_level_refusal,
    frozen_rule_replay,
)
from cleverly.sensitivity import omitted_variable_bounds
from cleverly.validation import refute
from cleverly.validation.refute import _placebo_null, refute_refusal
from tests import discrete_law_cde as exact
from tests.conftest import OracleDirectOutcome, OracleIntermediate, OracleTreatment

#: The reported name of the learned-rule value at each level.
NAME = "ey_learned_rule[learned rule]"
LEVELS = (0.0, 1.0)
COVARIATES = ["W1", "W2"]
FIT: dict[str, Any] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": COVARIATES,
    "intermediate": "Z",
}

#: W-j1: the declared floor of each level's epsilon (observed 0.035 to 0.16 over the schemes).
W_J1_EPSILON = 1e-2
#: W-j2: the declared share of rows on which the two levels' stub rules differ (observed 0.25).
W_J2_SHARE = 0.15
#: W-j2: the declared gap between the level-1 value and the value of the level-0 rule at
#: level 1, the fit that ignores Z (observed 0.019).
W_J2_GAP = 1e-2
#: W-j3: the declared floor of the score of the covariate without 1{Z = z} / q on the W-j1 stub
#: fit (observed 0.026 and 0.048 at the two levels).
W_J3_UNSOLVED = 1.5e-2
#: W-j3 over three arms: the declared floor of the same score on the learner fit of
#: :func:`_three_arms` (observed 0.013 and 0.011 at the two levels).
W_J3_UNSOLVED_ARMS = 5e-3
#: The declared share of rows on which the learner's rules at the two levels differ on the
#: fixture law (observed 0.36), and over three arms (observed 0.16).
LEVEL_RULE_SHARE = 0.2
ARMS_RULE_SHARE = 0.1


def cde_frame(n: int = 400, seed: int = 0, *, arms: int = 2) -> pd.DataFrame:
    """Draw a law whose blip of ``A`` changes with the intermediate.

    ``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, ``logit g0 = 0.3 W1 - 0.2 W2`` at two arms (a
    uniform draw over ``arms`` arms otherwise), ``logit P(Z = 1 | A, W) = 0.2 + 0.6 A -
    0.4 W1`` and ``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + 0.3 Z + A (0.1 + W1 - 0.9 Z)``.  The
    blip at ``Z = 0`` changes sign at ``W1 = -0.1`` and at ``Z = 1`` at ``W1 = 0.8``, so the
    two levels have different rules.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    if arms == 2:
        a = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    else:
        a = rng.integers(0, arms, n).astype(float)
    treated = (a > 0).astype(float)
    z = rng.binomial(1, expit(0.2 + 0.6 * treated - 0.4 * w1)).astype(float)
    effect = a * (0.1 + w1 - 0.9 * z) if arms == 2 else a * (0.1 + w1) * (1.0 - 1.2 * z)
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 - 0.3 * w2 + 0.3 * z + effect)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Z": z, "Y": y})


def outcome_learner() -> Pipeline:
    """The outcome learner on ``(A, W1, W2, Z)`` with every pairwise interaction."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(max_iter=5000)),
        ]
    )


class LevelStub(ClassifierMixin, BaseEstimator):
    """A fixed outcome regression on ``(A, W1, W2, Z)``: ``logit = 0.4 w1 + 0.3 z + a (w1 - s z)``.

    The blip has the sign of ``w1 - s z``.  With ``shift=0`` it is the sign of ``W1`` at both
    levels (W-j1); with ``shift=0.5`` the rule at ``Z = 1`` is ``1{W1 > 0.5}`` (W-j2).  It
    omits ``W2`` and the intercept of the law, so the fluctuation moves.  It learns nothing.
    """

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LevelStub:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        d = np.asarray(X, dtype=float)
        p = expit(0.4 * d[:, 1] + 0.3 * d[:, 3] + d[:, 0] * (d[:, 1] - self.shift * d[:, 3]))
        return np.column_stack([1.0 - p, p])


def cde_tmle(**overrides: Any) -> TMLE:
    """A fold-evaluated learned-rule estimator with explicit learners and five folds."""
    settings: dict[str, Any] = {
        "learned_rule": LearnedRule(),
        "outcome_learner": outcome_learner(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "intermediate_learner": LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "simultaneous": False,
        "random_state": 0,
        "cross_fit": True,
        "cv_evaluation": True,
    }
    settings.update(overrides)
    return TMLE(**settings)


def fit_cde(frame: pd.DataFrame | None = None, fit: dict[str, Any] | None = None, **overrides: Any):
    """Fit :func:`cde_tmle` on ``frame`` and return the result set, keyed by level."""
    frame = cde_frame() if frame is None else frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InSampleLearnerWarning)
        return cde_tmle(**overrides).fit(frame, **(FIT if fit is None else fit))


def _rule(threshold: float) -> Rule:
    return Rule(
        lambda w: (w["W1"] > threshold).astype(float), name=f"w1>{threshold:g}", rule_kind="known"
    )


def _known_fit(threshold: float, frame: pd.DataFrame | None = None, **overrides: Any):
    """The shipped controlled-direct-effect fit of the known rule ``1{W1 > threshold}``."""
    return fit_cde(frame, learned_rule=None, interventions=[_rule(threshold)], **overrides)


def _assert_bit_identical(a: Any, b: Any) -> None:
    """Two estimates with the same point, variance and curve, bit for bit."""
    assert a.psi == b.psi
    assert a.variance == b.variance
    np.testing.assert_array_equal(a.influence_curve, b.influence_curve)


@cache
def _planted() -> Any:
    """The learner fit at both levels on the fixture law."""
    return fit_cde()


# --------------------------------------------------------------------------- the fixture


class TestTheFixtureHasTeeth:
    def test_each_level_has_its_own_rules(self) -> None:
        results = _planted()
        assert set(results) == set(LEVELS)
        rules = [results[z].nuisance.regimes.fold_rules for z in LEVELS]
        assert rules[0].shape == (400, 5)
        assert float(np.mean(rules[0] != rules[1])) > LEVEL_RULE_SHARE

    def test_the_two_levels_report_different_values(self) -> None:
        results = _planted()
        assert abs(results[0.0].estimates[NAME].psi - results[1.0].estimates[NAME].psi) > 1e-2

    def test_the_intermediate_mechanism_is_not_constant(self) -> None:
        """Without a varying q the 1{Z = z} / q factor of W-j3 would be a constant."""
        q = np.asarray(_planted()[0.0].nuisance.intermediate, dtype=float)
        assert float(np.ptp(q)) > 0.2

    def test_the_nuisances_are_shared_across_levels(self) -> None:
        """One fold loop serves both levels: the mechanisms and folds are the same arrays."""
        results = _planted()
        a, b = (results[z].nuisance for z in LEVELS)
        np.testing.assert_array_equal(a.propensity.values, b.propensity.values)
        np.testing.assert_array_equal(a.intermediate, b.intermediate)
        np.testing.assert_array_equal(a.folds.assignment, b.folds.assignment)


# ------------------------------------------------------------------------------- W-j1


SCHEMES: dict[str, dict[str, Any]] = {
    "fold_evaluated": {},
    "stacked": {"cv_evaluation": False},
    "in_sample": {"cross_fit": False, "cv_evaluation": False},
    "fold_targeted": {"targeting_scheme": "fold"},
}


@cache
def _reduction(scheme: str, shift: float = 0.0, threshold: float = 0.0) -> tuple[Any, Any]:
    """The stub learned-rule fit and the shipped known-rule fit, on one draw."""
    settings = {"outcome_learner": LevelStub(shift), **SCHEMES[scheme]}
    learned = fit_cde(**settings)
    known = _known_fit(threshold, **settings)
    return learned, known


@pytest.mark.parametrize("scheme", list(SCHEMES))
class TestWj1Reduction:
    def test_the_rule_and_the_regime_densities_are_the_known_fit_s(self, scheme: str) -> None:
        learned, known = _reduction(scheme)
        for z in LEVELS:
            np.testing.assert_array_equal(
                learned[z].nuisance.regimes.values, known[z].nuisance.regimes.values
            )
            rule = (learned[z].data.covariates[:, 0] > 0).astype(np.int8)
            fold_rules = learned[z].nuisance.regimes.fold_rules
            np.testing.assert_array_equal(
                fold_rules, np.repeat(rule[:, None], 5 if scheme != "in_sample" else 1, axis=1)
            )

    def test_the_fluctuation_moves_at_each_level(self, scheme: str) -> None:
        learned, _ = _reduction(scheme)
        for z in LEVELS:
            epsilon = np.asarray(learned[z].fluctuations["regime"].epsilon, dtype=float)
            assert float(np.max(np.abs(epsilon))) > W_J1_EPSILON

    def test_each_level_equals_the_known_rule_bit_for_bit(self, scheme: str) -> None:
        learned, known = _reduction(scheme)
        for z in LEVELS:
            _assert_bit_identical(learned[z].estimates[NAME], known[z].estimates["ey_regime[w1>0]"])


class TestWj1Compositions:
    """The reduction beside known regimes, under baseline strata and over three splits."""

    def test_beside_a_known_regime_every_name_equals_the_regime_fit(self) -> None:
        learned = fit_cde(outcome_learner=LevelStub(), interventions=[Static(1)])
        known = fit_cde(
            outcome_learner=LevelStub(),
            learned_rule=None,
            interventions=[_rule(0.0), Static(1)],
            reference="always 1",
        )
        names = {
            NAME: "ey_regime[w1>0]",
            "ey_known_regime[always 1]": "ey_regime[always 1]",
            "ate_learned_rule[learned rule vs always 1]": "ate_regime[w1>0 vs always 1]",
        }
        for z in LEVELS:
            assert set(learned[z].estimates) == set(names)
            for mine, theirs in names.items():
                _assert_bit_identical(learned[z].estimates[mine], known[z].estimates[theirs])
            assert "known_regimes" in learned[z].config.support_key.qualifiers

    def test_with_baseline_strata_under_the_stacked_report(self) -> None:
        frame = cde_frame().assign(S=lambda d: (d["W1"] > 0.3).astype(int))
        fit = {**FIT, "covariates": ["W1", "W2", "S"], "strata": "S"}
        stub = {"outcome_learner": StrataStub(), "cv_evaluation": False}
        learned = fit_cde(frame, fit, **stub)
        known = fit_cde(frame, fit, learned_rule=None, interventions=[_rule(0.0)], **stub)
        for z in LEVELS:
            estimates = learned[z].estimates
            assert len(estimates) == 3
            for name, estimate in estimates.items():
                _assert_bit_identical(
                    estimate, known[z].estimates[name.replace(NAME, "ey_regime[w1>0]")]
                )

    def test_repeats_report_the_mean_of_the_single_split_fits_at_each_level(self) -> None:
        repeated = fit_cde(repeats=3)
        for z in LEVELS:
            seeds = [repeat.seed for repeat in repeated[z].repeats]
            singles = [fit_cde(random_state=seed)[z].estimates[NAME].psi for seed in seeds]
            assert repeated[z].estimates[NAME].psi == pytest.approx(np.mean(singles), rel=1e-12)
            assert repeated[z].extra["learned_rule"].n_splits == 3
        assert [r.seed for r in repeated[0.0].repeats] == [r.seed for r in repeated[1.0].repeats]


class StrataStub(LevelStub):
    """:class:`LevelStub` on the design ``(A, W1, W2, S, Z)`` of a stratified fit."""

    def predict_proba(self, X: Any) -> Any:
        d = np.asarray(X, dtype=float)
        return super().predict_proba(d[:, [0, 1, 2, 4]])


# ------------------------------------------------------------------------------- W-j2


class TestWj2LevelsDiffer:
    def test_the_rules_at_the_two_levels_differ_on_the_declared_share(self) -> None:
        learned, _ = _reduction("fold_evaluated", 0.5, 0.5)
        w1 = learned[0.0].data.covariates[:, 0]
        for z, threshold in zip(LEVELS, (0.0, 0.5), strict=True):
            rules = learned[z].nuisance.regimes.fold_rules
            np.testing.assert_array_equal(rules[:, 0], (w1 > threshold).astype(np.int8))
        differ = (
            learned[0.0].nuisance.regimes.fold_rules != learned[1.0].nuisance.regimes.fold_rules
        )
        assert float(np.mean(differ)) > W_J2_SHARE

    def test_the_level_one_fit_is_the_known_rule_of_its_level(self) -> None:
        learned, known = _reduction("fold_evaluated", 0.5, 0.5)
        _assert_bit_identical(
            learned[1.0].estimates[NAME], known[1.0].estimates["ey_regime[w1>0.5]"]
        )

    def test_control_a_fit_that_ignores_z_fails(self) -> None:
        """The level-0 rule held at level 1, which a rule that ignores Z would give."""
        learned, _ = _reduction("fold_evaluated", 0.5, 0.5)
        ignoring = _known_fit(0.0, outcome_learner=LevelStub(0.5))
        gap = learned[1.0].estimates[NAME].psi - ignoring[1.0].estimates["ey_regime[w1>0]"].psi
        assert abs(gap) > W_J2_GAP
        _assert_bit_identical(
            learned[0.0].estimates[NAME], ignoring[0.0].estimates["ey_regime[w1>0]"]
        )


# ------------------------------------------------------------------------------- W-j3


def _scores(result: Any, z: float) -> tuple[float, float, np.ndarray]:
    """The fold-average score with ``H`` and with ``H`` without ``1{Z = z} / q``, and the curve.

    ``H = 1{A = d} 1{Z = z} / (g(A | W) q(z | A, W))`` at the fit's bounds, with ``g`` and ``q``
    read at each row's own arm, so it holds at any number of arms.  The curve is
    ``H (Y - Qbar*) + Qbar*(d, z, W) - psi_v`` on each validation fold.
    """
    data = result.data
    nuisance = result.nuisance
    targeted = result.fluctuations["regime"].targeted
    lower, upper = result.config.g_bounds
    arms = np.asarray(nuisance.propensity.arms, dtype=float)
    a = np.asarray(data.treatment, dtype=float)
    index = np.searchsorted(arms, a)
    assert np.array_equal(arms[index], a)
    rows = np.arange(data.n)
    rule = arms[np.argmax(np.asarray(nuisance.regimes.values[:, :, 0]), axis=1)]
    by_arm = np.column_stack([np.asarray(nuisance.propensity.arm(arm)) for arm in arms])
    if arms.size == 2:
        # The binary fit bounds g(1 | W) and takes g(0 | W) as its complement.
        g1 = np.clip(by_arm[:, 1], lower, upper)
        by_arm = np.column_stack([1.0 - g1, g1])
    else:
        by_arm = np.clip(by_arm, lower, upper)
    g = by_arm[rows, index]
    density = np.asarray(nuisance.intermediate_density(z, result.config.missingness_bound))
    q = density[rows, index]
    at_level = (np.asarray(data.intermediate) == z).astype(float)
    on_rule = (a == rule).astype(float)
    h = on_rule * at_level / (g * q)
    h_mutated = on_rule / g
    y = np.asarray(nuisance.scaler.scale(data.outcome), dtype=float)
    residual = y - np.asarray(targeted.observed, dtype=float)
    q_arms = np.column_stack([np.asarray(targeted.arms[arm], dtype=float) for arm in arms])
    q_rule = q_arms[rows, np.searchsorted(arms, rule)]
    tests = [np.asarray(test) for _, test in nuisance.folds]
    curve = np.empty(data.n)
    for test in tests:
        curve[test] = (h * residual)[test] + q_rule[test] - np.mean(q_rule[test])

    def score(weights: np.ndarray) -> float:
        return float(np.mean([np.mean((weights * residual)[test]) for test in tests]))

    return score(h), score(h_mutated), curve


class TestWj3Mutation:
    @pytest.mark.parametrize("z", LEVELS)
    def test_the_score_of_the_clever_covariate_is_solved(self, z: float) -> None:
        for result in (_planted()[z], _reduction("fold_evaluated")[0][z], _three_arms()[z]):
            solved, _, _ = _scores(result, z)
            assert abs(solved) < 1e-10

    @pytest.mark.parametrize("z", LEVELS)
    def test_control_dropping_the_intermediate_factor_leaves_a_score(self, z: float) -> None:
        _, mutated, _ = _scores(_reduction("fold_evaluated")[0][z], z)
        assert abs(mutated) > W_J3_UNSOLVED

    @pytest.mark.parametrize("z", LEVELS)
    def test_control_over_three_arms_dropping_the_factor_leaves_a_score(self, z: float) -> None:
        _, mutated, _ = _scores(_three_arms()[z], z)
        assert abs(mutated) > W_J3_UNSOLVED_ARMS

    @pytest.mark.parametrize("z", LEVELS)
    def test_the_curve_and_variance_are_the_longhand_ones(self, z: float) -> None:
        result = _planted()[z]
        _, _, curve = _scores(result, z)
        tests = [np.asarray(test) for _, test in result.nuisance.folds]
        variance = np.sum([np.sum(curve[t] ** 2) / t.size**2 for t in tests]) / len(tests) ** 2
        assert result.estimates[NAME].variance == pytest.approx(float(variance), rel=1e-12)

    def test_the_estimate_is_the_mean_of_the_fold_values(self) -> None:
        """The target V^-1 sum_v Psi_{d^z_v}: the record's fold values average to the estimate."""
        for z in LEVELS:
            result = _planted()[z]
            (split,) = result.extra["learned_rule"].splits
            assert np.mean(split.fold_estimates) == pytest.approx(
                result.estimates[NAME].psi, abs=1e-12
            )


# ------------------------------------------------------------------------------- rule L4


def _longhand_rules(result: Any, learner: Any, z: float, *, by_arm: bool = False) -> np.ndarray:
    """Refit each fold's outcome regression on its training rows and take its rule at ``z``."""
    data = result.data
    arms = data.arm_codes
    y = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    columns = []
    for train, _ in result.nuisance.folds:
        train = np.asarray(train)
        if by_arm:
            design = np.column_stack([data.covariates, data.intermediate])
            at_z = np.column_stack([data.covariates, np.full(data.n, z)])
            predictions = []
            for arm in arms:
                rows = train[np.asarray(data.treatment)[train] == arm]
                model = fit_on_rows(learner, design, y, weights, rows, "classification", None)
                predictions.append(model.predict_proba(at_z)[:, 1])
        else:
            design = data.treatment_design(include_intermediate=True)
            model = fit_on_rows(learner, design, y, weights, train, "classification", None)
            predictions = [
                model.predict_proba(data.counterfactual_design(arm, intermediate_value=z))[:, 1]
                for arm in arms
            ]
        columns.append(np.argmax(np.vstack(predictions), axis=0))
    return np.column_stack(columns).astype(np.int8)


class TestRuleL4PerLevel:
    @pytest.mark.parametrize("z", LEVELS)
    def test_the_pooled_fold_rules_are_the_longhand_refit_s(self, z: float) -> None:
        result = _planted()[z]
        np.testing.assert_array_equal(
            result.nuisance.regimes.fold_rules, _longhand_rules(result, outcome_learner(), z)
        )

    @pytest.mark.parametrize("z", LEVELS)
    def test_the_per_arm_fold_rules_are_the_longhand_refit_s(self, z: float) -> None:
        results = _by_arm()
        result = results[z]
        np.testing.assert_array_equal(
            result.nuisance.regimes.fold_rules,
            _longhand_rules(result, _per_arm_learner(), z, by_arm=True),
        )
        other = results[1.0 - z].nuisance.regimes.fold_rules
        assert float(np.mean(other != result.nuisance.regimes.fold_rules)) > LEVEL_RULE_SHARE

    @pytest.mark.parametrize("z", LEVELS)
    def test_three_arms_take_the_argmax_at_each_level(self, z: float) -> None:
        results = _three_arms()
        result = results[z]
        rules = result.nuisance.regimes.fold_rules
        np.testing.assert_array_equal(rules, _longhand_rules(result, outcome_learner(), z))
        assert set(np.unique(rules)) == {0, 1, 2}
        assert result.config.support_key.treatment == ("arms3",)
        assert result.estimates[NAME].support.label == "experimental"
        other = results[1.0 - z].nuisance.regimes.fold_rules
        assert float(np.mean(other != rules)) > ARMS_RULE_SHARE

    def test_the_realized_rule_is_the_diagonal_at_each_level(self) -> None:
        for z in LEVELS:
            result = _planted()[z]
            regimes = result.nuisance.regimes
            treated = np.asarray(regimes.values[:, 1, 0], dtype=float)
            for fold, (_, test) in enumerate(result.nuisance.folds):
                np.testing.assert_array_equal(treated[test], regimes.fold_rules[test, fold])


def _per_arm_learner() -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(max_iter=5000)),
        ]
    )


@cache
def _by_arm() -> Any:
    return fit_cde(outcome_by_arm=True, outcome_learner=_per_arm_learner())


@cache
def _three_arms() -> Any:
    return fit_cde(cde_frame(600, 1, arms=3))


# ---------------------------------------------------------------------- exact law (Gateaux)


#: The observed rows of :mod:`tests.discrete_law_cde`: a complete-outcome law, exactly.
_EXACT_PROBS = exact.PROBS.copy()
_EXACT_PROBS[..., exact.UNOBSERVED] = 0.0
_EXACT_PROBS /= _EXACT_PROBS.sum()


def _exact_value(probs: Any, level: int, rule: np.ndarray, *, marginal_z: bool = False) -> Any:
    """``sum_w p(w) Qbar(d(w), z, w)``, or with ``Z`` marginalised over ``q(. | d, w)``."""
    p = probs[..., : exact.UNOBSERVED]
    p_w = p.sum(axis=(1, 2, 3))
    joint = p.sum(axis=3)
    q = p[..., exact.OBSERVED_ONE] / joint
    value = 0.0
    for w in range(3):
        a = int(rule[w])
        if marginal_z:
            share = joint[w, a] / joint[w, a].sum()
            value = value + p_w[w] * (share * q[w, a]).sum()
        else:
            value = value + p_w[w] * q[w, a, level]
    return value


def _exact_gateaux(level: int, rule: np.ndarray, point: tuple[int, ...], **kwargs: Any) -> float:
    step = 1e-30
    base = _EXACT_PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(_exact_value(perturbed, level, rule, **kwargs)) / step)


@cache
def _exact_fit() -> Any:
    dgp = exact.DiscreteLaw(_EXACT_PROBS)
    frame = exact.frame()
    frame = frame.loc[frame["Delta"] == 1.0, ["W", "A", "Z", "Y"]].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InSampleLearnerWarning)
        return TMLE(
            learned_rule=LearnedRule(),
            outcome_learner=OracleDirectOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            intermediate_learner=OracleIntermediate(dgp),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W"], intermediate="Z")


class TestTheExactLaw:
    """At oracle nuisances each level's in-sample curve is the influence function of its value."""

    def _pieces(self, z: float) -> tuple[Any, np.ndarray, np.ndarray]:
        result = _exact_fit()[z]
        w = np.rint(result.data.covariates[:, 0]).astype(int)
        treated = np.asarray(result.nuisance.regimes.values[:, 1, 0], dtype=float)
        rule = np.array([treated[np.flatnonzero(w == cell)[0]] for cell in range(3)])
        return result, w, rule

    @pytest.mark.parametrize("z", LEVELS)
    def test_the_rule_is_the_best_arm_given_z(self, z: float) -> None:
        _, _, rule = self._pieces(z)
        dgp = exact.DiscreteLaw(_EXACT_PROBS)
        best = np.argmax(dgp.q[:, :, int(z)], axis=1)
        np.testing.assert_array_equal(rule, best)

    def test_the_two_levels_learn_different_rules(self) -> None:
        assert not np.array_equal(self._pieces(0.0)[2], self._pieces(1.0)[2])

    @pytest.mark.parametrize("z", LEVELS)
    def test_the_curve_is_the_gateaux_derivative_at_each_point(self, z: float) -> None:
        result, w, rule = self._pieces(z)
        data = result.data
        a = np.rint(np.asarray(data.treatment)).astype(int)
        zz = np.rint(np.asarray(data.intermediate)).astype(int)
        y = np.rint(np.asarray(data.outcome)).astype(int)
        curve = np.asarray(result.estimates[NAME].influence_curve, dtype=float)
        for point in set(zip(w, a, zz, y, strict=True)):
            rows = (w == point[0]) & (a == point[1]) & (zz == point[2]) & (y == point[3])
            expected = _exact_gateaux(int(z), rule, point)
            np.testing.assert_allclose(curve[rows], expected, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("z", LEVELS)
    def test_control_the_value_that_lets_z_follow_a_is_caught(self, z: float) -> None:
        """The total-effect functional: Z drawn from q(. | d, W) instead of held at z."""
        result, w, rule = self._pieces(z)
        data = result.data
        a = np.rint(np.asarray(data.treatment)).astype(int)
        zz = np.rint(np.asarray(data.intermediate)).astype(int)
        y = np.rint(np.asarray(data.outcome)).astype(int)
        curve = np.asarray(result.estimates[NAME].influence_curve, dtype=float)
        gaps = []
        for point in set(zip(w, a, zz, y, strict=True)):
            rows = (w == point[0]) & (a == point[1]) & (zz == point[2]) & (y == point[3])
            wrong = _exact_gateaux(int(z), rule, point, marginal_z=True)
            gaps.append(float(np.max(np.abs(curve[rows] - wrong))))
        assert max(gaps) > 1e-2


# ------------------------------------------------------------------------------- refusals


class TestTheRefusals:
    @pytest.mark.parametrize(
        "rule",
        [
            LearnedRule(loss="blip", learner=LinearRegression()),
            LearnedRule(loss="weighted_classification", learner=LogisticRegression()),
        ],
        ids=["blip", "weighted_classification"],
    )
    def test_a_rule_learner_refuses_with_j2_and_the_remedy_runs(self, rule: LearnedRule) -> None:
        with pytest.raises(CapabilityError, match=r"X11 \(j2\)") as caught:
            fit_cde(learned_rule=rule)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "LearnedRule() (loss='plug_in')" in str(caught.value)
        assert "1{Z = z}" in str(caught.value)
        assert set(fit_cde()) == set(LEVELS)

    def test_missing_outcomes_keep_row_5(self) -> None:
        frame = cde_frame().assign(D=lambda d: (np.arange(len(d)) % 10 != 0).astype(int))
        frame.loc[frame["D"] == 0, "Y"] = np.nan
        with pytest.raises(CapabilityError, match="missing outcomes") as caught:
            fit_cde(frame, {**FIT, "delta": "D"})
        assert caught.value.refusal.roadmap == "F27"

    def test_strata_under_cv_evaluation_meet_the_x28_intermediate_refusal(self) -> None:
        frame = cde_frame().assign(S=lambda d: (d["W1"] > 0.3).astype(int))
        fit = {**FIT, "covariates": ["W1", "W2", "S"], "strata": "S"}
        with pytest.raises(CapabilityError, match=r"intermediate variable \(intermediate=\)"):
            fit_cde(frame, fit)
        assert set(fit_cde(frame, fit, cv_evaluation=False)) == set(LEVELS)

    def test_the_value_of_the_optimal_rule_refuses_with_f3(self) -> None:
        with pytest.raises(CapabilityError, match=r"X11 \(f3\)") as caught:
            fit_cde(estimands=["ey_optimal_rule"])
        assert "estimands=None" in str(caught.value)

    def test_the_bootstrap_keeps_its_row(self) -> None:
        with pytest.raises(CapabilityError, match="full-refit bootstrap"):
            fit_cde(n_bootstrap=10)

    def test_the_omitted_variable_bound_refuses_by_name_and_the_remedy_runs(self) -> None:
        result = _planted()[1.0]
        with pytest.raises(CapabilityError, match=r"X11 \(n2\)") as caught:
            omitted_variable_bounds(result, NAME, cf_y=0.05, cf_d=0.05)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "with an intermediate" in str(caught.value)
        known = _known_fit(0.0)[1.0]
        bound = omitted_variable_bounds(known, "ey_regime[w1>0]", cf_y=0.05, cf_d=0.05)
        assert bound.max_bias > 0.0

    def test_the_bound_refuses_beside_known_regimes_for_every_name(self) -> None:
        result = fit_cde(interventions=[Static(1)])[0.0]
        for name in result.estimates:
            with pytest.raises(CapabilityError, match=r"X11 \(n2\)"):
                omitted_variable_bounds(result, name, cf_y=0.05, cf_d=0.05)

    def test_the_method_layer_has_no_spelling(self) -> None:
        design = PointTreatment(
            outcome="Y", treatment="A", adjustment=tuple(COVARIATES), intermediate="Z"
        )
        study = CausalStudy(cde_frame(), design=design)
        with pytest.raises(CapabilityError, match=r"X11 \(j3\)") as caught:
            study.identify(LearnedRuleValue())
        assert "TMLE(learned_rule=LearnedRule())" in str(caught.value)
        with pytest.raises(CapabilityError, match=r"X11 \(f3\)"):
            study.identify(OptimalRuleValue())
        with pytest.raises(CapabilityError, match=r"X11 \(j2\)"):
            study.identify(
                LearnedRuleValue(rule=LearnedRule(loss="blip", learner=LinearRegression()))
            )

    def test_the_e_value_refuses(self) -> None:
        from cleverly.sensitivity.evalue import evalue

        with pytest.raises(CapabilityError, match="learned-rule fit"):
            evalue(_planted()[0.0], NAME)

    def test_simulated_confounding_refuses_an_engine_fit(self) -> None:
        from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding

        with pytest.raises(CapabilityError):
            simulated_confounding(
                _planted()[0.0],
                estimand=NAME,
                grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.2)),
            )


class TestTheFrozenRule:
    def test_freeze_keeps_the_level(self) -> None:
        for z in LEVELS:
            frozen = freeze_learned_rule(_planted()[z])
            assert frozen.intermediate_level == z

    def test_a_replay_at_its_level_reproduces_the_fit(self) -> None:
        result = _planted()[1.0]
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        again = replay._fit_single(result.data, intermediate_value=1.0)
        assert again.estimates[NAME].psi == pytest.approx(result.estimates[NAME].psi, abs=1e-12)
        line = again.extra["learned_rule"].describe()
        assert "holds the intermediate at Z = 1, where the original fit learned it" in line
        assert "reads the outcome regression" not in line
        assert (
            "reads the outcome regression at that level" in result.extra["learned_rule"].describe()
        )

    def test_a_replay_at_another_level_refuses(self) -> None:
        result = _planted()[1.0]
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        with pytest.raises(CapabilityError, match="Each level of the intermediate"):
            replay._fit_single(result.data, intermediate_value=0.0)
        refusal = frozen_level_refusal(replay.learned_rule, None)
        assert refusal is not None
        assert "no intermediate" in refusal.condition

    def test_fit_with_intermediate_refuses_a_frozen_rule(self) -> None:
        result = _planted()[1.0]
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        with pytest.raises(CapabilityError, match="every level"):
            replay.fit(cde_frame(), **FIT)

    def test_the_stability_tests_run_on_a_level(self) -> None:
        outcome = refute(
            _planted()[0.0], estimand=NAME, tests=("random_common_cause", "subset"), n_replicates=2
        )
        assert [test.name for test in outcome.tests] == ["random_common_cause", "subset"]


def strong_z_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """The fixture law with ``logit Qbar0 = -1 + 0.5 W1 + 2.5 Z + A (0.1 + W1 - 0.9 Z)``.

    ``Z`` moves ``Y`` strongly given ``W``, so the value of every rule at level ``z`` under the
    placebo law, ``E_W E(Y | Z = z, W)``, is far from the outcome mean ``E[Y]``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    z = rng.binomial(1, expit(0.2 + 0.6 * a - 0.4 * w1)).astype(float)
    y = rng.binomial(1, expit(-1.0 + 0.5 * w1 + 2.5 * z + a * (0.1 + w1 - 0.9 * z))).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Z": z, "Y": y})


def _strong_z_placebo_values(n: int = 400_000) -> tuple[float, dict[float, float]]:
    """``E[Y]`` and ``E_W E(Y | Z = z, W)`` at each level of :func:`strong_z_frame`'s law.

    The placebo permutes ``A`` alone, so ``E(Y | Z = z, W)`` mixes the arms with
    ``P(A = a | Z = z, W)``.  Both are exact given ``W``, averaged over a large draw of ``W``.
    """
    rng = np.random.default_rng(1)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    g = expit(0.3 * w1 - 0.2 * w2)

    def q1(a: float) -> np.ndarray:
        return expit(0.2 + 0.6 * a - 0.4 * w1)

    def qbar(a: float, z: float) -> np.ndarray:
        return expit(-1.0 + 0.5 * w1 + 2.5 * z + a * (0.1 + w1 - 0.9 * z))

    mean = 0.0
    values = {}
    for z in LEVELS:
        joint = {a: (g if a else 1.0 - g) * (q1(a) if z else 1.0 - q1(a)) for a in (0.0, 1.0)}
        mean += float(np.mean(sum(joint[a] * qbar(a, z) for a in joint)))
        given_z = sum(joint[a] * qbar(a, z) for a in joint) / (joint[0.0] + joint[1.0])
        values[z] = float(np.mean(given_z))
    return mean, values


@cache
def _strong_z() -> Any:
    """The learner fit beside ``always 1`` on :func:`strong_z_frame`, at both levels."""
    return fit_cde(strong_z_frame(), interventions=[Static(1)])


#: The declared gap between ``E[Y]`` and the placebo value at each level on the strong-Z law
#: (observed 0.27 and 0.17).
STRONG_Z_GAP = 0.1
CONTRAST = "ate_learned_rule[learned rule vs always 1]"


class TestThePlaceboWithAnIntermediate:
    """The placebo law gives no rule the value ``E[Y]`` when ``Z`` moves ``Y`` given ``W``."""

    def test_the_law_has_teeth(self) -> None:
        mean, values = _strong_z_placebo_values()
        for z in LEVELS:
            assert abs(values[z] - mean) > STRONG_Z_GAP

    @pytest.mark.parametrize("z", LEVELS)
    @pytest.mark.parametrize("name", [NAME, "ey_known_regime[always 1]"])
    def test_a_level_placebo_refuses(self, z: float, name: str) -> None:
        result = _strong_z()[z]
        assert _placebo_null(result, name) is None
        refusal = refute_refusal(result, estimand=name, tests=("placebo",))
        assert refusal is not None
        assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "E_W E(Y | Z = z, W)" in refusal.condition
        assert "random_common_cause" in refusal.alternative
        with pytest.raises(CapabilityError, match="no exact null"):
            refute(result, estimand=name, tests=("placebo",), n_replicates=2)

    def test_the_default_tests_refuse_on_a_level(self) -> None:
        with pytest.raises(CapabilityError, match="no exact null"):
            refute(_strong_z()[1.0], estimand=NAME)

    def test_a_stratum_alias_refuses(self) -> None:
        frame = cde_frame().assign(S=lambda d: (d["W1"] > 0.3).astype(int))
        fit = {**FIT, "covariates": ["W1", "W2", "S"], "strata": "S"}
        result = fit_cde(frame, fit, outcome_learner=StrataStub(), cv_evaluation=False)[0.0]
        aliases = [name for name in result.estimates if name != NAME]
        assert aliases
        for name in aliases:
            refusal = refute_refusal(result, estimand=name, tests=("placebo",))
            assert refusal is not None
            assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_the_contrast_placebo_runs_against_zero(self) -> None:
        result = _strong_z()[1.0]
        assert _placebo_null(result, CONTRAST) == 0.0
        assert refute_refusal(result, estimand=CONTRAST, tests=("placebo",)) is None
        test = refute(result, estimand=CONTRAST, tests=["placebo"], n_replicates=3).tests[0]
        assert abs(float(np.mean(test.values))) < 0.1

    def test_control_without_an_intermediate_the_level_placebo_runs(self) -> None:
        frame = strong_z_frame()
        fit = {key: value for key, value in FIT.items() if key != "intermediate"}
        result = fit_cde(frame, {**fit, "covariates": [*COVARIATES, "Z"]})[None]
        assert _placebo_null(result, NAME) == pytest.approx(float(np.mean(frame["Y"])))
        assert refute_refusal(result, estimand=NAME, tests=("placebo",)) is None


# -------------------------------------------------------------------- label and compositions


class TestTheLabel:
    def test_every_level_reads_experimental_with_no_row(self) -> None:
        for z in LEVELS:
            result = _planted()[z]
            key = result.config.support_key
            assert key.missing == ("intermediate",)
            assert key.family == frozenset({"P11"})
            support = result.estimates[NAME].support
            assert support.label == "experimental"
            assert support.row is None

    @pytest.mark.parametrize(
        ("fit", "settings"),
        [
            ({"weights": "WT"}, {}),
            ({"id": "C"}, {}),
            ({}, {"outcome_by_arm": True}),
            ({}, {"cv_evaluation": False}),
            ({}, {"cross_fit": False, "cv_evaluation": False}),
            ({}, {"targeting_scheme": "fold"}),
            ({}, {"repeats": 2}),
            ({"treatment_probabilities": "G"}, {}),
        ],
        ids=[
            "weights",
            "clusters",
            "by_arm",
            "stacked",
            "in_sample",
            "fold_targeted",
            "repeats",
            "known_mechanism",
        ],
    )
    def test_each_admitted_composition_runs_experimental(
        self, fit: dict[str, Any], settings: dict[str, Any]
    ) -> None:
        rng = np.random.default_rng(7)
        frame = cde_frame()
        frame = frame.assign(
            WT=rng.uniform(0.5, 1.5, 400),
            C=np.arange(400) % 80,
            G=expit(0.3 * frame["W1"] - 0.2 * frame["W2"]),
        )
        if "treatment_probabilities" in fit:
            g1 = frame["G"].to_numpy()
            fit = {"treatment_probabilities": np.column_stack([1.0 - g1, g1])}
        results = fit_cde(frame, {**FIT, **fit}, **settings)
        for z in LEVELS:
            result = results[z]
            key = result.config.support_key
            assert key.missing == ("intermediate",)
            assert result.estimates[NAME].support.label == "experimental"
            assert result.extra["learned_rule"].intermediate_level == z
        key = results[0.0].config.support_key
        if "weights" in fit:
            assert key.weighting != "none"
        if "id" in fit:
            assert key.dependence != "iid"
        if "treatment_probabilities" in fit:
            assert "known_mechanism" in key.qualifiers


# ---------------------------------------------------------------------- record and persistence


class TestTheRecord:
    def test_each_level_has_its_own_record(self) -> None:
        results = _planted()
        records = [results[z].extra["learned_rule"] for z in LEVELS]
        assert [record.intermediate_level for record in records] == list(LEVELS)
        assert {record.intermediate_name for record in records} == {"Z"}
        shares = [record.splits[0].arm_shares for record in records]
        assert shares[0] != shares[1]

    def test_the_summary_names_the_level(self) -> None:
        for z in LEVELS:
            result = _planted()[z]
            line = result.extra["learned_rule"].describe()
            assert f"Every rule holds the intermediate at Z = {z:g}" in line
            assert "not the best pair of treatment and intermediate" in line
            assert line in result.summary(detail="full").section("Method details").lines

    def test_a_fit_without_an_intermediate_names_no_level(self) -> None:
        frame = cde_frame()
        result = fit_cde(frame, {k: v for k, v in FIT.items() if k != "intermediate"})
        record = result.single().extra["learned_rule"]
        assert record.intermediate_level is None
        assert "intermediate" not in record.describe()

    @pytest.mark.parametrize("route", ["save", "dumps"])
    def test_a_round_trip_keeps_each_level(self, route: str, tmp_path: Any) -> None:
        for z in LEVELS:
            result = _planted()[z]
            if route == "save":
                path = tmp_path / f"level{z:g}.cleverly"
                result.save(path)
                back = load(path)
            else:
                back = loads(dumps(result))
            assert back.intermediate_value == z
            assert back.extra["learned_rule"] == result.extra["learned_rule"]
            np.testing.assert_array_equal(
                back.nuisance.regimes.fold_rules, result.nuisance.regimes.fold_rules
            )
            assert back.estimates[NAME].psi == result.estimates[NAME].psi

    def test_retarget_reads_the_level_s_saved_rule(self) -> None:
        for z in LEVELS:
            result = _planted()[z]
            estimates, _ = result.estimator.retarget(
                result.data, result.nuisance, estimands=[NAME.split("[")[0]], intermediate_value=z
            )
            assert estimates[NAME].psi == pytest.approx(result.estimates[NAME].psi, abs=1e-12)

    def test_the_nuisances_carry_one_regime_set_per_level(self) -> None:
        nuisance = _planted()[0.0].nuisance
        assert set(nuisance.regimes_by_level) == set(LEVELS)
        assert nuisance.fold_plug_in_rules_by_level == {}
        for z in LEVELS:
            np.testing.assert_array_equal(
                nuisance.at_level(z).regimes.fold_rules,
                _planted()[z].nuisance.regimes.fold_rules,
            )
