"""P1 part (c): the EP curve as the rule learner of the learned-rule value.

``LearnedRule(loss="ep", modifiers=V)`` learns each outer fold's rule as the sign of the EP
curve (van der Laan, Carone and Luedtke, arXiv:2402.01972v2) of the effect given ``V``, fitted
on the fold's training rows ``T_v`` from the shipped 5-fold inner cross-fit of ``T_v``.  The
outer fluctuation, estimate and variance are the plug-in rule's.

* W-c1 (exact law, nonzero witness): a finite population with a known mechanism and a supplied
  outcome regression that is wrong in the blip.  Every training fold is a whole number of copies
  of the population, so each fold's EP curve is the population EP curve, and with ``g`` exact it
  equals the true conditional effect given ``V`` in every cell to ``1e-10``.  The plug-in rule of
  the same regression has the wrong sign at ``V = 0``.  Five mutations each break it: no
  debiasing, unit ``1/g`` weights, no row weights, an unweighted final stage, and the ``(2A - 1)``
  factor replaced by 1.
* W-c2 (outer fold locality): flipping fold ``v``'s validation outcomes leaves fold ``v``'s curve
  unchanged bit for bit.  The mutation that fits the curve from the outer cross-fitted nuisances
  at ``T_v`` moves it.
* W-c3 (inner fold locality, plan finding 27): flipping the outcomes of inner fold ``j`` of
  ``T_v`` leaves the size step's curves scored on fold ``j`` unchanged.  The mutation whose size
  step fits on every row of ``T_v`` moves them.
* The refusals run before any learner; weights reach every inner, inner-inner and final fit;
  clusters stay whole; the declared final learner is the one fitted.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache, partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from cleverly import SuperLearner
from cleverly.bases import BSpline, Cosine, Modifiers, Sieve
from cleverly.estimators import TMLE
from cleverly.estimators import ep as ep_module
from cleverly.exceptions import CapabilityError, DataError, InSampleLearnerWarning
from cleverly.interventions import LearnedRule
from cleverly.interventions import _rule_learners as rule_learners
from cleverly.interventions._rule_learners import (
    INNER_FOLDS,
    EPRuleRecord,
    fit_fold_rule,
    inner_seed,
)
from cleverly.interventions.base import RegimeSet
from cleverly.interventions.learned import (
    EP_OPTIMAL_MODIFIERS_REFUSAL,
    OPTIMAL_RULE_TARGET,
    RULE_CLASS,
    ep_rule_condition,
    freeze_learned_rule,
)
from cleverly.learners import crossfit
from cleverly.learners.crossfit import random_partition
from tests.studies import _learned_rule_law as law
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The RM30 modifiers: every adjustment column, with the law's domain and levels.
MODIFIERS = Modifiers(["W1", "W2"], domains={"W1": (-1.0, 1.0)}, levels={"W2": (0.0, 1.0)})

#: The modifier ``W1`` alone, narrower than the adjustment set.
W1_ONLY = Modifiers(["W1"], domains={"W1": (-1.0, 1.0)})

#: A declared fixed size, which skips the size step and keeps the fast fits fast.
FIXED = Sieve.default(MODIFIERS, max_size=3, size=2)


def ep_rule(**keywords: Any) -> LearnedRule:
    return LearnedRule(loss="ep", **{"modifiers": MODIFIERS, **keywords})


# --------------------------------------------------------------------------- the learners


class NoWeightRegressor(RegressorMixin, BaseEstimator):
    """A regressor whose ``fit`` takes no ``sample_weight``."""

    def fit(self, X: Any, y: Any) -> NoWeightRegressor:  # pragma: no cover - refused first
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X: Any) -> Any:  # pragma: no cover - refused first
        return self.model_.predict(X)


class SpyFinal(RegressorMixin, BaseEstimator):
    """A linear final stage that records the design and the weights of every fit."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SpyFinal:
        SpyFinal.calls.append({"x": np.asarray(X, dtype=float), "weight": sample_weight})
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)


class WeightSpyOutcome(ClassifierMixin, BaseEstimator):
    """A logistic regression that records the design and the weights of every fit."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WeightSpyOutcome:
        type(self).calls.append({"x": np.asarray(X, dtype=float), "weight": sample_weight})
        self.model_ = LogisticRegression(max_iter=5000).fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)


class WeightSpyTreatment(WeightSpyOutcome):
    """The same spy for the treatment fits, with its own record."""

    calls: ClassVar[list[dict[str, Any]]] = []


# ------------------------------------------------------------------------- the declaration


class TestTheDeclaration:
    """``LearnedRule(loss="ep")`` refuses a declaration that does not suit it, before any data."""

    @pytest.mark.parametrize(
        ("keywords", "match"),
        [
            ({"loss": "ep"}, "needs modifiers="),
            ({"loss": "ep", "modifiers": ["W1"]}, "needs modifiers="),
            ({"modifiers": MODIFIERS}, "takes no modifiers="),
            (
                {"loss": "blip", "learner": LinearRegression(), "sieve": FIXED},
                "takes no sieve=",
            ),
            ({"loss": "ep", "modifiers": MODIFIERS, "sieve": "cv"}, "takes a cleverly.bases"),
            (
                {"loss": "ep", "modifiers": MODIFIERS, "learner": LogisticRegression()},
                "is a classifier",
            ),
            (
                {
                    "loss": "ep",
                    "modifiers": MODIFIERS,
                    "learner": SuperLearner(task="classification"),
                },
                "classifier",
            ),
            (
                {"loss": "ep", "modifiers": MODIFIERS, "learner": SuperLearner(clip=(0.0, 1.0))},
                "clips its predictions",
            ),
        ],
    )
    def test_a_declaration_that_does_not_suit_the_loss_refuses(
        self, keywords: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(CapabilityError, match=match) as caught:
            LearnedRule(**keywords)
        assert caught.value.refusal.reason.value == "invalid_declaration"

    def test_the_admitted_declarations(self) -> None:
        rule = ep_rule()
        assert (rule.loss, rule.learner, rule.sieve) == ("ep", None, None)
        ep_rule(sieve=FIXED, learner=LinearRegression())
        ep_rule(learner=NoWeightRegressor())  # admitted: an unweighted fit needs no weights


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    return support.law_frame(n, seed)


def _refused(rule: LearnedRule, frame: pd.DataFrame | None = None, **fit: Any) -> Any:
    """Fit ``rule`` with learners that fail if fitted; return the refusal, which came first."""
    estimator = support.learned_tmle(learned_rule=rule, **never_fit_learners())
    NeverFit.calls = 0
    frame = _frame() if frame is None else frame
    with pytest.raises((CapabilityError, DataError)) as caught:
        estimator.fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, **fit)
    assert NeverFit.calls == 0
    return caught.value


class TestTheRefusalsComeBeforeAnyLearner:
    def test_a_modifier_outside_the_adjustment_set_refuses_r6(self) -> None:
        frame = _frame().assign(Z=lambda f: f["W1"] ** 2)
        rule = LearnedRule(loss="ep", modifiers=Modifiers(["Z"], domains={"Z": (0.0, 1.0)}))
        error = _refused(rule, frame)
        assert isinstance(error, CapabilityError)
        assert "['Z'] are not adjustment columns" in str(error)
        assert error.refusal.reason.value == "invalid_declaration"

    def test_the_optimal_value_with_modifiers_narrower_than_w_refuses_r12(self) -> None:
        estimator = support.learned_tmle(
            learned_rule=LearnedRule(loss="ep", modifiers=W1_ONLY),
            estimands=[OPTIMAL_RULE_TARGET],
            **never_fit_learners(),
        )
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            estimator.fit(_frame(), outcome="Y", treatment="A", covariates=support.COVARIATES)
        assert NeverFit.calls == 0
        assert caught.value.refusal.reason.value == "different_estimand"
        assert "leave out the adjustment columns ['W2']" in str(caught.value)
        assert EP_OPTIMAL_MODIFIERS_REFUSAL.condition in str(caught.value)

    def test_the_learned_value_with_narrower_modifiers_is_admitted(self) -> None:
        result = support.fit(
            learned_rule=LearnedRule(loss="ep", modifiers=W1_ONLY, sieve=Sieve(Cosine(["W1"], 2)))
        )
        assert np.isfinite(result.estimates[support.NAME].psi)

    def test_more_than_two_arms_refuses_r10(self) -> None:
        frame = _frame().assign(A=lambda f: np.arange(len(f)) % 3)
        error = _refused(ep_rule(), frame)
        assert "loss='ep' learns one curve" in str(error)
        assert "X11 (d2)" in str(error)

    def test_an_intermediate_refuses_r10(self) -> None:
        frame = _frame().assign(Z=lambda f: (np.arange(len(f)) % 2).astype(float))
        error = _refused(ep_rule(), frame, intermediate="Z")
        assert "the EP fit does not build" in str(error)
        assert "X11 (j2)" in str(error)

    def test_a_basis_reading_an_undeclared_domain_refuses_r3(self) -> None:
        rule = LearnedRule(loss="ep", modifiers=Modifiers(["W1"]))
        error = _refused(rule)
        assert "domain" in str(error)

    def test_a_row_outside_a_declared_domain_refuses(self) -> None:
        rule = LearnedRule(loss="ep", modifiers=Modifiers(["W1"], domains={"W1": (-0.5, 0.5)}))
        error = _refused(rule)
        assert isinstance(error, DataError)
        assert "W1" in str(error)

    def test_a_declared_size_above_the_family_refuses(self) -> None:
        error = _refused(ep_rule(sieve=Sieve.default(MODIFIERS, max_size=3, size=5)))
        assert isinstance(error, CapabilityError)
        assert "has at most 3" in str(error)

    def test_a_schedule_above_the_family_refuses(self) -> None:
        error = _refused(ep_rule(sieve=Sieve.default(MODIFIERS, max_size=3, size={1: 4})))
        assert "has at most 3" in str(error)

    def test_a_schedule_is_read_at_each_outer_training_size(self) -> None:
        """At n = 120 the schedule gives 2, but each 5-fold training fold of 96 rows reads 4."""
        sieve = Sieve.default(MODIFIERS, max_size=3, size={1: 4, 110: 2})
        assert sieve.size_at(120) == 2 and sieve.size_at(96) == 4
        error = _refused(ep_rule(sieve=sieve))
        assert isinstance(error, CapabilityError)
        assert "at n = 96" in str(error) and "has at most 3" in str(error)

    def test_a_schedule_with_no_size_at_a_training_fold_refuses(self) -> None:
        error = _refused(ep_rule(sieve=Sieve.default(MODIFIERS, max_size=3, size={100: 2})))
        assert isinstance(error, DataError)
        assert "declares no size at n = 96" in str(error)

    def test_a_collinear_member_refuses(self) -> None:
        frame = _frame().assign(W1=lambda f: np.where(f["W1"] > 0, 0.5, -0.5))
        rule = LearnedRule(loss="ep", modifiers=W1_ONLY, sieve=Sieve(Cosine(["W1"], 3), size=3))
        error = _refused(rule, frame)
        assert isinstance(error, DataError)
        assert "collinear" in str(error)

    @pytest.mark.parametrize(
        "learner",
        [NoWeightRegressor(), SuperLearner(library=[LinearRegression(), KNeighborsRegressor()])],
        ids=["regressor", "superlearner_member"],
    )
    def test_a_weighted_fit_refuses_a_final_learner_without_weights(self, learner: Any) -> None:
        frame = _frame().assign(w=lambda f: 1.0 + 0.5 * f["W2"])
        error = _refused(ep_rule(learner=learner), frame, weights="w")
        assert isinstance(error, CapabilityError)
        assert error.refusal.reason.value == "invalid_declaration"
        assert "sample_weight" in str(error)


# ---------------------------------------------------------------------------- W-c1, exact

#: The population: ``V`` in {0, 1, 2}, ``W2`` in {0, 1}, the treated share of each cell, and
#: the cell masses.
G = {(0, 0): 0.3, (0, 1): 0.6, (1, 0): 0.5, (1, 1): 0.7, (2, 0): 0.4, (2, 1): 0.25}
P_W2 = {0: 0.5, 1: 0.3, 2: 0.6}
P_V = {0: 0.3, 1: 0.3, 2: 0.4}
COPIES = 20


def true_mean(a: float, v: int, w2: int) -> float:
    blip = (-0.6, 0.5, 0.2)[v] + 0.9 * w2 * (v == 1) - 0.8 * w2 * (v == 0)
    return float(expit(-0.2 + 0.3 * v - 0.4 * w2 + a * blip))


def wrong_mean(a: Any, v: Any, w2: Any) -> Any:
    """The supplied outcome regression: wrong in the blip, so its plug-in rule is wrong."""
    return expit(-0.2 + 0.3 * v - 0.4 * w2 + a * 0.3 * (1 - v) + 0.5 * w2 * a)


class WrongRegression(ClassifierMixin, BaseEstimator):
    """A fitted stub that returns :func:`wrong_mean` on the design ``(A, V, W2)``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WrongRegression:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        x = np.asarray(X, dtype=float)
        p = wrong_mean(x[:, 0], x[:, 1], x[:, 2])
        return np.column_stack([1.0 - p, p])


def _cell_effect(mean: Any, v: int) -> float:
    return sum(
        (P_W2[v] if w2 else 1.0 - P_W2[v]) * (mean(1.0, v, w2) - mean(0.0, v, w2)) for w2 in (0, 1)
    )


#: The conditional effect given ``V``, and the regression's wrong one.
THETA = {v: _cell_effect(true_mean, v) for v in range(3)}
WRONG = {v: _cell_effect(wrong_mean, v) for v in range(3)}


@cache
def population() -> pd.DataFrame:
    """:data:`COPIES` copies of the population, one cluster each, rows weighted by their mass."""
    rows = []
    for copy in range(COPIES):
        for v in range(3):
            for w2 in (0, 1):
                cell = P_V[v] * (P_W2[v] if w2 else 1.0 - P_W2[v])
                g = G[v, w2]
                for a in (0, 1):
                    mean = true_mean(a, v, w2)
                    for y in (0, 1):
                        mass = cell * (g if a else 1.0 - g) * (mean if y else 1.0 - mean)
                        rows.append(
                            {"V": v, "W2": w2, "A": a, "Y": y, "w": mass, "c": copy}
                            | {"p1": g, "p0": 1.0 - g}
                        )
    return pd.DataFrame(rows).astype({"V": float, "W2": float, "A": float, "Y": float})


def exact_fit(**rule: Any) -> Any:
    modifiers = Modifiers(["V"], levels={"V": (0.0, 1.0, 2.0)})
    estimator = TMLE(
        learned_rule=LearnedRule(loss="ep", modifiers=modifiers, **rule),
        n_folds=10,
        cv_evaluation=True,
        outcome_learner=WrongRegression(),
        random_state=3,
    )
    frame = population()
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["V", "W2"],
        weights="w",
        id="c",
        treatment_probabilities={1.0: "p1", 0.0: "p0"},
    ).single()


def cell_margins(result: Any) -> dict[int, np.ndarray]:
    """The distinct margins of the fold rules in each ``V`` cell."""
    margin = np.asarray(result.nuisance.regimes.margin, dtype=float)
    v = population()["V"].to_numpy()
    return {cell: np.unique(margin[v == cell]) for cell in range(3)}


def _far_from_theta(result: Any) -> float:
    margins = cell_margins(result)
    return max(float(np.max(np.abs(margins[v] - THETA[v]))) for v in range(3))


class TestWc1TheFoldCurveIsTheConditionalEffect:
    def test_the_declared_law_has_a_wrong_regression(self) -> None:
        assert THETA[0] < -0.15 and THETA[1] > 0.15 and THETA[2] > 0.04
        assert WRONG[0] > 0.1  # the plug-in sign is wrong at V = 0

    def test_every_fold_curve_equals_the_effect_given_v(self) -> None:
        result = exact_fit()
        for v, values in cell_margins(result).items():
            np.testing.assert_allclose(values, THETA[v], rtol=0, atol=1e-10)

    def test_the_plug_in_rule_of_the_same_regression_has_the_wrong_sign(self) -> None:
        plug_in = TMLE(
            learned_rule=LearnedRule(),
            n_folds=10,
            cv_evaluation=True,
            outcome_learner=WrongRegression(),
            random_state=3,
        ).fit(
            population(),
            outcome="Y",
            treatment="A",
            covariates=["V", "W2"],
            weights="w",
            id="c",
            treatment_probabilities={1.0: "p1", 0.0: "p0"},
        )
        at_zero = population()["V"].to_numpy() == 0
        assert law.rule_of(plug_in.single())[at_zero].all()
        assert not law.rule_of(exact_fit())[at_zero].any()

    def test_skipping_the_debiasing_gives_the_regressions_sign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ep_module.debias

        def no_debiasing(**inputs: Any) -> Any:
            fitted = original(**inputs)
            return replace(
                fitted,
                coefficients=0.0 * fitted.coefficients,
                targeted_t=np.asarray(inputs["initial_t"], dtype=float),
                targeted_r=np.asarray(inputs["initial_r"], dtype=float),
            )

        monkeypatch.setattr(ep_module, "debias", no_debiasing)
        margins = cell_margins(exact_fit())
        assert np.all(margins[0] > 0.1)

    @pytest.mark.parametrize("mutation", ["unit_inverse_g", "no_row_weights"])
    def test_a_debiasing_weight_mutation_breaks_the_identity(
        self, mutation: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ep_module.nuisance_pieces

        def mutated(*args: Any, **kwargs: Any) -> Any:
            pieces = original(*args, **kwargs)
            if mutation == "unit_inverse_g":
                return replace(pieces, weight=pieces.rows)
            return replace(pieces, weight=pieces.weight / pieces.rows, rows=0 * pieces.rows + 1)

        monkeypatch.setattr(ep_module, "nuisance_pieces", mutated)
        assert _far_from_theta(exact_fit()) > 1e-3

    def test_an_unweighted_final_stage_breaks_the_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ep_module.final_stage

        def unweighted(debiased: Any, design: Any, weights: Any, *args: Any, **kw: Any) -> Any:
            return original(debiased, design, np.ones_like(weights), *args, **kw)

        monkeypatch.setattr(ep_module, "final_stage", unweighted)
        assert _far_from_theta(exact_fit()) > 1e-3

    def test_dropping_the_arm_sign_breaks_the_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unsigned(design: Any, treatment: Any, pair: Any, scale: str) -> Any:
            return np.asarray(design, dtype=float)

        monkeypatch.setattr(ep_module, "debias_features", unsigned)
        assert _far_from_theta(exact_fit()) > 1e-3


# -------------------------------------------------------------------------- the record


@cache
def _ep_fit() -> Any:
    return support.fit(learned_rule=ep_rule())


@cache
def _fixed_fit() -> Any:
    return support.fit(learned_rule=ep_rule(sieve=FIXED))


def _refit(result: Any, fold: int, rule: LearnedRule, **changes: Any) -> Any:
    """Refit outer fold ``fold``'s rule through ``fit_fold_rule``, as the harness does."""
    (split,) = result.extra["learned_rule"].splits
    train, _ = list(result.nuisance.folds)[fold]
    settings = {
        "loss": "ep",
        "learner": rule.learner,
        "fit_inner": partial(result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler),
        "scaler": result.nuisance.scaler,
        "g_bounds": result.config.g_bounds,
        "seed": split.inner_seeds[fold],
        "modifiers": rule.modifiers,
        "sieve": rule.sieve,
        "alpha": result.estimator.alpha,
        **changes,
    }
    return fit_fold_rule(result.data, np.asarray(train), **settings)


class TestTheRecord:
    def test_the_rule_class_and_the_description_name_the_ep_curve(self) -> None:
        assert set(RULE_CLASS) == {"plug_in", "blip", "weighted_classification", "ep"}
        record = _ep_fit().extra["learned_rule"]
        assert record.loss == "ep"
        assert record.rule == RULE_CLASS["ep"]
        assert "EP curve" in record.rule and "1{theta_v(V) > 0}" in record.rule
        line = record.describe()
        assert "loss='ep'" in line
        assert "EP curve of the effect given the declared modifiers" in line

    def test_the_inner_seeds_are_the_split_seeds(self) -> None:
        result = _ep_fit()
        (split,) = result.extra["learned_rule"].splits
        assert split.inner_seeds == tuple(
            inner_seed(result.repeats[0].seed, fold) for fold in range(5)
        )

    def test_the_realized_rule_is_the_ep_curves_rule(self) -> None:
        result = _ep_fit()
        rule = law.rule_of(result)
        margin = np.asarray(result.nuisance.regimes.margin, dtype=float)
        covariates = np.asarray(result.data.covariates, dtype=float)
        assert not np.array_equal(rule, law.rule_of(support.fit()))
        for fold, (_, test) in enumerate(result.nuisance.folds):
            refit = _refit(result, fold, ep_rule())
            assert refit.loss == "ep" and refit.threshold == 0.0
            np.testing.assert_array_equal(refit.treat(covariates[test]), rule[test])
            np.testing.assert_array_equal(refit.margin(covariates[test]), margin[test])
            record = refit.ep
            assert isinstance(record, EPRuleRecord)
            assert record.mode == "cv"
            assert record.sizes == tuple(range(1, 7))
            assert len(record.cv_risks) == 6
            assert record.chosen == record.sizes[int(np.argmin(record.cv_risks))]
            assert len(set(record.seeds)) == INNER_FOLDS

    def test_a_declared_size_is_read_and_records_no_risks(self) -> None:
        refit = _refit(_fixed_fit(), 0, ep_rule(sieve=FIXED))
        assert refit.ep == EPRuleRecord("fixed", (2,), (), 2)

    def test_the_curve_reads_the_modifiers_only(self) -> None:
        """A rule given ``W1`` alone is constant in ``W2``, whatever the regression says."""
        rule = LearnedRule(loss="ep", modifiers=W1_ONLY, sieve=Sieve(Cosine(["W1"], 2)))
        result = support.fit(learned_rule=rule)
        refit = _refit(result, 0, rule)
        rows = np.column_stack([np.linspace(-0.9, 0.9, 7), np.zeros(7)])
        np.testing.assert_array_equal(refit.score(rows), refit.score(rows + np.array([0.0, 1.0])))

    def test_a_frozen_rule_keeps_the_declaration_and_the_rule(self) -> None:
        result = _ep_fit()
        frozen = freeze_learned_rule(result)
        assert (frozen.loss, frozen.modifiers, frozen.sieve) == ("ep", MODIFIERS, None)
        np.testing.assert_array_equal(frozen.treat, law.rule_of(result))

    def test_the_default_ep_rule_fit_reads_its_registered_row(self) -> None:
        """The ``learned-rule-ep`` study measures the default sieve, so row ``T51`` covers it."""
        label = _ep_fit().estimates[support.NAME].support
        assert label.label == "supported"
        assert label.row == "T51"

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"sieve": FIXED}, "a declared sieve="),
            ({"learner": LinearRegression()}, "a declared final learner (learner=)"),
        ],
        ids=["fixed_size", "final_learner"],
    )
    def test_a_declared_curve_keeps_the_row_and_reads_experimental(
        self, changes: dict[str, Any], named: str
    ) -> None:
        """The key holds neither choice, so a condition, not the row, labels the fit."""
        fitted = (
            _fixed_fit() if "sieve" in changes else support.fit(learned_rule=ep_rule(**changes))
        )
        label = fitted.estimates[support.NAME].support
        assert label.label == "experimental"
        assert "P1" in label.evidence
        (condition,) = fitted.config.support_conditions
        assert condition == ep_rule_condition(fitted.estimator)
        assert named in condition.condition
        assert "learned-rule-ep" in label.reason

    def test_the_explicit_default_sieve_is_the_measured_curve(self) -> None:
        """Mutation control: the condition reads the sieve itself, not whether one was passed."""
        explicit = ep_rule(sieve=Sieve.default(MODIFIERS))
        estimator = support.learned_tmle(learned_rule=explicit)
        assert ep_rule_condition(estimator) is None
        smaller = support.learned_tmle(
            learned_rule=ep_rule(sieve=Sieve.default(MODIFIERS, max_size=5))
        )
        assert ep_rule_condition(smaller) is not None
        in_sample = support.learned_tmle(
            learned_rule=ep_rule(sieve=FIXED), cross_fit=False, cv_evaluation=False
        )
        assert ep_rule_condition(in_sample) is None

    def test_the_c3_text_reads_the_ep_limit(self) -> None:
        from cleverly import CausalStudy, LearnedRuleValue, PointTreatment

        study = CausalStudy(
            support.law_frame(),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        effect = study.identify(LearnedRuleValue(rule=ep_rule()))
        assert "the EP curve of" in effect.functional.expression
        text = " ".join(effect.identification.assumptions)
        assert "For loss='ep' the limit score is the EP limit theta_1" in text
        assert "doubly robust" in text


# ---------------------------------------------------------------- W-c2, outer fold locality


class OuterNuisanceCurve(TMLE):
    """The W-c2 mutation: fold ``v``'s curve is fitted from the outer cross-fitted nuisances.

    At a row of ``T_v`` the outer prediction comes from a model trained on rows that include
    fold ``v``'s validation rows, so fold ``v``'s rule reads their outcomes.
    """

    margins: ClassVar[list[np.ndarray]] = []

    def _regimes(self, data: Any, estimates: Any, *, seed: int | None = None) -> RegimeSet:
        assert self.learned_rule is not None
        covariates = np.asarray(data.covariates, dtype=float)
        margin = np.zeros(data.n)
        for train, test in estimates.folds:
            curve = law.outer_ep_curve(self, data, estimates, np.asarray(train))
            margin[test] = curve(covariates[test])
        OuterNuisanceCurve.margins.append(margin)
        treat = margin > 0.0
        values = np.zeros((data.n, 2, 1), dtype=float)
        values[:, 1, 0] = treat
        values[:, 0, 0] = ~treat
        return RegimeSet((self.learned_rule.name,), values, 0.0)


def _locality_fits(kind: type[TMLE]) -> tuple[Any, Any, np.ndarray]:
    """A fit, and the fit with every outcome of fold 0's validation rows flipped."""
    frame = law.draw("misspecified_limit", 800, 3)
    settings = {
        "learned_rule": ep_rule(sieve=Sieve.default(MODIFIERS, max_size=6, size=6)),
        "n_folds": 5,
        "cv_evaluation": True,
        "outcome_learner": law.outcome_learner(),
        "treatment_learner": law.treatment_learner(),
        "random_state": 11,
    }
    covariates = list(law.COVARIATES)
    first = kind(**settings).fit(frame, outcome="Y", treatment="A", covariates=covariates)
    first = first.single()
    _, test = next(iter(first.nuisance.folds))
    test = np.asarray(test)
    flipped = frame.copy()
    flipped.loc[test, "Y"] = 1.0 - flipped.loc[test, "Y"]
    second = kind(**settings).fit(flipped, outcome="Y", treatment="A", covariates=covariates)
    return first, second.single(), test


class TestWc2TheRuleIsFoldLocal:
    def test_a_validation_outcome_never_moves_its_folds_curve(self) -> None:
        first, second, test = _locality_fits(TMLE)
        np.testing.assert_array_equal(
            first.nuisance.folds.assignment, second.nuisance.folds.assignment
        )
        margins = [np.asarray(r.nuisance.regimes.margin, dtype=float) for r in (first, second)]
        np.testing.assert_array_equal(margins[0][test], margins[1][test])
        # The flipped rows are training rows of every other fold, so those curves move.
        assert np.max(np.abs(margins[0] - margins[1])) > 1e-3

    def test_the_outer_nuisance_mutation_moves_the_curve(self) -> None:
        # The debiasing corrects most of the outcome regression's move, so the leak is second
        # order and moves the curve without flipping a rule on this draw.
        OuterNuisanceCurve.margins = []
        _, _, test = _locality_fits(OuterNuisanceCurve)
        first, second = OuterNuisanceCurve.margins
        assert np.max(np.abs(first[test] - second[test])) > 1e-4


# ---------------------------------------------------------------- W-c3, inner fold locality


def _size_step_predictions(result: Any, monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Refit outer fold 0's rule and record every curve the size step scores, in order."""
    recorded: list[np.ndarray] = []
    original = ep_module.one_step_risk

    def recording(prediction: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.append(np.asarray(prediction, dtype=float).copy())
        return original(prediction, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ep_module, "one_step_risk", recording)
        _refit(result, 0, ep_rule(), loss="ep")
    return recorded


def _inner_fold_flip() -> tuple[Any, Any]:
    """A fit, and the fit with the outcomes of inner fold 0 of outer fold 0's ``T_v`` flipped."""
    frame = law.draw("misspecified_limit", 600, 7)
    first = support.fit(frame, learned_rule=ep_rule())
    (split,) = first.extra["learned_rule"].splits
    train, _ = next(iter(first.nuisance.folds))
    train = np.asarray(train)
    inner = random_partition(train.size, INNER_FOLDS, seed=split.inner_seeds[0])
    _, held = next(iter(inner))
    flipped = frame.copy()
    rows = train[np.asarray(held)]
    flipped.loc[rows, "Y"] = 1.0 - flipped.loc[rows, "Y"]
    second = support.fit(flipped, learned_rule=ep_rule())
    np.testing.assert_array_equal(first.nuisance.folds.assignment, second.nuisance.folds.assignment)
    return first, second


class TestWc3TheSizeStepIsFoldLocal:
    SIZES = 6

    def test_an_inner_folds_outcomes_never_move_its_size_step_curves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first, second = _inner_fold_flip()
        before = _size_step_predictions(first, monkeypatch)
        after = _size_step_predictions(second, monkeypatch)
        assert len(before) == len(after) == INNER_FOLDS * self.SIZES
        for index in range(self.SIZES):
            np.testing.assert_array_equal(after[index], before[index])
        moved = [np.max(np.abs(after[i] - before[i])) for i in range(self.SIZES, len(before))]
        assert min(moved) > 1e-6

    def test_a_size_step_that_sees_the_fold_moves_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def leaking(inner: Any, train: Any, fit_inner: Any, seed: int, fold: Any) -> Any:
            split = random_partition(inner.n, INNER_FOLDS, cluster=inner.cluster, seed=seed)
            return inner, fit_inner(inner, split, seed)

        monkeypatch.setattr(rule_learners, "_size_step_nuisance", leaking)
        first, second = _inner_fold_flip()
        before = _size_step_predictions(first, monkeypatch)
        after = _size_step_predictions(second, monkeypatch)
        assert max(np.max(np.abs(after[i] - before[i])) for i in range(self.SIZES)) > 1e-6


# ------------------------------------------------------------- the size step's arithmetic

#: A continuous outcome on ``[LOW, HIGH]``: ``b - a = 6``, so a risk read off the unscaled
#: curve differs from the declared one.
LOW, HIGH = 1.0, 7.0


@cache
def _continuous_weighted_fit() -> Any:
    """A weighted fit of a continuous outcome with ``q_bounds=(1, 7)`` and the CV sieve."""
    frame = support.law_frame(400, 0)
    rng = np.random.default_rng(5)
    frame = frame.assign(
        Y=2.0 + 2.0 * frame["Y"] + rng.uniform(0.0, 2.0, len(frame)),
        w=1.0 + 1.5 * (frame["W1"] > 0.2),
    )
    return (
        support.learned_tmle(
            learned_rule=ep_rule(), outcome_learner=LinearRegression(), q_bounds=(LOW, HIGH)
        )
        .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
        .single()
    )


def _rebuilt_risks(result: Any, fold: int) -> dict[str, np.ndarray]:
    """The size step of outer fold ``fold``, rebuilt from the EP core outside the rule learner.

    For each inner fold ``j`` of ``T_v``: the rows ``T_v`` minus ``j``, their own 5-fold split
    at ``refit_seed(seed, j)`` and nuisances, the curve of each size through
    ``nuisance_pieces`` and ``curve_from_pieces``, and its weighted one-step loss
    ``theta^2 - 2 theta D`` on fold ``j`` on the scaled outcome, with ``D`` the pseudo-outcome
    at ``T_v``'s inner nuisances.  Returns the weighted, unweighted and sign-flipped risks.
    """
    from cleverly.effect_modification import modifier_frame

    (split,) = result.extra["learned_rule"].splits
    seed = split.inner_seeds[fold]
    train, _ = list(result.nuisance.folds)[fold]
    inner = result.data.subset(np.asarray(train))
    fit_inner = partial(result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler)
    folds = random_partition(inner.n, INNER_FOLDS, cluster=inner.cluster, seed=seed)
    nuisance = fit_inner(inner, folds, seed)
    bounds = result.config.g_bounds
    target = rule_learners.pseudo_outcome(
        nuisance,
        result.nuisance.scaler.scale(np.asarray(inner.outcome, dtype=float)),
        np.asarray(inner.treatment, dtype=float),
        bounds,
    )
    family = Sieve.default(MODIFIERS).basis
    weights = np.asarray(inner.weights, dtype=float)
    frame = modifier_frame(inner, MODIFIERS.columns)
    sizes = range(1, int(family.max_size) + 1)
    sums = {key: np.zeros(len(sizes)) for key in ("weighted", "unweighted", "flipped")}
    mass = {"weighted": 0.0, "unweighted": 0.0}
    for j, (rows_j, held_j) in enumerate(folds):
        rows_j, held_j = np.asarray(rows_j), np.asarray(held_j)
        refit = ep_module.refit_seed(seed, j)
        rows = inner.subset(rows_j)
        fitted = fit_inner(
            rows, random_partition(rows.n, INNER_FOLDS, cluster=rows.cluster, seed=refit), refit
        )
        values = modifier_frame(rows, MODIFIERS.columns)
        pieces = ep_module.nuisance_pieces(
            rows, fitted, pair=(1.0, 0.0), g_bounds=bounds, alpha=result.estimator.alpha
        )
        held = {name: np.asarray(column)[held_j] for name, column in frame.items()}
        d, w = target[held_j], weights[held_j]
        for position, size in enumerate(sizes):
            basis = ep_module.sieve_member(family, size, MODIFIERS)
            curve = ep_module.curve_from_pieces(
                pieces,
                np.asarray(basis.evaluate(values, MODIFIERS), dtype=float),
                basis=basis,
                modifiers=MODIFIERS,
                scale="difference",
                width=HIGH - LOW,
                clip=True,
                final_learner=None,
                values=values,
            )[0]
            theta = np.asarray(curve.predict(held), dtype=float) / (HIGH - LOW)
            sums["weighted"][position] += float(np.sum(w * (theta**2 - 2.0 * theta * d)))
            sums["unweighted"][position] += float(np.sum(theta**2 - 2.0 * theta * d))
            sums["flipped"][position] += float(np.sum(w * (theta**2 + 2.0 * theta * d)))
        mass["weighted"] += float(np.sum(w))
        mass["unweighted"] += float(held_j.size)
    return {
        "weighted": sums["weighted"] / mass["weighted"],
        "unweighted": sums["unweighted"] / mass["unweighted"],
        "flipped": sums["flipped"] / mass["weighted"],
    }


class TestTheSizeStepArithmetic:
    """Plan 3.7: the CV risks, recomputed outside the rule learner, and the chosen refit."""

    def test_the_fit_is_weighted_with_a_wide_continuous_outcome(self) -> None:
        result = _continuous_weighted_fit()
        assert result.data.family != "binomial" and result.data.is_weighted
        assert np.ptp(np.asarray(result.data.weights)) > 0.5

    @pytest.mark.parametrize("fold", [0, 3])
    def test_the_cv_risks_are_the_weighted_held_out_one_step_risks(self, fold: int) -> None:
        result = _continuous_weighted_fit()
        record = _refit(result, fold, ep_rule()).ep
        assert record is not None and record.mode == "cv"
        rebuilt = _rebuilt_risks(result, fold)
        np.testing.assert_allclose(record.cv_risks, rebuilt["weighted"], rtol=0, atol=1e-12)
        # nonzero witnesses: the weights and the sign of D each move the risks
        assert np.max(np.abs(rebuilt["weighted"] - rebuilt["unweighted"])) > 1e-5
        assert np.max(np.abs(rebuilt["weighted"] - rebuilt["flipped"])) > 1e-3

    @pytest.mark.parametrize("fold", [0, 3])
    def test_the_chosen_size_is_refitted_on_every_training_row(self, fold: int) -> None:
        """The CV fold curve equals, bit for bit, the declared-size refit at the chosen size."""
        result = _continuous_weighted_fit()
        cv = _refit(result, fold, ep_rule())
        assert cv.ep is not None
        sieve = Sieve(Sieve.default(MODIFIERS).basis, size=cv.ep.chosen)
        fixed = _refit(result, fold, ep_rule(sieve=sieve))
        covariates = np.asarray(result.data.covariates, dtype=float)
        np.testing.assert_array_equal(cv.margin(covariates), fixed.margin(covariates))
        assert fixed.ep == EPRuleRecord("fixed", (cv.ep.chosen,), (), cv.ep.chosen)


def test_a_sparse_categorical_level_names_its_fold_and_the_remedy() -> None:
    """A level held by 4 of 600 rows makes a fold's debiasing singular; the error says where."""
    frame = support.law_frame(600, 1)
    level = np.zeros(len(frame))
    level[:4], level[4:300] = 2.0, 1.0
    frame = frame.assign(W3=level)
    modifiers = Modifiers(["W1", "W3"], domains={"W1": (-1.0, 1.0)}, levels={"W3": (0.0, 1.0, 2.0)})
    rule = LearnedRule(
        loss="ep", modifiers=modifiers, sieve=Sieve.default(modifiers, max_size=3, size=2)
    )
    with pytest.raises(DataError) as caught:
        support.learned_tmle(learned_rule=rule).fit(
            frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"]
        )
    message = str(caught.value)
    assert "outer fold" in message and "W1, W3" in message
    assert "Merge the sparse level" in message


# ------------------------------------------------------------------------ compositions


def _weighted_frame() -> pd.DataFrame:
    return support.law_frame(400, 0).assign(w=lambda f: 1.0 + 1.5 * (f["W1"] > 0))


class TestTheCompositions:
    def test_every_inner_and_inner_inner_fit_is_weighted(self) -> None:
        """The 25 inner fits and the 125 size-step fits of each nuisance carry the weights."""
        WeightSpyOutcome.calls, WeightSpyTreatment.calls = [], []
        frame = _weighted_frame()
        fitted = (
            support.learned_tmle(
                learned_rule=ep_rule(),
                outcome_learner=WeightSpyOutcome(),
                treatment_learner=WeightSpyTreatment(),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        outer = max(np.asarray(train).size for train, _ in fitted.nuisance.folds)
        w1 = frame["W1"].to_numpy()
        for spy in (WeightSpyOutcome, WeightSpyTreatment):
            inner = [call for call in spy.calls if call["x"].shape[0] < outer]
            assert len(inner) == 5 * INNER_FOLDS * (1 + INNER_FOLDS)
            for call in inner:
                (column,) = [
                    j for j in range(call["x"].shape[1]) if np.isin(call["x"][:, j], w1).all()
                ]
                expected = 1.0 + 1.5 * (call["x"][:, column] > 0.0)
                assert call["weight"] is not None
                weight = np.asarray(call["weight"], dtype=float)
                np.testing.assert_allclose(
                    weight / weight.mean(), expected / expected.mean(), rtol=1e-12
                )
        assert fitted.estimates[support.NAME].support.label == "experimental"

    def test_the_declared_final_learner_is_fitted_with_the_weights(self) -> None:
        SpyFinal.calls = []
        frame = _weighted_frame()
        fitted = (
            support.learned_tmle(learned_rule=ep_rule(sieve=FIXED, learner=SpyFinal()))
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        assert len(SpyFinal.calls) == 5
        weights = np.asarray(fitted.data.weights, dtype=float)
        for call, (train, _) in zip(SpyFinal.calls, fitted.nuisance.folds, strict=True):
            expected = weights[np.asarray(train)]
            # The learner reads the modifier columns, not the basis.
            np.testing.assert_array_equal(
                call["x"][:, 0], np.asarray(fitted.data.covariates)[np.asarray(train), 0]
            )
            np.testing.assert_allclose(
                call["weight"] / call["weight"].mean(), expected / expected.mean(), rtol=1e-12
            )
        refit = _refit(fitted, 0, ep_rule(sieve=FIXED, learner=SpyFinal()))
        assert isinstance(refit.model.learner, SpyFinal)

    def test_least_squares_is_the_default_final_stage(self) -> None:
        refit = _refit(_fixed_fit(), 0, ep_rule(sieve=FIXED))
        assert refit.model.learner is None
        assert refit.model.beta is not None

    def test_clusters_stay_whole_at_both_inner_levels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn: list[tuple[Any, Any]] = []
        original = crossfit.random_partition

        def recording(n: int, n_folds: int, *, cluster: Any = None, seed: int) -> Any:
            folds = original(n, n_folds, cluster=cluster, seed=seed)
            drawn.append((cluster, folds))
            return folds

        monkeypatch.setattr(crossfit, "random_partition", recording)
        frame = support.law_frame(400, 0).assign(c=lambda f: np.arange(len(f)) // 4)
        support.learned_tmle(learned_rule=ep_rule()).fit(
            frame, outcome="Y", treatment="A", covariates=support.COVARIATES, id="c"
        )
        inner = [(cluster, folds) for cluster, folds in drawn if folds.n < len(frame)]
        assert len(inner) == 5 * (1 + INNER_FOLDS)
        for cluster, folds in inner:
            assert cluster is not None
            for code in np.unique(cluster):
                assert np.unique(folds.assignment[cluster == code]).size == 1

    @pytest.mark.parametrize(
        ("sieve", "family"),
        [
            (None, "adaptive"),
            (Sieve.default(MODIFIERS, size="cv"), "adaptive"),
            (FIXED, "parametric"),
        ],
    )
    def test_the_in_sample_key_reads_the_sieve_mode(self, sieve: Any, family: str) -> None:
        """A size chosen by cross-validation selects, so it keys adaptive; a fixed size does not."""
        from types import SimpleNamespace

        from cleverly._support import in_sample_learned_rule_learners

        estimator = support.learned_tmle(
            learned_rule=ep_rule(sieve=sieve), cross_fit=False, cv_evaluation=False
        )
        options = in_sample_learned_rule_learners(estimator, SimpleNamespace(known_treatment=None))
        assert options["rule_learner_family"] == family

    def test_an_in_sample_fit_learns_one_curve_on_every_row(self) -> None:
        # The support module's outcome learner is a Pipeline, which reads as adaptive in sample.
        with pytest.warns(InSampleLearnerWarning, match="adaptive outcome learner"):
            result = support.fit(
                learned_rule=ep_rule(sieve=FIXED), cross_fit=False, cv_evaluation=False
            )
        record = result.extra["learned_rule"]
        assert "fitted on every row, is positive (loss='ep')" in record.describe()
        data = result.data
        refit = fit_fold_rule(
            data,
            np.arange(data.n),
            loss="ep",
            learner=None,
            fit_inner=partial(
                result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler
            ),
            scaler=result.nuisance.scaler,
            g_bounds=result.config.g_bounds,
            seed=record.splits[0].inner_seeds[0],
            modifiers=MODIFIERS,
            sieve=FIXED,
            alpha=result.estimator.alpha,
        )
        covariates = np.asarray(data.covariates, dtype=float)
        np.testing.assert_array_equal(refit.treat(covariates), law.rule_of(result))


# ---------------------------------------------------------------------------- the harness


class TestTheStudyHarnessReproducesTheEPRule:
    def test_the_refit_reproduces_the_fits_rule(self) -> None:
        frame = law.draw("non_exceptional", 600, 5)
        _, truth, _ = law.measured(
            frame, law.outcome_learner(), "non_exceptional", 3, rule=ep_rule()
        )
        assert truth.rows_checked == 600
        assert 0.5 < truth.truth < 0.7

    def test_the_validation_row_control_learns_another_rule(self) -> None:
        frame = law.draw("non_exceptional", 600, 5)
        local = law.fit(frame, law.outcome_learner(), 3, rule=ep_rule())
        control = law.fit(frame, law.outcome_learner(), 3, mutation=True, rule=ep_rule())
        assert np.any(law.rule_of(local) != law.rule_of(control))
        truth = law.harness(control, law.outcome_learner(), "non_exceptional", source="validation")
        assert truth.rows_checked == 600
        with pytest.raises(law.HarnessMismatch):
            law.harness(control, law.outcome_learner(), "non_exceptional")


# ------------------------------------------------------------------ persistence and studies


@pytest.mark.parametrize("route", ["save", "dumps"])
def test_a_round_trip_keeps_the_rule_and_the_record(route: str, tmp_path: Any) -> None:
    from cleverly import load
    from cleverly.estimators.serialize import dumps, loads

    result = _fixed_fit()
    restored = (
        load(result.save(tmp_path / "ep.joblib")) if route == "save" else loads(dumps(result))
    )
    assert restored.extra["learned_rule"] == result.extra["learned_rule"]
    np.testing.assert_array_equal(restored.nuisance.regimes.values, result.nuisance.regimes.values)
    assert restored.estimator.learned_rule.modifiers == MODIFIERS


def test_a_study_fit_reports_the_engine_estimate() -> None:
    from cleverly import CausalStudy, LearnedRuleValue, OptimalRuleValue, PointTreatment

    study = CausalStudy(
        support.law_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    options = {
        "outcome_learner": support.outcome_learner(),
        "treatment_learner": support.treatment_learner(),
        "n_folds": 5,
        "cv_evaluation": True,
        "simultaneous": False,
        "random_state": 0,
    }
    learned = study.identify(LearnedRuleValue(rule=ep_rule(sieve=FIXED))).estimate(**options)
    assert learned.estimates[support.NAME].psi == pytest.approx(
        _fixed_fit().estimates[support.NAME].psi, abs=1e-12
    )
    optimal = study.identify(OptimalRuleValue(rule=ep_rule(sieve=FIXED))).estimate(**options)
    (name,) = optimal.estimates
    assert optimal.estimates[name].psi == learned.estimates[support.NAME].psi
    with pytest.raises(CapabilityError, match="leave out the adjustment columns"):
        study.identify(OptimalRuleValue(rule=LearnedRule(loss="ep", modifiers=W1_ONLY))).estimate(
            **options
        )


# ------------------------------------------- EP1, EP2, EP4: a declared sieve or final learner
#
# P1 PR 1 (plan 4.5).  EP1 reruns W-c1 with a declared fixed-size B-spline sieve on a finite
# population whose modifier takes nine levels, so the member does not saturate the levels and
# the fold curve is the projection of the effect onto the member, not the effect itself.  EP2
# reruns W-c2 with a declared final learner and records the rows each final-stage fit sees.
# EP4 reads the documented per-split diagnostic.

#: The EP1 population: nine levels of ``V`` on ``[0, 2]``, binary ``W2``.
EP1_LEVELS = np.linspace(0.0, 2.0, 9)
EP1_MODIFIERS = Modifiers(["V"], domains={"V": (0.0, 2.0)})
EP1_SIEVE = Sieve(BSpline(["V"], 4), size=1)


def ep1_true_mean(a: Any, v: Any, w2: Any) -> Any:
    blip = np.sin(4.0 * v) - 0.2 + 0.6 * w2 * (v - 1.0)
    return expit(-0.2 + 0.3 * v - 0.4 * w2 + a * blip)


def ep1_g(v: float, w2: int) -> float:
    return float(expit(-0.3 + 0.4 * v - 0.5 * w2))


def ep1_p_v(v: float) -> float:
    """``P(V = v)``, uneven so that an unweighted final stage misses the projection."""
    return float((0.2 + v**2) / np.sum(0.2 + EP1_LEVELS**2))


def ep1_p_w2(v: float) -> float:
    return float(expit(-0.5 + 0.6 * v))


class EP1WrongRegression(ClassifierMixin, BaseEstimator):
    """A fitted stub, wrong in the blip, on the design ``(A, V, W2)``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> EP1WrongRegression:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        x = np.asarray(X, dtype=float)
        a, v, w2 = x[:, 0], x[:, 1], x[:, 2]
        p = expit(-0.2 + 0.3 * v - 0.4 * w2 + a * (0.5 - 0.3 * v) + 0.2 * w2)
        return np.column_stack([1.0 - p, p])


@cache
def ep1_population() -> pd.DataFrame:
    rows = []
    for copy in range(COPIES):
        for v in EP1_LEVELS:
            for w2 in (0, 1):
                cell = (ep1_p_w2(v) if w2 else 1.0 - ep1_p_w2(v)) * ep1_p_v(v)
                g = ep1_g(v, w2)
                for a in (0, 1):
                    mean = float(ep1_true_mean(a, v, w2))
                    for y in (0, 1):
                        mass = cell * (g if a else 1.0 - g) * (mean if y else 1.0 - mean)
                        rows.append(
                            {"V": v, "W2": w2, "A": a, "Y": y, "w": mass, "c": copy}
                            | {"p1": g, "p0": 1.0 - g}
                        )
    return pd.DataFrame(rows).astype({"W2": float, "A": float, "Y": float})


def ep1_effect() -> np.ndarray:
    """``theta(v)``, the effect given ``V``, at each level."""
    p = np.array([ep1_p_w2(v) for v in EP1_LEVELS])
    effect = {
        w2: ep1_true_mean(1.0, EP1_LEVELS, w2) - ep1_true_mean(0.0, EP1_LEVELS, w2) for w2 in (0, 1)
    }
    return (1.0 - p) * effect[0] + p * effect[1]


def ep1_projection() -> np.ndarray:
    """The projection of ``theta`` onto the declared member under ``P(V)``, at each level."""
    from cleverly.estimators.ep import sieve_member

    member = sieve_member(EP1_SIEVE.basis, EP1_SIEVE.size, EP1_MODIFIERS)
    design = np.asarray(member.evaluate({"V": EP1_LEVELS}, EP1_MODIFIERS), dtype=float)
    root = np.sqrt([ep1_p_v(v) for v in EP1_LEVELS])
    beta = np.linalg.lstsq(design * root[:, None], ep1_effect() * root, rcond=None)[0]
    return design @ beta


def ep1_fit() -> Any:
    estimator = TMLE(
        learned_rule=LearnedRule(loss="ep", modifiers=EP1_MODIFIERS, sieve=EP1_SIEVE),
        n_folds=10,
        cv_evaluation=True,
        outcome_learner=EP1WrongRegression(),
        random_state=3,
    )
    return estimator.fit(
        ep1_population(),
        outcome="Y",
        treatment="A",
        covariates=["V", "W2"],
        weights="w",
        id="c",
        treatment_probabilities={1.0: "p1", 0.0: "p0"},
    ).single()


def ep1_gap(result: Any) -> float:
    """The largest distance of a fold curve from the projection, over the levels."""
    margin = np.asarray(result.nuisance.regimes.margin, dtype=float)
    v = ep1_population()["V"].to_numpy()
    target = ep1_projection()
    return max(
        float(np.max(np.abs(margin[np.isclose(v, level)] - target[i])))
        for i, level in enumerate(EP1_LEVELS)
    )


class TestEP1TheDeclaredSieveReachesTheCurve:
    def test_the_member_does_not_saturate_the_levels(self) -> None:
        # Nonzero witness: the projection differs from the effect, so a curve that ignored the
        # declared sieve (and fitted the effect at each level) fails the identity below.
        assert np.max(np.abs(ep1_projection() - ep1_effect())) > 1e-2

    def test_every_fold_curve_is_the_projection_onto_the_declared_member(self) -> None:
        assert ep1_gap(ep1_fit()) < 1e-10

    def test_skipping_the_debiasing_breaks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = ep_module.debias

        def no_debiasing(**inputs: Any) -> Any:
            fitted = original(**inputs)
            return replace(
                fitted,
                coefficients=0.0 * fitted.coefficients,
                targeted_t=np.asarray(inputs["initial_t"], dtype=float),
                targeted_r=np.asarray(inputs["initial_r"], dtype=float),
            )

        monkeypatch.setattr(ep_module, "debias", no_debiasing)
        assert ep1_gap(ep1_fit()) > 1e-3

    @pytest.mark.parametrize("mutation", ["unit_inverse_g", "no_row_weights"])
    def test_a_debiasing_weight_mutation_breaks_it(
        self, mutation: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ep_module.nuisance_pieces

        def mutated(*args: Any, **kwargs: Any) -> Any:
            pieces = original(*args, **kwargs)
            if mutation == "unit_inverse_g":
                return replace(pieces, weight=pieces.rows)
            return replace(pieces, weight=pieces.weight / pieces.rows, rows=0 * pieces.rows + 1)

        monkeypatch.setattr(ep_module, "nuisance_pieces", mutated)
        assert ep1_gap(ep1_fit()) > 1e-3

    def test_an_unweighted_final_stage_breaks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = ep_module.final_stage

        def unweighted(debiased: Any, design: Any, weights: Any, *args: Any, **kw: Any) -> Any:
            return original(debiased, design, np.ones_like(weights), *args, **kw)

        monkeypatch.setattr(ep_module, "final_stage", unweighted)
        assert ep1_gap(ep1_fit()) > 1e-3

    def test_dropping_the_arm_sign_breaks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unsigned(design: Any, treatment: Any, pair: Any, scale: str) -> Any:
            return np.asarray(design, dtype=float)

        monkeypatch.setattr(ep_module, "debias_features", unsigned)
        assert ep1_gap(ep1_fit()) > 1e-3


class RowSpyFinal(RegressorMixin, BaseEstimator):
    """A linear final learner that records the modifier rows of every fit."""

    calls: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RowSpyFinal:
        RowSpyFinal.calls.append(np.asarray(X, dtype=float).copy())
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)


def _spied_fit(
    monkeypatch: pytest.MonkeyPatch, *, all_rows: bool
) -> tuple[list[tuple[int, set[int]]], list[Any]]:
    """Fit with the spy learner; return, per final-stage fit, its outer fold and its rows.

    ``all_rows=True`` is the mutation: each fold's rule is fitted on every row.
    """
    frame = law.draw("misspecified_limit", 400, 3)
    original = rule_learners.fit_fold_rule
    current: list[int] = []

    def tracking(data: Any, rows: Any, **settings: Any) -> Any:
        current.append(int(settings["fold"]))
        return original(data, np.arange(data.n) if all_rows else rows, **settings)

    monkeypatch.setattr(rule_learners, "fit_fold_rule", tracking)
    RowSpyFinal.calls = []
    owners: list[int] = []
    spy = RowSpyFinal()
    real_fit = RowSpyFinal.fit

    def owned(self: RowSpyFinal, X: Any, y: Any, sample_weight: Any = None) -> RowSpyFinal:
        owners.append(current[-1])
        return real_fit(self, X, y, sample_weight)

    monkeypatch.setattr(RowSpyFinal, "fit", owned)
    result = TMLE(
        learned_rule=ep_rule(sieve=FIXED, learner=spy),
        n_folds=5,
        cv_evaluation=True,
        outcome_learner=law.outcome_learner(),
        treatment_learner=law.treatment_learner(),
        random_state=11,
    ).fit(frame, outcome="Y", treatment="A", covariates=list(law.COVARIATES))
    w1 = frame["W1"].to_numpy()
    index = {value: row for row, value in enumerate(w1)}
    assert len(index) == len(w1)
    calls = [{index[value] for value in x[:, 0]} for x in RowSpyFinal.calls]
    folds = list(result.single().nuisance.folds)
    return list(zip(owners, calls, strict=True)), folds


class TestEP2TheDeclaredLearnerIsFoldLocal:
    def test_no_final_stage_fit_sees_a_validation_row_of_its_fold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls, folds = _spied_fit(monkeypatch, all_rows=False)
        assert {owner for owner, _ in calls} == set(range(5))
        for owner, rows in calls:
            _, test = folds[owner]
            assert not rows & set(np.asarray(test).tolist())

    def test_the_mutation_that_fits_on_every_row_is_seen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls, folds = _spied_fit(monkeypatch, all_rows=True)
        leaked = [
            owner for owner, rows in calls if rows & set(np.asarray(folds[owner][1]).tolist())
        ]
        assert leaked


class TestEP4RuleDisagreement:
    @pytest.mark.parametrize(
        "changes", [{}, {"sieve": FIXED}, {"learner": LinearRegression()}], ids=str
    )
    def test_an_ep_rule_fit_reports_rule_disagreement_per_split(
        self, changes: dict[str, Any]
    ) -> None:
        result = support.fit(learned_rule=ep_rule(**({"sieve": FIXED} | changes)))
        splits = result.extra["learned_rule"].splits
        assert len(splits) == 1
        for split in splits:
            assert 0.0 <= split.rule_disagreement <= 1.0
