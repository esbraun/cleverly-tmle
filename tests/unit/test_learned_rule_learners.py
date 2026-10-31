"""X11 (g): the blip and weighted-classification rule learners of the learned-rule value.

``LearnedRule(loss="blip" | "weighted_classification", learner=...)`` learns each fold's rule
from the doubly robust blip pseudo-outcome
``D = (2A - 1) / g(A | W) (Y - Qbar(A, W)) + Qbar(1, W) - Qbar(0, W)``, with ``Qbar`` and ``g``
from a 5-fold cross-fit inside the fold's training rows.  The outer fluctuation, estimate and
variance do not change.

* W-g1 (nonzero): at ``misspecified_limit`` with the main-terms outcome learner, the plug-in
  rule and the blip rule with a ``W1**2`` basis differ on :data:`DECLARED_SHARE` of the
  population, declared from the population projections, and the blip rule's value is higher.
* W-g2 (exact identity): with a discrete covariate and saturated learners, the blip rule is the
  sign of each cell's mean pseudo-outcome and the weighted-classification rule the sign of each
  cell's sum, so the two agree on every row.  An unweighted classifier, which learns the
  majority sign, differs on the declared cell :data:`MAJORITY_CELL`.
* W-g3 (fold locality): fold ``v``'s rule does not move when fold ``v``'s validation outcomes
  do.  The mutation that builds the pseudo-outcome from the outer cross-fitted predictions
  moves it.
* W-g4 (exact): changing one training outcome by ``dY`` changes that row's pseudo-outcome by
  exactly ``(2A - 1) dY / g``, because its own inner nuisances did not train on it.  Two
  nonzero variants pin the components that vanish on the plain draw: at a row where a declared
  ``g_bounds`` binds the change is ``(2A - 1) dY / max(g, l)``, and on a continuous outcome in
  ``[a, b]`` it is ``(2A - 1) dY / {(b - a) g}``.
* The inner fits: under ``weights=`` every inner outcome and treatment fit receives the
  observation weights of its rows.  A training fold whose positive-weight rows carry one label
  gives the constant rule of that label, and a score exactly at its threshold assigns control.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache, partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import SuperLearner
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import LearnedRule
from cleverly.interventions._rule_learners import (
    INNER_FOLDS,
    FoldRule,
    fit_fold_rule,
    inner_seed,
    pseudo_outcome,
)
from cleverly.interventions.base import RegimeSet
from cleverly.interventions.learned import RULE_CLASS, freeze_learned_rule
from cleverly.learners import crossfit
from tests.studies import _learned_rule_law as law
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

# --------------------------------------------------------------------------- the learners


def blip_learner() -> Pipeline:
    """The blip learner of W-g1: a linear regression on every term of degree 2 in ``W``."""
    return Pipeline([("basis", PolynomialFeatures(2)), ("fit", LinearRegression())])


def classifier() -> Pipeline:
    """The weighted classifier on the same basis."""
    return Pipeline(
        [("basis", PolynomialFeatures(2)), ("fit", LogisticRegression(C=1e6, max_iter=5000))]
    )


def blip_rule() -> LearnedRule:
    return LearnedRule(loss="blip", learner=blip_learner())


class NoWeightRegressor(RegressorMixin, BaseEstimator):
    """A regressor whose ``fit`` takes no ``sample_weight``."""

    def fit(self, X: Any, y: Any) -> NoWeightRegressor:
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)


class NoWeightClassifier(ClassifierMixin, BaseEstimator):
    """A classifier whose ``fit`` takes no ``sample_weight``."""

    def fit(self, X: Any, y: Any) -> NoWeightClassifier:  # pragma: no cover - refused first
        self.model_ = LogisticRegression().fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - refused first
        return self.model_.predict_proba(X)


class SpyRegressor(RegressorMixin, BaseEstimator):
    """A linear regression that records the weights and the cluster codes of every fit."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None, groups: Any = None) -> SpyRegressor:
        SpyRegressor.calls.append(
            {"y": np.asarray(y, dtype=float), "weight": sample_weight, "groups": groups}
        )
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """A logistic regression that records the labels and weights of every fit."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SpyClassifier:
        SpyClassifier.calls.append({"y": np.asarray(y, dtype=float), "weight": sample_weight})
        self.model_ = LogisticRegression(max_iter=5000).fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)


class WeightSpyOutcome(ClassifierMixin, BaseEstimator):
    """A logistic regression that records the design and the weights of every outcome fit."""

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


class UnfittableClassifier(ClassifierMixin, BaseEstimator):
    """A classifier that takes ``sample_weight`` and fails if it is ever fitted."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> UnfittableClassifier:
        raise AssertionError("a one-class fold must not fit the classifier")

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - never fitted
        raise AssertionError("a one-class fold must not read the classifier")


class FixedScore(BaseEstimator):
    """A fitted stub whose score is the first covariate: the prediction or ``P(1)``."""

    classes_ = np.array([0.0, 1.0])

    def predict(self, X: Any) -> Any:
        return np.asarray(X, dtype=float)[:, 0]

    def predict_proba(self, X: Any) -> Any:
        p = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------------- the declaration


class TestTheDeclaration:
    """``LearnedRule`` refuses a learner that does not suit its loss, before any data."""

    def test_the_default_is_the_plug_in_rule(self) -> None:
        rule = LearnedRule()
        assert (rule.loss, rule.learner) == ("plug_in", None)

    def test_an_unknown_loss_is_a_data_error(self) -> None:
        with pytest.raises(DataError, match="loss= must be one of"):
            LearnedRule(loss="hinge")

    @pytest.mark.parametrize(
        ("keywords", "match"),
        [
            ({"learner": LinearRegression()}, "takes no learner="),
            ({"loss": "blip"}, "needs learner="),
            ({"loss": "weighted_classification"}, "needs learner="),
            ({"loss": "blip", "learner": LogisticRegression()}, "is a classifier"),
            ({"loss": "blip", "learner": SuperLearner(task="classification")}, "classifier"),
            ({"loss": "blip", "learner": SuperLearner(clip=(0.0, 1.0))}, "clips its"),
            ({"loss": "weighted_classification", "learner": LinearRegression()}, "regressor"),
            (
                {"loss": "weighted_classification", "learner": SuperLearner(task="regression")},
                "regressor",
            ),
            (
                {"loss": "weighted_classification", "learner": NoWeightClassifier()},
                "takes no sample_weight",
            ),
            (
                {
                    "loss": "weighted_classification",
                    "learner": SuperLearner(
                        library=[LogisticRegression(), KNeighborsClassifier()],
                        task="classification",
                    ),
                },
                "SuperLearner member KNeighborsClassifier, whose fit takes no sample_weight",
            ),
        ],
    )
    def test_a_learner_that_does_not_suit_the_loss_refuses(
        self, keywords: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(CapabilityError, match=match) as caught:
            LearnedRule(**keywords)
        assert caught.value.refusal.reason.value == "invalid_declaration"

    def test_the_admitted_learners(self) -> None:
        LearnedRule(loss="blip", learner=SuperLearner(library=[LinearRegression()]))
        LearnedRule(loss="blip", learner=NoWeightRegressor())
        LearnedRule(loss="weighted_classification", learner=classifier())

    def test_a_weighted_blip_fit_refuses_a_learner_without_weights_before_any_learner(
        self,
    ) -> None:
        estimator = support.learned_tmle(
            learned_rule=LearnedRule(loss="blip", learner=NoWeightRegressor()),
            **never_fit_learners(),
        )
        frame = support.law_frame(120, 3).assign(w=lambda f: 1.0 + 0.5 * f["W2"])
        NeverFit.calls = 0
        with pytest.raises(CapabilityError, match="rule of the unweighted population"):
            estimator.fit(
                frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w"
            )
        assert NeverFit.calls == 0

    def test_a_weighted_blip_fit_refuses_a_superlearner_member_without_weights(self) -> None:
        library = SuperLearner(library=[LinearRegression(), KNeighborsRegressor()])
        LearnedRule(loss="blip", learner=library)  # admitted: unweighted fits need no weights
        estimator = support.learned_tmle(
            learned_rule=LearnedRule(loss="blip", learner=library), **never_fit_learners()
        )
        frame = support.law_frame(120, 3).assign(w=lambda f: 1.0 + 0.5 * f["W2"])
        NeverFit.calls = 0
        with pytest.raises(CapabilityError, match="member KNeighborsRegressor") as caught:
            estimator.fit(
                frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w"
            )
        assert caught.value.refusal.reason.value == "invalid_declaration"
        assert NeverFit.calls == 0

    def test_the_weighted_learner_row_comes_after_the_setting_rows(self) -> None:
        estimator = support.learned_tmle(
            learned_rule=LearnedRule(loss="blip", learner=NoWeightRegressor()),
            n_bootstrap=10,
            **never_fit_learners(),
        )
        frame = support.law_frame(120, 3).assign(w=lambda f: 1.0 + 0.5 * f["W2"])
        with pytest.raises(CapabilityError, match="full-refit bootstrap"):
            estimator.fit(
                frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w"
            )

    def test_an_unweighted_blip_fit_takes_a_learner_without_weights(self) -> None:
        result = support.fit(learned_rule=LearnedRule(loss="blip", learner=NoWeightRegressor()))
        assert np.isfinite(result.estimates[support.NAME].psi)


# ------------------------------------------------------------------------------ the record


@cache
def _blip_fit() -> Any:
    return support.fit(learned_rule=blip_rule())


class TestTheRecord:
    def test_the_record_names_the_loss_and_the_learner(self) -> None:
        record = _blip_fit().extra["learned_rule"]
        assert record.loss == "blip"
        assert record.learner == "linear regression after polynomial terms"
        assert record.rule == RULE_CLASS["blip"]
        line = record.describe()
        assert "loss='blip'" in line
        assert "linear regression after polynomial terms" in line

    def test_each_loss_has_its_rule_class(self) -> None:
        assert set(RULE_CLASS) == {"plug_in", "blip", "weighted_classification"}
        assert "sample weights |D|" in RULE_CLASS["weighted_classification"]
        plug_in = support.fit().extra["learned_rule"]
        assert (plug_in.loss, plug_in.learner) == ("plug_in", None)
        assert plug_in.splits[0].inner_seeds == ()
        assert plug_in.splits[0].score_quantiles == ()
        assert "plug-in rule of the fold's outcome regression" in plug_in.describe()

    def test_the_inner_seeds_are_recorded_and_derived_from_the_split_seed(self) -> None:
        result = _blip_fit()
        (split,) = result.extra["learned_rule"].splits
        assert split.inner_seeds == tuple(
            inner_seed(result.repeats[0].seed, fold) for fold in range(5)
        )
        assert len(set(split.inner_seeds)) == 5

    def test_the_realized_rule_is_the_rule_learners_rule(self) -> None:
        """Each fold's realized rule is a refit of its rule learner, not the plug-in rule."""
        result = _blip_fit()
        rule = law.rule_of(result)
        covariates = np.asarray(result.data.covariates, dtype=float)
        (split,) = result.extra["learned_rule"].splits
        plug_in = support.fit()
        assert not np.array_equal(rule, law.rule_of(plug_in))
        for fold, (train, test) in enumerate(result.nuisance.folds):
            refit = fit_fold_rule(
                result.data,
                np.asarray(train),
                loss="blip",
                learner=blip_learner(),
                fit_inner=partial(
                    result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler
                ),
                scaler=result.nuisance.scaler,
                g_bounds=result.config.g_bounds,
                seed=split.inner_seeds[fold],
            )
            np.testing.assert_array_equal(refit.treat(covariates[test]), rule[test])
            assert np.mean(rule[test]) == split.treated_shares[fold]
            # The score quantiles describe the rule learner's score, not the outer blip.
            margin = refit.margin(covariates[test])
            np.testing.assert_allclose(
                split.score_quantiles[fold],
                np.quantile(margin, law_levels()),
                rtol=0,
                atol=1e-12,
            )
            assert split.score_quantiles[fold] != split.blip_quantiles[fold]

    def test_a_frozen_rule_keeps_the_loss_and_the_rule(self) -> None:
        result = _blip_fit()
        frozen = freeze_learned_rule(result)
        assert (frozen.loss, frozen.learner) == ("blip", result.estimator.learned_rule.learner)
        np.testing.assert_array_equal(frozen.treat, law.rule_of(result))

    @pytest.mark.parametrize(
        "rule",
        [blip_rule(), LearnedRule(loss="weighted_classification", learner=classifier())],
        ids=["blip", "weighted_classification"],
    )
    def test_an_unweighted_pooled_fit_with_a_rule_learner_reads_supported(
        self, rule: LearnedRule
    ) -> None:
        """``learned-rule-learners`` measures unweighted rows with pooled targeting, one split."""
        label = support.fit(learned_rule=rule).estimates[support.NAME].support
        assert label.label == "supported"
        assert label.row == f"T39.{rule.loss}"

    @pytest.mark.parametrize("setting", [{"targeting_scheme": "fold"}, {"repeats": 2}])
    def test_fold_targeting_and_repeats_with_a_rule_learner_stay_experimental(
        self, setting: dict[str, Any]
    ) -> None:
        """No cell measures a rule learner with either setting, so no row covers the fit."""
        label = support.fit(learned_rule=blip_rule(), **setting).estimates[support.NAME].support
        assert label.label == "experimental"
        assert label.row is None
        assert "Nearest row: T" in label.reason


def law_levels() -> tuple[float, ...]:
    from cleverly.interventions.learned import BLIP_QUANTILE_LEVELS

    return BLIP_QUANTILE_LEVELS


# ------------------------------------------------------------------------------------ W-g1

#: The population share of ``W`` on which the limit plug-in rule of ``law.outcome_learner()``
#: and the limit blip rule of :func:`blip_learner` differ at ``misspecified_limit``, from the
#: projections that :func:`_projections` computes.
DECLARED_SHARE = 0.1474

#: The population values of the two limit rules, and of the optimal rule.
DECLARED_VALUES = {"plug_in": 0.55475, "blip": 0.56019, "optimal": 0.56025}


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    """The ``W1`` grid of the study harness, and its trapezoid weights times the density 1/2."""
    grid = law.GRID
    weights = np.full(grid.size, grid[1] - grid[0])
    weights[[0, -1]] /= 2.0
    return grid, weights * 0.5


@cache
def _projections() -> dict[str, np.ndarray]:
    """The limit rules on the grid, stacked over ``W2 = 0`` then ``W2 = 1``.

    The plug-in limit is the blip of the logistic projection of ``Qbar0`` on the design of
    ``law.outcome_learner()`` under ``P0``.  The blip limit is the least-squares projection of
    ``E[D | W] = Qbar0(1, W) - Qbar0(0, W)`` on the basis of :func:`blip_learner`, which holds
    because ``g`` converges to the true mechanism.
    """
    grid, weights = _quadrature()
    scenario = "misspecified_limit"
    design, outcome, mass = [], [], []
    for w2 in law.W2_LEVELS:
        for a in (0.0, 1.0):
            treated = law.g0(grid, w2)
            design.append(np.column_stack([np.full_like(grid, a), grid, np.full_like(grid, w2)]))
            outcome.append(law.qbar0(a, grid, w2, scenario))
            mass.append(weights * 0.5 * (treated if a else 1.0 - treated))
    x, q, m = np.vstack(design), np.concatenate(outcome), np.concatenate(mass)
    projection = law.outcome_learner().set_params(model__C=1e10, model__tol=1e-14)
    projection.fit(
        np.vstack([x, x]),
        np.r_[np.ones(q.size), np.zeros(q.size)],
        model__sample_weight=np.r_[m * q, m * (1.0 - q)] * 2 * q.size,
    )
    covariates = np.vstack([np.column_stack([grid, np.full_like(grid, w2)]) for w2 in (0.0, 1.0)])
    ones = np.ones((covariates.shape[0], 1))
    plug_in = (
        projection.predict_proba(np.hstack([ones, covariates]))[:, 1]
        - projection.predict_proba(np.hstack([0.0 * ones, covariates]))[:, 1]
    )
    truth = np.concatenate(
        [
            law.qbar0(1.0, grid, w2, scenario) - law.qbar0(0.0, grid, w2, scenario)
            for w2 in law.W2_LEVELS
        ]
    )
    population = np.concatenate([weights * 0.5, weights * 0.5])
    blip = blip_learner().fit(covariates, truth, fit__sample_weight=population).predict(covariates)
    return {
        "plug_in": plug_in > 0.0,
        "blip": blip > 0.0,
        "optimal": truth > 0.0,
        "mass": population,
    }


def _population_value(rule: np.ndarray) -> float:
    grid, weights = _quadrature()
    halves = np.split(rule.astype(float), 2)
    return float(
        sum(
            np.sum(weights * 0.5 * law.qbar0(half, grid, w2, "misspecified_limit"))
            for half, w2 in zip(halves, law.W2_LEVELS, strict=True)
        )
    )


@cache
def _g1_fits() -> dict[str, tuple[Any, Any]]:
    frame = law.draw("misspecified_limit", 2000, 100)
    fits = {}
    for loss, rule in (("plug_in", None), ("blip", blip_rule())):
        result, truth, _ = law.measured(
            frame, law.outcome_learner(), "misspecified_limit", 7, rule=rule
        )
        fits[loss] = (result, truth)
    return fits


class TestWg1TheBlipLearnerMovesTheRule:
    def test_the_declared_share_and_values_are_the_projections(self) -> None:
        limits = _projections()
        share = float(np.sum(limits["mass"] * (limits["plug_in"] != limits["blip"])))
        assert share == pytest.approx(DECLARED_SHARE, abs=5e-4)
        for name, value in DECLARED_VALUES.items():
            assert _population_value(limits[name]) == pytest.approx(value, abs=5e-5)
        assert DECLARED_VALUES["blip"] > DECLARED_VALUES["plug_in"]

    def test_the_fits_differ_on_about_the_declared_share(self) -> None:
        fits = _g1_fits()
        rules = {loss: law.rule_of(result) for loss, (result, _) in fits.items()}
        share = float(np.mean(rules["plug_in"] != rules["blip"]))
        # The fold rules are estimates at n = 2,000, so the share is within 0.06 of the limit.
        assert abs(share - DECLARED_SHARE) < 0.06, share

    def test_the_blip_rules_have_the_higher_value(self) -> None:
        """The study harness integrates the value of each fit's own fold rules (rule L4)."""
        fits = _g1_fits()
        assert fits["blip"][1].truth > fits["plug_in"][1].truth + 0.002


# ------------------------------------------------------------------------------------ W-g2

#: The cells of the discrete law: ``P(A = 1 | cell)``, ``Qbar0(0, cell)``, ``Qbar0(1, cell)``.
CELLS = (
    (0.5, 0.3, 0.6),
    (0.5, 0.7, 0.3),
    (0.8, 0.05, 0.15),
    (0.5, 0.2, 0.6),
)

#: The cell where most pseudo-outcomes are negative and their sum is positive: 85% of its
#: treated rows have ``Y = 0`` and a small negative pseudo-outcome, and the rare ``Y = 1``
#: rows carry large positive ones.  The majority sign does not treat there, and the blip does.
MAJORITY_CELL = 2


def _cell_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cell = rng.integers(0, len(CELLS), n)
    g, q0, q1 = (np.array(column)[cell] for column in zip(*CELLS, strict=True))
    a = rng.binomial(1, g).astype(float)
    y = rng.binomial(1, np.where(a == 1.0, q1, q0)).astype(float)
    frame = pd.DataFrame({f"C{k}": (cell == k).astype(float) for k in range(len(CELLS))})
    return frame.assign(A=a, Y=y, cell=cell)


def _saturated_outcome() -> Pipeline:
    """A saturated outcome model: every treatment-by-cell term of ``(A, C0, ..., C3)``."""
    return Pipeline(
        [
            ("terms", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(C=1e8, max_iter=20000, tol=1e-12)),
        ]
    )


def _saturated_classifier() -> LogisticRegression:
    return LogisticRegression(C=1e8, max_iter=20000, tol=1e-12)


CELL_COVARIATES = [f"C{k}" for k in range(len(CELLS))]


@cache
def _cell_fits() -> dict[str, Any]:
    frame = _cell_frame()
    rules = {
        "blip": LearnedRule(loss="blip", learner=LinearRegression()),
        "weighted_classification": LearnedRule(
            loss="weighted_classification", learner=_saturated_classifier()
        ),
    }
    return {
        loss: support.learned_tmle(
            learned_rule=rule,
            outcome_learner=_saturated_outcome(),
            treatment_learner=_saturated_classifier(),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=CELL_COVARIATES)
        .single()
        for loss, rule in rules.items()
    }


def _fold_pseudo_outcomes(result: Any) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Each fold's training rows, test rows and pseudo-outcomes, refit from the record."""
    (split,) = result.extra["learned_rule"].splits
    rule = result.estimator.learned_rule
    out = []
    for fold, (train, test) in enumerate(result.nuisance.folds):
        fitted = fit_fold_rule(
            result.data,
            np.asarray(train),
            loss=rule.loss,
            learner=rule.learner,
            fit_inner=partial(
                result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler
            ),
            scaler=result.nuisance.scaler,
            g_bounds=result.config.g_bounds,
            seed=split.inner_seeds[fold],
        )
        out.append((np.asarray(train), np.asarray(test), fitted.pseudo_outcome))
    return out


class TestWg2TheTwoLossesAgreeOnSaturatedLearners:
    def test_each_cell_rule_is_the_sign_of_the_cell_sum_under_both_losses(self) -> None:
        fits = _cell_fits()
        cell = np.asarray(fits["blip"].data.covariates, dtype=float).argmax(axis=1)
        pieces = _fold_pseudo_outcomes(fits["blip"])
        # The same inner seeds and templates give the same pseudo-outcomes under both losses.
        for (_, _, blip), (_, _, weighted) in zip(
            pieces, _fold_pseudo_outcomes(fits["weighted_classification"]), strict=True
        ):
            np.testing.assert_array_equal(blip, weighted)
        rules = {loss: law.rule_of(result) for loss, result in fits.items()}
        np.testing.assert_array_equal(rules["blip"], rules["weighted_classification"])
        for train, test, target in pieces:
            for k in range(len(CELLS)):
                in_cell = cell[train] == k
                mean = float(np.mean(target[in_cell]))
                # The identity is checked away from a tie, so a solver tolerance cannot flip it.
                assert abs(mean) > 0.02, (k, mean)
                expected = bool(np.sum(target[in_cell]) > 0.0)
                np.testing.assert_array_equal(rules["blip"][test][cell[test] == k], expected)

    def test_the_blip_score_is_the_cell_mean_of_the_pseudo_outcome(self) -> None:
        result = _cell_fits()["blip"]
        covariates = np.asarray(result.data.covariates, dtype=float)
        cell = covariates.argmax(axis=1)
        (split,) = result.extra["learned_rule"].splits
        train, _ = next(iter(result.nuisance.folds))
        fitted = fit_fold_rule(
            result.data,
            np.asarray(train),
            loss="blip",
            learner=LinearRegression(),
            fit_inner=partial(
                result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler
            ),
            scaler=result.nuisance.scaler,
            g_bounds=result.config.g_bounds,
            seed=split.inner_seeds[0],
        )
        score = fitted.score(np.eye(len(CELLS)))
        means = [np.mean(fitted.pseudo_outcome[cell[train] == k]) for k in range(len(CELLS))]
        np.testing.assert_allclose(score, means, rtol=0, atol=1e-10)

    def test_the_majority_sign_mutation_differs_on_the_declared_cell(self) -> None:
        """An unweighted classifier learns the majority sign, which differs in one cell."""
        result = _cell_fits()["weighted_classification"]
        cell = np.asarray(result.data.covariates, dtype=float).argmax(axis=1)
        rule = law.rule_of(result)
        for train, test, target in _fold_pseudo_outcomes(result):
            covariates = np.eye(len(CELLS))[cell[train]]
            majority = _saturated_classifier().fit(covariates, (target > 0.0).astype(int))
            unweighted = majority.predict_proba(np.eye(len(CELLS)))[:, 1] > 0.5
            weighted = np.array([rule[test][cell[test] == k][0] for k in range(len(CELLS))])
            differs = np.flatnonzero(unweighted != weighted)
            np.testing.assert_array_equal(differs, [MAJORITY_CELL])
            assert weighted[MAJORITY_CELL]
            assert np.mean(target[cell[train] == MAJORITY_CELL] > 0.0) < 0.4


# ------------------------------------------------------------------------------------ W-g3


class OuterPseudoOutcome(TMLE):
    """The W-g3 mutation: the pseudo-outcome reads the outer cross-fitted predictions.

    At a row of fold ``v``'s training rows, the outer prediction comes from a model trained on
    rows that include fold ``v``'s validation rows, so fold ``v``'s rule reads their outcomes.
    """

    def _regimes(self, data: Any, estimates: Any, *, seed: int | None = None) -> RegimeSet:
        assert self.learned_rule is not None
        target = pseudo_outcome(
            estimates,
            estimates.scaler.scale(np.asarray(data.outcome, dtype=float)),
            np.asarray(data.treatment, dtype=float),
            self._rule_g_bounds(data),
        )
        covariates = np.asarray(data.covariates, dtype=float)
        treat = np.zeros(data.n, dtype=bool)
        for train, test in estimates.folds:
            model = blip_learner().fit(covariates[train], target[train])
            rule = FoldRule("blip", model, False, 0, target[train])
            treat[test] = rule.treat(covariates[test])
        values = np.zeros((data.n, 2, 1), dtype=float)
        values[:, 1, 0] = treat
        values[:, 0, 0] = ~treat
        return RegimeSet((self.learned_rule.name,), values, 0.0)


def _locality_fits(kind: type[TMLE]) -> tuple[Any, Any, np.ndarray]:
    """A fit, and the fit with every outcome of fold 0's validation rows flipped."""
    frame = law.draw("misspecified_limit", 1000, 3)
    settings = {
        "learned_rule": blip_rule(),
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


class TestWg3TheRuleIsFoldLocal:
    def test_a_validation_outcome_never_moves_its_folds_rule(self) -> None:
        first, second, test = _locality_fits(TMLE)
        np.testing.assert_array_equal(
            first.nuisance.folds.assignment, second.nuisance.folds.assignment
        )
        np.testing.assert_array_equal(law.rule_of(first)[test], law.rule_of(second)[test])
        # The flipped rows are training rows of every other fold, so those rules move.
        assert np.any(law.rule_of(first) != law.rule_of(second))

    def test_the_fold_rule_score_is_unchanged_bit_for_bit(self) -> None:
        first, second, test = _locality_fits(TMLE)
        (split,) = first.extra["learned_rule"].splits
        train, _ = next(iter(first.nuisance.folds))
        scores = []
        for result in (first, second):
            fitted = fit_fold_rule(
                result.data,
                np.asarray(train),
                loss="blip",
                learner=blip_learner(),
                fit_inner=partial(
                    result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler
                ),
                scaler=result.nuisance.scaler,
                g_bounds=result.config.g_bounds,
                seed=split.inner_seeds[0],
            )
            scores.append(fitted.score(np.asarray(result.data.covariates, dtype=float)[test]))
        np.testing.assert_array_equal(scores[0], scores[1])

    def test_the_outer_prediction_mutation_moves_the_rule(self) -> None:
        first, second, test = _locality_fits(OuterPseudoOutcome)
        moved = int(np.sum(law.rule_of(first)[test] != law.rule_of(second)[test]))
        assert moved >= 1


# ------------------------------------------------------------------------------------ W-g4


class TestWg4OwnRowLeakage:
    def test_a_training_outcome_moves_its_pseudo_outcome_by_its_own_term(self) -> None:
        result = _blip_fit()
        data = result.data
        train, _ = next(iter(result.nuisance.folds))
        train = np.asarray(train)
        seed = result.extra["learned_rule"].splits[0].inner_seeds[0]
        fit_inner = partial(result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler)
        bounds = result.config.g_bounds

        def pseudo(source: Any) -> np.ndarray:
            return fit_fold_rule(
                source,
                train,
                loss="blip",
                learner=blip_learner(),
                fit_inner=fit_inner,
                scaler=result.nuisance.scaler,
                g_bounds=bounds,
                seed=seed,
            ).pseudo_outcome

        position = 17
        row = train[position]
        outcome = np.asarray(data.outcome, dtype=float).copy()
        delta = 1.0 - 2.0 * outcome[row]
        outcome[row] += delta
        before, after = pseudo(data), pseudo(replace(data, outcome=outcome))
        inner = data.subset(train)
        folds = crossfit.random_partition(inner.n, INNER_FOLDS, cluster=None, seed=seed)
        nuisance = fit_inner(inner, folds, seed)
        a = float(inner.treatment[position])
        g = nuisance.bounded_propensity(bounds)[position, nuisance.propensity.column_for(a)]
        change = after - before
        assert change[position] == pytest.approx((2.0 * a - 1.0) * delta / g, rel=0, abs=1e-12)
        same_fold = folds.assignment == folds.assignment[position]
        same_fold[position] = False
        np.testing.assert_array_equal(change[same_fold], 0.0)
        # Every other inner fold trained on the row, so its pseudo-outcomes move.
        assert np.min(np.abs(change[~same_fold & (np.arange(train.size) != position)])) > 0.0

    def test_a_binding_truncation_divides_by_the_truncated_mechanism(self) -> None:
        """At a row where ``g_bounds`` binds, the change is ``(2A - 1) dY / max(g, l)``.

        The witness is nonzero: on this draw the row's fitted ``g(A | W)`` is below the bound,
        so dividing by the untruncated ``g`` gives another change.
        """
        result = _blip_fit()
        bounds = (0.45, 0.55)
        train = np.asarray(next(iter(result.nuisance.folds))[0])
        seed = result.extra["learned_rule"].splits[0].inner_seeds[0]
        fit_inner = partial(result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler)
        inner = result.data.subset(train)
        folds = crossfit.random_partition(inner.n, INNER_FOLDS, cluster=None, seed=seed)
        nuisance = fit_inner(inner, folds, seed)
        arms = np.asarray(inner.treatment, dtype=float)
        raw = np.where(arms == 1.0, nuisance.propensity.arm(1.0), nuisance.propensity.arm(0.0))
        position = int(np.flatnonzero(raw < 0.43)[0])
        g, a = float(raw[position]), float(arms[position])
        assert g < bounds[0]
        outcome = np.asarray(result.data.outcome, dtype=float).copy()
        row = train[position]
        delta = 1.0 - 2.0 * outcome[row]
        outcome[row] += delta

        def pseudo(source: Any) -> np.ndarray:
            return fit_fold_rule(
                source,
                train,
                loss="blip",
                learner=blip_learner(),
                fit_inner=fit_inner,
                scaler=result.nuisance.scaler,
                g_bounds=bounds,
                seed=seed,
            ).pseudo_outcome

        change = pseudo(replace(result.data, outcome=outcome)) - pseudo(result.data)
        expected = (2.0 * a - 1.0) * delta / bounds[0]
        assert change[position] == pytest.approx(expected, rel=0, abs=1e-12)
        assert abs((2.0 * a - 1.0) * delta / g - expected) > 0.05

    def test_a_continuous_outcome_moves_its_pseudo_outcome_on_the_scaled_outcome(self) -> None:
        """On ``Y`` in ``[a, b]`` with ``q_bounds=(a, b)``, the change is ``(2A - 1) dY / {(b - a) g}``.

        The witness is nonzero: ``b - a = 6``, so the raw outcome gives six times the change.
        """
        low, high = 1.0, 7.0
        frame = support.law_frame(400, 0)
        rng = np.random.default_rng(5)
        frame = frame.assign(Y=2.0 + 2.0 * frame["Y"] + rng.uniform(0.0, 2.0, len(frame)))
        result = (
            support.learned_tmle(
                learned_rule=blip_rule(),
                outcome_learner=LinearRegression(),
                q_bounds=(low, high),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES)
            .single()
        )
        assert result.data.family != "binomial"
        train = np.asarray(next(iter(result.nuisance.folds))[0])
        seed = result.extra["learned_rule"].splits[0].inner_seeds[0]
        fit_inner = partial(result.estimator._inner_rule_nuisances, scaler=result.nuisance.scaler)
        bounds = result.config.g_bounds

        def pseudo(source: Any) -> np.ndarray:
            return fit_fold_rule(
                source,
                train,
                loss="blip",
                learner=blip_learner(),
                fit_inner=fit_inner,
                scaler=result.nuisance.scaler,
                g_bounds=bounds,
                seed=seed,
            ).pseudo_outcome

        position, delta = 11, 0.5
        outcome = np.asarray(result.data.outcome, dtype=float).copy()
        outcome[train[position]] += delta
        change = pseudo(replace(result.data, outcome=outcome)) - pseudo(result.data)
        inner = result.data.subset(train)
        folds = crossfit.random_partition(inner.n, INNER_FOLDS, cluster=None, seed=seed)
        nuisance = fit_inner(inner, folds, seed)
        a = float(inner.treatment[position])
        g = nuisance.bounded_propensity(bounds)[position, nuisance.propensity.column_for(a)]
        expected = (2.0 * a - 1.0) * delta / ((high - low) * g)
        assert change[position] == pytest.approx(expected, rel=0, abs=1e-12)
        assert abs((2.0 * a - 1.0) * delta / g - expected) > 0.1


# ----------------------------------------------------------------- the rule of a fold, exactly


class TestTheFoldRule:
    """The one-class fold and the tie, which no draw of a continuous law reaches."""

    @pytest.mark.filterwarnings("ignore:targeting step did not drive")
    @pytest.mark.parametrize(("outcome", "treats"), [("A", True), ("1 - A", False)])
    def test_a_one_class_fold_gives_the_constant_rule_of_its_label(
        self, outcome: str, treats: bool
    ) -> None:
        """With ``Y = A`` every pseudo-outcome is positive, and with ``Y = 1 - A`` negative.

        Each fold's positive-weight rows then carry one label, so the rule is that label's
        constant and the classifier is never fitted.
        """
        frame = support.law_frame(400, 0)
        frame = frame.assign(Y=frame["A"] if outcome == "A" else 1.0 - frame["A"])
        rule = LearnedRule(loss="weighted_classification", learner=UnfittableClassifier())
        result = (
            support.learned_tmle(learned_rule=rule)
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES)
            .single()
        )
        np.testing.assert_array_equal(law.rule_of(result), treats)
        for target in (piece for _, _, piece in _fold_pseudo_outcomes(result)):
            assert np.all(target > 0.0) if treats else np.all(target < 0.0)

    @pytest.mark.parametrize(
        ("loss", "scores"),
        [("blip", (-0.2, 0.0, 0.2)), ("weighted_classification", (0.3, 0.5, 0.7))],
    )
    def test_a_score_at_the_threshold_assigns_control(
        self, loss: str, scores: tuple[float, ...]
    ) -> None:
        rule = FoldRule(loss, FixedScore(), False, 0, np.zeros(1))
        covariates = np.array(scores).reshape(-1, 1)
        np.testing.assert_array_equal(rule.treat(covariates), [False, False, True])
        np.testing.assert_allclose(rule.margin(covariates), [-0.2, 0.0, 0.2], atol=1e-15)

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.PositivityWarning")
    def test_an_inner_fold_without_an_arm_refuses_with_a_remedy(self) -> None:
        """Three treated rows: the outer split fits, and an inner complement has none."""
        frame = support.law_frame(60, 0)
        treated = np.zeros(len(frame))
        treated[[8, 48, 56]] = 1.0
        frame = frame.assign(A=treated)
        support.fit(frame)  # the plug-in rule fits this sample
        with pytest.raises(DataError, match=r"inner fold \d of outer fold \d") as caught:
            support.fit(frame, learned_rule=LearnedRule(loss="blip", learner=LinearRegression()))
        message = str(caught.value)
        assert "no row in treatment arm" in message
        assert "or with loss='plug_in'" in message


# ---------------------------------------------------------------------------- compositions


def _weighted_frame() -> pd.DataFrame:
    return support.law_frame(400, 0).assign(w=lambda f: 1.0 + 1.5 * (f["W1"] > 0))


class TestTheCompositions:
    """Weights, clusters, fold targeting and repeats run, and no study covers them."""

    def test_weights_reach_the_inner_fits_and_the_rule_learner(self) -> None:
        SpyRegressor.calls = []
        frame = _weighted_frame()
        fitted = (
            support.learned_tmle(learned_rule=LearnedRule(loss="blip", learner=SpyRegressor()))
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        weights = np.asarray(fitted.data.weights, dtype=float)
        assert len(SpyRegressor.calls) >= 5
        for call, (train, _) in zip(SpyRegressor.calls[-5:], fitted.nuisance.folds, strict=True):
            expected = weights[np.asarray(train)]
            np.testing.assert_allclose(
                call["weight"] / call["weight"].mean(), expected / expected.mean(), rtol=1e-12
            )
        assert fitted.estimates[support.NAME].support.label == "experimental"

    def test_every_inner_outcome_and_treatment_fit_is_weighted(self) -> None:
        """The 25 inner fits of each nuisance get their rows' observation weights."""
        WeightSpyOutcome.calls, WeightSpyTreatment.calls = [], []
        frame = _weighted_frame()
        fitted = (
            support.learned_tmle(
                learned_rule=LearnedRule(loss="blip", learner=LinearRegression()),
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
            assert len(inner) == 5 * INNER_FOLDS
            for call in inner:
                # The design holds W1, and the law's weight is 1 + 1.5 * (W1 > 0).
                (column,) = [
                    j for j in range(call["x"].shape[1]) if np.isin(call["x"][:, j], w1).all()
                ]
                expected = 1.0 + 1.5 * (call["x"][:, column] > 0.0)
                assert call["weight"] is not None
                weight = np.asarray(call["weight"], dtype=float)
                np.testing.assert_allclose(
                    weight / weight.mean(), expected / expected.mean(), rtol=1e-12
                )

    def test_weighted_classification_weights_by_the_size_of_the_pseudo_outcome(self) -> None:
        SpyClassifier.calls = []
        frame = _weighted_frame()
        rule = LearnedRule(loss="weighted_classification", learner=SpyClassifier())
        fitted = (
            support.learned_tmle(learned_rule=rule)
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        weights = np.asarray(fitted.data.weights, dtype=float)
        targets = _fold_pseudo_outcomes(fitted)
        for call, (train, _, target) in zip(SpyClassifier.calls[-5:], targets, strict=True):
            np.testing.assert_array_equal(call["y"], (target > 0.0).astype(float))
            expected = np.abs(target) * weights[train]
            np.testing.assert_allclose(
                call["weight"] / call["weight"].sum(), expected / expected.sum(), rtol=1e-10
            )

    def test_clusters_stay_whole_in_the_inner_folds_and_reach_the_rule_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn: list[tuple[Any, Any]] = []
        original = crossfit.random_partition

        def recording(n: int, n_folds: int, *, cluster: Any = None, seed: int) -> Any:
            folds = original(n, n_folds, cluster=cluster, seed=seed)
            drawn.append((cluster, folds))
            return folds

        monkeypatch.setattr(crossfit, "random_partition", recording)
        SpyRegressor.calls = []
        frame = support.law_frame(400, 0).assign(c=lambda f: np.arange(len(f)) // 4)
        fitted = (
            support.learned_tmle(learned_rule=LearnedRule(loss="blip", learner=SpyRegressor()))
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, id="c")
            .single()
        )
        inner = [(cluster, folds) for cluster, folds in drawn if folds.n < len(frame)]
        assert len(inner) == 5
        for cluster, folds in inner:
            assert cluster is not None
            for code in np.unique(cluster):
                assert np.unique(folds.assignment[cluster == code]).size == 1
        for call in SpyRegressor.calls[-5:]:
            assert call["groups"] is not None
        assert fitted.estimates[support.NAME].support.label == "experimental"

    def test_fold_targeting_runs(self) -> None:
        result = support.fit(learned_rule=blip_rule(), targeting_scheme="fold")
        assert np.isfinite(result.estimates[support.NAME].std_error)

    def test_the_mean_over_splits_is_the_mean_of_single_split_fits(self) -> None:
        repeated = support.fit(learned_rule=blip_rule(), repeats=2)
        singles = [
            support.fit(learned_rule=blip_rule(), random_state=repeat.seed)
            for repeat in repeated.repeats
        ]
        assert repeated.estimates[support.NAME].psi == np.mean(
            [single.estimates[support.NAME].psi for single in singles]
        )
        record = repeated.extra["learned_rule"]
        for split, single in zip(record.splits, singles, strict=True):
            assert split.inner_seeds == single.extra["learned_rule"].splits[0].inner_seeds


# ---------------------------------------------------------------------------- the harness


class TestTheStudyHarnessReproducesTheRuleLearners:
    """Rule L4 holds for a rule learner: the harness refits each fold's rule exactly."""

    @pytest.mark.parametrize(
        "rule",
        [blip_rule(), LearnedRule(loss="weighted_classification", learner=classifier())],
        ids=["blip", "weighted_classification"],
    )
    def test_the_refit_reproduces_the_fits_rule(self, rule: LearnedRule) -> None:
        frame = law.draw("misspecified_limit", 600, 5)
        _, truth, _ = law.measured(frame, law.outcome_learner(), "misspecified_limit", 3, rule=rule)
        assert truth.rows_checked == 600
        assert 0.4 < truth.truth < 0.7

    def test_the_validation_row_control_learns_another_rule(self) -> None:
        frame = law.draw("misspecified_limit", 600, 5)
        local = law.fit(frame, law.outcome_learner(), 3, rule=blip_rule())
        control = law.fit(frame, law.outcome_learner(), 3, mutation=True, rule=blip_rule())
        assert np.any(law.rule_of(local) != law.rule_of(control))
        truth = law.harness(
            control, law.outcome_learner(), "misspecified_limit", source="validation"
        )
        assert truth.rows_checked == 600
        with pytest.raises(law.HarnessMismatch):
            law.harness(control, law.outcome_learner(), "misspecified_limit")


# ------------------------------------------------------------------ persistence and studies


@pytest.mark.parametrize("route", ["save", "dumps"])
def test_a_round_trip_keeps_the_rule_and_the_record(route: str, tmp_path: Any) -> None:
    from cleverly import load
    from cleverly.estimators.serialize import dumps, loads

    result = _blip_fit()
    restored = (
        load(result.save(tmp_path / "blip.joblib")) if route == "save" else loads(dumps(result))
    )
    assert restored.extra["learned_rule"] == result.extra["learned_rule"]
    np.testing.assert_array_equal(restored.nuisance.regimes.values, result.nuisance.regimes.values)
    assert restored.estimator.learned_rule.loss == "blip"


def test_retarget_reads_the_saved_rule() -> None:
    """``retarget`` reads the rule on the saved nuisances, so it reproduces the fit."""
    result = _blip_fit()
    estimates, _ = result.estimator.retarget(
        result.data, result.nuisance, estimands=("ey_learned_rule",)
    )
    assert estimates[support.NAME].psi == pytest.approx(
        result.estimates[support.NAME].psi, abs=1e-12
    )


def test_a_study_fit_reports_the_engine_estimate_and_names_the_learner() -> None:
    from cleverly import CausalStudy, LearnedRuleValue, PointTreatment

    study = CausalStudy(
        support.law_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    effect = study.identify(LearnedRuleValue(rule=blip_rule()))
    assert "doubly robust pseudo-outcome" in effect.functional.expression
    fitted = effect.estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )
    assert fitted.estimates[support.NAME].psi == pytest.approx(
        _blip_fit().estimates[support.NAME].psi, abs=1e-12
    )
