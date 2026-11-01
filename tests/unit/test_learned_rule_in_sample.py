"""One learned rule fitted on all rows: ``cross_fit=False`` on a learned-rule fit (X11 (a)).

The in-sample fit learns one rule :math:`d_n` on every row and estimates the value of that
full-sample rule, :math:`\\Psi_{d_n}(P_0)` (van der Laan and Luedtke 2015, Section 6, Theorem 5),
with the shipped in-sample regime TMLE at :math:`d = d_n`.  It is neither the fold average of a
cross-fitted fit nor the optimal value.  The witnesses below are the plan's (``X11-wave2-plan.md``
rev 2.1, section 3), with W-a3 and W-a4 as rev 2 replaced them:

- W-a1 (reduction): a stub outcome regression whose blip sign is ``sign(W1)`` gives the same
  point, curve and standard error as the known rule ``1{W1 > 0}`` fitted in sample, at a
  nonzero fluctuation.  The omitted-variable bound of the two fits agrees too, limits included.
- W-a2 (rule, mutation): each row's rule is the sign of the blip of the outcome regression
  refitted on every row.  The rule read off cross-fitted predictions differs on at least 1% of
  the rows.  A rule learner learns its one rule on every row from an inner cross-fit over all
  rows.
- W-a3 (target distinction): on a fixed draw the full-sample rule and at least one fold rule of
  a 10-fold split disagree on at least 2% of the truth grid, and the fold-rule harness of the
  learned-rule studies refuses the in-sample fit by rule L4.
- W-a4 (known g enters H): with ``treatment_probabilities=`` the fit solves its score at the
  supplied probabilities.  The mutation that substitutes an estimated g leaves the score
  unsolved by more than 1e-3 and moves the point by more than 0.01 on a fixed draw.

The label is option C: under the in-sample scheme only, the support key reads the learner
families, and an adaptive learner makes the fit warn.  A wave-1 study fit keeps its row.
"""

from __future__ import annotations

import types
import warnings
from dataclasses import replace
from functools import cache, partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.datasets import learned_rule_truth, make_treatment_rules, treatment_rule_value
from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import InSampleLearnerWarning
from cleverly.inference import influence_variance
from cleverly.interventions import LearnedRule, Rule
from cleverly.interventions._rule_learners import fit_fold_rule, inner_seed
from cleverly.interventions.learned import TARGET_KIND, in_sample_learner_condition
from cleverly.sensitivity import omitted_variable_bounds
from tests.studies import _learned_rule_law as law
from tests.unit import _learned_rule_support as support

#: The in-sample scheme of a learned-rule fit.
IN_SAMPLE: dict[str, Any] = {"cross_fit": False, "cv_evaluation": False}

#: The warning of an in-sample fit with an adaptive learner.  The support module's learners
#: are Pipelines, which ``learner_family`` reads as adaptive.
WARNING = "This in-sample learned-rule fit"

pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.InSampleLearnerWarning")

#: The declared shares and moves of the witnesses, each below the value the fixed draw gives.
W_A2_ROW_SHARE = 0.01  # observed 0.018
W_A3_GRID_SHARE = 0.02  # observed 0.035 at the most distant fold
W_A4_POINT_MOVE = 0.01  # observed 0.017
W_A4_SCORE = 1e-3  # observed 0.017


class InteractionLogistic(ClassifierMixin, BaseEstimator):
    """The support module's outcome learner, marked parametric for :func:`learner_family`.

    Every pairwise product of ``(A, W1, W2)``, then a logistic regression: a fixed dimension
    and no selection, so condition C5a holds.  The plan's study declares this kind of class.
    """

    cleverly_learner_family: ClassVar[str] = "parametric"

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InteractionLogistic:
        options = {} if sample_weight is None else {"fit__sample_weight": sample_weight}
        self.model_ = support.outcome_learner().fit(np.asarray(X, dtype=float), y, **options)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(np.asarray(X, dtype=float))


class SignStub(ClassifierMixin, BaseEstimator):
    """A fixed outcome regression whose blip has the sign of ``W1``: W-a1's stub.

    ``logit Qbar(a, w) = 0.4 w1 + a w1`` on the design ``(A, W1, W2)``, so
    ``Qbar(1, w) - Qbar(0, w)`` has the sign of ``w1``.  It omits ``W2`` and the intercept of the
    law, so the fluctuation moves.  It learns nothing, so a fit in sample and a fit of the known
    rule read the same regression.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SignStub:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        p = expit(0.4 * design[:, 1] + design[:, 0] * design[:, 1])
        return np.column_stack([1.0 - p, p])


def _fit(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    """An in-sample learned-rule fit of the support law."""
    return support.fit(frame, **{**IN_SAMPLE, **overrides})


def _full_sample_model(result: Any, learner: Any) -> Any:
    """``learner`` refitted on every row of ``result``, with the fit's weights."""
    data = result.data
    return fit_on_rows(
        learner,
        np.asarray(data.treatment_design(), dtype=float),
        np.asarray(data.outcome, dtype=float),
        np.asarray(data.weights, dtype=float),
        np.arange(data.n),
        "classification",
        None,
    )


# ------------------------------------------------------------------------------- W-a1


@cache
def _reduction() -> tuple[Any, Any]:
    """The in-sample learned-rule fit and the known-rule fit of W-a1, on one draw."""
    frame = support.law_frame(400, 0)
    common: dict[str, Any] = {
        "outcome_learner": SignStub(),
        "treatment_learner": support.treatment_learner(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    fit = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
    learned = TMLE(learned_rule=LearnedRule(), **common).fit(frame, **fit).single()
    rule = Rule(lambda w: (w["W1"] > 0).astype(float), name="w1", rule_kind="known")
    known = (
        TMLE(interventions=[rule], estimands=("ey_regime",), **common).fit(frame, **fit).single()
    )
    return learned, known


class TestWa1Reduction:
    """W-a1: the in-sample fit at a known rule is the shipped in-sample regime TMLE."""

    def test_the_point_curve_and_standard_error_equal_the_known_rule_fit(self) -> None:
        learned, known = _reduction()
        a = learned.estimates[support.NAME]
        b = known.estimates["ey_regime[w1]"]
        assert np.array_equal(law.rule_of(learned), law.rule_of(known))
        assert a.psi == pytest.approx(b.psi, rel=1e-12, abs=1e-14)
        assert a.variance == pytest.approx(b.variance, rel=1e-12)
        np.testing.assert_allclose(a.influence_curve, b.influence_curve, rtol=0, atol=1e-12)

    def test_the_fluctuation_moves(self) -> None:
        """The nonzero witness: the comparison runs through a nonzero epsilon."""
        learned, _ = _reduction()
        assert abs(float(learned.fluctuations["regime"].epsilon[0])) > 1e-2

    def test_the_variance_is_the_centred_in_sample_rule(self) -> None:
        """The stored variance is ``{n(n-1)}^-1 sum (phi - phibar)^2``, the shipped rule."""
        learned, _ = _reduction()
        estimate = learned.estimates[support.NAME]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        n = curve.size
        assert abs(curve.mean()) < 1e-10
        assert estimate.variance == pytest.approx(influence_variance(curve), rel=1e-12)
        assert estimate.variance == pytest.approx(
            np.sum((curve - curve.mean()) ** 2) / (n * (n - 1)), rel=1e-12
        )

    def test_the_bound_equals_the_known_rule_bound_with_limits(self) -> None:
        """At one fold the combined representer is the known-regime representer at the rule,
        so the bound and its limits match, and the in-sample limits run (plan section 3)."""
        learned, known = _reduction()
        strengths = {"cf_y": 0.05, "cf_d": 0.05, "nu2_estimator": "doubly_robust"}
        a = omitted_variable_bounds(learned, support.NAME, **strengths)
        b = omitted_variable_bounds(known, "ey_regime[w1]", **strengths)
        for field in ("max_bias", "lower", "upper", "ci_lower", "ci_upper"):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-10), field
        assert np.isfinite(a.ci_lower) and a.ci_lower < a.lower
        assert a.limit_refusal is None
        assert a.limit_descriptive

    def test_the_cross_fitted_bound_still_withholds_its_limits(self) -> None:
        """The control: a fold-evaluated fit keeps the X11 (n2) refusal of its limits."""
        bounds = omitted_variable_bounds(
            support.fit(), support.NAME, cf_y=0.05, cf_d=0.05, nu2_estimator="doubly_robust"
        )
        assert bounds.limit_refusal is not None
        with pytest.raises(Exception, match="point bound"):
            _ = bounds.ci_lower


# ------------------------------------------------------------------------------- W-a2


@cache
def _law_fits() -> tuple[Any, Any]:
    """An in-sample fit and a 10-fold fold-evaluated fit of one study-law draw, n = 1,000."""
    frame = law.draw("non_exceptional", 1000, 11)

    def estimator(**overrides: Any) -> TMLE:
        settings: dict[str, Any] = {
            "learned_rule": LearnedRule(),
            "outcome_learner": law.outcome_learner(),
            "treatment_learner": law.treatment_learner(),
            "n_folds": 10,
            "simultaneous": False,
            "random_state": 3,
            **overrides,
        }
        return TMLE(**settings)

    fit = {"outcome": "Y", "treatment": "A", "covariates": list(law.COVARIATES)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InSampleLearnerWarning)
        in_sample = estimator(**IN_SAMPLE).fit(frame, **fit).single()
    cross_fitted = estimator(cv_evaluation=True).fit(frame, **fit).single()
    return in_sample, cross_fitted


class TestWa2TheRule:
    """W-a2: the rule of every row is the sign of the full-sample blip."""

    def test_each_row_reads_the_full_sample_blip(self) -> None:
        in_sample, _ = _law_fits()
        model = _full_sample_model(in_sample, law.outcome_learner())
        covariates = np.asarray(in_sample.data.covariates, dtype=float)
        assert np.array_equal(law._blip(model, covariates) > 0.0, law.rule_of(in_sample))

    def test_the_mutation_that_reads_cross_fitted_predictions_differs(self) -> None:
        in_sample, cross_fitted = _law_fits()
        share = float(np.mean(law.rule_of(in_sample) != law.rule_of(cross_fitted)))
        assert share >= W_A2_ROW_SHARE, share

    def test_the_weighted_rule_is_the_weighted_full_sample_rule(self) -> None:
        """With weights= the regression is weighted, as under cross-fitting (X11 (k))."""
        frame = support.law_frame(400, 0).assign(w=lambda f: 0.5 + f["W2"] + (f["W1"] > 0))
        result = (
            support.learned_tmle(**IN_SAMPLE)
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        covariates = np.asarray(result.data.covariates, dtype=float)
        model = _full_sample_model(result, support.outcome_learner())
        assert np.array_equal(support.blip(model, covariates) > 0.0, law.rule_of(result))
        # The mutation that drops the weights learns another rule.
        unweighted = fit_on_rows(
            support.outcome_learner(),
            np.asarray(result.data.treatment_design(), dtype=float),
            np.asarray(result.data.outcome, dtype=float),
            np.ones(result.data.n),
            np.arange(result.data.n),
            "classification",
            None,
        )
        assert np.any((support.blip(unweighted, covariates) > 0.0) != law.rule_of(result))

    @pytest.mark.parametrize(
        "rule",
        [
            LearnedRule(loss="blip", learner=LinearRegression()),
            LearnedRule(loss="weighted_classification", learner=InteractionLogistic()),
        ],
        ids=["blip", "weighted_classification"],
    )
    def test_a_rule_learner_learns_its_one_rule_on_every_row(self, rule: LearnedRule) -> None:
        result = _fit(learned_rule=rule)
        data, nuisance = result.data, result.nuisance
        split = result.extra["learned_rule"].splits[0]
        seed = int(result.repeats[0].seed)
        assert split.inner_seeds == (inner_seed(seed, 0),)
        refit = partial(
            fit_fold_rule,
            data,
            loss=rule.loss,
            learner=rule.learner,
            fit_inner=partial(result.estimator._inner_rule_nuisances, scaler=nuisance.scaler),
            scaler=nuisance.scaler,
            g_bounds=result.config.g_bounds,
            seed=split.inner_seeds[0],
        )
        covariates = np.asarray(data.covariates, dtype=float)
        assert np.array_equal(refit(np.arange(data.n)).treat(covariates), law.rule_of(result))
        # The mutation that learns the rule on a training complement differs on some row.
        half = np.arange(0, data.n, 2)
        assert np.any(refit(half).treat(covariates) != law.rule_of(result))
        assert len(split.score_quantiles) == 1


# ------------------------------------------------------------------------------- W-a3


class TestWa3TheTargetIsTheFullSampleRule:
    """W-a3: the full-sample rule is not a fold rule, and the fold harness refuses the fit."""

    def _grid_rules(self, model: Any) -> np.ndarray:
        return np.concatenate(
            [
                law._blip(model, np.column_stack([law.GRID, np.full_like(law.GRID, w2)])) > 0.0
                for w2 in law.W2_LEVELS
            ]
        )

    def test_a_fold_rule_disagrees_with_the_full_sample_rule_on_the_grid(self) -> None:
        in_sample, cross_fitted = _law_fits()
        full = self._grid_rules(_full_sample_model(in_sample, law.outcome_learner()))
        data = in_sample.data
        shares = []
        for train, _ in cross_fitted.nuisance.folds:
            model = fit_on_rows(
                law.outcome_learner(),
                np.asarray(data.treatment_design(), dtype=float),
                np.asarray(data.outcome, dtype=float),
                np.asarray(data.weights, dtype=float),
                np.asarray(train),
                "classification",
                None,
            )
            shares.append(float(np.mean(self._grid_rules(model) != full)))
        assert max(shares) >= W_A3_GRID_SHARE, shares

    def test_the_fold_rule_harness_refuses_the_in_sample_fit(self) -> None:
        """Rule L4: a truth refit of the fold rules differs from the fit's rule on a row."""
        in_sample, cross_fitted = _law_fits()
        view = types.SimpleNamespace(
            data=in_sample.data,
            nuisance=replace(in_sample.nuisance, folds=cross_fitted.nuisance.folds),
            estimator=in_sample.estimator,
        )
        with pytest.raises(law.HarnessMismatch, match="differs from the fit's rule"):
            law.harness(view, law.outcome_learner(), "non_exceptional")

    def test_the_full_sample_truth_checks_every_row(self) -> None:
        """The in-sample truth refits on every row, and its rule is the fit's on all of them."""
        in_sample, _ = _law_fits()
        model = _full_sample_model(in_sample, law.outcome_learner())
        covariates = np.asarray(in_sample.data.covariates, dtype=float)
        assert np.array_equal(law._blip(model, covariates) > 0.0, law.rule_of(in_sample))
        grid = {
            w2: law._blip(model, np.column_stack([law.GRID, np.full_like(law.GRID, w2)])) > 0.0
            for w2 in law.W2_LEVELS
        }
        value, variance = law.rule_value(grid, "non_exceptional")
        assert 0.0 < value < 1.0 and variance > 0.0

    def test_the_study_harness_reads_the_one_rule_of_an_in_sample_fit(self) -> None:
        """``_learned_rule_law.fold_rule`` learns a single fold's rule on every row.

        So the harness checks rule L4 on every row, its truth is the value of the one
        full-sample rule refitted independently here, and its oracle SE is ``sqrt(v / n)``.
        Without the branch the plug-in refit has no training rows and fails.
        """
        in_sample, _ = _law_fits()
        n = in_sample.data.n
        truth = law.harness(in_sample, law.outcome_learner(), "non_exceptional")
        model = _full_sample_model(in_sample, law.outcome_learner())
        grid = {
            w2: law._blip(model, np.column_stack([law.GRID, np.full_like(law.GRID, w2)])) > 0.0
            for w2 in law.W2_LEVELS
        }
        value, variance = law.rule_value(grid, "non_exceptional")
        assert truth.rows_checked == n
        assert truth.fold_values == pytest.approx((value,), abs=1e-12)
        assert truth.truth == pytest.approx(value, abs=1e-12)
        assert truth.oracle_se == pytest.approx(np.sqrt(variance / n), rel=1e-12)

    def test_the_study_fold_rule_reproduces_an_in_sample_rule_learner(self) -> None:
        result = _fit(learned_rule=LearnedRule(loss="blip", learner=LinearRegression()))
        covariates = np.asarray(result.data.covariates, dtype=float)
        rule = law.fold_rule(result, None, 0)
        assert np.array_equal(rule(covariates), law.rule_of(result))

    def test_the_truth_helper_refits_the_full_sample_rule(self) -> None:
        """``learned_rule_truth`` of an in-sample fit is the value of the one full-sample rule,
        refitted here independently, and differs from a fold rule's value."""
        frame, _ = make_treatment_rules(n=600, seed=4)
        learner = support.outcome_learner()
        fit = {"outcome": "home_30", "treatment": "offer", "covariates": ("need", "lives_alone")}
        result = (
            TMLE(
                learned_rule=LearnedRule(),
                outcome_learner=learner,
                treatment_learner=support.treatment_learner(),
                simultaneous=False,
                random_state=0,
                **IN_SAMPLE,
            )
            .fit(frame, **fit)
            .single()
        )
        design = frame[["offer", "need", "lives_alone"]].to_numpy(dtype=float)
        model = support.outcome_learner().fit(design, frame["home_30"].to_numpy())

        def rule(rows: pd.DataFrame) -> np.ndarray:
            return (support.blip(model, rows[["need", "lives_alone"]].to_numpy()) > 0.0).astype(
                float
            )

        truth = learned_rule_truth(result, support.outcome_learner())
        assert result.extra["learned_rule"].splits[0].fold_weights == (1.0,)
        assert truth == pytest.approx(treatment_rule_value(rule), abs=1e-12)
        half = support.outcome_learner().fit(design[::2], frame["home_30"].to_numpy()[::2])

        def fold_rule(rows: pd.DataFrame) -> np.ndarray:
            return (support.blip(half, rows[["need", "lives_alone"]].to_numpy()) > 0.0).astype(
                float
            )

        assert abs(truth - treatment_rule_value(fold_rule)) > 1e-4


# ------------------------------------------------------------------------------- W-a4


class TestWa4KnownProbabilitiesEnterTheCleverCovariate:
    """W-a4: the supplied treatment probabilities are the ``g`` of ``H``."""

    @staticmethod
    @cache
    def _fits() -> tuple[Any, Any, np.ndarray]:
        frame = law.draw("non_exceptional", 1000, 11)
        g1 = law.g0(frame["W1"].to_numpy(), frame["W2"].to_numpy())
        settings: dict[str, Any] = {
            "learned_rule": LearnedRule(),
            "outcome_learner": support.misspecified_learner(),
            "simultaneous": False,
            "random_state": 3,
            **IN_SAMPLE,
        }
        fit = {"outcome": "Y", "treatment": "A", "covariates": list(law.COVARIATES)}
        known = (
            TMLE(treatment_learner=law.treatment_learner(), **settings)
            .fit(frame, treatment_probabilities=np.column_stack([1.0 - g1, g1]), **fit)
            .single()
        )
        # The mutation: the same fit with an estimated g in place of the supplied one.
        estimated = (
            TMLE(treatment_learner=DummyClassifier(strategy="prior"), **settings)
            .fit(frame, **fit)
            .single()
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
        # The mutation: H at an estimated g leaves the score unsolved.
        g_hat = np.full_like(g1, treatment.mean())
        mutated = (treatment == pieces["rule"]) / np.where(treatment == 1.0, g_hat, 1.0 - g_hat)
        assert abs(float(np.mean(mutated * residual))) > W_A4_SCORE

    def test_the_mutation_moves_the_point(self) -> None:
        known, estimated, _ = self._fits()
        assert np.array_equal(law.rule_of(known), law.rule_of(estimated))
        move = known.estimates[support.NAME].psi - estimated.estimates[support.NAME].psi
        assert abs(move) > W_A4_POINT_MOVE, move

    def test_a_known_mechanism_keys_known(self) -> None:
        known, _, _ = self._fits()
        options = dict(known.config.support_key.options)
        assert options["treatment_learner_family"] == "known"
        assert "known_mechanism" in known.config.support_key.qualifiers


# ------------------------------------------------------------------------- the record


class TestTheRecord:
    """The record holds one pseudo-fold, whose value is the estimate."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return _fit()

    def test_one_pseudo_fold(self, result: Any) -> None:
        record = result.extra["learned_rule"]
        (split,) = record.splits
        psi = result.estimates[support.NAME].psi
        assert record.scheme == split.scheme == "in_sample"
        assert record.target == TARGET_KIND["in_sample"] == "full-sample data-adaptive value"
        assert split.fold_sizes == (result.data.n,)
        assert split.fold_weights == (1.0,)
        assert split.fold_estimates == pytest.approx((psi,), abs=1e-12)
        assert split.estimate == psi
        assert split.treated_shares == (float(np.mean(law.rule_of(result))),)

    def test_the_summary_names_the_full_sample_target(self, result: Any) -> None:
        line = result.extra["learned_rule"].describe()
        assert line.startswith("Target: the value of the one rule learned on every row")
        assert "a different target from the fold average" in line
        # The condition and the warning state the Donsker class; the target line does not.
        assert "Donsker" not in line

    def test_a_round_trip_keeps_the_record(self, result: Any) -> None:
        restored = loads(dumps(result))
        assert restored.extra["learned_rule"] == result.extra["learned_rule"]
        assert restored.extra["learned_rule"].scheme == "in_sample"

    def test_retarget_reproduces_the_fit(self, result: Any) -> None:
        estimates, *_ = result.estimator.retarget(
            result.data, result.nuisance, estimands=("ey_learned_rule",)
        )
        assert estimates[support.NAME].psi == result.estimates[support.NAME].psi


# -------------------------------------------------------------------- the compositions


class TestTheCompositions:
    """Weights, clusters and strata are admitted in sample, experimental (decision 7)."""

    def test_clusters_take_the_in_sample_cluster_variance_and_the_t_reference(self) -> None:
        frame = support.law_frame(400, 0).assign(c=lambda f: np.arange(len(f)) % 25)
        result = (
            support.learned_tmle(**IN_SAMPLE)
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, id="c")
            .single()
        )
        estimate = result.estimates[support.NAME]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        assert estimate.variance == pytest.approx(
            influence_variance(curve, result.data.cluster), rel=1e-12
        )
        assert estimate.variance != pytest.approx(influence_variance(curve), rel=1e-3)
        assert estimate.reference_df is not None
        assert result.config.support_key.dependence == "t"
        assert estimate.support.label == "experimental"

    def test_strata_read_the_one_rule_stratum_by_stratum(self) -> None:
        result = (
            support.learned_tmle(**IN_SAMPLE)
            .fit(
                support.law_frame(400, 0),
                outcome="Y",
                treatment="A",
                covariates=support.COVARIATES,
                strata=("W2",),
            )
            .single()
        )
        q = support.validation_pieces(result)["q_star_rule"]
        w2 = np.asarray(result.data.covariates, dtype=float)[:, 1]
        for level in (0.0, 1.0):
            name = f"{support.NAME}[W2={level}]"
            assert result.estimates[name].psi == pytest.approx(q[w2 == level].mean(), abs=1e-9)
            assert result.estimates[name].support.label == "experimental"
        assert result.estimates[support.NAME].psi == pytest.approx(q.mean(), abs=1e-9)

    def test_weights_run_experimental(self) -> None:
        frame = support.law_frame(400, 0).assign(w=lambda f: 0.5 + f["W2"])
        result = (
            support.learned_tmle(**IN_SAMPLE)
            .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, weights="w")
            .single()
        )
        assert result.config.support_key.weighting == "fixed"
        assert result.estimates[support.NAME].support.label == "experimental"


# ------------------------------------------------------------------- the label (option C)


class TestTheLabel:
    """Option C: the in-sample key reads the learner families; cross-fitted keys do not."""

    def test_an_adaptive_learner_keys_adaptive_and_warns(self) -> None:
        with pytest.warns(InSampleLearnerWarning, match="Donsker class") as caught:
            result = support.fit(**IN_SAMPLE)
        message = str(caught[0].message)
        assert message.startswith(WARNING)
        # Both spellings of the remedy: the engine's and the method's.
        assert (
            "Cross-fitting (cross_fit=True with n_folds of at least 2; "
            "CrossFitting(enabled=True) on the method) removes"
        ) in message
        assert "a different target" in message
        options = dict(result.config.support_key.options)
        assert options["outcome_learner_family"] == "adaptive"
        assert options["treatment_learner_family"] == "parametric"
        assert "rule_learner_family" not in options
        condition = in_sample_learner_condition(result.estimator, result.data)
        assert condition is not None and "outcome learner" in condition.condition
        assert condition in result.config.support_conditions
        assert result.estimates[support.NAME].support.label == "experimental"

    def test_parametric_learners_key_parametric_and_do_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = support.fit(outcome_learner=InteractionLogistic(), **IN_SAMPLE)
        options = dict(result.config.support_key.options)
        assert options["outcome_learner_family"] == "parametric"
        assert options["treatment_learner_family"] == "parametric"
        assert in_sample_learner_condition(result.estimator, result.data) is None
        # No in-sample row is registered until the learned-rule-in-sample study runs.
        assert result.estimates[support.NAME].support.label == "experimental"
        assert result.estimates[support.NAME].support.row is None

    def test_a_rule_learner_keys_its_family(self) -> None:
        rule = LearnedRule(loss="blip", learner=LinearRegression())
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = support.fit(
                learned_rule=rule, outcome_learner=InteractionLogistic(), **IN_SAMPLE
            )
        options = dict(result.config.support_key.options)
        assert options["rule_learner_family"] == "parametric"

    @staticmethod
    def _adaptive_role(role: str, **overrides: Any) -> Any:
        """Fit in sample with a parametric outcome learner and one adaptive ``role``."""
        with pytest.warns(InSampleLearnerWarning) as caught:
            result = support.fit(outcome_learner=InteractionLogistic(), **IN_SAMPLE, **overrides)
        (warning,) = [item for item in caught if item.category is InSampleLearnerWarning]
        message = str(warning.message)
        assert f"uses an adaptive {role}." in message, message
        assert "outcome learner" not in message, message
        condition = in_sample_learner_condition(result.estimator, result.data)
        assert condition is not None and role in condition.condition
        assert condition in result.config.support_conditions
        return result

    def test_an_adaptive_rule_learner_alone_warns(self) -> None:
        """Only the rule role can make this fit warn: its outcome learner is parametric."""
        rule = LearnedRule(loss="blip", learner=DecisionTreeRegressor(max_depth=2))
        result = self._adaptive_role("rule learner", learned_rule=rule)
        options = dict(result.config.support_key.options)
        assert options["outcome_learner_family"] == "parametric"
        assert options["treatment_learner_family"] == "parametric"
        assert options["rule_learner_family"] == "adaptive"

    def test_an_adaptive_treatment_learner_alone_warns(self) -> None:
        result = self._adaptive_role(
            "treatment learner", treatment_learner=DecisionTreeClassifier(max_depth=2)
        )
        options = dict(result.config.support_key.options)
        assert options["outcome_learner_family"] == "parametric"
        assert options["treatment_learner_family"] == "adaptive"

    def test_known_probabilities_beside_an_adaptive_treatment_learner_do_not_warn(self) -> None:
        """The fit does not learn ``g`` under ``treatment_probabilities=``: the key reads known."""
        frame = support.law_frame()
        g1 = expit(0.3 * frame["W1"] - 0.2 * frame["W2"]).to_numpy()
        estimator = support.learned_tmle(
            outcome_learner=InteractionLogistic(),
            treatment_learner=DecisionTreeClassifier(max_depth=2),
            **IN_SAMPLE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", InSampleLearnerWarning)
            result = estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=support.COVARIATES,
                treatment_probabilities=np.column_stack([1.0 - g1, g1]),
            ).single()
        options = dict(result.config.support_key.options)
        assert options["treatment_learner_family"] == "known"
        # The same learner without the supplied probabilities warns: the test above.
        assert in_sample_learner_condition(result.estimator, result.data) is None

    def test_the_cross_fitted_key_holds_no_family(self) -> None:
        for settings in ({"cv_evaluation": True}, {"cv_evaluation": False}):
            options = dict(support.fit(**settings).config.support_key.options)
            assert not {name for name in options if name.endswith("_learner_family")}
            assert in_sample_learner_condition(support.learned_tmle(**settings), None) is None

    def test_a_wave_1_study_fit_keeps_its_row(self) -> None:
        """R2: the T24 configuration (``learned-rule-cvtmle``, a Pipeline learner) reads T24."""
        frame = law.draw("non_exceptional", 2000, 5)
        result = law.fit(frame, law.outcome_learner(), 7)
        support_record = result.estimates[support.NAME].support
        assert support_record.row == "T24"
        assert support_record.label == "supported"

    def test_the_summary_prints_the_warning_and_no_generic_in_sample_note(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = support.fit(**IN_SAMPLE)
        text = str(result.summary())
        assert "Donsker class" in text
        assert "gives out-of-fold predictions" not in text
