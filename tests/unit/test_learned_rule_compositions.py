"""The learned-rule value with fold targeting, observation weights and clusters (X11 (h), (k), (l)).

Each composition reuses a construction that the fold-evaluated engine already ships for arm and
regime targets. The witnesses below check what the learned-rule path adds, and each one fails
when its component is wrong:

* fold targeting (h): every fold's own score is solved, which the pooled scheme leaves unsolved
  on the same draw; and a fit whose folds all learn the rule ``1{W1 > 0}`` equals the
  fold-targeted fit of that known rule.
* weights (k): the rule is the plug-in of the weighted outcome regression. On the
  ``misspecified_limit`` law the weighted and the unweighted rules have different limits, and the
  fit's rule is the weighted one. Fold-local normalisation of the weights is checked with the
  nuisances held fixed.
* clusters (l): folds move whole clusters, and the clustered variance exceeds the row variance
  when the outcome carries a cluster effect. Clusters of one row reproduce the row fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly import _support
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import LearnedRule, Rule
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

FIT = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}


def _scores_by_fold(result: Any) -> np.ndarray:
    """The targeted score ``sum_{i in v} H_i (Y_i - Qbar*(A_i, W_i))`` of each fold."""
    pieces = support.validation_pieces(result)
    residual = pieces["h"] * (pieces["y"] - pieces["q_star_observed"])
    return np.array([float(np.sum(residual[test])) for test in pieces["tests"]])


class TestFoldTargeting:
    """X11 (h): one fluctuation inside each validation fold."""

    def test_every_fold_score_is_solved_and_the_pooled_scheme_leaves_one_unsolved(self) -> None:
        frame = support.law_frame(203, 1)
        learner = support.misspecified_learner()
        fold = support.fit(frame, outcome_learner=learner, targeting_scheme="fold")
        pooled = support.fit(frame, outcome_learner=support.misspecified_learner())
        assert np.max(np.abs(_scores_by_fold(fold))) < 1e-6
        # The nonzero witness: one pooled epsilon solves only the weighted sum of the scores.
        assert np.max(np.abs(_scores_by_fold(pooled))) > 1e-2
        assert fold[support.NAME].psi != pytest.approx(pooled[support.NAME].psi, abs=1e-6)

    def test_a_common_rule_equals_the_fold_targeted_known_rule(self) -> None:
        """Every fold learns ``1{W1 > 0}``, so the fit is the fold-targeted fit of that rule."""
        frame = support.law_frame(400, 2)
        settings = {
            "outcome_learner": SignOfW1(),
            "treatment_learner": support.treatment_learner(),
            "n_folds": 5,
            "simultaneous": False,
            "random_state": 0,
            "targeting_scheme": "fold",
            **support.FOLD_EVALUATED,
        }
        learned = TMLE(learned_rule=LearnedRule(), **settings).fit(frame, **FIT).single()
        rule = Rule(lambda w: (w["W1"] > 0).astype(int), name="w1", rule_kind="known")
        known = TMLE(interventions=[rule], **settings).fit(frame, **FIT).single()
        assert [list(t) for _, t in learned.nuisance.folds] == [
            list(t) for _, t in known.nuisance.folds
        ]
        treat = np.asarray(learned.nuisance.regimes.values[:, 1, 0]) > 0.5
        np.testing.assert_array_equal(treat, frame["W1"].to_numpy() > 0)
        assert 0.2 < treat.mean() < 0.8
        value, regime = learned[support.NAME], known["ey_regime[w1]"]
        assert value.psi == pytest.approx(regime.psi, rel=1e-10)
        assert value.std_error == pytest.approx(regime.std_error, rel=1e-10)


class SignOfW1(ClassifierMixin, BaseEstimator):
    """A stub outcome model whose blip has the sign of ``W1`` exactly, in every fold.

    ``P(Y = 1 | A, W) = expit(0.2 + 0.3 W2 + A * 2 W1)``, whatever the training rows. The fit
    only records the classes, so every fold learns ``1{W1 > 0}``.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SignOfW1:
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        p = expit(0.2 + 0.3 * design[:, 2] + design[:, 0] * 2.0 * design[:, 1])
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------- weights (k)


def _misspecified_limit_frame(n: int, seed: int) -> pd.DataFrame:
    """The ``misspecified_limit`` law of the learned-rule studies, with the weight column ``w``.

    The blip is ``0.8 W1 + W1^2 - 0.3`` on the logit scale. The main-terms learner of
    :func:`support.outcome_learner` has no ``W1^2``, so its limit depends on the weights.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    blip = 0.8 * w1 + w1**2 - 0.3
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 - 0.3 * w2 + a * blip)).astype(float)
    weight = 1.0 + 1.5 * (w1 > 0.0)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "w": weight})


def _rule_of(frame: pd.DataFrame, weights: np.ndarray | None) -> np.ndarray:
    """The plug-in rule of the main-terms learner fitted on all of ``frame``."""
    design = frame[["A", "W1", "W2"]].to_numpy(dtype=float)
    model = support.outcome_learner().fit(design, frame["Y"].to_numpy(), fit__sample_weight=weights)
    return support.blip(model, frame[["W1", "W2"]].to_numpy(dtype=float)) > 0.0


#: The population shares of the ``misspecified_limit`` law, declared from :func:`_projection` at
#: 4,001 points of ``W1``: the rule of the weighted projection treats 48.15% of the covariate
#: law, the unweighted one 49.84%, and the two disagree on 1.69%.
WEIGHTED_TREATED_SHARE = 0.4815
UNWEIGHTED_TREATED_SHARE = 0.4984
DECLARED_DISAGREEMENT = 0.0169

#: How far a 400,000-row fit may sit from a declared population share. It is less than half
#: the disagreement, so a rule learned without the weights cannot pass.
SHARE_TOLERANCE = 0.0075


def _projection(weighted: bool, points: int = 4_001) -> tuple[np.ndarray, np.ndarray]:
    """The population rule of the learner's model on ``misspecified_limit``, by quadrature.

    Each grid point of ``(W1, W2)`` carries its trapezoid mass, each arm its propensity and each
    outcome its probability, so the weighted log-likelihood is the population one. Returns the
    rule and the covariate mass on the grid.
    """
    w1 = np.linspace(-1.0, 1.0, points)
    trapezoid = np.full(points, 2.0 / (points - 1))
    trapezoid[[0, -1]] /= 2.0
    mass = 0.25 * trapezoid
    design, sample_weight, outcome = [], [], []
    for w2 in (0.0, 1.0):
        g = expit(0.3 * w1 - 0.2 * w2)
        weight = 1.0 + 1.5 * (w1 > 0.0) if weighted else np.ones(points)
        for a in (0.0, 1.0):
            q = expit(0.2 + 0.5 * w1 - 0.3 * w2 + a * (0.8 * w1 + w1**2 - 0.3))
            for y, p_y in ((1.0, q), (0.0, 1.0 - q)):
                design.append(np.column_stack([np.full(points, a), w1, np.full(points, w2)]))
                sample_weight.append(mass * (g if a else 1.0 - g) * p_y * weight)
                outcome.append(np.full(points, y))
    model = support.outcome_learner().set_params(fit__C=1e10, fit__max_iter=10_000, fit__tol=1e-12)
    weights = np.concatenate(sample_weight)
    model.fit(
        np.vstack(design), np.concatenate(outcome), fit__sample_weight=weights / weights.mean()
    )
    rule = np.concatenate(
        [support.blip(model, np.column_stack([w1, np.full(points, w2)])) > 0.0 for w2 in (0.0, 1.0)]
    )
    return rule, np.concatenate([mass, mass])


class TestWeights:
    """X11 (k): known observation weights; the rule reads them."""

    @pytest.fixture(scope="class")
    def large(self) -> pd.DataFrame:
        return _misspecified_limit_frame(400_000, 11)

    def test_the_population_projections_give_the_declared_shares(self) -> None:
        """The nonzero witness, W-k1: the weighted and the unweighted limit rules differ."""
        weighted, mass = _projection(True)
        unweighted, _ = _projection(False)
        assert float(np.sum(mass * weighted)) == pytest.approx(WEIGHTED_TREATED_SHARE, abs=5e-4)
        assert float(np.sum(mass * unweighted)) == pytest.approx(UNWEIGHTED_TREATED_SHARE, abs=5e-4)
        disagreement = float(np.sum(mass * (weighted != unweighted)))
        assert disagreement == pytest.approx(DECLARED_DISAGREEMENT, abs=5e-4)

    def test_the_sample_projections_disagree_on_the_declared_share(
        self, large: pd.DataFrame
    ) -> None:
        weighted = _rule_of(large, large["w"].to_numpy())
        unweighted = _rule_of(large, None)
        disagreement = float(np.mean(weighted != unweighted))
        assert disagreement == pytest.approx(DECLARED_DISAGREEMENT, abs=SHARE_TOLERANCE)

    def test_the_fit_treats_the_weighted_share_at_400k_rows(self, large: pd.DataFrame) -> None:
        """W-k5: the fit's rules treat the weighted limit's share; the unweighted rule does not."""
        result = support.learned_tmle().fit(large, weights="w", **FIT).single()
        treated = float(np.mean(np.asarray(result.nuisance.regimes.values[:, 1, 0]) > 0.5))
        assert treated == pytest.approx(WEIGHTED_TREATED_SHARE, abs=SHARE_TOLERANCE)
        # The mutation, a rule learned without the weights, misses the weighted share.
        unweighted = float(np.mean(_rule_of(large, None)))
        assert abs(unweighted - WEIGHTED_TREATED_SHARE) > SHARE_TOLERANCE

    def test_the_fit_rule_is_the_weighted_rule_of_each_training_fold(self) -> None:
        frame = _misspecified_limit_frame(2_000, 3)
        result = support.learned_tmle().fit(frame, weights="w", **FIT).single()
        treat = np.asarray(result.nuisance.regimes.values[:, 1, 0]) > 0.5
        # The fit reads the weights normalised to mean one, as CausalData stores them.
        weights = np.asarray(result.data.weights, dtype=float)
        differs = 0
        for train, test in result.nuisance.folds:
            rows = frame.iloc[np.asarray(train)]
            design = rows[["A", "W1", "W2"]].to_numpy(dtype=float)
            covariates = frame.iloc[np.asarray(test)][["W1", "W2"]].to_numpy(dtype=float)
            weighted = support.outcome_learner().fit(
                design, rows["Y"].to_numpy(), fit__sample_weight=weights[np.asarray(train)]
            )
            unweighted = support.outcome_learner().fit(design, rows["Y"].to_numpy())
            np.testing.assert_array_equal(
                treat[np.asarray(test)], support.blip(weighted, covariates) > 0.0
            )
            differs += int(np.sum((support.blip(unweighted, covariates) > 0.0) != treat[test]))
        # The mutation, a rule learned without the weights, moves some rows on this draw.
        assert differs > 0

    def test_the_weighted_point_and_se_follow_the_weighted_fold_curve(self) -> None:
        """Longhand: weighted fold plug-ins and the curve ``(w / mean_v w) D`` rebuild the fit.

        Each fold's estimate is the weighted mean of ``Qbar*(d, W)`` over its validation rows,
        and its curve term scales the unweighted curve by the weight relative to the fold's mean
        weight. A curve that dropped the weight, or normalised it over all rows, gives another SE.
        """
        frame = _misspecified_limit_frame(1_000, 8)
        result = (
            support.learned_tmle(outcome_learner=support.misspecified_learner())
            .fit(frame, weights="w", **FIT)
            .single()
        )
        pieces = support.validation_pieces(result)
        weights = np.asarray(result.data.weights, dtype=float)
        tests = pieces["tests"]
        means, variance = [], 0.0
        for test in tests:
            w = weights[test] / np.mean(weights[test])
            psi_v = float(np.sum(w * pieces["q_star_rule"][test]) / test.size)
            core = pieces["h"][test] * (pieces["y"][test] - pieces["q_star_observed"][test])
            curve = w * (core + pieces["q_star_rule"][test] - psi_v)
            means.append(psi_v)
            variance += float(np.sum(curve**2)) / test.size**2
        variance /= len(tests) ** 2
        value = result[support.NAME]
        assert value.psi == pytest.approx(float(np.mean(means)), rel=1e-10)
        assert value.std_error == pytest.approx(np.sqrt(variance), rel=1e-8)
        # The weights vary, so the weightless curve is a different number.
        weightless = np.sqrt(
            sum(
                float(
                    np.sum(
                        (
                            pieces["h"][t] * (pieces["y"][t] - pieces["q_star_observed"][t])
                            + pieces["q_star_rule"][t]
                            - np.mean(pieces["q_star_rule"][t])
                        )
                        ** 2
                    )
                )
                / t.size**2
                for t in tests
            )
            / len(tests) ** 2
        )
        assert value.std_error != pytest.approx(weightless, rel=1e-3)

    def test_constant_weights_give_the_unweighted_fit(self) -> None:
        """A smoke test: CausalData normalises a constant weight away."""
        frame = support.law_frame(300, 4).assign(w=3.0)
        weighted = support.fit(frame.drop(columns="w"))
        result = support.learned_tmle().fit(frame, weights="w", **FIT).single()
        assert result[support.NAME].psi == pytest.approx(weighted[support.NAME].psi, rel=1e-12)
        assert result[support.NAME].std_error == pytest.approx(
            weighted[support.NAME].std_error, rel=1e-12
        )

    def test_estimated_weights_refuse_before_any_learner(self) -> None:
        frame = support.law_frame(120, 3).assign(w=lambda f: 1.0 + 0.5 * f["W2"])
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            support.learned_tmle(**never_fit_learners()).fit(
                frame, weights="w", weights_estimated=True, **FIT
            )
        assert NeverFit.calls == 0
        assert "X11 (k2)" in str(caught.value)


class TestFoldLocalWeightNormalisation:
    """Weights constant within each fold and different across folds normalise away.

    The fold-evaluated report normalises observation weights inside each validation fold. With
    the nuisances held fixed (a learner that ignores its weights and its rows), weights that are
    constant inside each fold give the unweighted point and SE. A global normalisation would not:
    it would reweight the folds against each other.
    """

    def test_fold_constant_weights_reproduce_the_unweighted_fit(self) -> None:
        frame = support.law_frame(400, 5)
        settings = {
            "outcome_learner": FixedOutcome(),
            "treatment_learner": FixedTreatment(),
        }
        plain = support.fit(frame, **settings)
        weights = np.ones(len(frame))
        for index, (_, test) in enumerate(plain.nuisance.folds):
            weights[np.asarray(test)] = 1.0 + 0.5 * index
        result = (
            support.learned_tmle(**settings)
            .fit(frame.assign(w=weights), weights="w", **FIT)
            .single()
        )
        assert [list(t) for _, t in result.nuisance.folds] == [
            list(t) for _, t in plain.nuisance.folds
        ]
        assert result[support.NAME].psi == pytest.approx(plain[support.NAME].psi, rel=1e-9)
        assert result[support.NAME].std_error == pytest.approx(
            plain[support.NAME].std_error, rel=1e-9
        )
        # The mutation: the same weights without the fold-local normalisation move the point.
        pieces = support.validation_pieces(plain)
        fold_means = np.array(
            [float(np.mean(pieces["q_star_rule"][test])) for test in pieces["tests"]]
        )
        fold_weights = np.array([weights[test].sum() for test in pieces["tests"]])
        assert float(np.dot(fold_means, fold_weights) / fold_weights.sum()) != pytest.approx(
            plain[support.NAME].psi, abs=1e-4
        )


class FixedOutcome(ClassifierMixin, BaseEstimator):
    """An outcome model that ignores its rows and weights: ``expit(0.1 + 0.4 W1 + A (W1 + 0.1))``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FixedOutcome:
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        p = expit(0.1 + 0.4 * design[:, 1] + design[:, 0] * (design[:, 1] + 0.1))
        return np.column_stack([1.0 - p, p])


class FixedTreatment(ClassifierMixin, BaseEstimator):
    """A treatment model that ignores its rows and weights: ``expit(0.3 W1 - 0.2 W2)``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FixedTreatment:
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        p = expit(0.3 * design[:, 0] - 0.2 * design[:, 1])
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------- clusters (l)


def _clustered_frame(clusters: int, size: int, seed: int, effect: float) -> pd.DataFrame:
    """The ``non_exceptional`` law with an outcome random effect shared inside each cluster."""
    rng = np.random.default_rng(seed)
    n = clusters * size
    cluster = np.repeat(np.arange(clusters), size)
    u = rng.normal(0.0, effect, clusters)[cluster]
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 - 0.3 * w2 + a * (0.1 + w1) + u)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "cluster": cluster})


class TestClusters:
    """X11 (l): the cluster is the unit."""

    def test_folds_move_whole_clusters(self) -> None:
        frame = _clustered_frame(60, 8, 1, 1.5)
        result = support.learned_tmle().fit(frame, id="cluster", **FIT).single()
        clusters = frame["cluster"].to_numpy()
        for train, test in result.nuisance.folds:
            assert not set(clusters[np.asarray(train)]) & set(clusters[np.asarray(test)])

    def test_the_clustered_variance_exceeds_the_row_variance(self) -> None:
        """The nonzero witness: a strong cluster effect inflates the variance.

        This is the ``clustered`` law of the learned-rule-compositions study, whose
        within-cluster correlation of the row curve is 0.144 by quadrature
        (``test_learned_rule_compositions_design.py``), so the clustered SE should be about
        ``sqrt(1 + 9 * 0.144) = 1.51`` times the row SE. With 100 clusters the relative SD of the
        clustered SE is near ``sqrt(1 / (2 * 100)) = 0.07``, so the floor 1.2 sits more than four
        SDs below the expected ratio and well above 1. This draw gives 1.44.
        """
        frame = _clustered_frame(100, 10, 2, 1.5)
        clustered = support.learned_tmle().fit(frame, id="cluster", **FIT).single()
        rows = support.learned_tmle().fit(frame.drop(columns="cluster"), **FIT).single()
        assert clustered[support.NAME].std_error > 1.2 * rows[support.NAME].std_error

    def test_clusters_of_one_row_reproduce_the_row_point(self) -> None:
        """With the same split, the point is the row point, and the SE follows the stated rule.

        The clustered fold term is the centred variance of the fold's cluster totals, with
        divisor ``J_v - 1``. With one row per cluster it is the row form, centred inside the
        fold and scaled by ``J_v / (J_v - 1)``.
        """
        frame = support.law_frame(300, 6).assign(cluster=np.arange(300))
        clustered = support.learned_tmle().fit(frame, id="cluster", **FIT).single()
        plan = [np.asarray(t) for _, t in clustered.nuisance.folds]
        pieces = support.validation_pieces(clustered)
        q_rule, q_obs = pieces["q_star_rule"], pieces["q_star_observed"]
        curve = pieces["h"] * (pieces["y"] - q_obs) + q_rule
        variance = 0.0
        for test in plan:
            values = curve[test] - np.mean(q_rule[test])
            size = test.size
            variance += size * np.var(values, ddof=1) / size**2
        variance /= len(plan) ** 2
        # A binary outcome is fitted on its own scale, so no unscaling enters.
        assert clustered.nuisance.scaler.range == 1.0
        fold_means = [float(np.mean(q_rule[test])) for test in plan]
        assert clustered[support.NAME].psi == pytest.approx(np.mean(fold_means), rel=1e-10)
        assert clustered[support.NAME].std_error == pytest.approx(np.sqrt(variance), rel=1e-8)
        # The mutation: the uncentred row form of the fold term gives another SE.
        rows = support.learned_tmle().fit(frame.drop(columns="cluster"), **FIT).single()
        assert rows[support.NAME].psi == pytest.approx(clustered[support.NAME].psi, rel=1e-10)
        assert rows[support.NAME].std_error != pytest.approx(
            clustered[support.NAME].std_error, rel=1e-6
        )


class TestTheLabelAndTheReference:
    """Until the learned-rule-compositions study registers, every lifted fit is experimental.

    When the study registers its rows, the label test is meant to fail and be inverted for the
    covered settings.
    """

    def test_each_lifted_setting_reads_experimental(self) -> None:
        fits = {
            "fold targeting": support.fit(support.law_frame(300, 1), targeting_scheme="fold"),
            "weights": support.learned_tmle()
            .fit(_misspecified_limit_frame(300, 1), weights="w", **FIT)
            .single(),
            "clusters": support.learned_tmle()
            .fit(_clustered_frame(60, 5, 1, 1.5), id="cluster", **FIT)
            .single(),
        }
        for setting, result in fits.items():
            label = result[support.NAME].support
            assert label.label == "experimental", setting
            assert label.row is None, setting
        # The weighted fit is near the learned-rule row and differs from it on the weights.
        assert "T24" in fits["weights"][support.NAME].support.reason
        assert "weighting" in fits["weights"][support.NAME].support.reason

    def test_unequal_cluster_sizes_carry_their_own_qualifier(self) -> None:
        """A learned-rule row for equal sizes never covers a fit with unequal sizes."""
        frame = _clustered_frame(60, 5, 1, 1.5)
        equal = support.learned_tmle().fit(frame, id="cluster", **FIT).single()
        unequal_ids = frame["cluster"].where(frame.index % 5 != 0, frame["cluster"] + 1)
        unequal = (
            support.learned_tmle()
            .fit(frame.assign(cluster=unequal_ids), id="cluster", **FIT)
            .single()
        )
        keys = [
            _support.point_key(support.learned_tmle(), result.data, (support.NAME,))[0]
            for result in (equal, unequal)
        ]
        assert "unequal_clusters" not in keys[0].qualifiers
        assert "unequal_clusters" in keys[1].qualifiers
        # An arm fit with the same unequal clusters keeps the shipped key.
        arm, _ = _support.point_key(
            TMLE(cross_fit=True, cv_evaluation=True), unequal.data, ("ate",)
        )
        assert "unequal_clusters" not in arm.qualifiers

    def test_few_clusters_take_the_t_reference(self) -> None:
        """J = 30 clusters in V = 5 folds: ``df = min(J - 2, J - V) = 25``."""
        result = (
            support.learned_tmle()
            .fit(_clustered_frame(30, 10, 1, 1.5), id="cluster", **FIT)
            .single()
        )
        assert result[support.NAME].reference_df == 25


class TestRefusalsThatStay:
    """Compositions that PR A does not lift keep their refusals."""

    def test_strata_with_fold_targeting_meets_the_learned_rule_strata_row(self) -> None:
        frame = support.law_frame(120, 3)
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            support.learned_tmle(**never_fit_learners(), targeting_scheme="fold").fit(
                frame, strata=("W2",), **FIT
            )
        assert NeverFit.calls == 0
        assert "X11 (m)" in str(caught.value)

    def test_strata_with_fold_targeting_still_refuses_on_an_arm_fit(self) -> None:
        frame = support.law_frame(120, 3)
        with pytest.raises(CapabilityError) as caught:
            TMLE(**never_fit_learners(), cross_fit=True, n_folds=5, targeting_scheme="fold").fit(
                frame, strata=("W2",), **FIT
            )
        assert "X61" in str(caught.value)
