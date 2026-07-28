"""Fold construction, covariate screening and the Super Learner."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from cleverly.learners import (
    CorrelationScreener,
    Folds,
    SuperLearner,
    fit_learner,
    infer_task,
    make_folds,
    predict_mean,
    resolve_library,
    resolve_n_folds,
    screen_by_correlation,
    supports_sample_weight,
)


@pytest.fixture
def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 4))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=n)
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-(x[:, 0] + 0.5 * x[:, 2])))).astype(float)
    return x, y, a


class TestFolds:
    def test_every_observation_is_held_out_exactly_once(self) -> None:
        folds = make_folds(200, 5, random_state=0)
        held_out = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(held_out), np.arange(200))

    def test_train_and_test_never_overlap(self) -> None:
        folds = make_folds(200, 4, random_state=0)
        for train, test in folds:
            assert set(train).isdisjoint(set(test))

    def test_stratification_keeps_both_arms_in_every_fold(self, sample) -> None:
        _, _, a = sample
        # A rare treatment arm can otherwise leave a fold with no treated units, and
        # a propensity model cannot be fit on a single-armed fold.
        rare = a.copy()
        rare[:] = 0.0
        rare[:12] = 1.0
        folds = make_folds(len(rare), 5, stratify=rare, random_state=0)
        for _, test in folds:
            assert rare[test].sum() >= 1

    def test_clusters_stay_intact(self) -> None:
        cluster = np.repeat(np.arange(40), 5)
        folds = make_folds(200, 5, cluster=cluster, random_state=0)
        for train, test in folds:
            assert set(cluster[train]).isdisjoint(set(cluster[test]))

    def test_clusters_and_stratification_combine(self) -> None:
        cluster = np.repeat(np.arange(40), 5)
        rng = np.random.default_rng(0)
        stratify = rng.binomial(1, 0.5, 200).astype(float)
        folds = make_folds(200, 4, stratify=stratify, cluster=cluster, random_state=1)
        for train, test in folds:
            assert set(cluster[train]).isdisjoint(set(cluster[test]))

    def test_the_split_is_reproducible(self) -> None:
        first = make_folds(150, 5, random_state=7).assignment
        second = make_folds(150, 5, random_state=7).assignment
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_splits(self) -> None:
        first = make_folds(150, 5, random_state=1).assignment
        second = make_folds(150, 5, random_state=2).assignment
        assert not np.array_equal(first, second)

    def test_single_fold_trains_and_predicts_on_everything(self) -> None:
        folds = Folds.single(50)
        assert folds.is_single
        train, test = next(iter(folds))
        assert len(test) == 50
        assert len(train) == 0

    def test_folds_are_capped_by_the_rarer_class(self) -> None:
        stratify = np.zeros(100)
        stratify[:3] = 1.0
        with pytest.warns(UserWarning, match="reducing n_folds"):
            assert resolve_n_folds(10, 100, stratify) == 3

    def test_too_few_folds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            resolve_n_folds(1, 100)

    def test_folds_are_capped_by_the_cluster_count(self) -> None:
        cluster = np.repeat(np.arange(3), 20)
        with pytest.warns(UserWarning, match="only 3 clusters"):
            folds = make_folds(60, 10, cluster=cluster, random_state=0)
        assert folds.n_folds == 3


class TestScreening:
    def test_associated_columns_are_kept_and_noise_dropped(self) -> None:
        rng = np.random.default_rng(0)
        n = 800
        x = rng.normal(size=(n, 5))
        a = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * x[:, 0]))).astype(float)
        keep = screen_by_correlation(x, a, threshold=0.01, min_retain=1)
        assert keep[0]
        assert keep.sum() < 5

    def test_min_retain_is_a_floor(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 6))
        a = rng.binomial(1, 0.5, 200).astype(float)
        # Nothing is associated, but screening must never empty the model.
        keep = screen_by_correlation(x, a, threshold=1e-12, min_retain=3)
        assert keep.sum() == 3

    def test_screener_works_inside_a_pipeline(self, sample) -> None:
        x, _, a = sample
        model = make_pipeline(CorrelationScreener(min_retain=2), LogisticRegression())
        model.fit(x, a)
        assert model.predict_proba(x).shape == (len(a), 2)

    def test_unfitted_screener_refuses_to_transform(self) -> None:
        with pytest.raises(AttributeError, match="not been fitted"):
            CorrelationScreener().get_support()


class TestFittingHelpers:
    def test_sample_weight_support_is_detected(self) -> None:
        assert supports_sample_weight(LinearRegression())
        assert supports_sample_weight(make_pipeline(LinearRegression()))

    def test_fitting_does_not_mutate_the_caller_estimator(self, sample) -> None:
        x, y, _ = sample
        template = LinearRegression()
        fitted = fit_learner(template, x, y)
        assert not hasattr(template, "coef_")
        assert hasattr(fitted, "coef_")

    def test_weights_route_through_a_pipeline(self, sample) -> None:
        x, y, _ = sample
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.5, 1.5, len(y))
        unweighted = fit_learner(make_pipeline(LinearRegression()), x, y)
        weighted = fit_learner(make_pipeline(LinearRegression()), x, y, weights)
        assert not np.allclose(unweighted[-1].coef_, weighted[-1].coef_, atol=1e-10)

    def test_a_learner_that_ignores_weights_warns_once(self, sample) -> None:
        class NoWeights(DummyRegressor):
            def fit(self, X, y):  # type: ignore[override]
                return super().fit(X, y)

        x, y, _ = sample
        with pytest.warns(UserWarning, match="does not accept sample_weight"):
            fit_learner(NoWeights(), x, y, np.ones(len(y)))

    def test_task_inference(self) -> None:
        assert infer_task(np.array([0.0, 1.0, 1.0])) == "classification"
        assert infer_task(np.array([0.0, 1.5, 2.0])) == "regression"

    def test_predict_mean_reads_the_positive_class_explicitly(self, sample) -> None:
        x, _, a = sample
        model = LogisticRegression().fit(x, a.astype(int))
        assert np.allclose(predict_mean(model, x, "classification"), model.predict_proba(x)[:, 1])

    def test_predict_mean_survives_a_single_class_fold(self, sample) -> None:
        from sklearn.dummy import DummyClassifier

        x, _, _ = sample
        # A fold can legitimately contain only one class when the outcome is rare. Such
        # a model returns a one-column predict_proba; predict_mean must read it as a
        # constant rather than indexing a column that is not there.
        model = DummyClassifier(strategy="prior").fit(x, np.ones(len(x), dtype=int))
        assert np.allclose(predict_mean(model, x, "classification"), 1.0)
        model_zero = DummyClassifier(strategy="prior").fit(x, np.zeros(len(x), dtype=int))
        assert np.allclose(predict_mean(model_zero, x, "classification"), 0.0)


class TestLibrary:
    @pytest.mark.parametrize("preset", ["glm", "fast", "default", "rich"])
    @pytest.mark.parametrize("task", ["regression", "classification"])
    def test_presets_resolve_to_named_estimators(self, preset: str, task: str) -> None:
        library = resolve_library(preset, task)  # type: ignore[arg-type]
        assert len(library) >= 2
        assert all(hasattr(estimator, "fit") for _, estimator in library)
        # SL.mean is always present: it is what stops the ensemble being dragged
        # around by a candidate that overfits.
        assert library[0][0] == "mean"

    def test_presets_grow_monotonically(self) -> None:
        sizes = [
            len(resolve_library(name, "regression")) for name in ("glm", "fast", "default", "rich")
        ]
        assert sizes == sorted(sizes)

    def test_bare_estimators_get_names(self) -> None:
        library = resolve_library([LinearRegression(), DummyRegressor()], "regression")
        assert [name for name, _ in library] == ["LinearRegression", "DummyRegressor"]

    def test_duplicate_names_are_disambiguated(self) -> None:
        library = resolve_library([LinearRegression(), LinearRegression()], "regression")
        assert [name for name, _ in library] == ["LinearRegression", "LinearRegression_2"]

    def test_explicit_pairs_pass_through(self) -> None:
        library = resolve_library([("mine", LinearRegression())], "regression")
        assert library[0][0] == "mine"

    def test_spline_learner_is_skipped_for_wide_data(self) -> None:
        names = [name for name, _ in resolve_library("default", "regression", n_features=200)]
        assert "gam" not in names

    def test_unknown_preset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown library preset"):
            resolve_library("magic", "regression")

    def test_empty_library_is_refused(self) -> None:
        with pytest.raises(ValueError, match="library is empty"):
            resolve_library([], "regression")


class TestSuperLearner:
    def test_weights_lie_on_the_simplex(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="fast", n_folds=3, random_state=0).fit(x, y)
        assert model.coef_.sum() == pytest.approx(1.0)
        assert np.all(model.coef_ >= -1e-12)

    def test_the_ensemble_beats_or_matches_its_worst_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="fast", n_folds=3, random_state=0).fit(x, y)
        assert model.diagnostics_.ensemble_cv_risk <= model.cv_risk_.max()

    def test_discrete_super_learner_picks_a_single_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(
            library="fast", meta_learner="discrete", n_folds=3, random_state=0
        ).fit(x, y)
        assert np.count_nonzero(model.coef_) == 1
        assert model.learner_names_[int(np.argmax(model.coef_))] == model.diagnostics_.best

    def test_discrete_selection_follows_the_cross_validated_risk(self, sample) -> None:
        x, y, _ = sample
        # A deliberately useless candidate alongside a good one: the discrete Super
        # Learner must not choose the useless one.
        library = [
            ("useless", DummyRegressor(strategy="constant", constant=1e6)),
            ("ols", LinearRegression()),
        ]
        model = SuperLearner(
            library=library, meta_learner="discrete", n_folds=3, random_state=0
        ).fit(x, y)
        assert model.diagnostics_.best == "ols"

    def test_classification_probabilities_stay_in_range(self, sample) -> None:
        x, _, a = sample
        model = SuperLearner(library="fast", n_folds=3, random_state=0, clip=(0.001, 0.999)).fit(
            x, a
        )
        p = model.predict(x)
        assert p.min() >= 0.001
        assert p.max() <= 0.999
        assert model.predict_proba(x).shape == (len(a), 2)
        assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)

    def test_nnloglik_is_the_default_for_a_binary_target(self, sample) -> None:
        x, _, a = sample
        model = SuperLearner(library="glm", n_folds=3, random_state=0).fit(x, a)
        assert model.diagnostics_.loss == "neg_log_likelihood"

    def test_out_of_fold_predictions_are_retained_for_every_row(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="glm", n_folds=4, random_state=0).fit(x, y)
        assert model.cv_predictions_.shape == (len(y), len(model.learner_names_))
        assert np.all(np.isfinite(model.cv_predictions_))

    def test_a_failing_candidate_is_dropped_not_fatal(self, sample) -> None:
        class Exploding(LinearRegression):
            def fit(self, X, y, sample_weight=None):  # type: ignore[override]
                raise RuntimeError("boom")

        x, y, _ = sample
        with pytest.warns(UserWarning, match="dropping learner"):
            model = SuperLearner(
                library=[("bad", Exploding()), ("ols", LinearRegression())],
                n_folds=3,
                random_state=0,
            ).fit(x, y)
        assert model.learner_names_ == ("ols",)

    def test_all_candidates_failing_raises(self, sample) -> None:
        class Exploding(LinearRegression):
            def fit(self, X, y, sample_weight=None):  # type: ignore[override]
                raise RuntimeError("boom")

        x, y, _ = sample
        with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="every learner"):
            SuperLearner(library=[("bad", Exploding())], n_folds=3).fit(x, y)

    def test_predicting_before_fitting_is_refused(self, sample) -> None:
        from cleverly.exceptions import NotFittedError

        x, _, _ = sample
        with pytest.raises(NotFittedError, match="has not been fitted"):
            SuperLearner().predict(x)

    def test_clusters_are_respected_by_the_inner_folds(self, sample) -> None:
        x, y, _ = sample
        groups = np.repeat(np.arange(len(y) // 10), 10)
        model = SuperLearner(library="glm", n_folds=4, random_state=0).fit(x, y, groups=groups)
        for train, test in model.folds_:
            assert set(groups[train]).isdisjoint(set(groups[test]))

    def test_weights_change_the_fit(self, sample) -> None:
        x, y, _ = sample
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.2, 2.0, len(y))
        plain = SuperLearner(library="glm", n_folds=3, random_state=0).fit(x, y)
        weighted = SuperLearner(library="glm", n_folds=3, random_state=0).fit(
            x, y, sample_weight=weights
        )
        assert not np.allclose(plain.predict(x), weighted.predict(x))

    def test_weights_mapping_matches_the_coefficients(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="glm", n_folds=3, random_state=0).fit(x, y)
        assert model.weights == dict(zip(model.learner_names_, model.coef_.tolist(), strict=True))

    def test_mismatched_lengths_are_refused(self, sample) -> None:
        x, y, _ = sample
        with pytest.raises(ValueError, match="expected"):
            SuperLearner(library="glm", n_folds=3).fit(x, y[:-1])
