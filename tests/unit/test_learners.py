"""Fold construction, covariate screening and the Super Learner."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from cleverly.exceptions import DataError
from cleverly.learners import (
    CorrelationScreener,
    CrossFitPlan,
    Folds,
    SuperLearner,
    check_integrity,
    fit_learner,
    infer_task,
    make_folds,
    predict_mean,
    refuse_scheme,
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


class TestFoldInvariants:
    """The prohibitions a cross-fitted estimate assumes, checked rather than assumed.

    Split the way the checks are: what an assignment alone can be wrong about is refused
    by ``Folds`` itself, and what needs the cluster vector beside it by
    ``check_integrity``.
    """

    def test_a_fold_index_outside_the_declared_range_is_refused(self) -> None:
        with pytest.raises(DataError, match=r"outside \[0, 3\)"):
            Folds(np.array([0, 1, 3, 2]), 3)

    def test_a_negative_fold_index_is_refused(self) -> None:
        with pytest.raises(DataError, match=r"outside \[0, 2\)"):
            Folds(np.array([0, -1, 1]), 2)

    def test_an_empty_fold_is_refused(self) -> None:
        # Fold 1 holds nothing, so it can produce no out-of-fold predictions and the
        # rows it was supposed to cover would silently never be predicted.
        with pytest.raises(DataError, match=r"fold\(s\) \[1\] hold no rows"):
            Folds(np.array([0, 0, 2, 2]), 3)

    def test_an_empty_assignment_is_refused(self) -> None:
        with pytest.raises(DataError, match="nothing to cross-fit"):
            Folds(np.array([], dtype=np.int64), 2)

    def test_the_degenerate_single_partition_is_still_allowed(self) -> None:
        # Folds.single is the cross_fit=False path: one fold, empty training set, by
        # design. The checks must not refuse a supported estimator.
        folds = Folds.single(20)
        assert folds.is_single
        check_integrity(folds, cluster=np.repeat(np.arange(4), 5))

    def test_a_cluster_spread_over_two_folds_is_refused(self) -> None:
        # Hand-built rather than produced by make_folds, because make_folds cannot
        # build this -- which is the point: the check exists for the Folds that
        # make_folds never saw, rehydrated from disk or assembled by a caller.
        cluster = np.repeat(np.arange(4), 5)
        assignment = np.repeat(np.arange(4), 5)
        assignment[0] = 3  # cluster 0 now straddles folds 3 and 0
        with pytest.raises(DataError, match="cluster code 0, spread over 2 folds"):
            check_integrity(Folds(assignment, 4), cluster=cluster)

    def test_an_unclustered_split_passes_trivially(self) -> None:
        check_integrity(make_folds(100, 5, random_state=0), cluster=None)

    def test_a_mismatched_cluster_length_is_refused(self) -> None:
        with pytest.raises(DataError, match="but the fold assignment has"):
            check_integrity(make_folds(100, 5, random_state=0), cluster=np.arange(50))

    def test_every_split_make_folds_builds_passes_its_own_check(self) -> None:
        cluster = np.repeat(np.arange(40), 5)
        rng = np.random.default_rng(0)
        stratify = rng.binomial(1, 0.3, 200).astype(float)
        for kwargs in (
            {},
            {"stratify": stratify},
            {"cluster": cluster},
            {"stratify": stratify, "cluster": cluster},
        ):
            folds = make_folds(200, 5, random_state=0, **kwargs)
            check_integrity(folds, cluster=kwargs.get("cluster"))

    def test_the_pre_1_6_grouped_fallback_still_keeps_clusters_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the path taken on scikit-learn before ``GroupKFold`` gained ``shuffle``.

        ``pyproject.toml`` declares ``scikit-learn>=1.3``, so this fallback is supported
        rather than vestigial -- and on a modern install it is never reached, which is
        exactly why it needs forcing. What it gives up is shuffling strength; what it
        must not give up is cluster integrity.
        """
        import sklearn.model_selection

        from cleverly.learners import crossfit

        real = sklearn.model_selection.GroupKFold

        class NoShuffle(real):  # type: ignore[misc, valid-type]
            def __init__(self, n_splits: int = 5, **kwargs: object) -> None:
                if "shuffle" in kwargs:
                    raise TypeError("__init__() got an unexpected keyword 'shuffle'")
                super().__init__(n_splits=n_splits)

        monkeypatch.setattr(crossfit, "GroupKFold", NoShuffle)
        cluster = np.repeat(np.arange(40), 5)
        folds = make_folds(200, 5, cluster=cluster, random_state=0)
        # The post-condition inside make_folds already ran; assert it directly too, so
        # this test fails on the claim rather than on an exception from elsewhere.
        check_integrity(folds, cluster=cluster)
        for train, test in folds:
            assert set(cluster[train]).isdisjoint(set(cluster[test]))


class TestRefusedSchemes:
    """Four schemes, four different reasons -- which is why they are named separately."""

    def test_blocked_temporal_names_the_missing_ordering(self) -> None:
        with pytest.raises(NotImplementedError, match="no node carries a time index"):
            refuse_scheme("blocked")

    def test_rolling_origin_names_the_storage_contract_not_the_time_index(self) -> None:
        # The distinction that matters: a rolling origin would still be refused after a
        # time index arrived, because one out-of-fold prediction per row is what
        # NuisanceEstimates is built on.
        with pytest.raises(NotImplementedError, match="different storage contract"):
            refuse_scheme("rolling_origin")

    def test_repeated_names_the_plumbing_rather_than_the_derivation(self) -> None:
        with pytest.raises(NotImplementedError, match="aggregation is not the obstacle"):
            refuse_scheme("repeated")

    def test_splitting_a_cluster_to_buy_more_folds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reduce n_folds, not to unfuse them"):
            refuse_scheme("row_within_cluster")

    def test_an_unknown_scheme_says_so(self) -> None:
        with pytest.raises(ValueError, match="unknown cross-fitting scheme"):
            refuse_scheme("nonsense")


class TestCrossFitPlan:
    """The policy a fit declared, as distinct from the split it realised."""

    def test_a_plan_is_only_numbers_and_so_compares_by_value(self) -> None:
        assert CrossFitPlan(n_folds=5) == CrossFitPlan(n_folds=5)
        assert CrossFitPlan(n_folds=5) != CrossFitPlan(n_folds=10)

    def test_one_fold_means_no_cross_fitting(self) -> None:
        assert not CrossFitPlan(n_folds=1).cross_fit
        assert CrossFitPlan(n_folds=2).cross_fit

    def test_describe_names_what_was_declared(self) -> None:
        plan = CrossFitPlan(n_folds=5, scheme="grouped", stratify_by=("A",))
        assert plan.describe() == "declared: 5-fold grouped stratified on A"
        assert "no cross-fitting" in CrossFitPlan(n_folds=1).describe()


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
        model = SuperLearner(library="glm", n_folds=3, random_state=0).fit(x, y)
        assert model.coef_.sum() == pytest.approx(1.0)
        assert np.all(model.coef_ >= -1e-12)

    def test_the_ensemble_beats_or_matches_its_worst_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="glm", n_folds=3, random_state=0).fit(x, y)
        assert model.diagnostics_.ensemble_cv_risk <= model.cv_risk_.max()

    def test_discrete_super_learner_picks_a_single_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library="glm", meta_learner="discrete", n_folds=3, random_state=0).fit(
            x, y
        )
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
        model = SuperLearner(library="glm", n_folds=3, random_state=0, clip=(0.001, 0.999)).fit(
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


class TestCrossFitRouting:
    """What an inner cross-validation is and is not told by the outer one.

    Two claims live here, and they are opposites.  The first is that the inner folds do
    *not* leak: a nuisance learner that cross-validates internally is handed only rows
    the outer training fold already owns, so its candidate weights cannot have been
    scored on a held-out row.  That is a property of
    :func:`~cleverly.estimators._nuisance.cross_fit_predictions` passing ``design[rows]``
    rather than the full design, and it is asserted rather than argued because it has
    been doubted.

    The second is that the inner folds must still be told the *cluster* structure, which
    is the one thing the row subset cannot convey.  Two rows of one cluster can sit in
    the same outer training fold and in different inner folds, and then a candidate is
    scored on a row correlated with its own training set.
    """

    @staticmethod
    def _spy() -> tuple[type, list[dict]]:
        """A SuperLearner subclass that records what each fit was given."""
        seen: list[dict] = []

        class Spy(SuperLearner):
            def fit(self, X, y, sample_weight=None, *, groups=None):  # type: ignore[no-untyped-def]
                fitted = super().fit(X, y, sample_weight, groups=groups)
                seen.append(
                    {
                        "rows": set(np.asarray(X)[:, 0].astype(int).tolist()),
                        "groups": None if groups is None else np.asarray(groups),
                        "folds": self.folds_,
                    }
                )
                return fitted

        return Spy, seen

    def test_the_inner_folds_never_see_an_outer_held_out_row(self) -> None:
        """The leak that is not there -- pinned so the claim stays checkable."""
        from cleverly.estimators._nuisance import cross_fit_predictions

        n = 200
        rng = np.random.default_rng(0)
        design = rng.normal(size=(n, 3))
        # Column 0 carries each row's own index, so the spy can identify what it was
        # handed without depending on float equality of the covariates.
        design[:, 0] = np.arange(n)
        treatment = (rng.random(n) < 0.5).astype(float)
        outer = make_folds(n, 4, stratify=treatment, random_state=0)

        Spy, seen = self._spy()
        cross_fit_predictions(
            Spy(library="glm", task="classification", n_folds=3, random_state=0, clip=(0.0, 1.0)),
            design,
            treatment,
            np.ones(n),
            outer,
            task="classification",
            predict_designs={"g1": design},
        )

        assert len(seen) == outer.n_folds
        for record, (train, test) in zip(seen, outer, strict=True):
            assert record["rows"] == set(train.tolist()), (
                "the inner cross-validation was handed rows the outer fold does not own"
            )
            assert record["rows"].isdisjoint(set(test.tolist()))

    @pytest.mark.parametrize("screen", [False, True])
    def test_cluster_codes_reach_the_inner_folds_through_screening(self, screen: bool) -> None:
        """Screening wraps the learner in a pipeline; the codes must still arrive.

        Parametrised over ``screen`` because the unwrapped case has always worked and the
        wrapped one silently did not -- running both is what makes the fix visible as a
        difference rather than as a bare assertion.
        """
        from cleverly.estimators._nuisance import _screened, cross_fit_predictions

        n = 240
        rng = np.random.default_rng(1)
        design = rng.normal(size=(n, 3))
        design[:, 0] = np.arange(n)
        treatment = (rng.random(n) < 0.5).astype(float)
        cluster = np.repeat(np.arange(n // 4), 4)
        outer = make_folds(n, 4, stratify=treatment, cluster=cluster, random_state=0)

        Spy, seen = self._spy()
        learner = Spy(
            library="glm", task="classification", n_folds=3, random_state=0, clip=(0.0, 1.0)
        )
        cross_fit_predictions(
            _screened(learner, 0.1, None) if screen else learner,
            design,
            treatment,
            np.ones(n),
            outer,
            task="classification",
            predict_designs={"g1": design},
            groups=cluster,
        )

        assert seen, "no inner fit was recorded"
        for record in seen:
            assert record["groups"] is not None, (
                "cluster codes were dropped on the way to the inner cross-validation"
            )
            # The codes are subset to the outer training rows, so they must line up with
            # the rows the fit actually saw rather than with the whole sample.
            rows = np.array(sorted(record["rows"]))
            assert np.array_equal(record["groups"], cluster[rows])
            for train, test in record["folds"]:
                assert set(record["groups"][train]).isdisjoint(set(record["groups"][test]))

    def test_weights_still_reach_a_pipeline_wrapped_learner(self) -> None:
        """The routing rewrite must not lose the parameter it already handled."""
        from cleverly.estimators._nuisance import _screened

        n = 200
        rng = np.random.default_rng(2)
        design = rng.normal(size=(n, 3))
        outcome = design[:, 0] + rng.normal(scale=0.5, size=n)
        weights = rng.uniform(0.2, 2.0, n)

        wrapped = _screened(SuperLearner(library="glm", n_folds=3, random_state=0), 0.1, None)
        plain = fit_learner(wrapped, design, outcome, None)
        weighted = fit_learner(wrapped, design, outcome, weights)
        assert not np.allclose(plain.predict(design), weighted.predict(design))

    def test_groups_are_withheld_from_a_learner_that_cannot_use_them(self) -> None:
        """A learner whose fit does not name ``groups`` is fitted without them.

        ``LinearRegression.fit`` would raise on an unexpected keyword, so this asserts the
        routing is selective rather than merely well-intentioned.
        """
        n = 60
        rng = np.random.default_rng(3)
        design = rng.normal(size=(n, 2))
        outcome = design[:, 0] + rng.normal(scale=0.1, size=n)
        model = fit_learner(
            LinearRegression(), design, outcome, None, groups=np.repeat(np.arange(n // 4), 4)
        )
        assert model.coef_.shape == (2,)
