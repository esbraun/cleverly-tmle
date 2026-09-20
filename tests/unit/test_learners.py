"""Fold construction, covariate screening and the Super Learner."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from cleverly.exceptions import DataError
from cleverly.learners import (
    CorrelationScreener,
    CrossFitPlan,
    Folds,
    SuperLearner,
    check_integrity,
    default_library,
    fit_learner,
    infer_task,
    make_folds,
    predict_mean,
    refuse_scheme,
    resolve_n_folds,
    screen_by_correlation,
    supports_sample_weight,
)
from tests.conftest import fast_tmle


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

    def test_one_cluster_cannot_be_cross_fitted(self) -> None:
        """The cluster cap can drive the count below two, and one fold is not a split.

        Every row is in the same cluster, so every row has to land in the same fold, and
        a single fold trains on nothing.  The cap fires first and says why the count
        moved; the refusal follows and says why the result is unusable.
        """
        cluster = np.zeros(60, dtype=np.int64)
        with (
            pytest.raises(ValueError, match="at least 2 clusters"),
            pytest.warns(UserWarning, match="only 1 clusters"),
        ):
            resolve_n_folds(10, 60, cluster=cluster)


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

    def test_a_grouped_split_keeps_clusters_intact_at_unequal_sizes(self) -> None:
        # The grouped split is package code now, so cluster integrity is this package's
        # own guarantee. Unequal sizes are the case a size-aware splitter treats
        # differently, so they are the case to check.
        cluster = _unequal_clusters(np.random.default_rng(3), n_clusters=40)
        folds = make_folds(cluster.size, 5, cluster=cluster, random_state=0)
        check_integrity(folds, cluster=cluster)
        for train, test in folds:
            assert set(cluster[train]).isdisjoint(set(cluster[test]))


#: Seeds for the generator grids: small, arbitrary, the largest seed
#: ``CrossFitPlan.seeds`` can return, and the largest ``RandomState`` accepts.
_GRID_SEEDS = (0, 1, 7, 2024, 2**31 - 2, 2**32 - 1)


def _unequal_clusters(rng: np.random.Generator, *, n_clusters: int) -> np.ndarray:
    """Cluster codes with sizes from one to nine, sparse labels, and shuffled rows.

    Sparse labels check that the split reads the sorted distinct labels rather than the
    codes ``0..m-1``. Shuffled rows check that it reads labels rather than row order.
    """
    labels = 3 + 7 * np.arange(n_clusters)
    sizes = rng.integers(1, 10, size=n_clusters)
    return rng.permutation(np.repeat(labels, sizes))


def _sklearn_assignment(splitter, n: int, groups: np.ndarray | None = None) -> np.ndarray:  # type: ignore[no-untyped-def]
    assignment = np.full(n, -1, dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros((n, 1)), None, groups)):
        assignment[test] = fold
    assert (assignment >= 0).all()
    return assignment


def _assert_row_split_matches_kfold() -> None:
    """W-gen1: the row-level generator is ``KFold(shuffle=True)`` under an integer seed.

    The grid runs through the production names at call time, ``make_folds`` included, so
    a mutant patched into ``crossfit.random_partition`` reaches both.
    """
    from sklearn.model_selection import KFold

    from cleverly.learners import crossfit

    for n in (2, 3, 7, 10, 31, 100, 257):
        for n_folds in (2, 3, 5, 10):
            if n_folds > n:
                continue
            for seed in _GRID_SEEDS:
                expected = _sklearn_assignment(
                    KFold(n_splits=n_folds, shuffle=True, random_state=seed), n
                )
                drawn = crossfit.random_partition(n, n_folds, seed=seed)
                routed = crossfit.make_folds(n, n_folds, random_state=seed)
                assert np.array_equal(drawn.assignment, expected), (n, n_folds, seed)
                assert np.array_equal(routed.assignment, expected), (n, n_folds, seed)


def _assert_grouped_split_matches_groupkfold() -> None:
    """W-gen2, first half: the grouped generator is 1.6's ``GroupKFold(shuffle=True)``."""
    from sklearn.model_selection import GroupKFold

    from cleverly.learners import crossfit

    rng = np.random.default_rng(11)
    for n_clusters in (2, 5, 13, 40):
        cluster = _unequal_clusters(rng, n_clusters=n_clusters)
        for n_folds in (2, 3, 5):
            if n_folds > n_clusters:
                continue
            for seed in _GRID_SEEDS:
                splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                expected = _sklearn_assignment(splitter, cluster.size, cluster)
                drawn = crossfit.random_partition(cluster.size, n_folds, cluster=cluster, seed=seed)
                routed = crossfit.make_folds(
                    cluster.size, n_folds, cluster=cluster, random_state=seed
                )
                assert np.array_equal(drawn.assignment, expected), (n_clusters, n_folds, seed)
                assert np.array_equal(routed.assignment, expected), (n_clusters, n_folds, seed)


def _label_folds(folds: Folds, cluster: np.ndarray) -> dict[int, int]:
    return {int(label): int(folds.assignment[cluster == label][0]) for label in np.unique(cluster)}


def _assert_grouped_split_reads_only_labels_and_seed() -> None:
    """W-gen2, second half: cluster sizes and row order do not move a cluster's fold.

    Two designs share the same distinct labels and differ in every cluster's size and in
    row order. Each label must land in the same fold in both. A splitter that balances
    rows by size, as ``GroupKFold`` without ``shuffle`` does, fails this.
    """
    from cleverly.learners import crossfit

    rng = np.random.default_rng(5)
    labels = 3 + 7 * np.arange(30)
    first = rng.permutation(np.repeat(labels, rng.integers(1, 12, size=labels.size)))
    second = rng.permutation(np.repeat(labels, rng.integers(1, 12, size=labels.size)))
    assert not np.array_equal(
        np.unique(first, return_counts=True)[1], np.unique(second, return_counts=True)[1]
    )
    for n_folds in (2, 3, 5):
        for seed in _GRID_SEEDS:
            one = crossfit.random_partition(first.size, n_folds, cluster=first, seed=seed)
            two = crossfit.random_partition(second.size, n_folds, cluster=second, seed=seed)
            assert _label_folds(one, first) == _label_folds(two, second), (n_folds, seed)


def _default_rng_mutant(
    n: int, n_folds: int, *, cluster: np.ndarray | None = None, seed: int
) -> Folds:
    """A plausible rewrite that draws the row order from ``default_rng`` instead."""
    from cleverly.learners import crossfit

    resolved = resolve_n_folds(n_folds, n, cluster=cluster)
    order = np.random.default_rng(seed).permutation(n)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = crossfit._contiguous_blocks(n, resolved)
    return Folds(assignment, resolved)


def _greedy_size_mutant(
    n: int, n_folds: int, *, cluster: np.ndarray | None = None, seed: int
) -> Folds:
    """The removed scikit-learn < 1.6 fallback: relabel, then assign greedily by size."""
    from sklearn.model_selection import GroupKFold

    assert cluster is not None
    resolved = resolve_n_folds(n_folds, n, cluster=cluster)
    unique = np.unique(cluster)
    relabel = np.random.default_rng(seed).permutation(unique.size)
    lookup = dict(zip(unique.tolist(), relabel.tolist(), strict=True))
    shuffled = np.array([lookup[int(code)] for code in cluster], dtype=np.int64)
    assignment = _sklearn_assignment(GroupKFold(n_splits=resolved), n, shuffled)
    return Folds(assignment, resolved)


class TestTheOuterFoldGenerator:
    """``random_partition`` owns the unstratified outer split, row-level and grouped.

    Each witness has a mutation control that patches a plausible wrong generator into
    production and requires the witness to fail. A witness that passes on the mutant
    would pass on a generator that does not do what the witness names.
    """

    def test_the_row_split_is_kfold_with_an_integer_seed(self) -> None:
        _assert_row_split_matches_kfold()

    def test_the_row_split_witness_rejects_a_default_rng_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cleverly.learners import crossfit

        monkeypatch.setattr(crossfit, "random_partition", _default_rng_mutant)
        with pytest.raises(AssertionError):
            _assert_row_split_matches_kfold()

    def test_the_grouped_split_is_shuffled_groupkfold(self) -> None:
        import sklearn

        major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
        if (major, minor) < (1, 6):
            pytest.skip("GroupKFold(shuffle=True) arrived in scikit-learn 1.6")
        _assert_grouped_split_matches_groupkfold()

    def test_the_grouped_split_reads_only_labels_and_seed(self) -> None:
        _assert_grouped_split_reads_only_labels_and_seed()

    @pytest.mark.parametrize(
        "witness",
        [
            _assert_grouped_split_matches_groupkfold,
            _assert_grouped_split_reads_only_labels_and_seed,
        ],
    )
    def test_each_grouped_witness_rejects_the_greedy_size_fallback(
        self, monkeypatch: pytest.MonkeyPatch, witness
    ) -> None:  # type: ignore[no-untyped-def]
        from cleverly.learners import crossfit

        monkeypatch.setattr(crossfit, "random_partition", _greedy_size_mutant)
        with pytest.raises(AssertionError):
            witness()

    def test_the_origin_records_the_request_before_the_cap(self) -> None:
        from cleverly.learners import FoldOrigin, random_partition

        cluster = np.repeat(np.array([4, 9, 20]), [2, 5, 3])
        with pytest.warns(UserWarning, match="only 3 clusters"):
            folds = random_partition(cluster.size, 5, cluster=cluster, seed=17)
        assert folds.n_folds == 3
        assert folds.origin == FoldOrigin(
            generator="cleverly.random_partition/1",
            scheme="grouped",
            requested_n_folds=5,
            seed=17,
        )
        row = random_partition(10, 4, seed=3)
        assert row.origin is not None
        assert (row.origin.scheme, row.origin.requested_n_folds, row.origin.seed) == (
            "vfold",
            4,
            3,
        )

    def test_a_stratified_split_carries_no_origin(self, sample) -> None:
        # The origin names the package generator, and a stratified split is scikit-learn's.
        _, _, a = sample
        assert make_folds(a.size, 5, stratify=a, random_state=0).origin is None

    def test_an_unseeded_split_records_a_seed_that_reproduces_it(self) -> None:
        from cleverly.learners import random_partition

        folds = make_folds(50, 5, random_state=None)
        assert folds.origin is not None
        again = random_partition(50, 5, seed=folds.origin.seed)
        assert np.array_equal(again.assignment, folds.assignment)

    def test_equality_ignores_the_origin(self) -> None:
        from dataclasses import fields

        assert {spec.name: spec.compare for spec in fields(Folds)}["origin"] is False

    def test_a_pickle_that_predates_the_origin_restores_with_none(self) -> None:
        import pickle

        restored = object.__new__(Folds)
        restored.__setstate__({"assignment": np.array([0, 1, 0, 1]), "n_folds": 2})
        assert restored.origin is None
        assert pickle.loads(pickle.dumps(restored)).origin is None

    @pytest.mark.parametrize(
        ("seed", "match"),
        [
            (-1, "must lie in"),
            (2**32, "must lie in"),
            (1.5, "must be an integer"),
            (True, "integer"),
        ],
    )
    def test_a_seed_outside_the_generator_range_is_refused(self, seed, match: str) -> None:  # type: ignore[no-untyped-def]
        from cleverly.learners import random_partition

        with pytest.raises(ValueError, match=match):
            random_partition(10, 2, seed=seed)


class TestRefusedSchemes:
    """Three schemes, three different reasons -- which is why they are named separately.

    Plus one that is no longer refused at all: ``"repeated"`` shipped as ``repeats=``, and
    the branch survives only to say it was never a scheme.
    """

    def test_blocked_temporal_names_the_missing_ordering(self) -> None:
        with pytest.raises(NotImplementedError, match="no node carries a time index"):
            refuse_scheme("blocked")

    def test_rolling_origin_names_the_storage_contract_not_the_time_index(self) -> None:
        # The distinction that matters: a rolling origin would still be refused after a
        # time index arrived, because one out-of-fold prediction per row is what
        # NuisanceEstimates is built on.
        with pytest.raises(NotImplementedError, match="different storage contract"):
            refuse_scheme("rolling_origin")

    def test_repeated_is_redirected_rather_than_refused(self) -> None:
        # A ValueError rather than NotImplementedError, and that is the whole point: the
        # feature exists, so the caller is pointed at repeats= rather than told to wait.
        with pytest.raises(ValueError, match="is not a scheme") as excinfo:
            refuse_scheme("repeated")
        assert not isinstance(excinfo.value, NotImplementedError)
        assert "repeats=" in str(excinfo.value)

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

    def test_describe_names_the_repeat_count_only_when_there_is_one(self) -> None:
        # An ordinary fit's line must not grow a clause saying "median over 1 draw":
        # a line that always appears is a line nobody reads.
        assert "draws" not in CrossFitPlan(n_folds=5, scheme="vfold").describe()
        assert CrossFitPlan(n_folds=5, scheme="vfold", repeats=4).describe() == (
            "declared: 5-fold vfold, median over 4 draws"
        )

    def test_one_repeat_passes_the_seed_straight_through(self) -> None:
        # The bit-for-bit guarantee rests on this: spawning from SeedSequence would give
        # repeats=1 a *different* seed from no repeats at all, and every fold assignment
        # would move.
        assert CrossFitPlan(random_state=7).seeds() == (7,)
        assert CrossFitPlan(random_state=None).seeds() == (None,)

    def test_repeats_get_distinct_reproducible_seeds(self) -> None:
        seeds = CrossFitPlan(random_state=7, repeats=4).seeds()
        assert len(seeds) == 4
        assert len(set(seeds)) == 4
        assert seeds == CrossFitPlan(random_state=7, repeats=4).seeds()
        assert seeds != CrossFitPlan(random_state=8, repeats=4).seeds()

    def test_unseeded_repeats_stay_unseeded(self) -> None:
        # None per draw rather than invented seeds: make_folds always shuffles, so the
        # draws differ anyway, and pinning them would promise a reproducibility the
        # caller declined by passing random_state=None.
        assert CrossFitPlan(random_state=None, repeats=3).seeds() == (None, None, None)


class TestTheDeclaredPlanIsRecordedOnAFit:
    """What the fit asked for, beside what it got -- which is the whole point.

    ``TMLEConfig.n_folds`` is the realised count and always has been.  The two agree in
    the ordinary case and come apart exactly when a cap fired, at which point the warning
    that explained it has already gone.
    """

    def _frame(self, n_clusters: int = 3, per_cluster: int = 20):  # type: ignore[no-untyped-def]
        # A binary outcome: this class fits with cross_fit=True (the default), and a
        # cross-fitted continuous outcome now needs a declared q_bounds. What this class
        # is about -- the declared fold plan beside the realised one -- does not depend
        # on the outcome's family, so a binary law keeps the fit accepted without one.
        pd = pytest.importorskip("pandas")
        n = n_clusters * per_cluster
        rng = np.random.default_rng(0)
        w = rng.normal(size=n)
        a = rng.binomial(1, 0.5, n).astype(float)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(w + a)))).astype(float)
        return pd.DataFrame(
            {
                "Y": y,
                "A": a,
                "W": w,
                "pid": np.repeat(np.arange(n_clusters), per_cluster),
            }
        )

    def _fit(self, frame, *, cluster_id: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
        fit_args = {"covariates": ["W"]}
        if cluster_id is not None:
            fit_args["id"] = cluster_id  # type: ignore[assignment]
        return fast_tmle(**kwargs).fit(frame, outcome="Y", treatment="A", **fit_args).single()

    def test_an_ordinary_fit_declares_what_it_ran(self) -> None:
        config = self._fit(self._frame(), n_folds=3).config
        assert config.crossfit.n_folds == 3 == config.n_folds
        assert config.crossfit.scheme == "vfold"
        assert config.crossfit.stratify_by == ()

    def test_a_capped_split_records_both_counts(self) -> None:
        # Three clusters cannot support ten folds. Before the plan, the realised 3 was
        # the only number on the result and the declared 10 was unrecoverable.
        frame = self._frame(n_clusters=3, per_cluster=20)
        with pytest.warns(UserWarning, match="only 3 clusters"):
            config = self._fit(frame, n_folds=10, cluster_id="pid").config
        assert config.crossfit.n_folds == 10
        assert config.n_folds == 3
        assert config.crossfit.scheme == "grouped"

    def test_the_summary_says_so_only_when_they_disagree(self) -> None:
        frame = self._frame(n_clusters=3, per_cluster=20)
        with pytest.warns(UserWarning, match="only 3 clusters"):
            capped = self._fit(frame, n_folds=10, cluster_id="pid").config
        assert any("10 folds were declared" in line for line in capped.describe())
        agreed = self._fit(self._frame(), n_folds=3).config
        assert not any("were declared" in line for line in agreed.describe())

    def test_no_cross_fitting_is_declared_as_one_fold(self) -> None:
        config = self._fit(self._frame(), cross_fit=False).config
        assert not config.crossfit.cross_fit
        assert config.crossfit.scheme == "none"
        assert config.crossfit.stratify_by == ()


class TestAnUndeclaredSeedIsResolved:
    """``random_state=None`` still draws every split from a concrete, recorded seed.

    The recorded seed is checked by replay: a fit declared with that seed must reproduce
    the unseeded fit exactly. A seed that reached the folds but not the learners or the
    selection folds would fail the replay, because those stages would then draw from
    unrecorded entropy.
    """

    def _frame(self):  # type: ignore[no-untyped-def]
        # A binary outcome: some tests here cross-fit (the default), and a cross-fitted
        # continuous outcome now needs a declared q_bounds, which is beside what seed
        # replay is about.
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(4)
        n = 120
        w = rng.normal(size=n)
        a = rng.binomial(1, 1.0 / (1.0 + np.exp(-w))).astype(float)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(w + a)))).astype(float)
        return pd.DataFrame({"Y": y, "A": a, "W": w})

    def _fit(self, estimator, frame):  # type: ignore[no-untyped-def]
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()

    @pytest.mark.parametrize("cross_fit", [True, False])
    def test_the_recorded_seed_replays_the_unseeded_fit(self, cross_fit: bool) -> None:
        frame = self._frame()
        unseeded = self._fit(fast_tmle(random_state=None, cross_fit=cross_fit), frame)
        seed = unseeded.repeats[0].seed
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1
        assert unseeded.config.crossfit.random_state is None
        replay = self._fit(fast_tmle(random_state=seed, cross_fit=cross_fit), frame)
        assert replay.repeats[0].seed == seed
        assert np.array_equal(replay.nuisance.folds.assignment, unseeded.nuisance.folds.assignment)
        assert replay["ate"].psi == unseeded["ate"].psi
        assert replay["ate"].std_error == unseeded["ate"].std_error

    def test_each_unseeded_repeat_gets_its_own_seed(self) -> None:
        result = self._fit(fast_tmle(random_state=None, repeats=3), self._frame())
        seeds = [repeat.seed for repeat in result.repeats]
        assert all(isinstance(seed, int) for seed in seeds)
        assert len(set(seeds)) == 3

    def test_a_declared_seed_passes_through(self) -> None:
        result = self._fit(fast_tmle(random_state=5), self._frame())
        assert result.repeats[0].seed == 5
        repeated = self._fit(fast_tmle(random_state=5, repeats=2), self._frame())
        plan = repeated.config.crossfit
        assert tuple(repeat.seed for repeat in repeated.repeats) == plan.seeds()

    def test_the_collaborative_selection_replays_from_the_recorded_seed(self) -> None:
        # C-TMLE draws selection folds even at cross_fit=False, and the selector reads the
        # draw seed. The replay therefore fails if the resolved seed misses the selector.
        from cleverly.estimators import CTMLE
        from tests.conftest import FAST_KWARGS

        frame = self._frame()
        kwargs = {**FAST_KWARGS, "estimands": ("ate",), "selection_folds": 3, "cross_fit": False}
        unseeded = self._fit(CTMLE(**{**kwargs, "random_state": None}), frame)
        seed = unseeded.repeats[0].seed
        assert isinstance(seed, int)
        replay = self._fit(CTMLE(**{**kwargs, "random_state": seed}), frame)
        assert replay["ate"].psi == unseeded["ate"].psi
        assert replay["ate"].std_error == unseeded["ate"].std_error


class TestStratifyingOnARareOutcome:
    """``stratify_folds="treatment+outcome"``: the one setting here that moves a number.

    The claim is narrow and checkable: crossing the outcome into the stratum caps the
    fold count at the rarest *cell*, so a fold cannot come out with no events in an arm.
    Asserted on the fold assignment rather than on an estimate -- exact, and it needs no
    targeting step, which on a deliberately rare outcome would only report separation.
    """

    def _rare_event_data(self, n: int = 300, n_events: int = 8):  # type: ignore[no-untyped-def]
        from cleverly.data import CausalData

        rng = np.random.default_rng(0)
        a = np.zeros(n)
        a[: n // 2] = 1.0
        y = np.zeros(n)
        # Every event in the treated arm, so an arm-stratified split can still leave a
        # fold with no treated events at all -- which is the failure being prevented.
        y[rng.choice(n // 2, size=n_events, replace=False)] = 1.0
        return CausalData.from_arrays(
            outcome=y, treatment=a, covariates=rng.normal(size=(n, 1)), family="binomial"
        )

    def _assignment(self, data, **kwargs):  # type: ignore[no-untyped-def]
        return fast_tmle(**kwargs)._folds(data).assignment

    def test_the_default_can_leave_a_fold_with_no_events(self) -> None:
        data = self._rare_event_data(n=300, n_events=8)
        assignment = self._assignment(data, n_folds=10)
        # 8 events over 10 arm-stratified folds: at least one fold has none of them.
        per_fold = [data.outcome[assignment == f].sum() for f in range(assignment.max() + 1)]
        assert min(per_fold) == 0

    def test_crossing_the_outcome_in_is_refused_under_cross_fitting(self) -> None:
        """ "treatment+outcome" used to reach every rare cell; cross-fitting refuses it now.

        No fold count can revive it: the policy is refused before any split is drawn,
        which is why the refusal reaches the caller through :meth:`TMLE._folds` exactly
        where the old capped assignment used to come from.
        """
        data = self._rare_event_data(n=300, n_events=8)
        with pytest.raises(ValueError, match=r"stratify_folds='treatment\+outcome'"):
            self._assignment(data, n_folds=10, stratify_folds="treatment+outcome")

    def test_the_fold_count_is_capped_at_the_rarest_cell(self) -> None:
        """The old cap never runs: the declaration is refused before ``data`` is read.

        ``n_folds=10`` and eight events in the rarest cell used to leave eight folds
        capped from ten.  Refusing the declaration at construction, with no ``data``
        argument at all, is stricter than that cap and makes it unreachable.
        """
        with pytest.raises(ValueError, match=r"stratify_folds='treatment\+outcome'"):
            fast_tmle(n_folds=10, stratify_folds="treatment+outcome")

    def test_the_default_is_now_none_and_the_old_default_is_refused(self) -> None:
        """``stratify_folds="treatment"`` was the default; it is refused under cross-fitting now."""
        data = self._rare_event_data()
        explicit_none = self._assignment(data, n_folds=5, stratify_folds="none")
        implied = self._assignment(data, n_folds=5)
        np.testing.assert_array_equal(explicit_none, implied)
        with pytest.raises(ValueError, match=r"stratify_folds='treatment'"):
            self._assignment(data, n_folds=5, stratify_folds="treatment")

    def test_a_missing_outcome_is_its_own_stratum_not_a_zero(self) -> None:
        # A fold with no *observed* outcomes in an arm cannot fit the regression either,
        # so Delta=0 must not be pooled with Y=0.
        pd = pytest.importorskip("pandas")
        from cleverly.data import CausalData

        rng = np.random.default_rng(0)
        n = 200
        frame = pd.DataFrame(
            {
                "Y": rng.binomial(1, 0.5, n).astype(float),
                "A": rng.binomial(1, 0.5, n).astype(float),
                "W": rng.normal(size=n),
                "D": np.ones(n),
            }
        )
        frame.loc[:5, "D"] = 0.0
        frame.loc[:5, "Y"] = np.nan
        # cross_fit=False: this checks _fold_strata's own per-row coding, which is
        # unaffected by the policy migration, and not the now-refused declaration of
        # stratify_folds="treatment+outcome" under cross-fitting.
        est = fast_tmle(stratify_folds="treatment+outcome", cross_fit=False)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"], delta="D")
        codes = est._fold_strata(data)
        assert codes is not None
        # The unobserved rows must not share a code with the observed Y=0 rows of the
        # same arm, or a fold could hold only unobserved outcomes there.
        unobserved = set(np.unique(codes[~data.observed]).tolist())
        observed = set(np.unique(codes[data.observed]).tolist())
        assert unobserved.isdisjoint(observed)

    def test_a_continuous_outcome_is_refused_by_name(self) -> None:
        from cleverly.data import CausalData

        rng = np.random.default_rng(0)
        data = CausalData.from_arrays(
            outcome=rng.normal(size=200),
            treatment=rng.binomial(1, 0.5, 200).astype(float),
            covariates=rng.normal(size=(200, 1)),
        )
        # cross_fit=False: reaches _fold_strata's own refusal rather than the
        # now-earlier one that a cross-fitted declaration of this policy raises first.
        with pytest.raises(DataError, match="needs a binary outcome"):
            fast_tmle(stratify_folds="treatment+outcome", cross_fit=False)._fold_strata(data)

    def test_an_unknown_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stratify_folds must be"):
            fast_tmle(stratify_folds="outcome")


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
    @pytest.mark.parametrize("task", ["regression", "classification"])
    def test_default_resolves_to_three_named_estimators(self, task: str) -> None:
        from cleverly.learners.library import _resolve_library

        library = _resolve_library(None, task)  # type: ignore[arg-type]
        assert [name for name, _ in library] == ["hist_gradient_boosting", "random_forest", "lasso"]
        assert all(hasattr(estimator, "fit") for _, estimator in library)

    def test_regression_default_uses_the_declared_sklearn_models(self) -> None:
        library = default_library("regression", random_state=3)
        assert isinstance(library[0][1], HistGradientBoostingRegressor)
        assert isinstance(library[1][1], RandomForestRegressor)
        assert isinstance(library[2][1], Pipeline)

    def test_bare_estimators_get_names(self) -> None:
        from cleverly.learners.library import _resolve_library

        library = _resolve_library([LinearRegression(), DummyRegressor()], "regression")
        assert [name for name, _ in library] == ["LinearRegression", "DummyRegressor"]

    def test_duplicate_names_are_disambiguated(self) -> None:
        from cleverly.learners.library import _resolve_library

        library = _resolve_library([LinearRegression(), LinearRegression()], "regression")
        assert [name for name, _ in library] == ["LinearRegression", "LinearRegression_2"]

    def test_explicit_pairs_pass_through(self) -> None:
        from cleverly.learners.library import _resolve_library

        library = _resolve_library([("mine", LinearRegression())], "regression")
        assert library[0][0] == "mine"

    def test_string_presets_are_refused(self) -> None:
        with pytest.raises(TypeError, match="estimator objects"):
            from cleverly.learners.library import _resolve_library

            _resolve_library("glm", "regression")  # type: ignore[arg-type]

    def test_empty_library_is_refused(self) -> None:
        with pytest.raises(ValueError, match="library is empty"):
            from cleverly.learners.library import _resolve_library

            _resolve_library([], "regression")


class TestSuperLearner:
    @staticmethod
    def library() -> list[object]:
        return [LinearRegression(), DummyRegressor()]

    @staticmethod
    def classifier_library() -> list[tuple[str, object]]:
        return [
            ("logistic", LogisticRegression(max_iter=200)),
            ("forest", RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0)),
        ]

    @staticmethod
    def assert_probabilities(model: SuperLearner, x: np.ndarray, n_classes: int) -> np.ndarray:
        probabilities = model.predict_proba(x)
        assert probabilities.shape == (len(x), n_classes)
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        return probabilities

    def test_weights_lie_on_the_simplex(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(x, y)
        assert model.coef_.sum() == pytest.approx(1.0)
        assert np.all(model.coef_ >= -1e-12)

    def test_the_ensemble_beats_or_matches_its_worst_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(x, y)
        assert model.diagnostics_.ensemble_cv_risk <= model.cv_risk_.max()

    def test_discrete_super_learner_picks_a_single_candidate(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(
            library=self.library(), meta_learner="discrete", n_folds=3, random_state=0
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
        model = SuperLearner(
            library=self.library(), n_folds=3, random_state=0, clip=(0.001, 0.999)
        ).fit(x, a)
        p = model.predict(x)
        assert p.min() >= 0.001
        assert p.max() <= 0.999
        assert model.predict_proba(x).shape == (len(a), 2)
        assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)

    def test_explicit_regression_drops_named_classifier_candidates(self, sample) -> None:
        x, y, _ = sample
        library = [*self.classifier_library(), ("ols", LinearRegression())]
        with pytest.warns(UserWarning, match="dropping learner"):
            model = SuperLearner(
                library=library,
                task="regression",
                n_folds=3,
                random_state=0,
            ).fit(x, y)
        assert model.learner_names_ == ("ols",)
        assert np.all(np.isfinite(model.predict(x)))

    def test_inferred_binary_classification_retains_named_classifiers(self, sample) -> None:
        x, _, treatment = sample
        model = SuperLearner(
            library=self.classifier_library(),
            n_folds=3,
            random_state=0,
        ).fit(x, treatment)
        assert model.task_ == "classification"
        assert model.learner_names_ == ("logistic", "forest")
        self.assert_probabilities(model, x, 2)

    def test_explicit_multiclass_classification_retains_each_ovr_library(self, sample) -> None:
        x, _, _ = sample
        lower, upper = np.quantile(x[:, 0], (1.0 / 3.0, 2.0 / 3.0))
        target = np.digitize(x[:, 0], (lower, upper)).astype(float)
        model = SuperLearner(
            library=self.classifier_library(),
            task="classification",
            n_folds=3,
            random_state=0,
        ).fit(x, target)
        assert model.is_multiclass
        assert model.classes_.tolist() == [0.0, 1.0, 2.0]
        self.assert_probabilities(model, x, 3)
        assert all(member.learner_names_ == ("logistic", "forest") for member in model.one_vs_rest_)

    def test_nnloglik_is_the_default_for_a_binary_target(self, sample) -> None:
        x, _, a = sample
        model = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(x, a)
        assert model.diagnostics_.loss == "neg_log_likelihood"

    def test_out_of_fold_predictions_are_retained_for_every_row(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library=self.library(), n_folds=4, random_state=0).fit(x, y)
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
        model = SuperLearner(library=self.library(), n_folds=4, random_state=0).fit(
            x, y, groups=groups
        )
        for train, test in model.folds_:
            assert set(groups[train]).isdisjoint(set(groups[test]))

    def test_weights_change_the_fit(self, sample) -> None:
        x, y, _ = sample
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.2, 2.0, len(y))
        plain = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(x, y)
        weighted = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(
            x, y, sample_weight=weights
        )
        assert not np.allclose(plain.predict(x), weighted.predict(x))

    def test_weights_mapping_matches_the_coefficients(self, sample) -> None:
        x, y, _ = sample
        model = SuperLearner(library=self.library(), n_folds=3, random_state=0).fit(x, y)
        assert model.weights == dict(zip(model.learner_names_, model.coef_.tolist(), strict=True))

    def test_mismatched_lengths_are_refused(self, sample) -> None:
        x, y, _ = sample
        with pytest.raises(ValueError, match="expected"):
            SuperLearner(library=self.library(), n_folds=3).fit(x, y[:-1])


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
            Spy(
                library=[LogisticRegression(max_iter=1000)],
                task="classification",
                n_folds=3,
                random_state=0,
                clip=(0.0, 1.0),
            ),
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
            library=[LogisticRegression(max_iter=1000)],
            task="classification",
            n_folds=3,
            random_state=0,
            clip=(0.0, 1.0),
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

        wrapped = _screened(
            SuperLearner(library=[LinearRegression()], n_folds=3, random_state=0), 0.1, None
        )
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
