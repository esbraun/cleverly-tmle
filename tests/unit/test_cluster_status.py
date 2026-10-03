"""Clustered fits withhold their interval below 10 clusters, and only there.

The status table of ``docs/technical-reference/inference.md`` gives one clustered status for
``TMLE`` and ``DRTMLE``, and F28 in the roadmap records its floor.

``"few_cluster_plugin"``
    A fit with fewer than :data:`~cleverly._inference_status.MINIMUM_INTERVAL_CLUSTERS`
    contributing clusters, in sample or cross-fitted, overall or in a stratum. 10 is the
    smallest count a registered study measures. From 10 to 39 such clusters every estimate
    keeps its interval on a Student t reference with ``J - 2`` degrees of freedom, which
    ``tests/unit/test_few_cluster_reference.py`` checks.

The cluster sizes do not enter the status. The cluster total of the curve is
``A_j - N_j psi``, the delta-method curve of the row-weighted mean, so unequal sizes keep the
interval in sample and cross-fitted. ``tests/unit/test_cluster_ratio_variance.py`` checks that
algebra. The unequal fit here is the RM20 probe: ``make_clustered(n=400, cluster_size=10,
seed=7)`` with rows removed from half the clusters, which leaves 315 rows in 40 clusters of 2
to 10 rows. It was the witness of the deleted unequal-size status, and it is now a control.

Each mutation in :class:`TestTheStatusIsTheHooksToWithhold` must fail the check its surface
passes.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    Inference,
    ModelSpec,
    PointTreatment,
    Runtime,
    TMLEMethod,
)
from cleverly._inference_status import MINIMUM_INTERVAL_CLUSTERS
from cleverly.datasets import make_clustered
from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import cluster as cluster_module
from cleverly.inference.cluster import cluster_inference_status
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_keeps_inference,
    assert_round_trips,
    assert_variable_importance_refuses,
    assert_withholds,
    at_or_below,
    restore,
)
from tests.unit._natural_course_support import never_fit_learners

pytestmark = pytest.mark.xdist_group("cluster_status")

#: The estimator module, whose ``cluster_inference_status`` the hook calls. The package
#: re-exports a function named ``tmle``, so the module is imported by its path.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

FEW = "few_cluster_plugin"

#: A cross-fitted continuous outcome needs its declared range, so every cross-fitted fit
#: here passes one wide enough for ``make_clustered``.
CROSS_FITTED: dict[str, Any] = {"cross_fit": True, "n_folds": 5, "q_bounds": (-15.0, 15.0)}
IN_SAMPLE: dict[str, Any] = {"cross_fit": False}

#: ``ey0`` beside ``ate`` gives the E-value its reported reference-arm mean, so the
#: E-value row of a status fit is unavailable for the status and not for a missing input.
ESTIMANDS = ("ate", "ey0")


def unequal_clusters(frame: Any, seed: int = 7) -> Any:
    """``frame`` with a random 2 to 10 rows kept in each cluster of its first half.

    The RM20 probe's construction. On ``make_clustered(n=400, cluster_size=10, seed=7)``
    it keeps 315 rows in 40 clusters.
    """
    rng = np.random.default_rng(seed)
    clusters = np.sort(frame["cluster"].unique())
    reduced = set(clusters[: len(clusters) // 2].tolist())
    keep: list[int] = []
    for cluster in clusters:
        rows = frame.index[frame["cluster"] == cluster].to_numpy()
        if cluster in reduced:
            rows = rng.choice(rows, size=int(rng.integers(2, 11)), replace=False)
        keep.extend(rows.tolist())
    return frame.loc[sorted(keep)].reset_index(drop=True)


def sizes(frame: Any) -> tuple[int, int, int]:
    """The cluster count and the smallest and largest cluster of ``frame``."""
    counts = frame.groupby("cluster").size()
    return int(counts.size), int(counts.min()), int(counts.max())


def fit(frame: Any, *, estimator: type = TMLE, **settings: Any) -> Any:
    return (
        estimator(**linear_in_sample(estimands=ESTIMANDS, **settings))
        .fit(frame, outcome="Y", treatment="A", id="cluster")
        .single()
    )


@pytest.fixture(scope="module")
def equal_frame() -> Any:
    """40 clusters of 10 rows."""
    return make_clustered(n=400, cluster_size=10, seed=7)[0]


@pytest.fixture(scope="module")
def unequal_frame(equal_frame: Any) -> Any:
    frame = unequal_clusters(equal_frame)
    # The witness that the construction is the probe's: 40 clusters, sizes 2 to 10.
    assert len(frame) == 315
    assert sizes(frame) == (40, 2, 10)
    return frame


@pytest.fixture(scope="module")
def few_frame() -> Any:
    """9 clusters of 40 rows, one below the floor."""
    frame = make_clustered(n=360, cluster_size=40, seed=7)[0]
    assert sizes(frame) == (MINIMUM_INTERVAL_CLUSTERS - 1, 40, 40)
    return frame


@pytest.fixture(scope="module")
def floor_frame() -> Any:
    """10 clusters of 40 rows, at the floor."""
    frame = make_clustered(n=400, cluster_size=40, seed=7)[0]
    assert sizes(frame) == (MINIMUM_INTERVAL_CLUSTERS, 40, 40)
    return frame


@pytest.fixture(scope="module", params=[TMLE, DRTMLE], ids=["TMLE", "DRTMLE"])
def unequal_result(request: pytest.FixtureRequest, unequal_frame: Any) -> Any:
    return fit(unequal_frame, estimator=request.param, **CROSS_FITTED)


class TestUnequalCrossFittedClustersKeepTheInterval:
    """The RM20 probe, cross-fitted: once the witness of a status, now a control."""

    def test_the_estimates_keep_their_inference(self, unequal_result: Any) -> None:
        assert_keeps_inference(unequal_result)
        # Forty clusters: the normal reference.
        assert {e.reference_df for e in unequal_result.estimates.values()} == {None}

    def test_the_summary_states_the_sizes_it_read(self, unequal_result: Any) -> None:
        assert "clusters = 40, sizes 2 to 10 (cluster-robust variance)" in (
            unequal_result.summary()
        )

    def test_the_evalue_is_available(self, unequal_result: Any) -> None:
        assert unequal_result.sensitivity.capability("evalue").available

    @pytest.mark.parametrize(
        "scheme",
        [
            pytest.param({"cv_evaluation": True}, id="cv_evaluation"),
            pytest.param({"targeting_scheme": "fold"}, id="fold targeting"),
            pytest.param({"repeats": 2, "simultaneous": False}, id="repeats"),
        ],
    )
    def test_every_cross_fitted_scheme_keeps_the_interval(
        self, unequal_frame: Any, scheme: dict[str, Any]
    ) -> None:
        result = fit(unequal_frame, **{**CROSS_FITTED, **scheme})
        assert_keeps_inference(result)
        report = getattr(result, "cv_targeting", None)
        if report is not None:
            assert report.inference == "influence_curve"
            assert np.all(np.isfinite(list(report.std_error.values())))


class TestTheNeighbouringFitsKeepTheirInterval:
    """The controls: each differs from a status fit in one setting."""

    def test_equal_sizes_cross_fitted_keep_the_interval(self, equal_frame: Any) -> None:
        result = fit(equal_frame, **CROSS_FITTED)
        assert_keeps_inference(result)
        # Equal sizes print the count alone.
        assert "clusters = 40 (cluster-robust variance)" in result.summary()

    @pytest.mark.parametrize("estimator", [TMLE, DRTMLE], ids=["TMLE", "DRTMLE"])
    def test_unequal_sizes_in_sample_keep_the_interval(
        self, unequal_frame: Any, estimator: type
    ) -> None:
        result = fit(unequal_frame, estimator=estimator, **IN_SAMPLE)
        assert_keeps_inference(result)
        assert result.sensitivity.capability("evalue").available

    @pytest.mark.parametrize(
        "settings",
        [pytest.param(IN_SAMPLE, id="in sample"), pytest.param(CROSS_FITTED, id="cross-fitted")],
    )
    def test_ten_clusters_keep_the_interval(
        self, floor_frame: Any, settings: dict[str, Any]
    ) -> None:
        """The boundary: a count equal to the floor is not below it."""
        result = fit(floor_frame, **settings)
        assert_keeps_inference(result)
        assert {e.reference_df for e in result.estimates.values()} == {8}


class TestFewClustersReportNoInterval:
    @pytest.mark.parametrize(
        ("estimator", "settings"),
        [
            pytest.param(TMLE, IN_SAMPLE, id="TMLE in sample"),
            pytest.param(TMLE, CROSS_FITTED, id="TMLE cross-fitted"),
            pytest.param(DRTMLE, IN_SAMPLE, id="DRTMLE in sample"),
        ],
    )
    def test_the_estimates_withhold_their_inference(
        self, few_frame: Any, estimator: type, settings: dict[str, Any]
    ) -> None:
        result = fit(few_frame, estimator=estimator, **settings)
        assert_withholds(result, FEW)
        # The diagnostic keeps the normal reference.
        assert {e.reference_df for e in result.estimates.values()} == {None}

    def test_the_nuisance_report_and_the_assessment_carry_the_note(self, few_frame: Any) -> None:
        assert_assessment_note(fit(few_frame, **IN_SAMPLE), FEW)

    def test_the_evalue_is_unavailable_with_the_reason(self, few_frame: Any) -> None:
        assert_evalue_unavailable(fit(few_frame, **IN_SAMPLE), FEW)

    def test_variable_importance_refuses_before_it_fits(self, few_frame: Any) -> None:
        """The status reads the prepared cluster labels, so it refuses before a learner."""
        assert_variable_importance_refuses(
            FEW,
            few_frame,
            covariates=["W1", "W2"],
            estimator=TMLE(**linear_in_sample(**never_fit_learners())),
            id="cluster",
        )


def stratified_frame(n_clusters: int, small: int) -> Any:
    """``n_clusters`` clusters of 10 rows, with a cluster-level stratum ``S``.

    The first ``small`` clusters hold ``S = "small"`` and the rest ``S = "big"``.
    """
    frame = make_clustered(n=10 * n_clusters, cluster_size=10, seed=7)[0]
    return frame.assign(S=np.where(frame["cluster"] < small, "small", "big"))


def fit_stratified(frame: Any, **settings: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=("ate",), **settings))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "S"],
            id="cluster",
            strata=["S"],
            **({"weights": "w"} if "w" in frame else {}),
        )
        .single()
    )


@pytest.fixture(scope="module")
def few_stratum_frame() -> Any:
    """50 clusters, 3 of them in the stratum ``S = "small"``."""
    frame = stratified_frame(50, small=3)
    assert sizes(frame) == (50, 10, 10)
    assert frame.loc[frame["S"] == "small", "cluster"].nunique() == 3
    return frame


class TestAStratumWithFewClustersReportsNoInterval:
    """Each stratum's estimate reads only its own clusters, so the count holds per stratum.

    The fit keeps one status, so a stratum of 3 clusters inside a fit of 50 withholds
    every estimate's interval.
    """

    def test_a_three_cluster_stratum_withholds_every_estimate(self, few_stratum_frame: Any) -> None:
        result = fit_stratified(few_stratum_frame)
        assert set(result.estimates) == {"ate", "ate[S='small']", "ate[S='big']"}
        assert_withholds(result, FEW)
        assert "clusters = 50, fewest in one stratum 3 (cluster-robust variance)" in (
            result.summary()
        )

    def test_a_stratum_at_the_floor_keeps_the_interval(self) -> None:
        """The control: 50 clusters split 10 and 40, the small stratum at the floor."""
        frame = stratified_frame(50, small=MINIMUM_INTERVAL_CLUSTERS)
        result = fit_stratified(frame)
        assert_keeps_inference(result)
        assert result["ate[S='small']"].reference_df == 8

    def test_a_count_that_ignores_the_strata_fails_the_check(
        self, few_stratum_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster_module, "fewest_clusters", whole_fit_count)
        # The forced status reaches the stratum's t reference, which refuses 3 clusters.
        with pytest.raises((AssertionError, ValueError)):
            assert_withholds(fit_stratified(few_stratum_frame), FEW)


def within_cluster_strata(frame: Any, first: int, second: int) -> Any:
    """Forty equal ten-row clusters, with two strata inside each cluster."""
    assert first + second == 10
    labels = np.concatenate(
        [
            np.r_[np.zeros(first if j < 20 else second), np.ones(second if j < 20 else first)]
            for j in range(40)
        ]
    )
    return frame.assign(S=labels.astype(int))


class TestUnequalSizesInsideAReportedStratumKeepTheInterval:
    def test_cross_fitted_fit_keeps_the_interval(self, equal_frame: Any) -> None:
        frame = within_cluster_strata(equal_frame, 3, 7)
        result = fit_stratified(frame, **CROSS_FITTED)
        assert sizes(frame) == (40, 10, 10)
        assert_keeps_inference(result)
        assert "within-stratum sizes 3 to 7" in result.summary()


def fit_weighted(frame: Any, **settings: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=ESTIMANDS, **settings))
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster", weights="w")
        .single()
    )


class TestUnequalWeightMassKeepsTheInterval:
    def test_equal_rows_and_unequal_mass_cross_fitted_keep_the_interval(
        self, equal_frame: Any
    ) -> None:
        frame = equal_frame.assign(w=np.where(equal_frame["cluster"] % 2 == 0, 0.5, 2.0))
        assert set(frame.groupby("cluster")["w"].sum()) == {5.0, 20.0}
        result = fit_weighted(frame, **CROSS_FITTED)
        assert_keeps_inference(result)
        # Weights are normalised to mean one, so the masses 5 and 20 print as 4 and 16.
        assert "clusters = 40, weight mass 4 to 16 (cluster-robust variance)" in (result.summary())


class TestThePrecedence:
    def test_estimated_weights_come_before_the_few_cluster_status(self, few_frame: Any) -> None:
        frame = few_frame.assign(w=np.random.default_rng(3).uniform(0.5, 2.0, len(few_frame)))
        result = (
            DRTMLE(**linear_in_sample(estimands=ESTIMANDS, **IN_SAMPLE))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                id="cluster",
                weights="w",
                weights_estimated=True,
            )
            .single()
        )
        assert_withholds(result, "estimated_weight_plugin")
        # The witness that the few-cluster status applies to this data on its own.
        assert cluster_inference_status(result.data.cluster) == FEW


def whole_fit_count(cluster: Any, strata: Any = None, weights: Any = None) -> int:
    """The mutant that counts the clusters of the whole fit and never reads the strata."""
    return int(np.unique(cluster).size)


def withholds_unequal_sizes(cluster: Any, **settings: Any) -> str:
    """The deleted size branch: withhold every fit whose clusters differ in row count."""
    counts = np.unique(cluster, return_counts=True)[1]
    if counts.min() != counts.max():
        return FEW
    return cluster_inference_status(cluster, **settings)


class TestZeroWeightClusters:
    def test_only_positive_mass_clusters_count(self, equal_frame: Any) -> None:
        frame = equal_frame.assign(w=(equal_frame["cluster"] < 3).astype(float))
        result = (
            TMLE(**linear_in_sample(estimands=("ate",), **IN_SAMPLE))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                id="cluster",
                weights="w",
            )
            .single()
        )
        assert result.data.n_clusters == 40
        assert_withholds(result, FEW)
        assert "positive weight mass in 3" in result.summary()

    def test_all_positive_clusters_keep_the_interval(self, equal_frame: Any) -> None:
        frame = equal_frame.assign(w=np.ones(len(equal_frame)))
        assert_keeps_inference(fit_weighted(frame, **IN_SAMPLE))

    def test_counting_zero_mass_clusters_fails(
        self, equal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = equal_frame.assign(w=(equal_frame["cluster"] < 3).astype(float))
        monkeypatch.setattr(cluster_module, "fewest_clusters", whole_fit_count)
        with pytest.raises((AssertionError, ValueError)):
            assert_withholds(
                TMLE(**linear_in_sample(estimands=("ate",), **IN_SAMPLE))
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2"],
                    id="cluster",
                    weights="w",
                )
                .single(),
                FEW,
            )


class TestTheStatusIsTheHooksToWithhold:
    """The mutation controls: each wrong rule fails the check its surface passes."""

    def test_restoring_the_size_branch_fails_the_unequal_control(
        self, unequal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "cluster_inference_status", withholds_unequal_sizes)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit(unequal_frame, **CROSS_FITTED))

    def test_a_floor_of_zero_fails_the_few_cluster_check(
        self, few_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rule's own module compares against a floor of zero."""
        monkeypatch.setattr(cluster_module, "MINIMUM_INTERVAL_CLUSTERS", 0)
        with pytest.raises((AssertionError, ValueError)):
            assert_withholds(fit(few_frame, **IN_SAMPLE), FEW)

    def test_an_at_or_below_comparison_fails_the_boundary_control(
        self, few_frame: Any, floor_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "cluster_inference_status", at_or_below)
        assert_withholds(fit(few_frame, **IN_SAMPLE), FEW)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit(floor_frame, **IN_SAMPLE))


class TestTheWorkflowPageRunsOnUnequalClusters:
    """``docs/workflow.md`` fits a cross-fitted weighted clustered TMLE and assesses it.

    Its households differ in size, and the fit keeps its interval. Every call the page makes
    after the fit must answer.
    """

    def test_the_page_calls_answer(self, tmp_path: Any) -> None:
        base = make_clustered(n=400, cluster_size=10, seed=7, family="binomial")[0]
        data = unequal_clusters(base)
        data = data.assign(sampling_weight=np.random.default_rng(7).uniform(0.5, 1.5, len(data)))
        study = CausalStudy(
            data,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W1", "W2"),
                weights="sampling_weight",
                cluster="cluster",
            ),
        )
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
            ),
            cross_fitting=CrossFitting(n_folds=5, learner_folds=3, repeats=1),
            inference=Inference(alpha=0.05, simultaneous=False),
            runtime=Runtime(random_state=17, n_jobs=1),
        )
        result = study.identify(ATE(reference=0)).estimate(method=method)
        assert result.inference_status == "influence_curve"
        for report in (
            result.diagnostics.support(),
            result.diagnostics.nuisance_models(),
            result.diagnostics.score_equations(),
            result.sensitivity.run_all(),
        ):
            assert report.summary()
        result.save(tmp_path / "analysis.joblib")


class TestASavedResultLoadsAsSaved:
    @pytest.mark.parametrize("route", ROUTES)
    def test_a_cv_evaluation_fit_keeps_its_interval(self, unequal_frame: Any, route: str) -> None:
        result = fit(unequal_frame, **CROSS_FITTED, cv_evaluation=True)
        assert_round_trips(result, "influence_curve", route)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_few_cluster_fit_keeps_its_status(self, few_frame: Any, route: str) -> None:
        assert_round_trips(fit(few_frame, **IN_SAMPLE), FEW, route)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_t_reference_round_trips(self, floor_frame: Any, route: str) -> None:
        result = fit(floor_frame, **IN_SAMPLE)
        restored = restore(result, route)
        for name, estimate in restored.estimates.items():
            assert estimate.reference_df == result[name].reference_df == 8
            assert estimate.ci == result[name].ci
