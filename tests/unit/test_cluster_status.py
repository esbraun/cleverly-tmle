"""Clustered fits withhold their interval at unequal cross-fitted sizes and with few clusters.

Roadmap row RM20 decides two clustered statuses for ``TMLE`` and ``DRTMLE``, and F22 holds
the route that reopens each one.

``"unequal_cluster_plugin"``
    A cross-fitted fit whose clusters hold different numbers of rows. The grouped
    cross-fitting argument needs equal sizes, because only then does the row-weighted
    target equal the cluster-weighted one. ``cv_evaluation=True`` and fold targeting are
    cross-fitted, so their fold-level reports carry the status too.
``"few_cluster_plugin"``
    A fit with fewer than :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD`
    clusters, in sample or cross-fitted. The package keeps its normal reference, and
    Nugent et al. (2024), Section 2.2, recommend a t reference below 40 clusters.

The unequal fit is the RM20 probe: ``make_clustered(n=400, cluster_size=10, seed=7)`` with
rows removed from half the clusters, which leaves 315 rows in 40 clusters of 2 to 10 rows.
Forty clusters keep the few-cluster status out of it. The few-cluster fits use 39 equal
clusters, and the boundary control uses 40.

The controls keep their interval: equal sizes cross-fitted, and the unequal clusters fitted
in sample. Each mutation in :class:`TestTheStatusIsTheHooksToWithhold` must fail the check
its surface passes.
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
from cleverly._inference_status import FEW_CLUSTER_THRESHOLD
from cleverly.datasets import make_clustered
from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import cluster as cluster_module
from cleverly.inference.cluster import cluster_inference_status
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_fold_report_withholds,
    assert_fold_reports_restamped,
    assert_keeps_inference,
    assert_restamped,
    assert_variable_importance_refuses,
    assert_withholds,
    stamp_headline_only,
)
from tests.unit._natural_course_support import never_fit_learners

pytestmark = pytest.mark.xdist_group("cluster_status")

#: The estimator module, whose ``cluster_inference_status`` the hook calls. The package
#: re-exports a function named ``tmle``, so the module is imported by its path.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

UNEQUAL = "unequal_cluster_plugin"
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
    """39 clusters of 10 rows, one below the threshold."""
    frame = make_clustered(n=390, cluster_size=10, seed=7)[0]
    assert sizes(frame) == (FEW_CLUSTER_THRESHOLD - 1, 10, 10)
    return frame


@pytest.fixture(scope="module")
def few_unequal_frame(unequal_frame: Any) -> Any:
    """The unequal frame without its last cluster: 39 clusters, sizes still 2 to 10."""
    last = unequal_frame["cluster"].max()
    frame = unequal_frame[unequal_frame["cluster"] != last].reset_index(drop=True)
    assert sizes(frame) == (FEW_CLUSTER_THRESHOLD - 1, 2, 10)
    return frame


@pytest.fixture(scope="module", params=[TMLE, DRTMLE], ids=["TMLE", "DRTMLE"])
def unequal_result(request: pytest.FixtureRequest, unequal_frame: Any) -> Any:
    return fit(unequal_frame, estimator=request.param, **CROSS_FITTED)


class TestUnequalCrossFittedClustersReportNoInterval:
    def test_the_estimates_withhold_their_inference(self, unequal_result: Any) -> None:
        assert_withholds(unequal_result, UNEQUAL)

    def test_the_summary_states_the_sizes_it_read(self, unequal_result: Any) -> None:
        assert "clusters = 40, sizes 2 to 10 (cluster-robust variance)" in (
            unequal_result.summary()
        )

    def test_the_nuisance_report_and_the_assessment_carry_the_note(
        self, unequal_result: Any
    ) -> None:
        assert_assessment_note(unequal_result, UNEQUAL)

    def test_the_evalue_is_unavailable_with_the_reason(self, unequal_result: Any) -> None:
        assert_evalue_unavailable(unequal_result, UNEQUAL)


class TestTheFoldReportIsStamped:
    """``cv_evaluation=True`` and fold targeting publish a fold-level report as well."""

    @pytest.mark.parametrize(
        "scheme",
        [
            pytest.param({"cv_evaluation": True}, id="cv_evaluation"),
            pytest.param({"targeting_scheme": "fold"}, id="fold targeting"),
        ],
    )
    def test_the_fold_report_withholds(self, unequal_frame: Any, scheme: dict[str, Any]) -> None:
        result = fit(unequal_frame, **CROSS_FITTED, **scheme)
        assert result.inference_status == UNEQUAL
        assert_fold_report_withholds(result, UNEQUAL)

    def test_a_stamp_that_skips_the_fold_reports_fails_the_check(
        self, unequal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_headline_only(monkeypatch)
        mutant = fit(unequal_frame, **CROSS_FITTED, cv_evaluation=True)
        assert mutant.inference_status == UNEQUAL
        with pytest.raises(AssertionError):
            assert_fold_report_withholds(mutant, UNEQUAL)


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
        # The nonzero witness for the E-value row above: it is available here.
        assert result.sensitivity.capability("evalue").available

    def test_forty_clusters_in_sample_keep_the_interval(self, equal_frame: Any) -> None:
        """The boundary: a count equal to the threshold is not below it."""
        assert sizes(equal_frame)[0] == FEW_CLUSTER_THRESHOLD
        assert_keeps_inference(fit(equal_frame, **IN_SAMPLE))


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
        assert_withholds(fit(few_frame, estimator=estimator, **settings), FEW)

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


def fit_stratified(frame: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=("ate",)))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "S"],
            id="cluster",
            strata=["S"],
        )
        .single()
    )


@pytest.fixture(scope="module")
def few_stratum_frame() -> Any:
    """50 clusters, 6 of them in the stratum ``S = "small"``: the R1 review's probe."""
    frame = stratified_frame(50, small=6)
    assert sizes(frame) == (50, 10, 10)
    assert frame.loc[frame["S"] == "small", "cluster"].nunique() == 6
    return frame


class TestAStratumWithFewClustersReportsNoInterval:
    """Each stratum's estimate reads only its own clusters, so the count holds per stratum.

    The fit keeps one status, so a stratum of 6 clusters inside a fit of 50 withholds
    every estimate's interval.
    """

    def test_a_six_cluster_stratum_withholds_every_estimate(self, few_stratum_frame: Any) -> None:
        result = fit_stratified(few_stratum_frame)
        assert set(result.estimates) == {"ate", "ate[S='small']", "ate[S='big']"}
        assert_withholds(result, FEW)
        assert "clusters = 50, fewest in one stratum 6 (cluster-robust variance)" in (
            result.summary()
        )

    def test_strata_of_forty_clusters_each_keep_the_interval(self) -> None:
        """The control: 80 clusters split 40 and 40, each stratum at the threshold."""
        frame = stratified_frame(80, small=FEW_CLUSTER_THRESHOLD)
        result = fit_stratified(frame)
        assert_keeps_inference(result)
        assert "clusters = 80, fewest in one stratum 40 (cluster-robust variance)" in (
            result.summary()
        )

    def test_a_count_that_ignores_the_strata_fails_the_check(
        self, few_stratum_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster_module, "fewest_clusters", whole_fit_count)
        with pytest.raises(AssertionError):
            assert_withholds(fit_stratified(few_stratum_frame), FEW)


def fit_weighted(frame: Any, **settings: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=ESTIMANDS, **settings))
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster", weights="w")
        .single()
    )


@pytest.fixture(scope="module")
def unequal_mass_frame(equal_frame: Any) -> Any:
    """40 clusters of 10 rows, with weight 0.5 in the even clusters and 2 in the odd ones.

    The R1 review's probe: equal row counts, and cluster weight mass 5 or 20.
    """
    frame = equal_frame.assign(w=np.where(equal_frame["cluster"] % 2 == 0, 0.5, 2.0))
    assert sizes(frame) == (40, 10, 10)
    assert set(frame.groupby("cluster")["w"].sum()) == {5.0, 20.0}
    return frame


class TestUnequalWeightMassIsAnUnequalSize:
    """A weighted fit targets the weight-weighted mean, so weight mass is a cluster size."""

    def test_equal_rows_and_unequal_mass_cross_fitted_withhold(
        self, unequal_mass_frame: Any
    ) -> None:
        result = fit_weighted(unequal_mass_frame, **CROSS_FITTED)
        assert_withholds(result, UNEQUAL)
        # Weights are normalised to mean one, so the masses 5 and 20 print as 4 and 16.
        assert "clusters = 40, weight mass 4 to 16 (cluster-robust variance)" in (result.summary())

    def test_equal_mass_weighted_cross_fitted_keeps_the_interval(self, equal_frame: Any) -> None:
        """The control: the weights vary inside each cluster, and every cluster sums to 10."""
        frame = equal_frame.assign(w=np.tile([0.5, 1.5], len(equal_frame) // 2))
        assert set(frame.groupby("cluster")["w"].sum()) == {10.0}
        result = fit_weighted(frame, **CROSS_FITTED)
        assert_keeps_inference(result)
        assert "clusters = 40 (cluster-robust variance)" in result.summary()

    def test_unequal_mass_in_sample_keeps_the_interval(self, unequal_mass_frame: Any) -> None:
        assert_keeps_inference(fit_weighted(unequal_mass_frame, **IN_SAMPLE))

    def test_a_rule_that_ignores_the_weights_fails_the_check(
        self, unequal_mass_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster_module, "unequal_cluster_sizes", rows_only)
        with pytest.raises(AssertionError):
            assert_withholds(fit_weighted(unequal_mass_frame, **CROSS_FITTED), UNEQUAL)

    def test_the_same_weights_summed_in_another_order_are_equal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tolerance absorbs rounding: each cluster holds the same ten weights, permuted.

        The nonzero witness is that the sums differ in the last place, so an exact
        comparison, the mutation, calls the clusters unequal.
        """
        rng = np.random.default_rng(0)
        base = rng.uniform(0.1, 3.0, 10)
        weights = np.concatenate([rng.permutation(base) for _ in range(40)])
        cluster = np.repeat(np.arange(40), 10)
        assert np.ptp(cluster_module.cluster_weight_mass(cluster, weights)) > 0
        assert not cluster_module.unequal_cluster_sizes(cluster, weights)
        monkeypatch.setattr(cluster_module, "WEIGHT_MASS_RTOL", 0.0)
        assert cluster_module.unequal_cluster_sizes(cluster, weights)


class TestThePrecedence:
    """A fit that meets several statuses takes the first in the table."""

    def test_unequal_and_few_cross_fitted_take_the_unequal_status(
        self, few_unequal_frame: Any
    ) -> None:
        result = fit(few_unequal_frame, **CROSS_FITTED)
        assert_withholds(result, UNEQUAL)
        # The nonzero witness: the same clusters in sample meet only the few-cluster status.
        assert fit(few_unequal_frame, **IN_SAMPLE).inference_status == FEW

    def test_estimated_weights_come_before_both_clustered_statuses(
        self, few_unequal_frame: Any
    ) -> None:
        frame = few_unequal_frame.assign(
            w=np.random.default_rng(3).uniform(0.5, 2.0, len(few_unequal_frame))
        )
        result = (
            DRTMLE(**linear_in_sample(estimands=ESTIMANDS, **CROSS_FITTED))
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
        # The witness that both clustered statuses apply to this data on their own.
        assert cluster_inference_status(result.data.cluster, cross_fit=True) == UNEQUAL
        assert cluster_inference_status(result.data.cluster, cross_fit=False) == FEW


def ignores_sizes(cluster: Any, *, cross_fit: bool, **settings: Any) -> str:
    """The mutant that never reads the row counts: only the cluster count decides."""
    few = np.unique(cluster).size < FEW_CLUSTER_THRESHOLD
    return FEW if few else "influence_curve"


def ignores_cross_fit(cluster: Any, *, cross_fit: bool, **settings: Any) -> str:
    """The mutant that treats every fit as cross-fitted."""
    return cluster_inference_status(cluster, cross_fit=True, **settings)


def at_or_below(cluster: Any, *, cross_fit: bool, **settings: Any) -> str:
    """The mutant that compares the cluster count with ``<=`` rather than ``<``."""
    status = cluster_inference_status(cluster, cross_fit=cross_fit, **settings)
    at_threshold = np.unique(cluster).size == FEW_CLUSTER_THRESHOLD
    return FEW if status == "influence_curve" and at_threshold else status


def rows_only(cluster: Any, weights: Any = None) -> bool:
    """The mutant that compares row counts and never reads the weight mass."""
    return bool(np.ptp(np.unique(cluster, return_counts=True)[1]) > 0)


def whole_fit_count(cluster: Any, strata: Any = None) -> int:
    """The mutant that counts the clusters of the whole fit and never reads the strata."""
    return int(np.unique(cluster).size)


class TestTheStatusIsTheHooksToWithhold:
    """The mutation controls: each wrong rule fails the check its surface passes."""

    def test_a_rule_that_ignores_the_sizes_fails_the_unequal_check(
        self, unequal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "cluster_inference_status", ignores_sizes)
        with pytest.raises(AssertionError):
            assert_withholds(fit(unequal_frame, **CROSS_FITTED), UNEQUAL)

    def test_a_rule_that_ignores_cross_fit_fails_the_in_sample_control(
        self, unequal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "cluster_inference_status", ignores_cross_fit)
        # The mutant still passes the status fit, so only the control can catch it.
        assert_withholds(fit(unequal_frame, **CROSS_FITTED), UNEQUAL)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit(unequal_frame, **IN_SAMPLE))

    def test_a_threshold_of_zero_fails_the_few_cluster_check(
        self, few_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rule's own module compares against a threshold of zero."""
        monkeypatch.setattr(cluster_module, "FEW_CLUSTER_THRESHOLD", 0)
        with pytest.raises(AssertionError):
            assert_withholds(fit(few_frame, **IN_SAMPLE), FEW)

    def test_an_at_or_below_comparison_fails_the_boundary_control(
        self, few_frame: Any, equal_frame: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "cluster_inference_status", at_or_below)
        assert_withholds(fit(few_frame, **IN_SAMPLE), FEW)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit(equal_frame, **IN_SAMPLE))


class TestTheWorkflowPageRunsOnUnequalClusters:
    """``docs/workflow.md`` fits a cross-fitted weighted clustered TMLE and assesses it.

    Its households differ in size, so the fit takes the unequal status. Every call the
    page makes after the fit must still answer.
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
        assert result.inference_status == UNEQUAL
        for report in (
            result.diagnostics.support(),
            result.diagnostics.nuisance_models(),
            result.diagnostics.score_equations(),
            result.sensitivity.run_all(),
        ):
            assert report.summary()
        result.save(tmp_path / "analysis.joblib")


class TestAnOlderArtifact:
    @pytest.mark.parametrize("route", ROUTES)
    def test_a_cv_evaluation_fit_saved_before_the_status_loads_under_it(
        self, unequal_frame: Any, route: str
    ) -> None:
        result = fit(unequal_frame, **CROSS_FITTED, cv_evaluation=True)
        restored = assert_restamped(result, UNEQUAL, route)
        assert_fold_reports_restamped(restored, result, UNEQUAL)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_few_cluster_fit_saved_before_the_status_loads_under_it(
        self, few_frame: Any, route: str
    ) -> None:
        assert_restamped(fit(few_frame, **IN_SAMPLE), FEW, route)
