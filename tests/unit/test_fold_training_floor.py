"""A fold-evaluated ``TMLE`` below its training-complement floor shows a labelled interval.

``MINIMUM_FOLD_TRAINING_CLUSTERS`` holds the floor that the registered
``few-cluster-fold-variance`` study measured for each treatment mechanism (X54 part (e)). A
fold-evaluated fit whose smallest training complement holds fewer positive-mass clusters takes
``"few_cluster_plugin"``. At the measured floor of 4 and the whole-fit floor of 10 clusters,
near-equal grouped folds of an unweighted fit never fall below it, so these tests raise the floor of one mechanism
to make the check fire, and keep the other as the control.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import _inference_status
from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.inference.cluster import smallest_training_clusters

tmle_module = sys.modules["cleverly.estimators.tmle"]


def _fit(*, known: bool, cv_evaluation: bool = True) -> Any:
    frame, _ = make_clustered(n=240, seed=0, cluster_size=20, family="binomial")  # 12 clusters
    columns: dict[str, Any] = {}
    if known:
        frame = frame.assign(_p1=0.5, _p0=0.5)
        columns = {"treatment_probabilities": {1: "_p1", 0: "_p0"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                cross_fit=True,
                n_folds=2,
                cv_evaluation=cv_evaluation,
                stratify_folds="none",
                estimands=("ate",),
                simultaneous=False,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                id="cluster",
                **columns,
            )
            .single()
        )


def test_the_shipped_floors_are_the_declared_route_of_the_committed_run() -> None:
    from tests.studies import few_cluster_fold_variance as study
    from tests.studies import few_cluster_fold_variance_properties as properties

    summary = pd.read_csv(study.STUDY.artifacts / "properties.csv")
    shipped = {}
    for arm in study.POSITIVE_ARMS:
        route = properties.declared_route(summary, arm)
        assert route.outcome == properties.TRAINING_FLOOR
        shipped[study.MECHANISMS[arm]] = route.minimum
    assert dict(_inference_status.MINIMUM_FOLD_TRAINING_CLUSTERS) == shipped


def test_the_smallest_training_complement_counts_positive_mass_clusters() -> None:
    cluster = np.repeat(np.arange(12), 2)
    folds = [np.arange(0, 12), np.arange(12, 24)]
    assert smallest_training_clusters(cluster, folds) == 6
    weights = np.where(cluster < 9, 1.0, 0.0)
    # The second fold's complement holds clusters 0 to 5; the first's holds 6 to 8 with mass.
    assert smallest_training_clusters(cluster, folds, weights) == 3


@pytest.mark.parametrize("known", [False, True])
def test_a_complement_below_the_floor_of_its_mechanism_is_labelled(
    monkeypatch: pytest.MonkeyPatch, known: bool
) -> None:
    mechanism = "known" if known else "estimated"
    assert _fit(known=known).estimates["ate"].inference == "influence_curve"
    # Six clusters train each fold. A floor of 7 for this mechanism labels the fit; the same
    # floor on the other mechanism leaves it unlabelled (the mutation control of the key).
    raised = {"estimated": 4, "known": 4, mechanism: 7}
    monkeypatch.setattr(tmle_module, "MINIMUM_FOLD_TRAINING_CLUSTERS", raised)
    result = _fit(known=known)
    assert result.nuisance.treatment_mechanism == mechanism
    assert result.estimates["ate"].inference == "few_cluster_plugin"
    assert np.isfinite(result.estimates["ate"].std_error)
    other = {"estimated": 4, "known": 4, ("estimated" if known else "known"): 7}
    monkeypatch.setattr(tmle_module, "MINIMUM_FOLD_TRAINING_CLUSTERS", other)
    assert _fit(known=known).estimates["ate"].inference == "influence_curve"


def test_the_floor_reads_only_the_fold_evaluated_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tmle_module, "MINIMUM_FOLD_TRAINING_CLUSTERS", {"estimated": 100, "known": 100}
    )
    stacked = _fit(known=False, cv_evaluation=False)
    assert stacked.estimates["ate"].inference == "influence_curve"


def test_the_shipped_floor_binds_through_weights_on_generated_folds() -> None:
    """The unpatched witness: zero weight on 4 clusters of one fold leaves a complement of 3."""
    frame, _ = make_clustered(n=280, seed=0, cluster_size=20, family="binomial")  # 14 clusters

    def fit(data: Any, **roles: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                TMLE(
                    outcome_learner=LogisticRegression(),
                    treatment_learner=LogisticRegression(),
                    cross_fit=True,
                    n_folds=2,
                    cv_evaluation=True,
                    stratify_folds="none",
                    estimands=("ate",),
                    simultaneous=False,
                    random_state=0,
                )
                .fit(
                    data, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster", **roles
                )
                .single()
            )

    unweighted = fit(frame)
    assert unweighted.estimates["ate"].inference == "influence_curve"
    cluster = frame["cluster"].to_numpy()
    folds = [test for _, test in unweighted.nuisance.folds]
    second = np.unique(cluster[folds[1]])
    weights = np.where(np.isin(cluster, second[:4]), 0.0, 1.0)
    weighted = fit(frame.assign(w=weights), weights="w")
    same = [test for _, test in weighted.nuisance.folds]
    assert all(np.array_equal(a, b) for a, b in zip(same, folds, strict=True))
    assert smallest_training_clusters(cluster, same, weights) == 3
    assert len(np.unique(cluster[weights > 0])) >= 10  # the whole-fit floor does not decide it
    assert weighted.estimates["ate"].inference == "few_cluster_plugin"
