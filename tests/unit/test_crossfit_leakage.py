"""What grouped cross-fitting buys, demonstrated rather than asserted.

Cluster integrity is checked structurally in :mod:`tests.unit.test_learners` -- no
cluster's rows land in two folds. That is a statement about fold *shape*, and it is not
the same claim as the one grouped cross-fitting is for: that a nuisance model does not get
to see, in training, rows that stand in for the ones it is about to predict.

This module makes the second claim exactly. The construction is rigged so that leakage is
not a matter of degree: one covariate is constant within a cluster and the outcome *is*
that covariate, with no noise at all. A one-nearest-neighbour learner then reproduces a
held-out row's outcome to the bit whenever a same-cluster row is available to train on,
and cannot come close when it is not. So the two assertions are array equality and array
inequality -- no tolerance, no seed sensitivity, no "the effect shrinks". A statistical
version of this test would be weaker and slower for no gain.

The end-to-end case at the bottom is the regression test proper: it guards the wiring that
carries cluster codes from ``id=`` down to the splitter, which the exact checks above say
nothing about.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor

from cleverly.estimators._nuisance import cross_fit_predictions
from cleverly.learners import make_folds
from tests.conftest import fast_tmle

#: 40 clusters of 8 rows. Enough clusters that a 5-fold grouped split exists, and enough
#: rows per cluster that an ungrouped split leaves every row a same-cluster training
#: neighbour. That second property is what the leaky case measures, so it is asserted as
#: a precondition rather than trusted: at 8 rows and 5 folds a cluster landing wholly
#: inside one fold has probability ~1e-5, but a test that quietly stops testing anything
#: on an unlucky seed is worse than one that says so.
N_CLUSTERS = 40
PER_CLUSTER = 8
N = N_CLUSTERS * PER_CLUSTER


def assert_every_row_has_a_same_cluster_neighbour(folds: object, cluster: np.ndarray) -> None:
    """The precondition the ungrouped case rests on."""
    for train, test in folds:  # type: ignore[attr-defined]
        stranded = set(cluster[test].tolist()) - set(cluster[train].tolist())
        assert not stranded, (
            f"cluster(s) {sorted(stranded)} fell entirely inside one validation fold, so "
            "the ungrouped split leaks nothing for them and the comparison below is "
            "vacuous; raise PER_CLUSTER or change the seed"
        )


def leaky_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A design whose outcome is *exactly* a cluster-constant covariate.

    ``W`` takes a distinct value per cluster and ``Y == W``. Nothing else varies, so a
    1-NN learner's prediction for a held-out row is the outcome of whichever training row
    is nearest in ``W`` -- a same-cluster row (identical ``W``, identical ``Y``) when the
    split let one through, and a different cluster's row otherwise.
    """
    cluster = np.repeat(np.arange(N_CLUSTERS), PER_CLUSTER).astype(np.int64)
    # Distinct, well-separated values so "nearest in W" is never ambiguous between
    # clusters, and equality within a cluster is exact rather than approximate.
    w = cluster.astype(float) * 10.0
    return w.reshape(-1, 1), w.copy(), cluster


def out_of_fold(design: np.ndarray, target: np.ndarray, folds: object) -> np.ndarray:
    predictions, _ = cross_fit_predictions(
        KNeighborsRegressor(n_neighbors=1),
        design,
        target,
        np.ones(N),
        folds,  # type: ignore[arg-type]
        task="regression",
        predict_designs={"observed": design},
    )
    return predictions["observed"]


class TestLeakageIsRealWhenFoldsIgnoreClusters:
    def test_an_ungrouped_split_reproduces_every_held_out_outcome_exactly(self) -> None:
        design, target, cluster = leaky_sample()
        folds = make_folds(N, 5, random_state=0)
        assert_every_row_has_a_same_cluster_neighbour(folds, cluster)
        # Every row's nearest training neighbour is another row of its own cluster, which
        # carries an identical outcome. The "out-of-fold" prediction is therefore the
        # held-out value itself -- perfect, and entirely spurious.
        np.testing.assert_array_equal(out_of_fold(design, target, folds), target)

    def test_a_grouped_split_cannot_reproduce_any_of_them(self) -> None:
        design, target, cluster = leaky_sample()
        folds = make_folds(N, 5, cluster=cluster, random_state=0)
        # No same-cluster row is available to train on, so the nearest neighbour belongs
        # to another cluster and carries a different outcome by construction.
        assert np.all(out_of_fold(design, target, folds) != target)

    def test_the_grouped_split_is_what_makes_the_difference(self) -> None:
        # The two runs differ in the fold assignment and in nothing else: same learner,
        # same design, same target, same seed, same fold count.
        design, target, cluster = leaky_sample()
        ungrouped = make_folds(N, 5, random_state=0)
        assert_every_row_has_a_same_cluster_neighbour(ungrouped, cluster)
        leaked = out_of_fold(design, target, ungrouped)
        honest = out_of_fold(design, target, make_folds(N, 5, cluster=cluster, random_state=0))
        assert np.abs(leaked - target).max() == 0.0
        assert np.abs(honest - target).min() > 0.0


class TestTheClusterCodesReachTheSplitter:
    """The wiring, end to end: ``id=`` has to arrive at the fold builder to matter."""

    def test_a_clustered_fit_keeps_every_cluster_in_one_fold(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        rng = np.random.default_rng(0)
        cluster = np.repeat(np.arange(N_CLUSTERS), PER_CLUSTER)
        w = rng.normal(size=N)
        a = rng.binomial(1, 0.5, N).astype(float)
        frame = pd.DataFrame(
            {"Y": w + a + rng.normal(scale=0.5, size=N), "A": a, "W": w, "pid": cluster}
        )
        result = (
            fast_tmle(n_folds=3)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], id="pid")
            .single()
        )
        assignment = result.nuisance.folds.assignment
        for code in np.unique(cluster):
            assert np.unique(assignment[cluster == code]).size == 1
