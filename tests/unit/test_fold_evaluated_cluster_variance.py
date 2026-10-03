"""The fold-evaluated CV-TMLE variance centres the cluster totals inside each fold.

``cv_evaluation=True`` averages fold estimates, and each fold curve is centred at the fold's
own plug-in. Inside a fold the plug-in part therefore sums to zero. An uncentred second moment
of the fold's cluster totals keeps only ``(J_v - 1) / J_v`` of that part's between-cluster
variance, where ``J_v`` counts the clusters of fold ``v``. With rows as units ``J_v = n_v`` and
the loss is negligible. With four clusters in a fold, a quarter of it is lost.

:func:`~cleverly.inference.cross_validated_variance` now takes, for each fold, the centred
``ddof=1`` variance of the cluster totals, which is
:func:`~cleverly.inference.influence_variance` on the fold rows. The row branch keeps Zheng and
van der Laan's uncentred second moment, bit for bit. A validation fold with fewer than two
clusters has no such variance, and the fit refuses it before any learner.

Mutation controls, each proved to fail the named test:

* restore the uncentred cluster sum in ``cross_validated_variance``:
  :class:`TestTheFoldEvaluatedClusterVariance` and :class:`TestTheFitVarianceCalibrates`;
* drop the fit's fold check: :class:`TestOneClusterFoldsAreRefused` reads a bare ``ValueError``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import SplitPlan
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference import cross_validated_variance
from cleverly.learners import random_partition
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("fold_evaluated_cluster_variance")

#: The estimator module. The package re-exports a function named ``tmle``.
tmle_module = importlib.import_module("cleverly.estimators.tmle")


def fold_centred_table(clusters_per_fold: int, n_folds: int = 3) -> tuple[Any, Any, Any]:
    """A curve whose cluster totals sum to zero inside each fold, as a fold plug-in part does.

    Cluster ``j`` holds ``1 + j % 3`` rows, so the folds also differ in row count.
    """
    rng = np.random.default_rng(20261003 + clusters_per_fold)
    n_clusters = clusters_per_fold * n_folds
    sizes = 1 + np.arange(n_clusters) % 3
    cluster = np.repeat(np.arange(n_clusters), sizes)
    fold_of_cluster = np.arange(n_clusters) % n_folds
    level = rng.normal(size=n_clusters)
    for fold in range(n_folds):
        members = fold_of_cluster == fold
        # Centre the cluster totals of the fold: total_j = level_j * size_j.
        totals = level[members] * sizes[members]
        level[members] = (totals - totals.mean()) / sizes[members]
    curve = level[cluster]
    indices = [np.flatnonzero(fold_of_cluster[cluster] == fold) for fold in range(n_folds)]
    return curve, indices, cluster


def by_hand(curve: Any, indices: Any, cluster: Any, *, centred: bool) -> float:
    """The fold-averaged cluster variance, from a longhand loop over clusters."""
    total = 0.0
    for index in indices:
        codes = cluster[index]
        sums = np.array([curve[index][codes == code].sum() for code in np.unique(codes)])
        if centred:
            total += sums.size * float(np.var(sums, ddof=1)) / index.size**2
        else:
            total += float(np.sum(sums**2)) / index.size**2
    return total / len(indices) ** 2


class TestTheFoldEvaluatedClusterVariance:
    def test_it_is_the_centred_cluster_variance_of_each_fold(self) -> None:
        curve, indices, cluster = fold_centred_table(4)
        assert cross_validated_variance(curve, indices, cluster) == pytest.approx(
            by_hand(curve, indices, cluster, centred=True), rel=1e-12
        )

    def test_it_exceeds_the_uncentred_rule_by_the_lost_cluster_share(self) -> None:
        """Nonzero witness: the gap is ``J_v / (J_v - 1)`` on a fold-centred curve."""
        for clusters_per_fold, share in ((4, 4 / 3), (12, 12 / 11)):
            curve, indices, cluster = fold_centred_table(clusters_per_fold)
            new = cross_validated_variance(curve, indices, cluster)
            old = by_hand(curve, indices, cluster, centred=False)
            assert new / old == pytest.approx(share, rel=1e-12)
        curve, indices, cluster = fold_centred_table(4)
        assert cross_validated_variance(curve, indices, cluster) > 1.2 * by_hand(
            curve, indices, cluster, centred=False
        )

    def test_a_fold_of_one_cluster_raises_with_the_fact(self) -> None:
        curve = np.arange(8, dtype=float)
        cluster = np.repeat(np.arange(4), 2)
        indices = [np.arange(6), np.arange(6, 8)]
        with pytest.raises(ValueError, match="validation fold 1 holds 1 cluster"):
            cross_validated_variance(curve, indices, cluster)


class TestTheRowBranchIsUnchanged:
    """``cluster=None`` keeps the pre-fix arithmetic, pinned as hexadecimal floats."""

    @pytest.mark.parametrize(
        ("seed", "n", "n_folds", "pinned"),
        [
            (101, 30, 3, "0x1.2bd119d11f00ep-5"),
            (102, 47, 4, "0x1.156a92641887ap-5"),
            (103, 100, 10, "0x1.546e2e392db98p-7"),
        ],
    )
    def test_the_row_variance_is_bit_identical(
        self, seed: int, n: int, n_folds: int, pinned: str
    ) -> None:
        curve = np.random.default_rng(seed).normal(size=n) + 0.2
        indices = [np.arange(n)[k::n_folds] for k in range(n_folds)]
        assert cross_validated_variance(curve, indices) == float.fromhex(pinned)


def clustered_frame(n_clusters: int, rows: int = 5, seed: int = 0) -> pd.DataFrame:
    """A binary outcome in ``n_clusters`` clusters of ``rows`` rows, with a shared covariate."""
    rng = np.random.default_rng(seed)
    w1 = np.repeat(rng.normal(size=n_clusters), rows)
    w2 = rng.normal(size=n_clusters * rows)
    a = rng.binomial(1, expit(0.3 * w1 + 0.3 * w2))
    y = rng.binomial(1, expit(-0.2 + 0.5 * a + 2.0 * w1 + 0.3 * w2))
    return pd.DataFrame(
        {"Y": y, "A": a, "W1": w1, "W2": w2, "cluster": np.repeat(np.arange(n_clusters), rows)}
    )


def fold_evaluated(**settings: Any) -> TMLE:
    defaults: dict[str, Any] = {
        "cv_evaluation": True,
        "simultaneous": False,
        "estimands": ("ey1",),
        "random_state": 0,
    }
    return TMLE(**{**defaults, **settings})


class TestOneClusterFoldsAreRefused:
    TEXT = "cv_evaluation=True needs at least 2 clusters in every validation fold"

    @pytest.mark.parametrize(("n_clusters", "n_folds", "fold_size"), [(6, 6, 1), (4, 5, 1)])
    def test_a_generated_split_is_refused_before_any_learner(
        self, n_clusters: int, n_folds: int, fold_size: int
    ) -> None:
        frame = clustered_frame(n_clusters)
        learners = never_fit_learners()
        with warnings.catch_warnings(), pytest.raises(CapabilityError) as caught:
            # Five folds over four clusters warn that the count is capped at four.
            warnings.simplefilter("ignore", UserWarning)
            fold_evaluated(n_folds=n_folds, **learners).fit(
                frame, outcome="Y", treatment="A", id="cluster"
            )
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert message.startswith(self.TEXT)
        assert f"holds {fold_size}." in message
        assert f"Request at most {n_clusters // 2} folds" in message

    def test_a_supplied_plan_is_checked_fold_by_fold(self) -> None:
        frame = clustered_frame(7)
        draw = random_partition(len(frame), 4, cluster=frame["cluster"].to_numpy(), seed=0)
        counts = [np.unique(frame["cluster"].to_numpy()[test]).size for _, test in draw]
        assert sorted(counts) == [1, 2, 2, 2]
        learners = never_fit_learners()
        with pytest.raises(CapabilityError, match="puts 7 clusters into 4 folds") as caught:
            fold_evaluated(n_folds=4, split_plan=SplitPlan.from_folds([draw]), **learners).fit(
                frame, outcome="Y", treatment="A", id="cluster"
            )
        assert NeverFit.calls == 0
        assert f"fold {counts.index(1)} holds 1." in str(caught.value)

    def test_two_clusters_per_fold_fit(self) -> None:
        estimate = (
            fold_evaluated(
                n_folds=3,
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
            )
            .fit(
                clustered_frame(6, rows=20),
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                id="cluster",
            )
            .single()["ey1"]
        )
        assert np.isfinite(estimate.plugin_std_error) and estimate.plugin_std_error > 0


class TestTheFitVarianceCalibrates:
    """The cluster-level-covariate law of the review probe: 40 clusters of 30 rows, 10 folds.

    ``W1`` is shared within a cluster, so the plug-in part carries the cluster effect. The
    mean fold-evaluated standard error over the empirical SD of 60 seeded fits reads 1.015 under
    the fixed rule and 0.895 under the uncentred one (measured when this test was written).
    The planning probe measured 1.018 and 0.900 on 300 draws.
    """

    REPLICATES = 60

    @pytest.fixture(scope="class")
    def ratios(self) -> tuple[float, float]:
        seen: list[tuple[float, float]] = []
        original = tmle_module.cross_validated_variance

        def recording(curve: Any, folds: Any, cluster: Any = None) -> float:
            folds = [np.asarray(fold) for fold in folds]
            new = original(curve, folds, cluster)
            seen.append(
                (by_hand(np.asarray(curve), folds, np.asarray(cluster), centred=False), new)
            )
            return new

        points = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(tmle_module, "cross_validated_variance", recording)
            for seed in range(self.REPLICATES):
                rng = np.random.default_rng(seed)
                frame = clustered_frame(40, rows=30, seed=int(rng.integers(2**31)))
                estimate = (
                    fold_evaluated(
                        n_folds=10,
                        random_state=seed,
                        outcome_learner=LogisticRegression(C=1e6, max_iter=1000),
                        treatment_learner=LogisticRegression(C=1e6, max_iter=1000),
                    )
                    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
                    .single()["ey1"]
                )
                points.append(estimate.psi)
        assert len(seen) == self.REPLICATES
        sd = float(np.std(points, ddof=1))
        old = float(np.mean(np.sqrt([pair[0] for pair in seen])))
        new = float(np.mean(np.sqrt([pair[1] for pair in seen])))
        return old / sd, new / sd

    def test_the_fixed_rule_calibrates(self, ratios: tuple[float, float]) -> None:
        assert 0.85 < ratios[1] < 1.20

    def test_the_fixed_rule_restores_the_lost_share_on_the_same_fits(
        self, ratios: tuple[float, float]
    ) -> None:
        """Mutation control: the uncentred rule gives a ratio of one. Measured: 1.134."""
        assert ratios[1] / ratios[0] > 1.08

    def test_the_uncentred_rule_reads_low_on_the_same_fits(
        self, ratios: tuple[float, float]
    ) -> None:
        assert ratios[0] < 0.92
