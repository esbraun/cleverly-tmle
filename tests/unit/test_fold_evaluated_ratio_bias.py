r"""The fold-evaluated bias at informative cluster sizes comes from the law, not from a fit.

``clustered-few-cluster-tmle`` reads a bias of 0.61, 0.43 and 0.32 empirical standard
deviations for ``TMLE(cv_evaluation=True)`` at 10, 20 and 30 clusters of informative size in
5 folds. The stacked and in-sample fits of the same draws read about one fifth of it.

``cv_evaluation=True`` reports the equal :math:`1/V` average of the fold estimates. Each fold
estimate is a row mean over the :math:`J/V` clusters of its fold, which is a ratio of two
cluster sums. The ratio bias of a mean over :math:`J/V` clusters is of order :math:`V/J`, so the
equal fold average carries about :math:`V` times the :math:`O(1/J)` ratio bias of the pooled
row mean. At equal sizes every fold mean has the same denominator, and the bias vanishes.

This oracle fits nothing. Each cluster carries the exact conditional effect
:math:`m_1(N) - m_0(N)` of the study law, so the bias comes from the law alone, with no
learner, fluctuation or variance code involved. These tests back the refusal of
``cv_evaluation=True`` on such clusters (``fold_evaluated_cluster_refusal`` in
``cleverly.estimators.tmle``), which is why the study's informative fold-evaluated cells left
its grid.

The ATT has the same mechanism through another denominator: each fold estimate divides by the
treated count of its clusters. At equal cluster sizes with treated counts that differ and an
effect that rises with the treated count, the equal fold average again carries about :math:`V`
times the pooled bias, and equal treated counts remove it.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from scipy import stats

from tests.studies import clustered_unequal_laws as laws
from tests.studies.clustered_few_cluster_properties import size_law_parameters
from tests.studies.clustered_few_cluster_tmle import CLUSTER_COUNTS, N_FOLDS

DRAWS = 200_000
SEED = 20261003
SIZE_LAW = "unequal_informative"
#: The cluster size of the treated-count law, and the slope of its treated effect in the
#: treated count. Each cluster treats ``1 + Binomial(TREATED_TRIALS, 0.5)`` of its rows, so no
#: fold is left without a treated row.
CLUSTER_SIZE = 10
TREATED_TRIALS = 9
TREATED_SLOPE = 0.05


@cache
def oracle(clusters: int, sizes: str = SIZE_LAW) -> tuple[float, float, float, float]:
    """Pooled and equal-fold-average bias of the exact cluster effects, with their MC SEs."""
    params = size_law_parameters(sizes)
    m0, m1 = laws.size_means(params["delta"], params["gamma"], params["effect_modifier"])
    truth = laws.informative_truth(**params)["ate"]
    effect = m1 - m0
    rng = np.random.default_rng(SEED + clusters)
    if sizes == "equal10":
        size = np.full((DRAWS, clusters), 10)
    else:
        size = rng.integers(int(laws.SIZES.min()), int(laws.SIZES.max()) + 1, (DRAWS, clusters))
    total = size * effect[size - int(laws.SIZES.min())]
    pooled = total.sum(axis=1) / size.sum(axis=1)
    # Clusters are exchangeable, so dealing them into folds in index order is a random split.
    by_fold = (DRAWS, N_FOLDS, clusters // N_FOLDS)
    folds = (total.reshape(by_fold).sum(axis=2) / size.reshape(by_fold).sum(axis=2)).mean(axis=1)
    root = np.sqrt(DRAWS)
    return (
        float(pooled.mean() - truth),
        float(pooled.std(ddof=1) / root),
        float(folds.mean() - truth),
        float(folds.std(ddof=1) / root),
    )


@pytest.mark.parametrize("clusters", CLUSTER_COUNTS)
def test_the_equal_fold_average_carries_about_v_times_the_pooled_bias(clusters: int) -> None:
    pooled, pooled_se, folds, _ = oracle(clusters)
    assert pooled > 4 * pooled_se
    assert 0.5 * N_FOLDS < folds / pooled < 2 * N_FOLDS


@pytest.mark.parametrize("clusters", CLUSTER_COUNTS)
def test_equal_sizes_carry_no_fold_average_bias(clusters: int) -> None:
    """The deliberate-mutation control: equal sizes remove the denominator that varies."""
    _, _, bias, se = oracle(clusters, "equal10")
    assert abs(bias) <= 4 * se + 1e-12


@cache
def treated_oracle(clusters: int, equal_treated: bool = False) -> tuple[float, float, float, float]:
    r"""Pooled and equal-fold-average ATT bias at equal sizes, with their MC SEs.

    Cluster :math:`j` treats :math:`T_j` of its ``CLUSTER_SIZE`` rows, and every treated row
    of it carries the effect :math:`\theta(T_j) = 0.05\,T_j`. The ATT is
    :math:`E[T\theta(T)]/E[T]`, and each fold estimate is the ratio of the fold's treated
    effect sum to its treated count.
    """
    rng = np.random.default_rng(SEED + 1000 + clusters)
    if equal_treated:
        treated = np.full((DRAWS, clusters), 1 + TREATED_TRIALS // 2)
        counts = np.array([1 + TREATED_TRIALS // 2])
        pmf = np.array([1.0])
    else:
        treated = 1 + rng.binomial(TREATED_TRIALS, 0.5, (DRAWS, clusters))
        counts = 1 + np.arange(TREATED_TRIALS + 1)
        pmf = stats.binom.pmf(np.arange(TREATED_TRIALS + 1), TREATED_TRIALS, 0.5)
    assert int(treated.max()) <= CLUSTER_SIZE
    truth = float(np.sum(pmf * counts * TREATED_SLOPE * counts) / np.sum(pmf * counts))
    total = treated * TREATED_SLOPE * treated
    pooled = total.sum(axis=1) / treated.sum(axis=1)
    by_fold = (DRAWS, N_FOLDS, clusters // N_FOLDS)
    folds = (total.reshape(by_fold).sum(axis=2) / treated.reshape(by_fold).sum(axis=2)).mean(axis=1)
    root = np.sqrt(DRAWS)
    return (
        float(pooled.mean() - truth),
        float(pooled.std(ddof=1) / root),
        float(folds.mean() - truth),
        float(folds.std(ddof=1) / root),
    )


@pytest.mark.parametrize("clusters", CLUSTER_COUNTS)
def test_an_unequal_treated_count_carries_about_v_times_the_pooled_att_bias(
    clusters: int,
) -> None:
    pooled, pooled_se, folds, _ = treated_oracle(clusters)
    # The effect rises with the treated count, so the ratio bias is negative here.
    assert abs(pooled) > 4 * pooled_se
    assert 0.5 * N_FOLDS < folds / pooled < 2 * N_FOLDS


@pytest.mark.parametrize("clusters", CLUSTER_COUNTS)
def test_equal_treated_counts_carry_no_fold_average_att_bias(clusters: int) -> None:
    """The deliberate-mutation control: equal treated counts fix every fold's denominator."""
    pooled, _, bias, se = treated_oracle(clusters, equal_treated=True)
    assert abs(pooled) <= 1e-12
    assert abs(bias) <= 4 * se + 1e-12
