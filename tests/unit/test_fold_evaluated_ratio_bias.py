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
:math:`m_1(N) - m_0(N)` of the study law. The test compares the equal fold average of these
effects with the bias that the committed study artifact measures for the shipped fit. Agreement
shows that the law alone produces the measured bias, with no learner, fluctuation or variance
code involved.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from tests.studies import clustered_unequal_laws as laws
from tests.studies.clustered_few_cluster_properties import size_law_parameters
from tests.studies.clustered_few_cluster_tmle import CLUSTER_COUNTS, N_FOLDS, STUDY

DRAWS = 200_000
SEED = 20261003
SIZE_LAW = "unequal_informative"
#: The cells whose bias the oracle rebuilds: the fold-evaluated fit at informative sizes.
CELL = "tmle_cv_evaluation__unequal_informative__j{clusters}__t_reference"
#: Combined Monte Carlo standard errors the oracle may sit from the measured bias.
TOLERANCE_SE = 4.0


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


@cache
def measured() -> pd.DataFrame:
    return pd.read_csv(STUDY.artifact("properties.csv"), float_precision="round_trip").set_index(
        "cell"
    )


@pytest.mark.parametrize("clusters", CLUSTER_COUNTS)
def test_the_equal_fold_average_of_the_law_rebuilds_the_measured_bias(clusters: int) -> None:
    pooled, _, bias, se = oracle(clusters)
    row = measured().loc[CELL.format(clusters=clusters)]
    allowed = TOLERANCE_SE * float(np.hypot(se, row["bias_se"]))
    assert abs(bias - row["bias"]) <= allowed, (
        f"J={clusters}: the oracle predicts a fold-evaluated bias of {bias:.4f}, "
        f"the study measured {row['bias']:.4f}"
    )
    # The witness discriminates: the pooled row mean, which the stacked report follows,
    # does not rebuild the measured bias.
    assert abs(pooled - row["bias"]) > allowed


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
