"""Fix-15: a negative-control reading tests no effect under the control fit's own reference.

The reading took a normal p-value from ``psi / plugin SE`` beside a clustered fit whose
interval reads a Student t with ``J - 2`` degrees of freedom. It now reads the control
refit's own Wald test, so the reference, the interval scale and any stopping-rule margin
are the ones its interval uses.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.validation import refute
from tests.conftest import linear_in_sample


def _reading(n: int, cluster_size: int) -> object:
    frame, _ = make_clustered(n=n, cluster_size=cluster_size, seed=7)
    fit = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A", id="cluster").single()
    control = np.random.default_rng(3).normal(size=len(frame))
    report = refute(
        fit,
        estimand="ate",
        tests=("negative_control_outcome",),
        negative_control_outcome=control,
        random_state=0,
    )
    return report["negative_control_outcome"]


def test_a_few_cluster_control_reads_the_t_reference() -> None:
    reading = _reading(360, 30)  # 12 clusters: t with 10 df
    z = reading.statistic
    np.testing.assert_allclose(reading.statistic_pvalue, 2.0 * stats.t.sf(abs(z), 10), rtol=1e-10)
    # Mutation control: the normal p-value the reading took before is smaller.
    assert reading.statistic_pvalue > 2.0 * stats.norm.sf(abs(z)) + 1e-4


def test_the_normal_reference_holds_at_the_threshold() -> None:
    reading = _reading(400, 10)  # 40 clusters
    z = reading.statistic
    np.testing.assert_allclose(reading.statistic_pvalue, 2.0 * stats.norm.sf(abs(z)), rtol=1e-10)
