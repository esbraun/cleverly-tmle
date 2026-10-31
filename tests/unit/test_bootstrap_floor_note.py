"""The below-floor bootstrap note names the floor of the fit's own kind.

A point fit is held to 10 clusters and a longitudinal fit to 20. The note once formatted the
point floor for both, so a longitudinal fit at 15 clusters printed "fewer than 10 clusters".
"""

from __future__ import annotations

import warnings
from typing import Any

from cleverly._inference_status import (
    MINIMUM_INTERVAL_CLUSTERS,
    MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
)
from cleverly.datasets import make_longitudinal
from tests.unit.test_ltmle_bootstrap import _end_columns, _estimator
from tests.unit.test_support_review_fixes import drtmle_estimated_weights


def _longitudinal(clusters: int) -> Any:
    data, _ = make_longitudinal(n=10 * clusters, seed=1, cluster_size=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(n_bootstrap=3, simultaneous=False).fit(data, id="id", **_end_columns())


def test_a_longitudinal_fit_at_fifteen_clusters_names_the_floor_of_twenty() -> None:
    estimate = next(iter(_longitudinal(15).estimates.values()))
    summary = estimate.bootstrap
    assert summary is not None and summary.below_floor
    assert summary.cluster_floor == MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS
    text = str(summary.summary())
    assert f"fewer than {MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS} clusters" in text
    assert f"fewer than {MINIMUM_INTERVAL_CLUSTERS} clusters" not in text


def test_a_point_fit_below_its_floor_names_the_floor_of_ten() -> None:
    """The control: the point floor is still 10, in the per-row and the fit summary."""
    result = drtmle_estimated_weights(7)
    summary = result["ate"].bootstrap
    assert summary is not None and summary.below_floor
    assert summary.cluster_floor == MINIMUM_INTERVAL_CLUSTERS
    for text in (str(summary.summary()), str(result.summary())):
        assert f"fewer than {MINIMUM_INTERVAL_CLUSTERS} clusters" in text


def test_a_fit_above_the_floor_carries_no_floor() -> None:
    estimate = next(iter(_longitudinal(20).estimates.values()))
    assert estimate.bootstrap is not None
    assert not estimate.bootstrap.below_floor
    assert estimate.bootstrap.cluster_floor is None
