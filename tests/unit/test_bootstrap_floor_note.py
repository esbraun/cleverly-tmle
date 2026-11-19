"""The below-floor bootstrap note names the floor of the fit's own family.

A point fit no floor study measures is held to 10 clusters, and an in-sample ``LTMLE`` fit to
its measured floor of 6. The note once formatted the point floor for both, so a longitudinal fit
below its own floor printed "fewer than 10 clusters".
"""

from __future__ import annotations

import warnings
from typing import Any

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS, DEFAULT_POINT_INTERVAL_CLUSTERS
from cleverly.datasets import make_longitudinal
from tests.unit.test_ltmle_bootstrap import _end_columns, _estimator
from tests.unit.test_support_review_fixes import drtmle_estimated_weights


def _longitudinal(clusters: int) -> Any:
    data, _ = make_longitudinal(n=10 * clusters, seed=1, cluster_size=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(n_bootstrap=3, simultaneous=False).fit(data, id="id", **_end_columns())


def test_a_longitudinal_fit_below_its_floor_names_its_own_floor() -> None:
    floor = CLUSTER_INTERVAL_FLOORS["ltmle_in_sample"]
    estimate = next(iter(_longitudinal(floor - 1).estimates.values()))
    summary = estimate.bootstrap
    assert summary is not None and summary.below_floor
    assert summary.cluster_floor == floor
    text = str(summary.summary())
    assert f"fewer than {floor} clusters" in text
    assert f"fewer than {DEFAULT_POINT_INTERVAL_CLUSTERS} clusters" not in text


def test_a_point_fit_below_its_floor_names_the_floor_of_ten() -> None:
    """The control: the point floor is still 10, in the per-row and the fit summary."""
    result = drtmle_estimated_weights(7)
    summary = result["ate"].bootstrap
    assert summary is not None and summary.below_floor
    assert summary.cluster_floor == DEFAULT_POINT_INTERVAL_CLUSTERS
    for text in (str(summary.summary()), str(result.summary())):
        assert f"fewer than {DEFAULT_POINT_INTERVAL_CLUSTERS} clusters" in text


def test_a_fit_above_the_floor_carries_no_floor() -> None:
    estimate = next(iter(_longitudinal(20).estimates.values()))
    assert estimate.bootstrap is not None
    assert not estimate.bootstrap.below_floor
    assert estimate.bootstrap.cluster_floor is None
