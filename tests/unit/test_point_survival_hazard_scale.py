"""The ``hazard_scale=`` knob of ``make_point_survival`` (X14 F11).

With the default ``None`` every draw is the draw X13's studies ran on, which the pinned digests
below record.  The digests were computed from the module as merged before the knob existed.
With a factor, the knob lowers the all-cause hazard at its node in the draws, the exact truth
and the efficient influence function alike.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_point_survival
from cleverly.datasets.survival_point import point_survival_truth
from tests.studies.point_survival_common import SURVIVAL, Target, efficiency_sd, static

#: ``make_point_survival(n=500, seed=7, **keywords)`` before the knob existed.
PINNED = {
    "survival": ({}, "2dab53c1e657da50"),
    "competing": ({"causes": 2, "n_times": 4}, "115cb7bab4387480"),
    "three arms, wide": (
        {"layout": "wide", "time_varying": True, "arms": 3, "n_times": 3},
        "2117fcc8cccf8902",
    ),
    "clustered": ({"cluster_size": 80}, "5d824b4509492847"),
}


def _digest(frame: Any) -> str:
    hasher = hashlib.sha256()
    for column in frame.columns:
        hasher.update(column.encode())
        hasher.update(np.ascontiguousarray(frame[column].to_numpy(dtype=float)).tobytes())
    return hasher.hexdigest()[:16]


@pytest.mark.parametrize("name", list(PINNED))
def test_no_scale_leaves_every_draw_bit_identical(name: str) -> None:
    keywords, digest = PINNED[name]
    frame, truth = make_point_survival(n=500, seed=7, **keywords)
    assert _digest(frame) == digest
    scaled, same = make_point_survival(n=500, seed=7, hazard_scale=None, **keywords)
    assert _digest(scaled) == digest
    assert same == truth


def test_a_scale_lowers_the_events_at_its_node_and_nowhere_before() -> None:
    """The same uniforms decide every event, so a scaled event is an unscaled one."""
    plain, _ = make_point_survival(n=4000, seed=3, layout="wide")
    scaled, _ = make_point_survival(n=4000, seed=3, layout="wide", hazard_scale={3: 0.4})
    for name in ("C1", "Y1", "C2", "Y2", "C3"):
        assert np.array_equal(plain[name].to_numpy(), scaled[name].to_numpy(), equal_nan=True)
    before = plain["Y3"].to_numpy() == 1.0
    after = scaled["Y3"].to_numpy() == 1.0
    assert np.all(before[after])
    assert after.sum() < 0.6 * before.sum()


def test_the_scale_enters_the_truth_at_its_node() -> None:
    plain = point_survival_truth(static(1, (0, 1)))
    scaled = point_survival_truth(static(1, (0, 1)), hazard_scale={3: 0.4})
    np.testing.assert_array_equal(plain[:2], scaled[:2])
    increment = np.diff(plain[:, 0])[1]
    assert np.diff(scaled[:, 0])[1] == pytest.approx(0.4 * increment, rel=1e-12)
    assert np.all(scaled[2:] < plain[2:])


def test_the_scale_enters_the_efficient_influence_function() -> None:
    from dataclasses import replace

    target = (Target(1.0, "static", 1, 3),)
    plain = efficiency_sd(SURVIVAL, target)
    scaled = efficiency_sd(replace(SURVIVAL, hazard_scale=((3, 0.4),)), target)
    assert scaled != plain


@pytest.mark.parametrize("scale", [{0: 0.5}, {6: 0.5}, {3: 0.0}, {3: 1.5}])
def test_a_scale_outside_the_nodes_or_the_unit_interval_is_refused(scale: dict) -> None:
    with pytest.raises(ValueError, match="hazard_scale maps a node"):
        make_point_survival(n=10, seed=0, hazard_scale=scale)


def test_a_scale_on_a_continuous_event_time_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a discrete event time"):
        make_point_survival(
            n=10, seed=0, n_times=4, continuous_event_time=True, hazard_scale={2: 0.5}
        )
