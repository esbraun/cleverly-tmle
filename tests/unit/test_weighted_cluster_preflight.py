"""A zero-mass cluster cannot supply the second total in a validation fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import SplitPlan
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.learners import random_partition
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("weighted_cluster_preflight")


def weighted_frame() -> pd.DataFrame:
    """Each of twelve clusters has both treatments and both outcomes."""
    return pd.DataFrame(
        {
            "Y": np.tile([0, 1, 1, 0], 12),
            "A": np.tile([0, 1, 0, 1], 12),
            "W": np.tile([-1.0, 0.0, 1.0, 2.0], 12),
            "cluster": np.repeat(np.arange(12), 4),
            "w": np.ones(48),
        }
    )


SETTINGS = (
    pytest.param({"cv_evaluation": True}, id="fold-evaluation"),
    pytest.param({"targeting_scheme": "fold"}, id="fold-targeting"),
)


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("positive", [0, 1])
def test_supplied_fold_counts_positive_mass_before_any_learner(
    settings: dict[str, Any], positive: int
) -> None:
    frame = weighted_frame()
    cluster = frame["cluster"].to_numpy()
    draw = random_partition(len(frame), 6, cluster=cluster, seed=0)
    test = draw.test_index(0)
    labels = np.unique(cluster[test])
    assert labels.size == 2
    frame.loc[frame["cluster"].isin(labels[positive:]), "w"] = 0.0
    learners = never_fit_learners()
    with pytest.raises(CapabilityError) as caught:
        TMLE(
            **settings,
            **learners,
            n_folds=6,
            split_plan=SplitPlan.from_folds([draw]),
            estimands=("ey1",),
            simultaneous=False,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W"], id="cluster", weights="w")
    assert NeverFit.calls == 0
    assert "at least 2 clusters with positive weight mass" in str(caught.value)
    assert f"fold 0 holds {positive}." in str(caught.value)
    assert "puts" in str(caught.value)
    assert f"{10 + positive} clusters with positive weight mass" in str(caught.value)


@pytest.mark.parametrize("settings", SETTINGS)
def test_generated_fold_counts_positive_mass_before_any_learner(
    settings: dict[str, Any],
) -> None:
    frame = weighted_frame()
    frame.loc[frame["cluster"] != 0, "w"] = 0.0
    learners = never_fit_learners()
    with pytest.raises(CapabilityError, match="clusters with positive weight mass"):
        TMLE(
            **settings,
            **learners,
            n_folds=6,
            estimands=("ey1",),
            simultaneous=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W"], id="cluster", weights="w")
    assert NeverFit.calls == 0


@pytest.mark.parametrize("settings", SETTINGS)
def test_two_positive_mass_clusters_per_fold_pass_the_guard(
    settings: dict[str, Any],
) -> None:
    frame = weighted_frame()
    cluster = frame["cluster"].to_numpy()
    draw = random_partition(len(frame), 3, cluster=cluster, seed=0)
    for _, test in draw:
        labels = np.unique(cluster[test])
        frame.loc[frame["cluster"].isin(labels[2:]), "w"] = 0.0
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W"], id="cluster", weights="w"
    )
    # This only checks the preflight. Six clusters remain below the inference floor,
    # and no fitted or variance quantity changes in this repair.
    TMLE(**settings)._preflight_cluster_validation_folds(data, [draw])
