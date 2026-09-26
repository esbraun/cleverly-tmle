"""Terminal outcome support is checked before longitudinal nuisance fitting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from cleverly.datasets import make_longitudinal
from cleverly.exceptions import LongitudinalError
from cleverly.learners import random_partition
from cleverly.longitudinal import LTMLE
from tests.unit._natural_course_support import Counting, CountingLogistic


def test_terminal_outcome_shortfall_precedes_mechanism_fits() -> None:
    """Two events stranded in one fold leave no event in its training complement."""
    Counting.calls = 0
    frame, _ = make_longitudinal(n=400, seed=0)
    assignment = random_partition(len(frame), 3, seed=0).assignment
    followers = (
        frame["A1"].eq(1)
        & frame["A2"].eq(1)
        & frame["C1"].eq(1)
        & frame["C2"].eq(1)
        & (assignment == 0)
    )
    rare_rows = frame.index[followers][:2]
    assert len(rare_rows) == 2
    frame.loc[frame["Y"].notna(), "Y"] = 0.0
    frame.loc[rare_rows, "Y"] = 1.0

    estimator = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner=CountingLogistic(max_iter=1000),
        treatment_learner=CountingLogistic(max_iter=1000),
        censoring_learner=CountingLogistic(max_iter=1000),
        pseudo_learner=LinearRegression(),
        n_folds=3,
        random_state=0,
    )
    with np.testing.assert_raises_regex(
        LongitudinalError,
        "regimen 'always' through time 2 in outer training fold 1 has the same outcome",
    ):
        estimator.fit(
            frame,
            baseline=["W1", "W2"],
            treatment=["A1", "A2"],
            censoring=["C1", "C2"],
            time_varying=[[], ["L2"]],
            outcome="Y",
        )
    assert Counting.calls == 0, f"{Counting.calls} learner fit(s) ran before the refusal"
