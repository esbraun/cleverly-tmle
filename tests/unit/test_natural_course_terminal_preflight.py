"""A natural-course outcome class must reach every outer training complement."""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.exceptions import DataError
from cleverly.learners import random_partition
from tests import discrete_law_mar as law
from tests.unit._natural_course_support import Counting, CountingLogistic, stacked_tmle


def test_rare_respondent_outcome_class_refuses_before_nuisance_fits() -> None:
    """Two positive respondents in one fold leave its complement with only zeroes."""
    Counting.calls = 0
    frame = law.frame()
    assignment = random_partition(len(frame), 2, seed=17).assignment
    rare_rows = np.flatnonzero(frame["Delta"].eq(1).to_numpy() & (assignment == 0))[:2]
    assert rare_rows.size == 2
    frame.loc[frame["Delta"].eq(1), "Y"] = 0.0
    frame.loc[rare_rows, "Y"] = 1.0

    estimator = stacked_tmle(
        outcome_learner=CountingLogistic(max_iter=1000),
        missingness_learner=CountingLogistic(max_iter=1000),
    )
    with pytest.raises(
        DataError, match="training complement contains no respondent with outcome 1"
    ):
        estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
    assert Counting.calls == 0, f"{Counting.calls} learner fit(s) ran before the refusal"
