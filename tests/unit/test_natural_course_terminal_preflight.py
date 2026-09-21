"""A natural-course outcome class must reach every outer training complement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import DataError
from cleverly.learners import random_partition
from tests import discrete_law_mar as law
from tests.unit._natural_course_support import stacked_tmle

_FIT_CALLS: list[int] = []


class CountingLogistic(LogisticRegression):
    """Count fits on sklearn clones of the outcome and response learners."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        _FIT_CALLS.append(len(y))
        return super().fit(X, y, sample_weight=sample_weight)


def test_rare_respondent_outcome_class_refuses_before_nuisance_fits() -> None:
    """Two positive respondents in one fold leave its complement with only zeroes."""
    _FIT_CALLS.clear()
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
    assert not _FIT_CALLS, f"{len(_FIT_CALLS)} learner fit(s) ran before the refusal"
