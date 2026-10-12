"""Fit failures that one sample causes raise ``DataError``, so a study records them (RM39b).

A repeated-sampling study records a ``DataError`` as one failed replicate and stops on a plain
``RuntimeError`` that ``cleverly`` raised.  These two failures are decided by the sample, not by
the configuration, so they take the recorded class.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from cleverly.datasets import make_binary_outcome
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.learners import SuperLearner
from cleverly.validation._failures import recordable
from tests.conftest import FAST_KWARGS


class _Refuses(RegressorMixin, BaseEstimator):
    def fit(self, x, y, sample_weight=None):  # type: ignore[no-untyped-def]
        raise np.linalg.LinAlgError("singular on this sample")

    def predict(self, x):  # type: ignore[no-untyped-def]
        return np.zeros(len(x))


def test_a_library_whose_every_learner_failed_raises_a_recordable_data_error() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 2)), rng.normal(size=60)
    with (
        pytest.warns(UserWarning, match="dropping learner"),
        pytest.raises(DataError, match="every learner in the library failed") as raised,
    ):
        SuperLearner(library=[_Refuses(), _Refuses()]).fit(x, y)
    assert recordable(raised.value)


def test_a_collapsed_fold_evaluation_raises_a_recordable_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The split that collapses to one realised fold needs a pathological sample; the patch
    # stands in for it so the test pins the class this path raises.
    monkeypatch.setattr(TMLE, "_cv_detail", lambda self, details, *, cluster: None)
    frame = make_binary_outcome(n=200, seed=3)[0]
    with pytest.raises(DataError, match="collapsed to one") as raised:
        TMLE(**{**FAST_KWARGS, "cv_evaluation": True, "estimands": ("ate",)}).fit(
            frame, outcome="Y", treatment="A"
        )
    assert recordable(raised.value)
