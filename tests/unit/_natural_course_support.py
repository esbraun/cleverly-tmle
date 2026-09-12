"""Shared builders for the in-sample and stacked natural-course TMLE unit tests."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from cleverly.estimators import TMLE
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome, fast_tmle
from tests.studies.missing_outcome_study_helpers import FailTreatment, NaturalCourseLaw


class NeverFit(BaseEstimator):
    """A learner whose fit is the failure a refusal must precede.

    ``calls`` is class-level, so it counts fits of every clone the estimator makes.
    """

    calls: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NeverFit:
        type(self).calls += 1
        raise AssertionError("a refusal or preflight must run before any learner is fitted")

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - fit must fail first
        raise AssertionError("a refusal or preflight must run before any prediction")


def never_fit_learners() -> dict[str, NeverFit]:
    """Every point-treatment learner role, each one a :class:`NeverFit`, with calls reset."""
    NeverFit.calls = 0
    return {
        "outcome_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "missingness_learner": NeverFit(),
        "intermediate_learner": NeverFit(),
    }


def in_sample_tmle(**overrides: Any) -> TMLE:
    """The ordinary in-sample natural-course TMLE."""
    return fast_tmle(**{"estimands": ("ey_obs",), "cross_fit": False, **overrides})


def stacked_tmle(**overrides: Any) -> TMLE:
    """The audited stacked cross-fitted natural-course CV-TMLE, with oracle nuisances."""
    settings: dict[str, Any] = {
        "outcome_learner": OracleOutcome(NaturalCourseLaw()),
        "treatment_learner": FailTreatment(),
        "missingness_learner": OracleMissingness(NaturalCourseLaw()),
        "estimands": ("ey_obs",),
        "cross_fit": True,
        "n_folds": 2,
        "repeats": 1,
        "stratify_folds": "none",
        "targeting_scheme": "pooled",
        "cv_evaluation": False,
        "random_state": 17,
        "max_iter": 100,
        "tol": 1e-12,
    }
    return fast_tmle(**{**settings, **overrides})


def exact_remainder(
    w: np.ndarray, a: np.ndarray, response: np.ndarray, targeted: np.ndarray
) -> float:
    r"""The exact MAR natural-course remainder over rows of :mod:`tests.discrete_law_mar`.

    .. math::

        R_2 = P[(1 - \pi_0 / \pi)(m^* - m_0)]

    ``response`` and ``targeted`` are the candidate :math:`\pi` and :math:`m^*` at each
    row's realized ``(w, a)``. The truth comes from the law's arrays alone.
    """
    return float(np.mean((1.0 - law.PI[w, a] / response) * (targeted - law.Q[w, a])))
