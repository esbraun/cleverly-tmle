"""The support label of an in-sample density-route shift reads whether its density is learned.

The shift studies behind T21 and T22 give the method the exact density of their law. A density
learned from the data adds error that the influence-curve standard error omits (X121: on the
interventions page's law, a regularized booster at n = 3,000 gave SE over SD 0.76 to 0.85), so a
learned density matches no row and reads experimental, with T21 named as the nearest row.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._support import density_learner_family
from cleverly.datasets import make_shift_dose
from cleverly.estimators import TMLE
from cleverly.interventions import Shift


class _KnownDensity(LogisticRegression):
    """A density that declares itself the law's exact density."""

    cleverly_density = "known"


class _OtherMarker(LogisticRegression):
    """A marker with another value is not a declaration of a known density."""

    cleverly_density = "approximate"


def _fit(treatment_learner: Any) -> Any:
    frame, _ = make_shift_dose(n=400, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(
            policies=[Shift(0.0, cap=None), Shift(0.25, cap=None)],
            cross_fit=False,
            ratio="density",
            density_bins=5,
            outcome_learner=LinearRegression(),
            treatment_learner=treatment_learner,
        ).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            treatment_kind="continuous",
        )


@pytest.mark.parametrize(
    ("learner", "family"),
    [
        (None, "adaptive"),
        (HistGradientBoostingClassifier(), "adaptive"),
        (LogisticRegression(), "parametric"),
        (_KnownDensity(), "known"),
        (_OtherMarker(), "adaptive"),
    ],
)
def test_the_density_family_reads_the_marker_then_the_learner_family(
    learner: Any, family: str
) -> None:
    assert density_learner_family(learner) == family


def test_a_known_density_reads_supported() -> None:
    frame = _fit(_KnownDensity(max_iter=1000)).to_frame()
    assert set(frame["support"]) == {"supported"}


@pytest.mark.parametrize(
    ("learner", "family"),
    [
        (HistGradientBoostingClassifier(max_depth=2, max_iter=50, random_state=0), "adaptive"),
        (LogisticRegression(max_iter=1000), "parametric"),
    ],
)
def test_a_learned_density_reads_experimental_beside_t21(learner: Any, family: str) -> None:
    frame = _fit(learner).to_frame()
    assert set(frame["support"]) == {"experimental"}
    reasons = set(frame["support_reason"])
    assert len(reasons) == 1
    (reason,) = reasons
    assert "Nearest row: T21" in reason
    assert f"density_learner_family={family!r}" in reason
