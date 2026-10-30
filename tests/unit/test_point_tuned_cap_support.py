"""A point fit with a cap tuned on the data reads ``experimental``.

Row ``T21`` (``shift-policies``) covers a shift with a declared cap on the density route. Its
study never fits a tuned cap, and the tuned-cap study ``mtp-tuning`` is longitudinal, so no point
row covers ``Shift(cap=DoseQuantile(...))``.

- Witness: the same shift with a declared cap reads ``supported``, so the tuned fit differs from
  a covered fit only in its cap.
- The tuned fit's key differs from the declared fit's key on the ``options`` axis alone, by
  ``tuned_cap``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression

import cleverly._support as support
from cleverly.datasets import make_shift_dose
from cleverly.estimators import TMLE
from cleverly.interventions import DoseQuantile, Shift

DECLARED = Shift(0.5, cap=5.0)
TUNED = Shift(0.5, cap=DoseQuantile(0.9), target="population")


def _fit(policy: Any, keys: list[Any]) -> Any:
    """Fit one shift, and record each point key the classifier reads."""
    frame, _ = make_shift_dose(n=400, seed=0)
    real = support.point_key

    def spy(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        keys.append(found[0])
        return found

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        patch.setattr(support, "point_key", spy)
        warnings.simplefilter("ignore")
        return TMLE(
            policies=[policy], cross_fit=False, ratio="density", outcome_learner=LinearRegression()
        ).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            treatment_kind="continuous",
        )


@pytest.fixture(scope="module")
def fits() -> dict[str, tuple[Any, list[Any]]]:
    found: dict[str, tuple[Any, list[Any]]] = {}
    for name, policy in (("declared", DECLARED), ("tuned", TUNED)):
        keys: list[Any] = []
        found[name] = (_fit(policy, keys), keys)
    return found


def test_a_declared_cap_reads_supported(fits: dict[str, Any]) -> None:
    assert set(fits["declared"][0].to_frame()["support"]) == {"supported"}


def test_a_tuned_cap_reads_experimental(fits: dict[str, Any]) -> None:
    assert set(fits["tuned"][0].to_frame()["support"]) == {"experimental"}


def test_the_keys_differ_only_by_the_tuned_cap_option(fits: dict[str, Any]) -> None:
    declared, tuned = fits["declared"][1][0], fits["tuned"][1][0]
    assert tuned.differing_axes(declared) == ("options",)
    assert ("tuned_cap", True) in tuned.options
    assert all(name != "tuned_cap" for name, _ in declared.options)
