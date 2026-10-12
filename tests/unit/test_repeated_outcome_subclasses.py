"""Every ``LTMLE`` subclass fits a repeated outcome by the reduction, or refuses it.

A subclass shares :meth:`LTMLE.fit`, so it would fit a repeated outcome by inheritance
whether or not its own estimator reduces per horizon.  ``LTMLE.fit`` refuses a repeated
container for any estimator kind other than the TMLE, naming roadmap row X32b.  This test
enumerates ``LTMLE`` and every subclass the package defines and requires, for each one,
either the TMLE's per-horizon parameters or that refusal before any learner runs.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import LTMLE, Repeated

from .test_repeated_outcome import panel

ROLES: dict[str, Any] = {
    "outcome": Repeated(["Y1", "Y2", "Y3"]),
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W"],
    "time_varying": [["L1"], ["L2"], ["L3"]],
    "censoring": ["C1", "C2", "C3"],
}


def _every_subclass() -> list[type[LTMLE]]:
    """``LTMLE`` and its subclasses, after importing every module of the package."""
    for module in pkgutil.walk_packages(cleverly.__path__, prefix="cleverly."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # pragma: no cover - an optional extra is absent
            continue
    found: list[type[LTMLE]] = [LTMLE]
    index = 0
    while index < len(found):
        found.extend(
            child
            for child in found[index].__subclasses__()
            if child.__module__.startswith("cleverly")
        )
        index += 1
    return found


class _Untouchable(LogisticRegression):
    """A learner that fails the test if a refusing estimator ever fits it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner ran before the repeated-outcome refusal")


def _fit(kind: type[LTMLE], learner: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kind(
            {"never": 0, "always": 1},
            reference="never",
            outcome_learner=learner,
            pseudo_learner=LinearRegression(),
            treatment_learner=learner,
            censoring_learner=learner,
            n_folds=1,
            simultaneous=False,
        ).fit(panel(), **ROLES)


@pytest.mark.parametrize("kind", _every_subclass(), ids=lambda kind: kind.__qualname__)
def test_each_estimator_reduces_or_refuses(kind: type[LTMLE]) -> None:
    if getattr(kind, "_estimator", "tmle") == "tmle":
        result = _fit(kind, LogisticRegression())
        assert {f"ey_regimen[always @ t={h}]" for h in (1, 2, 3)} <= set(result)
        return
    with pytest.raises(CapabilityError, match="X32b"):
        _fit(kind, _Untouchable())


def test_the_refusal_bites_on_a_subclass_of_another_kind() -> None:
    """A control: a subclass declaring another estimator kind is refused before a learner."""

    class _Other(LTMLE):
        _estimator = "sdr"

    with pytest.raises(CapabilityError, match="X32b"):
        _fit(_Other, _Untouchable())
