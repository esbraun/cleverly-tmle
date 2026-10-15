"""Every sequential estimator fits a repeated outcome by the reduction, or refuses it.

A subclass shares :meth:`LTMLE.fit`, so it would fit a repeated outcome by inheritance
whether or not its own estimator reduces per horizon.  ``LTMLE.fit`` refuses a repeated
container for any estimator kind outside ``REPEATED_OUTCOME_KINDS``, and the refusal names
the kinds that fit one.  ``SDR`` and ``IPSI`` subclass ``_SequentialEstimator`` and not
``LTMLE``, so this test enumerates ``_SequentialEstimator`` and every concrete subclass the
package defines.  For each one it requires either the refusal before any learner runs or, for
a kind in the allowlist, the reduction: the estimate and influence curve at each horizon equal
those of the same class fitted on the end-of-study panel truncated after that horizon.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Incremental
from cleverly.longitudinal import LTMLE, Repeated
from cleverly.longitudinal import estimator as sequential
from cleverly.longitudinal.estimator import _SequentialEstimator

from .test_repeated_outcome import panel

ROLES: dict[str, Any] = {
    "outcome": Repeated(["Y1", "Y2", "Y3"]),
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W"],
    "time_varying": [["L1"], ["L2"], ["L3"]],
    "censoring": ["C1", "C2", "C3"],
}

#: The message every refusal shares: it names the missing piece and the kinds that fit one.
REFUSAL = "has no per-horizon reduction for a repeated outcome"


def _every_subclass() -> list[type[_SequentialEstimator]]:
    """Every concrete subclass of ``_SequentialEstimator``, after importing the package."""
    for module in pkgutil.walk_packages(cleverly.__path__, prefix="cleverly."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # pragma: no cover - an optional extra is absent
            continue
    found: list[type[_SequentialEstimator]] = [_SequentialEstimator]
    index = 0
    while index < len(found):
        found.extend(
            child
            for child in found[index].__subclasses__()
            if child.__module__.startswith("cleverly") and child not in found
        )
        index += 1
    return [kind for kind in found if not inspect.isabstract(kind) and kind is not found[0]]


class _Untouchable(LogisticRegression):
    """A learner that fails the test if a refusing estimator ever fits it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner ran before the repeated-outcome refusal")


def _truncated_roles(horizon: int) -> dict[str, Any]:
    """The end-of-study roles of the panel truncated after ``horizon``."""
    nodes = range(1, horizon + 1)
    blocks = [[f"L{t}"] if t == 1 else [f"Y{t - 1}", f"L{t}"] for t in nodes]
    return {
        "outcome": f"Y{horizon}",
        "treatment": [f"A{t}" for t in nodes],
        "baseline": ["W"],
        "time_varying": blocks,
        "censoring": [f"C{t}" for t in nodes],
    }


def _plans(kind: type[_SequentialEstimator]) -> tuple[Any, str | None]:
    """The regimens a kind is handed, and its reference: tilts for ``IPSI``."""
    if getattr(kind, "_estimator", "tmle") == "ipsi":
        return [Incremental(0.5), Incremental(2.0)], None
    return {"never": 0, "always": 1}, "never"


def _fit(
    kind: type[_SequentialEstimator], learner: Any, roles: dict[str, Any] | None = None
) -> Any:
    regimens, reference = _plans(kind)
    # IPW fits no outcome regression and refuses an outcome learner when it is built.
    outcomes = (
        {}
        if getattr(kind, "_estimator", "tmle") == "ipw"
        else {"outcome_learner": learner, "pseudo_learner": LinearRegression()}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kind(
            regimens,
            reference=reference,
            **outcomes,
            treatment_learner=learner,
            censoring_learner=learner,
            n_folds=1,
            simultaneous=False,
            random_state=0,
        ).fit(panel(), **(ROLES if roles is None else roles))


def _assert_reduces(kind: type[_SequentialEstimator]) -> None:
    """W1 at one composition: each horizon equals the truncated end-of-study fit."""
    result = _fit(kind, LogisticRegression())
    labels = sorted(
        name[len("ey_regimen[") : -len(" @ t=1]")]
        for name in result
        if name.startswith("ey_regimen[") and name.endswith(" @ t=1]")
    )
    assert labels
    for horizon in (1, 2, 3):
        truncated = _fit(kind, LogisticRegression(), _truncated_roles(horizon))
        for label in labels:
            name = f"ey_regimen[{label} @ t={horizon}]"
            assert result.psi(name) == pytest.approx(
                truncated.psi(f"ey_regimen[{label}]"), rel=1e-10, abs=1e-12
            )
            np.testing.assert_allclose(
                result.influence_curves[name],
                truncated.influence_curves[f"ey_regimen[{label}]"],
                rtol=1e-10,
                atol=1e-12,
            )


def test_the_enumeration_reaches_every_sequential_estimator() -> None:
    """``SDR`` and ``IPSI`` are not ``LTMLE`` subclasses, so the walk starts above ``LTMLE``."""
    names = {kind.__name__ for kind in _every_subclass()}
    assert {"LTMLE", "SDR", "IPSI"} <= names
    assert not issubclass(cleverly.longitudinal.SDR, LTMLE)
    assert not issubclass(cleverly.longitudinal.IPSI, LTMLE)


@pytest.mark.parametrize("kind", _every_subclass(), ids=lambda kind: kind.__qualname__)
def test_each_estimator_reduces_or_refuses(kind: type[_SequentialEstimator]) -> None:
    if getattr(kind, "_estimator", "tmle") in sequential.REPEATED_OUTCOME_KINDS:
        _assert_reduces(kind)
        return
    with pytest.raises(CapabilityError, match=REFUSAL) as raised:
        _fit(kind, _Untouchable())
    assert str(raised.value).startswith(kind.__name__)
    for allowed in sequential.REPEATED_OUTCOME_KINDS:
        assert sequential._REPEATED_OUTCOME_NAMES[allowed] in str(raised.value)


@pytest.mark.parametrize("kind", _every_subclass(), ids=lambda kind: kind.__qualname__)
def test_an_allowlisted_kind_reduces(kind: type[_SequentialEstimator], monkeypatch: Any) -> None:
    """The allowlist is the only gate: add a kind to it and the fit reduces.

    This holds for every kind, which is why a later change that ships ``SDR`` or ``IPSI``
    edits that one line.  K1 (the fit key) is what lets the horizons of ``SDR`` and ``IPSI``
    stay apart.
    """
    monkeypatch.setattr(
        sequential,
        "REPEATED_OUTCOME_KINDS",
        frozenset({"tmle", getattr(kind, "_estimator", "tmle")}),
    )
    _assert_reduces(kind)


def test_the_refusal_bites_on_a_subclass_of_another_kind() -> None:
    """A control: a subclass declaring a kind outside the allowlist is refused first."""

    class _Other(LTMLE):
        _estimator = "x"

    with pytest.raises(CapabilityError, match=REFUSAL):
        _fit(_Other, _Untouchable())


def test_an_allowlist_that_accepts_every_kind_lets_a_foreign_kind_through(
    monkeypatch: Any,
) -> None:
    """The mutation control N2 in miniature: refusing nothing means the learner runs."""

    class _Other(LTMLE):
        _estimator = "x"

    class _Everything:
        def __contains__(self, kind: object) -> bool:
            return True

    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", _Everything())
    with pytest.raises(AssertionError, match="a learner ran before"):
        _fit(_Other, _Untouchable())


def test_the_refusal_names_the_kinds_of_the_allowlist(monkeypatch: Any) -> None:
    refusal = sequential.repeated_outcome_refusal("SDR")
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.message() == (
        "SDR has no per-horizon reduction for a repeated outcome (outcome=Repeated([...])). "
        "X32b in docs/roadmap.md tracks this. LTMLE fits one. Each horizon is the "
        "end-of-study fit of the panel truncated after it."
    )
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset({"tmle", "ipsi"}))
    assert "IPSI and LTMLE fit one." in sequential.repeated_outcome_refusal("SDR").message()
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset({"tmle", "ipsi", "sdr"}))
    assert "IPSI, LTMLE and SDR fit one." in sequential.repeated_outcome_refusal("X").message()
