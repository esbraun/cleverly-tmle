"""Shared scaffolding for the tests of a declared known function.

``tests/unit/test_msm_projection_weights.py`` (RM13) and
``tests/unit/test_stochastic_regime_densities.py`` (RM25) test the two users of
:class:`cleverly._declarations.FunctionDeclaration`.  Each file declares its own builders
and witnesses.  This module holds the parts that do not depend on which field is declared.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest

from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity.positivity import truncation_curve
from tests.unit._natural_course_support import NeverFit

#: ``cleverly.estimators`` exports a function named ``tmle``, which shadows the module.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: A fragment of every refusal of ``"estimated"``: the term the reported curve omits.
PATHWISE = "pathwise derivative"


def assert_refused(build: Callable[[], Any], error: type[Exception], *fragments: str) -> None:
    """``build()`` raises ``error`` with every fragment, or this raises ``AssertionError``.

    Written out rather than as ``pytest.raises``, whose failure is not an
    ``AssertionError``, so that a mutation control can require it to fail.
    """
    try:
        build()
    except error as raised:
        message = str(raised)
        for fragment in fragments:
            assert fragment in message, message
        return
    raise AssertionError(f"no {error.__name__} was raised")


def assert_refused_before_any_call(
    build: Callable[[], Any], spy: type, function: str, *fragments: str
) -> None:
    """``build()`` refuses before any learner fit and before any call of the spy ``function``.

    ``spy`` is the class whose class-level ``calls`` counts the calls of the declared
    function.  The caller resets it.  :func:`never_fit_learners` resets ``NeverFit``.
    """
    assert_refused(build, CapabilityError, *fragments)
    assert NeverFit.calls == 0, f"{NeverFit.calls} learner fit(s) ran before the refusal"
    assert spy.calls == 0, f"the {function} was evaluated before the refusal"


def assert_every_witness_fails(witnesses: Iterable[Callable[[], None]]) -> None:
    """Each witness raises ``AssertionError``: the teeth check of a mutation control."""
    for witness in witnesses:
        with pytest.raises(AssertionError):
            witness()


def restored(item: Any, field: str, kind: Any) -> Any:
    """``item`` with its declaration changed after construction, as a restore can leave it."""
    object.__setattr__(item, field, kind)
    return item


def restored_states(undeclared: str, estimated: str) -> dict[str, tuple[Any, tuple[str, ...]]]:
    """What a restored object can carry, and the fragments of the refusal each one meets."""
    return {"undeclared": (None, (undeclared,)), "estimated": ("estimated", (estimated, PATHWISE))}


def legacy_result(result: Any, field: str, select: Callable[[Any], list[Any]]) -> Any:
    """``result`` as an artifact written before ``field`` existed would restore it.

    ``select`` maps the estimator to the objects that carry ``field``.  It must select at
    least one, or the result would restore with nothing to drop.
    """
    old = loads(dumps(result))
    for item in select(old.estimator):
        vars(item).pop(field)
    old = loads(dumps(old))
    items = select(old.estimator)
    assert items, f"the estimator carries no object with {field}"
    assert all(getattr(item, field) is None for item in items)
    return old


def recomputations(result: Any, estimands: tuple[str, ...]) -> dict[str, Callable[[], Any]]:
    """Every entry to a recomputation a test drives: a sweep, a retarget, and a refit."""
    return {
        "truncation_curve": lambda: truncation_curve(result, bounds=[0.05]),
        "retarget": lambda: result.estimator.retarget(
            result.data, result.nuisance, estimands=estimands
        ),
        "refit": lambda: result.estimator.refit(result.data),
    }


def se_ratio(curve: np.ndarray, exact: np.ndarray, cell_p: np.ndarray) -> float:
    """The reported curve's SE over the exact curve's SE, from second moments.

    ``curve`` is the reported curve on the exact sample, and ``exact`` is the exact curve at
    each support point, whose probabilities are ``cell_p``.  ``std_error`` itself divides by
    ``n - 1``, which would move this by a factor of 1.0005.
    """
    return float(np.sqrt(np.mean(curve**2) / float(cell_p @ exact**2)))
