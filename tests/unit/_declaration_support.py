"""Shared scaffolding for the tests of a declared known function.

``tests/unit/test_msm_projection_weights.py`` (RM13),
``tests/unit/test_stochastic_regime_densities.py`` (RM25) and
``tests/unit/test_msm_design_declaration.py`` (RM27) test the three users of
:class:`cleverly._declarations.FunctionDeclaration`.  This module holds the parts that do
not depend on which field is declared: the refusal checks, the restored states, a legacy
point or longitudinal result and the checks of its status (RM28), the fit entries, the
two-node :func:`panel`, and the exact-law oracle fit.  ``tests/unit/_msm_declaration_support.py``
holds the MSM builders that the RM13 and RM27 files share, and
``tests/unit/_tilt_law_support.py`` holds the exact tilt law of the RM25 witness.  The test
files import these support modules and not each other.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CausalStudy, PointTreatment
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity.positivity import truncation_curve
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.unit._inference_status_support import assert_withholds
from tests.unit._natural_course_support import NeverFit

#: ``cleverly.estimators`` exports a function named ``tmle``, which shadows the module.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: A fragment of every refusal of ``"estimated"``: the term the reported curve omits.
PATHWISE = "pathwise derivative"

#: The status of a result restored with a function this version refuses (RM28).
UNDECLARED_STATUS = "undeclared_function_plugin"


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
    build: Callable[[], Any],
    spy: Any,
    function: str,
    *fragments: str,
    error: type[Exception] = CapabilityError,
) -> None:
    """``build()`` refuses before any learner fit and before any call of the spy ``function``.

    ``spy`` is the declared function, a :class:`tests.unit._confounding_support.Counter`
    whose ``calls`` starts at 0.  Pass the object the fit receives: a pickle round trip
    copies a counter.  :func:`never_fit_learners` resets ``NeverFit``.  ``error`` is the
    class of the refusal: a declaration that is refused raises ``CapabilityError``, and a
    value that is not a declaration raises ``DataError``.
    """
    assert_refused(build, error, *fragments)
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

    ``select`` maps the result to the objects that carry ``field``.  A point result holds
    them on ``result.estimator``, and a longitudinal result holds its resolved regimens on
    ``result.config.regimens``.  ``select`` must select at least one, or the result would
    restore with nothing to drop.
    """
    old = loads(dumps(result))
    for item in select(old):
        vars(item).pop(field)
    old = loads(dumps(old))
    items = select(old)
    assert items, f"the result carries no object with {field}"
    assert all(getattr(item, field) is None for item in items)
    return old


def assert_stored_interval_is_a_diagnostic(result: Any, old: Any) -> None:
    """``old``, ``result`` restored without its declaration, withholds inference (RM28).

    Every point estimate is unchanged.  Every estimate takes
    :data:`UNDECLARED_STATUS`, and ``ci``, ``pvalue`` and ``std_error`` refuse with its
    reason, which ``summary()`` prints.  The stored interval and standard error remain as
    ``plugin_interval`` and ``plugin_std_error``, and the simultaneous bands are dropped.
    The first line is the nonzero witness: the declared result reported an interval.
    """
    assert result.inference_status == "influence_curve"
    assert_withholds(old, UNDECLARED_STATUS)
    assert old.estimates.keys() == result.estimates.keys()
    for name, estimate in result.estimates.items():
        assert old.estimates[name].psi == estimate.psi
        assert old.estimates[name].plugin_interval == estimate.ci
        assert old.estimates[name].plugin_std_error == estimate.std_error
    assert old.simultaneous is None


def assert_keeps_its_interval(result: Any, old: Any) -> None:
    """``old``, ``result`` restored, keeps ``"influence_curve"`` and every stored interval."""
    assert old.inference_status == "influence_curve"
    for name, estimate in result.estimates.items():
        assert old.estimates[name].psi == estimate.psi
        assert old.estimates[name].ci == estimate.ci


def recomputations(result: Any, estimands: tuple[str, ...]) -> dict[str, Callable[[], Any]]:
    """Every entry to a recomputation a test drives: a sweep, a retarget, and a refit."""
    return {
        "truncation_curve": lambda: truncation_curve(result, bounds=[0.05]),
        "retarget": lambda: result.estimator.retarget(
            result.data, result.nuisance, estimands=estimands
        ),
        "refit": lambda: result.estimator.refit(result.data),
    }


def point_entries(
    declare: Callable[[Any], dict[str, Any]], target: Callable[[Any], Any]
) -> dict[str, Callable[[Any, dict[str, Any]], Any]]:
    """Every point-treatment fit entry that can reach a declared object, on the default law.

    ``declare`` maps the object to the ``TMLE`` keyword that carries it, and ``target``
    maps it to the target of ``CausalStudy.identify``.  Each entry takes the object and
    the learners: ``"fit"`` is ``TMLE.fit``, ``"study"`` is ``CausalStudy.estimate``, and
    ``"refit"`` is ``TMLE.refit`` on prepared data, which the refutations and the replay
    use.
    """

    def estimator(item: Any, learners: dict[str, Any]) -> TMLE:
        return TMLE(cross_fit=False, simultaneous=False, **declare(item), **learners)

    def fit(item: Any, learners: dict[str, Any]) -> Any:
        return estimator(item, learners).fit(law.frame(), outcome="Y", treatment="A")

    def study(item: Any, learners: dict[str, Any]) -> Any:
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        return (
            CausalStudy(law.frame(), design=design)
            .identify(target(item))
            .estimate(
                outcome_learner=learners["outcome_learner"],
                treatment_learner=learners["treatment_learner"],
                cross_fit=False,
                simultaneous=False,
            )
        )

    def refit(item: Any, learners: dict[str, Any]) -> Any:
        return estimator(item, learners).refit(
            CausalData.from_frame(law.frame(), outcome="Y", treatment="A")
        )

    return {"fit": fit, "study": study, "refit": refit}


def panel(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Two nodes, censoring, and a binary end-of-study outcome."""
    rng = np.random.default_rng(seed)
    c1 = (rng.random(n) < 0.9).astype(float)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    return pd.DataFrame(
        {
            "W1": rng.standard_normal(n),
            "A1": rng.integers(0, 2, n).astype(float),
            "C1": c1,
            "A2": np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan),
            "C2": c2,
            "Y": np.where(observed, rng.integers(0, 2, n).astype(float), np.nan),
        }
    )


# ------------------------------------------------------------------ the exact-law witness


def oracle_fit(counts: Any, **axis: Any) -> Any:
    """In sample on the law of ``counts``, with its own nuisances, so no learner error.

    ``counts`` is a :func:`tests.discrete_law.cell_counts` table, and ``axis`` is the
    ``TMLE`` keyword that carries the declared function.
    """
    dgp = law.DiscreteLaw(counts / law.N)
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
        **axis,
    )
    return estimator.fit(law.frame(counts), outcome="Y", treatment="A").single()


def cell_p(probs: Any) -> np.ndarray:
    """The probability of each support point, in the order of ``law.SUPPORT``."""
    return np.array([probs[cell] for cell in law.SUPPORT])


def se_ratio(curve: np.ndarray, exact: np.ndarray, cell_p: np.ndarray) -> float:
    """The reported curve's SE over the exact curve's SE, from second moments.

    ``curve`` is the reported curve on the exact sample, and ``exact`` is the exact curve at
    each support point, whose probabilities are ``cell_p``.  ``std_error`` itself divides by
    ``n - 1``, which would move this by a factor of 1.0005.
    """
    return float(np.sqrt(np.mean(curve**2) / float(cell_p @ exact**2)))
