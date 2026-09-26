"""Shared scaffolding for the tests of a declared known function.

``tests/unit/test_msm_projection_weights.py`` (RM13),
``tests/unit/test_stochastic_regime_densities.py`` (RM25) and
``tests/unit/test_msm_design_declaration.py`` (RM27) test the three users of
:class:`cleverly._declarations.FunctionDeclaration`.  This module holds the parts that do
not depend on which field is declared: the refusal checks, the modified states, a point or
longitudinal result whose declaration is removed after the fit (RM28), the checks that its
replay slots and rows agree with the calls they stand for (RM23), the fit entries, the
two-node :func:`panel` with its columns and ``NeverFit`` learners, and the exact-law oracle
fit.  ``tests/unit/_msm_declaration_support.py``
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
from cleverly.assessment import AssessmentStatus, replayability
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.sensitivity.positivity import truncation_curve
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
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
    """``item`` with its declaration changed after construction, as a caller can leave it."""
    object.__setattr__(item, field, kind)
    return item


def restored_states(undeclared: str, estimated: str) -> dict[str, tuple[Any, tuple[str, ...]]]:
    """What a modified object can carry, and the fragments of the refusal each one meets."""
    return {"undeclared": (None, (undeclared,)), "estimated": ("estimated", (estimated, PATHWISE))}


def undeclared_copy(result: Any, field: str, select: Callable[[Any], list[Any]]) -> Any:
    """A copy of ``result`` whose selected objects no longer declare ``field`` (RM28).

    ``select`` maps the copy to the objects that carry ``field``.  A point result holds
    them on ``result.estimator``, and a longitudinal result holds its resolved regimens on
    ``result.config.regimens``.  The constructors refuse an undeclared function, so the
    copy sets the field to ``None`` with ``object.__setattr__``, as a caller can modify a
    live object.  ``select`` must select at least one object, or the copy would carry
    nothing to refuse.
    """
    copy = loads(dumps(result))
    items = select(copy)
    assert items, f"the result carries no object with {field}"
    for item in items:
        assert getattr(item, field) is not None, f"{field} was already undeclared"
        object.__setattr__(item, field, None)
    return copy


# ------------------------------------------------------------------------ the replay slots

#: What a row refused by its replay slot appends to its sentence, before the codes.
_REPLAY_CODES = "reported codes: "


def replay_disagreements(result: Any, estimands: tuple[str, ...]) -> list[str]:
    """Every point replay slot of ``result`` that disagrees with the call it stands for.

    ``retarget_cached_nuisances`` stands for ``estimator.retarget`` on the cached
    nuisances, and ``refit_nuisances`` for ``estimator.refit`` on the result's own data.
    A :class:`~cleverly.exceptions.CapabilityError` or
    :class:`~cleverly.exceptions.DataError` is a refusal. Any other exception is a defect,
    so it propagates. An empty list is agreement.
    """
    replay = replayability(result)
    estimator = result.estimator
    calls: dict[str, Callable[[], Any]] = {
        "retarget_cached_nuisances": lambda: estimator.retarget(
            result.data, result.nuisance, estimands=estimands
        ),
        "refit_nuisances": lambda: estimator.refit(
            result.data, intermediate_value=result.intermediate_value
        ),
    }
    problems = []
    for slot, call in calls.items():
        try:
            call()
        except (CapabilityError, DataError) as error:
            refused: str | None = f"{type(error).__name__}: {error}"
        else:
            refused = None
        if getattr(replay, slot) != (refused is None):
            problems.append(f"{slot} reads {getattr(replay, slot)}, and the call gave {refused}")
    return problems


def assert_replay_agrees(result: Any, estimands: tuple[str, ...]) -> None:
    """Each point replay slot of ``result`` reads true exactly when its call runs."""
    assert replay_disagreements(result, estimands) == []


def replay_rows(result: Any) -> list[Any]:
    """Every capability row of ``result`` that declares a replay slot."""
    return [
        row
        for facade in (result.diagnostics, result.sensitivity)
        for row in facade.capabilities
        if row.requires_replay is not None
    ]


def assert_replay_rows_refused(result: Any, code: str) -> None:
    """Every replay row reads unavailable, the replay code refuses one, and a report runs.

    For a modified result whose replay slots read false, and ``code`` is the omission code
    that ``replayability`` reports for it. A row that a rule of its own refuses keeps that
    sentence, so the code need not name every row. The ``refute`` row reads unavailable
    too. A longitudinal ``refute`` row declares no replay slot, because it is unavailable
    for every longitudinal result.
    """
    assert code in replayability(result).unreconstructible
    rows = replay_rows(result)
    assert rows
    assert [row.operation for row in rows if row.available] == []
    named = [row.operation for row in rows if f"{_REPLAY_CODES}[{code!r}]" in (row.reason or "")]
    assert named, [row.reason for row in rows]
    assert not result.diagnostics.capability("refute").available
    report = result.assess(include_refits=True, include_retargets=True, random_state=0)
    assert report.diagnostics["refute"].status is AssessmentStatus.UNAVAILABLE


def assert_replay_rows_available(result: Any) -> None:
    """No replay row of ``result`` is refused by its slot, and one of them answers.

    The mirror of :func:`assert_replay_rows_refused`, for a declared result.
    Every slot a row declares reads true, and no row quotes a replay code. A row can still
    be refused by a rule of its own, so the check is that one row reads available or
    deferred.
    """
    replay = replayability(result)
    assert replay.unreconstructible == ()
    rows = replay_rows(result)
    assert rows
    assert all(getattr(replay, row.requires_replay) for row in rows)
    assert [row.operation for row in rows if _REPLAY_CODES in (row.reason or "")] == []
    assert any(row.available or row.status is AssessmentStatus.DEFERRED for row in rows)


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


#: The columns of :func:`panel`, as ``LTMLE.fit`` and ``LongitudinalData.from_frame`` read them.
PANEL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1"],
    "censoring": ["C1", "C2"],
}


def never_fit_longitudinal_learners() -> dict[str, NeverFit]:
    """Every longitudinal learner slot, each one a :class:`NeverFit`, with calls reset."""
    NeverFit.calls = 0
    return {
        "outcome_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
    }


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
