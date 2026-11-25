"""Kernel-smoothed dose-response fits, whose summaries the contract test checks (X123 (b)).

A direct fit of :class:`~cleverly.estimators.DoseResponseTMLE` and the same curve fitted
through :class:`~cleverly.CausalStudy` with a study protocol, each with explicit fast learners.
The result carries no ``data``, so each fixture names the columns it prints and reads every
printed number back from the result.
"""

from __future__ import annotations

import functools
import warnings
from types import SimpleNamespace
from typing import Any

from cleverly._summary import vocabulary
from tests.unit.summary_fixtures import Case
from tests.unit.summary_fixtures._check_fidelity import header_values

__all__ = ["FIXTURES"]

_NAMES = ("Y", "A", "W")


def read_dose_response(result: Any, text: str) -> None:
    """Check every header number and every estimates cell against the result.

    Parameters
    ----------
    result : DoseResponseResult
        The fitted curve.
    text : str
        Its short summary.
    """
    import pandas as pd

    from tests.unit.test_summary_contract import _check_fidelity

    facts = header_values(text)
    curve = result.curve
    record = result.nuisance_record
    assert facts[vocabulary.ROWS] == f"{record.outcome.size:,}"
    assert facts[vocabulary.DOSE_COUNT] == str(len(curve.doses))
    assert facts[vocabulary.DOSE_BANDWIDTHS] == str(len(curve.bandwidths))
    low, high = curve.support
    assert facts[vocabulary.DOSE_RANGE] == f"[{low:g}, {high:g}]"
    folds = int(record.folds.max()) + 1
    expected = f"{folds} folds" if record.cross_fitted else "no"
    assert facts[vocabulary.OUT_OF_FOLD] == expected
    frame = pd.concat([estimate.to_frame() for estimate in result.estimates.values()])
    _check_fidelity(SimpleNamespace(to_frame=lambda: frame), text)


@functools.cache
def _frame() -> Any:
    from tests.unit.test_dose_response_assessment import _binary

    return _binary()


def direct() -> Any:
    """A two-fold direct fit of a binary outcome at three doses, two of them thin."""
    from tests.unit.test_dose_response_assessment import _curve, _fit

    return _fit(_curve(50.0), _frame())


def study() -> Any:
    """The same curve fitted through ``CausalStudy``, with a study protocol."""
    from sklearn.linear_model import LogisticRegression

    from cleverly import CausalStudy, DoseResponseCurve, PointTreatment
    from tests.unit.summary_fixtures.setup import protocol
    from tests.unit.test_dose_response_assessment import _curve

    effect = CausalStudy(
        _frame(),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
        protocol=protocol(
            treatment_strategies=["Each declared dose"], treatment_versions=["One dose given once"]
        ),
    ).identify(DoseResponseCurve(_curve(50.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=0,
            n_folds=2,
        )


FIXTURES: dict[str, Any] = {
    "direct": Case(direct, names=_NAMES, fidelity=read_dose_response),
    "study": Case(study, names=_NAMES, fidelity=read_dose_response),
}
