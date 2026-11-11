"""Continuous-time survival fits whose summaries the contract test checks (X91 part G).

A single-cause fit on the full sample, a two-cause fit cross-fitted over three folds, and the
single-cause fit with its targeting record marked as stopped early, so the warning fires.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from cleverly.survival import ContinuousSurvivalTMLE

from . import Case
from ._check_fidelity import close, header_values

__all__ = ["FIXTURES", "read_continuous_survival"]


def _frame(causes: int) -> pd.DataFrame:
    rng = np.random.default_rng(29 + causes)
    n = 240
    age = rng.normal(size=n)
    arm = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * age)))
    times = [rng.exponential(1.0 / (0.6 * np.exp(0.3 * age - 0.4 * arm))) for _ in range(causes)]
    first = np.min(times, axis=0)
    cause = np.argmin(times, axis=0) + 1
    censor = rng.exponential(2.5, n)
    return pd.DataFrame(
        {
            "followup": np.minimum(first, censor),
            "status": np.where(first <= censor, cause, 0),
            "drug": arm,
            "age": age,
        }
    )


@functools.cache
def _fit(causes: int, cross_fit: bool) -> Any:
    engine = ContinuousSurvivalTMLE(
        (0.5, 1.0), cross_fit=cross_fit, n_folds=3, learner_folds=3, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.fit(
            _frame(causes), time="followup", event="status", treatment="drug", baseline=["age"]
        )


def single_cause() -> Any:
    """One cause, fitted on the full sample."""
    return _fit(1, False)


def two_causes() -> Any:
    """Two competing causes, cross-fitted over three folds."""
    return _fit(2, True)


def stopped_early() -> Any:
    """The single-cause fit with its targeting marked as stopped early (synthetic)."""
    fit = single_cause()
    return replace(fit, record=replace(fit.record, converged=False))


def read_continuous_survival(result: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``result``.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fit.
    text : str
        Its short printed summary.
    """
    from tests.unit.test_summary_contract import _check_fidelity

    facts = header_values(text)
    assert close(facts["Units"], float(result.n)), facts
    assert facts["Update steps"].split(",")[0] == str(result.record.steps), facts
    _check_fidelity(result, text)


def _case(build: Any) -> Case:
    return Case(
        build, names=("followup", "status", "drug", "age"), fidelity=read_continuous_survival
    )


FIXTURES: dict[str, Case] = {
    "single_cause": _case(single_cause),
    "two_causes": _case(two_causes),
    "stopped_early": _case(stopped_early),
}
