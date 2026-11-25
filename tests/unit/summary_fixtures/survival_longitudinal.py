"""Continuous-time longitudinal fits whose summaries the contract test checks (X6 part (e)).

One full-sample fit on a law CT draw with two causes, and the same fit with its node walks
marked as stopped short, so the not-converged note fires.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

from tests.unit import _capability_sweep_support as sweep

from . import Case
from ._check_fidelity import close, header_values

__all__ = ["FIXTURES", "read_continuous_longitudinal"]


@functools.cache
def fitted() -> Any:
    """Two causes, the regimes always and never, fitted on the full sample."""
    return sweep.fit_continuous_longitudinal()


def stopped_short() -> Any:
    """The fit with its node walks marked as stopped short of the cutoff (synthetic)."""
    fit = fitted()
    return replace(fit, record=replace(fit.record, converged=False))


def read_continuous_longitudinal(result: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``result``.

    Parameters
    ----------
    result : ContinuousLongitudinalResult
        The fit.
    text : str
        Its short printed summary.
    """
    from tests.unit.test_summary_contract import _check_fidelity

    facts = header_values(text)
    assert close(facts["Units"], float(result.n)), facts
    steps = sum(int(v) for row in result.record.steps for v in row)
    assert str(steps) in facts["Update steps"], facts
    _check_fidelity(result, text)


def _case(build: Any) -> Case:
    return Case(
        build,
        names=("always", "never", "1", "2"),
        fidelity=read_continuous_longitudinal,
    )


FIXTURES: dict[str, Case] = {
    "fitted": _case(fitted),
    "stopped_short": _case(stopped_short),
}
