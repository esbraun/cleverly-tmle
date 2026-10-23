r"""Every public call on an AdaptiveIPTW result returns a value or refuses by name.

Each fit of :mod:`tests.unit._capability_sweep_support` is built once: an arm fit with an
estimated and with a known mechanism, a deterministic regime fit, a stochastic regime fit, the
natural course, a composite fit with a missing outcome, a missing treatment in sample and
cross-fitted, and a regime fit with missing outcomes. The test calls every public method of the
result, of ``result.diagnostics`` and of ``result.sensitivity`` with no arguments, and ``assess()``.
A call returns a value or raises :class:`~cleverly.exceptions.CapabilityError`. The calls never
raise ``AttributeError``, ``RuntimeError``, ``ValueError`` or ``NotImplementedError``, and the
row ``capability(operation)`` agrees with the call in both directions.

``result.psi()`` and ``result.estimate`` raise ``ValueError`` on a fit with several estimates, as
they do on a TMLE fit, so the walker leaves them out and one test pins that equality.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from cleverly.exceptions import CapabilityError
from tests.unit import _capability_sweep_support as sweep

FITS: dict[str, Callable[[], Any]] = {
    "estimated": sweep.fit_iptw,
    "known": sweep.fit_iptw_known,
    "regime": sweep.fit_iptw_regime,
    "stochastic": sweep.fit_iptw_stochastic,
    "natural_course": sweep.fit_iptw_natural_course,
    "missing_outcome": sweep.fit_iptw_missing,
    "missing_treatment": sweep.fit_iptw_missing_treatment,
    "missing_treatment_crossfit": sweep.fit_iptw_missing_treatment_crossfit,
    "regime_missing": sweep.fit_iptw_regime_missing,
}

#: Methods whose no-argument call raises ``ValueError`` on any fit with several estimates.
AMBIGUOUS = frozenset({"psi"})

#: Operations the walker maps to a capability row of the same name.
DIAGNOSTIC_ROWS = ("nuisance_models", "score_equations", "support", "refute", "truncation_curve")
SENSITIVITY_ROWS = (
    "benchmark",
    "contour",
    "elements",
    "evalue",
    "missingness",
    "omitted_confounding",
    "robustness_value",
    "simulated_confounding",
    "tipping_gamma",
)

FORBIDDEN = (AttributeError, RuntimeError, ValueError, NotImplementedError)


@pytest.fixture(scope="module", params=sorted(FITS))
def result(request: pytest.FixtureRequest) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FITS[request.param]()


def _public_calls(owner: Any, *, skip: frozenset[str] = frozenset()) -> list[str]:
    names = []
    for name in dir(owner):
        if name.startswith("_") or name in skip or name == "capability":
            continue
        try:
            attribute = getattr(owner, name)
        except (KeyError, ValueError):
            # An estimand property such as ``ate`` raises when the fit did not report it.
            continue
        if not callable(attribute):
            continue
        required = [
            p
            for p in inspect.signature(attribute).parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        if not required:
            names.append(name)
    return names


def _call(owner: Any, name: str) -> Any:
    """The value of ``owner.name()``, or the :class:`CapabilityError` it raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return getattr(owner, name)()
        except CapabilityError as error:
            return error
        except FORBIDDEN as error:  # pragma: no cover - reaching here fails the test
            pytest.fail(f"{type(owner).__name__}.{name}() raised {type(error).__name__}: {error}")


def test_every_public_call_on_the_result_returns_or_refuses_by_name(result: Any) -> None:
    outcomes = {name: _call(result, name) for name in _public_calls(result, skip=AMBIGUOUS)}
    assert "summary" in outcomes and "assess" in outcomes
    assert not isinstance(outcomes["summary"], CapabilityError)
    assert not isinstance(outcomes["assess"], CapabilityError)
    assert outcomes["assess"] is not None


def test_a_refused_call_carries_a_record_with_a_reason_and_an_alternative_or_a_roadmap(
    result: Any,
) -> None:
    for owner in (result, result.diagnostics, result.sensitivity):
        for name in _public_calls(owner, skip=AMBIGUOUS):
            outcome = _call(owner, name)
            if isinstance(outcome, CapabilityError):
                refusal = outcome.refusal
                assert refusal.condition
                assert refusal.alternative or refusal.roadmap or refusal.reason is not None


@pytest.mark.parametrize("operation", DIAGNOSTIC_ROWS)
def test_the_diagnostic_rows_agree_with_the_calls(result: Any, operation: str) -> None:
    row = result.diagnostics.capability(operation)
    outcome = _call(result.diagnostics, operation)
    ran = not isinstance(outcome, CapabilityError)
    assert row.available is ran, operation
    if not ran:
        assert row.refusal is not None
        assert row.refusal.condition in str(outcome)


@pytest.mark.parametrize("operation", SENSITIVITY_ROWS)
def test_the_sensitivity_rows_agree_with_the_calls(result: Any, operation: str) -> None:
    row = result.sensitivity.capability(operation)
    outcome = _call(result.sensitivity, operation)
    ran = not isinstance(outcome, CapabilityError)
    assert row.available is ran, operation
    if not ran:
        assert row.refusal is not None
        assert row.refusal.condition in str(outcome)


def test_the_ambiguous_accessors_raise_what_a_tmle_fit_raises() -> None:
    adaptive = sweep.fit_iptw()
    tmle = sweep.fit_ordinary()
    for name in AMBIGUOUS:
        with pytest.raises(ValueError, match="multiple parameters"):
            getattr(adaptive, name)()
        with pytest.raises(ValueError, match="multiple parameters"):
            getattr(tmle, name)()


def test_the_outcome_regression_rows_refuse_by_name_on_every_fit(result: Any) -> None:
    """No fit has an outcome regression, so the rows that read one refuse on all of them."""
    for operation in ("refute", "truncation_curve"):
        assert not result.diagnostics.capability(operation).available
    for operation in ("omitted_confounding", "contour", "robustness_value", "benchmark"):
        assert not result.sensitivity.capability(operation).available
    assert result.diagnostics.capability("nuisance_models").available
    # The natural-course mean alone has no arm to support.
    if "ey_obs" not in result.estimates or len(result.estimates) > 1:
        assert result.diagnostics.capability("support").available


def test_the_known_mechanism_fit_answers_the_evalue_and_the_estimated_one_refuses_it() -> None:
    known = sweep.fit_iptw_known()
    estimated = sweep.fit_iptw()
    assert known.sensitivity.capability("evalue").available
    assert not estimated.sensitivity.capability("evalue").available
    assert not isinstance(_call(known.sensitivity, "evalue"), CapabilityError)
