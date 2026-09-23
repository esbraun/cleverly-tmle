"""Shared checks for the tests that follow an inference status through a fitted result.

Two shapes recur in every status test: an estimate or a report that must withhold its
inferential numbers by the reason of one status, and an artifact saved before its
configuration took that status, which must load re-stamped. The forced-status reach test
and each surface's own test read them here, so a later status inherits the same checks.
"""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from cleverly._inference_status import NON_INFERENTIAL
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, capitalize_first

#: The two ways an artifact is restored: the package's own serializer and a bare pickle.
ROUTES = ("serialize", "pickle")

#: The columns a frame publishes only when the package supplies inference.
INFERENTIAL_COLUMNS = frozenset({"std_err", "ci_lower", "ci_upper", "p_value"})

#: The columns a frame publishes in their place on a non-inferential status.
DIAGNOSTIC_COLUMNS = frozenset(
    {"inference", "plugin_std_err", "plugin_interval_lower", "plugin_interval_upper"}
)


def assert_refused_by(status: str, raised: pytest.ExceptionInfo[CapabilityError]) -> None:
    """The raise ends with the reason of ``status``."""
    assert NON_INFERENTIAL[status].reason in str(raised.value)


def assert_withholds(result: Any, status: str) -> None:
    """Every estimate of ``result`` carries ``status`` and withholds its inference.

    The status on the fit and on each estimate, a refusal with the status reason from
    each of ``ci``, ``pvalue`` and ``std_error``, a finite retained diagnostic, the frame
    columns, and the ``summary()`` label and paragraph. A mutation that restores
    ``"influence_curve"`` on the surface fails the first line.
    """
    assert result.inference_status == status
    for estimate in result.estimates.values():
        assert estimate.inference == status
        for accessor in ("ci", "pvalue", "std_error"):
            with pytest.raises(CapabilityError) as raised:
                getattr(estimate, accessor)
            assert_refused_by(status, raised)
        assert estimate.plugin_std_error > 0
    columns = set(result.to_frame().columns)
    assert not INFERENTIAL_COLUMNS & columns
    assert columns >= DIAGNOSTIC_COLUMNS
    record = NON_INFERENTIAL[status]
    text = result.summary()
    assert record.summary_label in text
    assert capitalize_first(record.reason) in text
    assert "95% CI" not in text


def assert_keeps_inference(result: Any) -> None:
    """The control: every estimate answers ``ci``, ``pvalue`` and ``std_error``."""
    assert result.inference_status == "influence_curve"
    for estimate in result.estimates.values():
        low, high = estimate.ci
        assert low < high
        assert 0.0 <= estimate.pvalue <= 1.0
        assert estimate.std_error == estimate.plugin_std_error
    assert set(result.to_frame().columns) >= INFERENTIAL_COLUMNS


def legacy_copy(result: Any) -> Any:
    """A copy saved as an ordinary fit: every estimate and any fold report inferential.

    The post-RM12, pre-RM20 shape. Its estimates carry ``inference="influence_curve"``
    explicitly, and it holds a simultaneous band and an assessment answer that a status
    fit would not have.
    """
    legacy = pickle.loads(pickle.dumps(result))
    reports = [legacy.estimates]
    detail = legacy.cv_targeting
    if detail is not None:
        reports.extend([detail.pooled, detail.canonical])
    for report in reports:
        for estimate in report.values():
            estimate.__dict__["inference"] = "influence_curve"
    legacy.__dict__["simultaneous"] = "bands built before the status"
    legacy.__dict__["assessment_cache"] = {"sensitivity.evalue": "an answer read off .ci"}
    return legacy


def restore(artifact: Any, route: str) -> Any:
    """``artifact`` saved and loaded by ``route``, one of :data:`ROUTES`."""
    if route == "serialize":
        return loads(dumps(artifact))
    return pickle.loads(pickle.dumps(artifact))


def assert_restamped(result: Any, status: str, route: str) -> Any:
    """A legacy copy of ``result`` loads under ``status``, with its diagnostic unchanged.

    Returns the restored result, so a caller can check the reports it adds.
    """
    legacy = legacy_copy(result)
    # The nonzero witness: the copy really is the inferential shape before it loads.
    assert legacy.inference_status == "influence_curve"
    restored = restore(legacy, route)
    assert restored.inference_status == status
    for name, estimate in restored.estimates.items():
        with pytest.raises(CapabilityError) as raised:
            _ = estimate.ci
        assert_refused_by(status, raised)
        assert estimate.plugin_std_error == result.estimates[name].plugin_std_error
        assert estimate.plugin_interval == result.estimates[name].plugin_interval
    assert restored.simultaneous is None
    assert restored.assessment_cache == {}
    return restored
