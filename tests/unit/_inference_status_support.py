"""Shared checks for the tests that follow an inference status through a fitted result.

Two shapes recur in every status test: an estimate or a report that must withhold its
inferential numbers by the reason of one status, and an artifact saved before its
configuration took that status, which must load re-stamped. The forced-status reach test
and each surface's own test read them here, so a later status inherits the same checks.
The fold-level report checks and the stamp mutation that must fail them live here too,
because the forced-status test and the clustered surfaces both fit ``cv_evaluation=True``.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import replace
from typing import Any

import pytest

from cleverly._inference_status import NON_INFERENTIAL
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, capitalize_first
from cleverly.inference.influence import _DIAGNOSTIC_NAMES

#: The two ways an artifact is restored: the package's own serializer and a bare pickle.
ROUTES = ("serialize", "pickle")

#: The columns a frame publishes only when the package supplies inference.
INFERENTIAL_COLUMNS = frozenset({"std_err", "ci_lower", "ci_upper", "p_value"})

#: The columns a frame publishes in their place on a non-inferential status.
DIAGNOSTIC_COLUMNS = frozenset(
    {"inference", "plugin_std_err", "plugin_interval_lower", "plugin_interval_upper"}
)


#: Every name a report publishes only when the package supplies inference: each key of
#: the diagnostic-name table that is a column or a key, and the p-value columns, which
#: have no diagnostic name at all.
INFERENTIAL_NAMES = frozenset(
    name for name in _DIAGNOSTIC_NAMES if re.fullmatch(r"[a-z_]+", name)
) | {"p_value", "p_value_adjusted"}

#: Text that claims an interval or a standard error. A status reason says "no confidence
#: interval", so the bare phrase is not here: these are the forms a published number takes.
FORBIDDEN_TEXT = (
    re.compile(r"\bCIs?\b"),
    re.compile(r"confidence[- ](limit|bound)"),
    re.compile(r"(?<![\w-])std_err\b"),
    re.compile(r"\bp_value\b"),
    re.compile(r"\bRVa?\b.*confidence"),
)


def assert_no_inferential_name(names: Any) -> None:
    leaked = INFERENTIAL_NAMES & set(names)
    assert not leaked, f"inferential names published on a diagnostic fit: {sorted(leaked)}"


def assert_no_inferential_text(text: str) -> None:
    for pattern in FORBIDDEN_TEXT:
        match = pattern.search(text)
        assert match is None, f"{pattern.pattern!r} matched {match.group(0)!r} in:\n{text}"


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


def assert_fold_report_withholds(result: Any, status: str) -> None:
    """The fold-level report carries the status and publishes no inferential name."""
    report = result.cv_targeting
    assert report is not None
    assert report.inference == status
    for name in report.pooled:
        with pytest.raises(CapabilityError) as raised:
            _ = report.pooled[name].ci
        assert_refused_by(status, raised)
        with pytest.raises(CapabilityError):
            _ = report.canonical[name].std_error
    with pytest.raises(CapabilityError) as raised:
        _ = report.std_error
    assert_refused_by(status, raised)
    assert report.plugin_std_error
    columns = set(report.to_frame().columns)
    assert_no_inferential_name(columns)
    assert {"inference", "cv_plugin_std_err", "pooled_plugin_std_err"} <= columns
    assert_no_inferential_text(report.summary())


def assert_fold_reports_restamped(restored: Any, result: Any, status: str) -> None:
    """Both fold-level reports of a restored ``restored`` carry ``status`` again.

    ``result`` is the live fit the legacy copy was made from, so the diagnostic of each
    pooled estimate is compared bit for bit.
    """
    detail = restored.cv_targeting
    assert detail.inference == status
    for name in detail.pooled:
        assert detail.pooled[name].inference == status
        assert detail.canonical[name].inference == status
        assert (
            detail.pooled[name].plugin_std_error
            == result.cv_targeting.pooled[name].plugin_std_error
        )


def _unstamped(report: dict[str, Any]) -> dict[str, Any]:
    return {
        name: replace(estimate, inference="influence_curve") for name, estimate in report.items()
    }


def stamp_headline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation of the stamp site: only the headline estimates carry the status.

    Wraps ``TMLE._retarget_detailed`` so the two fold-level reports it returns keep
    ``"influence_curve"``, which is the stamp before RM20 reached them.
    """
    stamped = TMLE._retarget_detailed

    def headline_only(self: Any, *args: Any, **kwargs: Any) -> Any:
        ordered, fluctuations, detail = stamped(self, *args, **kwargs)
        if detail is not None:
            detail = replace(
                detail,
                pooled=_unstamped(detail.pooled),
                canonical=_unstamped(detail.canonical),
            )
        return ordered, fluctuations, detail

    monkeypatch.setattr(TMLE, "_retarget_detailed", headline_only)


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
