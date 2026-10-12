"""RM39a2: every comparator fit runs inside the harness's ``study_stage``, and only the fit does.

CI installs no R, so these checks read the runners as text.  The harness smoke
(``tests/canonical/harness_fixture/smoke.py``) runs the stage itself.  What is checked here is
the shape each runner must keep so that a recorded failure means "the comparator raised":

* every comparator runner reaches ``study_stage``, in its own text or through a file it sources;
* no runner or adapter keeps a local ``stage <- function`` re-signal helper;
* every warning promotion raises ``study_fatal_error``, the class a stage never records (Q-a);
* no function a runner defines sits inside a stage argument, except the allow-listed comparator
  wrappers that call package code only (M7), and no stage argument defines a function;
* the two point adapters take no stage (m9);
* the drift test's ``SHARED`` names the new shared functions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.studies.evidence.registry import ROOT
from tests.unit.test_canonical_runner_parity import COLLECTORS
from tests.unit.test_harness_drift import SHARED

CANONICAL = ROOT / "tests" / "canonical"
HARNESS = CANONICAL / "study_harness.R"

#: Files that collect through the harness but fit no comparator a study publishes.  Their
#: errors stay fatal (M3).
NO_STAGE = {
    "tmle_mar_attributable/probe_scale_workaround.R": "probe: every error stays fatal (M3)",
    "tmle_mar_natural_course/probe_scale_workaround.R": "probe: every error stays fatal (M3)",
}

#: Runner-defined functions that may sit inside a stage argument: each is a comparator wrapper
#: that calls package code only, with no check of its own.
STAGE_WRAPPERS = frozenset({"fit_conditional", "fit_cv"})

#: Every R file a comparator fit runs through: the runners and the adapters beside the harness.
COMPARATOR_FILES = tuple(
    sorted(
        path
        for path in CANONICAL.rglob("*.R")
        if path != HARNESS
        and "harness_fixture" not in path.parts
        and "lmtp_crossfit" not in path.parts
        and not path.name.startswith("probe_")
    )
)

_SOURCED = re.compile(r"""source\(["']/fixture/(?P<path>[^"']+)["']\)""")
_DEFINED = re.compile(r"^\s*(?P<name>[A-Za-z_.][A-Za-z0-9_.]*) <- function\(", re.M)
_CALLED = re.compile(r"(?<![\w.$@])(?P<name>[A-Za-z_.][A-Za-z0-9_.]*)\(")


def _label(path: Path) -> str:
    return path.relative_to(CANONICAL).as_posix()


def _sourced(path: Path, seen: set[Path] | None = None) -> list[Path]:
    """``path`` and every file it sources, transitively, without the harness."""
    seen = set() if seen is None else seen
    if path in seen:
        return []
    seen.add(path)
    found = [path]
    for match in _SOURCED.finditer(path.read_text(encoding="utf-8")):
        target = CANONICAL / match.group("path")
        if target != HARNESS:
            found += _sourced(target, seen)
    return found


def _code(source: str) -> str:
    """``source`` with comments blanked, so a word in a comment is never read as code."""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False
    for character in source:
        if comment:
            if character == "\n":
                comment = False
                out.append(character)
            else:
                out.append(" ")
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            out.append(character)
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "#":
            comment = True
            out.append(" ")
            continue
        out.append(character)
    return "".join(out)


def _without_strings(source: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', source)


def _argument(source: str, opening: int) -> str:
    """The text between the bracket at ``opening`` and its match, comments already blanked."""
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError("unbalanced brackets")


def _calls(source: str, name: str) -> list[str]:
    """The argument text of every ``name(`` call in ``source``."""
    code = _code(source)
    pattern = re.compile(rf"(?<![\w.]){re.escape(name)}\(")
    return [_argument(code, match.end() - 1) for match in pattern.finditer(code)]


COMPARATOR_RUNNERS = tuple(path for path in COLLECTORS if _label(path) not in NO_STAGE)


def test_the_comparator_runners_are_found() -> None:
    assert len(COMPARATOR_RUNNERS) >= 45
    assert set(NO_STAGE) <= {_label(path) for path in COLLECTORS}


@pytest.mark.parametrize("runner", COMPARATOR_RUNNERS, ids=_label)
def test_every_comparator_runner_reaches_a_stage(runner: Path) -> None:
    """Q-i: no comparator runner is left without a stage."""
    reached = [path for path in _sourced(runner) if _calls(path.read_text("utf-8"), "study_stage")]
    assert reached, (
        f"{_label(runner)} fits a comparator without study_stage, so a comparator error stops "
        f"the run instead of costing one group. Wrap the comparator call, or list the file in "
        f"NO_STAGE with its reason"
    )


@pytest.mark.parametrize("path", COMPARATOR_FILES, ids=_label)
def test_no_file_keeps_a_local_stage_helper(path: Path) -> None:
    code = _code(path.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w.])stage <- function", code), (
        f"{_label(path)} defines its own stage helper; study_stage replaces it"
    )


@pytest.mark.parametrize("path", COMPARATOR_FILES, ids=_label)
def test_every_warning_promotion_raises_the_fatal_class(path: Path) -> None:
    """Q-a and m11: a promoted warning is never recorded as a fitting failure."""
    for argument in _calls(path.read_text(encoding="utf-8"), "withCallingHandlers"):
        match = re.search(r"\bwarning\s*=\s*function", argument)
        if match is None:
            continue
        handler = argument[match.start() :]
        assert "study_fatal_error(" in handler, (
            f"{_label(path)}: a warning handler must raise study_fatal_error"
        )
        assert not re.search(r"(?<![\w.])stop\(", _without_strings(handler)), (
            f"{_label(path)}: a warning handler raises a plain error, which a stage would record"
        )


def test_both_known_promotions_are_checked() -> None:
    promoting = {
        _label(path)
        for path in COMPARATOR_FILES
        if any(
            "warning" in argument
            for argument in _calls(path.read_text("utf-8"), "withCallingHandlers")
        )
    }
    assert {"ltmle_regimen_adapter.R", "ltmle_few_cluster/run_study.R"} <= promoting


@pytest.mark.parametrize("path", COMPARATOR_FILES, ids=_label)
def test_a_stage_holds_the_comparator_call_only(path: Path) -> None:
    """M7: no runner-defined function, and no function definition, inside a stage argument."""
    files = _sourced(path)
    defined = {
        match.group("name")
        for file in files
        for match in _DEFINED.finditer(_code(file.read_text(encoding="utf-8")))
    }
    for argument in _calls(path.read_text(encoding="utf-8"), "study_stage"):
        body = _without_strings(argument)
        assert not re.search(r"(?<![\w.])function\s*\(", body), (
            f"{_label(path)}: a stage argument defines a function: {argument.strip()[:120]}"
        )
        called = {match.group("name") for match in _CALLED.finditer(body)}
        local = sorted((called & defined) - STAGE_WRAPPERS)
        assert not local, (
            f"{_label(path)}: runner-defined {local} run inside a stage, so their checks would be "
            f"recorded as comparator failures: {argument.strip()[:120]}"
        )
        assert not {name for name in called if name.startswith("study_")}, (
            f"{_label(path)}: a harness function runs inside a stage"
        )


def test_the_allow_listed_wrappers_hold_no_check() -> None:
    """A wrapper allowed inside a stage must not carry a contract stop of its own."""
    for path in COMPARATOR_FILES:
        code = _code(path.read_text(encoding="utf-8"))
        for match in _DEFINED.finditer(code):
            if match.group("name") in STAGE_WRAPPERS:
                body = _without_strings(_argument(code, code.index("{", match.end())))
                assert not re.search(r"(?<![\w.])(stop|stopifnot)\(", body), (
                    f"{_label(path)}: {match.group('name')} carries a check; move it outside "
                    f"the stage, as tmle3_cvtmle's cv_task does"
                )


@pytest.mark.parametrize("name", ("tmle_point_adapter.R", "tmle_continuous_point_adapter.R"))
def test_the_point_adapters_take_no_stage(name: str) -> None:
    """m9: they transcribe a finished fit and call no comparator."""
    assert not _calls((CANONICAL / name).read_text(encoding="utf-8"), "study_stage")


def test_the_lmtp_smokes_source_the_harness_before_an_adapter() -> None:
    """m2: the adapters call study_stage, which only the harness defines."""
    for path in sorted((CANONICAL / "lmtp_crossfit").glob("smoke*.R")):
        sourced = [match.group("path") for match in _SOURCED.finditer(path.read_text("utf-8"))]
        assert sourced and sourced[0] == "study_harness.R", path.name


def test_the_rm19_wrapper_turns_recording_off() -> None:
    """m1: a diagnostic must see every comparator error raw."""
    wrapper = ROOT / "tests" / "diagnostics" / "rm19_one_sided_increment" / "count_rounds.R"
    assert 'Sys.setenv(CLEVERLY_R_NO_RECORD = "1")' in wrapper.read_text(encoding="utf-8")


def test_the_drift_test_covers_the_failure_functions() -> None:
    added = {
        "study_stage",
        "study_fatal_error",
        "study_failures",
        "study_systematic",
        "study_invalid",
        "study_finish",
    }
    assert added <= set(SHARED)


def test_the_harness_writes_the_failure_columns_the_driver_reads() -> None:
    """The record's columns are ``FAILURE_COLUMNS`` without the driver's ``implementation``."""
    from tests.studies.evidence.failures import FAILURE_COLUMNS

    harness = HARNESS.read_text(encoding="utf-8")
    (row,) = _calls(harness[harness.index("study_failure_row <- function") :], "data.frame")[:1]
    written = [
        name for name in re.findall(r"^\s*([a-z_]+) = ", row, re.M) if name != "stringsAsFactors"
    ]
    assert tuple(written) == tuple(
        column for column in FAILURE_COLUMNS if column != "implementation"
    )


def test_the_failures_file_name_matches_the_driver() -> None:
    """``Path.stem`` strips the last suffix only; the harness regex must do the same."""
    harness = HARNESS.read_text(encoding="utf-8")
    assert 'sub("(.)\\\\.[^.]+$", "\\\\1", basename(output)), ".failures.csv"' in harness
