"""Warnings a fit raises are recorded on its result (``cleverly._warnings``, X89 plan 6.5).

Every warning of the package goes through ``cleverly._warnings.warn``, which records it and
then raises it where the call site always did. These tests pin that the record is complete
under any filter and any ``n_jobs``, that what the user sees does not change, and that a
resampled or nested fit keeps its warnings to itself.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import _warnings
from cleverly.datasets import make_linear_ate
from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.utils.parallel import map_parallel

SOURCE = Path(cleverly.__file__).parent


def _frame(**columns: Any) -> Any:
    return make_linear_ate(n=200, seed=1)[0].assign(**columns)


def _estimator(**overrides: Any) -> TMLE:
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "estimands": ("ate",),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return TMLE(**settings)


def _capped_fit(**overrides: Any) -> Any:
    """Tight probability limits, so the fit raises its positivity warning."""
    return (
        _estimator(g_bounds=(0.4, 0.6), **overrides)
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


def _positivity(result: Any) -> list[_warnings.FitWarning]:
    return [record for record in result.fit_warnings if record.category == "PositivityWarning"]


def _check_recorded(result: Any) -> None:
    """The constant covariate's warning is on the result and printed as a note."""
    (record,) = [item for item in result.fit_warnings if "constant" in item.message]
    assert record.category == "UserWarning"
    assert record.message in result.summary().note("fit_warning_other").text


def _constant_covariate_fit() -> Any:
    return _estimator().fit(_frame(W5=1.0), outcome="Y", treatment="A").single()


def test_a_warned_fit_records_the_warning_and_prints_it() -> None:
    with pytest.warns(UserWarning, match="constant"):
        result = _constant_covariate_fit()
    _check_recorded(result)


def test_a_restated_warning_prints_once_as_its_note() -> None:
    """The positivity warning is recorded, and the weights note that states the same rows
    prints among the warnings in its place."""
    with pytest.warns(PositivityWarning):
        result = _capped_fit()
    (record,) = _positivity(result)
    summary = result.summary()
    assert summary.note("weights_capped").kind == "warning"
    assert "fit_warning_positivity" not in summary.note_codes
    assert record.message not in str(summary)


def test_a_second_identical_fit_records_the_same_warnings() -> None:
    """The once-per-location registry hides the second warning from the screen, not from
    the record."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("default")
        first = _capped_fit()
        second = _capped_fit()
    assert _positivity(first)
    assert first.fit_warnings == second.fit_warnings


def _warns_in_a_task(index: int) -> int:
    _warnings.warn(f"task {index % 2} warned", UserWarning)
    return index


def test_parallel_and_serial_tasks_record_the_same_set() -> None:
    def run(n_jobs: int) -> tuple[list[int], list[_warnings.FitWarning]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _warnings.recording() as records:
                values = map_parallel(_warns_in_a_task, range(4), n_jobs=n_jobs)
        return values, list(records)

    serial, parallel = run(1), run(2)
    assert serial == parallel
    assert [record.message for record in serial[1]] == ["task 0 warned", "task 1 warned"]
    assert [record.count for record in parallel[1]] == [2, 2]


def test_filters_change_what_shows_and_not_what_is_recorded() -> None:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("ignore")
        result = _capped_fit()
    assert shown == []
    assert _positivity(result)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PositivityWarning):
            _capped_fit()


def _helper_at_two() -> None:
    _warnings.warn("from a helper", UserWarning, stacklevel=2)


def test_the_reported_location_is_the_call_sites() -> None:
    """The recorder's own frame never shows: a direct call, a helper at ``stacklevel=2`` and
    a package site through a fit each point where ``warnings.warn`` would have."""
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        warnings.warn("plain", UserWarning, stacklevel=1)
        plain = shown[-1]
        _warnings.warn("recorded", UserWarning, stacklevel=1)
        recorded = shown[-1]
        _helper_at_two()
        helper = shown[-1]
        _capped_fit()
        (positivity,) = [item for item in shown if item.category is PositivityWarning]
    assert plain.filename == recorded.filename == helper.filename == __file__
    assert recorded.lineno == plain.lineno + 2
    assert helper.lineno == recorded.lineno + 2
    # The positivity site warns at stacklevel=3: _warn_on_positivity, _fit_single, fit.
    assert positivity.filename == str(SOURCE / "estimators" / "tmle.py")
    line = Path(positivity.filename).read_text(encoding="utf-8").splitlines()[positivity.lineno - 1]
    assert "self._fit_single(" in line


def test_resampled_and_nested_fits_keep_their_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = _capped_fit()
        resampled = _capped_fit(n_bootstrap=4)
        with _warnings.recording() as outer:
            nested = _capped_fit()
    # Each replicate refits under the same limits and warns too, silenced and unrecorded.
    assert [record.count for record in _positivity(resampled)] == [1]
    assert _positivity(resampled) == _positivity(plain)
    assert _positivity(nested)
    assert outer == []


def test_the_record_check_fails_when_nothing_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: a recorder that drops every record fails the first test's check."""
    monkeypatch.setattr(_warnings._Recorder, "add", lambda self, record: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _constant_covariate_fit()
    with pytest.raises((AssertionError, ValueError)):
        _check_recorded(result)


def _direct_warn_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "warnings":
            found.extend(
                f"{path}:{node.lineno} imports warn" for name in node.names if name.name == "warn"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warn"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "warnings"
        ):
            found.append(f"{path}:{node.lineno} calls warnings.warn")
    return found


def test_every_package_warning_goes_through_the_recorder() -> None:
    files = [path for path in SOURCE.rglob("*.py") if path.name != "_warnings.py"]
    assert len(files) > 100
    found = [item for path in files for item in _direct_warn_calls(path)]
    assert found == []


def test_the_source_check_finds_both_forms(tmp_path: Path) -> None:
    """Witness: the scan above sees a direct call and an imported ``warn``."""
    sample = tmp_path / "sample.py"
    sample.write_text("import warnings\nfrom warnings import warn\nwarnings.warn('x')\n")
    assert len(_direct_warn_calls(sample)) == 2
