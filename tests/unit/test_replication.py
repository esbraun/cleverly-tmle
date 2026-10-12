"""``fit_replicates``: one failed or killed fit costs one replicate, and nothing else moves."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, DataError, NotFittedError
from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import failure_call, recordable
from tests.studies.evidence import replication
from tests.studies.evidence.failures import FAILURE_COLUMNS
from tests.studies.evidence.replication import (
    FAILURE_FIELDS,
    KilledReplicate,
    SystematicFailure,
    checkpoints,
    fit_replicates,
)

KEYS = [(cell, replicate) for cell in ("a", "b") for replicate in range(25)]


def _draw(key):  # type: ignore[no-untyped-def]
    return np.random.default_rng([ord(key[0]), key[1]]).normal(size=7)


def _fit(key, drawn):  # type: ignore[no-untyped-def]
    return (key, float(drawn.mean()), float(drawn.std()))


def _singular_on(bad):  # type: ignore[no-untyped-def]
    def fit(key, drawn):  # type: ignore[no-untyped-def]
        if key in bad:
            np.linalg.inv(np.zeros((2, 2)))
        return _fit(key, drawn)

    return fit


def _call(keys, fit_one, **options):  # type: ignore[no-untyped-def]
    options.setdefault("n_jobs", 2)
    options.setdefault("checkpoint", None)
    return fit_replicates(
        keys,
        fit_one,
        draw=_draw,
        label=options.pop("label", "test"),
        implementation="subject",
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        **options,
    )


def _reference(keys):  # type: ignore[no-untyped-def]
    return map_parallel(lambda key: _fit(key, _draw(key)), [(key,) for key in keys], n_jobs=1)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_without_failure_the_values_are_map_parallels(n_jobs: int) -> None:
    outcomes = _call(KEYS, _fit, n_jobs=n_jobs)
    assert outcomes.fitted == _reference(KEYS)
    assert outcomes.failed == []
    assert outcomes.failures.empty


def test_the_two_call_split_keeps_key_order() -> None:
    # A deliberate-mutation control: the split runs the prefix of each cell before the rest,
    # so values that came back in run order would interleave the cells.
    outcomes = _call(KEYS, _fit, _prefix=1)
    assert [value[0] for value in outcomes.fitted] == KEYS
    run_order = [*(key for key in KEYS if key[1] == 0), *(key for key in KEYS if key[1] != 0)]
    assert [value[0] for value in outcomes.fitted] != run_order


def test_a_recorded_failure_keeps_the_rest_and_says_why() -> None:
    outcomes = _call(KEYS, _singular_on({("b", 3)}), seed_of=lambda key: 100 + key[1])
    assert outcomes.failed == [KEYS.index(("b", 3))]
    assert outcomes.fitted[outcomes.failed[0]] is None
    assert list(outcomes.failures.columns) == list(FAILURE_FIELDS) == list(FAILURE_COLUMNS)
    record = outcomes.failures.iloc[0].to_dict()
    assert record == {
        "implementation": "subject",
        "group_id": "test/b/r000000003",
        "scenarios": "b",
        "replicates": "3",
        "attempt": 1,
        "class": "fitting_failure",
        "stage": "cleverly fit",
        "call": "numpy.linalg._linalg:_raise_linalgerror_singular",
        "message": "LinAlgError: Singular matrix",
        "seed": "103",
    }
    assert outcomes.values == [
        value for key, value in zip(KEYS, _reference(KEYS), strict=True) if key != ("b", 3)
    ]


def test_on_failure_supplies_the_failed_keys_value_from_its_draw() -> None:
    outcomes = _call(
        KEYS, _singular_on({("a", 4)}), on_failure=lambda key, drawn: ("drawn", drawn.sum())
    )
    assert outcomes.fitted[4] == ("drawn", _draw(("a", 4)).sum())


def test_a_rejected_row_is_an_invalid_result() -> None:
    outcomes = _call(
        KEYS, _fit, validate=lambda value: "non-finite estimate" if value[0] == ("a", 9) else None
    )
    record = outcomes.failures.iloc[0]
    assert (record["class"], record["stage"], record["message"]) == (
        "invalid_result",
        "row check",
        "non-finite estimate",
    )


@pytest.mark.parametrize(
    "error",
    [TypeError("bug"), KeyError("bug"), ValueError("ours"), CapabilityError("no")],
    ids=lambda error: type(error).__name__,
)
def test_a_non_recordable_error_stops_the_run_with_its_class(error: BaseException) -> None:
    def fit(key, drawn):  # type: ignore[no-untyped-def]
        if key == ("a", 5):
            raise error
        return _fit(key, drawn)

    with pytest.raises(type(error)):
        _call(KEYS, fit)


def test_an_error_in_the_draw_stops_the_run() -> None:
    def draw(key):  # type: ignore[no-untyped-def]
        raise np.linalg.LinAlgError("a law defect")

    with pytest.raises(np.linalg.LinAlgError):
        fit_replicates(
            KEYS,
            _fit,
            draw=draw,
            n_jobs=1,
            label="test",
            implementation="subject",
            cell_of=lambda key: key[0],
            replicate_of=lambda key: key[1],
            checkpoint=None,
        )


def test_the_systematic_stop_reads_each_cell_and_needs_twenty() -> None:
    nineteen = {("a", replicate) for replicate in range(19)}
    outcomes = _call(KEYS, _singular_on(nineteen))
    assert len(outcomes.failed) == 19
    with pytest.raises(SystematicFailure, match="the first 20 replicates of a failed"):
        _call(KEYS, _singular_on({("a", replicate) for replicate in range(20)}))


def test_every_failed_key_stops_the_run() -> None:
    few = [("a", replicate) for replicate in range(3)]
    with pytest.raises(RuntimeError, match="every replicate of test failed"):
        _call(few, _singular_on(set(few)))


def test_a_checkpointed_run_resumes_without_refitting(tmp_path: Path) -> None:
    calls = tmp_path / "calls"
    calls.mkdir()

    def fit(key, drawn):  # type: ignore[no-untyped-def]
        (calls / f"{key[0]}{key[1]}-{os.getpid()}-{len(list(calls.iterdir()))}").touch()
        return _fit(key, drawn)

    root = tmp_path / "checkpoints"
    first = _call(KEYS, fit, checkpoint=root)
    assert len(list(calls.iterdir())) == len(KEYS)
    second = _call(KEYS, fit, checkpoint=root)
    assert len(list(calls.iterdir())) == len(KEYS)
    assert second.fitted == first.fitted == _reference(KEYS)
    assert second.reused == len(KEYS)


def _killing(flags: Path, victims: set[tuple[str, int]], times: int):  # type: ignore[no-untyped-def]
    def fit(key, drawn):  # type: ignore[no-untyped-def]
        if key in victims:
            flag = flags / f"{key[0]}{key[1]}"
            count = int(flag.read_text()) if flag.exists() else 0
            if count < times:
                flag.write_text(str(count + 1))
                os._exit(1)
        return _fit(key, drawn)

    return fit


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_a_killed_worker_costs_one_more_attempt_alone(tmp_path: Path, n_jobs: int) -> None:
    flags = tmp_path / "flags"
    flags.mkdir()
    outcomes = _call(
        KEYS,
        _killing(flags, {("a", 2), ("b", 21)}, times=1),
        n_jobs=n_jobs,
        checkpoint=tmp_path / "checkpoints",
    )
    assert outcomes.fitted == _reference(KEYS)
    assert outcomes.failed == []


def test_a_replicate_killed_alone_stops_the_run(tmp_path: Path) -> None:
    flags = tmp_path / "flags"
    flags.mkdir()
    with pytest.raises(KilledReplicate, match="replicate 2 of test"):
        _call(KEYS, _killing(flags, {("a", 2)}, times=2), checkpoint=tmp_path / "c")


def test_memory_exhaustion_is_a_kill_not_a_record(tmp_path: Path) -> None:
    flags = tmp_path / "flags"
    flags.mkdir()

    def fit(key, drawn):  # type: ignore[no-untyped-def]
        flag = flags / f"{key[0]}{key[1]}"
        if key == ("b", 7) and not flag.exists():
            flag.touch()
            raise MemoryError
        return _fit(key, drawn)

    outcomes = _call(KEYS, fit, checkpoint=tmp_path / "c")
    assert outcomes.failed == []
    assert outcomes.fitted == _reference(KEYS)
    # Outside the driver a MemoryError is fatal, because there is no worker to lose.
    with pytest.raises(MemoryError):
        _call(KEYS, lambda key, drawn: (_ for _ in ()).throw(MemoryError()), n_jobs=1)


def test_a_stale_marker_from_a_killed_driver_runs_its_key_alone(tmp_path: Path) -> None:
    root = tmp_path / "c"
    _call(KEYS[:3], _fit, checkpoint=root)
    directory = root / "subject" / "test"
    stored = sorted(directory.glob("*.pkl.gz"))[0]
    ident = stored.name.removesuffix(".pkl.gz")
    stored.unlink()
    (directory / f"{ident}.started").write_text("1\n")
    outcomes = _call(KEYS[:3], _fit, checkpoint=root)
    assert outcomes.fitted == _reference(KEYS[:3])
    assert (directory / f"{ident}.kills").read_text().strip() == "1"
    assert not list(directory.glob("*.started"))


def test_a_label_is_used_once_per_run(tmp_path: Path) -> None:
    with checkpoints(tmp_path):
        _call(KEYS[:2], _fit, checkpoint="ambient")
        with pytest.raises(ValueError, match="used twice"):
            _call(KEYS[:2], _fit, checkpoint="ambient")
    assert replication.checkpoint_root() is None


def _raised(function):  # type: ignore[no-untyped-def]
    try:
        function()
    except BaseException as error:
        return error
    raise AssertionError("nothing was raised")


class TestTheClassifier:
    def test_listed_fitting_failures_are_recorded(self) -> None:
        for error in (DataError("x"), ZeroDivisionError("x"), OverflowError("x")):
            assert recordable(error)
        assert recordable(_raised(lambda: np.linalg.inv(np.zeros((2, 2)))))

    def test_a_library_plain_value_error_is_recorded(self) -> None:
        from sklearn.linear_model import LogisticRegression

        error = _raised(lambda: LogisticRegression().fit(np.ones((4, 1)), [1, 1, 1, 1]))
        assert type(error) is ValueError
        assert recordable(error)
        assert failure_call(error).startswith("sklearn.")

    def test_a_library_value_error_subclass_is_fatal(self) -> None:
        from sklearn.linear_model import LogisticRegression

        error = _raised(lambda: LogisticRegression(C="large").fit(np.ones((4, 1)), [0, 1, 0, 1]))
        assert type(error).__name__ == "InvalidParameterError"
        assert not recordable(error)

    def test_cleverly_runtime_errors_and_subclasses_are_fatal(self) -> None:
        assert not recordable(_raised(lambda: (_ for _ in ()).throw(RuntimeError("ours"))))
        assert not recordable(NotFittedError("x"))
        assert not recordable(CapabilityError("x"))

    def test_a_numpy_error_raised_in_c_under_our_frame_is_fatal(self) -> None:
        def ours() -> None:
            np.ones(3).reshape(2, 2)

        error = _raised(ours)
        assert type(error) is ValueError
        assert failure_call(error).endswith(":ours")
        assert not recordable(error)

    def test_an_explicit_translation_is_read_as_the_outer_class(self) -> None:
        def translated() -> None:
            try:
                np.linalg.inv(np.zeros((2, 2)))
            except np.linalg.LinAlgError as error:
                raise RuntimeError("translated") from error

        assert not recordable(_raised(translated))


_ROW = {
    "scenario": "law",
    "estimand": "ate",
    "truth": 0.0,
    "estimate": 0.1,
    "inference_estimate": 0.1,
    "std_error": 0.1,
    "ci_lower": -0.2,
    "ci_upper": 0.4,
    "covered": 1,
}


class TestRowRules:
    """``invalid_rows`` applies the reference harness's ``study_invalid`` rules."""

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"estimate": float("nan")}, "non-finite estimate for ate of law (row 2)"),
            ({"std_error": float("inf")}, "non-finite std_error"),
            ({"std_error": 0.0}, "non-positive std_error"),
            ({"ci_lower": 0.5}, "ci_lower above ci_upper"),
            ({"covered": 0}, "covered disagrees with the interval and truth"),
        ],
    )
    def test_each_rule_names_the_first_broken_row(self, change: dict, message: str) -> None:
        rows = [_ROW, {**_ROW, **change}]
        broken = replication.invalid_rows(rows)
        assert broken is not None and broken.startswith(message)
        assert replication.invalid_rows(pd.DataFrame(rows)) == broken

    def test_sound_rows_and_absent_columns_break_nothing(self) -> None:
        assert replication.invalid_rows([_ROW, _ROW]) is None
        assert replication.invalid_rows([{"estimate": 1.0}]) is None
        assert replication.invalid_rows([]) is None


def test_induced_records_name_the_subject_group() -> None:
    from tests.studies.evidence import failures
    from tests.studies.evidence.registry import StudyRecord

    record = StudyRecord(
        name="x",
        slug="x",
        artifacts=Path("."),
        document="x.md",
        anchor="x",
        scenarios={"law": ("ate",)},
        replicates=10,
        n=50,
        seed=1,
        reference="R ref",
    )
    outcomes = _call(KEYS, _singular_on({("a", 4)}), seed_of=lambda key: 7)
    subject = outcomes.failures.assign(scenarios="law")
    induced = failures.induced(subject, record=record)
    assert induced.loc[0, ["implementation", "class", "stage", "seed", "replicates"]].tolist() == [
        "R ref",
        failures.PAIRED_INPUT_MISSING,
        "subject fit",
        "7",
        "4",
    ]
    assert "test/a/r000000004" in induced.loc[0, "message"]
    with pytest.raises(ValueError, match="no reference"):
        failures.induced(subject, record=dataclasses.replace(record, reference=None))
