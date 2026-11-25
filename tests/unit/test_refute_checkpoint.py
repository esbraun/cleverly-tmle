"""Checkpoints and progress for the long refutation tests (X106 part G3, comparison plan
"Equivalence witnesses").

Every comparison goes through a bitwise field walk (``_refute_checkpoint_support.walk``), so two
``nan`` values compare equal and ``-0.0`` differs from ``0.0``.

- W1 a resume equals the uninterrupted call, for each stored test and family, a ratio with a
  nonpositive draw, and a stop and resume in fresh processes.
- W2 a resume refits only the missing draws, with a mutation control that ignores the directory.
- W3 a changed request refuses, naming the part.
- W4 without the keywords the tests equal the loop G3 replaced, and nothing is written.
- W5 the facade's cache ignores the two keywords, and a fully loaded resume stays pinned.
- W6 the progress records, W7 a leftover temporary file, W8 variable importance, W9 the preview.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError
from cleverly.validation import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    GaussianIndependentOutcome,
    GaussianNoise,
    RefutationCheckpoint,
    RefutationProgress,
    planned_refits,
    refute,
)
from tests.unit import _refute_checkpoint_support as support
from tests.unit.test_planned_refits import _gaussian_panel, _point, _structural

refute_module: Any = sys.modules["cleverly.validation.refute"]
checkpoint_module: Any = sys.modules["cleverly.validation._refute_checkpoint"]

GENERATED_DRAWS = 10
BOOTSTRAP_DRAWS = 5


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


@functools.cache
def _in_sample() -> Any:
    return support.in_sample()


@functools.cache
def _survival_panel_fit() -> Any:
    from tests.unit.test_longitudinal_refutation import _ltmle, _survival_panel

    return _quiet(
        lambda: _ltmle(n_folds=1).fit(
            _survival_panel(n=400),
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    )


def _measurement(draws: int, **settings: Any) -> dict[str, Any]:
    return {
        "bootstrap_measurement_error": BootstrapMeasurementError(("W1",), **settings),
        "measurement_error_rule": EmpiricalInclusionRule(alpha=2.0 / draws, minimum_draws=draws),
    }


def _survival_arguments() -> dict[str, Any]:
    from cleverly.validation import LogisticHazardOutcome

    return {
        "estimand": "ate_regimen[never vs always @ t=2]",
        "simulated_outcome": LogisticHazardOutcome(
            intercept=-1.5, effect=0.8, adjustment_scale=0.5
        ),
    }


#: case: (fit, test, draws, extra keywords).
CASES: dict[str, tuple[Callable[[], Any], str, int, Callable[[], dict[str, Any]]]] = {
    "point dummy_outcome": (_in_sample, "dummy_outcome", GENERATED_DRAWS, dict),
    "point simulated_outcome": (_in_sample, "simulated_outcome", GENERATED_DRAWS, dict),
    "point plasmode": (_in_sample, "plasmode", GENERATED_DRAWS, dict),
    "point bootstrap_stability": (_point, "bootstrap_stability", BOOTSTRAP_DRAWS, dict),
    "point bootstrap_measurement_error": (
        _point,
        "bootstrap_measurement_error",
        BOOTSTRAP_DRAWS,
        lambda: _measurement(BOOTSTRAP_DRAWS),
    ),
    "longitudinal dummy_outcome": (
        _gaussian_panel,
        "dummy_outcome",
        GENERATED_DRAWS,
        lambda: {"estimand": next(name for name in _gaussian_panel() if name.startswith("ate"))},
    ),
    "survival simulated_outcome": (
        _survival_panel_fit,
        "simulated_outcome",
        GENERATED_DRAWS,
        _survival_arguments,
    ),
    "structural bootstrap_stability": (
        _structural,
        "bootstrap_stability",
        BOOTSTRAP_DRAWS,
        lambda: {"estimand": "blip[A1:1.0:intercept]"},
    ),
}


def _arguments(case: str) -> tuple[Any, dict[str, Any]]:
    build, test, draws, extra = CASES[case]
    return build(), {"tests": [test], "n_replicates": draws, "random_state": 0, **extra()}


@functools.cache
def _uninterrupted(case: str) -> Any:
    result, arguments = _arguments(case)
    return _quiet(lambda: refute(result, **arguments))


def _stop_and_resume(result: Any, arguments: dict[str, Any], directory: Path, after: int) -> Any:
    checkpoint = RefutationCheckpoint(directory)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: refute(
                result, **arguments, checkpoint=checkpoint, progress=support.stopper(after)
            )
        )
    return _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint))


# ----------------------------------------------------------------- W1 resume equals uninterrupted


@pytest.mark.parametrize("position", ["first", "mid", "last"])
@pytest.mark.parametrize("case", list(CASES))
def test_w1_a_resume_equals_the_uninterrupted_call(
    case: str, position: str, tmp_path: Path
) -> None:
    result, arguments = _arguments(case)
    draws = arguments["n_replicates"]
    after = {"first": 1, "mid": draws // 2, "last": draws - 1}[position]
    resumed = _stop_and_resume(result, arguments, tmp_path, after)
    assert support.walk(resumed) == support.walk(_uninterrupted(case))


def _negate(draw: Callable[..., Any], replicate: int) -> Callable[..., Any]:
    """Make draw ``replicate`` return a nonpositive estimate, before the ratio filter."""

    def patched(*args: Any, **kwargs: Any) -> Any:
        outcome = draw(*args, **kwargs)
        if kwargs["replicate"] == replicate and hasattr(outcome, "estimate"):
            return dataclasses.replace(outcome, estimate=-abs(outcome.estimate))
        return outcome

    return patched


def test_w1_a_stored_nonpositive_ratio_meets_the_filter_after_the_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint stores the raw draw, so a loaded nonpositive ratio becomes the same
    retained failure the uninterrupted call records."""
    from cleverly import RiskRatio
    from cleverly.validation import BernoulliAdjustmentOutcome
    from tests.unit.test_generated_point_families import _fit, _frame

    result = _quiet(lambda: _fit(_frame(300), "Yb", RiskRatio()))
    arguments: dict[str, Any] = {
        "estimand": "rr",
        "tests": ["simulated_outcome"],
        "simulated_outcome": BernoulliAdjustmentOutcome(),
        "n_replicates": GENERATED_DRAWS,
        "random_state": 4,
    }
    monkeypatch.setattr(refute_module, "_empirical_draw", _negate(refute_module._empirical_draw, 2))
    uninterrupted = _quiet(lambda: refute(result, **arguments))
    (failure,) = uninterrupted["simulated_outcome"].failures
    assert failure.replicate == 2 and "not positive" in failure.message
    resumed = _stop_and_resume(result, arguments, tmp_path, 5)
    stored = json.loads((tmp_path / "simulated_outcome" / "draw-000000002.json").read_text())
    assert stored["kind"] == "record" and stored["fields"]["estimate"] < 0
    assert support.walk(resumed) == support.walk(uninterrupted)


def test_w1_a_stop_and_resume_in_fresh_processes_equal_the_call(tmp_path: Path) -> None:
    """The learner's repr holds an address, so a key that read it would differ per process."""
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd()), "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "tests.unit._refute_checkpoint_support"]
    subprocess.run([*command, "stop", str(tmp_path), "4"], check=True, env=environment)
    resumed = subprocess.run(
        [*command, "resume", str(tmp_path), "0"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = support.in_sample(learner=support.AddressedLinear())
    expected = _quiet(lambda: refute(result, tests=["dummy_outcome"], n_replicates=10))
    assert resumed == support.digest(expected)


# ----------------------------------------------------------------- W2 no stored draw is refit


def _counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    original = refute_module._empirical_draw

    def counted(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["replicate"])
        return original(*args, **kwargs)

    monkeypatch.setattr(refute_module, "_empirical_draw", counted)
    return seen


@pytest.mark.parametrize("ignore", [False, True], ids=["checkpoint", "mutation"])
def test_w2_a_resume_refits_only_the_missing_draws(
    ignore: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, arguments = _arguments("point dummy_outcome")
    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: refute(result, **arguments, checkpoint=checkpoint, progress=support.stopper(4))
        )
    seen = _counted(monkeypatch)
    if ignore:
        # Mutation: a resume that never reads the directory.
        monkeypatch.setattr(checkpoint_module.RefutationCheckpoint, "load", lambda *a: None)
    _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint))
    assert (seen == list(range(4, GENERATED_DRAWS))) is not ignore


# ----------------------------------------------------------------- W3 mismatches refuse


def _stored(tmp_path: Path, **changes: Any) -> tuple[Any, dict[str, Any], RefutationCheckpoint]:
    """A directory holding four draws of the point dummy-outcome request."""
    result, arguments = _arguments("point dummy_outcome")
    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: refute(result, **arguments, checkpoint=checkpoint, progress=support.stopper(4))
        )
    return result, {**arguments, **changes}, checkpoint


def _refusal(call: Callable[[], Any]) -> str:
    with pytest.raises(CapabilityError) as error:
        _quiet(call)
    return str(error.value)


@pytest.mark.parametrize(
    ("changes", "part"),
    [
        ({"random_state": 1}, "Pass random_state=0"),
        ({"n_replicates": 12}, "its draws differs"),
        (
            {"dummy_outcome": GaussianIndependentOutcome(noise=GaussianNoise(0.0, 2.0))},
            "its process differs",
        ),
        ({"outcome_bands": "wider"}, "its bands differs"),
    ],
    ids=["seed", "draws", "process", "bands"],
)
def test_w3_a_changed_request_refuses_naming_the_part(
    changes: dict[str, Any], part: str, tmp_path: Path
) -> None:
    from cleverly.validation import EquivalenceBands

    if changes.get("outcome_bands") == "wider":
        changes = {"outcome_bands": EquivalenceBands(bias=0.5)}
    result, arguments, checkpoint = _stored(tmp_path, **changes)
    assert part in _refusal(lambda: refute(result, **arguments, checkpoint=checkpoint))


def test_w3_another_estimand_refuses(tmp_path: Path) -> None:
    """A direct fit reports five aliases, each of which the bootstrap test refits."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly.estimators import TMLE

    estimator = TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
        random_state=0,
    )
    result = _quiet(
        lambda: estimator.fit(
            support.frame(), outcome="Y", treatment="A", covariates=["W1", "W2"]
        ).single()
    )
    arguments: dict[str, Any] = {"tests": ["bootstrap_stability"], "n_replicates": 3}
    checkpoint = RefutationCheckpoint(tmp_path)
    _quiet(lambda: refute(result, estimand="ate", **arguments, checkpoint=checkpoint))
    message = _refusal(lambda: refute(result, estimand="att", **arguments, checkpoint=checkpoint))
    assert "its request differs" in message


def test_w3_another_fit_with_the_same_seed_refuses(tmp_path: Path) -> None:
    _, arguments, checkpoint = _stored(tmp_path)
    other = support.in_sample(seed=1)
    message = _refusal(lambda: refute(other, **arguments, checkpoint=checkpoint))
    assert "its data differs" in message


def test_w3_a_changed_resampling_refuses(tmp_path: Path) -> None:
    """``"iid"`` and ``"auto"`` resolve alike on unclustered data, and the declaration still
    differs, so the directory refuses rather than mix two declarations."""
    result, arguments = _arguments("point bootstrap_measurement_error")
    checkpoint = RefutationCheckpoint(tmp_path)
    _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint))
    changed = {**arguments, **_measurement(BOOTSTRAP_DRAWS, resampling="iid")}
    message = _refusal(lambda: refute(result, **changed, checkpoint=checkpoint))
    assert "its declaration differs" in message


def test_w3_an_unseeded_call_names_the_stored_seed(tmp_path: Path) -> None:
    result = support.in_sample(random_state=None)
    arguments: dict[str, Any] = {"tests": ["dummy_outcome"], "n_replicates": GENERATED_DRAWS}
    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: refute(result, **arguments, checkpoint=checkpoint, progress=support.stopper(3))
        )
    root = json.loads((tmp_path / "request.json").read_text())["common"]["seeds"]["root"]
    message = _refusal(lambda: refute(result, **arguments, checkpoint=checkpoint))
    assert f"Pass random_state={root}" in message
    resumed = _quiet(lambda: refute(result, **arguments, random_state=root, checkpoint=checkpoint))
    expected = _quiet(lambda: refute(result, **arguments, random_state=root))
    assert support.walk(resumed) == support.walk(expected)


def _shift(values: Any) -> Any:
    return values


def test_w3_a_lambda_inside_a_learner_refuses_by_name(tmp_path: Path) -> None:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    learner = make_pipeline(FunctionTransformer(lambda values: values), LinearRegression())
    result = support.in_sample(learner=learner)
    message = _refusal(
        lambda: refute(
            result,
            tests=["dummy_outcome"],
            n_replicates=GENERATED_DRAWS,
            checkpoint=RefutationCheckpoint(tmp_path),
        )
    )
    assert "<lambda>" in message and "func" in message
    # A module-level function keys by its name.
    named = make_pipeline(FunctionTransformer(_shift), LinearRegression())
    _quiet(
        lambda: refute(
            support.in_sample(learner=named),
            tests=["dummy_outcome"],
            n_replicates=GENERATED_DRAWS,
            checkpoint=RefutationCheckpoint(tmp_path / "named"),
        )
    )


@pytest.mark.parametrize("drop", [False, True], ids=["key", "mutation"])
def test_w3_a_fit_that_differs_only_in_its_weights_refuses(
    drop: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data fingerprint omits weights, so the data part adds them; without that
    addition the estimates part still refuses, one part later."""
    if drop:
        # Mutation: the data part without its weights and cluster additions.
        monkeypatch.setattr(
            checkpoint_module,
            "_data_key",
            lambda result: {"fingerprint": result.provenance.data_fingerprint},
        )
    n = 300
    arguments: dict[str, Any] = {"tests": ["dummy_outcome"], "n_replicates": GENERATED_DRAWS}
    first = support.in_sample(weights=np.ones(n))
    second = support.in_sample(weights=np.linspace(0.5, 1.5, n))
    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: refute(first, **arguments, checkpoint=checkpoint, progress=support.stopper(2))
        )
    message = _refusal(lambda: refute(second, **arguments, checkpoint=checkpoint))
    assert ("its estimates differs" if drop else "its data differs") in message


def test_w3_an_addressed_setting_refuses_by_path_and_an_object_array_keys(
    tmp_path: Path,
) -> None:
    arguments: dict[str, Any] = {"tests": ["dummy_outcome"], "n_replicates": GENERATED_DRAWS}
    addressed = support.in_sample(learner=support.TaggedLinear(tag=support.Opaque()))
    message = _refusal(
        lambda: refute(addressed, **arguments, checkpoint=RefutationCheckpoint(tmp_path / "a"))
    )
    assert "outcome_learner.params['tag']" in message and "memory address" in message
    tagged = support.in_sample(learner=support.TaggedLinear(tag=np.array(["a", "b"], dtype=object)))
    report = _quiet(
        lambda: refute(tagged, **arguments, checkpoint=RefutationCheckpoint(tmp_path / "b"))
    )
    assert support.walk(report) == support.walk(_quiet(lambda: refute(tagged, **arguments)))


def test_w3_a_directory_with_no_draw_takes_a_new_request(tmp_path: Path) -> None:
    """A call stopped before its first draw finished leaves a request and no draw."""
    result, arguments = _arguments("point dummy_outcome")
    checkpoint = RefutationCheckpoint(tmp_path)

    def stop_first(*args: Any, **kwargs: Any) -> Any:
        raise support.Stop

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(refute_module, "_empirical_draw", stop_first)
        with pytest.raises(support.Stop):
            _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint))
    assert (tmp_path / "request.json").exists()
    changed = {**arguments, "n_replicates": 12}
    report = _quiet(lambda: refute(result, **changed, checkpoint=checkpoint))
    assert support.walk(report) == support.walk(_quiet(lambda: refute(result, **changed)))


def test_w3_a_tampered_version_or_seed_refuses(tmp_path: Path) -> None:
    result, arguments, checkpoint = _stored(tmp_path)
    request = tmp_path / "request.json"
    original = request.read_text()
    tampered = json.loads(original)
    tampered["common"]["library versions"]["numpy"] = "0.0"
    request.write_text(json.dumps(tampered))
    assert "library versions differs" in _refusal(
        lambda: refute(result, **arguments, checkpoint=checkpoint)
    )
    request.write_text(original)
    draw = tmp_path / "dummy_outcome" / "draw-000000001.json"
    stored = json.loads(draw.read_text())
    stored["fields"]["seed"] += 1
    draw.write_text(json.dumps(stored))
    assert "stores draw 1 of dummy_outcome" in _refusal(
        lambda: refute(result, **arguments, checkpoint=checkpoint)
    )


# ----------------------------------------------------------------- W4 no checkpoint, no change


def _baseline_refits(
    result: Any,
    *,
    estimand: str,
    draws: Any,
    replacement: Any,
    expected_family: str,
    retain_preparation_failures: bool = False,
    refit_options: Any = None,
    refit_estimator: Any = None,
    session: Any = None,
    test: str = "",
) -> Any:
    """The loop G3 replaced, copied as it stood (``session`` and ``test`` are accepted and
    ignored)."""
    from cleverly.validation.refute import EmpiricalRefitRecord
    from cleverly.validation.simulation import ReplicationFailure

    records = []
    failures = []
    for replicate, child_seed in draws:
        try:
            prepared = replacement(replicate, child_seed)
        except Exception as error:
            if not retain_preparation_failures:
                raise
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            continue
        try:
            estimator = result.estimator if refit_estimator is None else refit_estimator(replicate)
            refitted = estimator.refit(
                prepared,
                **refute_module._intermediate_option(result),
                random_state=child_seed,
                **({} if refit_options is None else refit_options(replicate)),
            )
        except Exception as error:
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            continue
        fitted_family = str(refitted.data.family)
        assert fitted_family == expected_family
        estimate = refitted[estimand]
        if not np.isfinite(estimate.psi) or not np.isfinite(estimate.plugin_std_error):
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type="ValueError",
                    message="refitted estimate or standard error is non-finite",
                )
            )
            continue
        records.append(
            EmpiricalRefitRecord(
                replicate=replicate,
                seed=child_seed,
                estimand=estimand,
                estimate=float(estimate.psi),
                std_error=float(estimate.plugin_std_error),
                family=fitted_family,
            )
        )
    return tuple(records), tuple(failures)


@pytest.mark.parametrize("case", [case for case in CASES if case.startswith("point")])
def test_w4_without_the_keywords_the_tests_equal_the_replaced_loop(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, arguments = _arguments(case)
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(refute_module, "_run_empirical_refits", _baseline_refits)
        baseline = _quiet(lambda: refute(result, **arguments))
    assert support.walk(_quiet(lambda: refute(result, **arguments))) == support.walk(baseline)
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------- W5 the facade cache


def test_w5_the_facade_cache_ignores_the_keywords(tmp_path: Path) -> None:
    result = dataclasses.replace(_in_sample())
    _, arguments = _arguments("point dummy_outcome")
    seen: list[Any] = []
    first = _quiet(
        lambda: result.diagnostics.refute(
            **arguments, checkpoint=RefutationCheckpoint(tmp_path), progress=seen.append
        )
    )
    assert len(seen) == GENERATED_DRAWS
    second = _quiet(lambda: result.diagnostics.refute(**arguments))
    third = _quiet(
        lambda: result.diagnostics.refute(
            **arguments, checkpoint=RefutationCheckpoint(tmp_path / "unused"), progress=seen.append
        )
    )
    direct = _uninterrupted("point dummy_outcome")
    assert support.walk(first) == support.walk(second) == support.walk(third)
    assert support.walk(first) == support.walk(direct)
    assert len(seen) == GENERATED_DRAWS
    assert not (tmp_path / "unused").exists()


def test_w5_run_all_leaves_the_keywords_out_of_its_key(tmp_path: Path) -> None:
    result = dataclasses.replace(_in_sample())
    _, arguments = _arguments("point dummy_outcome")
    arguments.pop("random_state")
    keyed = {**arguments, "checkpoint": RefutationCheckpoint(tmp_path), "progress": print}
    first = _quiet(
        lambda: result.diagnostics.run_all(
            include_refits=True, arguments={"refute": keyed}, random_state=0
        )
    )
    before = len(result.assessment_cache)
    second = _quiet(
        lambda: result.diagnostics.run_all(
            include_refits=True, arguments={"refute": arguments}, random_state=0
        )
    )
    assert len(result.assessment_cache) == before
    (item,) = [row for row in first.items if row.name == "refute"]
    assert "checkpoint" not in item.arguments and "progress" not in item.arguments
    assert support.walk(first.items) == support.walk(second.items)


@pytest.mark.parametrize("mark", [True, False], ids=["marked", "mutation"])
def test_w5_a_fully_loaded_resume_files_a_pinned_answer(
    mark: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed rule keeps every refit anyway, so the witness installs a rule that would
    evict it: the mark alone must then keep the answer."""
    from cleverly import _assessment_cache
    from cleverly._assessment_cache import _is_evictable

    monkeypatch.setattr(_assessment_cache, "_EVICTION_RULE", [lambda *args: True])
    _, arguments = _arguments("point dummy_outcome")
    checkpoint = RefutationCheckpoint(tmp_path)
    _quiet(lambda: refute(_in_sample(), **arguments, checkpoint=checkpoint))
    if not mark:
        # Mutation: loading a draw no longer marks a learner fit.
        monkeypatch.setattr(refute_module, "note_learner_fit", lambda: None)
    result = dataclasses.replace(_in_sample())
    seen = _counted(monkeypatch)
    _quiet(lambda: result.diagnostics.refute(**arguments, checkpoint=checkpoint))
    assert seen == []
    cache = result.assessment_cache
    keys = [key for key in cache if "refute" in key]
    assert keys
    assert any(_is_evictable(cache, key) for key in keys) is not mark


# ----------------------------------------------------------------- W6 progress


def test_w6_progress_counts_each_draw_after_its_file_exists(tmp_path: Path) -> None:
    result, arguments = _arguments("point dummy_outcome")
    seen: list[RefutationProgress] = []
    pid = os.getpid()

    def progress(record: RefutationProgress) -> None:
        assert os.getpid() == pid
        path = tmp_path / record.test / f"draw-{record.completed - 1:09d}.json"
        assert path.exists()
        seen.append(record)
        if record.completed == 3:
            raise support.Stop

    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint, progress=progress))
    stored = sorted((tmp_path / "dummy_outcome").iterdir())
    assert [json.loads(path.read_text())["kind"] for path in stored] == ["record"] * 3
    seen.clear()
    _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint, progress=seen.append))
    assert [record.completed for record in seen] == list(range(1, GENERATED_DRAWS + 1))
    assert [record.resumed for record in seen] == [1, 2, 3] + [3] * (GENERATED_DRAWS - 3)
    assert {(record.test, record.requested, record.failed) for record in seen} == {
        ("dummy_outcome", GENERATED_DRAWS, 0)
    }


def test_w6_progress_runs_without_a_checkpoint(tmp_path: Path) -> None:
    result, arguments = _arguments("point bootstrap_stability")
    seen: list[RefutationProgress] = []
    report = _quiet(lambda: refute(result, **arguments, progress=seen.append))
    assert [record.completed for record in seen] == list(range(1, BOOTSTRAP_DRAWS + 1))
    assert support.walk(report) == support.walk(_uninterrupted("point bootstrap_stability"))


def test_w6_a_wrong_kind_of_keyword_is_refused_first() -> None:
    with pytest.raises(TypeError, match="RefutationCheckpoint"):
        refute(_in_sample(), tests=["dummy_outcome"], checkpoint="a/path")
    with pytest.raises(TypeError, match="progress must be callable"):
        refute(_in_sample(), tests=["dummy_outcome"], progress=3)


# ----------------------------------------------------------------- W7 atomic write


def test_w7_a_leftover_temporary_file_is_ignored(tmp_path: Path) -> None:
    result, arguments, checkpoint = _stored(tmp_path)
    leftover = tmp_path / "dummy_outcome" / "draw-000000004.json.123.tmp"
    leftover.write_text("{ a partial wri")
    resumed = _quiet(lambda: refute(result, **arguments, checkpoint=checkpoint))
    assert support.walk(resumed) == support.walk(_uninterrupted("point dummy_outcome"))


# ----------------------------------------------------------------- W8 variable importance


def test_w8_each_candidate_keeps_its_own_subdirectory(tmp_path: Path) -> None:
    from tests.unit._capability_sweep_support import KINDS

    build = KINDS["variable_importance"].build
    arguments: dict[str, Any] = {
        "tests": ["bootstrap_stability"],
        "n_replicates": 3,
        "random_state": 0,
    }
    uninterrupted = _quiet(lambda: build().diagnostics.refute(**arguments))
    names = list(uninterrupted)
    seen: list[RefutationProgress] = []

    def progress(record: RefutationProgress) -> None:
        seen.append(record)
        if record.candidate == names[-1] and record.completed == 2:
            raise support.Stop

    checkpoint = RefutationCheckpoint(tmp_path)
    with pytest.raises(support.Stop):
        _quiet(
            lambda: build().diagnostics.refute(
                **arguments, checkpoint=checkpoint, progress=progress
            )
        )
    assert seen[0].candidate == names[0]
    mapping = json.loads((tmp_path / "candidates.json").read_text())
    assert sorted(mapping.values()) == sorted(names)
    assert len({path.name for path in tmp_path.glob("candidate-*")}) == len(names)
    resumed = _quiet(lambda: build().diagnostics.refute(**arguments, checkpoint=checkpoint))
    assert list(resumed) == names
    for name in names:
        assert support.walk(resumed[name]) == support.walk(uninterrupted[name])


def test_w8_two_names_on_one_subdirectory_refuse(tmp_path: Path) -> None:
    checkpoint = RefutationCheckpoint(tmp_path)
    child = checkpoint.child("W1")
    digest = Path(child.directory).name.removeprefix("candidate-")
    (tmp_path / "candidates.json").write_text(json.dumps({digest: "W9"}))
    with pytest.raises(CapabilityError, match="maps candidates 'W9' and 'W1'"):
        checkpoint.child("W1")
    assert Path(checkpoint.child("../x").directory).parent == tmp_path


# ----------------------------------------------------------------- W9 the preview


def _never(record: Any) -> None:
    raise AssertionError("the preview calls no progress callback")


def test_w9_the_preview_counts_only_the_missing_draws(tmp_path: Path) -> None:
    result, arguments, checkpoint = _stored(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    full = planned_refits(result, **arguments)
    plan = planned_refits(result, **arguments, checkpoint=checkpoint, progress=_never)
    (test,) = plan.tests
    assert test.stored == 4 and test.draws == GENERATED_DRAWS
    assert test.refits == GENERATED_DRAWS - 4 == full.tests[0].refits - 4
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert "Draws already stored" in str(plan.summary())
    assert list(plan.to_frame()["stored"]) == [4]


def test_w9_the_preview_refuses_as_refute_does(tmp_path: Path) -> None:
    result, arguments, checkpoint = _stored(tmp_path)
    changed = {**arguments, "n_replicates": 12}
    expected = _refusal(lambda: refute(result, **changed, checkpoint=checkpoint))
    assert _refusal(lambda: planned_refits(result, **changed, checkpoint=checkpoint)) == expected
    # A fit that draws its seed: the preview checks the parts in refute's order, so another
    # estimator refuses there as in refute, and the same fit refuses at the seed.
    without = {key: value for key, value in arguments.items() if key != "random_state"}
    unseeded = support.in_sample(random_state=None)

    def both(target: RefutationCheckpoint) -> tuple[str, str]:
        return (
            _refusal(lambda: refute(unseeded, **without, checkpoint=target)),
            _refusal(lambda: planned_refits(unseeded, **without, checkpoint=target)),
        )

    expected, preview = both(checkpoint)
    assert preview == expected and "its estimator differs" in preview
    stopped = RefutationCheckpoint(tmp_path / "unseeded")
    with pytest.raises(support.Stop):
        _quiet(lambda: refute(unseeded, **without, checkpoint=stopped, progress=support.stopper(2)))
    expected, preview = both(stopped)
    assert preview == expected and "draws a fresh one" in preview
    fresh = RefutationCheckpoint(tmp_path / "fresh")
    assert planned_refits(result, **arguments, checkpoint=fresh).tests[0].stored == 0
    assert not (tmp_path / "fresh").exists()
