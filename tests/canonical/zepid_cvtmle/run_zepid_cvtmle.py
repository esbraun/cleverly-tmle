"""Run zEpid's native single-crossfit TMLE on the study's exact rows and folds.

RM39 part b: a fit that raises a recorded fitting failure costs its replicate, as a failed
group costs an R reference run (``tests/canonical/study_harness.R``).  The runner writes one
record per failed replicate to ``<output stem>.failures.csv``.  The rule copies
``cleverly.validation._failures``, without the ``cleverly`` classes, which the image does not
install, and with ``zepid`` among the libraries.  A row that breaks a row rule is recorded as
``invalid_result``.  The run stops when the first 20 replicates of a scenario all failed, when
every replicate failed, and on any other error.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd

#: Always fatal; checked first.  ``cleverly.validation._failures.FATAL`` without its
#: ``cleverly`` classes.
FATAL: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    MemoryError,
    TypeError,
    AttributeError,
    NameError,
    ImportError,
    AssertionError,
)

#: Recorded wherever raised.  ``cleverly.validation._failures.RECORDED`` without its
#: ``cleverly`` classes.
RECORDED: tuple[type[BaseException], ...] = (
    np.linalg.LinAlgError,
    FloatingPointError,
    ZeroDivisionError,
    OverflowError,
)

#: Packages whose own plain ``ValueError`` or ``RuntimeError`` is a fitting failure.
LIBRARY_MODULES: tuple[str, ...] = ("numpy", "scipy", "sklearn", "statsmodels", "zepid")

#: The R harness's systematic stop: the first 20 groups of one scenario all failed.
SYSTEMATIC_PREFIX = 20

FAILURE_COLUMNS = (
    "group_id",
    "scenarios",
    "replicates",
    "attempt",
    "class",
    "stage",
    "call",
    "message",
    "seed",
)


def _innermost(traceback: TracebackType | None) -> TracebackType | None:
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback


def recordable(error: BaseException) -> bool:
    """Whether ``error`` costs its replicate rather than the run."""
    if isinstance(error, FATAL):
        return False
    if isinstance(error, RECORDED):
        return True
    if type(error) in (ValueError, RuntimeError):
        innermost = _innermost(error.__traceback__)
        module = "" if innermost is None else str(innermost.tb_frame.f_globals.get("__name__", ""))
        return any(module == name or module.startswith(f"{name}.") for name in LIBRARY_MODULES)
    return False


def failure_call(error: BaseException) -> str:
    """``module:function`` of the frame that raised ``error``."""
    innermost = _innermost(error.__traceback__)
    if innermost is None:
        return ""
    frame = innermost.tb_frame
    return f"{frame.f_globals.get('__name__', '')}:{frame.f_code.co_name}"


def invalid(row: dict[str, Any]) -> str | None:
    """The first row rule ``row`` breaks: the harness's ``study_invalid`` rules."""
    where = f"{row['estimand']} of {row['scenario']} (row 1)"
    for column in ("estimate", "inference_estimate", "std_error"):
        if not np.isfinite(float(row[column])):
            return f"non-finite {column} for {where}"
    if not float(row["std_error"]) > 0:
        return f"non-positive std_error for {where}"
    if not float(row["ci_lower"]) <= float(row["ci_upper"]):
        return f"ci_lower above ci_upper for {where}"
    inside = float(row["ci_lower"]) <= float(row["truth"]) <= float(row["ci_upper"])
    if int(row["covered"]) != int(inside):
        return f"covered disagrees with the interval and truth for {where}"
    return None


def assert_native_split_identity(
    frame: pd.DataFrame, native_splits: Sequence[pd.DataFrame]
) -> None:
    """Abort unless every native split contains the declared row identities."""
    labels = sorted(frame["fold"].unique().tolist())
    if labels != list(range(len(native_splits))):
        raise RuntimeError(
            f"declared fold labels {labels} do not identify {len(native_splits)} native splits"
        )
    for fold, split in enumerate(native_splits):
        expected = set(frame.loc[frame["fold"] == fold, "row_id"].astype(int))
        observed = set(split["row_id"].astype(int))
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            unexpected = sorted(observed - expected)[:5]
            raise RuntimeError(
                "zEpid native split changed the declared row identities for "
                f"fold {fold}: missing={missing}, unexpected={unexpected}"
            )


def _fit_one(payload: tuple[pd.DataFrame, float]) -> dict[str, Any]:
    frame, truth = payload
    from sklearn.linear_model import LogisticRegression
    from zepid.causal.doublyrobust import SingleCrossfitTMLE
    from zepid.causal.doublyrobust import crossfit as crossfit_module

    scenario = str(frame["scenario"].iloc[0])
    replicate = int(frame["replicate"].iloc[0])
    partition_random_state = int(frame["partition_random_state"].iloc[0])
    native_split = crossfit_module._sample_split_
    native_targeting = crossfit_module.targeting_step
    initial_estimates: list[float] = []

    def checked_split(data: pd.DataFrame, n_splits: int, random_state: int) -> Any:
        splits = native_split(data, n_splits=n_splits, random_state=random_state)
        assert_native_split_identity(data, splits)
        return splits

    def recorded_targeting(*args: Any, **kwargs: Any) -> Any:
        py_a = kwargs["py_a"] if "py_a" in kwargs else args[2]
        py_n = kwargs["py_n"] if "py_n" in kwargs else args[3]
        initial_estimates.append(float(np.mean(np.asarray(py_a) - np.asarray(py_n))))
        return native_targeting(*args, **kwargs)

    crossfit_module._sample_split_ = checked_split
    crossfit_module.targeting_step = recorded_targeting
    try:
        fit = SingleCrossfitTMLE(frame.copy(), exposure="A", outcome="Y", alpha=0.05)
        learner = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
        covariates = " + ".join(column for column in frame if column.startswith("W"))
        fit.exposure_model(covariates, estimator=learner, bound=[0.025, 0.975])
        fit.outcome_model(f"A + {covariates}", estimator=learner)
        fit.fit(
            n_splits=2,
            n_partitions=1,
            method="median",
            random_state=partition_random_state,
        )
    finally:
        crossfit_module._sample_split_ = native_split
        crossfit_module.targeting_step = native_targeting

    if len(initial_estimates) != 1:
        raise RuntimeError(
            "the one-partition zEpid fit did not expose exactly one pre-targeting plug-in"
        )

    estimate = float(fit.risk_difference)
    std_error = float(fit.risk_difference_se)
    low, high = (float(value) for value in fit.risk_difference_ci)
    return {
        "implementation": "zepid-single-crossfit-tmle",
        "scenario": scenario,
        "replicate": replicate,
        "n": len(frame),
        "estimand": "ate",
        "truth": truth,
        "estimate": estimate,
        "inference_estimate": estimate,
        "std_error": std_error,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": initial_estimates[0],
    }


def _record(
    group_id: str, payload: tuple[pd.DataFrame, float], kind: str, stage: str, call: str, text: str
) -> dict[str, Any]:
    frame = payload[0]
    return {
        "group_id": group_id,
        "scenarios": str(frame["scenario"].iloc[0]),
        "replicates": str(int(frame["replicate"].iloc[0])),
        "attempt": 1,
        "class": kind,
        "stage": stage,
        "call": call,
        "message": text.replace("\r", " ").replace("\n", " "),
        "seed": "NA",
    }


def _attempt(task: tuple[str, tuple[pd.DataFrame, float]]) -> tuple[str, dict[str, Any]]:
    """One replicate's row, or its failure record.  It runs where the fit ran, so the
    classification reads the fit's own frames."""
    group_id, payload = task
    try:
        row = _fit_one(payload)
    except Exception as error:
        if not recordable(error):
            raise
        message = f"{type(error).__name__}: {error}"
        record = _record(
            group_id, payload, "fitting_failure", "zepid fit", failure_call(error), message
        )
        return "failure", record
    broken = invalid(row)
    if broken is not None:
        return "failure", _record(group_id, payload, "invalid_result", "row check", "none", broken)
    return "row", row


def _map(tasks: list[Any], cores: int) -> list[tuple[str, dict[str, Any]]]:
    if cores == 1:
        return [_attempt(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=cores) as pool:
        return list(pool.map(_attempt, tasks))


def fit_all(
    groups: list[tuple[pd.DataFrame, float]], cores: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows and the failure records of every group, in group order.

    Groups take the harness's ids ``g000000001, ...``.  The first 20 groups of each scenario
    run first, and the run stops when all of one scenario's 20 failed.
    """
    tasks = [(f"g{index:09d}", group) for index, group in enumerate(groups, start=1)]
    by_scenario: dict[str, list[int]] = {}
    for index, (_, (frame, _)) in enumerate(tasks):
        by_scenario.setdefault(str(frame["scenario"].iloc[0]), []).append(index)
    first = sorted(
        index for members in by_scenario.values() for index in members[:SYSTEMATIC_PREFIX]
    )
    done = dict(zip(first, _map([tasks[index] for index in first], cores), strict=True))
    for scenario, members in by_scenario.items():
        prefix = members[:SYSTEMATIC_PREFIX]
        if len(prefix) == SYSTEMATIC_PREFIX and all(
            done[index][0] == "failure" for index in prefix
        ):
            record = done[prefix[0]][1]
            raise RuntimeError(
                f"systematic: the first {SYSTEMATIC_PREFIX} groups of {scenario} failed; first: "
                f"{record['class']} in {record['stage']} (group {record['group_id']}): "
                f"{record['message']}"
            )
    rest = [index for index in range(len(tasks)) if index not in done]
    done.update(zip(rest, _map([tasks[index] for index in rest], cores), strict=True))
    outcomes = [done[index] for index in range(len(tasks))]
    rows = [value for kind, value in outcomes if kind == "row"]
    failures = [value for kind, value in outcomes if kind == "failure"]
    if failures and not rows:
        raise RuntimeError(f"every group failed; first: {failures[0]['message']}")
    return rows, failures


def _cores(groups: int) -> int:
    raw = os.getenv("CLEVERLY_REFERENCE_CORES", os.getenv("CLEVERLY_R_CORES", "1"))
    try:
        requested = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid reference core count {raw!r}") from error
    return max(1, min(groups, requested))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_zepid_cvtmle.py SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
    samples_path, truths_path, output_path = map(Path, sys.argv[1:])
    samples = pd.read_csv(samples_path)
    truths = pd.read_csv(truths_path)
    truth_lookup = {
        (str(row.scenario), int(row.replicate)): float(row.truth_ate) for row in truths.itertuples()
    }
    groups = [
        (frame.copy(), truth_lookup[(str(scenario), int(replicate))])
        for (scenario, replicate), frame in samples.groupby(["scenario", "replicate"], sort=True)
    ]
    rows, failures = fit_all(groups, _cores(len(groups)))
    if len(rows) + len(failures) != len(groups):
        raise RuntimeError(
            f"zEpid returned {len(rows)} rows and {len(failures)} failures for {len(groups)} "
            "replications"
        )
    failures_path = output_path.with_name(f"{output_path.stem}.failures.csv")
    failures_path.unlink(missing_ok=True)
    if failures:
        for record in failures:
            print(f"recorded {record['class']} for group {record['group_id']}: {record['message']}")
        pd.DataFrame(failures, columns=list(FAILURE_COLUMNS)).to_csv(failures_path, index=False)
    pd.DataFrame(rows).to_csv(output_path, index=False, na_rep="NA")


if __name__ == "__main__":
    main()
