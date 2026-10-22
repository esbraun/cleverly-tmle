"""Run EconML 0.17.0 ``DynamicDML`` on the rows and folds of the study ``structural-nested-mean``.

Usage: ``run_study.py SAMPLES.csv.gz TASKS.csv OUTPUT.csv``.  The study's primary phase writes the
samples and the task table (``tests/studies/canonical_structural_nested_mean.py``).  The image
installs no ``cleverly``, so this file reads only those two inputs.

**Tasks.**  Each task row names a scenario, a variant, an estimand and its truth, and the
settings of the variant's fit: the treatment kind, the ``GLM`` order, the baseline features of
the blips, and the two treatment vectors of a contrast.  The study module is the one source of
these settings.  This runner fits each declared variant once for each replicate of its scenario.

==========  ========================================================================================
variant     fit
==========  ========================================================================================
``paired``  The aligned fit.  The state of period ``t`` is the full history ``H_t``, the folds
            are the subject's own, and the nuisances are the subject's ``GLM`` design.  Its rows
            go to ``OUTPUT``, and the driver pairs them with the subject's rows.
``markov``  The native fit.  The state of period ``t`` is ``(W, S_t, A_{t-1})``, the Markov state
            of the published estimator, and each nuisance is saturated in that state.  Its rows
            go to ``<OUTPUT stem>.markov.csv``.  The study publishes them as a descriptive table
            of the declared differences.
==========  ========================================================================================

**Nuisances.**  :class:`InteractionLinear` and :class:`IndicatorLogistic` copy the design and
the fit of ``cleverly.learners.GLM``: an intercept, the columns, and every product of two to
``order`` distinct columns in :func:`itertools.combinations` order, fitted by unpenalized
maximum likelihood.  A categorical treatment is fitted one indicator at a time, as the subject
fits it.  A ``VarianceThreshold`` step first drops the zero padding of the early periods.

**Plug-in witness.**  A g-estimator has no targeting step.  Its analogue is the residualization
by the nuisances, so each paired row's ``initial_estimate`` is the same ``DynamicDML`` fit with
constant nuisances (``DummyRegressor`` and ``DummyClassifier``), on the same folds.

**Failures** (RM39 part b).  A fit that raises a recorded fitting failure costs its replicate,
as a failed group costs an R reference run.  The rule copies ``cleverly.validation._failures``
without the ``cleverly`` classes, which the image does not install, and with ``econml`` among the
libraries.  A paired row that breaks a row rule is recorded as ``invalid_result``.  The paired
records go to ``<OUTPUT stem>.failures.csv`` and the Markov records to
``<OUTPUT stem>.markov-failures.csv``.  The run stops when the first 20 replicates of one
scenario and variant all failed, when every replicate failed, and on any other error.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

#: The normal quantile of a 95% interval, as ``scipy.stats.norm.ppf(0.975)`` gives it.
Z_95 = 1.959963984540054

#: The variants this runner fits, and the file stem suffix that holds each one's rows.
PAIRED = "paired"
MARKOV = "markov"
VARIANTS = (PAIRED, MARKOV)

#: The implementation name each row carries.
IMPLEMENTATION = "econml"

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
LIBRARY_MODULES: tuple[str, ...] = ("numpy", "scipy", "sklearn", "statsmodels", "econml")

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

#: The task-table columns the study writes.
TASK_COLUMNS = (
    "scenario",
    "variant",
    "estimand",
    "truth",
    "kind",
    "nodes",
    "order",
    "features",
    "t1",
    "t0",
)

#: The row columns of ``OUTPUT``, in the order of the study's replicate schema.
ROW_COLUMNS = (
    "implementation",
    "scenario",
    "replicate",
    "n",
    "estimand",
    "truth",
    "estimate",
    "inference_estimate",
    "std_error",
    "ci_lower",
    "ci_upper",
    "inference_scale",
    "covered",
    "initial_estimate",
)

#: A linear predictor beyond this magnitude puts a fitted probability within about 1e-13 of 0
#: or 1.  ``cleverly.learners.GLM`` refuses such a fit as separated; this copy records it.
SEPARATION_ETA = 30.0
#: The Newton steps and the tolerance of the logistic fit.
NEWTON_STEPS = 100
NEWTON_TOLERANCE = 1e-10


# ------------------------------------------------------------------------------- nuisances


def interaction_design(x: Any, order: int) -> np.ndarray:
    """An intercept, the columns of ``x``, and each product of 2 to ``order`` distinct columns.

    The order of the columns is that of ``cleverly.learners.GLM.full_design``.
    """
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    blocks = [np.ones((matrix.shape[0], 1)), matrix]
    for size in range(2, int(order) + 1):
        for chosen in combinations(range(matrix.shape[1]), size):
            blocks.append(np.prod(matrix[:, list(chosen)], axis=1, keepdims=True))
    return np.hstack(blocks)


def _expit(eta: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * eta)), dtype=float)


def logistic_coefficients(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The unpenalized logistic MLE of ``target`` (0/1) on ``design``, by Newton steps.

    Each step solves its weighted least-squares problem by ``lstsq``, so an aliased column takes
    the minimum-norm share and the fitted probabilities stay unique.  A fit whose linear
    predictor leaves ``[-30, 30]`` is separated, and it raises ``LinAlgError``.
    """
    coef = np.zeros(design.shape[1])
    for _ in range(NEWTON_STEPS):
        eta = design @ coef
        mu = _expit(eta)
        weight = np.maximum(mu * (1.0 - mu), 1e-12)
        root = np.sqrt(weight)
        step = np.linalg.lstsq(design * root[:, None], (target - mu) / root, rcond=None)[0]
        coef = coef + step
        if np.max(np.abs(step)) < NEWTON_TOLERANCE:
            break
    else:
        raise np.linalg.LinAlgError("the logistic fit did not converge")
    if np.max(np.abs(design @ coef)) > SEPARATION_ETA:
        raise np.linalg.LinAlgError("the logistic fit is separated: a fitted probability is 0 or 1")
    return coef


class InteractionLinear(BaseEstimator, RegressorMixin):
    """Least squares on :func:`interaction_design`: the gaussian ``GLM`` of ``cleverly``.

    Parameters
    ----------
    order : int
        The highest order of the products.
    """

    def __init__(self, order: int = 1) -> None:
        self.order = order

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> InteractionLinear:
        """Fit the coefficients by least squares."""
        design = interaction_design(x, self.order)
        self.coef_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]
        return self

    def predict(self, x: Any) -> Any:
        """Return the fitted means."""
        return interaction_design(x, self.order) @ self.coef_


class IndicatorLogistic(BaseEstimator, ClassifierMixin):
    """One logistic fit of each non-reference indicator on :func:`interaction_design`.

    ``cleverly`` fits a categorical treatment's nuisance one coding column at a time, so this
    copy does the same.  ``predict_proba`` puts one minus the sum of the fitted indicators in
    the reference column, which ``DynamicDML`` drops.  A training fold with one class has a
    constant fit.

    Parameters
    ----------
    order : int
        The highest order of the products.
    """

    def __init__(self, order: int = 1) -> None:
        self.order = order

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> IndicatorLogistic:
        """Fit one logistic regression per non-reference class."""
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        design = interaction_design(x, self.order)
        self.coef_ = []
        for level in self.classes_[1:]:
            target = (labels == level).astype(float)
            self.coef_.append(logistic_coefficients(design, target))
        return self

    def predict_proba(self, x: Any) -> Any:
        """Return the reference column and one fitted probability per indicator."""
        design = interaction_design(x, self.order)
        columns = [_expit(design @ coef) for coef in self.coef_]
        rest = np.column_stack(columns) if columns else np.zeros((design.shape[0], 0))
        return np.column_stack([1.0 - rest.sum(axis=1), rest])

    def predict(self, x: Any) -> Any:
        """Return the class with the largest fitted probability."""
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


# -------------------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class Task:
    """The settings of one variant of one scenario, and the estimands it reports.

    Parameters
    ----------
    scenario : str
        The scenario.
    variant : str
        ``"paired"`` or ``"markov"``.
    kind : str
        ``"binary"``, ``"categorical"`` or ``"continuous"``: every node's treatment type.
    nodes : int
        The number of treatment nodes.
    order : int
        The ``GLM`` order of every nuisance.
    features : tuple of str
        The baseline columns of the blips; empty for ``phi = 1``.
    estimands : tuple of tuple
        ``(estimand, truth, t1, t0)``; ``t1`` and ``t0`` are ``None`` for a blip.
    """

    scenario: str
    variant: str
    kind: str
    nodes: int
    order: int
    features: tuple[str, ...]
    estimands: tuple[tuple[str, float, tuple[float, ...] | None, tuple[float, ...] | None], ...]


def _vector(text: Any) -> tuple[float, ...] | None:
    if not isinstance(text, str) or not text:
        return None
    return tuple(float(part) for part in text.split(";"))


def read_tasks(frame: pd.DataFrame) -> dict[tuple[str, str], Task]:
    """The tasks of the task table, by ``(scenario, variant)``."""
    missing = [column for column in TASK_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"the task table lacks the columns {missing}")
    out: dict[tuple[str, str], Task] = {}
    for (scenario, variant), group in frame.groupby(["scenario", "variant"], sort=True):
        if variant not in VARIANTS:
            raise RuntimeError(f"unknown variant {variant!r} for {scenario}")
        settings = group[["kind", "nodes", "order", "features"]].drop_duplicates()
        if len(settings) != 1:
            raise RuntimeError(f"{scenario}/{variant} has more than one setting")
        kind, nodes, order, features = settings.iloc[0]
        text = "" if not isinstance(features, str) else features
        out[(str(scenario), str(variant))] = Task(
            scenario=str(scenario),
            variant=str(variant),
            kind=str(kind),
            nodes=int(nodes),
            order=int(order),
            features=tuple(part for part in text.split(";") if part),
            estimands=tuple(
                (str(row.estimand), float(row.truth), _vector(row.t1), _vector(row.t0))
                for row in group.itertuples(index=False)
            ),
        )
    return out


def _coding(kind: str, values: np.ndarray, levels: int) -> np.ndarray:
    """The coding of a treatment column: drop-first indicators, or the value itself."""
    if kind == "categorical":
        return np.column_stack([(values == level).astype(float) for level in range(1, levels)])
    return values.reshape(-1, 1).astype(float)


def _levels(kind: str, frame: pd.DataFrame, nodes: int) -> int:
    if kind != "categorical":
        return 2
    return int(max(frame[f"A{t}"].max() for t in range(1, nodes + 1))) + 1


def states(task: Task, frame: pd.DataFrame) -> np.ndarray:
    """The state rows of the long panel, ``(n * nodes, columns)``, zero-padded.

    Row ``nodes * i + t - 1`` is unit ``i`` at period ``t``.  The paired state is ``[W, S_1, ...,
    S_t]`` and the codings of ``A_1, ..., A_{t-1}``.  The Markov state is ``[W, S_t]`` and the
    coding of ``A_{t-1}``.  A column a period has not observed yet is zero there.
    """
    n, nodes = len(frame), task.nodes
    levels = _levels(task.kind, frame, nodes)
    w = frame["W"].to_numpy(dtype=float).reshape(-1, 1)
    covariates = [frame[f"S{t}"].to_numpy(dtype=float) for t in range(1, nodes + 1)]
    codings = [
        _coding(task.kind, frame[f"A{t}"].to_numpy(dtype=float), levels)
        for t in range(1, nodes + 1)
    ]
    width = codings[0].shape[1]
    blocks = []
    for t in range(1, nodes + 1):
        if task.variant == PAIRED:
            seen = np.column_stack([covariates[s] if s < t else np.zeros(n) for s in range(nodes)])
            past = [codings[s] if s < t - 1 else np.zeros((n, width)) for s in range(nodes - 1)]
            blocks.append(np.hstack([w, seen, *past]))
        else:
            previous = codings[t - 2] if t > 1 else np.zeros((n, width))
            blocks.append(np.hstack([w, covariates[t - 1].reshape(-1, 1), previous]))
    stacked = np.stack(blocks, axis=1)
    return stacked.reshape(n * nodes, -1)


def splits(folds: np.ndarray, nodes: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The subject's unit folds as ``DynamicDML``'s ``cv`` iterable over the long rows."""
    out = []
    for fold in np.unique(folds):
        units = (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        out.append(
            tuple(
                np.sort(np.concatenate([nodes * chosen + t for t in range(nodes)]))
                for chosen in units
            )
        )
    return out  # type: ignore[return-value]


def _models(task: Task, constant: bool) -> tuple[Any, Any]:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import make_pipeline

    if constant:
        treatment = DummyRegressor() if task.kind == "continuous" else DummyClassifier()
        return DummyRegressor(), treatment
    order = task.order
    outcome = make_pipeline(VarianceThreshold(0.0), InteractionLinear(order))
    if task.kind == "continuous":
        treatment: Any = make_pipeline(VarianceThreshold(0.0), InteractionLinear(order))
    else:
        treatment = make_pipeline(VarianceThreshold(0.0), IndicatorLogistic(order))
    return outcome, treatment


def fit_dynamic_dml(task: Task, frame: pd.DataFrame, *, constant: bool = False) -> Any:
    """Fit ``DynamicDML`` on one replicate under ``task``'s settings."""
    from econml.panel.dml import DynamicDML

    n, nodes = len(frame), task.nodes
    model_y, model_t = _models(task, constant)
    features = (
        np.repeat(frame[list(task.features)].to_numpy(dtype=float), nodes, axis=0)
        if task.features
        else None
    )
    treatment = np.column_stack(
        [frame[f"A{t}"].to_numpy(dtype=float) for t in range(1, nodes + 1)]
    ).reshape(-1)
    estimator = DynamicDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=task.kind != "continuous",
        cv=splits(frame["fold"].to_numpy(dtype=int), nodes),
        random_state=0,
    )
    with warnings.catch_warnings():
        # DynamicDML warns that a cv iterable ignores its own grouping; the folds here are the
        # subject's own unit folds, so the warning names no change.
        warnings.simplefilter("ignore")
        estimator.fit(
            np.repeat(frame["Y"].to_numpy(dtype=float), nodes),
            treatment,
            X=features,
            W=states(task, frame),
            groups=np.repeat(np.arange(n), nodes),
        )
    return estimator


def _blip_position(task: Task, estimand: str, levels: int) -> tuple[int, str]:
    """The coefficient row of a blip estimand, and its feature (``intercept`` or a column)."""
    inner = estimand[len("blip[") : -1]
    treatment, level, feature = inner.split(":")
    node = int(treatment[1:])
    width = levels - 1 if task.kind == "categorical" else 1
    offset = 0 if level == "dose" or task.kind != "categorical" else int(float(level)) - 1
    return (node - 1) * width + offset, feature


def estimates(
    task: Task, estimator: Any, frame: pd.DataFrame, *, inference: bool = True
) -> list[tuple[str, float, float, float, float, float]]:
    """``(estimand, truth, estimate, std_error, ci_lower, ci_upper)`` of each task estimand."""
    levels = _levels(task.kind, frame, task.nodes)
    features = frame[list(task.features)].to_numpy(dtype=float) if task.features else None
    intercept = np.asarray(estimator.intercept_, dtype=float).reshape(-1)
    coef = (
        np.asarray(estimator.coef_, dtype=float).reshape(len(intercept), -1)
        if task.features
        else np.zeros((len(intercept), 0))
    )
    if inference:
        intercept_se = np.asarray(estimator.intercept__inference().stderr, dtype=float).reshape(-1)
        coef_se = (
            np.asarray(estimator.coef__inference().stderr, dtype=float).reshape(coef.shape)
            if task.features
            else coef
        )
    out = []
    for estimand, truth, t1, t0 in task.estimands:
        if estimand.startswith("blip["):
            row, feature = _blip_position(task, estimand, levels)
            if feature == "intercept":
                point = float(intercept[row])
                se = float(intercept_se[row]) if inference else float("nan")
            else:
                column = task.features.index(feature)
                point = float(coef[row, column])
                se = float(coef_se[row, column]) if inference else float("nan")
        elif estimand.startswith("ate_regimen["):
            if t1 is None or t0 is None:
                raise RuntimeError(f"{estimand} of {task.scenario} has no treatment vectors")
            # One treatment row per feature row: the contrast averages the effect over X.
            count = 1 if features is None else len(features)
            one = np.tile(np.asarray(t1, dtype=float), (count, 1))
            zero = np.tile(np.asarray(t0, dtype=float), (count, 1))
            if inference:
                result = estimator.ate_inference(X=features, T0=zero, T1=one)
                point = float(np.asarray(result.mean_point).reshape(-1)[0])
                se = float(np.asarray(result.stderr_mean).reshape(-1)[0])
            else:
                point = float(np.asarray(estimator.ate(X=features, T0=zero, T1=one)).reshape(-1)[0])
                se = float("nan")
        else:
            raise RuntimeError(f"DynamicDML reports no {estimand}")
        out.append((estimand, truth, point, se, point - Z_95 * se, point + Z_95 * se))
    return out


def rows_of(task: Task, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of one fit: one per task estimand, with the constant-nuisance witness."""
    estimator = fit_dynamic_dml(task, frame)
    fitted = estimates(task, estimator, frame)
    initial: dict[str, float] = {}
    if task.variant == PAIRED:
        witness = fit_dynamic_dml(task, frame, constant=True)
        initial = {
            name: value for name, _, value, *_ in estimates(task, witness, frame, inference=False)
        }
    replicate = int(frame["replicate"].iloc[0])
    out = []
    for estimand, truth, point, se, low, high in fitted:
        out.append(
            {
                "implementation": IMPLEMENTATION,
                "scenario": task.scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": estimand,
                "truth": truth,
                "estimate": point,
                "inference_estimate": point,
                "std_error": se,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": initial.get(estimand, float("nan")),
            }
        )
    return out


# ------------------------------------------------------------------------------- failures


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


def _record(
    group_id: str, task: Task, replicate: int, kind: str, stage: str, call: str, text: str
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "scenarios": task.scenario,
        "replicates": str(replicate),
        "attempt": 1,
        "class": kind,
        "stage": stage,
        "call": call,
        "message": text.replace("\r", " ").replace("\n", " "),
        "seed": "NA",
    }


#: One unit of work: the group id, the task and the replicate's frame.
Job = tuple[str, Task, pd.DataFrame]


def attempt(job: Job) -> tuple[str, Any]:
    """One fit's rows, or its failure record.  It runs where the fit ran, so the
    classification reads the fit's own frames."""
    group_id, task, frame = job
    replicate = int(frame["replicate"].iloc[0])
    try:
        rows = rows_of(task, frame)
    except Exception as error:
        if not recordable(error):
            raise
        message = f"{type(error).__name__}: {error}"
        stage = f"dynamic dml {task.variant} fit"
        return "failure", _record(
            group_id, task, replicate, "fitting_failure", stage, failure_call(error), message
        )
    for row in rows:
        broken = invalid(row)
        if broken is not None:
            return "failure", _record(
                group_id, task, replicate, "invalid_result", "row check", "none", broken
            )
    return "rows", rows


def _map(jobs: Sequence[Job], cores: int) -> list[tuple[str, Any]]:
    if cores == 1:
        return [attempt(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=cores) as pool:
        return list(pool.map(attempt, jobs, chunksize=4))


def fit_all(
    jobs: Sequence[Job], cores: int
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """The rows and the failure records of every job, by variant, in job order.

    The first 20 jobs of each scenario and variant run first, and the run stops when all of one
    scenario and variant's 20 failed.
    """
    by_cell: dict[tuple[str, str], list[int]] = {}
    for index, (_, task, _) in enumerate(jobs):
        by_cell.setdefault((task.scenario, task.variant), []).append(index)
    first = sorted(index for members in by_cell.values() for index in members[:SYSTEMATIC_PREFIX])
    done = dict(zip(first, _map([jobs[index] for index in first], cores), strict=True))
    for (scenario, variant), members in by_cell.items():
        prefix = members[:SYSTEMATIC_PREFIX]
        if len(prefix) == SYSTEMATIC_PREFIX and all(done[i][0] == "failure" for i in prefix):
            record = done[prefix[0]][1]
            raise RuntimeError(
                f"systematic: the first {SYSTEMATIC_PREFIX} replicates of {scenario} "
                f"({variant}) failed; first: {record['class']} in {record['stage']} "
                f"(group {record['group_id']}): {record['message']}"
            )
    rest = [index for index in range(len(jobs)) if index not in done]
    done.update(zip(rest, _map([jobs[index] for index in rest], cores), strict=True))
    out: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
        variant: ([], []) for variant in VARIANTS
    }
    for index in range(len(jobs)):
        kind, value = done[index]
        rows, failures = out[jobs[index][1].variant]
        if kind == "rows":
            rows.extend(value)
        else:
            failures.append(value)
    for variant, (rows, failures) in out.items():
        if failures and not rows:
            raise RuntimeError(f"every {variant} fit failed; first: {failures[0]['message']}")
    return out


def jobs_of(samples: pd.DataFrame, tasks: dict[tuple[str, str], Task]) -> list[Job]:
    """One job per declared task and replicate.  Group ids follow the harness's
    ``g000000001`` form, over ``(scenario, replicate, variant)`` in sorted order."""
    out: list[Job] = []
    scenarios = {scenario for scenario, _ in tasks}
    unknown = sorted(set(samples["scenario"].astype(str)) - scenarios)
    if unknown:
        raise RuntimeError(f"the samples hold scenarios with no task: {unknown}")
    for (scenario, _replicate), frame in samples.groupby(["scenario", "replicate"], sort=True):
        frame = frame.dropna(axis=1, how="all").reset_index(drop=True)
        for variant in VARIANTS:
            task = tasks.get((str(scenario), variant))
            if task is not None:
                out.append((f"g{len(out) + 1:09d}", task, frame))
    return out


def _cores(jobs: int) -> int:
    raw = os.getenv("CLEVERLY_REFERENCE_CORES", os.getenv("CLEVERLY_R_CORES", "1"))
    try:
        requested = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid reference core count {raw!r}") from error
    return max(1, min(jobs, requested))


def sibling(output: Path, suffix: str) -> Path:
    """The file beside ``output`` that holds ``suffix``: ``<stem>.<suffix>.csv``."""
    return output.with_name(f"{output.stem}.{suffix}.csv")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_study.py SAMPLES.csv.gz TASKS.csv OUTPUT.csv")
    samples_path, tasks_path, output_path = map(Path, sys.argv[1:])
    tasks = read_tasks(pd.read_csv(tasks_path, keep_default_na=False, na_values=[""]))
    samples = pd.read_csv(samples_path)
    jobs = jobs_of(samples, tasks)
    del samples
    results = fit_all(jobs, _cores(len(jobs)))
    expected = dict.fromkeys(VARIANTS, 0)
    for _, task, _ in jobs:
        expected[task.variant] += 1
    targets = {
        PAIRED: (output_path, sibling(output_path, "failures")),
        MARKOV: (sibling(output_path, MARKOV), sibling(output_path, "markov-failures")),
    }
    for variant, (rows, failures) in results.items():
        counts = pd.Series([(row["scenario"], row["replicate"]) for row in rows]).value_counts()
        if len(counts) + len(failures) != expected[variant]:
            raise RuntimeError(
                f"DynamicDML returned {len(counts)} {variant} fits and {len(failures)} failures "
                f"for {expected[variant]} jobs"
            )
        wrong = [
            key for key, count in counts.items() if count != len(tasks[(key[0], variant)].estimands)
        ]
        if wrong:
            raise RuntimeError(f"{variant} fits with a row count other than their tasks: {wrong}")
        rows_path, failures_path = targets[variant]
        failures_path.unlink(missing_ok=True)
        if failures:
            for record in failures:
                print(
                    f"recorded {record['class']} for {variant} group {record['group_id']}: "
                    f"{record['message']}"
                )
            pd.DataFrame(failures, columns=list(FAILURE_COLUMNS)).to_csv(failures_path, index=False)
        pd.DataFrame(rows, columns=list(ROW_COLUMNS)).to_csv(rows_path, index=False, na_rep="NA")


if __name__ == "__main__":
    main()
