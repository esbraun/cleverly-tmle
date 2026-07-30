"""Uniform fitting and prediction across heterogeneous learners.

Nuisance models can be bare scikit-learn estimators, pipelines, grid searches or
a :class:`~cleverly.learners.SuperLearner`.  Two operations need to work the same
way for all of them: passing observation weights, and getting back a conditional
*mean* (a probability for a binary target).  Both are handled here so the
estimator code never branches on learner type.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Literal

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .._typing import FloatArray, IntArray, Learner
from ._threads import thread_limit

__all__ = [
    "Task",
    "accepts_groups",
    "fit_learner",
    "predict_mean",
    "supports_sample_weight",
]

Task = Literal["regression", "classification"]


def _final_estimator(estimator: Learner) -> tuple[Learner, str | None]:
    """The estimator that actually consumes ``sample_weight``, and its step name."""
    if isinstance(estimator, Pipeline):
        name, step = estimator.steps[-1]
        return step, name
    return estimator, None


def _qualified(param: str, step_name: str | None) -> str:
    """The name a fit parameter must take to reach a pipeline's final step.

    Screening wraps a learner in a :class:`~sklearn.pipeline.Pipeline`
    (:func:`cleverly.estimators._nuisance._screened`), and a bare ``groups=`` handed to
    a pipeline is not an error -- it is silently ignored.  Every fit parameter therefore
    goes through here rather than being spelled out at the call site, so a wrapped
    learner cannot quietly stop receiving one.
    """
    return param if step_name is None else f"{step_name}__{param}"


def supports_sample_weight(estimator: Learner) -> bool:
    """Whether ``estimator.fit`` (or its pipeline's final step) takes weights."""
    target, _ = _final_estimator(estimator)
    fit = getattr(target, "fit", None)
    if fit is None:
        return False
    try:
        signature = inspect.signature(fit)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    if "sample_weight" in signature.parameters:
        return True
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def accepts_groups(estimator: Learner) -> bool:
    """Whether ``estimator.fit`` (or its pipeline's final step) takes cluster codes.

    Unlike :func:`supports_sample_weight` this does **not** count a ``**kwargs`` fit as
    accepting them.  Weights may safely be offered to a learner that swallows unknown
    keywords -- the worst case is that they are ignored, and the caller is warned.
    Cluster codes are different: a learner that appears to take them but discards them
    would report inner folds that look cluster-respecting and are not, which is the
    failure this routing exists to prevent.  So ``groups`` travels only to a learner
    that names it.
    """
    target, _ = _final_estimator(estimator)
    fit = getattr(target, "fit", None)
    if fit is None:
        return False
    try:
        signature = inspect.signature(fit)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "groups" in signature.parameters


def fit_learner(
    estimator: Learner,
    x: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray | None = None,
    *,
    groups: IntArray | None = None,
    copy: bool = True,
    warn_unweighted: bool = True,
) -> Learner:
    """Fit ``estimator``, routing weights and cluster codes to wherever they belong.

    Returns a *fitted clone* by default so a learner passed by the user is never
    mutated and can be reused across folds.  A learner that cannot accept weights
    is fitted without them and warns once per call, rather than silently
    discarding a weighting scheme the user asked for.

    ``groups`` reaches any learner whose ``fit`` names it -- a
    :class:`~cleverly.learners.SuperLearner`, whose inner folds must keep a cluster
    intact, but also a wrapped search whose own cross-validation should respect the
    same structure.  It is dropped for a learner that has no use for it.
    """
    model = clone(estimator) if copy else estimator
    _, step_name = _final_estimator(model)
    params: dict[str, Any] = {}

    if groups is not None and accepts_groups(model):
        params[_qualified("groups", step_name)] = np.asarray(groups)

    if sample_weight is not None:
        if supports_sample_weight(model):
            weights = np.asarray(sample_weight, dtype=float)
            params[_qualified("sample_weight", step_name)] = weights
        elif warn_unweighted:
            warnings.warn(
                f"{type(model).__name__} does not accept sample_weight; fitting it unweighted. "
                "Observation weights will still be applied in the targeting step, but the "
                "nuisance fit ignores them.",
                UserWarning,
                stacklevel=2,
            )

    with thread_limit():
        model.fit(x, y, **params)
    return model


def predict_mean(estimator: Learner, x: FloatArray, task: Task) -> FloatArray:
    """Predicted conditional mean of the target at ``x``.

    For ``task="classification"`` this is ``P(y = 1 | x)``, read off
    ``predict_proba`` with the class ordering resolved explicitly.  A model fit on
    a degenerate fold (a single observed class) returns that class as a constant
    instead of raising, which keeps a cross-fit from failing outright on rare
    outcomes.
    """
    matrix = np.asarray(x, dtype=float)
    if task == "regression":
        with thread_limit():
            return np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)

    if hasattr(estimator, "predict_proba"):
        with thread_limit():
            proba = np.asarray(estimator.predict_proba(matrix), dtype=float)
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]), dtype=float).reshape(-1)
        if proba.ndim == 1:  # pragma: no cover - non-standard estimator
            return proba.reshape(-1)
        if proba.shape[1] == 1:
            constant = 1.0 if classes.size and classes[0] == 1.0 else 0.0
            return np.full(matrix.shape[0], constant, dtype=float)
        positive = np.flatnonzero(classes == 1.0)
        column = int(positive[0]) if positive.size else proba.shape[1] - 1
        return np.asarray(proba[:, column], dtype=float)

    if hasattr(estimator, "decision_function"):  # pragma: no cover - margin-only models
        from ..utils.bounds import expit

        with thread_limit():
            margin = np.asarray(estimator.decision_function(matrix), dtype=float).reshape(-1)
        return expit(margin)

    with thread_limit():
        return np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)


def infer_task(y: FloatArray) -> Task:
    """``"classification"`` for a 0/1 target, ``"regression"`` otherwise."""
    values = np.asarray(y, dtype=float)
    unique = np.unique(values[np.isfinite(values)])
    if unique.size <= 2 and np.all(np.isin(unique, (0.0, 1.0))):
        return "classification"
    return "regression"


def as_target(y: FloatArray, task: Task) -> Any:
    """Coerce the target to the dtype the task expects."""
    values = np.asarray(y, dtype=float).reshape(-1)
    if task == "classification":
        return values.astype(int)
    return values
