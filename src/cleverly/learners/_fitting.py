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

from .._typing import FloatArray, Learner
from ._threads import thread_limit

__all__ = ["Task", "fit_learner", "predict_mean", "supports_sample_weight"]

Task = Literal["regression", "classification"]


def _final_estimator(estimator: Learner) -> tuple[Learner, str | None]:
    """The estimator that actually consumes ``sample_weight``, and its step name."""
    if isinstance(estimator, Pipeline):
        name, step = estimator.steps[-1]
        return step, name
    return estimator, None


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


def fit_learner(
    estimator: Learner,
    x: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray | None = None,
    *,
    copy: bool = True,
    warn_unweighted: bool = True,
) -> Learner:
    """Fit ``estimator``, routing weights to wherever they belong.

    Returns a *fitted clone* by default so a learner passed by the user is never
    mutated and can be reused across folds.  A learner that cannot accept weights
    is fitted without them and warns once per call, rather than silently
    discarding a weighting scheme the user asked for.
    """
    model = clone(estimator) if copy else estimator
    if sample_weight is None:
        with thread_limit():
            model.fit(x, y)
        return model

    weights = np.asarray(sample_weight, dtype=float)
    if not supports_sample_weight(model):
        if warn_unweighted:
            warnings.warn(
                f"{type(model).__name__} does not accept sample_weight; fitting it unweighted. "
                "Observation weights will still be applied in the targeting step, but the "
                "nuisance fit ignores them.",
                UserWarning,
                stacklevel=2,
            )
        with thread_limit():
            model.fit(x, y)
        return model

    _, step_name = _final_estimator(model)
    with thread_limit():
        if step_name is None:
            model.fit(x, y, sample_weight=weights)
        else:
            model.fit(x, y, **{f"{step_name}__sample_weight": weights})
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
