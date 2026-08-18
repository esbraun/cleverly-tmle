"""Concrete estimator libraries for :class:`~cleverly.learners.SuperLearner`.

Nuisance-model APIs accept estimator objects, not names. The only implicit
library is used when ``SuperLearner.library`` is omitted, and is materialized
here as three ordinary scikit-learn estimators for the target task.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .._typing import Learner
from ._fitting import Task

__all__ = ["default_library"]

LibraryEntry = tuple[str, Learner]


def default_library(task: Task, *, random_state: int | None = None) -> list[LibraryEntry]:
    """Return the concrete default ensemble for one prediction task."""
    if task not in {"classification", "regression"}:
        raise ValueError(f"task must be 'classification' or 'regression'; got {task!r}")
    boost_common: dict[str, Any] = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": random_state,
    }
    forest_common: dict[str, Any] = {
        "n_estimators": 300,
        "min_samples_leaf": 10,
        "n_jobs": 1,
        "random_state": random_state,
    }
    if task == "classification":
        lasso: Learner = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=np.logspace(-3, 2, 8),
                        cv=5,
                        penalty="l1",
                        solver="saga",
                        scoring="neg_log_loss",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return [
            ("hist_gradient_boosting", HistGradientBoostingClassifier(**boost_common)),
            ("random_forest", RandomForestClassifier(**forest_common)),
            ("lasso", lasso),
        ]
    return [
        ("hist_gradient_boosting", HistGradientBoostingRegressor(**boost_common)),
        ("random_forest", RandomForestRegressor(**forest_common)),
        (
            "lasso",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LassoCV(cv=5, max_iter=5000, random_state=random_state)),
                ]
            ),
        ),
    ]


def _validate_learner(value: Any, name: str, *, optional: bool = True) -> None:
    """Refuse non-estimator nuisance declarations at configuration time."""
    if value is None and optional:
        return
    if isinstance(value, str):
        raise TypeError(
            f"{name} must be an sklearn-compatible estimator object, not {value!r}; "
            "pass a model such as LinearRegression(), LogisticRegression(), or SuperLearner()"
        )
    if (
        value is None
        or not hasattr(value, "fit")
        or not (hasattr(value, "predict") or hasattr(value, "predict_proba"))
    ):
        raise TypeError(
            f"{name} must implement fit and predict or predict_proba; got {type(value).__name__}"
        )


def _resolve_library(
    library: Sequence[LibraryEntry] | Sequence[Learner] | None,
    task: Task,
    *,
    random_state: int | None = None,
) -> list[LibraryEntry]:
    """Normalize estimator objects into uniquely named library entries."""
    if isinstance(library, str):
        raise TypeError(
            "SuperLearner.library must contain sklearn-compatible estimator objects, "
            f"not preset name {library!r}"
        )
    if library is None:
        return default_library(task, random_state=random_state)

    entries: list[LibraryEntry] = []
    seen: set[str] = set()
    for item in library:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            name, estimator = item
        else:
            estimator = item
            name = type(estimator).__name__
        _validate_learner(estimator, f"SuperLearner.library[{name!r}]", optional=False)
        candidate = name
        suffix = 2
        while candidate in seen:
            candidate = f"{name}_{suffix}"
            suffix += 1
        seen.add(candidate)
        entries.append((candidate, estimator))
    if not entries:
        raise ValueError("learner library is empty")
    return entries
