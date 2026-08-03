"""Named learner libraries for the :class:`~cleverly.learners.SuperLearner`.

The presets mirror the algorithm sets that R's ``tmle`` and ``SuperLearner``
default to, expressed with scikit-learn (and optionally LightGBM) so the heavy
numerical work happens in compiled code rather than Python:

===============  ===========================================================
R algorithm      cleverly equivalent
===============  ===========================================================
``SL.mean``      :class:`~sklearn.dummy.DummyRegressor` / ``DummyClassifier``
``SL.glm``       unpenalised linear / logistic regression
``SL.glmnet``    :class:`~sklearn.linear_model.ElasticNetCV` / ``LogisticRegressionCV``
``SL.gam``       spline basis expansion + penalised linear / logistic fit
``SL.dbarts``    histogram gradient boosting (LightGBM when installed)
``SL.randomForest``  :class:`~sklearn.ensemble.RandomForestRegressor` / ``Classifier``
===============  ===========================================================

Including ``SL.mean`` is not ceremony: it is what keeps the ensemble from being
dragged around by a learner that overfits a small sample, since the convex
weights can fall back on the marginal mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNetCV,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .._typing import Learner
from ._fitting import Task
from ._threads import refresh_thread_pools

__all__ = ["LIBRARY_PRESETS", "has_lightgbm", "resolve_library"]

LibraryEntry = tuple[str, Learner]

#: Above this many covariates the spline expansion in the GAM-style learner is
#: skipped: the basis grows linearly in the number of columns and stops being a
#: smoother and starts being a memory problem.
_SPLINE_MAX_FEATURES = 25


def has_lightgbm() -> bool:
    """Whether the optional LightGBM extra is installed.

    Importing LightGBM loads an OpenMP runtime, which is a thread pool the fits are
    supposed to be limiting.  :mod:`cleverly.learners._threads` caches its controller
    across calls -- see that module for why -- so the import has to say that the set of
    loaded pools has changed; nothing else in the package loads one lazily.
    """
    global _LIGHTGBM
    if _LIGHTGBM is None:
        try:  # pragma: no cover - depends on installed extras
            import lightgbm  # noqa: F401
        except ImportError:
            _LIGHTGBM = False
        else:
            _LIGHTGBM = True
            refresh_thread_pools()
    return _LIGHTGBM


#: Tri-state: unknown, then True or False once the import has been attempted.  The
#: refresh above must happen on the import and not on every call.
_LIGHTGBM: bool | None = None


def _glm(task: Task) -> Learner:
    if task == "classification":
        # C=1e6 rather than penalty=None: effectively unpenalised, and stable
        # across scikit-learn versions (penalty=None is deprecated from 1.8).
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1e6, max_iter=1000)),
            ]
        )
    return LinearRegression()


def _glmnet(task: Task, random_state: int | None) -> Learner:
    """Regularisation-path learner, with the penalty strength chosen by CV."""
    if task == "classification":
        # The search sits *inside* the pipeline so that observation weights can be
        # routed to the final step as `model__sample_weight`; a search wrapping a
        # pipeline would have no way to forward them.
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GridSearchCV(
                        LogisticRegression(solver="liblinear", max_iter=2000),
                        param_grid={"C": np.logspace(-3, 2, 8)},
                        scoring="neg_log_loss",
                        cv=5,
                        refit=True,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(l1_ratio=[0.5, 1.0], cv=5, max_iter=5000, random_state=random_state),
            ),
        ]
    )


def _gam(task: Task) -> Learner:
    """Additive spline model: a smoother that stays a *linear* model in the basis."""
    basis = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    if task == "classification":
        return Pipeline(
            [
                ("spline", basis),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        )
    return Pipeline([("spline", basis), ("model", Ridge(alpha=1.0))])


def _boost(task: Task, random_state: int | None) -> Learner:
    if has_lightgbm():  # pragma: no cover - depends on installed extras
        import lightgbm as lgb

        common: dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "verbose": -1,
            "random_state": random_state,
        }
        if task == "classification":
            return lgb.LGBMClassifier(**common)
        return lgb.LGBMRegressor(**common)

    fallback: dict[str, Any] = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": random_state,
    }
    if task == "classification":
        return HistGradientBoostingClassifier(**fallback)
    return HistGradientBoostingRegressor(**fallback)


def _forest(task: Task, random_state: int | None) -> Learner:
    common: dict[str, Any] = {
        "n_estimators": 300,
        "min_samples_leaf": 10,
        "n_jobs": 1,
        "random_state": random_state,
    }
    if task == "classification":
        return RandomForestClassifier(**common)
    return RandomForestRegressor(**common)


def _mean(task: Task) -> Learner:
    if task == "classification":
        return DummyClassifier(strategy="prior")
    return DummyRegressor(strategy="mean")


#: Preset names accepted wherever a library is expected.
LIBRARY_PRESETS: tuple[str, ...] = ("glm", "fast", "default", "rich")


def resolve_library(
    spec: str | Sequence[LibraryEntry] | Sequence[Learner] | None,
    task: Task,
    *,
    n_features: int | None = None,
    random_state: int | None = None,
) -> list[LibraryEntry]:
    """Turn a library specification into ``(name, estimator)`` pairs.

    ``spec`` may be a preset name, a sequence of estimators, or a sequence of
    ``(name, estimator)`` pairs.  ``n_features`` lets the presets drop the spline
    learner when the covariate count would make the basis expansion unreasonable.
    """
    if spec is None:
        spec = "default"

    if isinstance(spec, str):
        return _preset(spec, task, n_features=n_features, random_state=random_state)

    entries: list[LibraryEntry] = []
    seen: set[str] = set()
    for item in spec:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            name, estimator = item
        else:
            estimator = item
            name = type(estimator).__name__
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


def _preset(
    name: str,
    task: Task,
    *,
    n_features: int | None,
    random_state: int | None,
) -> list[LibraryEntry]:
    if name not in LIBRARY_PRESETS:
        raise ValueError(f"unknown library preset {name!r}; choose from {LIBRARY_PRESETS}")

    entries: list[LibraryEntry] = [("mean", _mean(task)), ("glm", _glm(task))]
    if name == "glm":
        return entries

    splines_ok = n_features is None or n_features <= _SPLINE_MAX_FEATURES
    if name == "fast":
        if splines_ok:
            entries.append(("gam", _gam(task)))
        entries.append(("boost", _boost(task, random_state)))
        return entries

    entries.append(("glmnet", _glmnet(task, random_state)))
    if splines_ok:
        entries.append(("gam", _gam(task)))
    entries.append(("boost", _boost(task, random_state)))
    if name == "rich":
        entries.append(("forest", _forest(task, random_state)))
    return entries
