"""Super Learner: cross-validated convex stacking of a library of algorithms.

TMLE's asymptotic guarantees rest on the nuisance estimators converging fast
enough, which in practice means "do not commit to one parametric model".  The
Super Learner (van der Laan, Polley & Hubbard, 2007) solves this by fitting a
library of candidates, scoring each by cross-validated risk, and combining them
with weights on the simplex.  It is guaranteed to do asymptotically at least as
well as the best single candidate in the library.

Two meta-learners are available, matching R's ``SuperLearner``:

* ``"nnls"`` -- non-negative least squares on the cross-validated predictions,
  normalised to sum to one (``method.NNLS``);
* ``"nnloglik"`` -- non-negative log-likelihood on the simplex
  (``method.NNloglik``), the right choice for a binary target;

plus ``"discrete"``, which puts all weight on the single best candidate (the
"discrete Super Learner", R's ``discreteSL=TRUE``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, clone

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..utils.parallel import map_parallel
from ._fitting import Task, as_target, fit_learner, infer_task, predict_mean
from .crossfit import Folds, make_folds
from .library import _resolve_library

__all__ = ["MetaLearner", "SuperLearner", "SuperLearnerDiagnostics", "resolve_learner"]

MetaLearner = Literal["auto", "nnls", "nnloglik", "discrete"]

# Probabilities are pulled inside this margin before entering a log-likelihood.
_LOGLIK_EPS = 1e-9


@dataclass(frozen=True)
class SuperLearnerDiagnostics:
    """What the ensemble learned, for reporting and validation.

    Attributes
    ----------
    names:
        Candidate names, in the order the weights refer to.
    cv_risk:
        Cross-validated risk per candidate (mean squared error for a regression
        task, negative log-likelihood for a classification task).
    weights:
        Convex ensemble weights.
    best:
        Name of the lowest-risk candidate -- what the discrete Super Learner
        would have selected.
    ensemble_cv_risk:
        Cross-validated risk of the weighted ensemble itself.  Comparing it with
        ``min(cv_risk)`` shows whether stacking bought anything.
    failed:
        Candidates dropped because fitting raised.
    """

    names: tuple[str, ...]
    cv_risk: FloatArray
    weights: FloatArray
    best: str
    ensemble_cv_risk: float
    loss: str
    failed: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "learner": list(self.names),
            "cv_risk": self.cv_risk.tolist(),
            "weight": self.weights.tolist(),
        }


class SuperLearner(BaseEstimator):
    """Cross-validated convex ensemble of a library of learners.

    Parameters
    ----------
    library:
        A sequence of scikit-learn estimators or ``(name, estimator)`` pairs. When
        omitted, the concrete default library is histogram gradient boosting, a
        random forest, and a task-appropriate lasso model.
    task:
        ``"classification"`` for a 0/1 target, ``"regression"`` otherwise;
        ``None`` infers it from ``y`` at fit time.
    meta_learner:
        How to combine candidates -- see the module docstring.  ``"auto"`` picks
        ``"nnloglik"`` for a classification task and ``"nnls"`` otherwise.
    n_folds:
        Inner cross-validation folds used to score candidates.
    clip:
        Optional ``(low, high)`` bounds applied to predictions.  Nuisance models
        for probabilities pass ``(0, 1)``, since a regression candidate can
        otherwise predict outside the unit interval.
    random_state, n_jobs:
        Reproducibility and parallelism.  Candidate fits across folds are
        independent and are dispatched through joblib.

    Attributes
    ----------
    coef_:
        Convex ensemble weights, aligned with ``learner_names_``.
    cv_risk_:
        Cross-validated risk per candidate.
    cv_predictions_:
        ``(n, n_candidates)`` matrix of out-of-fold predictions -- reused by the
        validation module to assess calibration without refitting.
    """

    def __init__(
        self,
        library: Sequence[Any] | None = None,
        *,
        task: Task | None = None,
        meta_learner: MetaLearner = "auto",
        n_folds: int = 10,
        clip: tuple[float, float] | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.library = library
        self.task = task
        self.meta_learner = meta_learner
        self.n_folds = n_folds
        self.clip = clip
        self.random_state = random_state
        self.n_jobs = n_jobs
        if library is not None:
            _resolve_library(library, task or "regression", random_state=random_state)

    # -------------------------------------------------------------------- fit

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        sample_weight: FloatArray | None = None,
        *,
        groups: IntArray | None = None,
    ) -> SuperLearner:
        """Fit the ensemble.

        ``groups`` keeps clustered observations inside the same inner fold, so a
        candidate cannot be scored on a row correlated with its training set.
        """
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        target = np.asarray(y, dtype=float).reshape(-1)
        n = x.shape[0]
        if target.shape[0] != n:
            raise ValueError(f"y has length {target.shape[0]}, expected {n}")

        resolved_task: Task = self.task or infer_task(target)
        weights = (
            np.ones(n, dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )

        levels = np.unique(target)
        if resolved_task == "classification" and levels.size > 2:
            return self._fit_one_vs_rest(x, target, weights, levels, groups)

        library = _resolve_library(
            self.library,
            resolved_task,
            random_state=self.random_state,
        )
        names = [name for name, _ in library]

        folds = make_folds(
            n,
            self.n_folds,
            stratify=target if resolved_task == "classification" else None,
            cluster=groups,
            random_state=self.random_state,
        )

        cv_predictions = self._cross_validate(x, target, weights, library, folds, resolved_task)

        usable: BoolArray = np.asarray(~np.isnan(cv_predictions).any(axis=0), dtype=bool)
        failed = tuple(name for name, ok in zip(names, usable, strict=True) if not ok)
        if failed:
            warnings.warn(
                f"dropping learner(s) {list(failed)} from the ensemble: fitting raised",
                UserWarning,
                stacklevel=2,
            )
        if not usable.any():
            raise RuntimeError("every learner in the library failed to fit")

        kept_names = [name for name, ok in zip(names, usable, strict=True) if ok]
        kept_predictions = cv_predictions[:, usable]
        risks = np.array(
            [
                _risk(kept_predictions[:, k], target, weights, resolved_task)
                for k in range(kept_predictions.shape[1])
            ]
        )
        coefficients = self._combine(kept_predictions, target, weights, risks, resolved_task)

        self.task_ = resolved_task
        self.learner_names_ = tuple(kept_names)
        self.coef_ = coefficients
        self.cv_risk_ = risks
        self.cv_predictions_ = kept_predictions
        self.folds_ = folds
        self.n_features_in_ = x.shape[1]
        self.fitted_learners_ = [
            fit_learner(
                estimator,
                x,
                as_target(target, resolved_task),
                weights,
                warn_unweighted=sample_weight is not None,
            )
            for name, estimator in library
            if name in set(kept_names)
        ]
        ensemble_cv = _risk(kept_predictions @ coefficients, target, weights, resolved_task)
        self.diagnostics_ = SuperLearnerDiagnostics(
            names=tuple(kept_names),
            cv_risk=risks,
            weights=coefficients,
            best=kept_names[int(np.argmin(risks))],
            ensemble_cv_risk=float(ensemble_cv),
            loss="mse" if resolved_task == "regression" else "neg_log_likelihood",
            failed=failed,
        )
        return self

    def _fit_one_vs_rest(
        self,
        x: FloatArray,
        target: FloatArray,
        weights: FloatArray,
        levels: FloatArray,
        groups: IntArray | None,
    ) -> SuperLearner:
        """Fit one binary ensemble per class, for a target with more than two levels.

        The ensemble itself is single-output: it stacks candidates' *conditional means*
        and combines them with one coefficient vector, and both meta-learners
        (non-negative least squares, non-negative log-likelihood) score a scalar
        prediction against a scalar target.  A genuinely multinomial Super Learner would
        need a simplex-valued stack and a multi-class loss -- a different estimator, not a
        wider loop.

        One-vs-rest reuses the whole binary path instead: ``K`` independent ensembles of
        :math:`P(A = a \\mid W)`, each with its own library weights and cross-validated
        risk, normalised at prediction time so the row sums to one.  That normalisation is
        what one-vs-rest costs -- the ``K`` ensembles are not fit under a shared
        constraint, so nothing makes them sum to one before it is applied, and a class the
        ensembles jointly under-predict borrows from the others in proportion rather than
        by any principled rule.  It is reported as a modelling choice for that reason;
        pass a multinomial classifier directly if you would rather have one.

        The binary path is not reached at all when there are two levels, so nothing here
        can perturb it.
        """
        per_class: list[SuperLearner] = []
        for level in levels:
            member = SuperLearner(
                library=self.library,
                meta_learner=self.meta_learner,
                n_folds=self.n_folds,
                task="classification",
                clip=self.clip,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            member.fit(x, (target == level).astype(float), weights, groups=groups)
            per_class.append(member)

        self.task_ = "classification"
        self.classes_by_fit_ = np.asarray(levels, dtype=float)
        self.one_vs_rest_ = per_class
        self.n_features_in_ = x.shape[1]
        # Per-class ensembles have their own library weights and risks, so there is no one
        # ``diagnostics_`` to report. Left unset rather than filled with the first class's,
        # which would read as the ensemble's own. ``diagnostics_by_class_`` has them all.
        self.diagnostics_by_class_ = {
            float(level): member.diagnostics_
            for level, member in zip(levels, per_class, strict=True)
        }
        self.learner_names_ = per_class[0].learner_names_
        return self

    def _cross_validate(
        self,
        x: FloatArray,
        y: FloatArray,
        weights: FloatArray,
        library: list[tuple[str, Learner]],
        folds: Folds,
        task: Task,
    ) -> FloatArray:
        """Out-of-fold predictions, one column per candidate."""
        jobs = [
            (index, train, test, estimator)
            for index, (_, estimator) in enumerate(library)
            for train, test in folds
        ]

        def run(
            index: int, train: IntArray, test: IntArray, estimator: Learner
        ) -> tuple[int, IntArray, FloatArray | None]:
            try:
                model = fit_learner(
                    estimator,
                    x[train],
                    as_target(y[train], task),
                    weights[train],
                    warn_unweighted=False,
                )
                return index, test, self._clip(predict_mean(model, x[test], task))
            except Exception:
                return index, test, None

        predictions = np.full((x.shape[0], len(library)), np.nan)
        for index, test, values in map_parallel(run, jobs, n_jobs=self.n_jobs):
            if values is None:
                predictions[:, index] = np.nan
            else:
                predictions[test, index] = values
        return predictions

    def _combine(
        self,
        z: FloatArray,
        y: FloatArray,
        weights: FloatArray,
        risks: FloatArray,
        task: Task,
    ) -> FloatArray:
        """Solve for the convex ensemble weights."""
        method = self.meta_learner
        if method == "auto":
            method = "nnloglik" if task == "classification" else "nnls"

        if z.shape[1] == 1:
            return np.ones(1)
        if method == "discrete":
            return _one_hot(int(np.argmin(risks)), z.shape[1])
        if method == "nnls":
            return _solve_nnls(z, y, weights, risks)
        if method == "nnloglik":
            return _solve_nnloglik(z, y, weights, risks)
        raise ValueError(
            f"meta_learner must be 'auto', 'nnls', 'nnloglik' or 'discrete'; got {method!r}"
        )

    # ---------------------------------------------------------------- predict

    @property
    def is_multiclass(self) -> bool:
        """Whether this was fit one-vs-rest on a target with more than two levels."""
        return hasattr(self, "one_vs_rest_")

    def predict(self, X: FloatArray) -> FloatArray:
        """Ensemble prediction of the conditional mean."""
        self._check_fitted()
        if self.is_multiclass:
            raise AttributeError(
                "predict is not defined for a multi-class ensemble: there is no single "
                f"conditional mean over classes {self.classes_.tolist()}. Use "
                "predict_proba, whose columns follow classes_."
            )
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        columns = np.column_stack(
            [self._clip(predict_mean(model, x, self.task_)) for model in self.fitted_learners_]
        )
        return self._clip(columns @ self.coef_)

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Class probabilities, columns following :attr:`classes_`.

        ``(n, 2)`` for a binary target, ``(n, K)`` for a one-vs-rest multi-class fit,
        where the per-class ensembles are normalised to sum to one across the row.
        """
        if self.task_ != "classification":
            raise AttributeError("predict_proba is only available for a classification task")
        if self.is_multiclass:
            stacked = np.column_stack(
                [member.predict_proba(X)[:, 1] for member in self.one_vs_rest_]
            )
            # Nothing constrains K independently fit ensembles to sum to one, so the row is
            # normalised here. A row whose ensembles all predicted zero has no information
            # to normalise, and falls back to the uniform distribution rather than dividing
            # by zero -- which would be a silent NaN in a denominator of the estimating
            # equation.
            total = stacked.sum(axis=1, keepdims=True)
            uniform = 1.0 / stacked.shape[1]
            return np.where(total > 0.0, stacked / np.where(total > 0.0, total, 1.0), uniform)
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    @property
    def classes_(self) -> FloatArray:
        """Return fitted outcome classes."""
        self._check_fitted()
        if self.is_multiclass:
            return np.asarray(self.classes_by_fit_, dtype=float)
        return np.array([0.0, 1.0])

    def _clip(self, values: FloatArray) -> FloatArray:
        if self.clip is None:
            return np.asarray(values, dtype=float)
        low, high = self.clip
        return np.clip(np.asarray(values, dtype=float), low, high)

    def _check_fitted(self) -> None:
        # A one-vs-rest fit has no ``coef_`` of its own -- its coefficients live on the
        # per-class members -- so fittedness is either of the two markers.
        if not hasattr(self, "coef_") and not hasattr(self, "one_vs_rest_"):
            from ..exceptions import NotFittedError

            raise NotFittedError("SuperLearner has not been fitted yet; call fit first")

    # ------------------------------------------------------------ reporting

    @property
    def weights(self) -> dict[str, float]:
        """Ensemble weight per candidate, as a mapping."""
        self._check_fitted()
        return dict(zip(self.learner_names_, self.coef_.tolist(), strict=True))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if not hasattr(self, "coef_"):
            return f"SuperLearner(library={self.library!r}, unfitted)"
        parts = ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(self.learner_names_, self.coef_, strict=True)
            if weight > 1e-4
        )
        return f"SuperLearner({parts})"


def _risk(predictions: FloatArray, y: FloatArray, weights: FloatArray, task: Task) -> float:
    """Weighted cross-validated risk under the task's loss."""
    p = np.asarray(predictions, dtype=float)
    if task == "classification":
        q = np.clip(p, _LOGLIK_EPS, 1.0 - _LOGLIK_EPS)
        loss = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    else:
        loss = (y - p) ** 2
    return float(np.average(loss, weights=weights))


def _one_hot(index: int, size: int) -> FloatArray:
    out = np.zeros(size)
    out[index] = 1.0
    return out


def _solve_nnls(z: FloatArray, y: FloatArray, weights: FloatArray, risks: FloatArray) -> FloatArray:
    """Non-negative least squares, normalised onto the simplex."""
    root = np.sqrt(weights)
    try:
        coefficients, _ = optimize.nnls(z * root[:, None], y * root)
    except RuntimeError:  # pragma: no cover - nnls failure is rare
        return _one_hot(int(np.argmin(risks)), z.shape[1])
    total = coefficients.sum()
    if not np.isfinite(total) or total <= 0:
        return _one_hot(int(np.argmin(risks)), z.shape[1])
    return np.asarray(coefficients / total, dtype=float)


def _solve_nnloglik(
    z: FloatArray, y: FloatArray, weights: FloatArray, risks: FloatArray
) -> FloatArray:
    """Minimise the weighted negative log-likelihood over the simplex."""
    n_candidates = z.shape[1]
    bounded = np.clip(z, _LOGLIK_EPS, 1.0 - _LOGLIK_EPS)

    def objective(alpha: FloatArray) -> float:
        p = np.clip(bounded @ alpha, _LOGLIK_EPS, 1.0 - _LOGLIK_EPS)
        return float(-np.sum(weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

    def gradient(alpha: FloatArray) -> FloatArray:
        p = np.clip(bounded @ alpha, _LOGLIK_EPS, 1.0 - _LOGLIK_EPS)
        residual = weights * (y / p - (1.0 - y) / (1.0 - p))
        return np.asarray(-bounded.T @ residual, dtype=float)

    start = _one_hot(int(np.argmin(risks)), n_candidates)
    start = 0.5 * start + 0.5 / n_candidates
    result = optimize.minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_candidates,
        constraints=[{"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0)}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return _solve_nnls(z, y, weights, risks)
    alpha = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    total = alpha.sum()
    if total <= 0:  # pragma: no cover - guarded by the simplex constraint
        return _one_hot(int(np.argmin(risks)), n_candidates)
    return np.asarray(alpha / total, dtype=float)


def resolve_learner(
    spec: Learner | None,
    *,
    task: Task,
    n_folds: int = 5,
    random_state: int | None = None,
    fallback: Learner | None = None,
) -> Learner:
    """Turn a learner specification into an estimator ready to be fitted per fold.

    ``None`` falls back to ``fallback`` and then to a concrete default
    :class:`SuperLearner`. An explicit SuperLearner with no ``random_state`` inherits
    the enclosing estimator's seed on a clone, so repeated cross-fitting redraws its
    inner folds without mutating the caller's template.

    A free function rather than a method so that the C-TMLE candidate search -- and
    anything else that needs the same learner the estimator would have built -- does
    not have to reach into a private method on the estimator to get one.
    """
    if spec is None:
        spec = fallback
    if spec is None:
        return SuperLearner(
            task=task,
            n_folds=n_folds,
            clip=(0.0, 1.0),
            random_state=random_state,
            n_jobs=1,
        )
    if isinstance(spec, SuperLearner) and spec.random_state is None and random_state is not None:
        inherited = clone(spec)
        inherited.set_params(random_state=random_state)
        return inherited
    from .library import _validate_learner

    _validate_learner(spec, "nuisance learner", optional=False)
    return spec
