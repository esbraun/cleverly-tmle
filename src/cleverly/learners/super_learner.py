"""Super Learner: cross-validated convex stacking of a library of algorithms.

TMLE's asymptotic guarantees rest on the nuisance estimators converging fast
enough, which in practice means "do not commit to one parametric model".  The
Super Learner (van der Laan, Polley & Hubbard, 2007) solves this by fitting a
library of candidates, scoring each by cross-validated risk, and combining them
with weights on the simplex.

The oracle result behind it covers the combination that *minimises the
cross-validated risk over the simplex*: under bounded data and a bounded loss,
that combination does asymptotically as well as the best combination of the
library, and so at least as well as the best single candidate.  Polley (2010,
Theorem 1) states it for squared-error loss, which covers ``"cc_ls"`` and
``"discrete"`` on a regression task.  ``"nnloglik"`` and ``"discrete"`` on a
classification task minimise the cross-validated log-likelihood risk instead;
that loss is bounded here only because every candidate prediction is clipped to
``[1e-9, 1 - 1e-9]`` before it enters the log, and the general bounded-loss form
of the result (van der Laan, Polley & Hubbard, 2007) then applies.

The meta-learners, matching R's ``SuperLearner``:

* ``"cc_ls"`` -- least squares on the cross-validated predictions with the
  weights constrained to the simplex (non-negative, summing to one), solved as a
  convex quadratic programme (``method.CC_LS``).  The default for regression;
* ``"nnloglik"`` -- non-negative log-likelihood on the simplex
  (``method.NNloglik``), the default for a binary target;
* ``"nnls"`` -- non-negative least squares *without* the sum-to-one constraint,
  then divided by the sum of the weights (``method.NNLS``).  This is R's heuristic,
  kept for comparison with R.  The division does not give the simplex minimiser, so
  ``"nnls"`` carries no oracle guarantee: it can put all weight on a candidate
  whose risk is far above the best one's, at every sample size;
* ``"discrete"`` -- all weight on the single candidate of smallest cross-validated
  risk (the "discrete Super Learner", R's ``discreteSL=TRUE``).

When a constrained solver fails, ``"cc_ls"`` falls back to ``"discrete"`` and
``"nnloglik"`` falls back to ``"cc_ls"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator

from .._typing import BoolArray, FloatArray, IntArray, Learner
from .._warnings import warn as _warn
from ..exceptions import DataError
from ..utils.parallel import map_parallel
from ._fitting import Task, as_target, fit_learner, infer_task, predict_mean, seed_unset
from .crossfit import Folds, make_folds
from .library import _resolve_library

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary

__all__ = ["MetaLearner", "SuperLearner", "SuperLearnerDiagnostics", "resolve_learner"]

MetaLearner = Literal["auto", "cc_ls", "nnls", "nnloglik", "discrete"]

# Probabilities are pulled inside this margin before entering a log-likelihood.
_LOGLIK_EPS = 1e-9


@dataclass(frozen=True)
class SuperLearnerDiagnostics:
    """What the ensemble learned, for reporting and validation.

    Parameters
    ----------
    names : tuple of str
        Candidate names, in the order the weights refer to.
    cv_risk : ndarray
        Cross-validated risk per candidate (mean squared error for a regression
        task, negative log-likelihood for a classification task).
    weights : ndarray
        Convex ensemble weights.
    best : str
        Name of the lowest-risk candidate -- what the discrete Super Learner
        would have selected.
    ensemble_cv_risk : float
        Cross-validated risk of the weighted ensemble itself.  Comparing it with
        ``min(cv_risk)`` shows whether stacking bought anything.
    loss : str
        Name of the loss the risks above were scored under.
    failed : tuple of str
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
        """Return a JSON-compatible representation.

        Returns
        -------
        dict of str to Any
            Candidate names, cross-validated risks, and weights, as JSON-compatible lists.
        """
        return {
            "learner": list(self.names),
            "cv_risk": self.cv_risk.tolist(),
            "weight": self.weights.tolist(),
        }

    def plot(self, **kwargs: Any) -> Any:
        """Plot each candidate's cross-validated risk and its ensemble weight.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.learners`.

        Returns
        -------
        Figure
            Two panels with the candidates in the same order, lowest risk first. The
            ensemble's own risk is not drawn.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression, Ridge
        >>> from cleverly.learners import SuperLearner
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(120, 2))
        >>> y = x[:, 0] + rng.normal(size=120)
        >>> ensemble = SuperLearner(
        ...     library=[("ols", LinearRegression()), ("ridge", Ridge(alpha=5.0))],
        ...     task="regression",
        ...     n_folds=3,
        ... ).fit(x, y)
        >>> fig = ensemble.diagnostics_.plot()
        >>> fig.axes[1].get_xlabel()
        'Ensemble weight'
        >>> plt.close(fig)
        """
        from ..plotting import learners

        return learners(self, **kwargs)

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the ensemble, like a statsmodels summary.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds a Settings section.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text, with the parts as data.
        """
        from .._summary import check_detail
        from ._ensemble_summary import diagnostics_summary

        return diagnostics_summary(self, check_detail(detail))

    def __repr__(self) -> str:
        from ._ensemble_summary import diagnostics_repr

        return diagnostics_repr(self)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


class SuperLearner(BaseEstimator):
    """Cross-validated convex ensemble of a library of learners.

    Parameters
    ----------
    library : sequence of estimators or None
        A sequence of scikit-learn estimators or ``(name, estimator)`` pairs. When
        omitted, the concrete default library is histogram gradient boosting, a
        random forest, and a task-appropriate lasso model.
    task : {"classification", "regression"} or None
        ``"classification"`` for a 0/1 target, ``"regression"`` otherwise;
        ``None`` infers it from ``y`` at fit time.
    meta_learner : {"auto", "cc_ls", "nnls", "nnloglik", "discrete"}
        How to combine candidates.  See the module docstring.  ``"auto"`` picks
        ``"nnloglik"`` for a classification task and ``"cc_ls"`` otherwise.
        ``"nnls"`` is R's normalised heuristic and has no oracle guarantee.
    n_folds : int
        Inner cross-validation folds used to score candidates.
    clip : tuple of float or None
        Optional ``(low, high)`` bounds applied to predictions.  Nuisance models
        for probabilities pass ``(0, 1)``, since a regression candidate can
        otherwise predict outside the unit interval.
    random_state : int or None
        Seed for the inner fold split, and for every library candidate whose own
        ``random_state`` is unset (at any depth of its ``get_params``).  A candidate's
        explicit seed is kept.  ``None`` seeds neither, and a stochastic candidate with no
        seed of its own then draws from numpy's global generator.
    n_jobs : int
        Number of joblib workers.  Candidate fits across folds are independent
        and are dispatched in parallel.

    Attributes
    ----------
    coef_ : ndarray
        Convex ensemble weights, aligned with ``learner_names_``.
    cv_risk_ : ndarray
        Cross-validated risk per candidate.
    cv_predictions_ : ndarray
        ``(n, n_candidates)`` matrix of out-of-fold predictions.  The validation
        module reuses it to assess calibration without refitting.

    See Also
    --------
    cleverly.ModelSpec : Where an ensemble is passed as a nuisance learner.
    cleverly.learners.default_library : The candidates used when none are given.
    cleverly.learners.SuperLearnerDiagnostics : Per-candidate risk and weight.

    Notes
    -----
    The Super Learner oracle result covers the weights that minimise the cross-validated risk
    over the simplex: with bounded data and a bounded loss, the ensemble then does
    asymptotically as well as the best weighted combination of the library, and so at least as
    well as its best candidate.  Polley (2010, *Super Learner*, U.C. Berkeley dissertation,
    Theorem 1) states it for squared-error loss, which ``"cc_ls"`` and ``"discrete"`` minimise
    on a regression task.  ``"nnloglik"``, and ``"discrete"`` on a classification task,
    minimise the log-likelihood risk; that loss is bounded here only because candidate
    predictions are clipped to ``[1e-9, 1 - 1e-9]`` before the log, and the bounded-loss form
    of the result (van der Laan, Polley and Hubbard 2007) then applies.  ``"nnls"`` divides
    unconstrained non-negative weights by their sum, which is not the simplex minimiser, so it
    has no oracle guarantee and can keep all weight on a much worse candidate at every sample
    size.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> from cleverly import SuperLearner
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = X[:, 0] + rng.normal(size=200)
    >>> ensemble = SuperLearner(
    ...     library=[LinearRegression(), Ridge(alpha=1.0)], n_folds=3, random_state=0
    ... )
    >>> ensemble.fit(X, y).learner_names_
    ('LinearRegression', 'Ridge')

    The fitted weights form a convex combination:

    >>> float(round(ensemble.coef_.sum(), 6))
    1.0
    >>> ensemble.predict(X).shape
    (200,)
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

        Parameters
        ----------
        X : ndarray
            ``(n, p)`` covariates.
        y : ndarray
            ``(n,)`` target.
        sample_weight : ndarray or None
            ``(n,)`` weights, or ``None`` for equal weights.
        groups : ndarray or None
            ``(n,)`` cluster codes. Rows of a cluster stay in the same inner fold, so a
            candidate cannot be scored on a row correlated with its training set.

        Returns
        -------
        SuperLearner
            This object, fitted, as scikit-learn expects.
        """
        self._clear_fitted_state()
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

        library = [
            (name, seed_unset(estimator, self.random_state))
            for name, estimator in _resolve_library(
                self.library,
                resolved_task,
                random_state=self.random_state,
            )
        ]
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
            _warn(
                f"dropping learner(s) {list(failed)} from the ensemble: fitting raised",
                UserWarning,
                stacklevel=2,
            )
        if not usable.any():
            # The data decided it: each learner raised on this sample.  A repeated-sampling
            # study records a ``DataError`` as one failed replicate rather than a stopped run.
            raise DataError("every learner in the library failed to fit")

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

    def _clear_fitted_state(self) -> None:
        """Remove every fitted attribute, so a refit carries nothing from an earlier fit.

        The binary and one-vs-rest branches set different attributes, and
        :attr:`is_multiclass` reads which ones exist.  Fitted attributes follow the
        scikit-learn convention of a trailing underscore; constructor parameters have none.
        """
        for name in [key for key in vars(self) if key.endswith("_") and not key.startswith("_")]:
            delattr(self, name)

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
            method = "nnloglik" if task == "classification" else "cc_ls"

        if method not in ("cc_ls", "nnls", "nnloglik", "discrete"):
            raise ValueError(
                "meta_learner must be 'auto', 'cc_ls', 'nnls', 'nnloglik' or 'discrete'; "
                f"got {method!r}"
            )
        if z.shape[1] == 1:
            return np.ones(1)
        if method == "discrete":
            return _one_hot(int(np.argmin(risks)), z.shape[1])
        if method == "cc_ls":
            return _solve_cc_ls(z, y, weights, risks)
        if method == "nnls":
            return _solve_nnls(z, y, weights, risks)
        return _solve_nnloglik(z, y, weights, risks)

    # ---------------------------------------------------------------- predict

    @property
    def is_multiclass(self) -> bool:
        """Whether this was fit one-vs-rest on a target with more than two levels."""
        return hasattr(self, "one_vs_rest_")

    def predict(self, X: FloatArray) -> FloatArray:
        """Ensemble prediction of the conditional mean.

        Parameters
        ----------
        X : ndarray
            ``(n, p)`` covariates.

        Returns
        -------
        ndarray
            ``(n,)`` ensemble prediction of the conditional mean.
        """
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

        Parameters
        ----------
        X : ndarray
            ``(n, p)`` covariates.

        Returns
        -------
        ndarray
            ``(n, n_classes)`` probabilities, columns following :attr:`classes_`.
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


def _solve_cc_ls(
    z: FloatArray, y: FloatArray, weights: FloatArray, risks: FloatArray
) -> FloatArray:
    """Weighted least squares over the simplex: R's ``method.CC_LS``.

    Minimises the weighted cross-validated squared error of ``z @ alpha`` over
    ``alpha >= 0`` with ``sum(alpha) = 1``, a convex quadratic programme in the Gram
    matrix of the stacked predictions, posed in units of the outcome's root mean square.
    SLSQP finds the active set; the weights are then re-solved exactly on it from the
    equality-constrained normal equations.  A
    solver failure, or a solution whose risk exceeds the best single candidate's,
    falls back to that candidate (the discrete selector), which is a feasible point.
    """
    n_candidates = z.shape[1]
    fallback = _one_hot(int(np.argmin(risks)), n_candidates)
    w = weights / weights.sum()
    # The weights do not depend on the outcome's scale, but SLSQP's stopping rule is absolute,
    # so the problem is solved in units of the outcome's root mean square.
    scale = float(np.sqrt(np.sum(w * y * y)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    zs, ys = z / scale, y / scale
    gram = (zs * w[:, None]).T @ zs
    cross = (zs * w[:, None]).T @ ys

    def objective(alpha: FloatArray) -> float:
        return float(0.5 * alpha @ gram @ alpha - cross @ alpha)

    def gradient(alpha: FloatArray) -> FloatArray:
        return np.asarray(gram @ alpha - cross, dtype=float)

    ones = np.ones(n_candidates)
    try:
        result = optimize.minimize(
            objective,
            np.full(n_candidates, 1.0 / n_candidates),
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_candidates,
            constraints=[
                {"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0), "jac": lambda a: ones}
            ],
            options={"maxiter": 500, "ftol": 1e-15},
        )
    except (ValueError, ArithmeticError):  # pragma: no cover - SLSQP raising is rare
        return fallback
    if not result.success or not np.all(np.isfinite(result.x)):
        return fallback
    alpha = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    if alpha.sum() <= 0:  # pragma: no cover - guarded by the simplex constraint
        return fallback
    alpha = alpha / alpha.sum()
    alpha = _polish_on_active_set(alpha, gram, cross, objective)
    if objective(alpha) > objective(fallback) + 1e-12 * max(1.0, abs(objective(fallback))):
        return fallback
    return alpha


def _polish_on_active_set(
    alpha: FloatArray, gram: FloatArray, cross: FloatArray, objective: Any
) -> FloatArray:
    """Re-solve the simplex least squares exactly on the weights SLSQP left positive.

    On the active set the problem is least squares under one equality constraint,
    whose KKT system is linear.  The exact solution replaces SLSQP's iterate only when
    it stays non-negative and its objective is no larger, so this step can only help.
    """
    active = np.flatnonzero(alpha > 1e-10)
    k = active.size
    system = np.zeros((k + 1, k + 1))
    system[:k, :k] = gram[np.ix_(active, active)]
    system[:k, k] = 1.0
    system[k, :k] = 1.0
    rhs = np.concatenate([cross[active], [1.0]])
    solution = np.linalg.lstsq(system, rhs, rcond=None)[0][:k]
    if not np.all(np.isfinite(solution)) or np.any(solution < 0.0):
        return alpha
    polished = np.zeros_like(alpha)
    polished[active] = solution
    polished = polished / polished.sum()
    return polished if objective(polished) <= objective(alpha) else alpha


def _solve_nnls(z: FloatArray, y: FloatArray, weights: FloatArray, risks: FloatArray) -> FloatArray:
    """Non-negative least squares, then divided by its sum: R's ``method.NNLS`` heuristic.

    The division does not give the least-squares minimiser on the simplex, so this has
    no oracle guarantee (see the module docstring); ``"cc_ls"`` solves that problem.
    """
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
        return _solve_cc_ls(z, y, weights, risks)
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
    :class:`SuperLearner`. When the enclosing estimator passes a seed, every ``random_state``
    of the learner that is ``None`` takes it, on a clone
    (:func:`~cleverly.learners._fitting.seed_unset`), so a seeded fit is a function of its
    data and seed and the caller's template is never mutated. A bare stochastic learner then
    fits with the estimator's seed, and an explicit SuperLearner with no seed of its own
    draws its inner folds from it, so repeated cross-fitting redraws them; the ensemble in
    turn seeds its candidates. An explicit seed is kept, and ``random_state=None`` leaves the
    learner as it is.

    A free function rather than a method so that the C-TMLE candidate search -- and
    anything else that needs the same learner the estimator would have built -- does
    not have to reach into a private method on the estimator to get one.

    The seeding clone goes through :func:`~cleverly.learners._fitting.clone_learner`
    for the same reason every per-fold clone does.  Cloning a SuperLearner clones its whole
    library, so a nested member that refuses would otherwise escape here as a raw
    scikit-learn exception naming neither the library nor the remedy.
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
    from .library import _validate_learner

    _validate_learner(spec, "nuisance learner", optional=False)
    return seed_unset(spec, random_state)
