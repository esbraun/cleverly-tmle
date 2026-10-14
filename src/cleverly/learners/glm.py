r"""A parametric generalized linear model whose score the package can stack.

A Super Learner or a gradient-boosted model has no estimating equation the package can
differentiate, so an estimator that fits one reports no stacked interval. :class:`GLM` is the
declared parametric model: an unpenalized maximum-likelihood fit with a canonical link, whose
design, coefficients, score and information are all public. The sequential g-computation and
IPW estimators of :mod:`cleverly.longitudinal` read them to build the stacked
estimating-equation curve of a fit.

The families follow R's ``glm``:

- ``binomial`` takes a 0/1 label or a fraction in [0, 1], with
  :math:`\mu = \operatorname{expit}(x^\top\beta)`. A fractional target is the quasi-binomial
  fit of R's ``glm(family = quasibinomial())``, whose score :math:`x(y - \mu)` is the binomial
  one.
- ``multinomial`` takes three or more integer labels, with baseline-category logits and the first
  label as the reference.
- ``gaussian`` takes any real target, with :math:`\mu = x^\top\beta` fitted by least squares.

``family="auto"`` reads the target: integer labels give ``binomial`` (two labels) or
``multinomial`` (more), a float target in [0, 1] gives ``binomial``, and any other float target
gives ``gaussian``.

The design is an intercept, then the columns of ``x``, then (with ``interactions`` above one)
every product of two to ``interactions`` distinct columns, in
:func:`itertools.combinations` order. The design reads nothing from the data, so it adds no
estimated step. A column that is a linear combination of earlier columns on the weighted fit rows
is aliased, as R's ``glm`` aliases it: its coefficient is fixed at zero and it leaves the score.
On the followers of a static regimen, for example, an earlier treatment column is constant and
is aliased with the intercept.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator

from .._typing import BoolArray, FloatArray
from ..exceptions import DataError

__all__ = ["GLM", "GLMFamily"]

#: The families a :class:`GLM` fits.
GLMFamily = Literal["auto", "binomial", "multinomial", "gaussian"]

#: Relative tolerance under which a design column counts as aliased with earlier columns. It is
#: the tolerance of R's ``glm.fit``, ``min(1e-07, control$epsilon / 1000)`` with the default
#: ``epsilon = 1e-8``.
_ALIAS_TOLERANCE = 1e-11

#: A linear predictor beyond this magnitude puts a fitted probability within about 1e-13 of 0
#: or 1, where the logistic MLE is at or near separation.
_SEPARATION_ETA = 30.0


def _expit(eta: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-eta)), dtype=float)


def _softmax(eta: FloatArray) -> FloatArray:
    """Row softmax of ``(n, K)`` logits, stabilized by the row maximum."""
    shifted = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(shifted)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


class GLM(BaseEstimator):
    """An unpenalized generalized linear model with a canonical link.

    Parameters
    ----------
    family : {"auto", "binomial", "multinomial", "gaussian"}
        The model family. ``"auto"`` chooses from the target, as the module docstring states.
    interactions : int
        The highest order of the products of distinct columns in the design. ``1`` is main
        terms, as R's default ``glm`` formula ``y ~ .``.
    max_iter : int
        The most Newton steps of a binomial or multinomial fit.
    tol : float
        The convergence tolerance on the largest Newton step, relative to one plus the
        largest coefficient.

    Attributes
    ----------
    family_ : str
        The fitted family.
    classes_ : numpy.ndarray
        The labels of a ``binomial`` fit on labels or of a ``multinomial`` fit, in sorted order.
        Absent on a fit of a float target.
    active_ : numpy.ndarray
        The boolean mask of design columns that are not aliased.
    coef_ : numpy.ndarray
        The coefficients of the active columns. ``(p,)`` for ``binomial`` and ``gaussian``, and
        ``(K - 1, p)`` for ``multinomial``, one row per non-reference label.
    n_iter_ : int
        The Newton steps the fit took. Zero for ``gaussian``.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.learners import GLM
    >>> x = np.array([[0.0], [0.0], [1.0], [1.0]])
    >>> model = GLM().fit(x, np.array([0.2, 0.4, 0.6, 0.8]))
    >>> model.family_
    'binomial'
    >>> np.round(model.predict(np.array([[0.0], [1.0]])), 6)
    array([0.3, 0.7])
    """

    def __init__(
        self,
        family: GLMFamily = "auto",
        interactions: int = 1,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        self.family = family
        self.interactions = interactions
        self.max_iter = max_iter
        self.tol = tol

    # -- design -------------------------------------------------------------------------

    def full_design(self, x: FloatArray) -> FloatArray:
        """The design before aliasing: an intercept, the columns, and their products.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The ``(n, P)`` design.
        """
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        columns = [np.ones(matrix.shape[0]), *matrix.T]
        for order in range(2, int(self.interactions) + 1):
            for chosen in combinations(range(matrix.shape[1]), order):
                columns.append(np.prod(matrix[:, list(chosen)], axis=1))
        return np.column_stack(columns)

    def design(self, x: FloatArray) -> FloatArray:
        """The design of the active columns, the one the score and the information read.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The ``(n, p)`` active design.
        """
        return self.full_design(x)[:, self.active_]

    # -- fit ----------------------------------------------------------------------------

    def fit(self, x: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None) -> GLM:
        """Fit the model by maximum likelihood.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.
        y : numpy.ndarray
            The target.
        sample_weight : numpy.ndarray or None
            Nonnegative observation weights. ``None`` weights every row one.

        Returns
        -------
        GLM
            The fitted model.

        Raises
        ------
        DataError
            When the fit has no row with positive weight, when a binomial target leaves
            [0, 1], or when the Newton iteration does not converge or reaches separation.
        """
        if int(self.interactions) < 1:
            raise ValueError(f"interactions must be at least 1; got {self.interactions}")
        target = np.asarray(y)
        weights = (
            np.ones(target.shape[0])
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if not np.any(weights > 0):
            raise DataError("GLM: no row has positive weight, so the model has no fit rows")
        family = self._resolve_family(target)
        self.family_ = family
        full = self.full_design(x)
        self.active_ = _active_columns(full, weights)
        design = full[:, self.active_]
        if family == "gaussian":
            values = target.astype(float)
            root = np.sqrt(weights)
            coef, *_ = np.linalg.lstsq(design * root[:, None], values * root, rcond=None)
            self.coef_ = np.asarray(coef, dtype=float)
            self.n_iter_ = 0
            return self
        if family == "binomial":
            values = self._binomial_target(target)
            self.coef_, self.n_iter_ = self._newton_binomial(design, values, weights)
            return self
        labels = np.unique(target)
        self.classes_ = labels
        indicator = (target.reshape(-1, 1) == labels.reshape(1, -1)).astype(float)
        self.coef_, self.n_iter_ = self._newton_multinomial(design, indicator, weights)
        return self

    def _resolve_family(self, target: np.ndarray) -> str:
        if self.family != "auto":
            if self.family not in ("binomial", "multinomial", "gaussian"):
                raise ValueError(
                    "family must be 'auto', 'binomial', 'multinomial' or 'gaussian'; "
                    f"got {self.family!r}"
                )
            return str(self.family)
        if np.issubdtype(target.dtype, np.integer) or target.dtype == bool:
            return "binomial" if np.unique(target).size <= 2 else "multinomial"
        values = target.astype(float)
        if np.all((values >= 0.0) & (values <= 1.0)):
            return "binomial"
        return "gaussian"

    def _binomial_target(self, target: np.ndarray) -> FloatArray:
        if np.issubdtype(target.dtype, np.integer) or target.dtype == bool:
            labels = np.unique(target)
            self.classes_ = labels
            if labels.size == 1:
                raise DataError(
                    f"GLM: every fit row holds the label {labels[0]!r}, so a binomial model "
                    "has no finite maximum-likelihood estimate"
                )
            return np.asarray(target == labels[-1], dtype=float)
        values = target.astype(float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise DataError("GLM: a binomial target must lie in [0, 1]")
        return np.asarray(values, dtype=float)

    def _newton_binomial(
        self, design: FloatArray, y: FloatArray, weights: FloatArray
    ) -> tuple[FloatArray, int]:
        def score_and_information(coef: FloatArray) -> tuple[FloatArray, FloatArray]:
            mu = _expit(design @ coef)
            score = design.T @ (weights * (y - mu))
            return score, (design * (weights * mu * (1.0 - mu))[:, None]).T @ design

        coef, steps = self._newton(np.zeros(design.shape[1]), score_and_information, "binomial")
        self._check_separation(design @ coef, weights)
        return coef, steps

    def _newton_multinomial(
        self, design: FloatArray, indicator: FloatArray, weights: FloatArray
    ) -> tuple[FloatArray, int]:
        shape = (indicator.shape[1] - 1, design.shape[1])

        def score_and_information(flat: FloatArray) -> tuple[FloatArray, FloatArray]:
            probabilities = self._multinomial_probabilities(design, flat.reshape(shape))
            score = self._multinomial_score(design, indicator, probabilities, weights)
            return score, _multinomial_information(design, probabilities, weights)

        flat, steps = self._newton(
            np.zeros(shape).reshape(-1), score_and_information, "multinomial"
        )
        coef = flat.reshape(shape)
        self._check_separation(self._multinomial_logits(design, coef), weights)
        return coef, steps

    def _newton(
        self,
        start: FloatArray,
        score_and_information: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
        family: str,
    ) -> tuple[FloatArray, int]:
        """Newton's method on a concave log-likelihood, stopped by the size of its step.

        A step-size rule, not a score rule, is what detects separation: there the score
        decays towards zero while the coefficients keep growing by a step of order one, so
        the iteration runs out of steps and raises. A finite maximum makes the steps shrink
        quadratically, so the last step leaves the score at rounding error.
        """
        coef = start
        for step in range(1, int(self.max_iter) + 1):
            score, information = score_and_information(coef)
            move = _solve(information, score)
            coef = coef + move
            if np.max(np.abs(move)) <= self.tol * (1.0 + np.max(np.abs(coef))):
                return coef, step
        raise DataError(
            f"GLM: the {family} fit did not converge in {self.max_iter} Newton steps. The "
            "usual cause is separation: a design cell whose rows all hold one label, where no "
            "maximum-likelihood estimate exists. Pool sparse covariate levels or lower "
            "interactions="
        )

    @staticmethod
    def _multinomial_logits(design: FloatArray, coef: FloatArray) -> FloatArray:
        return np.column_stack([np.zeros(design.shape[0]), design @ coef.T])

    def _multinomial_probabilities(self, design: FloatArray, coef: FloatArray) -> FloatArray:
        return _softmax(self._multinomial_logits(design, coef))

    @staticmethod
    def _multinomial_score(
        design: FloatArray, indicator: FloatArray, probabilities: FloatArray, weights: FloatArray
    ) -> FloatArray:
        residual = (indicator - probabilities)[:, 1:] * weights[:, None]
        return np.asarray((residual.T @ design).reshape(-1), dtype=float)

    @staticmethod
    def _check_separation(eta: FloatArray, weights: FloatArray) -> None:
        if np.any(np.abs(eta[weights > 0]) > _SEPARATION_ETA):
            raise DataError(
                "GLM: a fitted probability is within 1e-13 of 0 or 1, so the logistic fit is "
                "at separation and its maximum-likelihood estimate is not finite. Pool sparse "
                "covariate levels or lower interactions="
            )

    # -- prediction ---------------------------------------------------------------------

    def linear_predictor(self, x: FloatArray) -> FloatArray:
        """The linear predictor: ``(n,)``, or ``(n, K)`` logits for ``multinomial``.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The linear predictor.
        """
        design = self.design(x)
        if self.family_ == "multinomial":
            return self._multinomial_logits(design, self.coef_)
        return np.asarray(design @ self.coef_, dtype=float)

    def predict(self, x: FloatArray) -> FloatArray:
        """The fitted mean: the probability of the last label for ``binomial``.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The ``(n,)`` mean. A ``multinomial`` fit returns the most probable label.
        """
        eta = self.linear_predictor(x)
        if self.family_ == "gaussian":
            return eta
        if self.family_ == "binomial":
            return _expit(eta)
        return np.asarray(self.classes_[np.argmax(eta, axis=1)])

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """The label probabilities, one column per label of :attr:`classes_`.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The ``(n, K)`` probabilities.
        """
        eta = self.linear_predictor(x)
        if self.family_ == "multinomial":
            return _softmax(eta)
        if self.family_ != "binomial":
            raise AttributeError("a gaussian GLM has no predict_proba")
        if getattr(self, "classes_", np.array([0, 1])).size == 1:
            return np.ones((eta.shape[0], 1))
        positive = _expit(eta)
        return np.column_stack([1.0 - positive, positive])

    # -- estimating equation ------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """The count of free coefficients."""
        return int(np.asarray(self.coef_).size)

    def mean_jacobian(self, x: FloatArray) -> FloatArray:
        r"""The derivative of the fitted mean in the coefficients.

        For ``binomial`` and ``gaussian`` it is :math:`\mu'(\eta)\, x`, an ``(n, p)`` array.
        For ``multinomial`` it is the ``(n, K, (K - 1) p)`` derivative of each label's
        probability, in the order of ``coef_.reshape(-1)``.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.

        Returns
        -------
        numpy.ndarray
            The Jacobian.
        """
        design = self.design(x)
        if self.family_ == "gaussian":
            return design
        if self.family_ == "binomial":
            mu = _expit(design @ self.coef_)
            return design * (mu * (1.0 - mu))[:, None]
        probabilities = _softmax(self._multinomial_logits(design, self.coef_))
        n, n_classes = probabilities.shape
        p = design.shape[1]
        out = np.zeros((n, n_classes, (n_classes - 1) * p))
        for label in range(n_classes):
            for other in range(1, n_classes):
                delta = 1.0 if label == other else 0.0
                factor = probabilities[:, label] * (delta - probabilities[:, other])
                out[:, label, (other - 1) * p : other * p] = factor[:, None] * design
        return out

    def score_contributions(
        self, x: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None
    ) -> FloatArray:
        """Each row's weighted score at the fitted coefficients, ``(n, n_parameters)``.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates.
        y : numpy.ndarray
            The target, coded as at the fit.
        sample_weight : numpy.ndarray or None
            The weights; a row the fit did not read takes weight zero.

        Returns
        -------
        numpy.ndarray
            The score rows. Their sum is zero on the fit rows, to the fit's tolerance.
        """
        design = self.design(x)
        target = np.asarray(y)
        weights = (
            np.ones(design.shape[0])
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if self.family_ == "multinomial":
            indicator = (target.reshape(-1, 1) == self.classes_.reshape(1, -1)).astype(float)
            probabilities = _softmax(self._multinomial_logits(design, self.coef_))
            residual = (indicator - probabilities)[:, 1:] * weights[:, None]
            return np.asarray(
                (residual[:, :, None] * design[:, None, :]).reshape(design.shape[0], -1),
                dtype=float,
            )
        if self.family_ == "binomial":
            values = (
                np.asarray(target == self.classes_[-1], dtype=float)
                if hasattr(self, "classes_") and self.classes_.size == 2
                else target.astype(float)
            )
            mu = _expit(design @ self.coef_)
        else:
            values = target.astype(float)
            mu = design @ self.coef_
        return np.asarray(design * (weights * (values - mu))[:, None], dtype=float)

    def information(self, x: FloatArray, sample_weight: FloatArray | None = None) -> FloatArray:
        """The negative derivative of the summed score, ``(n_parameters, n_parameters)``.

        Parameters
        ----------
        x : numpy.ndarray
            The ``(n, d)`` covariates of the fit rows.
        sample_weight : numpy.ndarray or None
            The weights; zero on rows the fit did not read.

        Returns
        -------
        numpy.ndarray
            The information matrix (not divided by n).
        """
        design = self.design(x)
        weights = (
            np.ones(design.shape[0])
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if self.family_ == "gaussian":
            return np.asarray((design * weights[:, None]).T @ design, dtype=float)
        if self.family_ == "binomial":
            mu = _expit(design @ self.coef_)
            return np.asarray((design * (weights * mu * (1.0 - mu))[:, None]).T @ design)
        probabilities = _softmax(self._multinomial_logits(design, self.coef_))
        return _multinomial_information(design, probabilities, weights)


def _multinomial_information(
    design: FloatArray, probabilities: FloatArray, weights: FloatArray
) -> FloatArray:
    """The baseline-category information, blocks ordered as ``coef_.reshape(-1)``."""
    n_classes = probabilities.shape[1]
    p = design.shape[1]
    info = np.zeros(((n_classes - 1) * p, (n_classes - 1) * p))
    for row in range(1, n_classes):
        for col in range(1, n_classes):
            delta = 1.0 if row == col else 0.0
            factor = weights * probabilities[:, row] * (delta - probabilities[:, col])
            block = (design * factor[:, None]).T @ design
            info[(row - 1) * p : row * p, (col - 1) * p : col * p] = block
    return info


def _solve(info: FloatArray, score: FloatArray) -> FloatArray:
    try:
        return np.asarray(np.linalg.solve(info, score), dtype=float)
    except np.linalg.LinAlgError as error:
        raise DataError(
            "GLM: the information matrix is singular, so the Newton step is undefined. The "
            "usual cause is separation or a design cell with no row"
        ) from error


def _active_columns(design: FloatArray, weights: FloatArray) -> BoolArray:
    """Keep each column unless it is a combination of the kept earlier columns.

    The weighted Gram-Schmidt order of R's pivoting QR with ``glm.fit``'s tolerance: a column whose
    residual norm, after projection on the kept columns over the rows with positive weight, is
    below the tolerance times its own norm is aliased.
    """
    rows = weights > 0
    root = np.sqrt(weights[rows])
    matrix = design[rows] * root[:, None]
    keep = np.zeros(design.shape[1], dtype=bool)
    basis: list[FloatArray] = []
    for column in range(design.shape[1]):
        vector = matrix[:, column].copy()
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        for unit in basis:
            vector -= (unit @ vector) * unit
        residual = float(np.linalg.norm(vector))
        if residual > _ALIAS_TOLERANCE * norm:
            basis.append(vector / residual)
            keep[column] = True
    return keep
