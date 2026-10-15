r"""The mechanism term of the plug-in :math:`\nu^2` curve, for a declared parametric mechanism.

``nu2_estimator="plugin"`` estimates :math:`\nu^2` by :math:`E_n[\hat\alpha^2]`, where
:math:`\hat\alpha` is built from the fitted treatment mechanism :math:`\hat g`.  The estimate
moves at first order with :math:`\hat g`, so its curve needs a term for that fit.  When the
fit's own treatment learner is a scikit-learn ``LogisticRegression`` (binary or multinomial,
with no penalty or a fixed L2 penalty, fitted bare or after ``StandardScaler`` steps), the
mechanism is an M-estimator, and the stacked estimating equation of its score and the second
moment gives the term (Saul and Hudgens 2020, *Journal of Statistical Software* 92(2),
Section 2, Equations (1) and (2); Section 3.5 for a propensity-model nuisance).

For fold :math:`k` with coefficients :math:`\theta_k` in reference form (the lowest arm is the
reference), the estimating function of row :math:`i` is

.. math::

    u_{k,i} = w_i\, x_i \otimes (e_{A_i} - g_k(x_i))_{-r}\, 1\{i \in \mathrm{train}_k\}
              - (w_i / n) P \theta_k / C,

where the second term allocates the fixed L2 penalty under the mean-one weight convention.
The information matrix is :math:`J_k = H_k + P/C`, with
:math:`H_k = \sum_{j \in \mathrm{train}_k} w_j [\mathrm{diag}(g_{-r}) - g_{-r} g_{-r}^\top]
\otimes x_j x_j^\top`.  With :math:`D_k = n^{-1} \sum_{j \in \mathrm{test}_k} w_j
\partial_\theta \alpha_j^2`, the mechanism term is
:math:`\psi^\gamma_i = n \sum_k D_k J_k^{+} u_{k,i}`.  An in-sample fit is one fold whose
training and test rows are all rows.

The derivative of :math:`\alpha` is zero on every cell the truncation bounds moved, so the term
is the curve of the truncated plug-in functional :math:`\gamma \mapsto E[\alpha_b(g_\gamma)^2]`.
That functional equals :math:`\nu_0^2` only when the truncation is asymptotically inactive.

A declared known mechanism (:class:`~cleverly.data.KnownMechanism`) is the degenerate case:
:math:`\hat\alpha = \alpha(g_0)`, so the term is zero.

:func:`plugin_mechanism_term` returns the term, or the reason the fit does not qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray
from ..estimators._mechanism_fit import MechanismFit

__all__ = [
    "CONDITION_LIMIT",
    "MISMATCH_TOLERANCE",
    "PENALTY_BIAS_FRACTION",
    "ROOT_TOLERANCE",
    "MechanismRefusal",
    "plugin_mechanism_term",
]

#: The largest Newton step from the captured coefficients to the root of the fold's
#: penalized score, in standard errors of the coefficients, that still counts as a root.
ROOT_TOLERANCE = 0.05

#: The largest condition number of a fold's information matrix, on the range of its design.
CONDITION_LIMIT = 1e10

#: The largest absolute difference between the recomputed and the stored mechanism.
MISMATCH_TOLERANCE = 1e-8

#: The largest first-order shrinkage of ``nu^2`` by a fixed L2 penalty, as a fraction of the
#: standard error of ``nu^2``.
PENALTY_BIAS_FRACTION = 0.1

#: Relative eigenvalue cutoff below which a direction counts as outside the design's range.
_RANK_CUTOFF = 1e-12

_STEM = (
    "the plug-in nu^2 has a standard error only through the score of the fit's own treatment "
    "mechanism. E_n[alpha_hat^2] squares the fitted Riesz representer, so it moves at first "
    "order with the fitted mechanism. The limits need that mechanism to be declared known, or "
    "to be a scikit-learn LogisticRegression, binary or multinomial, with no penalty or a "
    "fixed L2 penalty, fitted bare or after StandardScaler. For that model the stacked "
    "estimating equation of the mechanism score and the second moment gives the curve (Saul "
    "and Hudgens 2020). No source this package cites gives the curve for any other mechanism "
    "learner. The bias-adjusted bounds, rv, max_bias, benchmark() and contour() remain "
    "available. docs/roadmap.md F26 tracks this stop"
)

_DOUBLY_ROBUST = (
    "Fit the bound with nu2_estimator='doubly_robust', which has derived limits for any "
    "mechanism learner."
)


@dataclass(frozen=True)
class MechanismRefusal:
    """Why a plug-in bound publishes no limit.

    Parameters
    ----------
    key : str
        The clause key, for example ``"other_learner"``.
    text : str
        The full reason, ``"nu2_estimator='plugin': <stem>. This fit does not qualify:
        <clause>."``.
    alternative : str
        What the reader can do instead.
    """

    key: str
    text: str
    alternative: str


def _refusal(key: str, clause: str, alternative: str = _DOUBLY_ROBUST) -> MechanismRefusal:
    return MechanismRefusal(
        key=key,
        text=f"nu2_estimator='plugin': {_STEM}. This fit does not qualify: {clause}.",
        alternative=alternative,
    )


def _resolved_penalty(fit: MechanismFit) -> str:
    """``"none"``, ``"l2"`` or ``"other"``, as scikit-learn's ``fit`` resolves them."""
    if np.isinf(fit.C):
        return "none"
    if fit.penalty is None or fit.penalty == "none":
        return "none"
    if fit.penalty == "l2":
        return "l2"
    if fit.penalty == "deprecated" and (fit.l1_ratio is None or fit.l1_ratio == 0.0):
        return "l2"
    return "other"


def _learner_refusal(fit: MechanismFit, n_arms: int) -> MechanismRefusal | None:
    """Checks 1 to 7: the fit-level rule, read off one fold's record."""
    if "CorrelationScreener" in fit.step_names:
        return _refusal(
            "screened",
            "screen_treatment=True selects the covariates of the treatment model inside each "
            "fold, so the mechanism is not one fixed parametric model",
        )
    if fit.kind == "logistic" and fit.is_cv:
        return _refusal(
            "cv_penalty",
            "the treatment learner is LogisticRegressionCV, which selects C by "
            "cross-validation, so the penalty is data-adaptive. Pass LogisticRegression with "
            "a fixed C",
        )
    if fit.kind == "other" and fit.learner_name == "SuperLearner":
        return _refusal(
            "super_learner",
            "the treatment learner is a SuperLearner. A library that contains "
            "LogisticRegression does not qualify, because the bound reads the ensemble's "
            "fitted mechanism and not one member's",
        )
    if fit.kind == "other":
        return _refusal(
            "other_learner", f"the treatment learner is {fit.learner_name}, not LogisticRegression"
        )
    if fit.kind != "logistic":
        return None
    if fit.steps is None:
        offending = next(
            name for name in fit.step_names if name not in ("StandardScaler", "passthrough")
        )
        return _refusal(
            "pipeline",
            f"the treatment learner is a Pipeline with the step {offending} before the "
            "LogisticRegression. Only StandardScaler steps qualify. Transform the covariates "
            "before the fit, or pass the bare LogisticRegression",
        )
    if not fit.fit_intercept and any(step.mean is not None for step in fit.steps):
        return _refusal(
            "pipeline_intercept",
            "a StandardScaler centers the covariates and the LogisticRegression sets "
            "fit_intercept=False, so the mechanism depends on the estimated column means. Set "
            "fit_intercept=True",
        )
    if _resolved_penalty(fit) == "other":
        return _refusal(
            "l1_penalty",
            "the LogisticRegression has an L1 or elastic-net penalty, whose score has no "
            "derivative where a coefficient is zero. Use l1_ratio=0 for an L2 penalty, or "
            "C=numpy.inf",
        )
    if fit.solver == "liblinear":
        return _refusal(
            "liblinear",
            "the LogisticRegression uses solver='liblinear', which penalizes the intercept. "
            "Use another solver",
        )
    if fit.class_weight is not None:
        return _refusal(
            "class_weight",
            f"the LogisticRegression sets class_weight={fit.class_weight}, which changes its "
            "score. Declare observation weights with weights= instead",
        )
    if n_arms > 2 and fit.multi_class == "ovr":
        return _refusal(
            "one_vs_rest",
            "the LogisticRegression fits one-vs-rest, which is not the multinomial model",
        )
    return None


@dataclass(frozen=True)
class LogisticStack:
    """One fold's logistic mechanism in reference form.

    Parameters
    ----------
    design : ndarray
        ``(n, d)`` design of every row: the pipeline transform of the covariates, with the
        intercept column first when the model fits one.
    theta : ndarray
        ``(K - 1, d)`` coefficients in reference form.
    penalty : ndarray
        ``(p, p)`` penalty matrix ``P / C`` with ``p = (K - 1) d``; zero with no penalty.
    probabilities : ndarray
        ``(n, K)`` mechanism the coefficients give at every row.
    """

    design: FloatArray
    theta: FloatArray
    penalty: FloatArray
    probabilities: FloatArray

    @property
    def n_classes(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.theta.size)

    def score(self, treatment_index: Any, rows: Any, weights: FloatArray) -> FloatArray:
        """``(len(rows), p)`` weighted unpenalized score rows ``w x (e_A - g)_{-r}``."""
        g = self.probabilities[rows]
        indicator = np.zeros_like(g)
        indicator[np.arange(g.shape[0]), np.asarray(treatment_index)[rows]] = 1.0
        residual = (indicator - g)[:, 1:]
        x = self.design[rows]
        rows_score = residual[:, :, None] * x[:, None, :]
        return np.asarray(weights[rows][:, None] * rows_score.reshape(g.shape[0], -1), dtype=float)

    def information(self, rows: Any, weights: FloatArray) -> FloatArray:
        """``H + P / C`` over ``rows``."""
        g = self.probabilities[rows][:, 1:]
        x = self.design[rows]
        w = weights[rows]
        k = g.shape[1]
        d = x.shape[1]
        out = np.zeros((k, d, k, d))
        for m in range(k):
            for q in range(m, k):
                factor = g[:, m] * ((1.0 if m == q else 0.0) - g[:, q])
                block = (x * (w * factor)[:, None]).T @ x
                out[m, :, q, :] = block
                out[q, :, m, :] = block.T
        return np.asarray(out.reshape(k * d, k * d) + self.penalty, dtype=float)


def _reference_form(fit: MechanismFit, n_arms: int) -> FloatArray:
    """``(K - 1, d)`` coefficients against the lowest arm, intercept first when fitted."""
    assert fit.coef is not None and fit.intercept is not None
    coef = np.asarray(fit.coef, dtype=float)
    intercept = np.asarray(fit.intercept, dtype=float).reshape(-1)
    if fit.fit_intercept:
        full = np.column_stack([intercept, coef]) if coef.ndim == 2 else coef
    else:
        full = coef
    if n_arms == 2:
        return np.asarray(full[:1], dtype=float)
    return np.asarray(full[1:] - full[0], dtype=float)


def _penalty_matrix(fit: MechanismFit, n_arms: int, d: int) -> FloatArray:
    """``P / C`` in reference form, from scikit-learn's own penalty."""
    p = (n_arms - 1) * d
    if _resolved_penalty(fit) == "none":
        return np.zeros((p, p))
    slopes = np.eye(d)
    if fit.fit_intercept:
        slopes[0, 0] = 0.0
    if n_arms == 2:
        arm = np.ones((1, 1))
    else:
        arm = np.eye(n_arms - 1) - np.ones((n_arms - 1, n_arms - 1)) / n_arms
    return np.asarray(np.kron(arm, slopes) / fit.C, dtype=float)


def _softmax_reference(design: FloatArray, theta: FloatArray) -> FloatArray:
    """``(n, K)`` probabilities with the reference column first."""
    eta = np.column_stack([np.zeros(design.shape[0]), design @ theta.T])
    eta = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


def _design(fit: MechanismFit, covariates: FloatArray) -> FloatArray:
    x = np.asarray(covariates, dtype=float)
    for step in fit.steps or ():
        x = step.transform(x)
    if fit.fit_intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])
    return np.asarray(x, dtype=float)


def _pseudo_inverse(matrix: FloatArray, rank: int) -> tuple[FloatArray, float, int]:
    """The pseudo-inverse on the top ``rank`` eigen-directions, the condition number there,
    and the count of eigenvalues above the cutoff."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    top = float(values[0]) if values.size else 0.0
    above = int(np.sum(values > _RANK_CUTOFF * max(top, 0.0)))
    kept = values[:rank]
    if rank == 0 or kept[-1] <= 0.0:
        return np.zeros_like(matrix), float("inf"), above
    inverse = (vectors[:, :rank] / kept) @ vectors[:, :rank].T
    return np.asarray(inverse, dtype=float), float(kept[0] / kept[-1]), above


@dataclass(frozen=True)
class _FoldStack:
    stack: LogisticStack
    train: Any
    test: Any
    inverse: FloatArray


def _fold_rows(folds: Any, n: int) -> list[tuple[Any, Any]]:
    if folds.n_folds == 1:
        everything = np.arange(n)
        return [(everything, everything)]
    return list(folds)


def _stacks(
    fits: tuple[MechanismFit, ...],
    nuisance: Any,
    data: Any,
) -> list[_FoldStack] | MechanismRefusal:
    """Checks 8 to 11 per fold, in fold order, and each fold's stack."""
    arms = tuple(float(a) for a in nuisance.arms)
    k = len(arms)
    n = data.n
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    stored = np.asarray(nuisance.propensity.values, dtype=float)
    rows = _fold_rows(nuisance.folds, n)
    if len(rows) != len(fits):
        return _refusal(
            "mismatch",
            "the stored treatment mechanism does not equal the LogisticRegression's own "
            "predictions",
        )
    out: list[_FoldStack] = []
    for fold, (fit, (train, test)) in enumerate(zip(fits, rows, strict=True)):
        classes = None if fit.classes is None else tuple(float(c) for c in fit.classes)
        if fit.kind == "constant" or classes != arms:
            seen = set() if classes is None else set(classes)
            missing = next((a for a in arms if a not in seen), arms[0])
            return _refusal(
                "missing_arm",
                f"the training rows of fold {fold} hold no unit of arm {missing:g}, so the "
                f"LogisticRegression of that fold is not the {k}-arm model",
            )
        design = _design(fit, data.covariates)
        theta = _reference_form(fit, k)
        stack = LogisticStack(
            design=design,
            theta=theta,
            penalty=_penalty_matrix(fit, k, design.shape[1]),
            probabilities=_softmax_reference(design, theta),
        )
        information = stack.information(train, weights)
        design_rank = int(np.linalg.matrix_rank(design[train] * np.sqrt(weights[train])[:, None]))
        expected = stack.dimension if _resolved_penalty(fit) == "l2" else (k - 1) * design_rank
        inverse, condition, above = _pseudo_inverse(information, expected)
        residual = stack.score(treatment_index, train, weights).sum(axis=0) - (
            stack.penalty @ theta.reshape(-1)
        )
        step = inverse @ residual
        scale = np.sqrt(np.clip(np.diag(inverse), 0.0, None))
        visible = scale > 0.0
        largest = float(np.max(np.abs(step[visible]) / scale[visible])) if visible.any() else 0.0
        if not np.isfinite(largest) or largest > ROOT_TOLERANCE:
            remedy = (
                " The solver stopped at max_iter."
                if fit.max_iter and fit.n_iter >= fit.max_iter
                else ""
            )
            return _refusal(
                "score_not_zero",
                f"the score of the LogisticRegression of fold {fold} is not zero at its "
                f"coefficients: the largest Newton step is {largest:.3g} standard errors."
                f"{remedy} Lower tol, raise max_iter, or use solver='newton-cholesky'",
                alternative="Refit with a tighter LogisticRegression tol or a larger max_iter.",
            )
        if above < expected or condition > CONDITION_LIMIT:
            return _refusal(
                "singular",
                f"the information matrix of the mechanism in fold {fold} is singular "
                f"(condition number {condition:.3g}): the fitted probabilities reach 0 or 1, "
                "so the coefficients are not identified",
                alternative="Remove the covariates that separate the arms, or use a fixed C.",
            )
        recomputed = stack.probabilities[test]
        if float(np.max(np.abs(recomputed - stored[test]))) > MISMATCH_TOLERANCE:
            return _refusal(
                "mismatch",
                "the stored treatment mechanism does not equal the LogisticRegression's own "
                "predictions",
            )
        out.append(_FoldStack(stack=stack, train=train, test=test, inverse=inverse))
    return out


def representer_g_derivative(
    group: str,
    arm: float,
    versus: float | None,
    arms: tuple[float, ...],
    treatment: FloatArray,
    bounded: FloatArray,
    share: float | None,
) -> tuple[FloatArray, FloatArray]:
    r"""The representer and its derivative in each column of the bounded mechanism.

    Parameters
    ----------
    group : str
        ``"mean"``, ``"att"`` or ``"atc"``.
    arm : float
        The parameter's arm.
    versus : float or None
        The arm it is contrasted with, or ``None`` for a mean.
    arms : tuple of float
        The arm codes, in column order.
    treatment : ndarray
        ``(n,)`` observed arm codes.
    bounded : ndarray
        ``(n, K)`` truncated mechanism.
    share : float or None
        The share of the conditioning arm, for ``"att"`` and ``"atc"``.

    Returns
    -------
    tuple of ndarray
        ``(n,)`` values of :math:`\alpha` and ``(n, K)`` values of
        :math:`\partial\alpha/\partial g_c`.
    """
    g = np.asarray(bounded, dtype=float)
    a = np.asarray(treatment, dtype=float)
    column = {level: j for j, level in enumerate(arms)}
    n = g.shape[0]
    alpha = np.zeros(n)
    derivative = np.zeros_like(g)
    if group == "mean":
        terms = [(arm, 1.0)] if versus is None else [(arm, 1.0), (versus, -1.0)]
        for level, sign in terms:
            j = column[level]
            on = (a == level).astype(float)
            alpha += sign * on / g[:, j]
            derivative[:, j] += -sign * on / g[:, j] ** 2
        return alpha, derivative
    assert versus is not None and share is not None
    i, r = column[arm], column[versus]
    on_a = (a == arm).astype(float)
    on_r = (a == versus).astype(float)
    if group == "att":
        alpha = (on_a - on_r * g[:, i] / g[:, r]) / share
        derivative[:, i] = -on_r / (g[:, r] * share)
        derivative[:, r] = on_r * g[:, i] / (g[:, r] ** 2 * share)
        return alpha, derivative
    alpha = (on_a * g[:, r] / g[:, i] - on_r) / share
    derivative[:, r] = on_a / (g[:, i] * share)
    derivative[:, i] = -on_a * g[:, r] / (g[:, i] ** 2 * share)
    return alpha, derivative


def plugin_mechanism_term(
    result: Any,
    repeat: Any,
    parameter: Any,
    representer: FloatArray,
    bounds: tuple[float, float],
    share: float | None,
) -> FloatArray | MechanismRefusal:
    r"""The mechanism term :math:`\psi^\gamma` of the plug-in :math:`\nu^2` curve.

    Parameters
    ----------
    result : TMLEResult
        The fitted result whose bound is computed.
    repeat : RepeatFit
        The cross-fitting draw the bound reads.
    parameter : ArmParameter
        The bounded parameter.
    representer : ndarray
        ``(n,)`` Riesz representer the bound squares, as the package built it.
    bounds : tuple of float
        The truncation bounds of the parameter's group.
    share : float or None
        The share of the conditioning arm, for ATT and ATC.

    Returns
    -------
    ndarray or MechanismRefusal
        The ``(n,)`` term, or the reason the fit does not qualify.
    """
    nuisance = repeat.nuisance
    data = result.data
    truncation = nuisance.propensity.truncate(bounds)
    if nuisance.treatment_mechanism == "known":
        if bool(np.any(truncation.clipped)):
            return _refusal(
                "known_truncated",
                "the truncation bounds move a declared known mechanism, so the representer "
                "is not alpha(g_0)",
                alternative="Pass g_bounds=(lower, upper) that contains every known probability.",
            )
        return np.zeros(data.n)
    fits: tuple[MechanismFit, ...] | None = nuisance.treatment_fits
    if not fits:
        return _refusal(
            "unrecorded",
            "the fit records no fitted treatment learner, so the mechanism score is not available",
        )
    arms = tuple(float(a) for a in nuisance.arms)
    refusal = _learner_refusal(fits[0], len(arms))
    if refusal is not None:
        return refusal
    folds = _stacks(fits, nuisance, data)
    if isinstance(folds, MechanismRefusal):
        return folds

    alpha, d_alpha = representer_g_derivative(
        parameter.group,
        parameter.arm,
        parameter.versus,
        arms,
        np.asarray(data.treatment, dtype=float),
        truncation.values,
        share,
    )
    if not np.allclose(alpha, representer, rtol=1e-10, atol=1e-12):
        raise RuntimeError(  # pragma: no cover - an internal consistency guard
            f"the closed-form representer of {parameter.name!r} departs from the fitted one"
        )
    moved: BoolArray = np.asarray(truncation.clipped, dtype=bool)
    b = np.where(moved, 0.0, d_alpha)
    n = data.n
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    term = np.zeros(n)
    for fold in folds:
        stack = fold.stack
        g = stack.probabilities
        # dalpha/dtheta_m = x * g_m (b_m - sum_c b_c g_c), with the reference column dropped.
        inner = (b * g).sum(axis=1)
        t = g[:, 1:] * (b[:, 1:] - inner[:, None])
        test = fold.test
        coefficient = (weights[test] * 2.0 * alpha[test])[:, None] * t[test]
        gradient = (coefficient[:, :, None] * stack.design[test][:, None, :]).sum(axis=0) / n
        direction = fold.inverse @ gradient.reshape(-1)
        rows = stack.score(treatment_index, fold.train, weights)
        term[fold.train] += n * rows @ direction
        allocation = weights / n * float(direction @ (stack.penalty @ stack.theta.reshape(-1)))
        term -= n * allocation
    return term
