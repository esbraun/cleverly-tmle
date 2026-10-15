"""The score of the fluctuation submodel, and the scale it is judged against.

These four helpers are shared by every solver and by the estimator that stitches
fold-specific fluctuations back together.  They live in their own module because
they had previously been written out three times -- once in
:mod:`cleverly.fluctuation.iterative`, once again in
:mod:`cleverly.fluctuation.one_step` (which imported the private names from its
sibling), and a third near-verbatim copy in :mod:`cleverly.estimators.tmle`.  Three
implementations of one quantity is three chances for them to drift, and the score is
precisely the quantity the whole method is defined by: TMLE is the estimator that
solves ``P_n D*(hat P) = 0``, and :func:`cleverly.validation.score_check` audits how
close to zero it got.

Everything here takes and returns plain arrays, so this module imports nothing from
its siblings and no import cycle is possible.

The second half holds the stopping rules of ``Targeting(stopping=)``.  A rule judges the
score against a threshold that is computed from the data, once, in :func:`rule_ratio_at`,
which every solver and every outer loop that judges a score calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._refusal import Refusal, RefusalReason
from .._typing import BoolArray, FloatArray, FluctuationKind, IntArray
from ..exceptions import MethodConfigurationError
from ..inference.delta import reference_quantile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..inference.influence import ParameterEstimate

__all__ = [
    "LINEAR_RULE_REFUSAL",
    "RESIDUAL_LINE",
    "RULE_FAILURE_LINE",
    "STOPPING_RULES",
    "STOP_CONTEXT_MISSING",
    "StopContext",
    "absolute_score_weights",
    "check_stopping_rule",
    "quasi_loglik",
    "relative_score",
    "residual_line",
    "residual_se",
    "rule_contributions",
    "rule_failure_line",
    "rule_ratio_at",
    "rule_thresholds",
    "ruled_warning",
    "score_columns",
    "score_scale",
    "unit_influence",
    "wald_coverage_bound",
]

#: The values of ``Targeting(stopping=)``, shipped rule first.
STOPPING_RULES = ("relative", "scaled_var", "sample_size")

#: The refusal of a stopping rule (and, once they exist, an iterate or a step bound) beside
#: ``fluctuation="linear"``, which solves the score equation in one closed-form step.
LINEAR_RULE_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "fluctuation='linear' solves the score equation in one closed-form step, so it has no "
    "iterate, stopping rule or step bound to choose. Remove iterate=, stopping= and "
    "step_bound=, or use fluctuation='logistic'",
)

#: Raised by a solver that is asked to judge a declared rule and holds no
#: :class:`StopContext`.  A call site that misses the context fails here, loudly, rather than
#: judging the score on a threshold it had to invent.
STOP_CONTEXT_MISSING = (
    "a declared stopping rule needs a StopContext (the unit of each row and the outcome "
    "range) to compute its threshold, and this call passed none. Pass stop= to the solver"
)


def absolute_score_weights(h: FloatArray, weights: FloatArray, mask: BoolArray) -> FloatArray:
    """Return ``abs(w_i * h_ij)`` on exactly the rows an outcome score uses.

    Unlike :func:`score_scale`, this keeps the row axis.  A fitted fluctuation retains
    the result so post-fit diagnostics can inspect the score that was actually solved,
    including estimator-specific submodels and validation-risk weights that cannot be
    reconstructed from the initial nuisance estimates alone.
    """
    selected = np.asarray(mask, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    covariate = np.asarray(h, dtype=float)
    return np.asarray(np.abs(w[selected])[:, None] * np.abs(covariate[selected]), dtype=float)


def score_columns(
    y: FloatArray,
    q_star: FloatArray,
    h: FloatArray,
    weights: FloatArray,
    mask: BoolArray,
) -> FloatArray:
    """``mean(w * h * (Y - Q*))`` over observed rows, scaled by the full sample.

    The mean is taken over *all* ``n`` rows, not just the observed ones, because the
    estimating equation carries a ``Delta`` factor: unobserved rows contribute a
    genuine zero rather than being excluded from the average.

    ``h`` is always the *unweighted* clever covariate, even when the fluctuation was
    fit in its weighted form (R's ``target.gwt``).  The weighted form is a
    reparameterisation that solves the same equation, so the score has to be reported
    on the one canonical scale or two fits of the same model would not be comparable.
    """
    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q_star[mask]
    contribution = (weights * residual)[:, None] * h
    return np.asarray(contribution.mean(axis=0), dtype=float)


def score_scale(h: FloatArray, weights: FloatArray, mask: BoolArray) -> FloatArray:
    """Per-column ``mean(|w * h|)``: the largest the score could be.

    The residual ``Y - Q*`` is bounded by one on the ``[0, 1]`` outcome scale, so this
    bounds ``|score|`` and makes the ratio dimensionless.
    """
    contribution = np.zeros_like(h)
    contribution[mask] = np.abs(weights[mask])[:, None] * np.abs(h[mask])
    return np.asarray(contribution.mean(axis=0), dtype=float)


def relative_score(score: FloatArray, scale: FloatArray) -> float:
    """Largest score component relative to its maximum possible magnitude."""
    if score.size == 0:
        return 0.0
    return float(np.max(np.abs(score) / np.maximum(scale, 1e-300)))


def quasi_loglik(y: FloatArray, p: FloatArray, weights: FloatArray) -> float:
    """Weighted binomial quasi-log-likelihood, valid for ``y`` in ``[0, 1]``."""
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(np.sum(weights * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def check_stopping_rule(stopping: object, fluctuation: FluctuationKind | str) -> None:
    """Refuse a stopping rule that does not exist, or that its fluctuation cannot take.

    Raises
    ------
    ValueError
        If ``stopping`` is not one of :data:`STOPPING_RULES`.
    MethodConfigurationError
        If a declared rule meets ``fluctuation="linear"``.
    """
    if stopping not in STOPPING_RULES:
        raise ValueError(
            f"stopping must be 'relative', 'scaled_var' or 'sample_size'; got {stopping!r}"
        )
    if stopping != "relative" and fluctuation == "linear":
        raise MethodConfigurationError.from_refusal(LINEAR_RULE_REFUSAL)


@dataclass(frozen=True)
class StopContext:
    """What a declared stopping rule needs besides the score itself.

    Parameters
    ----------
    units : ndarray of int or None
        The unit code of each row the solver receives.  Rows that share a code are one
        unit, a cluster.  ``None`` means every row is its own unit.
    span : float
        The outcome range, :attr:`cleverly.utils.bounds.OutcomeScaler.range`.  The solver
        works on the ``[0, 1]`` scale and a rule judges the score in the outcome's units.
    """

    units: IntArray | None
    span: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.span) and self.span > 0.0):
            raise ValueError(f"span must be a positive finite number; got {self.span!r}")
        if self.units is not None and np.asarray(self.units).ndim != 1:
            raise ValueError("units must be a one-dimensional array with one code per row")

    def restrict(self, rows: IntArray | BoolArray) -> StopContext:
        """The context of a subset of the rows, as a fold or a candidate solves them.

        Parameters
        ----------
        rows : ndarray
            Row indices, or a boolean mask, into the rows this context describes.

        Returns
        -------
        StopContext
            The same ``span``, with the unit codes of those rows.  The unit count of the
            subset is the number of distinct codes it holds.
        """
        if self.units is None:
            return self
        return StopContext(np.asarray(self.units)[np.asarray(rows)], self.span)


def unit_influence(contributions: FloatArray, units: IntArray | None) -> tuple[FloatArray, int]:
    """Per-unit influence ``IC_uj = (n_u / n) * sum_{i in u} c_ij``, and the unit count.

    Without ``units`` every row is a unit and the contributions are returned as they are.
    """
    c = np.asarray(contributions, dtype=float)
    n = c.shape[0]
    if units is None:
        return c, n
    codes = np.asarray(units)
    if codes.shape[0] != n:
        raise ValueError(
            f"the stop context holds {codes.shape[0]} unit code(s) for {n} row(s); a "
            "solver's context must describe exactly the rows it receives"
        )
    _, inverse = np.unique(codes, return_inverse=True)
    n_units = int(inverse.max()) + 1 if n else 0
    sums = np.column_stack(
        [np.bincount(inverse, weights=c[:, j], minlength=n_units) for j in range(c.shape[1])]
    )
    return (n_units / n) * sums, n_units


def rule_contributions(
    y: FloatArray,
    q_star: FloatArray,
    h: FloatArray,
    weights: FloatArray,
    mask: BoolArray,
    span: float,
    arm_curves: Mapping[int, FloatArray] | None = None,
) -> FloatArray:
    """Per-row contribution of each score column, in the outcome's units.

    The default column is the score column ``span * w * h * (Y - Q*)``, zero where the
    outcome is unobserved.  A column listed in ``arm_curves`` targets one arm mean and takes
    the full curve ``span * (w * h * (Y - Q*) + Q*_a(W) - P_n Q*_a)``, whose sample SD is the
    arm mean's standard error times ``sqrt(n)``.

    Parameters
    ----------
    y, q_star, h, weights, mask
        As :func:`score_columns` reads them.
    span : float
        The outcome range.
    arm_curves : mapping of int to ndarray, optional
        Column index to the targeted counterfactual prediction ``Q*_a(W)`` of the arm that
        column targets.
    """
    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q_star[mask]
    score_part = (weights * residual)[:, None] * h
    out = span * score_part
    for column, q_arm in (arm_curves or {}).items():
        out[:, column] = span * (score_part[:, column] + q_arm - q_arm.mean())
    return np.asarray(out, dtype=float)


def rule_thresholds(contributions: FloatArray, stop: StopContext, stopping: str) -> FloatArray:
    """The threshold of each score column under a declared rule, in the outcome's units.

    ``sample_size`` is ``1 / n_u``.  ``scaled_var`` is ``sd_j / (sqrt(n_u) * min(log n_u,
    10))`` with ``sd_j`` the sample SD (``ddof=1``) of the per-unit influence of column ``j``,
    which is the ``tmle3`` ``check_convergence`` formula.
    """
    if stopping not in ("scaled_var", "sample_size"):
        raise ValueError(f"no threshold is defined for stopping={stopping!r}")
    influence, n_units = unit_influence(contributions, stop.units)
    columns = influence.shape[1]
    if stopping == "sample_size":
        return np.full(columns, 1.0 / n_units)
    if n_units < 2:
        raise ValueError(
            "stopping='scaled_var' needs at least two units to estimate a standard deviation; "
            f"got {n_units}"
        )
    sd = np.std(influence, axis=0, ddof=1)
    return np.asarray(sd / (np.sqrt(n_units) * min(np.log(n_units), 10.0)), dtype=float)


def rule_ratio_at(
    score: FloatArray, contributions: FloatArray, stop: StopContext, stopping: str
) -> tuple[FloatArray, float]:
    """Thresholds, and the largest score over its threshold, at one pair of score and fit.

    This is the single place thresholds are computed.  A ratio is valid only at the pair it
    was computed at: a caller that moves the fit after a solve calls this again on the
    score and the contributions of the moved fit.

    Parameters
    ----------
    score : ndarray
        ``P_n S_j`` per column, on the ``[0, 1]`` scale, as :func:`score_columns` returns it.
    contributions : ndarray
        :func:`rule_contributions` at the same fit, one column per score column.
    stop : StopContext
        The units of the rows and the outcome range.
    stopping : {"scaled_var", "sample_size"}
        The declared rule.

    Returns
    -------
    thresholds : ndarray
        One per column, in the outcome's units.
    ratio : float
        ``max_j |P_n S_j| * span / threshold_j``.  The rule is met when it is at most one.
    """
    thresholds = rule_thresholds(contributions, stop, stopping)
    if score.size == 0:
        return thresholds, 0.0
    statistic = np.abs(score) * stop.span
    return thresholds, float(np.max(statistic / np.maximum(thresholds, 1e-300)))


def ruled_warning(stopping: str, steps: int, ratio: float, cause: str) -> str:
    """The warning of a targeting step that did not meet its declared rule."""
    return (
        f"targeting step did not meet the {stopping} stopping rule after {steps} "
        f"iteration(s) (largest score over its threshold = {ratio:.3g}): {cause}"
    )


#: The line a result prints under a declared stopping rule: the largest score residual in
#: standard errors, and the coverage a Wald interval keeps at that residual.
RESIDUAL_LINE = (
    "stopping rule {rule}: the largest score residual is {b:.3g} standard errors, on {name}. "
    "At that residual a {level:.0%} Wald interval keeps at least {bound:.4f} coverage."
)

#: The line a result prints when the targeting did not meet its declared stopping rule.
RULE_FAILURE_LINE = "the targeting did not meet the {rule} stopping rule; {detail}"


def wald_coverage_bound(residual: float, alpha: float) -> float:
    r"""Coverage a Wald interval keeps when its estimate carries a bias of ``residual`` SE.

    :math:`\Phi(z - b) - \Phi(-z - b)` with :math:`z = \Phi^{-1}(1 - \alpha/2)`.  It is the
    coverage of the interval :math:`\hat\psi \pm z\,\widehat{se}` when
    :math:`(\hat\psi - \psi) / \widehat{se}` is normal with mean :math:`b` and unit variance.
    A Student t reference (few clusters) has a larger critical value, so the coverage it keeps
    is at least this one.

    Parameters
    ----------
    residual : float
        The bias, in standard errors.  The sign does not matter.
    alpha : float
        The significance level of the interval.

    Returns
    -------
    float
        The coverage, ``nan`` for a non-finite residual.

    Examples
    --------
    >>> from cleverly.fluctuation._score import wald_coverage_bound
    >>> f"{wald_coverage_bound(0.0, 0.05):.4f}"
    '0.9500'
    >>> f"{wald_coverage_bound(0.1, 0.05):.4f}"
    '0.9489'
    """
    if not np.isfinite(residual):
        return float("nan")
    b = abs(float(residual))
    z = reference_quantile(alpha)
    return float(_normal_cdf(z - b) - _normal_cdf(-z - b))


def _normal_cdf(x: float) -> float:
    """The standard normal distribution function, from the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def residual_se(estimate: ParameterEstimate) -> float:
    """The mean of an estimate's influence curve, in its own standard errors.

    ``|P_n IC| / se`` on the scale the estimate is inferred on.  It is the bias that an
    unsolved score equation leaves, expressed so that it compares with an interval's half
    width.  ``nan`` when the standard error is zero or not finite.
    """
    error = estimate.plugin_std_error
    if not (np.isfinite(error) and error > 0.0):
        return float("nan")
    return float(abs(estimate.score) / error)


def residual_line(estimates: Mapping[str, ParameterEstimate], rule: str, alpha: float) -> str:
    """The pinned line of a result under a declared stopping rule.

    Parameters
    ----------
    estimates : mapping of str to ParameterEstimate
        The result's estimates.
    rule : str
        The declared rule, ``"scaled_var"`` or ``"sample_size"``.
    alpha : float
        The significance level of the result's intervals.

    Returns
    -------
    str
        :data:`RESIDUAL_LINE` filled at the estimand with the largest residual, or the empty
        string when no estimate has a finite residual.
    """
    residuals = {name: residual_se(estimate) for name, estimate in estimates.items()}
    finite = {name: value for name, value in residuals.items() if np.isfinite(value)}
    if not finite:
        return ""
    name = max(finite, key=lambda key: finite[key])
    return RESIDUAL_LINE.format(
        rule=rule,
        b=finite[name],
        name=name,
        level=1.0 - alpha,
        bound=wald_coverage_bound(finite[name], alpha),
    )


def rule_failure_line(rule: str, detail: str) -> str:
    """The pinned line of a result whose targeting did not meet its declared rule."""
    return RULE_FAILURE_LINE.format(rule=rule, detail=detail)
