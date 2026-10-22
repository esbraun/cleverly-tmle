r"""Bounds on arm means and the ATE under the marginal sensitivity model.

Tan (2006, Section 4.2, eq. (9)) lets an unobserved ``U`` move the odds of treatment by at
most a factor :math:`\Gamma \ge 1` against the odds given the covariates alone:

.. math::

    \Gamma^{-1} \le \frac{P(A = 1 \mid W, U) / P(A = 0 \mid W, U)}
                          {e(W) / (1 - e(W))} \le \Gamma,

with :math:`(Y(1), Y(0)) \perp A \mid (W, U)`.  :math:`\Gamma = 1` is no unmeasured
confounding.

**The sharp arm bound.**  Under the model :math:`E[Y(1) \mid A = 0, W] = E[w Y \mid A = 1, W]`
for a weight with values in :math:`[\Gamma^{-1}, \Gamma]` and conditional mean one.  The
largest such value is a linear program whose dual is

.. math::

    \rho^+(W) = \min_q \; q + E[\varphi_\Gamma(Y - q) \mid A = 1, W],
    \qquad \varphi_\Gamma(u) = \Gamma u_+ - \Gamma^{-1} u_-.

Any :math:`q` gives an upper bound (weak duality), and the :math:`\tau`-quantile of
:math:`Y \mid A = 1, W` with :math:`\tau = \Gamma / (1 + \Gamma)` attains it, for a
continuous or a discrete outcome.  The arm bound is
:math:`\psi^+_1 = E[A Y + (1 - A) \rho^+(W)]`.  The lower bound applies the same step to
:math:`-Y`, and arm 0 exchanges the roles of the arms.  The ATE bounds are
:math:`[\psi^-_1 - \psi^+_0, \psi^+_1 - \psi^-_0]`.

**The estimator.**  With :math:`R(q) = q + \varphi_\Gamma(Y - q)` the bound is estimated by
the cross-fitted mean of

.. math::

    A Y + (1 - A) \hat\rho(W) + A \frac{1 - \hat e(W)}{\hat e(W)}
    \bigl(R(\hat q) - \hat\rho(W)\bigr),

which is consistent for a valid bound when :math:`\hat e` or :math:`\hat\rho` is consistent,
and for the sharp bound when :math:`\hat q` also converges to the quantile.  The derivative
of :math:`E[R(q) \mid A = 1, W]` in :math:`q` vanishes at the quantile of a continuous
outcome, so the quantile error enters at second order.  :math:`\hat\rho` is
:math:`\hat\mu / \Gamma + (1 - 1/\Gamma)\hat c` with :math:`\hat\mu` the fit's out-of-fold
outcome regression and :math:`\hat c` a regression of the tail pseudo-outcome
:math:`q + (Y - q)_+ / (1 - \tau)`.  For each predicted fold, the quantile trains on one
half of the other folds and the tail regression on the other half, as the authors' code
does, so an overfitted quantile cannot bias a training pseudo-outcome.  On a binary
outcome both nuisances are closed forms of :math:`\hat\mu`:
:math:`\hat q = 1\{\hat\mu > 1 - \tau\}` and
:math:`\hat\rho^+ = \min(\Gamma\hat\mu, 1 - (1 - \hat\mu)/\Gamma)`.  These are the formulas of
the authors' replication code for Dorn, Guo and Kallus,
``CausalML/DoublyValidDoublySharpSensitivity`` (``simplified_dvds.R``).

**The interval.**  ``[lower - z se_lower, upper + z se_upper]`` with
:math:`z = \Phi^{-1}(1 - (1 - \text{level})/2)` covers the identified set with asymptotic
probability at least ``level`` by the union bound.  It is a set interval: the tighter
interval for the parameter alone (Imbens and Manski 2004) is not shipped.  The binary
quantile is a plug-in threshold, so its sharpness needs a margin condition: the outcome
regression puts little mass near :math:`1 - \tau`.  Validity needs no such condition.

**What the interval may claim.**  It resolves through the registry of
:mod:`cleverly.validation._interpretation`, instrument :data:`MSM_BOUND_INSTRUMENT`.  Until
a registered coverage cell adds a row, it prints the registry's descriptive sentence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm
from sklearn.base import clone

from .._inference_status import status_record
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from ..exceptions import CapabilityError
from ..inference.cluster import influence_variance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult
    from ..validation._interpretation import Resolution

__all__ = [
    "MSM_BOUND_INSTRUMENT",
    "MarginalSensitivityBounds",
    "MarginalSensitivityRow",
    "marginal_sensitivity_bounds",
]

FloatArray = np.ndarray

#: The registry instrument of the bound's confidence limits.  The rule is the outcome kind,
#: ``"continuous"`` or ``"binary"``, with ``"+clustered"`` on a clustered fit.
MSM_BOUND_INSTRUMENT = "msm_bound_limit"

#: The default grid of the sensitivity parameter.
DEFAULT_GAMMA_GRID = (1.0, 1.5, 2.0)

#: The fold count of the bound's own split when the fit was not cross-fitted.
_IN_SAMPLE_FOLDS = 5

#: The roadmap row of the compositions with no known calculation.
_OWNER = "X87"

#: The roadmap row of the compositions X82 part H2 adds.
_PART_H2 = "X82"

#: The fitted methods whose stored mechanism estimates ``P(A | W)``.
_FULL_MECHANISM_METHODS = frozenset({"tmle", "drtmle"})


@dataclass(frozen=True)
class MarginalSensitivityRow:
    """The bounds at one value of the sensitivity parameter.

    Parameters
    ----------
    gamma : float
        The sensitivity parameter :math:`\\Gamma`.
    estimand : str
        The bounded parameter.
    lower : float
        The estimated lower bound.
    upper : float
        The estimated upper bound.
    lower_std_error : float
        The standard error of ``lower``.
    upper_std_error : float
        The standard error of ``upper``.
    lower_limit : float
        ``lower`` less ``z`` standard errors.
    upper_limit : float
        ``upper`` plus ``z`` standard errors.
    """

    gamma: float
    estimand: str
    lower: float
    upper: float
    lower_std_error: float
    upper_std_error: float
    lower_limit: float
    upper_limit: float


@dataclass(frozen=True)
class MarginalSensitivityBounds:
    """Marginal sensitivity model bounds over a grid of :math:`\\Gamma`.

    Parameters
    ----------
    rows : tuple of MarginalSensitivityRow
        One row per :math:`\\Gamma`, in increasing order.
    estimand : str
        The bounded parameter.
    level : float
        The coverage level of the set interval.
    null : float or None
        The null value of the estimand, ``0.0`` for the ATE and ``None`` for an arm mean.
    outcome_kind : {"continuous", "binary"}
        Which nuisance route the bounds used.
    quantile_learner : tuple of str
        The quantile learner each side and arm selected, or ``()`` for a binary outcome.
    pseudo_outcome_learner : tuple of str
        The tail-regression learner each side and arm selected, or ``()``.
    limit_reading : Resolution or None
        The registry's reading of the confidence limits.

    Attributes
    ----------
    breakdown : float or None
    limit_breakdown : float or None
    limit_descriptive : bool
    """

    rows: tuple[MarginalSensitivityRow, ...]
    estimand: str
    level: float
    null: float | None
    outcome_kind: str
    quantile_learner: tuple[str, ...] = ()
    pseudo_outcome_learner: tuple[str, ...] = ()
    limit_reading: Resolution | None = field(default=None, compare=False)

    @property
    def breakdown(self) -> float | None:
        """The smallest grid :math:`\\Gamma` whose point bounds contain the null.

        ``None`` for an arm mean, or when no grid value reaches the null.
        """
        if self.null is None:
            return None
        for row in self.rows:
            if row.lower <= self.null <= row.upper:
                return row.gamma
        return None

    @property
    def limit_breakdown(self) -> float | None:
        """The smallest grid :math:`\\Gamma` whose set interval contains the null."""
        if self.null is None:
            return None
        for row in self.rows:
            if row.lower_limit <= self.null <= row.upper_limit:
                return row.gamma
        return None

    @property
    def limit_descriptive(self) -> bool:
        """Whether the confidence limits print without a licensed interpretation."""
        return self.limit_reading is None or not self.limit_reading.licensed

    def to_frame(self) -> Any:
        """Return one row per :math:`\\Gamma` as a pandas data frame.

        Returns
        -------
        pandas.DataFrame
            The fields of :class:`MarginalSensitivityRow`.
        """
        import pandas as pd

        return pd.DataFrame([row.__dict__ for row in self.rows])

    def summary(self) -> str:
        """Return a text report of the bounds and how to read them.

        Returns
        -------
        str
            The grid, the breakdown values, and the limit reading.
        """
        lines = [
            f"Marginal sensitivity model bounds for {self.estimand} "
            f"({self.outcome_kind} outcome, {self.level:.0%} set interval)",
        ]
        for row in self.rows:
            lines.append(
                f"  gamma={row.gamma:g}: bounds [{row.lower:.4g}, {row.upper:.4g}], "
                f"interval [{row.lower_limit:.4g}, {row.upper_limit:.4g}]"
            )
        if self.null is not None:
            lines.append(f"  breakdown gamma (bounds): {self.breakdown}")
            lines.append(f"  breakdown gamma (interval): {self.limit_breakdown}")
        if self.limit_reading is not None:
            lines.append(f"  confidence limit: {self.limit_reading.sentence}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Refusals


def _gap(condition: str, roadmap: str = _OWNER) -> Refusal:
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP, condition, alternative=NO_ALTERNATIVE, roadmap=roadmap
    )


#: The refusal on a structural nested mean model fit.
STRUCTURAL_REFUSAL = Refusal(
    RefusalReason.DIFFERENT_ESTIMAND,
    "the marginal sensitivity model bounds an arm mean of a point treatment. A structural "
    "nested mean model reports a blip parameter, which is a different functional.",
    alternative=NO_ALTERNATIVE,
)


def marginal_bound_refusal(result: Any) -> Refusal | None:
    """The refusal of the bounds on ``result``, or ``None`` when they apply.

    Every rule reads the fitted result only, so the facade can declare the refusal
    without running a learner.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        A fitted result.

    Returns
    -------
    Refusal or None
        The first failed condition.
    """
    family = getattr(result, "assessment_family", "point")
    if family == "longitudinal":
        return _gap(
            "the marginal sensitivity model bounds are derived for a point treatment. The "
            "longitudinal analogue of Tan (2025) gives population bounds only, with no sample "
            "estimator.",
            "F16",
        )
    if family == "structural":
        return STRUCTURAL_REFUSAL
    data = result.data
    config = result.config
    if getattr(config, "continuous_treatment", False) or getattr(data, "policies", None):
        return Refusal(
            RefusalReason.DIFFERENT_ESTIMAND,
            "the marginal sensitivity model bounds the odds of a discrete treatment. A dose, "
            "a policy or a regime asks a different question that the model does not state.",
            alternative=NO_ALTERNATIVE,
        )
    if getattr(config, "parameter_axis", "arm") != "arm":
        return Refusal(
            RefusalReason.DIFFERENT_ESTIMAND,
            "the marginal sensitivity model bounds arm means and their difference. A regime, "
            "shift, incremental or MSM parameter is a different functional.",
            alternative=NO_ALTERNATIVE,
        )
    method = getattr(result, "fitted_method", "unknown")
    if method not in _FULL_MECHANISM_METHODS:
        return _gap(
            f"a {method} fit stores a selected or targeted treatment mechanism, and the bound "
            "divides by an estimate of P(A | W). A refit of the full mechanism is the step.",
            _PART_H2,
        )
    if getattr(data, "intermediate", None) is not None:
        return _gap(
            "a controlled direct effect also needs a bound on confounding of the intermediate "
            "and the outcome, which the model does not state."
        )
    if getattr(data, "treatment_observed", None) is not None:
        return _gap(
            "a missing treatment needs a joint model of U on the treatment and on its "
            "observation, which no read source defines."
        )
    if bool(np.any(~np.asarray(data.observed, dtype=bool))):
        return _gap("the bounds on a fit with missing outcomes are not implemented yet.", _PART_H2)
    if not data.is_binary_treatment:
        return _gap(
            "the bounds on a treatment with more than two arms are not implemented yet.", _PART_H2
        )
    if data.weights_name is not None:
        return _gap("the bounds on a weighted fit are not implemented yet.", _PART_H2)
    if getattr(data, "strata", None) is not None:
        return _gap("the stratum-conditional bounds are not implemented yet.", _PART_H2)
    if data.family not in ("gaussian", "binomial"):
        return Refusal(
            RefusalReason.DIFFERENT_ESTIMAND,
            f"the bounds read a continuous or a binary outcome; this fit's family is "
            f"{data.family!r}.",
            alternative=NO_ALTERNATIVE,
        )
    if data.family == "binomial":
        values = np.unique(np.asarray(data.outcome, dtype=float))
        if not np.all(np.isin(values, (0.0, 1.0))):
            return Refusal(
                RefusalReason.DIFFERENT_ESTIMAND,
                "a binomial fit with outcomes other than 0 and 1 is neither binary nor "
                "continuous, and the bound's quantile route covers neither case.",
                alternative=NO_ALTERNATIVE,
            )
    status = result.inference_status
    status = status() if callable(status) else status
    if status != "influence_curve":
        record = status_record(status)
        return Refusal(
            RefusalReason.INFERENCE_GAP,
            "the bound's interval is built from its influence curve on a fit whose "
            "nuisances support one. " + record.reason,
            alternative=record.alternative or NO_ALTERNATIVE,
            roadmap=record.reopened_by,
        )
    return None


# ---------------------------------------------------------------------------------------
# Learners


QuantileFactory = Callable[[float], Any]


def _default_quantile_candidates(tau: float, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import QuantileRegressor

    return {
        "linear quantile": QuantileRegressor(quantile=tau, alpha=0.0, solver="highs"),
        "boosted quantile": HistGradientBoostingRegressor(
            loss="quantile",
            quantile=tau,
            max_depth=3,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        ),
    }


def _default_mean_candidates(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    return {
        "linear": LinearRegression(),
        "boosted": HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, early_stopping=True, random_state=seed
        ),
    }


def _pinball(y: FloatArray, q: FloatArray, tau: float) -> float:
    u = y - q
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))


# ---------------------------------------------------------------------------------------
# One side of one arm


@dataclass(frozen=True)
class _Side:
    """The cross-fitted summand of one arm bound, on the outcome scale."""

    value: float
    curve: FloatArray
    quantile_learner: str | None = None
    pseudo_outcome_learner: str | None = None


def _binary_side(
    y: FloatArray, treated: FloatArray, e: FloatArray, mu: FloatArray, gamma: float, upper: bool
) -> _Side:
    """The binary-outcome side, with both nuisances closed forms of ``mu``."""
    tau = gamma / (1.0 + gamma)
    if upper:
        q = (mu > 1.0 - tau).astype(float)
        rho = np.minimum(1.0 - 1.0 / gamma + mu / gamma, gamma * mu)
        residual = y - q
        r = q + np.where(residual > 0, gamma, 1.0 / gamma) * residual
    else:
        q = (mu > tau).astype(float)
        rho = np.maximum(1.0 - gamma + gamma * mu, mu / gamma)
        residual = y - q
        r = q + np.where(residual < 0, gamma, 1.0 / gamma) * residual
    summand = treated * y + (1.0 - treated) * rho + treated * (1.0 - e) / e * (r - rho)
    value = float(np.mean(summand))
    return _Side(value, summand - value)


def _continuous_side(
    x: FloatArray,
    y: FloatArray,
    treated: FloatArray,
    e: FloatArray,
    mu: FloatArray,
    gamma: float,
    upper: bool,
    assignment: Any,
    n_folds: int,
    quantile_learner: QuantileFactory | None,
    pseudo_outcome_learner: Any,
    seed: int,
) -> _Side:
    """The continuous-outcome side: the upper bound of ``sign * Y``, read back by sign."""
    sign = 1.0 if upper else -1.0
    y_s, mu_s = sign * y, sign * mu
    if gamma == 1.0:
        # phi_1(u) = u, so R = Y and rho = mu: the AIPW mean, with no learner to fit.
        summand = treated * y_s + (1.0 - treated) * mu_s + treated * (1.0 - e) / e * (y_s - mu_s)
        value = float(np.mean(summand))
        return _Side(sign * value, sign * (summand - value))
    tau = gamma / (1.0 + gamma)
    rows = treated > 0.5
    n = y.shape[0]

    def tail(q: FloatArray, target: FloatArray) -> FloatArray:
        return q + np.maximum(target - q, 0.0) / (1.0 - tau)

    # For fold k the other folds split by parity of their order: the quantile trains on
    # one half and the tail regression on the other, so no training pseudo-outcome reads a
    # quantile fitted on its own row. This is the split of the authors' code
    # (``DVDS.R``, ``semiadaptive = FALSE``).
    halves = []
    for k in range(n_folds):
        others = assignment[assignment != k]
        order = np.where(others > k, others - 1, others)
        first = np.zeros(n, dtype=bool)
        first[assignment != k] = order % 2 == 0
        second = (assignment != k) & ~first
        halves.append((assignment == k, first, second))

    if quantile_learner is None:
        q_candidates = _default_quantile_candidates(tau, seed)
    else:
        q_candidates = {"quantile_learner": quantile_learner(tau)}
    q_name: str | None = None
    q_hat = np.empty(n, dtype=float)
    q_second: list[FloatArray] = []
    best_risk = np.inf
    for name, learner in q_candidates.items():
        held_out = np.empty(n, dtype=float)
        on_second: list[FloatArray] = []
        for test, first, second in halves:
            model = clone(learner).fit(x[rows & first], y_s[rows & first])
            held_out[test] = model.predict(x[test])
            on_second.append(np.asarray(model.predict(x[rows & second]), dtype=float))
        risk = _pinball(y_s[rows], held_out[rows], tau)
        if risk < best_risk:
            q_name, q_hat, q_second, best_risk = name, held_out, on_second, risk

    c_candidates = (
        _default_mean_candidates(seed)
        if pseudo_outcome_learner is None
        else {"pseudo_outcome_learner": pseudo_outcome_learner}
    )
    c_name: str | None = None
    c_hat = np.empty(n, dtype=float)
    best_risk = np.inf
    for name, learner in c_candidates.items():
        predictions = np.empty(n, dtype=float)
        for (test, _first, second), q_on_second in zip(halves, q_second, strict=True):
            train = rows & second
            target = tail(q_on_second, y_s[train])
            model = clone(learner).fit(x[train], target)
            predictions[test] = model.predict(x[test])
        # The held-out loss reads each row's pseudo-outcome from the quantile fitted
        # without its fold, as the evaluation does.
        risk = float(np.mean((tail(q_hat[rows], y_s[rows]) - predictions[rows]) ** 2))
        if risk < best_risk:
            c_name, c_hat, best_risk = name, predictions, risk
    rho = mu_s / gamma + (1.0 - 1.0 / gamma) * c_hat
    r = y_s / gamma + (1.0 - 1.0 / gamma) * tail(q_hat, y_s)
    summand = treated * y_s + (1.0 - treated) * rho + treated * (1.0 - e) / e * (r - rho)
    value = float(np.mean(summand))
    return _Side(sign * value, sign * (summand - value), q_name, c_name)


# ---------------------------------------------------------------------------------------
# The public function


def _estimand_arms(result: TMLEResult, estimand: str) -> tuple[float, float | None]:
    """``(arm, versus)``: the arm of a mean, or the arms of the difference ``arm - versus``."""
    arms = tuple(float(level) for level in result.repeats[0].nuisance.propensity.arms)
    reference = float(result.config.reference_arm)
    other = next(arm for arm in arms if arm != reference)
    if estimand == "ate":
        return other, reference
    names = {f"ey{int(arm)}": arm for arm in arms if float(arm).is_integer()}
    if estimand in names:
        return names[estimand], None
    raise ValueError(
        f"estimand must be 'ate' or an arm mean ({', '.join(sorted(names))}); got {estimand!r}"
    )


def _gammas(gamma: float | Sequence[float]) -> tuple[float, ...]:
    values = (gamma,) if isinstance(gamma, (int, float)) else tuple(gamma)
    if not values:
        raise ValueError("gamma needs at least one value")
    out = tuple(sorted({float(value) for value in values}))
    if any(not np.isfinite(value) or value < 1.0 for value in out):
        raise ValueError(f"every gamma must be a finite number of at least 1; got {values!r}")
    return out


def _draw_sides(
    result: TMLEResult,
    repeat: Any,
    gamma: float,
    arms: tuple[float, float | None],
    quantile_learner: QuantileFactory | None,
    pseudo_outcome_learner: Any,
) -> dict[tuple[float, bool], _Side]:
    """Every side the estimand needs, on one cross-fitting draw."""
    data = result.data
    nuisance = repeat.nuisance
    folds = nuisance.folds
    seed = int(repeat.seed) if repeat.seed is not None else 0
    if folds.n_folds < 3:
        # The quantile and the tail regression each train on one half of the folds outside
        # the predicted fold, which needs at least three folds.
        from ..learners.crossfit import make_folds

        folds = make_folds(
            data.n,
            _IN_SAMPLE_FOLDS,
            stratify=np.asarray(data.treatment, dtype=float),
            cluster=data.cluster,
            random_state=seed,
        )
    assignment, n_folds = np.asarray(folds.assignment), int(folds.n_folds)
    level_index = {float(level): i for i, level in enumerate(nuisance.propensity.arms)}
    g = nuisance.propensity.bounded(result.config.g_bounds)
    y = np.asarray(data.outcome, dtype=float)
    x = np.asarray(data.covariates, dtype=float)
    binary = data.family == "binomial"
    sides: dict[tuple[float, bool], _Side] = {}
    arm, versus = arms
    wanted = [(arm, True), (arm, False)]
    if versus is not None:
        wanted = [(arm, True), (arm, False), (versus, True), (versus, False)]
    for level, upper in wanted:
        treated = (np.asarray(data.treatment, dtype=float) == level).astype(float)
        e = g[:, level_index[level]]
        mu = nuisance.scaler.unscale_levels(np.asarray(nuisance.outcome.arms[level], dtype=float))
        if binary:
            sides[(level, upper)] = _binary_side(y, treated, e, mu, gamma, upper)
        else:
            sides[(level, upper)] = _continuous_side(
                x,
                y,
                treated,
                e,
                mu,
                gamma,
                upper,
                assignment,
                n_folds,
                quantile_learner,
                pseudo_outcome_learner,
                seed,
            )
    return sides


def _median_rule(values: FloatArray, variances: FloatArray) -> tuple[float, float]:
    """The repeat median and its variance (Chernozhukov et al. 2018, eq. 3.14)."""
    centre = float(np.median(values))
    return centre, float(np.median(variances + (values - centre) ** 2))


def _limit_reading(result: TMLEResult, outcome_kind: str, estimand: str) -> Resolution:
    from ..validation._interpretation import fit_configuration, resolve

    rule = outcome_kind + ("+clustered" if result.data.cluster is not None else "")
    try:
        fit = fit_configuration(result, estimand)
    except (AttributeError, KeyError, TypeError, ValueError):
        fit = None
    return resolve(MSM_BOUND_INSTRUMENT, rule, fit)


def marginal_sensitivity_bounds(
    result: TMLEResult,
    gamma: float | Sequence[float] = DEFAULT_GAMMA_GRID,
    *,
    estimand: str | None = None,
    level: float = 0.95,
    quantile_learner: QuantileFactory | None = None,
    pseudo_outcome_learner: Any = None,
) -> MarginalSensitivityBounds:
    """Bound an arm mean or the ATE under the marginal sensitivity model.

    Parameters
    ----------
    result : TMLEResult
        A fitted two-arm ``TMLE`` or ``DRTMLE`` result with a Gaussian or binary outcome.
    gamma : float or sequence of float, default=DEFAULT_GAMMA_GRID
        The sensitivity parameter :math:`\\Gamma \\ge 1` (Tan's :math:`\\Lambda`), or a grid.
        :math:`\\Gamma = 1` reproduces the cross-fitted AIPW estimate.
    estimand : str or None, default=None
        ``"ate"`` or an arm mean such as ``"ey1"``. ``None`` bounds the ATE, which every
        two-arm fit defines, whatever estimands the fit reported.
    level : float, default=0.95
        The coverage level of the set interval.
    quantile_learner : callable, optional
        A factory ``tau -> regressor`` for the conditional ``tau``-quantile of a continuous
        outcome. By default the bounds select, by out-of-fold pinball loss, between a
        linear quantile regression and a shallow boosted quantile regression.
    pseudo_outcome_learner : regressor, optional
        The regression of the tail pseudo-outcome. By default the bounds select, by
        out-of-fold squared loss, between a linear regression and a shallow booster.

    Returns
    -------
    MarginalSensitivityBounds
        One row per :math:`\\Gamma`.

    Raises
    ------
    CapabilityError
        When :func:`marginal_bound_refusal` refuses the fit.
    ValueError
        When ``gamma`` holds a value below 1, or ``estimand`` or ``level`` is invalid.

    See Also
    --------
    cleverly.sensitivity.omitted_variable_bounds : The partial-:math:`R^2` scale bound.
    cleverly.sensitivity.evalue : The risk-ratio scale summary.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.estimators import TMLE
    >>> from cleverly.sensitivity import marginal_sensitivity_bounds
    >>> rng = np.random.default_rng(0)
    >>> w = rng.normal(size=300)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-(w + a - 0.5))))
    >>> frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    >>> fit = TMLE(
    ...     outcome_learner=LogisticRegression(),
    ...     treatment_learner=LogisticRegression(),
    ...     random_state=0,
    ... ).fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()
    >>> bounds = marginal_sensitivity_bounds(fit, gamma=(1.0, 2.0))
    >>> row_1, row_2 = bounds.rows
    >>> bool(np.isclose(row_1.lower, row_1.upper))
    True
    >>> bool(row_2.lower < row_1.lower and row_2.upper > row_1.upper)
    True
    """
    refusal = marginal_bound_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level!r}")
    grid = _gammas(gamma)
    estimand = "ate" if estimand is None else estimand
    arm, versus = _estimand_arms(result, estimand)
    cluster = result.data.cluster
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    outcome_kind = "binary" if result.data.family == "binomial" else "continuous"
    rows: list[MarginalSensitivityRow] = []
    quantile_names: set[str] = set()
    pseudo_names: set[str] = set()
    for value in grid:
        lowers, uppers, lower_vars, upper_vars = [], [], [], []
        for repeat in result.repeats:
            sides = _draw_sides(
                result, repeat, value, (arm, versus), quantile_learner, pseudo_outcome_learner
            )
            for side in sides.values():
                if side.quantile_learner is not None:
                    quantile_names.add(side.quantile_learner)
                if side.pseudo_outcome_learner is not None:
                    pseudo_names.add(side.pseudo_outcome_learner)
            up, low = sides[(arm, True)], sides[(arm, False)]
            if versus is not None:
                up_v, low_v = sides[(versus, True)], sides[(versus, False)]
                up = _Side(up.value - low_v.value, up.curve - low_v.curve)
                low = _Side(low.value - up_v.value, low.curve - up_v.curve)
            lowers.append(low.value)
            uppers.append(up.value)
            lower_vars.append(influence_variance(low.curve, cluster))
            upper_vars.append(influence_variance(up.curve, cluster))
        lower, lower_var = _median_rule(np.asarray(lowers), np.asarray(lower_vars))
        upper, upper_var = _median_rule(np.asarray(uppers), np.asarray(upper_vars))
        lower_se, upper_se = float(np.sqrt(lower_var)), float(np.sqrt(upper_var))
        rows.append(
            MarginalSensitivityRow(
                gamma=value,
                estimand=estimand,
                lower=lower,
                upper=upper,
                lower_std_error=lower_se,
                upper_std_error=upper_se,
                lower_limit=lower - z * lower_se,
                upper_limit=upper + z * upper_se,
            )
        )
    return MarginalSensitivityBounds(
        rows=tuple(rows),
        estimand=estimand,
        level=level,
        null=0.0 if versus is not None else None,
        outcome_kind=outcome_kind,
        quantile_learner=tuple(sorted(quantile_names)),
        pseudo_outcome_learner=tuple(sorted(pseudo_names)),
        limit_reading=_limit_reading(result, outcome_kind, estimand),
    )
