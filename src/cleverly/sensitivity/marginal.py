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

**Compositions.**  Each arm bound is the mean of a per-row summand, so the other
parameters are means of the same summands.

* More than two arms: the model restricts the odds of ``A = a`` against ``A != a``, so
  ``e`` is ``P(A = a | W)`` and the summand reads the indicator of arm ``a``.
* ``att`` and ``atc`` (two arms): :math:`E[Y(0) \mid A = 1] P(A = 1)` is the mean of the
  arm-0 summand less its observed-arm part, and the mean is divided by the share of the
  conditioning arm (a delta-method curve).
* ``rr`` and ``or``: the ratio map is increasing in the arm and decreasing in the
  reference, so ``RR+ = psi+_1 / psi-_0``. Each end's limits come from its log-scale curve.
  The set is componentwise, with no sharpness claim beyond the arm bounds.
* Sampling weights: every mean is a weighted mean and every regression is weighted.
* Baseline strata: a stratum-conditional parameter is the mean over the stratum's rows,
  divided by the stratum's share.
* Missing outcomes: under :math:`\Delta \perp Y \mid A, W`, with ``U`` acting on ``A``
  only, the summand's residual terms are multiplied by
  :math:`\Delta / P(\Delta = 1 \mid A, W)` from the fit's response model, and the
  quantile and the tail regression train on observed rows.
* A selecting ``CTMLE`` fit stores a mechanism that does not estimate ``P(A | W)``, so the
  bounds refit the fit's treatment learner on every covariate, on the fit's folds, as
  :func:`~cleverly.sensitivity.omitted_variable_bounds` does under
  ``assume_consistent_mechanism=True``.

**The interval.**  ``[lower - z se_lower, upper + z se_upper]`` with
:math:`z = \Phi^{-1}(1 - (1 - \text{level})/2)` covers the identified set with asymptotic
probability at least ``level`` by the union bound.  It is a set interval: the tighter
interval for the parameter alone (Imbens and Manski 2004) is not shipped.  The binary
quantile is a plug-in threshold, so its sharpness needs a margin condition: the outcome
regression puts little mass near :math:`1 - \tau`.  Validity needs no such condition.

**What the interval may claim.**  It resolves through the registry of
:mod:`cleverly.validation._interpretation`, instrument :data:`MSM_BOUND_INSTRUMENT`.  Study
``marginal-bound-limit-coverage`` (X82 SH) registers the ``continuous`` and ``binary`` rows
for the ATE of a five-fold cross-fitted TMLE with GLM nuisances at ``n >= 8000``.  Every
other fit prints the registry's descriptive sentence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import status_record
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from ..exceptions import CapabilityError
from ..inference.cluster import influence_variance
from ..inference.delta import reference_quantile
from ..utils.frames import emit_frame

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
#: ``"continuous"`` or ``"binary"``, followed by one ``"+feature"`` per composition the
#: bound used (:func:`_limit_reading`), then ``"+repeats"`` on a fit with more than one
#: cross-fitting draw and ``"+clustered"`` on a clustered fit.
MSM_BOUND_INSTRUMENT = "msm_bound_limit"

#: The default grid of the sensitivity parameter.
DEFAULT_GAMMA_GRID = (1.0, 1.5, 2.0)

#: The fold count of the bound's own split when the fit was not cross-fitted.
_IN_SAMPLE_FOLDS = 5

#: The roadmap row of the compositions with no known calculation.
_OWNER = "X87"

#: The roadmap row of the compositions X82 part H2 adds.
_PART_H2 = "X82"

#: The fitted methods the bounds read. A selecting C-TMLE has its mechanism refitted.
_BOUNDED_METHODS = frozenset({"tmle", "drtmle", "collaborative_tmle"})

#: The statuses a collaborative fit withholds its interval under because its mechanism is
#: selected. The bounds refit the mechanism on every covariate, so their own curve does not
#: read the selection.
_SELECTION_STATUSES = frozenset({"working_mechanism_plugin", "generated_design_plugin"})


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
        The null value of the estimand: ``0.0`` for a difference, ``1.0`` for a ratio and
        ``None`` for an arm mean.
    outcome_kind : {"continuous", "binary"}
        Which nuisance route the bounds used.
    quantile_learner : tuple of str
        The quantile learner each side and arm selected, or ``()`` for a binary outcome.
    pseudo_outcome_learner : tuple of str
        The tail-regression learner each side and arm selected, or ``()``.
    limit_reading : Resolution or None
        The registry's reading of the confidence limits.
    backend : str or None
        The dataframe backend of the fit's data, which :meth:`to_frame` emits.
    scale : {"identity", "log"}
        The scale of the standard errors. A risk ratio or an odds ratio reports its ends
        and limits on the ratio scale and its standard errors on the log scale.

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
    backend: str | None = None
    scale: str = "identity"

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
        """Return one row per :math:`\\Gamma`, in the backend of the fit's data.

        Returns
        -------
        DataFrame
            The fields of :class:`MarginalSensitivityRow`.
        """
        names = [item.name for item in fields(MarginalSensitivityRow)]
        payload = {name: [getattr(row, name) for row in self.rows] for name in names}
        return emit_frame(payload, backend=self.backend)

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
    if method == "adaptive_iptw":
        from ..estimators.iptw import OPERATION_REFUSALS

        return OPERATION_REFUSALS["marginal_bounds"]
    if method not in _BOUNDED_METHODS:
        return _gap(
            f"the bounds read the outcome regression and the treatment mechanism of a TMLE, "
            f"DR-TMLE or C-TMLE fit, and a {method} fit is none of these.",
            _PART_H2,
        )
    if method == "collaborative_tmle" and _refits_mechanism(result):
        refusal = _collaborative_refusal(result)
        if refusal is not None:
            return refusal
    elif _mechanism_unfitted(result):
        return Refusal(
            RefusalReason.DIFFERENT_ESTIMAND,
            "this fit reports the natural-course mean only and fitted no treatment "
            "mechanism. The model bounds a counterfactual arm mean through the odds of "
            "treatment, and the observed-treatment mean is not one.",
            alternative="Fit the arm means ('ey') or a contrast, then call the bounds.",
        )
    if getattr(data, "intermediate", None) is not None:
        return _gap(
            "a controlled direct effect also needs a bound on confounding of the intermediate "
            "and the outcome, which the model does not state."
        )
    if data.has_missing_treatment:
        return _gap(
            "a missing treatment needs a joint model of U on the treatment and on its "
            "observation, which no read source defines."
        )
    if data.family not in ("gaussian", "binomial"):
        return Refusal(
            RefusalReason.DIFFERENT_ESTIMAND,
            f"the bounds read a continuous or a binary outcome; this fit's family is "
            f"{data.family!r}.",
            alternative=NO_ALTERNATIVE,
        )
    status = result.inference_status
    status = status() if callable(status) else status
    if status != "influence_curve" and not (
        status in _SELECTION_STATUSES and _refits_mechanism(result)
    ):
        record = status_record(status)
        return Refusal(
            RefusalReason.INFERENCE_GAP,
            "the bound's interval is built from its influence curve on a fit whose "
            "nuisances support one. " + record.reason,
            alternative=record.alternative or NO_ALTERNATIVE,
            roadmap=record.reopened_by,
        )
    return None


def _mechanism_unfitted(result: Any) -> bool:
    """Whether a draw of ``result`` kept the never-fitted treatment mechanism."""
    from ..estimators._nuisance import UnfittedPropensity

    return any(
        isinstance(repeat.nuisance.propensity, UnfittedPropensity)
        for repeat in getattr(result, "repeats", ())
    )


#: The refusal of ``att`` or ``atc`` on a fit with more than two arms.
MULTI_ARM_CONDITIONAL_REFUSAL = _gap(
    "on more than two arms the model bounds the odds of each arm against all the others. "
    "The effect on the treated also needs E[Y(b) | A = a] for one other arm a, which that "
    "model does not bound."
)


def _collaborative_refusal(result: Any) -> Refusal | None:
    """Why the mechanism of a selecting C-TMLE fit cannot be refitted, or ``None``."""
    from ..estimators.ctmle import is_per_arm_oat

    if result.estimator is None:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "the bounds refit the treatment mechanism of a selecting CTMLE fit on every "
            "covariate, which reads the fitted estimator, and this result carries none.",
            alternative=NO_ALTERNATIVE,
        )
    if is_per_arm_oat(result.estimator):
        return _gap(
            "a per-arm outcome-adaptive CTMLE declares a treatment learner that reads one "
            "column, the outcome prediction at one arm, so no learner for the mechanism on "
            "every covariate exists to refit.",
            _PART_H2,
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


def _pinball(y: FloatArray, q: FloatArray, tau: float, weights: FloatArray | None = None) -> float:
    u = y - q
    return float(np.average(np.maximum(tau * u, (tau - 1.0) * u), weights=weights))


# ---------------------------------------------------------------------------------------
# One side of one arm


@dataclass(frozen=True)
class _Side:
    """The cross-fitted summand of one arm bound, on the outcome scale.

    ``summand`` is the per-row estimating function and ``own`` its observed-arm part,
    ``1{A = a}`` times the arm's augmented outcome. ``value`` and ``curve`` are the
    unweighted marginal mean and its centred curve.
    """

    value: float
    curve: FloatArray
    summand: FloatArray
    own: FloatArray
    quantile_learner: str | None = None
    pseudo_outcome_learner: str | None = None


def _side(
    summand: FloatArray,
    own: FloatArray,
    quantile_learner: str | None = None,
    pseudo_outcome_learner: str | None = None,
) -> _Side:
    value = float(np.mean(summand))
    return _Side(value, summand - value, summand, own, quantile_learner, pseudo_outcome_learner)


def _observation(
    y: FloatArray, observed_weight: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """The outcome with unobserved rows set to zero, and ``Delta / P(Delta = 1 | A, W)``."""
    if observed_weight is None:
        return y, np.ones_like(y)
    return np.where(observed_weight > 0, y, 0.0), observed_weight


def _binary_side(
    y: FloatArray,
    treated: FloatArray,
    e: FloatArray,
    mu: FloatArray,
    gamma: float,
    upper: bool,
    observed_weight: FloatArray | None = None,
) -> _Side:
    """The binary-outcome side, with both nuisances closed forms of ``mu``.

    ``observed_weight`` is ``Delta / P(Delta = 1 | A, W)`` on a fit with missing outcomes.
    """
    y, obs = _observation(y, observed_weight)
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
    own = treated * (mu + obs * (y - mu))
    summand = own + (1.0 - treated) * rho + treated * obs * (1.0 - e) / e * (r - rho)
    return _side(summand, own)


def _fit(learner: Any, x: FloatArray, y: FloatArray, weights: FloatArray | None) -> Any:
    from ..learners._fitting import fit_learner

    return fit_learner(learner, x, y, sample_weight=weights)


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
    *,
    weights: FloatArray | None = None,
    observed_weight: FloatArray | None = None,
) -> _Side:
    """The continuous-outcome side: the upper bound of ``sign * Y``, read back by sign.

    ``weights`` are the fit's sampling weights, which every regression and every held-out
    loss reads. ``observed_weight`` is ``Delta / P(Delta = 1 | A, W)`` on a fit with missing
    outcomes: the quantile and the tail regression train on observed rows only.
    """
    y, obs = _observation(y, observed_weight)
    sign = 1.0 if upper else -1.0
    y_s, mu_s = sign * y, sign * mu
    own_s = treated * (mu_s + obs * (y_s - mu_s))
    if gamma == 1.0:
        # phi_1(u) = u, so R = Y and rho = mu: the AIPW mean, with no learner to fit.
        summand = own_s + (1.0 - treated) * mu_s + treated * obs * (1.0 - e) / e * (y_s - mu_s)
        return _side(sign * summand, sign * own_s)
    tau = gamma / (1.0 + gamma)
    rows = (treated > 0.5) & (obs > 0)
    n = y.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    def tail(q: FloatArray, target: FloatArray) -> FloatArray:
        return q + np.maximum(target - q, 0.0) / (1.0 - tau)

    def train_weights(mask: FloatArray) -> FloatArray | None:
        return None if weights is None else w[mask]

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
            train = rows & first
            model = _fit(learner, x[train], y_s[train], train_weights(train))
            held_out[test] = model.predict(x[test])
            on_second.append(np.asarray(model.predict(x[rows & second]), dtype=float))
        risk = _pinball(y_s[rows], held_out[rows], tau, w[rows])
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
            model = _fit(learner, x[train], target, train_weights(train))
            predictions[test] = model.predict(x[test])
        # The held-out loss reads each row's pseudo-outcome from the quantile fitted
        # without its fold, as the evaluation does.
        loss = (tail(q_hat[rows], y_s[rows]) - predictions[rows]) ** 2
        risk = float(np.average(loss, weights=w[rows]))
        if risk < best_risk:
            c_name, c_hat, best_risk = name, predictions, risk
    rho = mu_s / gamma + (1.0 - 1.0 / gamma) * c_hat
    r = y_s / gamma + (1.0 - 1.0 / gamma) * tail(q_hat, y_s)
    summand = own_s + (1.0 - treated) * rho + treated * obs * (1.0 - e) / e * (r - rho)
    return _side(sign * summand, sign * own_s, q_name, c_name)


# ---------------------------------------------------------------------------------------
# The bounded parameter


#: The estimand stems, and the scale each reports on.
_LINEAR_STEMS = ("ey", "ate", "att", "atc")
_RATIO_STEMS = ("rr", "or")


@dataclass(frozen=True)
class _Target:
    """One bounded parameter: its stem, its arms and the stratum it conditions on.

    ``arm`` is the arm a level reports and the non-reference arm of a contrast; ``versus``
    is the reference arm, or ``None`` for a level. ``stratum`` is a code into
    ``data.strata_levels``, or ``None`` for a marginal parameter.
    """

    name: str
    stem: str
    arm: float
    versus: float | None = None
    stratum: int | None = None

    @property
    def ratio(self) -> bool:
        return self.stem in _RATIO_STEMS

    @property
    def null(self) -> float | None:
        if self.versus is None:
            return None
        return 1.0 if self.ratio else 0.0


def _named_targets(result: TMLEResult) -> dict[str, _Target]:
    """Every marginal parameter the bounds can take, by its reported name."""
    from ..targets.base import arm_alias, parameter_name

    data = result.data
    reference = float(result.config.reference_arm)
    label = data.arm_label
    binary = bool(data.is_binary_treatment)
    out: dict[str, _Target] = {}
    for arm in data.arm_codes:
        name = parameter_name("ey", arm=label(arm))
        out[name] = _Target(name, "ey", float(arm))
    if binary:
        out["ey1"] = _Target("ey1", "ey", 1.0)
        out["ey0"] = _Target("ey0", "ey", 0.0)
    for arm in data.arm_codes:
        if float(arm) == reference:
            continue
        for stem in ("ate", "att", "atc", *_RATIO_STEMS):
            name = arm_alias(stem, arm=label(arm), versus=label(reference), collapse=binary)
            out[name] = _Target(name, stem, float(arm), reference)
    # A fit with structured parameter keys may report aliases of its own.
    from ._parameters import arm_parameters

    for name, parameter in arm_parameters(result).items():
        stem = "ey" if parameter.stem in ("ey1", "ey0") else parameter.stem
        out.setdefault(name, _Target(name, stem, parameter.arm, parameter.versus))
    if data.strata is not None:
        # The conditional report suffixes the marginal alias with the stratum's label.
        from ..targets.base import stratum_alias

        for code in range(data.n_strata):
            label_s = data.stratum_label(code)
            for marginal in list(out.values()):
                if marginal.stratum is None:
                    name = stratum_alias(marginal.name, label_s)
                    out[name] = replace(marginal, name=name, stratum=code)
    return out


def _resolve_target(result: TMLEResult, estimand: str | None) -> _Target:
    """The parameter an ``estimand`` name asks for, with ``None`` read as the default."""
    from ._parameters import stratum_parameter

    data = result.data
    named = _named_targets(result)
    if estimand is None:
        if data.is_binary_treatment:
            return named["ate"]
        # A fit with more than two arms has no single ATE: take its first reported
        # marginal contrast or mean, else the first contrast against the reference.
        for name in result.estimates:
            if name in named and not named[name].ratio:
                return named[name]
        return next(target for target in named.values() if target.stem == "ate")
    if estimand in named:
        target = named[estimand]
    else:
        conditional = stratum_parameter(result, estimand)
        if conditional is None:
            raise ValueError(
                f"estimand must name an arm mean, a contrast, att, atc, rr or or of this fit "
                f"({', '.join(sorted(named))}), or a reported stratum-conditional alias; "
                f"got {estimand!r}"
            )
        stem = "ey" if conditional.stem in ("ey1", "ey0") else conditional.stem
        target = _Target(estimand, stem, conditional.arm, conditional.versus, conditional.stratum)
    if target.stem in ("att", "atc") and not data.is_binary_treatment:
        raise CapabilityError.from_refusal(MULTI_ARM_CONDITIONAL_REFUSAL)
    if target.ratio:
        observed = np.asarray(data.outcome, dtype=float)[np.asarray(data.observed, dtype=bool)]
        if data.family != "binomial" and target.stem == "or":
            raise CapabilityError(
                "the odds ratio bounds read a binary outcome; this fit's family is "
                f"{data.family!r}.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Ask for the risk ratio ('rr') of a nonnegative outcome.",
            )
        if np.any(observed < 0):
            raise CapabilityError(
                "the risk ratio bounds take the ratio of two arm bounds, which needs a "
                "nonnegative outcome; this outcome takes negative values.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Bound the difference ('ate') instead.",
            )
    return target


def _gammas(gamma: float | Sequence[float]) -> tuple[float, ...]:
    values = (gamma,) if isinstance(gamma, (int, float)) else tuple(gamma)
    if not values:
        raise ValueError("gamma needs at least one value")
    out = tuple(sorted({float(value) for value in values}))
    if any(not np.isfinite(value) or value < 1.0 for value in out):
        raise ValueError(f"every gamma must be a finite number of at least 1; got {values!r}")
    return out


def _refits_mechanism(result: TMLEResult) -> bool:
    """Whether the bounds refit the treatment mechanism instead of reading the fit's.

    A selecting C-TMLE conditions its mechanism on a selected function of the
    covariates, which does not estimate ``P(A | W)``. A derivative C-TMLE (C-TMLE0)
    selects nothing, so its mechanism is the ordinary one.
    """
    if getattr(result, "fitted_method", None) != "collaborative_tmle":
        return False
    from ..estimators.ctmle import CTMLEDerivativeFit

    return not isinstance(result.extra.get("ctmle"), CTMLEDerivativeFit)


def _mechanism(result: TMLEResult, repeat: Any) -> Any:
    """The treatment mechanism of one draw: the fit's, or one refit on every covariate."""
    if not _refits_mechanism(result):
        return repeat.nuisance.propensity
    from .omitted_variable import _full_mechanism

    return _full_mechanism(result, repeat)


def _draw_sides(
    result: TMLEResult,
    repeat: Any,
    gamma: float,
    wanted: Sequence[tuple[float, bool]],
    quantile_learner: QuantileFactory | None,
    pseudo_outcome_learner: Any,
) -> dict[tuple[float, bool], _Side]:
    """Every ``(arm, upper)`` side in ``wanted``, on one cross-fitting draw."""
    from ..estimators._nuisance import tilt_missingness

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
    propensity = _mechanism(result, repeat)
    level_index = {float(level): i for i, level in enumerate(propensity.arms)}
    g = propensity.bounded(result.config.g_bounds)
    y = np.asarray(data.outcome, dtype=float)
    x = np.asarray(data.covariates, dtype=float)
    weights = np.asarray(data.weights, dtype=float) if data.weights_name is not None else None
    response = None
    if data.has_missing_outcome:
        response = tilt_missingness(nuisance, result.config.missingness_bound)
    observed = np.asarray(data.observed, dtype=bool)
    binary = data.family == "binomial"
    treatment = np.asarray(data.treatment, dtype=float)
    sides: dict[tuple[float, bool], _Side] = {}
    for level, upper in wanted:
        treated = (treatment == level).astype(float)
        e = g[:, level_index[level]]
        mu = nuisance.scaler.unscale_levels(np.asarray(nuisance.outcome.arms[level], dtype=float))
        observed_weight = None
        if response is not None:
            column = list(nuisance.arms).index(level)
            observed_weight = np.where(observed, 1.0 / response[:, column], 0.0)
        if binary:
            sides[(level, upper)] = _binary_side(y, treated, e, mu, gamma, upper, observed_weight)
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
                weights=weights,
                observed_weight=observed_weight,
            )
    return sides


def _wanted(target: _Target) -> list[tuple[float, bool]]:
    """The sides the target reads: both sides of its arm and, for a contrast, of ``versus``."""
    wanted = [(target.arm, True), (target.arm, False)]
    if target.versus is not None:
        wanted += [(target.versus, True), (target.versus, False)]
    return wanted


@dataclass(frozen=True)
class _End:
    """One end of the bound on one draw: its value and influence curve, on the bound's scale."""

    value: float
    curve: FloatArray


def _mean(numerator: FloatArray, denominator: FloatArray, mass: FloatArray) -> _End:
    """``E[mass numerator] / E[mass denominator]`` and its influence curve.

    ``mass`` is the sampling weight times the stratum indicator. The denominator is one
    for a marginal mean and the indicator of the conditioning arm for ``att`` and ``atc``.
    """
    share = float(np.mean(mass * denominator))
    value = float(np.mean(mass * numerator)) / share
    return _End(value, mass * (numerator - value * denominator) / share)


def _log_end(end: _End, odds: bool) -> _End:
    """The end on the log scale (of the mean, or of its odds)."""
    value = end.value
    if not value > 0.0 or (odds and not value < 1.0):
        raise CapabilityError(
            f"an arm bound of {value:.4g} has no log{' odds' if odds else ''} at this gamma, so "
            "the ratio of the arm bounds is undefined there.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Ask for a smaller gamma, or bound the difference ('ate').",
        )
    scale = value * (1.0 - value) if odds else value
    transformed = np.log(value / (1.0 - value)) if odds else np.log(value)
    return _End(float(transformed), end.curve / scale)


def _target_ends(
    result: TMLEResult, target: _Target, sides: dict[tuple[float, bool], _Side]
) -> tuple[_End, _End]:
    """``(lower, upper)`` of the target on one draw, a ratio on the log scale."""
    data = result.data
    n = data.n
    mass = np.asarray(data.weights, dtype=float).copy()
    if target.stratum is not None:
        mass = mass * (np.asarray(data.strata) == target.stratum)
    ones = np.ones(n)
    arm_up, arm_low = sides[(target.arm, True)], sides[(target.arm, False)]
    if target.versus is None:
        return _mean(arm_low.summand, ones, mass), _mean(arm_up.summand, ones, mass)
    ref_up, ref_low = sides[(target.versus, True)], sides[(target.versus, False)]
    if target.stem == "ate":
        return (
            _mean(arm_low.summand - ref_up.summand, ones, mass),
            _mean(arm_up.summand - ref_low.summand, ones, mass),
        )
    treatment = np.asarray(data.treatment, dtype=float)
    if target.stem == "att":
        # E[Y(a) | A = a] is identified; E[Y(ref) | A = a] reads the reference arm's bound
        # without its observed-arm part.
        given = (treatment == target.arm).astype(float)
        own = arm_up.own
        return (
            _mean(own - (ref_up.summand - ref_up.own), given, mass),
            _mean(own - (ref_low.summand - ref_low.own), given, mass),
        )
    if target.stem == "atc":
        given = (treatment == target.versus).astype(float)
        own = ref_up.own
        return (
            _mean((arm_low.summand - arm_low.own) - own, given, mass),
            _mean((arm_up.summand - arm_up.own) - own, given, mass),
        )
    # A ratio of two arm bounds is monotone in each: the upper end divides the arm's upper
    # bound by the reference's lower bound.
    odds = target.stem == "or"
    arms = {key: _log_end(_mean(side.summand, ones, mass), odds) for key, side in sides.items()}
    lower = _End(
        arms[(target.arm, False)].value - arms[(target.versus, True)].value,
        arms[(target.arm, False)].curve - arms[(target.versus, True)].curve,
    )
    upper = _End(
        arms[(target.arm, True)].value - arms[(target.versus, False)].value,
        arms[(target.arm, True)].curve - arms[(target.versus, False)].curve,
    )
    return lower, upper


def _median_rule(values: FloatArray, variances: FloatArray) -> tuple[float, float]:
    """The repeat median and its variance (Chernozhukov et al. 2018, eq. 3.14)."""
    centre = float(np.median(values))
    return centre, float(np.median(variances + (values - centre) ** 2))


def _limit_reading(
    result: TMLEResult, outcome_kind: str, target: _Target, custom_learners: bool
) -> Resolution:
    from ..validation._interpretation import fit_configuration, resolve

    # Study SH covers the default quantile and tail learners and the marginal ATE of an
    # unweighted, fully observed two-arm fit with one draw. Each other feature names a rule
    # of its own, which no row covers.
    data = result.data
    features = [outcome_kind]
    if outcome_kind == "continuous" and custom_learners:
        features.append("custom_learners")
    if not data.is_binary_treatment:
        features.append("multi_arm")
    if target.stem in ("att", "atc"):
        features.append("conditional")
    if target.ratio:
        features.append("ratio")
    if target.stratum is not None:
        features.append("stratum")
    if data.weights_name is not None:
        features.append("weighted")
    if data.has_missing_outcome:
        features.append("missing_outcome")
    if _refits_mechanism(result):
        features.append("refit_mechanism")
    if len(getattr(result, "repeats", ())) > 1:
        # Study SH measured one cross-fitting draw; the median rule over draws is unmeasured.
        features.append("repeats")
    if data.cluster is not None:
        features.append("clustered")
    rule = "+".join(features)
    try:
        fit = fit_configuration(result, target.name)
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
        A fitted ``TMLE``, ``DRTMLE`` or ``CTMLE`` result with a Gaussian or binary outcome.
        It may have more than two arms, sampling weights, baseline strata or missing
        outcomes.
    gamma : float or sequence of float, default=DEFAULT_GAMMA_GRID
        The sensitivity parameter :math:`\\Gamma \\ge 1` (Tan's :math:`\\Lambda`), or a grid.
        :math:`\\Gamma = 1` reproduces the cross-fitted AIPW estimate. On a selecting
        ``CTMLE`` fit that estimate reads the refitted mechanism, so it is not the C-TMLE
        estimate.
    estimand : str or None, default=None
        The reported name of an arm mean (``"ey1"``), a difference (``"ate"``), an effect on
        the treated or the controls (``"att"``, ``"atc"``, two arms only), a ratio
        (``"rr"``, ``"or"``), any of these against a named reference arm, or a
        stratum-conditional alias such as ``"ate[V=1]"``. The fit need not have reported
        it. ``None`` bounds the ATE of a two-arm fit, and the first reported contrast or
        mean of a fit with more arms.
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
        When :func:`marginal_bound_refusal` refuses the fit, when ``att`` or ``atc`` is
        asked of a fit with more than two arms, or when a ratio's outcome scale does not
        fit it or an arm bound has no logarithm.
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
    target = _resolve_target(result, estimand)
    cluster = result.data.cluster
    z = reference_quantile(1.0 - level)
    outcome_kind = "binary" if result.data.family == "binomial" else "continuous"
    rows: list[MarginalSensitivityRow] = []
    quantile_names: set[str] = set()
    pseudo_names: set[str] = set()
    for value in grid:
        lowers, uppers, lower_vars, upper_vars = [], [], [], []
        for repeat in result.repeats:
            sides = _draw_sides(
                result, repeat, value, _wanted(target), quantile_learner, pseudo_outcome_learner
            )
            for side in sides.values():
                if side.quantile_learner is not None:
                    quantile_names.add(side.quantile_learner)
                if side.pseudo_outcome_learner is not None:
                    pseudo_names.add(side.pseudo_outcome_learner)
            try:
                low, up = _target_ends(result, target, sides)
            except CapabilityError as error:
                if not target.ratio:
                    raise
                defined = rows[-1].gamma if rows else None
                reach = (
                    f"The ratio is defined up to gamma={defined:g}; pass a grid that stops there."
                    if defined is not None
                    else "No gamma of the grid gives a defined ratio."
                )
                raise CapabilityError(
                    f"at gamma={value:g}, {error.refusal.condition} {reach}",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=error.refusal.alternative,
                ) from error
            lowers.append(low.value)
            uppers.append(up.value)
            lower_vars.append(influence_variance(low.curve, cluster))
            upper_vars.append(influence_variance(up.curve, cluster))
        lower, lower_var = _median_rule(np.asarray(lowers), np.asarray(lower_vars))
        upper, upper_var = _median_rule(np.asarray(uppers), np.asarray(upper_vars))
        lower_se, upper_se = float(np.sqrt(lower_var)), float(np.sqrt(upper_var))
        lower_limit, upper_limit = lower - z * lower_se, upper + z * upper_se
        if target.ratio:
            lower, upper = float(np.exp(lower)), float(np.exp(upper))
            lower_limit, upper_limit = float(np.exp(lower_limit)), float(np.exp(upper_limit))
        rows.append(
            MarginalSensitivityRow(
                gamma=value,
                estimand=target.name,
                lower=lower,
                upper=upper,
                lower_std_error=lower_se,
                upper_std_error=upper_se,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
        )
    return MarginalSensitivityBounds(
        rows=tuple(rows),
        estimand=target.name,
        level=level,
        null=target.null,
        outcome_kind=outcome_kind,
        quantile_learner=tuple(sorted(quantile_names)),
        pseudo_outcome_learner=tuple(sorted(pseudo_names)),
        limit_reading=_limit_reading(
            result,
            outcome_kind,
            target,
            custom_learners=quantile_learner is not None or pseudo_outcome_learner is not None,
        ),
        backend=result.data.backend,
        scale="log" if target.ratio else "identity",
    )
