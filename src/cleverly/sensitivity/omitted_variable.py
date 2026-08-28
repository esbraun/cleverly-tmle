r"""Omitted-variable-bias sensitivity analysis.

Positivity diagnostics tell you whether the data can support the estimate.  This
module answers a different question: *how strong would an unmeasured confounder
have to be to overturn the conclusion?*

Following Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), the bias from
omitting a confounder is bounded by a product of three interpretable pieces:

.. math::

    |\mathrm{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1 - c_D^2}}\; c_Y\;
                       \underbrace{\sqrt{\sigma^2 \nu^2}}_{\text{max bias}}

where, writing :math:`\alpha` for the Riesz representer of the target parameter,

* :math:`\sigma^2 = E[(Y - \bar Q(A, W))^2]` -- residual outcome variance,
* :math:`\nu^2 = E[\alpha(A, W)^2]` -- how hard the estimand has to work to
  extrapolate (it is large exactly when overlap is poor),
* ``cf_y`` -- the share of *residual* outcome variance the confounder would explain,
* ``cf_d`` -- the corresponding gain in the Riesz representer, i.e. how much the
  confounder would improve prediction of treatment,
* ``rho`` -- how adversarially aligned those two are; ``rho = 1`` is the worst case.

Two things make this useful rather than merely formal.  First, the *robustness
value* :func:`robustness_value` inverts the bound: it reports the single number
``cf_y = cf_d = RV`` at which the conclusion would flip, so there is no need to
guess sensitivity parameters at all.  Second, :func:`benchmark` calibrates
``cf_y``/``cf_d`` against covariates you *did* measure -- "a confounder as strong as
age" is a claim a reader can evaluate, where "cf_y = 0.03" is not.

The parameterisation, including the definition of the benchmark gain statistics,
matches DoubleML's ``sensitivity_analysis`` so numbers are comparable across the two
libraries.

Scope: the bound applies to the linear functionals this library estimates -- the
counterfactual means, their contrasts against the reference arm, and the two conditional
effects.  With more than two arms those are named for the arms they are about, so the
estimand to ask for is ``"ate[medium vs low]"`` rather than ``"ate"``; it is one bound
per contrast, because :math:`\nu^2` is the second moment of *that contrast's* Riesz
representer.  Ratios are not linear functionals of the outcome regression, so use
:mod:`cleverly.sensitivity.evalue` for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

from .._typing import FloatArray
from ..estimators.targeting import build_submodel
from ..exceptions import CapabilityError, refuse_after_repeats
from ..inference.cluster import influence_variance
from ..targets import parameter_stem
from ..utils.bounds import g_bounds_for
from ..utils.text import format_table
from ._parameters import ArmParameter, arm_parameters, stratum_refusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators._nuisance import RepeatFit
    from ..estimators.base import TMLEResult

__all__ = [
    "BenchmarkResult",
    "SensitivityBounds",
    "SensitivityElements",
    "benchmark",
    "omitted_variable_bounds",
    "robustness_value",
    "sensitivity_elements",
]

#: Targets for which the Riesz representer -- and therefore this bound -- is defined.
#:
#: Stems rather than reported names, since a multi-valued treatment names each parameter
#: for the arms it is about: ``ate[medium vs low]`` has a stem of ``ate`` and a
#: representer of its own.  :func:`~cleverly.sensitivity._parameters.arm_parameters` is
#: what turns a fit's arms into the names these stems produce.
LINEAR_ESTIMANDS: frozenset[str] = frozenset({"ate", "ey", "ey1", "ey0", "att", "atc"})


@dataclass(frozen=True)
class SensitivityElements:
    r"""The ingredients of the bias bound for one estimand.

    Parameters
    ----------
    estimand : str
        Alias of the estimand these elements describe.
    sigma2 : float
        :math:`E[(Y - \bar Q^*(A, W))^2]`, the residual outcome variance.
    nu2 : float
        :math:`E[\alpha(A, W)^2]`, the second moment of the Riesz representer.
    max_bias : float
        :math:`\sqrt{\sigma^2 \nu^2}` -- the largest bias any confounder could produce
        if it explained *all* the residual variation on both sides.
    psi_sigma2 : ndarray
        Influence curve of ``sigma2``.
    psi_nu2 : ndarray
        Influence curve of ``nu2``.
    psi_max_bias : ndarray
        Influence curve of ``max_bias``, so the bias-adjusted bounds get confidence
        intervals rather than being treated as known constants.

    riesz_representer : ndarray
        ``(n,)`` values of :math:`\\alpha(A, W)` for the targeted functional.
    nu2_estimator : str
        Which estimator of ``nu2`` produced these values.
    """

    estimand: str
    sigma2: float
    nu2: float
    max_bias: float
    psi_sigma2: FloatArray
    psi_nu2: FloatArray
    psi_max_bias: FloatArray
    riesz_representer: FloatArray
    nu2_estimator: str


def sensitivity_elements(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    nu2_estimator: str = "auto",
) -> SensitivityElements:
    r"""Compute :math:`\sigma^2`, :math:`\nu^2` and the maximal bias for one estimand.

    Parameters
    ----------
    nu2_estimator:
        ``"doubly_robust"`` uses :math:`E[2 m(W, \alpha) - \alpha^2]`, which is less
        sensitive to error in the estimated propensity than the plug-in
        :math:`E[\alpha^2]` (``"plugin"``).  Both are consistent; ``"auto"`` picks the
        doubly robust form wherever the functional's :math:`m(W, \alpha)` has a closed
        form, which is all of :data:`LINEAR_ESTIMANDS`.
    """
    refuse_after_repeats(
        result.n_repeats,
        operation="omitted-variable sensitivity",
        reason=(
            "A coordinatewise median of the bound's influence terms would not be the "
            "influence function of the median bound. Fit one split for this analysis."
        ),
    )
    parameter = resolve_parameter(result, estimand)
    return _elements_for(result, result.repeats[0], parameter, nu2_estimator)


def resolve_parameter(result: TMLEResult, estimand: str) -> ArmParameter:
    """The arms a requested estimand is about, or a refusal that says why not.

    One bound is one linear functional, so this is where "which contrast" is decided --
    ``ate`` on a two-armed fit and ``ate[medium vs low]`` on a wider one, each with its
    own Riesz representer.  The order of the checks matters: a name this bound could
    never apply to is refused for *that* reason, before the fit is consulted at all, so
    that asking for a risk ratio is not reported as a missing estimand.
    """
    if parameter_stem(estimand) not in LINEAR_ESTIMANDS:
        raise CapabilityError(
            f"the omitted-variable bound applies to {sorted(LINEAR_ESTIMANDS)}, not "
            f"{estimand!r}. For a risk ratio or odds ratio use sensitivity.evalue()."
        )
    known = arm_parameters(result)
    available = {name: parameter for name, parameter in known.items() if name in result.estimates}
    if estimand in available:
        return available[estimand]
    # Before the coverage message: this one *was* reported, so "not requested" would be
    # false. It is the derivation that is missing, not the parameter.
    conditional = stratum_refusal(result, estimand, "the omitted-variable bound")
    if conditional is not None:
        raise CapabilityError(conditional)
    if not available:
        raise CapabilityError(
            "the omitted-variable bound applies to the arm-indexed linear estimands, and "
            f"this fit reports none: its parameters are indexed by "
            f"{result.config.parameter_axis!r} and it reported {sorted(result.estimates)}. "
            "The bias is bounded through the Riesz representer of a mean or a contrast of "
            "arms, which a fit whose counterfactuals are not arms does not have."
        )
    raise CapabilityError(
        f"estimand {estimand!r} was not requested in this fit. The bound is available for "
        f"{sorted(available)} -- one per contrast, since nu^2 is the second moment of "
        "that contrast's own Riesz representer."
    )


def _elements_for(
    result: TMLEResult,
    repeat: RepeatFit,
    parameter: ArmParameter,
    nu2_estimator: str,
) -> SensitivityElements:
    """The bound's pieces under one cross-fitting draw.

    Takes the targeted ``Qbar`` and the mechanism from the same
    :class:`~cleverly.estimators._nuisance.RepeatFit`, which is what makes ``sigma2`` the
    residual variance of the regression whose propensity ``nu2`` was computed from.
    """
    data = result.data
    scaler = repeat.nuisance.scaler
    group = parameter.group
    fluctuation = repeat.fluctuations[group]
    bounds = g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)
    reference = result.config.reference_arm
    submodel = build_submodel(
        data,
        repeat.nuisance,
        group,
        bounds=bounds,
        nuisance_bound=result.config.missingness_bound,
        intermediate_value=result.intermediate_value,
        # The conditional-effect fluctuations contrast against the arm this fit declared,
        # so the covariate rebuilt here must be the one it was targeted with.
        reference=reference,
    )
    # The margin of the arm the estimand *conditions on*: ``_m_alpha`` weights the
    # contrast by the density ratio dP(W | A = c) / dP(W), which is g_c / P(A = c).
    # ``None`` for a mean or an unconditional contrast, which reweight nobody -- and the
    # arm is read off the parameter rather than assumed to be the other one, since with
    # K arms ``att[medium vs low]`` and ``att[high vs low]`` condition on different
    # populations.
    arms = repeat.nuisance.arms
    conditioning = parameter.conditions_on
    propensity: FloatArray | None = None
    conditioning_share: float | None = None
    if conditioning is not None:
        index = arms.index(conditioning)
        propensity = repeat.nuisance.bounded_propensity(bounds)[:, index]
        conditioning_share = float(data.arm_fractions[index])

    # sigma^2: residual variance of the targeted outcome regression, on the original
    # outcome scale so the bound is reported in the units the estimate uses.
    scaled = scaler.scale(data.outcome)
    residual = np.where(data.observed, scaled - fluctuation.targeted.observed, 0.0)
    if not scaler.is_identity:
        residual = residual * scaler.range
    weights = data.weights
    sigma2_element = residual**2
    sigma2 = float(np.average(sigma2_element[data.observed], weights=weights[data.observed]))
    psi_sigma2 = np.where(data.observed, sigma2_element - sigma2, 0.0) * weights

    representer = _riesz_representer(parameter, submodel)
    plugin = float(np.average(representer**2, weights=weights))

    method = nu2_estimator
    if method == "auto":
        method = "doubly_robust"
    if method == "doubly_robust":
        m_alpha = _m_alpha(parameter, submodel, propensity, conditioning_share)
        nu2_element = 2.0 * m_alpha - representer**2
        nu2 = float(np.average(nu2_element, weights=weights))
        if nu2 <= 0:  # pragma: no cover - only with a pathological propensity fit
            nu2, nu2_element, method = plugin, representer**2, "plugin"
    elif method == "plugin":
        nu2_element = representer**2
        nu2 = plugin
    else:
        raise ValueError(
            f"nu2_estimator must be 'auto', 'doubly_robust' or 'plugin'; got {nu2_estimator!r}"
        )
    psi_nu2 = (nu2_element - nu2) * weights

    max_bias = float(np.sqrt(sigma2 * nu2))
    psi_max_bias = (sigma2 * psi_nu2 + nu2 * psi_sigma2) / (2.0 * max_bias)
    return SensitivityElements(
        estimand=parameter.name,
        sigma2=sigma2,
        nu2=nu2,
        max_bias=max_bias,
        psi_sigma2=psi_sigma2,
        psi_nu2=psi_nu2,
        psi_max_bias=psi_max_bias,
        riesz_representer=representer,
        nu2_estimator=method,
    )


def _riesz_representer(parameter: ArmParameter, submodel: Any) -> FloatArray:
    r"""``alpha(A, W)`` for the requested estimand.

    The clever covariate *is* the Riesz representer -- that is why the same object
    both drives the targeting step and controls how much an omitted confounder can
    move the estimate.  Poor overlap inflates both simultaneously.

    Every column is reached by the arm it carries rather than by position, so
    ``ate[high vs low]`` reads the two arms it names out of a ``K``-column covariate
    instead of the two an implementation that counted to two would have found.
    """
    if parameter.group in ("att", "atc"):
        # One column per non-reference arm, and this parameter's is the one carrying the
        # contrast it is named for.
        return submodel.contrast_column_for(parameter.arm)
    if parameter.versus is None:
        return submodel.column_for(parameter.arm)
    return np.asarray(
        submodel.column_for(parameter.arm) - submodel.column_for(parameter.versus), dtype=float
    )


def _m_alpha(
    parameter: ArmParameter,
    submodel: Any,
    propensity: FloatArray | None,
    conditioning_share: float | None,
) -> FloatArray:
    r"""The target functional applied to the Riesz representer, ``m(W, alpha)``.

    Used by the doubly robust estimator of :math:`\nu^2`, which relies on the Riesz
    identity :math:`E[m(W, \alpha)] = E[\alpha^2]`.

    ``propensity`` and ``conditioning_share`` belong to the arm the estimand conditions
    on -- the contrast arm for an ATT and the reference for an ATC -- rather than to arm
    1 and its complement.  With two arms and the default reference those are the same two
    numbers; naming the arm is what keeps them right when a binary fit declares the other
    one, and what gives each of ``K - 1`` contrasts its own population.  They are
    ``None`` for a mean or an unconditional contrast, which reweight nobody.
    """
    # ``arms[a][:, c]`` is the covariate at arm ``a`` in the column targeting arm ``c``:
    # the mean submodel has one column per arm, so both indices are arm levels and
    # neither is a positional 0 or 1.
    if parameter.group == "mean":
        columns = submodel.arm_columns
        at = parameter.arm
        if parameter.versus is None:
            return np.asarray(submodel.arms[at][:, columns[at]], dtype=float)
        versus = parameter.versus
        first, second = submodel.arms[at], submodel.arms[versus]
        return np.asarray(
            first[:, columns[at]]
            - first[:, columns[versus]]
            - (second[:, columns[at]] - second[:, columns[versus]]),
            dtype=float,
        )

    # ATT / ATC: the functional carries an arm-membership weight, since it averages the
    # contrast over the conditioning arm's subpopulation rather than over everyone. The
    # column is read by the arm whose contrast it carries, not as a literal 0.
    assert parameter.versus is not None
    assert propensity is not None and conditioning_share is not None
    column = submodel.contrast_columns[parameter.arm]
    difference = np.asarray(
        submodel.arms[parameter.arm][:, column] - submodel.arms[parameter.versus][:, column],
        dtype=float,
    )
    return np.asarray((propensity / conditioning_share) * difference, dtype=float)


@dataclass(frozen=True)
class SensitivityBounds:
    """Bias-adjusted bounds under an assumed confounder strength.

    Parameters
    ----------
    estimand : str
        Alias of the estimand these bounds describe.
    psi : float
        The unadjusted point estimate.
    cf_y : float
        Share of the residual outcome variation the assumed confounder explains.
    cf_d : float
        Share of the residual treatment variation the assumed confounder explains.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.
    confounding_strength : float
        The product those three imply.
    max_bias : float
        Largest bias a confounder of that strength could produce.
    lower : float
        Bias-adjusted lower bound on the estimate.
    upper : float
        Bias-adjusted upper bound on the estimate.
    ci_lower : float
        Lower confidence limit of the adjusted bound.
    ci_upper : float
        Upper confidence limit of the adjusted bound.
    level : float
        Coverage level of those limits.
    robustness_value : float
        Confounding strength that would move the point estimate to the null.
    robustness_value_ci : float
        The same strength for the confidence limit rather than the point estimate.
    null_hypothesis : float
        The value the robustness values are measured against.
    """

    estimand: str
    psi: float
    cf_y: float
    cf_d: float
    rho: float
    confounding_strength: float
    max_bias: float
    lower: float
    upper: float
    ci_lower: float
    ci_upper: float
    level: float
    robustness_value: float
    robustness_value_ci: float
    null_hypothesis: float

    @property
    def bias(self) -> float:
        """Magnitude of the bias the assumed confounder could produce."""
        return self.confounding_strength * self.max_bias

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        return {
            "estimand": self.estimand,
            "psi": self.psi,
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "rho": self.rho,
            "max_bias": self.max_bias,
            "bias": self.bias,
            "lower": self.lower,
            "upper": self.upper,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "robustness_value": self.robustness_value,
            "robustness_value_ci": self.robustness_value_ci,
        }

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        conclusion = (
            "the sign of the effect survives"
            if (self.lower - self.null_hypothesis) * (self.upper - self.null_hypothesis) > 0
            else "the effect could be explained away"
        )
        return "\n".join(
            [
                f"Omitted-variable sensitivity for {self.estimand!r}",
                "-" * 44,
                f"estimate {self.psi:.5g}; maximal bias sqrt(sigma^2 nu^2) = {self.max_bias:.5g}",
                f"assumed confounding: cf_y = {self.cf_y:.3g}, cf_d = {self.cf_d:.3g}, "
                f"rho = {self.rho:.3g}"
                f" -> bias <= {self.bias:.5g}",
                f"bias-adjusted bounds:  [{self.lower:.5g}, {self.upper:.5g}]",
                f"with {self.level:.0%} one-sided CIs: [{self.ci_lower:.5g}, {self.ci_upper:.5g}] "
                f"({conclusion})",
                "",
                f"robustness value RV   = {self.robustness_value:.4f}: a confounder explaining "
                f"{self.robustness_value:.1%} of the residual variation in BOTH the outcome and "
                f"treatment would move the estimate to {self.null_hypothesis:g}.",
                f"robustness value RVa  = {self.robustness_value_ci:.4f}: the same, for the "
                f"{self.level:.0%} confidence bound.",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _confounding_strength(cf_y: float, cf_d: float, rho: float) -> float:
    """``|rho| sqrt(cf_y cf_d / (1 - cf_d))``, the multiplier on the maximal bias."""
    for name, value in (("cf_y", cf_y), ("cf_d", cf_d)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1); got {value}")
    if not 0.0 <= abs(rho) <= 1.0:
        raise ValueError(f"|rho| must lie in [0, 1]; got {rho}")
    return float(abs(rho) * np.sqrt(cf_y * cf_d / (1.0 - cf_d)))


def omitted_variable_bounds(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    cf_y: float = 0.03,
    cf_d: float = 0.03,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
) -> SensitivityBounds:
    """Bias-adjusted bounds and robustness values for one estimand.

    Defaults follow the convention of assuming a confounder that explains 3% of the
    residual variation on each side, with worst-case alignment (``rho = 1``).  Prefer
    reading :attr:`SensitivityBounds.robustness_value`, which needs no assumption at
    all.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to bound.
    cf_y : float
        Assumed share of residual outcome variation the confounder explains.
    cf_d : float
        Assumed share of residual treatment variation it explains.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.
    level : float
        Coverage level of the reported limits.
    null_hypothesis : float
        Value the robustness values are measured against.
    nu2_estimator : {"auto", "analytic", "riesz"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    SensitivityBounds
        Adjusted bounds, their confidence limits, and the robustness values.
    """
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    estimate = result[estimand]
    strength = _confounding_strength(cf_y, cf_d, rho)

    lower = estimate.psi - strength * elements.max_bias
    upper = estimate.psi + strength * elements.max_bias

    # One-sided confidence bounds on each end, accounting for uncertainty in the bias
    # term itself as well as in the estimate.
    quantile = float(stats.norm.ppf(level))
    se_lower = _bound_std_error(estimate.influence_curve, -strength * elements.psi_max_bias, result)
    se_upper = _bound_std_error(estimate.influence_curve, strength * elements.psi_max_bias, result)
    ci_lower = lower - quantile * se_lower
    ci_upper = upper + quantile * se_upper

    rv, rva = _robustness_values(result, elements, estimate.psi, rho, level, null_hypothesis)
    return SensitivityBounds(
        estimand=estimand,
        psi=estimate.psi,
        cf_y=cf_y,
        cf_d=cf_d,
        rho=rho,
        confounding_strength=strength,
        max_bias=elements.max_bias,
        lower=lower,
        upper=upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        level=level,
        robustness_value=rv,
        robustness_value_ci=rva,
        null_hypothesis=null_hypothesis,
    )


def _bound_std_error(psi_estimate: FloatArray, psi_bias: FloatArray, result: TMLEResult) -> float:
    """Standard error of a bias-adjusted bound."""
    combined = np.asarray(psi_estimate, dtype=float) + np.asarray(psi_bias, dtype=float)
    return float(np.sqrt(influence_variance(combined, result.data.cluster)))


def _robustness_values(
    result: TMLEResult,
    elements: SensitivityElements,
    psi: float,
    rho: float,
    level: float,
    null_hypothesis: float,
) -> tuple[float, float]:
    """Solve for ``cf_y = cf_d = v`` at which a bound reaches the null."""
    side = 1.0 if null_hypothesis > psi else -1.0
    quantile = float(stats.norm.ppf(level))

    def bound_at(value: float, *, with_ci: bool) -> float:
        strength = _confounding_strength(value, value, rho)
        bias = strength * elements.max_bias
        edge = psi + side * bias
        if not with_ci:
            return edge
        se = _bound_std_error(
            result[elements.estimand].influence_curve,
            side * strength * elements.psi_max_bias,
            result,
        )
        return edge + side * quantile * se

    def objective(value: float, with_ci: bool) -> float:
        return float((bound_at(value, with_ci=with_ci) - null_hypothesis) ** 2)

    rv = float(
        optimize.minimize_scalar(objective, bounds=(0.0, 0.9999), method="bounded", args=(False,)).x
    )
    rva = float(
        optimize.minimize_scalar(objective, bounds=(0.0, 0.9999), method="bounded", args=(True,)).x
    )
    return rv, rva


@dataclass(frozen=True)
class BenchmarkResult:
    """Confounder strengths calibrated against covariates that *were* observed.

    Interpretation: ``cf_y`` and ``cf_d`` are the sensitivity parameters an unobserved
    confounder would need in order to be "as important as" the benchmark covariates,
    measured by how much dropping those covariates degrades the outcome regression and
    the Riesz representer.  ``delta_psi`` is how much the estimate actually moved when
    they were dropped, and ``rho`` is the implied degree of adversity -- a value well
    below 1 says the worst-case ``rho = 1`` is pessimistic for confounders like these.

    Parameters
    ----------
    estimand : str
        Alias of the estimand benchmarked.
    covariates : tuple of str
        The observed covariates the strength is calibrated against.
    cf_y : float
        Share of residual outcome variation those covariates explain.
    cf_d : float
        Share of residual treatment variation those covariates explain.
    rho : float
        The implied alignment of the two.
    delta_psi : float
        How far the estimate moved when they were dropped.
    psi_long : float
        Estimate with the benchmark covariates adjusted for.
    psi_short : float
        Estimate with them dropped.
    sigma2_long : float
        Residual outcome variance with them adjusted for.
    sigma2_short : float
        Residual outcome variance with them dropped.
    nu2_long : float
        Riesz second moment with them adjusted for.
    nu2_short : float
        Riesz second moment with them dropped.
    """

    estimand: str
    covariates: tuple[str, ...]
    cf_y: float
    cf_d: float
    rho: float
    delta_psi: float
    psi_long: float
    psi_short: float
    sigma2_long: float
    sigma2_short: float
    nu2_long: float
    nu2_short: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        return {
            "estimand": self.estimand,
            "covariates": ", ".join(self.covariates),
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "rho": self.rho,
            "delta_psi": self.delta_psi,
            "psi_long": self.psi_long,
            "psi_short": self.psi_short,
        }

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        return "\n".join(
            [
                f"Benchmark for {self.estimand!r} against {list(self.covariates)}",
                "-" * 48,
                format_table(
                    ["quantity", "with covariates", "without"],
                    [
                        ["estimate", f"{self.psi_long:.5g}", f"{self.psi_short:.5g}"],
                        ["sigma^2", f"{self.sigma2_long:.5g}", f"{self.sigma2_short:.5g}"],
                        ["nu^2", f"{self.nu2_long:.5g}", f"{self.nu2_short:.5g}"],
                    ],
                ),
                "",
                f"implied cf_y = {self.cf_y:.4f}, cf_d = {self.cf_d:.4f}, rho = {self.rho:.4f}",
                f"the estimate moved by {self.delta_psi:+.5g} when these covariates were dropped",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def benchmark(
    result: TMLEResult,
    covariates: Any,
    *,
    estimand: str = "ate",
    nu2_estimator: str = "auto",
) -> BenchmarkResult:
    """Calibrate ``cf_y`` and ``cf_d`` against observed covariates.

    Refits the whole model *without* the named covariates (the "short" model) and
    compares it with the full fit.  The resulting gain statistics say how strong a
    confounder like the dropped ones would be, on the ``cf_y``/``cf_d`` scale, which
    turns an abstract sensitivity parameter into a concrete comparison.

    Note this is a genuine refit, so it costs about as much as the original fit.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    covariates : sequence of str
        Observed covariates to calibrate against.
    estimand : str
        Alias to benchmark.
    nu2_estimator : {"auto", "analytic", "riesz"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    BenchmarkResult
        The sensitivity parameters a confounder as important as those covariates
        would need, and how far the estimate moved without them.
    """
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError("benchmark needs the fitted estimator that produced the result")
    names = tuple([covariates] if isinstance(covariates, str) else covariates)

    long_elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    short_data = result.data.without_covariates(names)
    short_result = estimator.refit(short_data, intermediate_value=result.intermediate_value)
    short_elements = sensitivity_elements(short_result, estimand, nu2_estimator=nu2_estimator)

    var_y = float(np.var(result.data.outcome[result.data.observed]))
    r2_long = 1.0 - long_elements.sigma2 / var_y
    r2_short = 1.0 - short_elements.sigma2 / var_y
    r2_riesz = short_elements.nu2 / long_elements.nu2

    cf_y = float(np.clip((r2_long - r2_short) / (1.0 - r2_long), 0.0, 1.0))
    cf_d = float(np.clip((1.0 - r2_riesz) / r2_riesz, 0.0, 1.0)) if r2_riesz > 0 else 1.0

    delta = short_result[estimand].psi - result[estimand].psi
    var_g = short_elements.sigma2 - long_elements.sigma2
    var_riesz = long_elements.nu2 - short_elements.nu2
    if var_g > 0 and var_riesz > 0:
        rho = float(np.clip(abs(delta) / np.sqrt(var_g * var_riesz), 0.0, 1.0)) * float(
            np.sign(delta)
        )
    else:
        rho = float(np.sign(delta))

    return BenchmarkResult(
        estimand=estimand,
        covariates=names,
        cf_y=cf_y,
        cf_d=cf_d,
        rho=rho,
        delta_psi=delta,
        psi_long=result[estimand].psi,
        psi_short=short_result[estimand].psi,
        sigma2_long=long_elements.sigma2,
        sigma2_short=short_elements.sigma2,
        nu2_long=long_elements.nu2,
        nu2_short=short_elements.nu2,
    )


def robustness_value(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
) -> dict[str, float]:
    """The confounding strength that would explain the effect away.

    Returns ``{"rv": ..., "rva": ...}``: the value of ``cf_y = cf_d`` at which the point
    estimate reaches ``null_hypothesis``, and the value at which its confidence bound
    does.  This is the single most useful number in this module, because it requires no
    guess about how strong an unmeasured confounder might be -- it reports the
    threshold and lets the reader judge whether it is plausible.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to report on.
    rho : float
        How adversarially the two sensitivity parameters are aligned.
    level : float
        Coverage level used for the confidence-limit value.
    null_hypothesis : float
        Value the strength is measured against.
    nu2_estimator : {"auto", "analytic", "riesz"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    dict of str to float
        The strength that moves the point estimate to the null, and the one that
        moves the confidence limit there.
    """
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    rv, rva = _robustness_values(
        result, elements, result[estimand].psi, rho, level, null_hypothesis
    )
    return {"rv": rv, "rva": rva, "max_bias": elements.max_bias}


def contour_data(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    rho: float = 1.0,
    grid_size: int = 20,
    grid_bounds: tuple[float, float] = (0.15, 0.15),
    bound: str = "lower",
    nu2_estimator: str = "auto",
) -> Any:
    """A ``cf_d`` x ``cf_y`` grid of bias-adjusted bounds, for a contour plot.

    Returned as a long-format frame (``cf_d``, ``cf_y``, ``value``) rather than a plot, so
    it can be rendered with whatever plotting stack the caller already uses.
    """
    if bound not in ("lower", "upper"):
        raise ValueError(f"bound must be 'lower' or 'upper'; got {bound!r}")
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    psi = result[estimand].psi
    sign = -1.0 if bound == "lower" else 1.0

    cf_d_grid = np.linspace(0.0, grid_bounds[0], grid_size)
    cf_y_grid = np.linspace(0.0, grid_bounds[1], grid_size)
    rows_d, rows_y, values = [], [], []
    for cf_d in cf_d_grid:
        for cf_y in cf_y_grid:
            strength = _confounding_strength(float(cf_y), float(cf_d), rho)
            rows_d.append(float(cf_d))
            rows_y.append(float(cf_y))
            values.append(psi + sign * strength * elements.max_bias)
    return result.data.frame_like({"cf_d": rows_d, "cf_y": rows_y, "value": values})
