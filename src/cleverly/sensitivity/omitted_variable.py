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

Scope: the bound applies to the linear functionals this library estimates
(``ate``, ``ey1``, ``ey0``, ``att``, ``atc``).  Ratios are not linear functionals of the
outcome regression, so use :mod:`cleverly.sensitivity.evalue` for those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

from .._typing import FloatArray
from ..estimators.base import format_table
from ..estimators.targeting import build_submodel
from ..inference.cluster import influence_variance
from ..utils.bounds import g_bounds_for

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

#: Estimands for which the Riesz representer -- and therefore this bound -- is defined.
LINEAR_ESTIMANDS: frozenset[str] = frozenset({"ate", "ey1", "ey0", "att", "atc"})


@dataclass(frozen=True)
class SensitivityElements:
    r"""The ingredients of the bias bound for one estimand.

    Attributes
    ----------
    sigma2:
        :math:`E[(Y - \bar Q^*(A, W))^2]`, the residual outcome variance.
    nu2:
        :math:`E[\alpha(A, W)^2]`, the second moment of the Riesz representer.
    max_bias:
        :math:`\sqrt{\sigma^2 \nu^2}` -- the largest bias any confounder could produce
        if it explained *all* the residual variation on both sides.
    psi_max_bias:
        Influence curve of ``max_bias``, so the bias-adjusted bounds get confidence
        intervals rather than being treated as known constants.
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
    if estimand not in LINEAR_ESTIMANDS:
        raise ValueError(
            f"the omitted-variable bound applies to {sorted(LINEAR_ESTIMANDS)}, not "
            f"{estimand!r}. For a risk ratio or odds ratio use sensitivity.evalue()."
        )
    # Before the "was not requested" check: on a multi-arm fit no parameter is named
    # plainly `ate` -- they are `ate[mid vs low]` and so on -- so that check would fire
    # first and report a missing estimand when the real answer is that this bound does
    # not apply to the fit at all.
    if not result.data.is_binary_treatment:
        raise ValueError(
            "the omitted-variable bound is derived for a binary treatment; this fit has "
            f"{result.data.n_arms} arms {list(result.data.treatment_levels)}. The bound "
            "rests on a scalar confounding strength in the treatment equation, and with "
            "more than two arms an omitted covariate has one such strength per arm -- a "
            "different derivation, not a wider loop. Use sensitivity.evalue() for a "
            "contrast, or restrict the fit to the two arms being compared."
        )
    if estimand not in result.estimates:
        raise ValueError(f"estimand {estimand!r} was not requested in this fit")

    per_repeat = [
        _elements_for(result, repeat, estimand, nu2_estimator) for repeat in result.repeats
    ]
    return _average_elements(per_repeat)


def _average_elements(per_repeat: Sequence[SensitivityElements]) -> SensitivityElements:
    """Average the bound's pieces over the cross-fitting draws.

    Every field is averaged, scalars and per-unit arrays alike, which is the same rule
    :func:`~cleverly.inference.average_estimates` applies to the estimate itself: the
    reported bound is the mean of the per-draw bounds, and the curve that goes with it is
    the mean of theirs.  ``max_bias`` is averaged rather than recomputed from the averaged
    ``sigma2`` and ``nu2`` for exactly that reason -- ``sqrt`` of the averages is not the
    average of the ``sqrt``s, and it is the bound that is being reported.
    """
    if len(per_repeat) == 1:
        return per_repeat[0]
    methods = {elements.nu2_estimator for elements in per_repeat}
    return SensitivityElements(
        estimand=per_repeat[0].estimand,
        sigma2=float(np.mean([e.sigma2 for e in per_repeat])),
        nu2=float(np.mean([e.nu2 for e in per_repeat])),
        max_bias=float(np.mean([e.max_bias for e in per_repeat])),
        psi_sigma2=np.mean([e.psi_sigma2 for e in per_repeat], axis=0),
        psi_nu2=np.mean([e.psi_nu2 for e in per_repeat], axis=0),
        psi_max_bias=np.mean([e.psi_max_bias for e in per_repeat], axis=0),
        riesz_representer=np.mean([e.riesz_representer for e in per_repeat], axis=0),
        # Named as a mixture rather than as whichever came first when the draws disagree,
        # which happens only when the doubly-robust nu2 went non-positive on some of them
        # and fell back. That is a diagnosis of the propensity fit, and hiding it behind
        # one draw's label would lose it.
        nu2_estimator=(
            per_repeat[0].nu2_estimator if len(methods) == 1 else "+".join(sorted(methods))
        ),
    )


def _elements_for(
    result: TMLEResult,
    repeat: RepeatFit,
    estimand: str,
    nu2_estimator: str,
) -> SensitivityElements:
    """The bound's pieces under one cross-fitting draw.

    Takes the targeted ``Qbar`` and the mechanism from the same
    :class:`~cleverly.estimators._nuisance.RepeatFit`, which is what makes ``sigma2`` the
    residual variance of the regression whose propensity ``nu2`` was computed from.
    """
    data = result.data
    scaler = repeat.nuisance.scaler
    group = "mean" if estimand in ("ate", "ey1", "ey0") else estimand
    fluctuation = repeat.fluctuations[group]
    bounds = g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)
    submodel = build_submodel(
        data,
        repeat.nuisance,
        group,
        bounds=bounds,
        nuisance_bound=result.config.missingness_bound,
        intermediate_value=result.intermediate_value,
    )
    # The arm-1 margin: ``_m_alpha`` weights the ATT/ATC contrast by ``g1`` and its
    # complement, which is a two-arm statement -- guarded above.
    propensity = repeat.nuisance.bounded_propensity(bounds)[:, repeat.nuisance.arms.index(1.0)]

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

    representer = _riesz_representer(estimand, submodel, data.treatment)
    plugin = float(np.average(representer**2, weights=weights))

    method = nu2_estimator
    if method == "auto":
        method = "doubly_robust"
    if method == "doubly_robust":
        m_alpha = _m_alpha(estimand, submodel, propensity, data.treated_fraction)
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
        estimand=estimand,
        sigma2=sigma2,
        nu2=nu2,
        max_bias=max_bias,
        psi_sigma2=psi_sigma2,
        psi_nu2=psi_nu2,
        psi_max_bias=psi_max_bias,
        riesz_representer=representer,
        nu2_estimator=method,
    )


def _riesz_representer(estimand: str, submodel: Any, treatment: FloatArray) -> FloatArray:
    r"""``alpha(A, W)`` for the requested estimand.

    The clever covariate *is* the Riesz representer -- that is why the same object
    both drives the targeting step and controls how much an omitted confounder can
    move the estimate.  Poor overlap inflates both simultaneously.
    """
    if estimand == "ate":
        return np.asarray(submodel.observed[:, 1] - submodel.observed[:, 0], dtype=float)
    if estimand == "ey1":
        return np.asarray(submodel.observed[:, 1], dtype=float)
    if estimand == "ey0":
        return np.asarray(submodel.observed[:, 0], dtype=float)
    return np.asarray(submodel.observed[:, 0], dtype=float)


def _m_alpha(
    estimand: str, submodel: Any, propensity: FloatArray, treated_fraction: float
) -> FloatArray:
    r"""The target functional applied to the Riesz representer, ``m(W, alpha)``.

    Used by the doubly robust estimator of :math:`\nu^2`, which relies on the Riesz
    identity :math:`E[m(W, \alpha)] = E[\alpha^2]`.
    """
    # ``arms[a][:, c]`` is the covariate at arm ``a`` in the column targeting arm ``c``:
    # the mean submodel has one column per arm, so both indices are arm levels and
    # neither is a positional 0 or 1.
    if estimand == "ate":
        one, zero = submodel.arms[1.0], submodel.arms[0.0]
        treated, control = submodel.arm_columns[1.0], submodel.arm_columns[0.0]
        return np.asarray(
            one[:, treated] - one[:, control] - (zero[:, treated] - zero[:, control]),
            dtype=float,
        )
    if estimand == "ey1":
        return np.asarray(submodel.arms[1.0][:, submodel.arm_columns[1.0]], dtype=float)
    if estimand == "ey0":
        return np.asarray(submodel.arms[0.0][:, submodel.arm_columns[0.0]], dtype=float)

    # ATT / ATC: the functional carries an arm-membership weight, since it averages the
    # contrast over the treated (or control) subpopulation rather than over everyone.
    # Column 0 is the sole column of a contrast submodel, not an arm's column.
    contrast = np.asarray(submodel.arms[1.0][:, 0] - submodel.arms[0.0][:, 0], dtype=float)
    if estimand == "att":
        weight = propensity / treated_fraction
    else:
        weight = (1.0 - propensity) / (1.0 - treated_fraction)
    return np.asarray(weight * contrast, dtype=float)


@dataclass(frozen=True)
class SensitivityBounds:
    """Bias-adjusted bounds under an assumed confounder strength."""

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
    """
    estimator = result.estimator
    if estimator is None:
        raise ValueError("benchmark needs the fitted estimator that produced the result")
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
