"""Shared scalar-result composition.

Point-treatment and longitudinal estimators produce different scientific artifacts, but
their mapping, covariance, and delta-method operations have exactly the same algebra.
Keeping that algebra here prevents result classes from drifting while leaving their
method-specific reporting and diagnostics separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, Protocol, TypeVar

import numpy as np

from .._inference_status import PARAMETER_STATUSES, InferenceStatus, precedent_status
from .._typing import FloatArray
from ..exceptions import CapabilityError
from .bootstrap import BootstrapResult
from .cluster import influence_covariance, stacked_second_moment_covariance
from .delta import Transform, delta_method, log_odds_ratio_influence, log_ratio_influence
from .influence import (
    _SECOND_MOMENT_CLUSTER_REFUSAL,
    BootstrapSummary,
    CovarianceRule,
    ParameterEstimate,
    Scale,
    make_estimate,
    minimum_reference_df,
)

__all__ = [
    "attach_bootstrap",
    "covariance_rule",
    "derived_bootstrap",
    "estimate_covariance",
    "estimate_curves",
    "inference_status",
    "linear_functional",
    "ratio_contrast",
    "refuse_odds_ratio",
    "reported_status",
    "select_estimates",
    "smooth_contrast",
    "sole_estimate",
]

RatioKind = Literal["rr", "or"]


def sole_estimate(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
    """Return the sole estimate, refusing to guess on a multi-parameter result."""
    if len(estimates) != 1:
        raise ValueError(
            "this result contains multiple parameters; index the one you want from "
            f"{list(estimates)}"
        )
    return next(iter(estimates.values()))


def select_estimates(
    estimates: Mapping[str, ParameterEstimate], names: Sequence[str] | None
) -> tuple[str, ...]:
    """Normalize and validate a requested ordered subset of estimates."""
    chosen = tuple(estimates) if names is None else tuple(names)
    if not chosen:
        raise ValueError(f"no parameters selected; this result reports {list(estimates)}")
    missing = [name for name in chosen if name not in estimates]
    if missing:
        raise KeyError(f"unknown parameter(s) {missing}; this result reports {list(estimates)}")
    return chosen


def estimate_curves(estimates: Mapping[str, ParameterEstimate]) -> dict[str, FloatArray]:
    """Influence curves in the result's stable report order."""
    return {name: estimate.influence_curve for name, estimate in estimates.items()}


def covariance_rule(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str],
    *,
    cluster: Any = None,
) -> CovarianceRule:
    """The one covariance rule every selected estimate declares.

    A selection that mixes rules is refused. No derivation here supplies the
    cross-covariance between a centred curve and a raw second moment.
    """
    rules = {estimates[name].covariance_rule for name in names}
    if len(rules) != 1:
        raise ValueError(
            f"the selected estimates {list(names)} declare different covariance rules "
            f"{sorted(rules)}; select estimates that share one rule"
        )
    rule = rules.pop()
    if rule == "second_moment" and cluster is not None:
        raise ValueError(_SECOND_MOMENT_CLUSTER_REFUSAL)
    return rule


def inference_status(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str],
) -> InferenceStatus:
    """The one inference status every selected estimate declares.

    A selection that mixes fit statuses is refused. A contrast of one inferential estimate
    and one working-mechanism plug-in estimate is not inference, and inheriting the
    inferential status would launder the refused half into an interval.

    A status of :data:`~cleverly._inference_status.PARAMETER_STATUSES` belongs to one
    parameter and sits beside its fit's status. A selection that reads such a parameter
    takes it, through :func:`~cleverly._inference_status.precedent_status`, so a contrast of
    a constant-node risk and an ordinary one is a constant-node diagnostic.
    """
    statuses = {estimates[name].inference for name in names}
    fit_level = {status for status in statuses if status not in PARAMETER_STATUSES}
    if len(fit_level) > 1:
        raise ValueError(
            f"the selected estimates {list(names)} declare different inference statuses "
            f"{sorted(statuses)}; a contrast is inferential output or it is not"
        )
    if len(fit_level) < len(statuses):
        return precedent_status(statuses)
    return statuses.pop()


def reported_status(estimates: Mapping[str, ParameterEstimate]) -> InferenceStatus:
    """The one inference status of the fit a report holds.

    A report with no estimate refuses nothing, so its status is ``"influence_curve"``.
    Otherwise this is :func:`inference_status` over every estimate, which refuses a mix of
    fit statuses. A parameter status
    (:data:`~cleverly._inference_status.PARAMETER_STATUSES`) is not the fit's, so it is left
    out unless every estimate declares one.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The estimates of one report, keyed by name.

    Returns
    -------
    str
        One of :data:`~cleverly.inference.influence.InferenceStatus`.

    Raises
    ------
    ValueError
        When the estimates declare two different fit statuses. A parameter status beside
        the fit status does not raise.
    """
    if not estimates:
        return "influence_curve"
    fitted = tuple(
        name for name, estimate in estimates.items() if estimate.inference not in PARAMETER_STATUSES
    )
    return inference_status(estimates, fitted or tuple(estimates))


def estimate_covariance(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str] | None,
    *,
    cluster: Any = None,
) -> FloatArray:
    """Joint covariance under the rule the selected estimates declare.

    The ``"centered"`` rule is the sample covariance at the observation or declared cluster
    unit. The ``"second_moment"`` rule is the raw second moment. Under either rule, an
    estimate's diagonal entry does not depend on which other estimates are selected.
    """
    chosen = select_estimates(estimates, names)
    rule = covariance_rule(estimates, chosen, cluster=cluster)
    curves = np.column_stack([estimates[name].influence_curve for name in chosen])
    if rule == "second_moment":
        return stacked_second_moment_covariance(curves)
    return influence_covariance(curves, cluster=cluster)


def smooth_contrast(
    estimates: Mapping[str, ParameterEstimate],
    function: Callable[[FloatArray], float],
    names: Sequence[str],
    *,
    n: int,
    cluster: Any = None,
    alpha: float = 0.05,
    name: str | None = None,
    scale: Scale = "difference",
    gradient: Callable[[FloatArray], FloatArray] | None = None,
    transform: Transform | None = None,
) -> ParameterEstimate:
    """Apply the delta method to a smooth function of jointly estimated parameters.

    The derived estimate inherits the covariance rule of its inputs, and its variance
    applies that rule to the derived influence curve. Under either rule, that variance
    equals the gradient's quadratic form with the joint covariance on compatible scales.
    :func:`estimate_covariance` reads input inference scales. For untransformed inputs on
    their reported scales, it can be used directly; otherwise the chain rule must first
    map the covariance or gradient to the same scales.

    It inherits the inference status of its inputs too. A contrast of estimates the
    package supplies no inference for is itself refused, rather than becoming an
    interval that its inputs do not have. Its Student t reference takes the smallest
    degrees of freedom of its inputs (:func:`~cleverly.inference.influence.minimum_reference_df`).

    With ``scale="ratio"`` the value must be positive. The estimate stores
    ``log_psi = log(value)`` and the curve divided by the value, which is the chain rule
    for the log. With ``transform=`` the curve is multiplied by the transform's slope at
    the value, and the interval is mapped back through the inverse. Input curves are first
    mapped from their inference scales to the reported scales that ``function`` reads.

    Two mechanisms carry one idea here: ``scale="ratio"`` with ``log_psi`` and
    ``transform=Transform.log()``. Both stay, because every registered ``rr`` and ``or``
    estimate stores ``log_psi``, and saved frames name it. A test pins that the two give
    the same standard error, interval and p-value.
    """
    if transform is not None:
        if scale == "ratio":
            raise ValueError(
                "a ratio-scale estimate already carries the log transform. Pass "
                'scale="ratio" or transform=, not both'
            )
        if scale not in ("level", "difference"):
            raise ValueError(
                "with transform=, scale names the scale of psi and must be 'level' or "
                f"'difference'; the transform carries the inference scale. Got {scale!r}"
            )
    chosen = select_estimates(estimates, names)
    rule = covariance_rule(estimates, chosen, cluster=cluster)
    status = inference_status(estimates, chosen)
    value, curve = delta_method(
        function,
        [estimates[key].psi for key in chosen],
        [_reported_curve(estimates[key]) for key in chosen],
        gradient=gradient,
    )
    log_psi: float | None = None
    if scale == "ratio":
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"a ratio-scale contrast needs a positive value; got {value:.6g}")
        log_psi = float(np.log(value))
        # The same arithmetic as ``Transform.log()`` below, so the two routes agree bit for
        # bit.
        curve = (1.0 / value) * curve
    if transform is not None:
        mapped = float(transform.forward(value))
        slope = transform.slope(value)
        if not (np.isfinite(mapped) and np.isfinite(slope)):
            raise ValueError(
                f"the contrast value {value:.6g} has no finite value or slope under the "
                f"transform {transform.name!r}"
            )
        curve = slope * curve
    estimate = make_estimate(
        name or f"contrast({', '.join(chosen)})",
        value,
        curve,
        n=n,
        cluster=cluster,
        scale=scale,
        alpha=alpha,
        log_psi=log_psi,
        covariance_rule=rule,
        inference=status,
        reference_df=minimum_reference_df([estimates[key] for key in chosen]),
    )
    return estimate if transform is None else replace(estimate, transform=transform)


def _reported_curve(estimate: ParameterEstimate) -> FloatArray:
    """The input curve on the reported scale that a contrast's function reads.

    Ratios store the curve of their log, and transformed estimates store the curve of
    their forward map. Undo that map before applying the contrast's derivative.
    """
    curve = estimate.influence_curve
    if estimate.transform is not None:
        mapped = float(estimate.transform.forward(estimate.psi))
        slope = estimate.transform.slope(estimate.psi)
        if not (np.isfinite(mapped) and np.isfinite(slope) and slope != 0.0):
            raise ValueError(
                f"the input estimate {estimate.name!r} needs a finite transformed value "
                "and a finite nonzero slope to recover its reported-scale influence curve"
            )
        return np.asarray(curve / slope, dtype=float)
    if estimate.scale == "ratio":
        if not (np.isfinite(estimate.psi) and estimate.psi > 0.0):
            raise ValueError(
                f"the ratio input {estimate.name!r} needs a positive finite value to "
                "recover its reported-scale influence curve"
            )
        return np.asarray(estimate.psi * curve, dtype=float)
    return curve


def ratio_contrast(
    estimates: Mapping[str, ParameterEstimate],
    numerator: str,
    denominator: str,
    *,
    kind: RatioKind = "rr",
    complement: bool = False,
    n: int,
    cluster: Any = None,
    alpha: float = 0.05,
    name: str,
) -> ParameterEstimate:
    r"""The risk ratio or odds ratio of two level estimates, with inference on the log scale.

    For a risk ratio, :math:`\log\psi = \log\psi_a - \log\psi_b` with curve
    :math:`IC_a/\psi_a - IC_b/\psi_b`. For an odds ratio the curve is
    :math:`IC_a/(\psi_a(1-\psi_a)) - IC_b/(\psi_b(1-\psi_b))`. This is the curve of
    ``lmtp_contrast(type = "rr")`` and ``type = "or"`` in R ``lmtp`` 1.5.4. With
    ``complement=True`` each level is mapped to :math:`1 - \psi` and its curve to
    :math:`-IC` first, which gives the ratio of two survival probabilities.

    The derived estimate inherits the covariance rule, the inference status and the
    smallest reference degrees of freedom of its inputs, as :func:`smooth_contrast` does.
    A transformed level's curve is first mapped back to its reported probability scale.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The estimates of one fit.
    numerator : str
        The name of the level in the numerator.
    denominator : str
        The name of the level in the denominator.
    kind : {"rr", "or"}, default="rr"
        A ratio of the levels, or a ratio of their odds.
    complement : bool, default=False
        Whether to take the ratio of :math:`1 - \psi` instead of :math:`\psi`.
    n : int
        Number of observations behind the estimates.
    cluster : ndarray or None, default=None
        Cluster codes of the fit, or ``None`` for independent rows.
    alpha : float, default=0.05
        Significance level of the interval.
    name : str
        The name of the derived estimate.

    Returns
    -------
    ParameterEstimate
        The ratio, on the ``"ratio"`` scale with ``log_psi`` set.

    Raises
    ------
    ValueError
        When an input is not reported on the ``"level"`` scale, when the selection mixes
        covariance rules or statuses, or when a level is outside the domain of the log.
    """
    chosen = select_estimates(estimates, (numerator, denominator))
    for key in chosen:
        if estimates[key].scale != "level":
            raise ValueError(
                f"a ratio compares two levels, and {key!r} is reported on the "
                f"{estimates[key].scale!r} scale. Pass two level estimates, such as two "
                "counterfactual means"
            )
    rule = covariance_rule(estimates, chosen, cluster=cluster)
    status = inference_status(estimates, chosen)
    top, bottom = estimates[numerator], estimates[denominator]
    psi_a, ic_a = top.psi, _reported_curve(top)
    psi_b, ic_b = bottom.psi, _reported_curve(bottom)
    if complement:
        psi_a, ic_a = 1.0 - psi_a, -ic_a
        psi_b, ic_b = 1.0 - psi_b, -ic_b
    if kind not in ("rr", "or"):
        raise ValueError(f"kind must be 'rr' or 'or'; got {kind!r}")
    influence = log_ratio_influence if kind == "rr" else log_odds_ratio_influence
    try:
        log_psi, curve = influence(psi_a, ic_a, psi_b, ic_b)
    except ValueError:
        # The helpers name EY1 and EY0, which is wrong for a regimen level or a survival
        # view, so the refusal is restated with the caller's own inputs.
        view = "1 - " if complement else ""
        domain = "strictly positive" if kind == "rr" else "strictly inside (0, 1)"
        raise ValueError(
            f"the {'risk' if kind == 'rr' else 'odds'} ratio needs both inputs {domain}; got "
            f"{view}{numerator}={psi_a:.6g} and {view}{denominator}={psi_b:.6g}"
        ) from None
    return make_estimate(
        name,
        float(np.exp(log_psi)),
        curve,
        n=n,
        cluster=cluster,
        scale="ratio",
        alpha=alpha,
        log_psi=log_psi,
        covariance_rule=rule,
        inference=status,
        reference_df=minimum_reference_df([estimates[key] for key in chosen]),
    )


def derived_bootstrap(
    estimates: Mapping[str, ParameterEstimate],
    bootstrap: BootstrapResult | None,
    names: Sequence[str],
    function: Callable[[FloatArray], float],
    *,
    alpha: float,
) -> BootstrapSummary | None:
    """The bootstrap summary of a derived estimate, from its inputs' replicate draws.

    Each bootstrap replicate already holds a point estimate of every reported parameter, in
    aligned rows of :attr:`BootstrapResult.draws`.  Applying ``function`` to each row gives
    the replicate value of the derived estimate, so its percentile interval is the one the
    full refit implies. A replicate with a missing input gives a missing value. Domain-invalid
    ratios also give missing values. The failure count includes every requested replicate
    without a finite derived value. User callable exceptions propagate. The summary is
    licensed as inference only when every input's summary is.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The estimates of the fit.
    bootstrap : BootstrapResult or None
        The fit's replicate draws.
    names : sequence of str
        The inputs, in the order ``function`` reads them.
    function : callable
        The derived value as a function of the inputs.
    alpha : float
        Significance level of the percentile interval.

    Returns
    -------
    BootstrapSummary or None
        ``None`` when the fit carries no bootstrap, or an input has no draws.
    """
    if bootstrap is None or any(name not in bootstrap.draws for name in names):
        return None
    summaries = [estimates[name].bootstrap for name in names]
    if any(summary is None for summary in summaries):
        return None
    matrix = np.column_stack([np.asarray(bootstrap.draws[name], dtype=float) for name in names])
    values = np.array(
        [float(function(row)) if np.all(np.isfinite(row)) else np.nan for row in matrix]
    )
    derived = BootstrapResult(
        draws={"derived": values},
        n_requested=bootstrap.n_requested,
        n_failed=bootstrap.n_requested - int(np.count_nonzero(np.isfinite(values))),
        resampling=bootstrap.resampling,
    ).summary("derived", alpha)
    licensed = all(summary.inferential for summary in summaries if summary is not None)
    return replace(derived, inferential=licensed)


def with_derived_bootstrap(
    derived: ParameterEstimate,
    estimates: Mapping[str, ParameterEstimate],
    bootstrap: BootstrapResult | None,
    names: Sequence[str],
    function: Callable[[FloatArray], float],
) -> ParameterEstimate:
    """``derived`` with the bootstrap summary :func:`derived_bootstrap` gives, if any."""
    summary = derived_bootstrap(estimates, bootstrap, names, function, alpha=derived.alpha)
    return derived if summary is None else derived.with_bootstrap(summary)


def ratio_function(kind: RatioKind, complement: bool) -> Callable[[FloatArray], float]:
    """The ratio of two levels as a function of the pair, for replicate draws."""

    def value(pair: FloatArray) -> float:
        a, b = float(pair[0]), float(pair[1])
        if complement:
            a, b = 1.0 - a, 1.0 - b
        if kind == "rr":
            if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
                return float("nan")
            return a / b
        if not (0.0 < a < 1.0 and 0.0 < b < 1.0):
            return float("nan")
        return (a / (1.0 - a)) / (b / (1.0 - b))

    return value


def linear_function(
    weights: Sequence[float], constant: float = 0.0
) -> Callable[[FloatArray], float]:
    """``constant + weights . values``, for replicate draws."""
    vector = np.asarray(weights, dtype=float)

    def value(values: FloatArray) -> float:
        return float(constant + vector @ np.asarray(values, dtype=float))

    return value


def linear_functional(
    estimates: Mapping[str, ParameterEstimate],
    weights: Mapping[str, float],
    *,
    constant: float = 0.0,
    n: int,
    cluster: Any = None,
    alpha: float = 0.05,
    name: str,
    scale: Scale,
) -> ParameterEstimate:
    r"""A constant plus a weighted sum of estimates, with its exact influence curve.

    The value is :math:`c + \sum_k w_k \psi_k` and the curve is
    :math:`\sum_k w_k IC_k`. It runs through :func:`smooth_contrast` with the analytic
    gradient, so the status and the covariance rule are inherited and the curve is exact.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The estimates of one fit.
    weights : Mapping of str to float
        The weight of each selected estimate, in the order of the sum.
    constant : float, default=0.0
        The constant :math:`c`.
    n : int
        Number of observations behind the estimates.
    cluster : ndarray or None, default=None
        Cluster codes of the fit, or ``None`` for independent rows.
    alpha : float, default=0.05
        Significance level of the interval.
    name : str
        The name of the derived estimate.
    scale : {"level", "difference", "ratio", "fraction"}
        The reported scale of the derived estimate.

    Returns
    -------
    ParameterEstimate
        The linear functional.
    """
    chosen = tuple(weights)
    vector = np.array([float(weights[key]) for key in chosen], dtype=float)

    def value(psi: FloatArray) -> float:
        return float(constant + vector @ np.asarray(psi, dtype=float))

    def gradient(psi: FloatArray) -> FloatArray:
        return vector

    return smooth_contrast(
        estimates,
        value,
        chosen,
        n=n,
        cluster=cluster,
        alpha=alpha,
        name=name,
        scale=scale,
        gradient=gradient,
    )


def refuse_odds_ratio(family: str) -> None:
    """Refuse an odds ratio on a fit whose outcome is not binary.

    The registry's own admission rule for ``or``, ``TARGETS["or"].requires_family``, so a
    post-fit ratio and a fitted one are admitted by one rule.

    Parameters
    ----------
    family : str
        The outcome family of the fit.

    Raises
    ------
    CapabilityError
        When the family is not the one the odds ratio requires.
    """
    from ..targets import TARGETS

    required = TARGETS["or"].requires_family
    if required is not None and family != required:
        raise CapabilityError(
            "an odds ratio compares two probabilities, and this fit's outcome family is "
            f'{family!r}. Use kind="rr" for a ratio of means'
        )


class _BootstrappableResult(Protocol):
    @property
    def estimates(self) -> Mapping[str, ParameterEstimate]: ...

    @property
    def config(self) -> Any: ...


_ResultT = TypeVar("_ResultT", bound=_BootstrappableResult)


def attach_bootstrap(
    result: _ResultT, bootstrap: BootstrapResult, *, inferential: bool = True
) -> _ResultT:
    """Return a copy of ``result`` with bootstrap summaries attached.

    One function for a point-treatment and a longitudinal result, so both publish the
    same summaries under the same rule.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        The fitted result.
    bootstrap : BootstrapResult
        The replicate draws.
    inferential : bool, default=True
        Whether a registered study licenses the percentile interval as inference for
        this fit. Stored on each :class:`~cleverly.inference.BootstrapSummary`.

    Returns
    -------
    TMLEResult or LongitudinalResult
        A copy whose estimates carry their bootstrap summaries.
    """
    alpha = result.config.alpha_sig
    estimates = {
        name: (
            estimate.with_bootstrap(
                replace(bootstrap.summary(name, alpha), inferential=inferential)
            )
            if name in bootstrap.draws
            else estimate
        )
        for name, estimate in result.estimates.items()
    }
    return replace(result, estimates=estimates, bootstrap=bootstrap)  # type: ignore[type-var]
