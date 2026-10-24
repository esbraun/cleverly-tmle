"""Shared scalar-result composition.

Point-treatment and longitudinal estimators produce different scientific artifacts, but
their mapping, covariance, and delta-method operations have exactly the same algebra.
Keeping that algebra here prevents result classes from drifting while leaving their
method-specific reporting and diagnostics separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary

from .._inference_status import PARAMETER_STATUSES, InferenceStatus, precedent_status
from .._support import Classification, Kind, combine_support, support_for
from .._typing import FloatArray, IntArray
from ..exceptions import CapabilityError, RefusalReason
from .bootstrap import BootstrapInterval, BootstrapResult, bca_acceleration
from .cluster import influence_covariance, stacked_second_moment_covariance
from .delta import Transform, delta_method, log_odds_ratio_influence, log_ratio_influence
from .influence import (
    _SECOND_MOMENT_CLUSTER_REFUSAL,
    BootstrapSummary,
    CovarianceRule,
    ParameterEstimate,
    Scale,
    _median_adjusted_variance,
    make_estimate,
    median_estimates,
    minimum_reference_df,
    spread_name,
    with_precedent_status,
)

__all__ = [
    "RepeatSpread",
    "attach_bootstrap",
    "check_transform_scale",
    "covariance_rule",
    "derived_bootstrap",
    "estimate_covariance",
    "estimate_curves",
    "inference_status",
    "linear_functional",
    "median_covariance",
    "median_derived",
    "ratio_contrast",
    "refuse_odds_ratio",
    "repeat_spread",
    "reported_status",
    "select_estimates",
    "smooth_contrast",
    "sole_estimate",
    "split_noise_lines",
    "transform_estimate",
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
    check_transform_scale(scale, transform)
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
        curve = _transform_slope(value, transform) * curve
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
        support=combine_support((estimates[key].support for key in chosen), kind="derived"),
    )
    return estimate if transform is None else replace(estimate, transform=transform)


def check_transform_scale(scale: Scale, transform: Transform | None) -> None:
    """Refuse a reported scale that cannot carry ``transform``.

    Parameters
    ----------
    scale : str
        The reported scale of a derived estimate.
    transform : Transform or None
        The map of its inference scale.

    Raises
    ------
    ValueError
        When ``transform`` is given with ``scale="ratio"`` or a scale other than
        ``"level"`` and ``"difference"``.
    """
    if transform is None:
        return
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


def _transform_slope(value: float, transform: Transform) -> float:
    """The transform's slope at ``value``, refusing a value outside its domain."""
    mapped = float(transform.forward(value))
    slope = transform.slope(value)
    if not (np.isfinite(mapped) and np.isfinite(slope)):
        raise ValueError(
            f"the contrast value {value:.6g} has no finite value or slope under the "
            f"transform {transform.name!r}"
        )
    return float(slope)


def transform_estimate(
    estimate: ParameterEstimate, transform: Transform, *, cluster: Any = None
) -> ParameterEstimate:
    """``estimate`` carried to the inference scale of ``transform``.

    The arithmetic of :func:`smooth_contrast` with ``transform=``: the curve is multiplied
    by the transform's slope at the value, and the variance is read off that curve under the
    estimate's covariance rule.  So ``transform_estimate(smooth_contrast(...), T)`` equals
    ``smooth_contrast(..., transform=T)`` bit for bit.

    Parameters
    ----------
    estimate : ParameterEstimate
        An untransformed estimate on the ``"level"`` or ``"difference"`` scale.
    transform : Transform
        The monotone map of the inference scale.
    cluster : ndarray or None, default=None
        Cluster codes of the fit, or ``None`` for independent rows.

    Returns
    -------
    ParameterEstimate
        The estimate with ``transform`` set.

    Raises
    ------
    ValueError
        When the value has no finite value or slope under ``transform``.
    """
    slope = _transform_slope(estimate.psi, transform)
    mapped = make_estimate(
        estimate.name,
        estimate.psi,
        slope * estimate.influence_curve,
        n=estimate.n,
        cluster=cluster,
        scale=estimate.scale,
        alpha=estimate.alpha,
        log_psi=estimate.log_psi,
        covariance_rule=estimate.covariance_rule,
        inference=estimate.inference,
        reference_df=estimate.reference_df,
    )
    return replace(mapped, transform=transform)


def _named_draw(error: Exception, draw: int, total: int) -> Exception:
    """``error`` again, of the same type, with its message prefixed by the draw."""
    message = error.args[0] if len(error.args) == 1 and isinstance(error.args[0], str) else error
    text = f"draw {draw} of {total}: {message}"
    try:
        return type(error)(text)
    except TypeError:  # pragma: no cover - an exception type with a custom constructor
        return ValueError(text)


def median_derived(
    per_draw: Sequence[Mapping[str, ParameterEstimate]],
    function: Callable[[Mapping[str, ParameterEstimate]], ParameterEstimate],
    *,
    transform: Transform | None = None,
    cluster: Any = None,
) -> ParameterEstimate:
    """A derived estimate of a repeated fit: computed in every draw, then combined.

    ``function`` maps one draw's estimates to the derived estimate, such as a contrast, a
    ratio or a restricted mean.  It runs on every draw.  A refusal it raises in draw ``r``
    is raised again with its own type, prefixed ``draw r of R:``.  With ``transform=``,
    every draw's value is carried to the transform's scale before any median, and a draw
    where the transform is undefined is refused.  Each draw's result is then stamped with
    the precedent of the draws' statuses
    (:func:`~cleverly.inference.influence.with_precedent_status`), and
    :func:`~cleverly.inference.influence.median_estimates` combines them on their inference
    scale.  So the report is the median of the per-draw functional, never the functional
    of the medians.

    Parameters
    ----------
    per_draw : sequence of mappings
        One estimate mapping per draw.
    function : callable
        The derived estimate of one draw, as a function of that draw's estimates.  It
        returns an untransformed estimate when ``transform`` is given.
    transform : Transform or None, default=None
        The map of the inference scale, applied to each draw's result.
    cluster : ndarray or None, default=None
        Cluster codes of the fit, or ``None`` for independent rows.

    Returns
    -------
    ParameterEstimate
        The median-combined derived estimate.

    Raises
    ------
    ValueError
        When the transform has no finite value or slope at one draw's value, or when
        ``function`` raises it in one draw.
    KeyError
        When ``function`` raises it in one draw, such as for a name a draw lacks.
    TypeError
        When ``function`` raises it in one draw.
    CapabilityError
        When ``function`` refuses one draw.  Each error from ``function`` keeps its type,
        and its message starts ``draw r of R:``.
    """
    total = len(per_draw)
    results: list[ParameterEstimate] = []
    for index, estimates in enumerate(per_draw, start=1):
        try:
            results.append(function(estimates))
        except (ValueError, KeyError, TypeError, CapabilityError) as error:
            raise _named_draw(error, index, total) from error
    if transform is not None:
        mapped: list[ParameterEstimate] = []
        for index, result in enumerate(results, start=1):
            try:
                mapped.append(transform_estimate(result, transform, cluster=cluster))
            except ValueError as error:
                raise CapabilityError(
                    f"draw {index} of {total}: the contrast value {result.psi:.6g} has no "
                    f"finite value or slope under the transform {transform.name!r}. The "
                    "median over draws is taken on the transform's scale, so the transform "
                    "must be defined at every draw's value.",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=(
                        "Use a transform defined there, drop transform= for the natural "
                        "scale, or read each draw in result.repeats."
                    ),
                ) from error
        results = mapped
    name = results[0].name
    stamped = with_precedent_status([{name: result} for result in results])
    return median_estimates(stamped)[name]


def median_covariance(
    per_draw: Sequence[Mapping[str, ParameterEstimate]],
    names: Sequence[str],
    *,
    cluster: Any = None,
) -> FloatArray:
    r"""The joint covariance of a repeated fit, by the matrix rule of equation (3.14).

    Chernozhukov et al. (2018, Definition 3.3 and equation 3.14, pp. C29-C30).  Let
    :math:`t_r` be draw :math:`r`'s estimates of ``names`` on their inference scales,
    :math:`\tilde t` their coordinatewise median, :math:`d_r = t_r - \tilde t`, and
    :math:`\Sigma_r` draw :math:`r`'s :func:`estimate_covariance` of ``names``.  Each draw
    gives the candidate :math:`C_r = \Sigma_r + d_r d_r^\top`.  The result is the candidate
    with the median operator norm, its largest eigenvalue, with ties to the lower draw
    index.  At an even draw count it is the mean of the two candidates with the central
    operator norms.  That mean is nonnegative definite, and its norm can be below both
    candidates' norms.

    The matrix is the rule for the **selected** vector.  The chosen draw depends on the
    selection, so an entry, the diagonal included, can differ between two selections that
    share a name, and from that name's reported variance.  One name is the exception: its
    1x1 matrix is the reported variance bit for bit, because it is computed by the same
    scalar rule.  Each draw is first stamped with the precedent of the draws' statuses, so
    the selection supplies inference only when every draw does.

    Parameters
    ----------
    per_draw : sequence of mappings
        One estimate mapping per draw.
    names : sequence of str
        The selected estimates, in output order.
    cluster : ndarray or None, default=None
        Cluster codes of the fit, or ``None`` for independent rows.

    Returns
    -------
    ndarray
        The ``(k, k)`` matrix.
    """
    stamped = with_precedent_status(per_draw)
    chosen = select_estimates(stamped[0], names)
    if len(chosen) == 1:
        parts = [draw[chosen[0]] for draw in stamped]
        points = np.asarray([part.inference_value for part in parts], dtype=float)
        variance = _median_adjusted_variance(points, [part.variance for part in parts])
        return np.array([[variance]], dtype=float)
    values = np.array(
        [[draw[name].inference_value for name in chosen] for draw in stamped], dtype=float
    )
    displacement = values - np.median(values, axis=0, keepdims=True)
    candidates = [
        estimate_covariance(draw, chosen, cluster=cluster) + np.outer(offset, offset)
        for draw, offset in zip(stamped, displacement, strict=True)
    ]
    norms = np.array([float(np.linalg.eigvalsh(candidate)[-1]) for candidate in candidates])
    order = np.argsort(norms, kind="stable")
    middle = len(candidates) // 2
    if len(candidates) % 2:
        return np.asarray(candidates[int(order[middle])], dtype=float)
    low, high = int(order[middle - 1]), int(order[middle])
    return np.asarray((candidates[low] + candidates[high]) / 2.0, dtype=float)


class RepeatSpread(dict[str, float]):
    """How much each estimate moved between random splits into folds.

    What :meth:`~cleverly.estimators.TMLEResult.repeat_spread` and
    :meth:`~cleverly.longitudinal.LongitudinalResult.repeat_spread` return. It is a ``dict``
    from estimand name to the standard deviation of that estimate across the splits, on the
    scale its standard error is on, and it prints a summary that sets each spread beside the
    standard error it should be read against.

    Parameters
    ----------
    spreads : mapping of str to float
        The standard deviation across the splits, per estimand; ``nan`` where a split gave
        a value the scale cannot take.
    n_repeats : int
        The number of splits.
    std_errors : mapping of str to float
        The spread each estimate's interval uses: the standard error, or the plug-in
        diagnostic of an estimate that supplies no inference.
    inferential : mapping of str to bool
        Whether each estimate supplies inference.
    scales : mapping of str to str
        The scale each spread is taken on: ``"log"`` for a ratio, otherwise ``"original"``.
    """

    def __init__(
        self,
        spreads: Mapping[str, float],
        n_repeats: int,
        std_errors: Mapping[str, float],
        inferential: Mapping[str, bool],
        scales: Mapping[str, str],
    ) -> None:
        super().__init__(spreads)
        self.n_repeats = int(n_repeats)
        self.std_errors = dict(std_errors)
        self.inferential = dict(inferential)
        self.scales = dict(scales)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            RepeatSpread,
            (dict(self), self.n_repeats, self.std_errors, self.inferential, self.scales),
        )

    def to_frame(self) -> Any:
        """One row per estimand: the spread, the standard error it is read against, and
        their ratio.

        Returns
        -------
        dataframe
            The columns ``estimand``, ``spread``, ``std_err`` (``None`` for an estimate that
            supplies no inference), ``plugin_std_err``, ``ratio`` and ``scale``, in the
            default dataframe backend.
        """
        from ..utils.frames import frame_from_dict

        names = list(self)
        return frame_from_dict(
            {
                "estimand": names,
                "spread": [self[name] for name in names],
                "std_err": [
                    self.std_errors[name] if self.inferential[name] else None for name in names
                ],
                "plugin_std_err": [self.std_errors[name] for name in names],
                "ratio": [
                    self[name] / self.std_errors[name]
                    if self.std_errors[name] > 0
                    else float("nan")
                    for name in names
                ],
                "scale": [self.scales[name] for name in names],
            }
        )

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the spreads beside the standard errors.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds which scale a ratio's spread is on.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text, with the parts as data.
        """
        from .._summary import check_detail
        from ._summaries import spread_summary

        return spread_summary(self, check_detail(detail))

    def __repr__(self) -> str:
        from ._summaries import spread_repr

        return spread_repr(self, self.n_repeats)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


def repeat_spread(
    estimates: Mapping[str, ParameterEstimate],
    draws: Sequence[Mapping[str, float]],
) -> RepeatSpread:
    """Split spread of each estimate across the draws, on its inference scale.

    The body of :meth:`cleverly.estimators.TMLEResult.repeat_spread` and
    :meth:`cleverly.longitudinal.LongitudinalResult.repeat_spread`, which document it.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The combined estimates, which give each name's reported scale.
    draws : sequence of mappings
        Each draw's point estimates, keyed by name.

    Returns
    -------
    RepeatSpread
        The standard deviation across the draws, per estimand present in every draw, as a
        mapping that also prints a summary.

    Raises
    ------
    CapabilityError
        When there are fewer than two draws.
    """
    if len(draws) < 2:
        raise CapabilityError(
            "repeat_spread() describes how much psi moved between draws of the "
            f"cross-fitting split, and this fit has {len(draws)}.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Fit with repeats=2 or more.",
        )
    shared = [name for name in estimates if all(name in draw for draw in draws)]
    spreads: dict[str, float] = {}
    for name in shared:
        values = np.asarray([draw[name] for draw in draws], dtype=float)
        # Outside the ratio branch, because a draw that is not finite has no spread on
        # any scale.  Leaving it inside sent an infinite difference-scale draw into
        # ``np.std``, which warns and therefore raises under this project's
        # ``error::RuntimeWarning`` filter rather than yielding the promised ``nan``.
        if not np.all(np.isfinite(values)):
            spreads[name] = float("nan")
            continue
        if estimates[name].scale == "ratio":
            if np.any(values <= 0.0):
                spreads[name] = float("nan")
                continue
            values = np.log(values)
        spreads[name] = float(np.std(values, ddof=1))
    return RepeatSpread(
        spreads,
        n_repeats=len(draws),
        std_errors={name: float(estimates[name].plugin_std_error) for name in spreads},
        inferential={name: bool(estimates[name].supplies_inference) for name in spreads},
        scales={
            name: "log" if estimates[name].scale == "ratio" else "original" for name in spreads
        },
    )


def split_noise_lines(
    estimates: Mapping[str, ParameterEstimate],
    spreads: Mapping[str, float],
    n_repeats: int,
    status: InferenceStatus,
) -> list[str]:
    """The lines a repeated fit's summary prints under its table.

    A note that the report is coordinatewise, then the split spread of each estimand beside
    its plug-in standard error.  Shared by the point-treatment and longitudinal summaries.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The combined estimates.
    spreads : Mapping of str to float
        :func:`repeat_spread` of the fit.
    n_repeats : int
        The number of draws.
    status : str
        The fit's inference status, which names the spread column.

    Returns
    -------
    list of str
        The lines, with an empty line before each block.
    """
    lines = [
        "",
        # Printed under the table rather than left to the scope document, because a
        # reader who subtracts two rows of that table gets a third number the fit
        # never reported.
        "the report above is coordinatewise: each row is the median of its own "
        "draws, so algebraic identities among the rows (ate versus ey1 - ey0, "
        "say) do not hold.",
        "",
        # Beside the standard error rather than in a separate report, because the
        # comparison is the whole content of the number: on its own "0.0065" says
        # nothing about whether the split mattered.
        f"split noise: spread across the {n_repeats} draws on the inference "
        "scale (the log scale for a ratio), a diagnostic and not a standard error:",
    ]
    for name, value in spreads.items():
        # The plug-in accessor: this share is a diagnostic ratio, and the fit may
        # be one whose standard error the package refuses to name as one.
        error = estimates[name].plugin_std_error
        # "-" and a named reason, as every other table in this package renders a
        # cell it has no value for.
        if not np.isfinite(value):
            lines.append(f"  {name:<5s} -  (no spread on the inference scale)")
            continue
        # Named as ``to_frame`` names the same number, so a selector-path fit
        # does not describe its diagnostic as a standard error here.
        label = spread_name("std_err", status)
        share = f"{value / error:.0%} of {label}" if error > 0 else f"{label} unavailable"
        lines.append(f"  {name:<5s} {value:.4g}  ({share})")
    return lines


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
        support=combine_support((estimates[key].support for key in chosen), kind="derived"),
    )


def derived_bootstrap(
    estimates: Mapping[str, ParameterEstimate],
    bootstrap: BootstrapResult | None,
    names: Sequence[str],
    function: Callable[[FloatArray], float],
    *,
    alpha: float,
    derived: ParameterEstimate | None = None,
    cluster: IntArray | None = None,
) -> BootstrapSummary | None:
    """The bootstrap summary of a derived estimate, from its inputs' replicate draws.

    Each bootstrap replicate already holds a point estimate of every reported parameter, in
    aligned rows of :attr:`BootstrapResult.draws`.  Applying ``function`` to each row gives
    the replicate value of the derived estimate, so its interval is the one the full refit
    implies. A replicate with a missing input gives a missing value. Domain-invalid ratios
    also give missing values. The failure count includes every requested replicate without
    a finite derived value. User callable exceptions propagate. The summary is supported
    only when every input's summary is and each row of the inputs lists the ``derived``
    kind.

    The derived summary takes the interval of its inputs. A BCa interval reads its
    acceleration from the derived estimate's influence curve on the scale of its draws
    (the delta-method curve of :func:`smooth_contrast` with a ratio's log and a
    ``transform=`` undone), and its bias correction from the derived point estimate.

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
        Significance level of the interval.
    derived : ParameterEstimate or None, default=None
        The derived estimate. Required when the inputs carry BCa summaries.
    cluster : ndarray or None, default=None
        ``(n,)`` cluster codes of the fit. A BCa acceleration sums the curve within clusters
        when the bootstrap resampled clusters.

    Returns
    -------
    BootstrapSummary or None
        ``None`` when the fit carries no bootstrap, or an input has no draws.

    Raises
    ------
    ValueError
        When the inputs carry BCa summaries and ``derived`` is ``None``, or the inputs mix
        two intervals.
    """
    if bootstrap is None or any(name not in bootstrap.draws for name in names):
        return None
    summaries = [estimates[name].bootstrap for name in names]
    if any(summary is None for summary in summaries):
        return None
    intervals = {summary.interval for summary in summaries if summary is not None}
    if len(intervals) != 1:
        raise ValueError(f"the inputs of a derived bootstrap mix intervals {sorted(intervals)}")
    interval = intervals.pop()
    matrix = np.column_stack([np.asarray(bootstrap.draws[name], dtype=float) for name in names])
    values = np.array(
        [float(function(row)) if np.all(np.isfinite(row)) else np.nan for row in matrix]
    )
    replicates = BootstrapResult(
        draws={"derived": values},
        n_requested=bootstrap.n_requested,
        n_failed=bootstrap.n_requested - int(np.count_nonzero(np.isfinite(values))),
        resampling=bootstrap.resampling,
    )
    if interval == "bca":
        if derived is None:
            raise ValueError("a BCa summary of a derived estimate needs the derived estimate")
        summary = replicates.interval(
            "derived",
            alpha,
            interval="bca",
            estimate=derived.psi,
            acceleration=bca_acceleration(
                _reported_curve(derived), cluster if bootstrap.resampling == "cluster" else None
            ),
        )
    else:
        summary = replicates.interval("derived", alpha)
    kept = [item for item in summaries if item is not None]
    return replace(
        summary,
        diagnostic=any(item.diagnostic for item in kept),
        below_floor=any(item.below_floor for item in kept),
        support=combine_support([item.support for item in kept], kind="derived"),
    )


def with_derived_bootstrap(
    derived: ParameterEstimate,
    estimates: Mapping[str, ParameterEstimate],
    bootstrap: BootstrapResult | None,
    names: Sequence[str],
    function: Callable[[FloatArray], float],
    *,
    cluster: IntArray | None = None,
) -> ParameterEstimate:
    """``derived`` with the bootstrap summary :func:`derived_bootstrap` gives, if any."""
    summary = derived_bootstrap(
        estimates,
        bootstrap,
        names,
        function,
        alpha=derived.alpha,
        derived=derived,
        cluster=cluster,
    )
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
            f"{family!r}.",
            reason=RefusalReason.API_REDIRECTION,
            alternative='Use kind="rr" for a ratio of means.',
        )


class _BootstrappableResult(Protocol):
    @property
    def estimates(self) -> Mapping[str, ParameterEstimate]: ...

    @property
    def config(self) -> Any: ...


_ResultT = TypeVar("_ResultT", bound=_BootstrappableResult)


def attach_bootstrap(
    result: _ResultT,
    bootstrap: BootstrapResult,
    *,
    diagnostic: bool = False,
    classification: Classification | None = None,
    interval: BootstrapInterval = "percentile",
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
    diagnostic : bool, default=False
        Whether the fit declares its bootstrap interval a diagnostic. Stored on each
        :class:`~cleverly.inference.BootstrapSummary`.
    classification : Classification or None, default=None
        The fit's classification. Each summary takes the support that
        :func:`~cleverly._support.support_for` gives for the interval's kind: ``"bootstrap"``
        for a percentile interval and ``"bootstrap_bca"`` for a BCa interval. ``None``
        leaves it unclassified.
    interval : {"percentile", "bca"}, default="percentile"
        The pointwise interval. A BCa interval reads each estimate's acceleration from its
        own influence curve (:func:`~cleverly.inference.bootstrap.bca_acceleration`), summed within
        clusters when the bootstrap resamples clusters.

    Returns
    -------
    TMLEResult or LongitudinalResult
        A copy whose estimates carry their bootstrap summaries.
    """
    alpha = result.config.alpha_sig
    below_floor = classification is not None and classification.key.dependence == "below_floor"
    kind: Kind = "bootstrap_bca" if interval == "bca" else "bootstrap"
    # A cluster bootstrap resamples clusters, so its acceleration reads the cluster sums.
    unit = (
        getattr(getattr(result, "data", None), "cluster", None)
        if bootstrap.resampling == "cluster"
        else None
    )

    def summary(name: str, estimate: ParameterEstimate) -> BootstrapSummary:
        if interval == "bca":
            drawn = bootstrap.interval(
                name,
                alpha,
                interval="bca",
                estimate=estimate.psi,
                acceleration=bca_acceleration(estimate.influence_curve, unit),
            )
        else:
            drawn = bootstrap.interval(name, alpha)
        return replace(
            drawn,
            diagnostic=diagnostic,
            below_floor=below_floor,
            support=support_for(
                classification, estimate.inference, estimate.reference_df, kind=kind
            ),
        )

    estimates = {
        name: (
            estimate.with_bootstrap(summary(name, estimate))
            if name in bootstrap.draws
            else estimate
        )
        for name, estimate in result.estimates.items()
    }
    return replace(result, estimates=estimates, bootstrap=bootstrap)  # type: ignore[type-var]
