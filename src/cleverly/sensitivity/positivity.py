r"""Positivity and overlap diagnostics.

Positivity -- every unit having some chance of either treatment given ``W`` -- is the
assumption that fails quietly.  Unlike confounding, it leaves a visible fingerprint
in the estimated propensity scores, and unlike model misspecification it is not
fixed by a better learner: if no treated unit resembles a given control unit, no
estimator can say what would have happened to that control unit under treatment.

What to look at, in order of how much it tells you:

**Effective sample size.**  The clever covariate reweights the sample.  Kish's
effective sample size, :math:`(\sum_i \omega_i)^2 / \sum_i \omega_i^2` for
:math:`\omega_i = 1/g(W_i)` in the treated arm, says how many observations the
weighted analysis is really using.  An ESS of 40 out of 500 treated units means the
estimate rests on a small effective subsample, whatever the nominal ``n`` says.

**Weight concentration.**  The share of the estimating equation contributed by the
largest few weights.  If the top 1% of units carry 30% of it, the estimate is a
statement about those units.

**Truncation load.**  How many propensity scores were clipped, and how far.  Every
clipped unit contributes bias in exchange for variance; a large clipped fraction
means the reported estimate is not the estimand that was asked for.

Observation weights are folded into :math:`\omega_i` rather than reported separately,
because the two costs multiply: a design that halves the effective sample size and a
clever covariate that halves it again leave a quarter, and a diagnostic that showed only
one of them would look comfortable.  For the weighting cost on its own -- and for the
estimand statement that goes with it -- see
:meth:`~cleverly.data.CausalData.weight_report` and :mod:`cleverly.data.weighting`.

Use :func:`truncation_curve` to see how much the answer actually moves as the
truncation bound changes -- a flat curve is reassuring in a way that no single
diagnostic can be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray
from ..estimators.base import format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["PositivityReport", "positivity_report", "truncation_curve"]

#: Quantiles reported for the propensity distribution in each arm.
_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

#: Thresholds at which the mass of extreme propensity scores is reported.
_THRESHOLDS = (0.01, 0.025, 0.05, 0.1)


@dataclass(frozen=True)
class PositivityReport:
    """Overlap diagnostics for a fitted TMLE.

    Attributes
    ----------
    propensity_quantiles:
        Quantiles of ``g(W)``, overall and within each treatment arm.
    tail_mass:
        Fraction of units with ``g(W)`` below each threshold, and above its mirror.
    effective_sample_size:
        Kish ESS of the inverse-probability weights per arm, with the nominal arm
        size alongside for comparison.
    weight_share:
        Share of the total weight held by the largest 1% and 5% of weights.
    truncated:
        Count and fraction of propensity scores clipped by the truncation bounds, and
        the most extreme untruncated value.
    clever_covariate_max:
        Largest absolute clever-covariate value per targeted estimand family -- the
        single most direct summary of how much one observation can move the estimate.
    mechanisms:
        Overlap for the *other* denominators in the clever covariate:
        ``P(Delta = 1 | A, W)`` when outcomes are missing, and ``P(Z = z | A, W)`` for a
        controlled direct effect.  Each carries the smallest and lowest-quantile value,
        how many rows the ``nuisance_bound`` clipped, and ``ess_ratio`` -- the Kish
        effective sample size the ``1 / mechanism`` weights leave behind, on the same
        scale as the propensity's, so the two can be read side by side.  Empty when
        neither applies.

        These deserve reporting for exactly the reason ``g`` does: they enter the
        estimating equation as a denominator, so a value near zero gives one observation
        unbounded leverage.  Unlike ``g`` they are one-sided -- only the approach to zero
        matters -- and they are easy to overlook, because a fit can have perfectly
        healthy propensity overlap and still be resting on a handful of rows that were
        very unlikely to be observed at all.
    nuisance_bound:
        The lower bound applied to those mechanisms.
    """

    propensity_quantiles: dict[str, dict[float, float]]
    tail_mass: dict[float, dict[str, float]]
    effective_sample_size: dict[str, dict[str, float]]
    weight_share: dict[str, dict[str, float]]
    truncated: dict[str, float]
    clever_covariate_max: dict[str, float]
    bounds: tuple[float, float]
    n: int
    mechanisms: dict[str, dict[str, float]] = field(default_factory=dict)
    nuisance_bound: float = 0.0

    def to_frame(self, data: Any = None) -> Any:
        """Propensity quantiles as a tidy frame."""
        from ..utils.frames import frame_from_dict

        rows: list[tuple[str, float, float]] = [
            (group, quantile, value)
            for group, quantiles in self.propensity_quantiles.items()
            for quantile, value in quantiles.items()
        ]
        payload = {
            "group": [row[0] for row in rows],
            "quantile": [row[1] for row in rows],
            "propensity": [row[2] for row in rows],
        }
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def summary(self) -> str:
        """A printable overlap report."""
        lines = [
            "Positivity / overlap diagnostics",
            "-" * 32,
            f"n = {self.n}; propensity truncated to [{self.bounds[0]:.4g}, {self.bounds[1]:.4g}]",
            "",
        ]
        quantiles = sorted(next(iter(self.propensity_quantiles.values())))
        lines.append(
            format_table(
                ["g(W) quantile", *[f"{q:.0%}" for q in quantiles]],
                [
                    [group, *[f"{values[q]:.4f}" for q in quantiles]]
                    for group, values in self.propensity_quantiles.items()
                ],
            )
        )
        lines.append("")
        lines.append(
            format_table(
                ["arm", "n", "effective n", "ESS / n", "top 1% weight", "top 5% weight"],
                [
                    [
                        arm,
                        f"{ess['n']:.0f}",
                        f"{ess['effective']:.1f}",
                        f"{ess['ratio']:.3f}",
                        f"{self.weight_share[arm]['top_1pct']:.3f}",
                        f"{self.weight_share[arm]['top_5pct']:.3f}",
                    ]
                    for arm, ess in self.effective_sample_size.items()
                ],
            )
        )
        lines.append("")
        lines.append(
            format_table(
                ["threshold", "P(g < t)", "P(g > 1-t)"],
                [
                    [f"{threshold:.3g}", f"{mass['below']:.4f}", f"{mass['above']:.4f}"]
                    for threshold, mass in sorted(self.tail_mass.items())
                ],
            )
        )
        lines.append("")
        lines.append(
            f"truncated: {self.truncated['count']:.0f} unit(s) "
            f"({self.truncated['fraction']:.2%}); most extreme untruncated g(W) = "
            f"{self.truncated['most_extreme']:.5g}"
        )
        for group, value in self.clever_covariate_max.items():
            lines.append(f"max |clever covariate| ({group}): {value:.4g}")
        if self.mechanisms:
            lines.append("")
            lines.append(
                format_table(
                    ["mechanism", "min", "1%", "5%", "median", "ESS / n", "clipped"],
                    [
                        [
                            name,
                            f"{stats['min']:.4f}",
                            f"{stats['q01']:.4f}",
                            f"{stats['q05']:.4f}",
                            f"{stats['median']:.4f}",
                            f"{stats['ess_ratio']:.3f}",
                            f"{stats['clipped']:.0f} ({stats['clipped_fraction']:.2%})",
                        ]
                        for name, stats in self.mechanisms.items()
                    ],
                )
            )
            lines.append(
                f"(truncated to [{self.nuisance_bound:.4g}, 1]; each row counts both arms)"
            )
        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)

    def verdict(self) -> str:
        """A one-line reading of the diagnostics."""
        worst_ratio = min(ess["ratio"] for ess in self.effective_sample_size.values())
        fraction = self.truncated["fraction"]
        for name, stats in self.mechanisms.items():
            # Checked before the propensity verdict, because this is the failure a reader
            # is least likely to be looking for: overlap in `g` can be immaculate while
            # the estimate rests on a few rows that were very unlikely to be observed.
            # Judged on the same scale as the propensity -- the effective sample size the
            # 1/mechanism weights leave behind -- so the two are directly comparable.
            if stats["clipped_fraction"] > 0.01 or stats["ess_ratio"] < 0.6:
                return (
                    f"VERDICT: {name} strains the estimate. It falls to {stats['min']:.4g} at "
                    f"its smallest and leaves an effective {stats['ess_ratio']:.0%} of the "
                    f"rows it weights ({stats['clipped_fraction']:.2%} clipped at "
                    f"{self.nuisance_bound:.4g}). It divides the clever covariate exactly as "
                    "g(W) does, so those rows carry outsized leverage whatever the propensity "
                    "overlap looks like. Check truncation_curve(mechanism=True)."
                )
        if fraction > 0.05 or worst_ratio < 0.3:
            return (
                "VERDICT: serious positivity problem. The weighted analysis uses far fewer "
                "observations than it appears to, and/or a large share of units were "
                "truncated. Treat the point estimate as describing the region of overlap "
                "only, and check truncation_curve() before drawing conclusions."
            )
        if fraction > 0.01 or worst_ratio < 0.6:
            return (
                "VERDICT: some positivity strain. Report truncation_curve() alongside the "
                "estimate so readers can see how much the answer depends on the bound."
            )
        return "VERDICT: overlap looks adequate; no truncation-driven fragility detected."

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def positivity_report(result: TMLEResult) -> PositivityReport:
    """Compute overlap diagnostics for a fitted TMLE."""
    data = result.data
    bounds = result.config.g_bounds
    raw = result.nuisance.propensity
    treated = data.treatment == 1.0

    quantiles: dict[str, dict[float, float]] = {
        "overall": {q: float(np.quantile(raw, q)) for q in _QUANTILES},
        "treated": {q: float(np.quantile(raw[treated], q)) for q in _QUANTILES},
        "control": {q: float(np.quantile(raw[~treated], q)) for q in _QUANTILES},
    }

    tail_mass = {
        threshold: {
            "below": float(np.mean(raw < threshold)),
            "above": float(np.mean(raw > 1.0 - threshold)),
        }
        for threshold in _THRESHOLDS
    }

    bounded = np.clip(raw, bounds[0], bounds[1])
    ess: dict[str, dict[str, float]] = {}
    share: dict[str, dict[str, float]] = {}
    for arm, mask, weights in (
        ("treated", treated, 1.0 / bounded),
        ("control", ~treated, 1.0 / (1.0 - bounded)),
    ):
        arm_weights = weights[mask] * data.weights[mask]
        ess[arm] = {
            "n": float(mask.sum()),
            "effective": _kish_ess(arm_weights),
            "ratio": _kish_ess(arm_weights) / float(mask.sum()) if mask.any() else float("nan"),
        }
        share[arm] = {
            "top_1pct": _top_share(arm_weights, 0.01),
            "top_5pct": _top_share(arm_weights, 0.05),
        }

    clipped = (raw < bounds[0]) | (raw > bounds[1])
    inside = raw[~clipped]
    most_extreme = float(min(inside.min(), 1.0 - inside.max())) if inside.size else float("nan")

    return PositivityReport(
        propensity_quantiles=quantiles,
        tail_mass=tail_mass,
        effective_sample_size=ess,
        weight_share=share,
        truncated={
            "count": float(clipped.sum()),
            "fraction": float(clipped.mean()),
            "most_extreme": most_extreme,
        },
        clever_covariate_max={
            group: _max_abs_covariate(result, group) for group in result.fluctuations
        },
        bounds=bounds,
        n=data.n,
        mechanisms=_mechanism_overlap(result),
        nuisance_bound=result.config.missingness_bound,
    )


def _mechanism_overlap(result: TMLEResult) -> dict[str, dict[str, float]]:
    """Overlap for the denominators other than ``g`` -- ``pi`` and the intermediate density.

    Two views, because they answer different questions.  The quantiles pool both arms,
    since both columns are used: the treated arm's covariate divides by the mechanism at
    ``a = 1`` and the control arm's by the one at ``a = 0``, so the union is the set of
    values that appear as denominators anywhere.  The effective sample size instead takes
    the weights the estimating equation *actually* forms -- ``1 / pi`` at each unit's
    realised arm, over the rows whose residual it multiplies -- and is reported on the
    same scale as the propensity's ESS so that the two can be read side by side.
    """
    data = result.data
    nuisance = result.nuisance
    bound = result.config.missingness_bound
    treated = data.treatment == 1.0
    out: dict[str, dict[str, float]] = {}

    candidates: list[tuple[str, Any]] = [("P(Delta=1|A,W)", nuisance.missingness)]
    if nuisance.intermediate is not None and result.intermediate_value is not None:
        # The covariate divides by P(Z = z | A, W) for the targeted z, which is the
        # fitted probability or its complement -- so report the one actually used.
        candidates.append(
            (
                f"P(Z={result.intermediate_value:.0f}|A,W)",
                nuisance.intermediate_density(result.intermediate_value, 0.0),
            )
        )

    for name, values in candidates:
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        flat = array.reshape(-1)
        clipped = flat < bound
        # The weight the estimating equation forms: the mechanism at the realised arm,
        # on the rows whose residual term it multiplies.  Rows with no outcome contribute
        # a genuine zero to that term, so they are not weighted by it and do not belong
        # in its effective sample size.
        at_arm = np.where(treated, array[:, 1], array[:, 0])
        used = np.maximum(at_arm[data.observed], bound)
        out[name] = {
            "min": float(flat.min()),
            "q01": float(np.quantile(flat, 0.01)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.median(flat)),
            "ess_ratio": (_kish_ess(1.0 / used) / float(used.size) if used.size else float("nan")),
            "clipped": float(clipped.sum()),
            "clipped_fraction": float(clipped.mean()),
        }
    return out


def _kish_ess(weights: FloatArray) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / np.sum(w**2))


def _top_share(weights: FloatArray, fraction: float) -> float:
    """Share of total weight held by the largest ``fraction`` of units."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(fraction * w.size)))
    largest = np.sort(w)[-count:]
    total = w.sum()
    return float(largest.sum() / total) if total > 0 else float("nan")


def _max_abs_covariate(result: TMLEResult, group: str) -> float:
    """Largest absolute clever-covariate value for one targeted family."""
    estimator = result.estimator
    if estimator is None:  # pragma: no cover - only for hand-built results
        return float("nan")
    bounds = result.config.g_bounds if group == "mean" else result.config.g_bounds_conditional
    submodel = estimator._submodel(
        result.data, result.nuisance, group, bounds, result.intermediate_value, None
    )
    return submodel.max_abs


def truncation_curve(
    result: TMLEResult,
    bounds: Any = None,
    *,
    estimands: Any = None,
    mechanism: bool = False,
) -> Any:
    """Re-estimate across a grid of truncation bounds.

    Returns a tidy frame with one row per ``(bound, estimand)`` giving the point
    estimate and confidence interval.  Only the targeting step is re-run -- the
    nuisance fits are cached -- so a 10-point curve costs a small fraction of the
    original fit.

    A curve that is flat over the plausible range of bounds says the estimate does not
    hinge on the truncation choice.  A curve that drifts monotonically says the
    estimand being reported changes with the bound, and the bound should be reported
    with the estimate.

    Parameters
    ----------
    bounds:
        Lower truncation values to try.  ``None`` uses a grid from 0.001 to 0.2 that
        includes the bound the fit actually used.
    estimands:
        Restrict to a subset; defaults to everything the fit reported.
    mechanism:
        Sweep the bound on ``P(Delta = 1 | A, W)`` (and the intermediate density)
        instead of the one on ``g(W)``.  That probability divides the clever covariate
        exactly as the propensity does, so it has a truncation curve for exactly the
        same reason -- and it is the one that goes unexamined, because it has no
        familiar name.  Requires a fit with ``delta=`` or ``intermediate=``.

        Note what the curve does and does not show.  Truncating a mechanism cannot move
        the *estimand*: the plug-in is an average of targeted predictions and contains
        no mechanism at all.  What moves is the second-order remainder, so a curve that
        drifts is saying the estimate is leaning on rows the bound is holding up.
    """
    estimator = result.estimator
    if estimator is None:
        raise ValueError("truncation_curve needs the fitted estimator that produced the result")

    names = tuple(result.estimates) if estimands is None else tuple(estimands)
    if mechanism:
        if result.nuisance.missingness is None and result.nuisance.intermediate is None:
            raise ValueError(
                "mechanism=True needs a fit with missing outcomes or an intermediate "
                "variable; without one there is no mechanism in the clever covariate to "
                "truncate. Pass delta=<column> or intermediate=<column> to fit()."
            )
        fitted_bound = result.config.missingness_bound
    else:
        fitted_bound = result.config.g_bounds[0]

    if bounds is None:
        grid = sorted({0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, round(fitted_bound, 6)})
    else:
        grid = sorted(float(value) for value in bounds)

    rows: list[dict[str, Any]] = []
    for lower in grid:
        if not 0.0 < lower < 0.5:
            raise ValueError(f"truncation bounds must lie in (0, 0.5); got {lower}")
        pair = (lower, 1.0 - lower)
        estimates, _ = estimator.retarget(
            result.data,
            result.nuisance,
            estimands=names,
            intermediate_value=result.intermediate_value,
            g_bounds=None if mechanism else pair,
            g_bounds_conditional=None if mechanism else pair,
            nuisance_bound=lower if mechanism else None,
        )
        for name, estimate in estimates.items():
            low, high = estimate.ci
            rows.append(
                {
                    "bound": lower,
                    "estimand": name,
                    "psi": estimate.psi,
                    "std_err": estimate.std_error,
                    "ci_lower": low,
                    "ci_upper": high,
                    "truncated_fraction": _clipped_fraction(result, lower, mechanism),
                    "is_fitted_bound": bool(np.isclose(lower, fitted_bound, atol=1e-9)),
                }
            )

    payload = {key: [row[key] for row in rows] for key in rows[0]}
    return result.data.frame_like(payload)


def _clipped_fraction(result: TMLEResult, lower: float, mechanism: bool) -> float:
    """Share of nuisance values the bound would clip, for whichever bound is swept."""
    if not mechanism:
        propensity = result.nuisance.propensity
        return float(np.mean((propensity < lower) | (propensity > 1.0 - lower)))
    parts = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in (result.nuisance.missingness, result.nuisance.intermediate)
        if values is not None
    ]
    return float(np.mean(np.concatenate(parts) < lower))
