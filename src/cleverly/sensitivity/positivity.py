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
from ..data.weighting import effective_sample_size
from ..estimators.direct_effect import targeted_rows
from ..estimators.targeting import build_submodel
from ..exceptions import DataError
from ..inference.influence import average_estimates
from ..targets import parameter_stem
from ..utils.bounds import g_bounds_for
from ..utils.frames import emit_frame
from ..utils.text import format_table

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
    simplex_deviation:
        Largest ``|sum_a g(a | W) - 1|`` across rows *after* truncation, and ``0`` for a
        two-armed fit, where the complement form preserves the sum exactly.

        Non-zero is expected rather than alarming: with more than two arms the bounds are
        applied arm by arm and deliberately **not** renormalised, because rescaling a row
        back onto the simplex can push a column below the floor and so undo the only
        thing truncation is for.  The number is reported because it is the size of that
        deliberate inconsistency, and a large value says the bounds are binding hard --
        which is a positivity finding, not a bookkeeping one.  It does not bias the
        plug-in: the plug-in averages targeted predictions and contains no mechanism at
        all.
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
    simplex_deviation: float = 0.0
    #: How many cross-fitting draws the fit averaged over.  Everything above describes the
    #: **first** of them, and this is here so a reader knows that.  Overlap is a property
    #: of one fitted mechanism, and averaging ``R`` propensity vectors would produce a
    #: perfectly good estimate of ``g`` that is nonetheless not the object any reported
    #: ``psi`` was computed from -- a different aggregation from the one the estimates use,
    #: under the same heading.  The draws share the data and differ only in the split, so
    #: their overlap is near identical in practice; when it is not, that is itself worth
    #: seeing rather than averaging away.
    n_repeats: int = 1
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None

    def to_frame(self, data: Any = None) -> Any:
        """Propensity quantiles as a tidy frame."""
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
        return emit_frame(payload, data, backend=self.backend)

    def summary(self) -> str:
        """A printable overlap report."""
        lines = [
            "Positivity / overlap diagnostics",
            "-" * 32,
            f"n = {self.n}; propensity truncated to [{self.bounds[0]:.4g}, {self.bounds[1]:.4g}]",
        ]
        if self.n_repeats > 1:
            lines.append(
                f"describing draw 1 of {self.n_repeats}: overlap is a property of one "
                "fitted mechanism, not of the averaged estimate"
            )
        lines.append("")
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
    """Compute overlap diagnostics for a fitted TMLE.

    Two arms and more than two are reported by separate functions rather than one
    parameterised by arm count, because the *questions* differ.  With two arms there is
    one propensity and overlap is symmetric: ``g`` near zero and ``g`` near one are the
    same problem seen from either arm, and the interesting split is treated versus
    control.  With more there is no single margin and no mirror -- each arm has its own
    denominator, which has to be reported and truncated in its own right.  Collapsing
    the two into one function would mean picking definitions that read oddly in both.

    A **continuous** treatment is refused rather than given a third branch.  Every field
    of :class:`PositivityReport` is per arm -- quantiles of ``g(a | W)``, tail mass,
    effective sample size, weight share -- and a dose has none, so the report would come
    back empty with a ``simplex_deviation`` of ``1.0`` computed from a zero-column
    mechanism: the largest value the field can take, reported as a finding.  The question
    a shift fit actually has to answer is about the *density ratio* at the shifted dose,
    which :func:`~cleverly.interventions.check_shift_support` answers.
    """
    if result.data.is_continuous_treatment:
        raise DataError(
            f"{result.data.treatment_name} is continuous, so there is no per-arm "
            "propensity to tabulate and this report has no rows to fill. A shift's "
            "positivity question is whether the density ratio g(a - delta | W) / "
            "g(a | W) stays bounded, not whether an arm probability does -- use "
            "res.sensitivity.shift_support(). With delta= or intermediate= there is a "
            "mechanism in the denominator as well, and it is the one bound this axis "
            "actually has: res.sensitivity.truncation_curve(mechanism=True) sweeps it."
        )
    if result.data.is_binary_treatment:
        return _binary_positivity_report(result)
    return _multi_arm_positivity_report(result)


def _binary_positivity_report(result: TMLEResult) -> PositivityReport:
    """Overlap for a two-armed treatment, in terms of the single propensity ``g(W)``."""
    data = result.data
    bounds = result.config.g_bounds
    raw = result.nuisance.propensity.arm(1.0)
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
        n_repeats=result.n_repeats,
        backend=data.backend,
    )


def _multi_arm_positivity_report(result: TMLEResult) -> PositivityReport:
    r"""Overlap for a ``K``-armed treatment, arm by arm.

    Each arm's probability :math:`g_a(W)` is summarised over **all** rows, not just the
    rows in that arm.  That is the distribution positivity actually depends on: the
    clever covariate divides by :math:`g_a` and the plug-in evaluates
    :math:`\bar Q(a, W)` at every unit, so a unit that could never have received arm
    ``a`` is a problem for arm ``a`` whichever arm it did receive.

    ``tail_mass`` loses its mirror here.  With two arms ``g > 1 - t`` is the same
    statement as ``g < t`` read from the control arm; with more arms an arm being
    *likely* is not another arm being unlikely, so ``below`` counts unit-arm pairs whose
    probability sits under the threshold and ``above`` counts the unit-arm pairs that are
    nearly deterministic -- related, but no longer the same number counted twice.
    """
    data = result.data
    bounds = result.config.g_bounds
    propensity = result.nuisance.propensity
    raw = np.asarray(propensity.values, dtype=float)
    labels = {arm: str(data.arm_label(arm)) for arm in propensity.arms}

    quantiles: dict[str, dict[float, float]] = {
        f"g[{labels[arm]}]": {q: float(np.quantile(propensity.arm(arm), q)) for q in _QUANTILES}
        for arm in propensity.arms
    }

    tail_mass = {
        threshold: {
            "below": float(np.mean(raw < threshold)),
            "above": float(np.mean(raw > 1.0 - threshold)),
        }
        for threshold in _THRESHOLDS
    }

    bounded = propensity.bounded(bounds)
    ess: dict[str, dict[str, float]] = {}
    share: dict[str, dict[str, float]] = {}
    for arm in propensity.arms:
        mask = data.treatment == arm
        column = bounded[:, propensity.column_for(arm)]
        arm_weights = (1.0 / column)[mask] * data.weights[mask]
        nominal = float(mask.sum())
        ess[labels[arm]] = {
            "n": nominal,
            "effective": _kish_ess(arm_weights),
            "ratio": _kish_ess(arm_weights) / nominal if mask.any() else float("nan"),
        }
        share[labels[arm]] = {
            "top_1pct": _top_share(arm_weights, 0.01),
            "top_5pct": _top_share(arm_weights, 0.05),
        }

    clipped_cells = (raw < bounds[0]) | (raw > bounds[1])
    clipped_units = np.any(clipped_cells, axis=1)
    inside = raw[~clipped_cells]
    # Only the approach to zero matters per arm, so the "most extreme" untruncated value
    # is the smallest surviving probability rather than the two-sided minimum the binary
    # report uses.
    most_extreme = float(inside.min()) if inside.size else float("nan")

    return PositivityReport(
        propensity_quantiles=quantiles,
        tail_mass=tail_mass,
        effective_sample_size=ess,
        weight_share=share,
        truncated={
            "count": float(clipped_units.sum()),
            "fraction": float(clipped_units.mean()),
            "most_extreme": most_extreme,
        },
        clever_covariate_max={
            group: _max_abs_covariate(result, group) for group in result.fluctuations
        },
        bounds=bounds,
        n=data.n,
        mechanisms=_mechanism_overlap(result),
        nuisance_bound=result.config.missingness_bound,
        simplex_deviation=float(np.max(np.abs(bounded.sum(axis=1) - 1.0))),
        n_repeats=result.n_repeats,
        backend=data.backend,
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

    Which rows those are depends on the estimand.  A row with no recorded outcome
    contributes a genuine zero to the residual term, and so does a row whose intermediate
    is not the level being targeted; neither is weighted by any mechanism, so neither
    belongs in an effective sample size.  For a controlled direct effect that is a real
    difference rather than a technicality -- roughly half the sample is typically at the
    other level, and averaging it in would report an ESS for a weighting that never
    happened.
    """
    data = result.data
    nuisance = result.nuisance
    bound = result.config.missingness_bound
    treated = data.treatment == 1.0
    contributing = targeted_rows(data, result.intermediate_value)
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
        used = np.maximum(at_arm[contributing], bound)
        out[name] = {
            "min": float(flat.min()),
            "q01": float(np.quantile(flat, 0.01)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.median(flat)),
            "ess_ratio": (_kish_ess(1.0 / used) / float(used.size) if used.size else float("nan")),
            "clipped": float(clipped.sum()),
            "clipped_fraction": float(clipped.mean()),
        }

    # The product is a derived denominator, not a third fitted or targeted mechanism.
    # Report it without storing it on the nuisance state so the treatment and observation
    # probabilities cannot become stale relative to a cached product.
    missing_reduction = getattr(nuisance.reduced, "reduction", None) == "missing_outcome"
    if missing_reduction and nuisance.missingness is not None:
        lower = result.config.g_bounds[0] * result.config.missingness_bound
        array = np.asarray(nuisance.propensity.values, dtype=float) * np.asarray(
            nuisance.missingness, dtype=float
        )
        flat = array.reshape(-1)
        clipped = flat < lower
        at_arm = np.where(treated, array[:, 1], array[:, 0])
        used = np.maximum(at_arm[contributing], lower)
        out["P(A=a,Delta=1|W)"] = {
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
    """Kish's ESS, answering ``nan`` for an arm with no rows rather than zero.

    The distinction is this module's own and is why it still has a wrapper: an arm nothing
    was selected for has no effective sample size to report, while an arm whose weights
    sum to zero has one and it is zero.  The formula itself is
    :func:`~cleverly.data.weighting.effective_sample_size`.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    return effective_sample_size(w, on_degenerate=0.0)


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
    """Largest absolute clever-covariate value for one targeted family.

    Rebuilt from the data, the nuisance estimates and the config rather than from the
    estimator, so this stays a real number on a result whose estimator is gone.
    """
    bounds = g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)
    submodel = build_submodel(
        result.data,
        result.nuisance,
        group,
        bounds=bounds,
        nuisance_bound=result.config.missingness_bound,
        intermediate_value=result.intermediate_value,
        # The fit's own reference, so a conditional-effect covariate is rebuilt with the
        # contrasts it was targeted with rather than with the lowest arm's.
        reference=result.config.reference_arm,
        # A working model with a non-identity link has a covariate that reads its own
        # coefficients, so "the covariate" is only defined once they are named. The ones
        # the fit *reports* at are the right choice and the only defensible one: they are
        # the equation this fit solved, and under fold-wise targeting they are the single
        # beta a per-fold covariate has no other summary of.
        msm_beta=_reported_beta(result, group),
    )
    return submodel.max_abs


def _reported_beta(result: TMLEResult, group: str) -> Any:
    """The working model's coefficients, or ``None`` where the covariate does not read them."""
    projection = getattr(result.fluctuations.get(group), "projection", None)
    return None if projection is None else projection.beta


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

    # `retarget` takes *target* names, while `result.estimates` is keyed by the parameter
    # names those targets reported -- the same thing for a two-armed fit, and `ey[high]`
    # against `ey` for a wider one. Mapping back through the stem keeps the sweep asking
    # for the targets it already has, rather than for names the registry never had.
    reported = tuple(result.estimates) if estimands is None else tuple(estimands)
    names = tuple(dict.fromkeys(parameter_stem(name) for name in reported))
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
        # Every draw, then averaged the way the fit averaged them. Sweeping one draw and
        # calling the answer the fit's would compare a bound's effect on one split against
        # a reported estimate that came from R -- and the difference between the two curves
        # would read as sensitivity to the bound. Costs R times the sweep, which is still a
        # fraction of one refit.
        estimates = average_estimates(
            [
                estimator.retarget(
                    result.data,
                    repeat.nuisance,
                    estimands=names,
                    intermediate_value=result.intermediate_value,
                    g_bounds=None if mechanism else pair,
                    g_bounds_conditional=None if mechanism else pair,
                    nuisance_bound=lower if mechanism else None,
                )[0]
                for repeat in result.repeats
            ],
            cluster=result.data.cluster,
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
        divisor = np.asarray(result.nuisance.propensity.values, dtype=float)
        return float(np.mean((divisor < lower) | (divisor > 1.0 - lower)))
    # The intermediate entry must be the density for the level being targeted, not the
    # raw P(Z = 1 | A, W): at z = 0 the covariate divides by the complement, so reading
    # the array directly counts the wrong tail and reports a mirror-inverted fraction.
    # This column is not a nicety -- it is the only part of the mechanism truncation
    # curve that can detect a q_z positivity problem at all.  A density that is clipped
    # to a constant rescales both clever-covariate columns by the same factor, and the
    # fluctuation ``epsilon * h`` is invariant to that, so ``psi`` sits flat across the
    # whole sweep however badly the bound is binding.
    candidates = [result.nuisance.missingness]
    if result.nuisance.intermediate is not None and result.intermediate_value is not None:
        candidates.append(result.nuisance.intermediate_density(result.intermediate_value, 0.0))
    parts = [
        np.asarray(values, dtype=float).reshape(-1) for values in candidates if values is not None
    ]
    return float(np.mean(np.concatenate(parts) < lower))
