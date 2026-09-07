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

**Truncation load.**  How many propensity scores were clipped, and how far.  Truncation
trades variance for potential second-order bias.  It regularises the finite-sample
procedure, but it does not change the requested estimand.  A large clipped fraction
means that the estimate relies on extrapolation.

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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray
from ..data.weighting import effective_sample_size, top_weight_share
from ..estimators.direct_effect import targeted_rows
from ..estimators.targeting import build_submodel
from ..exceptions import CapabilityError, DataError
from ..inference.influence import median_estimates
from ..targets import TARGETS, parameter_stem
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

#: The derived denominators the clever covariate divides by: ``g_a(W) pi_a(W)`` when the
#: outcome can be missing, ``g_a(W) q_z(a, W)`` for a controlled direct effect, and the
#: three-factor product when a controlled direct effect also has missing outcomes.  Each is
#: a *product of separately truncated factors* rather than a fitted or targeted mechanism of
#: its own, which is why they are named once here: the report has to treat them differently
#: from the rows that were held to a single bound.
JOINT_MECHANISM = "P(A=a,Delta=1|W)"
JOINT_INTERMEDIATE = "P(A=a,Z=z|W)"
JOINT_MISSING_INTERMEDIATE = "P(A=a,Delta=1,Z=z|W)"

#: Which derived row a fit reports, keyed by ``(an observation mechanism was fitted, an
#: intermediate density was)``.  The two derived facts below come from this one table, so a
#: further composition is one entry rather than three edits that have to agree.
_COMPOSED_ROWS: dict[tuple[bool, bool], str] = {
    (True, False): JOINT_MECHANISM,
    (False, True): JOINT_INTERMEDIATE,
    (True, True): JOINT_MISSING_INTERMEDIATE,
}

#: How many ``[nuisance_bound, 1]`` factors stand beside ``g`` in each derived row.
_COMPOSED_NUISANCE_FACTORS = {name: sum(key) for key, name in _COMPOSED_ROWS.items()}

_COMPOSED_MECHANISMS = frozenset(_COMPOSED_ROWS.values())

#: The target groups whose clever covariate really divides by ``g_a(W) pi_a(W) q_z(a, W)``
#: -- :func:`~cleverly.fluctuation.submodel.mean_submodel`, ``regime_submodel`` and
#: ``msm_submodel`` all build ``1 / (g * pi * pz)`` arm by arm.  The rest do not, and a
#: derived row would name a denominator they never form: ``att`` and ``atc`` divide by
#: ``P(A = a)`` in place of ``g_a`` and reweight the reference arm by the propensity odds,
#: ``ipsi`` discards the propensity outright, and ``mtp`` divides by a density ratio.
#: Those three also read ``g_bounds_conditional`` where it applies, which is a second bound
#: this row does not quote.  A fit that targets none of the groups below therefore reports
#: its factor rows and no product.
_COMPOSED_GROUPS = frozenset({"mean", "regime", "msm"})


@dataclass(frozen=True)
class PositivityReport:
    """Overlap diagnostics for a fitted TMLE.

    Parameters
    ----------
    propensity_quantiles : dict of str to dict of float to float
        Quantiles of ``g(W)``, overall and within each treatment arm.
    tail_mass : dict of float to dict of str to float
        Fraction of units with ``g(W)`` below each threshold, and above its mirror.
    effective_sample_size : dict of str to dict of str to float
        Kish ESS of the inverse-probability weights per arm, with the nominal arm
        size alongside for comparison.
    weight_share : dict of str to dict of str to float
        Share of the total weight held by the largest 1% and 5% of weights.
    truncated : dict of str to float
        Count and fraction of propensity scores clipped by the truncation bounds, and
        the most extreme untruncated value.
    clever_covariate_max : dict of str to float
        Largest absolute clever-covariate value per targeted estimand family -- the
        single most direct summary of how much one observation can move the estimate.
    bounds : tuple of float
        Truncation bounds the fit applied to the treatment mechanism.
    n : int
        Number of observations the diagnostics were computed over.
    mechanisms : dict of str to dict of str to float
        Overlap for the *other* denominators in the clever covariate:
        ``P(Delta = 1 | A, W)`` when outcomes are missing, and ``P(Z = z | A, W)`` for a
        controlled direct effect.  Every row carries the same nine keys: ``min``, ``q01``,
        ``q05`` and ``median`` of the mechanism as fitted; ``clipped`` and
        ``clipped_fraction`` for the cells the bound moved; and ``ess_ratio``,
        ``top_1pct`` and ``top_5pct`` for the load the weights leave behind.  Empty when
        neither applies.

        The three load measures are taken over the weights the estimating equation
        actually forms: the *bounded* mechanism at each unit's realised arm, times that
        unit's observation weight, over the rows whose residual it multiplies.  They are
        on the same scale as the propensity's :attr:`effective_sample_size` and
        :attr:`weight_share`, and fold in the design for the same reason those do, so the
        two tables can be read side by side.

        These deserve reporting for exactly the reason ``g`` does: they enter the
        estimating equation as a denominator, so a value near zero gives one observation
        unbounded leverage.  Unlike ``g`` they are one-sided -- only the approach to zero
        matters -- and they are easy to overlook, because a fit can have perfectly
        healthy propensity overlap and still be resting on a handful of rows that were
        very unlikely to be observed at all.

        **A derived row reports the complete denominator** for a missing-outcome fit
        (``P(A=a,Delta=1|W)``), a controlled direct effect (``P(A=a,Z=z|W)``), or both at
        once (``P(A=a,Delta=1,Z=z|W)``).  The ``z`` in those names is the level the fit
        targets, which the factor row beside them states outright.  It is not a further
        fitted mechanism and it was never held to a bound of its own: the estimator
        truncates each factor separately and multiplies.  So its ``min`` and quantiles
        stay on the untruncated product, as the fitted rows' do, while its ``clipped``
        counts the cells where *any* factor moved -- which is the truncation this
        denominator underwent, and which a comparison of the two products would miss
        wherever a zero in one factor hides a clip in another.

        The derived row is the reason this table exists.  Each factor can look
        comfortable on its own while their product concentrates the weight, because the
        costs multiply, and only the product is what the clever covariate divides by.
        It describes the fit's **marginal-mean** estimands -- the ``ey`` family, its
        contrasts, a regime, and a marginal structural model.  A fit that targets none of
        those gets its factor rows and no product row, because an ATT or ATC covariate
        divides by ``P(A = a)`` rather than ``g_a(W)`` and an incremental one drops the
        propensity entirely.
    nuisance_bound : float
        The lower bound applied to the *fitted* mechanisms above.  A derived row has no
        single such bound -- see :meth:`_bound_label`.
    simplex_deviation : float
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

    n_repeats : int
        Cross-fitting draws the fit combined. Everything else here describes
        the **first** of them, because overlap is a property of one fitted
        mechanism and a combination of draws would describe one no estimate came from.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns when ``data`` is omitted.
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
    #: How many cross-fitting draws the fit combined. Everything above describes the
    #: **first** of them, and this is here so a reader knows that.  Overlap is a property
    #: of one fitted mechanism, and combining ``R`` propensity vectors would produce a
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
        """Propensity quantiles as a tidy frame.

        Parameters
        ----------
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses
            :attr:`backend`.

        Returns
        -------
        dataframe
            One row per ``(arm, quantile)`` of the treatment mechanism.
        """
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

    def _bound_label(self, name: str) -> str:
        """How a mechanism row was truncated, as text a message can print.

        A derived row is the only kind with **more than one** bound: the estimator clips
        each factor separately and multiplies them, so naming any one alone -- which both
        call sites used to do, always ``nuisance_bound`` -- quotes a bound that row was
        never held to.  How many it names comes from
        :data:`_COMPOSED_NUISANCE_FACTORS`, so a further composition does not need a
        second place to agree with.
        """
        if name in _COMPOSED_MECHANISMS:
            nuisance_factors = _COMPOSED_NUISANCE_FACTORS[name]
            tail = " x ".join(f"[{self.nuisance_bound:.4g}, 1]" for _ in range(nuisance_factors))
            return f"[{self.bounds[0]:.4g}, {self.bounds[1]:.4g}] x {tail}, factor by factor"
        return f"[{self.nuisance_bound:.4g}, 1]"

    def summary(self) -> str:
        """A printable overlap report.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        lines = [
            "Positivity / overlap diagnostics",
            "-" * 32,
            f"n = {self.n}; propensity truncated to [{self.bounds[0]:.4g}, {self.bounds[1]:.4g}]",
        ]
        if self.n_repeats > 1:
            lines.append(
                f"describing draw 1 of {self.n_repeats}: overlap is a property of one "
                "fitted mechanism, not of the median-combined estimate"
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
                    [
                        "mechanism",
                        "min",
                        "1%",
                        "5%",
                        "median",
                        "ESS / n",
                        "top 1% weight",
                        "top 5% weight",
                        "clipped",
                    ],
                    [
                        [
                            name,
                            f"{stats['min']:.4f}",
                            f"{stats['q01']:.4f}",
                            f"{stats['q05']:.4f}",
                            f"{stats['median']:.4f}",
                            f"{stats['ess_ratio']:.3f}",
                            f"{stats['top_1pct']:.3f}",
                            f"{stats['top_5pct']:.3f}",
                            f"{stats['clipped']:.0f} ({stats['clipped_fraction']:.2%})",
                        ]
                        for name, stats in self.mechanisms.items()
                    ],
                )
            )
            truncations = "; ".join(
                f"{name} truncated to {self._bound_label(name)}" for name in self.mechanisms
            )
            lines.append(f"({truncations}; each row counts both arms)")
        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)

    @property
    def severity(self) -> Literal["adequate", "strain", "serious"]:
        """Which tier this report's own thresholds place the fit in.

        Returns
        -------
        {"adequate", "strain", "serious"}
            The tier behind :meth:`verdict`, for a caller that must branch on it.
        """
        return self._verdict_parts()[0]

    def verdict(self) -> str:
        """A one-line reading of the diagnostics.

        Returns
        -------
        str
            One line saying whether the overlap supports the reported estimate.
        """
        return self._verdict_parts()[1]

    def _verdict_parts(self) -> tuple[Literal["adequate", "strain", "serious"], str]:
        """Decide the tier and the sentence that reports it, in one place.

        A reader reads :meth:`verdict`; the combined assessment row reads
        :attr:`severity`.  Both come from here because the alternative is what shipped:
        the aggregate row re-derived its own thresholds and so reported ``passed`` for a
        fit this method called a serious positivity problem.  No caller outside this
        method may restate the numbers below.

        **The effective-sample-size ratio is reported and never graded.**  A Kish ratio
        is a descriptive quantity with no derived cutoff, so a tier keyed on it would be
        a house convention presented as a finding, and a reader would take ``passed`` as
        a positivity clearance this package cannot give.  Every branch states the share
        instead, and the analyst judges it against the question.  What is graded is the
        truncated fraction, which is not a judgement about the data: it counts the rows
        the fit clipped at a bound the caller configured, and a clipped row contributes
        extrapolation rather than data.
        """
        # Non-finite ratios are dropped rather than compared.  An arm with no rows stores
        # NaN here by construction, and `min` over a sequence containing NaN returns
        # whichever value it met first, so an unfiltered comparison made this report's
        # reading depend on arm order.
        ratios = [
            float(ess["ratio"])
            for ess in self.effective_sample_size.values()
            if np.isfinite(ess["ratio"])
        ]
        fraction = self.truncated["fraction"]
        share = (
            (
                f"The weighted analysis uses an effective {min(ratios):.0%} of the rows in its "
                "narrowest arm. No threshold is applied to that share, because none is derived; "
                "read it against what the estimate is for."
            )
            if ratios
            else "No arm reports a finite effective sample size."
        )
        for name, stats in self.mechanisms.items():
            # Checked before the propensity verdict, because this is the failure a reader
            # is least likely to be looking for: overlap in `g` can be immaculate while
            # the estimate rests on a few rows that were very unlikely to be observed.
            # Two triggers, one of which only reports.  A clipped fraction is truncation
            # and grades like the propensity's.  A low mechanism effective sample size
            # earns the *sentence*, because a reader who never sees the number cannot
            # weigh it, but it does not move the tier -- grading it would be the invented
            # cutoff this report refuses to apply to the propensity.
            composed = name in _COMPOSED_MECHANISMS
            # A derived row's clipped cells are the union over its factors, and its
            # propensity factor is the one the branch below already reports -- accurately,
            # and against the curve that can move it. Triggering here on that union put a
            # mechanism's name on a verdict whose every number came from `g`, and sent the
            # reader to `mechanism=True`, which sweeps `nuisance_bound` alone. So a derived
            # row earns its sentence on joint leverage only, which is the one thing neither
            # its factor rows nor the propensity branch can show. Each factor's own
            # clipping still triggers on that factor's own row.
            clipped = not composed and stats["clipped_fraction"] > 0.01
            if clipped or stats["ess_ratio"] < 0.6:
                leverage = (
                    "It is the whole denominator of the clever covariate for this fit's "
                    "marginal-mean estimands, so those rows carry outsized leverage "
                    "however each factor looks on its own."
                    if composed
                    else "It divides the clever covariate exactly as g(W) does, so those "
                    "rows carry outsized leverage whatever the propensity overlap looks like."
                )
                # Either bound can have moved a derived row's cells, so naming one curve
                # would send the reader to a knob that cannot move what they were shown.
                advice = (
                    "Check truncation_curve() and truncation_curve(mechanism=True); this "
                    "row is clipped by either bound."
                    if composed
                    else "Check truncation_curve(mechanism=True)."
                )
                return (
                    "serious"
                    if fraction > 0.05
                    else "strain"
                    if fraction > 0.01 or clipped
                    else "adequate",
                    f"VERDICT: {name} strains the estimate. It falls to {stats['min']:.4g} at "
                    f"its smallest and leaves an effective {stats['ess_ratio']:.0%} of the "
                    f"rows it weights ({stats['clipped_fraction']:.2%} clipped at "
                    f"{self._bound_label(name)}). {leverage} {advice}",
                )
        if fraction > 0.05:
            return (
                "serious",
                f"VERDICT: truncation is carrying this estimate. {fraction:.1%} of units were "
                "clipped, so their contributions rest on extrapolation rather than data. Treat "
                "the estimate as sensitive to this finite-sample regularisation. The bound "
                "does not change the requested estimand. Check "
                f"truncation_curve() before drawing conclusions. {share}",
            )
        if fraction > 0.01:
            return (
                "strain",
                f"VERDICT: some truncation. {fraction:.1%} of units were clipped at the bound. "
                "Report truncation_curve() alongside the estimate so readers can see how much "
                f"the answer depends on it. {share}",
            )
        return (
            "adequate",
            f"VERDICT: no truncation-driven fragility detected ({fraction:.1%} of units "
            f"clipped). {share}",
        )

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

    Parameters
    ----------
    result : TMLEResult
        A fitted result.

    Returns
    -------
    PositivityReport
        Overlap read off the stored nuisance fits, without refitting.
    """
    if result.data.is_continuous_treatment:
        raise DataError(
            f"{result.data.treatment_name} is continuous, so there is no per-arm "
            "propensity to tabulate and this report has no rows to fill. A shift's "
            "positivity question is whether the density ratio g(a - delta | W) / "
            "g(a | W) stays bounded, not whether an arm probability does, and "
            "check_shift_support answers it -- which res.diagnostics.support() reaches "
            "for a fit that declared shifts=. Reaching this report instead means none "
            "were declared. With delta= or intermediate= there is a mechanism in the "
            "denominator as well, and it is the one bound this axis actually has: "
            "res.diagnostics.truncation_curve(mechanism=True) sweeps it."
        )
    if result.data.is_binary_treatment:
        return _binary_positivity_report(result)
    return _multi_arm_positivity_report(result)


def _binary_positivity_report(result: TMLEResult) -> PositivityReport:
    """Overlap for a two-armed treatment, in terms of the single propensity ``g(W)``."""
    data = result.data
    bounds = result.config.g_bounds
    propensity = result.nuisance.propensity
    raw = propensity.arm(1.0)
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

    # Both denominators come out of the one truncation the estimator applies, rather than
    # from a clip written here and a complement written beside it.  The two forms agree
    # only while the bound pair is symmetric.
    truncation = propensity.truncate(bounds)
    bounded = truncation.values
    ess: dict[str, dict[str, float]] = {}
    share: dict[str, dict[str, float]] = {}
    for arm, mask, weights in (
        ("treated", treated, 1.0 / bounded[:, propensity.column_for(1.0)]),
        ("control", ~treated, 1.0 / bounded[:, propensity.column_for(0.0)]),
    ):
        arm_weights = weights[mask] * data.weights[mask]
        ess[arm] = {
            "n": float(mask.sum()),
            "effective": _kish_ess(arm_weights),
            "ratio": _kish_ess(arm_weights) / float(mask.sum()) if mask.any() else float("nan"),
        }
        share[arm] = {
            "top_1pct": top_weight_share(arm_weights, 0.01),
            "top_5pct": top_weight_share(arm_weights, 0.05),
        }

    clipped = truncation.units
    inside = raw[~clipped]
    # This line still reads the two columns as complements, which the rest of the function
    # no longer does: `raw` is g1 alone, so `1 - inside.max()` is the control denominator
    # only on the simplex. `truncate` sends a two-arm `simplex=False` mechanism down its
    # column-by-column branch instead. Nothing in `src/` builds such a mechanism today, so
    # this is recorded rather than restructured.
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

    truncation = propensity.truncate(bounds)
    bounded = truncation.values
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
            "top_1pct": top_weight_share(arm_weights, 0.01),
            "top_5pct": top_weight_share(arm_weights, 0.05),
        }

    clipped_cells = truncation.clipped
    clipped_units = truncation.units
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

    Two views, because they answer different questions.  The quantiles pool every arm,
    since every column is used: each arm's covariate divides by the mechanism at that
    arm, so the union is the set of values that appear as denominators anywhere.  The
    effective sample size instead takes the weights the estimating equation *actually*
    forms -- ``1 / pi`` at each unit's realised arm, over the rows whose residual it
    multiplies -- and is reported on the same scale as the propensity's ESS so that the
    two can be read side by side.

    Beside each fitted mechanism the report carries the *product* the clever covariate
    divides by, named in :data:`_COMPOSED_ROWS`.  It is built here rather than stored on
    the nuisance state, so a cached product cannot go stale against the factors it came
    from, and each factor is bounded through the accessor that owns its rule before the
    multiplication -- which is the order the estimator uses.

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
    missingness_bound = result.config.missingness_bound
    contributing = targeted_rows(data, result.intermediate_value)
    out: dict[str, dict[str, float]] = {}

    # Which column of an `(n, K)` mechanism each row realised.  Written as a gather rather
    # than as `np.where(treatment == 1, values[:, 1], values[:, 0])`, which reports arm 0's
    # denominator for every unit outside arms 0 and 1 -- silently, and on a report that is
    # otherwise per arm.  A two-armed fit gathers the same two columns that expression did.
    realised = np.zeros(data.n, dtype=int)
    for arm in nuisance.propensity.arms:
        realised[data.treatment == arm] = nuisance.propensity.column_for(arm)
    rows = np.arange(data.n)

    def summarize(
        raw: FloatArray, bounded: FloatArray, clipped: BoolArray | None = None
    ) -> dict[str, float]:
        """One mechanism row: raw quantiles, bounded-weight load, and the cells that moved.

        The two arrays are deliberately different.  Quantiles pool every column, because
        every column appears as a denominator somewhere, and they describe the mechanism
        as fitted.  The weights are the ones the estimating equation actually forms: the
        *bounded* array at each unit's realised arm, times that unit's observation weight,
        over the rows whose residual it multiplies.  ``clipped`` defaults to the cells the
        bound moved, which is the rule for a single factor; a derived row passes the union
        of its factors' masks instead, because its own product was never bounded.
        """
        flat = raw.reshape(-1)
        at_arm = bounded[rows, realised]
        weights = data.weights[contributing] / at_arm[contributing]
        changed = raw != bounded if clipped is None else clipped
        return {
            "min": float(flat.min()),
            "q01": float(np.quantile(flat, 0.01)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.median(flat)),
            "ess_ratio": (
                _kish_ess(weights) / float(weights.size) if weights.size else float("nan")
            ),
            "top_1pct": top_weight_share(weights, 0.01),
            "top_5pct": top_weight_share(weights, 0.05),
            "clipped": float(np.count_nonzero(changed)),
            "clipped_fraction": float(np.mean(changed)),
        }

    # Each fitted nuisance factor, as the pair the report needs: the array as fitted, and
    # the same array as the estimator bounds it at targeting time.  Both come from the
    # accessors that own those rules -- `bounded_missingness`, and `intermediate_density`,
    # which is the one place the `z` / `1 - z` convention lives, so the row reports the
    # level the covariate actually divides by rather than its complement.
    level = result.intermediate_value
    has_observation = nuisance.missingness is not None
    has_intermediate = nuisance.intermediate is not None and level is not None
    factors: list[tuple[str, FloatArray, FloatArray]] = []
    if has_observation:
        factors.append(
            (
                "P(Delta=1|A,W)",
                np.asarray(nuisance.missingness, dtype=float),
                np.asarray(nuisance.bounded_missingness(missingness_bound), dtype=float),
            )
        )
    if has_intermediate and level is not None:
        factors.append(
            (
                f"P(Z={level:.0f}|A,W)",
                np.asarray(nuisance.intermediate_density(level, 0.0), dtype=float),
                np.asarray(nuisance.intermediate_density(level, missingness_bound), dtype=float),
            )
        )

    for name, raw, bounded_factor in factors:
        out[name] = summarize(raw, bounded_factor)

    # The product is a derived denominator, not a further fitted or targeted mechanism.
    # Report it without storing it on the nuisance state so the treatment, observation and
    # intermediate probabilities cannot become stale relative to a cached product.  It is
    # reported only for the groups that form it -- see `_COMPOSED_GROUPS`, whose comment
    # names what the others divide by instead.  Refusing the row there is the point: an
    # ATT fit's covariate never forms `g_a * pi_a`, so a row asserting that product is its
    # whole denominator would describe an estimand nobody asked for.
    if factors and not _COMPOSED_GROUPS.isdisjoint(result.fluctuations):
        # The estimator truncates each factor **separately** and multiplies -- see
        # `build_submodel`'s `1 / (g * pi * pz)` and the missing-outcome reductions -- so
        # `g_bounds[0] * missingness_bound` is a floor no code applies. Counting the cells
        # beneath it reports a strict subset of the cells truncation altered, because a
        # small factor beside a large one leaves the product above the product of the
        # floors: at the shipped defaults that floor is around 5e-4, so the row read zero
        # on fits where a third of the observation mechanism was pinned. What is reported
        # instead is the union of the factors' own clipped cells, which is the truncation
        # this row's denominator actually underwent -- and which still marks a cell that a
        # zero in another factor would have hidden from a product-level comparison.  The
        # propensity contributes `Truncation`'s mask rather than a rebuilt predicate,
        # because two arms on the simplex are clipped through `g1` and its complement, so
        # a cell can move without ever standing below the bound.
        truncation = nuisance.propensity.truncate(result.config.g_bounds)
        raws = [np.asarray(nuisance.propensity.values, dtype=float)]
        boundeds = [np.asarray(truncation.values, dtype=float)]
        masks = [np.asarray(truncation.clipped, dtype=bool)]
        for _, raw, bounded_factor in factors:
            raws.append(raw)
            boundeds.append(bounded_factor)
            masks.append(raw != bounded_factor)
        out[_COMPOSED_ROWS[has_observation, has_intermediate]] = summarize(
            np.prod(np.stack(raws), axis=0),
            np.prod(np.stack(boundeds), axis=0),
            np.logical_or.reduce(np.stack(masks)),
        )
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

    Returns a tidy frame with one row per evaluated bound pair and estimand, giving the
    point estimate and confidence interval.  A scalar bound supplied by the caller remains
    shorthand for the symmetric treatment-mechanism pair ``(bound, 1 - bound)``; an
    observation- or intermediate-mechanism sweep uses ``(bound, 1)``.  On an ordinary,
    collaborative, or unguarded
    doubly-robust fit only the targeting step is re-run -- the nuisance fits are cached
    -- so each bound costs a small fraction of the original fit.

    A **guarded DR-TMLE** fit is the exception, and it is why the combined report prices
    this operation per result rather than per operation.  Its targeting step alternates
    against the reduced-dimension regressions, so every bound refits them.  The ordinary
    closure refits them at the fitted reduced bounds; the missing-outcome construction is
    handed the swept bounds instead, because they define two of its regression targets.
    The primary outcome regression and the propensity stay cached, so one bound costs
    less than a fit -- but the default grid has several of them, so the **whole curve can
    cost more than the fit it describes**.  Budget it as refitting work.

    A curve that is flat over the plausible range of bounds says the estimate does not
    hinge on the truncation choice.  A curve that drifts monotonically says the estimate
    is sensitive to this finite-sample regularisation.  It does not change the requested
    estimand.  The bound should be reported with the estimate.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    bounds : sequence of float or None
        Lower truncation values to try.  ``None`` uses a grid from 0.001 to 0.2 and adds
        every exact bound pair the requested parameters used.  An explicit sequence is
        evaluated exactly as requested and is never expanded with fitted points.
    estimands : sequence of str or None
        Restrict the emitted rows to these parameters.  ``None`` uses every parameter the
        fit reported.  Two forms are accepted.  A reported parameter selects its own row,
        so ``"ey[high]"`` gives one row per bound pair.  A registered target name selects
        every reported alias of that target, in report order, so ``"ey"`` on a three-armed
        fit gives three rows per bound pair and on a two-armed fit gives one.  A name that
        is neither is refused, because it has no fitted pair and no fitted estimate for a
        row to reference, and an empty selection is refused because it names no row.
    mechanism : bool
        Sweep the bound on ``P(Delta = 1 | A, W)`` (and the intermediate density)
        instead of the one on ``g(W)``.  That probability divides the clever covariate
        exactly as the propensity does, so it has a truncation curve for exactly the
        same reason -- and it is the one that goes unexamined, because it has no
        familiar name.  Requires a fit with ``delta=`` or ``intermediate=``.

        Note what the curve does and does not show.  Truncating a mechanism cannot move
        the *estimand*: the plug-in is an average of targeted predictions and contains
        no mechanism at all.  What moves is the second-order remainder, so a curve that
        drifts is saying the estimate is leaning on rows the bound is holding up.

    Returns
    -------
    dataframe
        One row per evaluated bound pair and estimand, with the point estimate, interval,
        fitted pair and estimate, and signed movement from that fitted estimate.
    """
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "truncation_curve needs the fitted estimator that produced the result"
        )

    # Every reported alias against the target that answers it, in report order. The
    # selection below reads it, and so does the row loop, so the registry is asked once.
    reported_targets = {name: _target_name(result, name) for name in result.estimates}
    reported = (
        tuple(result.estimates) if estimands is None else _select(estimands, reported_targets)
    )
    if mechanism and result.nuisance.missingness is None and result.nuisance.intermediate is None:
        raise CapabilityError(
            "mechanism=True needs a fit with missing outcomes or an intermediate "
            "variable; without one there is no mechanism in the clever covariate to "
            "truncate. Pass delta=<column> or intermediate=<column> to fit()."
        )

    def pair_for(lower: float) -> tuple[float, float]:
        """One lower bound as the pair the sweep evaluates, in the documented shorthand."""
        return (lower, 1.0 if mechanism else 1.0 - lower)

    # `retarget` takes *target* names, while `result.estimates` is keyed by the parameter
    # names those targets reported -- the same thing for a two-armed fit, and `ey[high]`
    # against `ey` for a wider one. The rows narrow back to `reported`: an arm-narrowed fit
    # reports `ey[high]` alone while the estimator still retargets every arm of `ey`.
    targets = {name: reported_targets[name] for name in reported}
    names = tuple(dict.fromkeys(targets.values()))
    pairs = {
        name: _fitted_bound_pair(result, target, mechanism=mechanism)
        for name, target in targets.items()
    }
    fitted_psi = {name: float(result.estimates[name].psi) for name in reported}

    canned = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
    if bounds is None:
        grid = sorted({*(pair_for(lower) for lower in canned), *pairs.values()})
    else:
        lowers = sorted(float(value) for value in bounds)
        for lower in lowers:
            if not 0.0 < lower < 0.5:
                raise ValueError(f"truncation bounds must lie in (0, 0.5); got {lower}")
        grid = [pair_for(lower) for lower in lowers]

    rows: list[dict[str, Any]] = []
    for lower, upper in grid:
        if not 0.0 < lower < upper <= 1.0:  # pragma: no cover - fitted config is validated
            raise RuntimeError(f"invalid fitted truncation pair {(lower, upper)}")
        pair = (lower, upper)
        # Every draw, then combined the way the fit combines them. Sweeping one draw and
        # calling the answer the fit's would compare a bound's effect on one split against
        # a reported estimate that came from R -- and the difference between the two curves
        # would read as sensitivity to the bound. Costs R times the sweep, which is still a
        # fraction of one refit.
        estimates = median_estimates(
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
        )
        # A retarget of `ey` answers every arm, and a fit narrowed to one arm asked for
        # one row. Dropping the extra names is the ordinary case; missing a requested one
        # is an internal inconsistency, so it is raised rather than silently omitted.
        missing = [name for name in reported if name not in estimates]
        if missing:
            raise RuntimeError(
                f"retargeting at bounds {pair} reported {sorted(estimates)}, which leaves "
                f"{missing} of the requested parameters {list(reported)} unanswered"
            )
        # One truncation load per bound pair: it reads the pair and the mechanism flag,
        # and nothing that varies between the parameters sharing that pair.
        truncated_fraction = _clipped_fraction(result, pair, mechanism)
        for name in reported:
            estimate = estimates[name]
            low, high = estimate.ci
            fitted_lower, fitted_upper = pairs[name]
            reference = fitted_psi[name]
            rows.append(
                {
                    "bound": lower,
                    "estimand": name,
                    "psi": estimate.psi,
                    "std_err": estimate.std_error,
                    "ci_lower": low,
                    "ci_upper": high,
                    "truncated_fraction": truncated_fraction,
                    "is_fitted_bound": pair == (fitted_lower, fitted_upper),
                    # Additive metadata follows the legacy columns so positional consumers
                    # retain the order they saw before the fitted-reference contract grew.
                    "upper_bound": upper,
                    "fitted_lower_bound": fitted_lower,
                    "fitted_upper_bound": fitted_upper,
                    "fitted_psi": reference,
                    "delta_from_fitted": float(estimate.psi - reference),
                }
            )

    payload = {key: [row[key] for row in rows] for key in rows[0]}
    return result.data.frame_like(payload)


def _select(estimands: Any, reported_targets: dict[str, str]) -> tuple[str, ...]:
    """The reported parameters an ``estimands=`` request selects, in report order.

    A reported parameter selects its own row.  A registered target name selects every
    reported alias of that target, because the two spellings coincide only on a two-armed
    fit: ``estimands=["ey"]`` has to keep working on a fit that reports ``ey[low]``,
    ``ey[medium]`` and ``ey[high]``, where the target is the name a caller has.

    Every other name is refused here, before the first retarget.  It has no fitted pair
    and no fitted estimate, so every row it would fill is undefined.  A typed arm label
    the fit does not carry, such as ``ey[nope]``, is one of those: it is not stemmed back
    to ``ey`` and answered with the whole target.  An empty request is refused for the
    same reason, and at the same point, because it names no row for the sweep to emit.
    """
    requested = tuple(dict.fromkeys(estimands))
    if not requested:
        raise CapabilityError(
            "estimands= selected no parameter; pass at least one of "
            f"{list(reported_targets)}, or None for every parameter this fit reported"
        )
    targets = [
        name for name in dict.fromkeys(reported_targets.values()) if name not in reported_targets
    ]
    unknown = [
        name
        for name in requested
        if name not in reported_targets and name not in reported_targets.values()
    ]
    if unknown:
        also = f"; or a target name whose reported arms it selects: {targets}" if targets else ""
        raise CapabilityError(
            f"parameter(s) {unknown} were not reported by this fit; "
            f"choose from {list(reported_targets)}{also}"
        )
    selected = [
        alias
        for name in requested
        for alias in (
            (name,)
            if name in reported_targets
            else tuple(a for a, target in reported_targets.items() if target == name)
        )
    ]
    return tuple(dict.fromkeys(selected))


def _target_name(result: TMLEResult, name: str) -> str:
    """Registered target for one reported alias, with a legacy-result fallback."""
    # The mapping is a dataclass field and always exists, but a raw estimator result
    # leaves it empty. The stem is that path's answer, and it is the only one available.
    keys = result.parameter_keys
    key = keys.get(name) if keys else None
    return key.estimand if key is not None else parameter_stem(name)


def _fitted_bound_pair(result: TMLEResult, target: str, *, mechanism: bool) -> tuple[float, float]:
    """The fitted mechanism pair applicable to one registered target."""
    if mechanism:
        return float(result.config.missingness_bound), 1.0
    group = TARGETS[target].group
    return g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)


def _clipped_fraction(result: TMLEResult, pair: tuple[float, float], mechanism: bool) -> float:
    """Share of the nuisance the bound would clip, for whichever bound is swept.

    The treatment branch asks :meth:`~cleverly.estimators._nuisance.Propensity.truncate`
    and reports the share of **units** it moves at ``pair``.  A unit is one row of the
    mechanism, and one binding denominator is enough to make that row's contribution
    extrapolation.  The share therefore belongs to the pair it was evaluated at, and it
    equals the :func:`positivity_report` figure only at that report's own pair.  A
    continuous treatment fits no arms at all, and the share is ``nan`` there rather than
    a claim that the bound moved nothing.

    The mechanism branch has one column per arm as well, but its bound is applied cell by
    cell and is reported that way.
    """
    if not mechanism:
        return result.nuisance.propensity.truncate(pair).fraction
    lower, upper = pair
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
    divisor = np.concatenate(parts)
    return float(np.mean((divisor < lower) | (divisor > upper)))
