"""Shared estimator infrastructure: configuration, results and reporting.

Every TMLE variant in this library produces the same shape of result -- a set of
:class:`~cleverly.inference.ParameterEstimate` objects plus the nuisance fits and
targeting details that produced them -- so sensitivity analysis, validation and
reporting are written once, here, rather than per estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray, ParameterAxis
from ..data.causal_data import CausalData
from ..fluctuation.iterative import Fluctuation
from ..inference.bootstrap import BootstrapResult
from ..inference.cluster import influence_covariance
from ..inference.delta import delta_method
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.multiplier import SimultaneousBands
from ..learners.crossfit import CrossFitPlan
from ..provenance import Provenance
from ..targets import TARGETS, all_names, resolve_estimands
from ..utils.frames import emit_frame
from ..utils.text import format_pvalue, format_table
from ._nuisance import NuisanceEstimates, RepeatFit
from .direct_effect import describe as describe_direct_effect
from .targeting import TargetingSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..assessment import DiagnosticsFacade, Replayability, SensitivityFacade, ValidationReport
    from ..sensitivity.api import SensitivityAnalysis
    from ..validation.api import ValidationSuite
    from ..validation.score import ScoreCheck

__all__ = [
    "ALL_ESTIMANDS",
    "DEFAULT_ESTIMANDS",
    "MEAN_GROUP_ESTIMANDS",
    "CVTargeting",
    "TMLEConfig",
    "TMLEResult",
    "TMLEResultSet",
    "resolve_estimands",
]

#: Estimands available from a classic point-treatment fit, in report order.  Derived
#: from the registry rather than declared here, so a registered target appears in it
#: automatically and the two cannot disagree.
ALL_ESTIMANDS: tuple[str, ...] = all_names()

#: Estimands produced by the two-column ``mean`` fluctuation.
MEAN_GROUP_ESTIMANDS: frozenset[str] = frozenset(
    name for name, target in TARGETS.items() if target.group == "mean"
)

#: The default report for a continuous outcome.  A binary outcome additionally gets
#: ``rr`` and ``or``, which are only defined when the means are probabilities.
#:
#: Spans every parameter axis, so it is a *listing* rather than a report any one fit
#: produces: no fit reports ``ate`` and ``ey_regime`` and ``ey_shift`` together, because
#: declaring ``interventions=`` or ``shifts=`` switches which axis is in scope.  Use
#: :func:`~cleverly.targets.default_names` with the fit's axis for the report itself.
DEFAULT_ESTIMANDS: tuple[str, ...] = tuple(
    name for name, target in TARGETS.items() if target.in_default_set
)


@dataclass(frozen=True)
class TMLEConfig:
    """A snapshot of the settings a fit actually used.

    Recorded on the result so a reported number can always be traced back to the
    bounds, folds and submodel that produced it -- including values the estimator
    resolved itself, such as ``g_bounds="auto"``.
    """

    family: str
    #: Everything the targeting step needs that is not data.  Held as one object
    #: rather than as loose fields so that re-solving a fluctuation -- the truncation
    #: curve, the MNAR tilt, the C-TMLE search -- needs the config and not the live
    #: estimator that produced it.
    targeting_spec: TargetingSpec
    targeting_scheme: str
    cross_fit: bool
    n_folds: int
    g_bounds: tuple[float, float]
    g_bounds_conditional: tuple[float, float]
    missingness_bound: float
    q_bounds: tuple[float, float] | None
    screen_treatment: bool
    estimands: tuple[str, ...]
    alpha_sig: float
    random_state: int | None = None
    n_bootstrap: int = 0
    cv_evaluation: bool = False
    auto_bounds_n: float | None = None
    #: Which mechanisms ``missingness_bound`` was actually applied to -- empty when the
    #: fit had neither missing outcomes nor an intermediate variable, in which case the
    #: bound exists on the config but never touched anything.
    bounded_mechanisms: tuple[str, ...] = ()
    #: The arm code every contrast estimand is taken against.  Part of the *estimand*
    #: rather than a setting: ``ate[medium vs low]`` and ``ate[medium vs high]`` are
    #: different parameters, so which one was reported has to be recorded alongside the
    #: number.  ``0.0`` -- the lowest arm -- unless the caller chose otherwise.
    reference_arm: float = 0.0
    #: What this fit's parameters are indexed by -- see
    #: :attr:`~cleverly.Target.parameter_axis`.  Recorded rather than inferred from the
    #: estimand names, so a result read back from disk can say what it reported without
    #: parsing them.
    parameter_axis: ParameterAxis = "arm"
    #: The fold policy the caller *declared*, as against ``n_folds`` above, which is the
    #: count the fit actually ran.  The two come apart without leaving a trace otherwise:
    #: ``resolve_n_folds`` caps the count at the rarest stratum and ``make_folds`` caps it
    #: again at the cluster count, each with a warning that is gone by the time anyone
    #: reads the result.  Defaulted so that every existing construction and
    #: ``dataclasses.replace`` keeps working untouched.
    crossfit: CrossFitPlan = field(default_factory=CrossFitPlan)

    # Read-through to the spec, so the settings appear once and cannot drift.
    @property
    def fluctuation(self) -> str:
        return self.targeting_spec.fluctuation

    @property
    def targeting(self) -> str:
        return self.targeting_spec.targeting

    @property
    def alpha(self) -> float:
        return self.targeting_spec.alpha

    @property
    def target_weights(self) -> bool:
        return self.targeting_spec.target_weights

    @property
    def estimator_name(self) -> str:
        """Which of the three cross-fitting constructions this fit actually ran.

        Worth naming rather than reporting the raw settings: one common update over the
        validation losses is the source CV-TMLE targeting step, while a separate update
        per fold is an optional extension with different finite-sample behavior.
        """
        if not self.cross_fit:
            return "TMLE (in-sample nuisances)"
        if self.targeting_scheme == "pooled":
            return "fold-evaluated CV-TMLE" if self.cv_evaluation else "stacked CV-TMLE (Levy)"
        return (
            "fold-specific fold-evaluated TMLE"
            if self.cv_evaluation
            else "fold-specific targeted TMLE"
        )

    def describe(self) -> list[str]:
        """Human-readable lines for :meth:`TMLEResult.summary`."""
        lines = [
            f"outcome family: {self.family}; fluctuation: {self.fluctuation} ({self.targeting})",
        ]
        if self.cross_fit:
            lines.append(
                f"{self.estimator_name}: nuisances cross-fitted over {self.n_folds} folds; "
                f"targeting: {self.targeting_scheme}"
            )
            if self.crossfit.repeated:
                # Always shown when it applies, unlike the capped-folds line below: a
                # repeated fit reports a different estimator from an ordinary one, and a
                # reader who cannot tell them apart cannot compare two summaries.
                lines.append(
                    f"  (averaged over {self.crossfit.repeats} independent draws of the "
                    "split; the influence curve is the mean of theirs)"
                )
                if self.cv_evaluation:
                    # Which variance rule produced the interval is not recoverable from
                    # the number, and here it is not the one the line above implies.
                    lines.append(
                        "  (the standard error is the mean of the draws' cross-validated "
                        "variances, not the variance of that curve)"
                    )
        else:
            lines.append(f"{self.estimator_name}: nuisances fitted in-sample (cross_fit=False)")
        if self.cross_fit and self.crossfit.n_folds != self.n_folds:
            # Only when they disagree, because agreeing is the ordinary case and a line
            # that always appears is a line nobody reads. When they do disagree the
            # warning that explained it was emitted at fit time and is long gone, so this
            # is the only place a reader can find out that the split was capped.
            lines.append(
                f"  ({self.crossfit.n_folds} folds were declared; the split was capped "
                f"at {self.n_folds} by the rarest stratum or the cluster count)"
            )
        # A shift fit's mechanism is a conditional density, not a propensity: nothing is
        # truncated into g_bounds and reporting the bound would name a step that did not
        # happen. What bounds a density ratio there is the cap the analyst declared,
        # which is part of the estimand and so appears in the parameter names instead.
        if self.parameter_axis != "shift":
            bounds = f"propensity truncated to [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}]"
            if self.auto_bounds_n is not None:
                # Named because it is a deliberate divergence from R's rule, and because a
                # reader comparing two fits needs to know the bound moved with the weights.
                bounds += f" (auto, resolved at the effective n of {self.auto_bounds_n:.0f})"
            if self.g_bounds_conditional != self.g_bounds:
                bounds += (
                    f"; ATT/ATC to [{self.g_bounds_conditional[0]:.4g}, "
                    f"{self.g_bounds_conditional[1]:.4g}]"
                )
            lines.append(bounds)
        if self.bounded_mechanisms:
            # The propensity is not the only denominator in the clever covariate, and a
            # reader comparing two fits needs to see every bound that shaped the estimate
            # -- not just the one with a familiar name.
            named = " and ".join(self.bounded_mechanisms)
            lines.append(f"{named} truncated to [{self.missingness_bound:.4g}, 1]")
        if self.q_bounds is not None:
            lines.append(
                f"outcome scaled from [{self.q_bounds[0]:.4g}, {self.q_bounds[1]:.4g}] to [0, 1]"
            )
        if self.target_weights:
            lines.append("fluctuation in weighted form (target_weights=True)")
        return lines


@dataclass(frozen=True)
class CVTargeting:
    """Fold-level detail from a cross-validated TMLE fit.

    The original CV-TMLE evaluates updated fold-specific distributions one fold at a
    time.  Its usual pooled targeting step fits one common ``epsilon`` by minimising the
    equal average of their validation losses; :attr:`epsilon` records that update.  Levy's
    easy implementation instead stacks the out-of-fold predictions, targets and evaluates
    them just like one ordinary TMLE. That stacked construction is the package default,
    defined here by Levy (2018). The pinned ``tmle3`` source snapshot in the references
    implements the same path by requesting its ``"validation"`` likelihood in
    ``tmle3_Update``; it is corroborating implementation evidence, not the specification.
    The optional ``targeting_scheme="fold"`` extension fits one update per fold; those
    coefficients are kept separately in :attr:`fold_epsilon`.

    The two reports here are genuinely different estimators, not two views of one.
    :attr:`pooled` stitches the updated validation predictions together and evaluates the
    estimand once over the whole sample; :attr:`canonical` (the compatibility name for
    the original fold-evaluated report) evaluates it inside each fold and averages, which
    is the construction Zheng & van der Laan (2011) analyse.  They
    coincide exactly for estimands linear in the targeted predictions at equal fold
    sizes -- ``ate``, ``ey1``, ``ey0`` -- and diverge for ``rr``, ``or``, ``att`` and
    ``atc``, where a ratio of means is not a mean of ratios and the pooled conditional
    effects weight by the whole sample's arm share rather than each fold's.  Which one
    ``result[name]`` carries is set by ``TMLE(cv_evaluation=...)``; both are always here.

    Under ``repeats=R`` the fields divide by what they are.  The three that are
    *estimates* -- :attr:`pooled`, :attr:`canonical` and :attr:`variance` -- follow every
    draw, exactly as the headline report does.  The four that are indexed *by fold* --
    :attr:`n_folds`, :attr:`fold_sizes`, :attr:`fold_estimates`, :attr:`fold_epsilon` --
    describe the first draw alone, because fold 3 of one draw is not fold 3 of another
    and there is no correspondence along which to average them.  :attr:`repeats` says how
    many draws the first three cover.

    Attributes
    ----------
    variance, std_error:
        The cross-validated variance of Zheng & van der Laan (2011) -- the fold-averaged
        second moment of the fold-specific influence curves -- per estimand.  This is the
        standard error attached to :attr:`canonical`; the pooled report carries the
        ordinary influence-curve one.  The two agree when the folds are balanced and the
        score equation is solved, so a gap between them is itself informative.  Over
        ``R`` draws it is the mean of their ``R`` cross-validated variances; see
        ``cleverly.estimators.tmle._with_cross_validated_variance`` for why that, and not
        a cross-validated variance of the averaged curve, is the reported quantity.
    fold_estimates:
        Per-estimand tuple of fold-specific plug-in estimates, from the first draw.
        Estimands that some fold could not evaluate (no units in the conditioning arm, a
        boundary counterfactual mean) are absent, and their omission was warned about at
        fit time.
    epsilon:
        Per-targeting-group common fluctuation coefficient.  For the fold-specific
        extension this is only the mass-weighted summary reported by ``Fluctuation``.
    fold_epsilon:
        Per-targeting-group tuple of fold-specific fluctuation coefficients, from the
        first draw. Empty for common pooled-validation targeting.
    pooled, canonical:
        The two reports, per estimand, averaged over every draw.
    repeats:
        How many cross-fitting draws :attr:`pooled`, :attr:`canonical` and
        :attr:`variance` were averaged over.
    """

    n_folds: int
    fold_sizes: tuple[int, ...]
    variance: dict[str, float]
    fold_estimates: dict[str, tuple[float, ...]]
    epsilon: dict[str, tuple[float, ...]]
    fold_epsilon: dict[str, tuple[tuple[float, ...], ...]]
    pooled: dict[str, ParameterEstimate] = field(default_factory=dict)
    canonical: dict[str, ParameterEstimate] = field(default_factory=dict)
    repeats: int = 1
    #: Name of the dataframe backend the fit's data arrived in, as on every other
    #: report here.
    backend: str | None = None

    @property
    def std_error(self) -> dict[str, float]:
        """Cross-validated standard error per estimand."""
        return {name: float(np.sqrt(value)) for name, value in self.variance.items()}

    @property
    def fold_evaluated(self) -> dict[str, ParameterEstimate]:
        """The original fold-evaluated report (clear alias for ``canonical``)."""
        return self.canonical

    def to_frame(self, data: CausalData | None = None) -> Any:
        """One row per estimand: both reports, the CV standard error and the spread."""
        names = list(self.variance)
        errors = self.std_error
        payload: dict[str, Any] = {
            "estimand": names,
            "canonical_psi": [self.canonical[name].psi for name in names],
            "pooled_psi": [self.pooled[name].psi for name in names],
            "cv_std_err": [errors[name] for name in names],
            "pooled_std_err": [self.pooled[name].std_error for name in names],
            "fold_sd": [
                float(np.std(self.fold_estimates[name], ddof=1))
                if len(self.fold_estimates.get(name, ())) > 1
                else float("nan")
                for name in names
            ],
        }
        return emit_frame(payload, data, backend=self.backend)

    def summary(self) -> str:
        """A printable report of the fold-level targeting."""
        errors = self.std_error
        rows = []
        for name, values in self.fold_estimates.items():
            spread = f"{np.std(values, ddof=1):.4g}" if len(values) > 1 else "n/a"
            rows.append(
                [
                    name,
                    f"{self.canonical[name].psi:.5g}",
                    f"{errors[name]:.4g}",
                    f"{self.pooled[name].psi:.5g}",
                    f"{self.pooled[name].std_error:.4g}",
                    spread,
                    f"[{min(values):.5g}, {max(values):.5g}]",
                ]
            )
        table = format_table(
            [
                "estimand",
                "fold-evaluated",
                "cv std_err",
                "pooled",
                "std_err",
                "fold sd",
                "fold range",
            ],
            rows,
        )
        header = [
            f"Cross-validated targeting over {self.n_folds} folds "
            f"(sizes {min(self.fold_sizes)}-{max(self.fold_sizes)})",
            "fold-evaluated: evaluated inside each fold and averaged (original CV-TMLE).",
            "pooled: this fit's updated validation predictions stitched and evaluated once.",
        ]
        if self.repeats > 1:
            header.append(
                f"both reports and the cv std_err average {self.repeats} draws of the "
                "split; the fold columns describe the first."
            )
        header.append("")
        epsilon_lines = ["", "common fluctuation coefficients:"]
        for group, eps in self.epsilon.items():
            formatted = "(" + ", ".join(f"{e:.4g}" for e in eps) + ")"
            epsilon_lines.append(f"  {group}: {formatted}")
        if self.fold_epsilon:
            epsilon_lines.append("fold-specific fluctuation coefficients:")
        for group, per_fold in self.fold_epsilon.items():
            formatted = ", ".join(
                "(" + ", ".join(f"{e:.4g}" for e in eps) + ")" for eps in per_fold
            )
            epsilon_lines.append(f"  {group}: {formatted}")
        return "\n".join([*header, table, *epsilon_lines])


@dataclass(frozen=True)
class TMLEResult:
    """The result of a TMLE fit.

    Behaves like a mapping from estimand name to
    :class:`~cleverly.inference.ParameterEstimate`, and carries the nuisance fits so
    that sensitivity and validation analyses can be run without refitting.

    Attributes
    ----------
    repeats:
        One :class:`~cleverly.estimators._nuisance.RepeatFit` per draw of the
        cross-fitting split -- a one-element tuple for an ordinary fit, ``R`` of them
        under ``repeats=R``.  This is where the nuisances and the fluctuations actually
        live; :attr:`nuisance` and :attr:`fluctuations` read through to the first entry,
        which is what keeps every analysis written against a single fit working unchanged.
        Anything that must account for *all* the draws -- and every analysis that produces
        a number, as against one that describes a mechanism, must -- iterates this.
    intermediate_value:
        The level of the intermediate variable this fit targets, or ``None`` for an
        ordinary point-treatment fit.  It is part of the *estimand*, not a setting: every
        estimate here is a controlled direct effect holding ``Z`` at this level, and the
        same data yields a different parameter at the other one.  See
        :mod:`cleverly.estimators.direct_effect`.
    """

    estimates: dict[str, ParameterEstimate]
    repeats: tuple[RepeatFit, ...]
    data: CausalData
    config: TMLEConfig
    estimator: Any = None
    provenance: Provenance | None = None
    simultaneous: SimultaneousBands | None = None
    bootstrap: BootstrapResult | None = None
    intermediate_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: Present for fits made through ``CausalStudy``. Legacy estimator calls leave it
    #: absent while the clean-break migration is in progress.
    identified_effect: Any = None
    #: The normalized typed method configuration used by ``IdentifiedEffect.estimate``.
    method: Any = None
    #: Alias-to-structured-key mapping. Routing must read this rather than parse aliases.
    parameter_keys: dict[str, Any] = field(default_factory=dict)
    #: Persistent assessment results, keyed by operation plus normalized arguments.
    #: Filling this mapping never changes the fitted parameter or its summary.
    assessment_cache: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- repeats

    @property
    def nuisance(self) -> NuisanceEstimates:
        """The first draw's nuisance predictions.

        The attribute every sensitivity analysis and validation diagnostic was written
        against, kept pointing at one object so that a fit with no repeats behaves exactly
        as it always has.  On a repeated fit this is draw zero of ``R`` and *nothing
        else*: it is the right thing to describe a fitted mechanism with, and the wrong
        thing to compute a reported number from.  Use :attr:`repeats` for the latter.
        """
        return self.repeats[0].nuisance

    @property
    def fluctuations(self) -> dict[str, Fluctuation]:
        """The first draw's solved fluctuations, keyed by target group.

        Reads through to :attr:`repeats` for the reason :attr:`nuisance` does, and carries
        the same warning: on a repeated fit this is one draw's ``epsilon`` and one draw's
        targeted ``Qbar``.
        """
        return self.repeats[0].fluctuations

    @property
    def nuisances(self) -> tuple[NuisanceEstimates, ...]:
        """Every draw's nuisance predictions, in fit order."""
        return tuple(repeat.nuisance for repeat in self.repeats)

    @property
    def n_repeats(self) -> int:
        """How many draws of the cross-fitting split this fit averaged over."""
        return len(self.repeats)

    def repeat_spread(self) -> dict[str, float]:
        r"""Standard deviation of ``psi`` across the cross-fitting draws, per estimand.

        A *diagnostic*, and emphatically not a standard error.  It measures how much the
        arbitrary fold assignment moved the answer: the ``R`` draws differ in nothing but
        the split, so :math:`\mathrm{sd}(\psi_r)` is the size of the fold noise a single
        fit carries silently, and :math:`\mathrm{sd}(\psi_r)/\sqrt{R}` is roughly what
        survives of it in the reported average.  Read against
        :attr:`~cleverly.ParameterEstimate.std_error`: a spread that is an appreciable
        fraction of the standard error means the split mattered, and one near zero means
        the nuisance fits were stable enough that repeating bought little.

        What it must not be used for is inference.  It says nothing about the *sampling*
        variability of the estimand, so it is neither an alternative to the influence-curve
        standard error nor something to add to it -- the reported interval already covers
        the estimator that was reported, which is the average.

        Raises when there is only one draw, since the standard deviation of one number is
        not a diagnostic but an artefact.
        """
        if self.n_repeats < 2:
            raise ValueError(
                "repeat_spread() describes how much psi moved between draws of the "
                f"cross-fitting split, and this fit has {self.n_repeats}. Fit with "
                "repeats=2 or more."
            )
        shared = [name for name in self.estimates if all(name in r.psi for r in self.repeats)]
        return {
            name: float(np.std([repeat.psi[name] for repeat in self.repeats], ddof=1))
            for name in shared
        }

    # ------------------------------------------------------------- accessors

    def __getitem__(self, name: str) -> ParameterEstimate:
        try:
            return self.estimates[name]
        except KeyError:
            raise KeyError(
                f"estimand {name!r} was not requested; available: {list(self.estimates)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self.estimates

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    @property
    def estimate(self) -> ParameterEstimate:
        """The sole parameter estimate, with an explicit refusal for multi-parameter fits."""
        if len(self.estimates) != 1:
            raise ValueError(
                "this result contains multiple parameters; index the one you want from "
                f"{list(self.estimates)}"
            )
        return next(iter(self.estimates.values()))

    def psi(self, name: str | None = None) -> float:
        """Point estimate for ``name``, or for the sole parameter when omitted."""
        return self.estimate.psi if name is None else self[name].psi

    @property
    def ate(self) -> ParameterEstimate:
        """The average treatment effect, the most commonly wanted estimand."""
        return self["ate"]

    @property
    def n(self) -> int:
        return self.data.n

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        return {name: estimate.influence_curve for name, estimate in self.estimates.items()}

    @property
    def cv_targeting(self) -> CVTargeting | None:
        """Fold-level detail for fold evaluation or fold-specific targeting.

        Carries both the stacked and original fold-evaluated reports whichever one
        ``result[name]`` was configured to show, so the two can always be compared. The
        latter retains the :attr:`CVTargeting.canonical` compatibility name.
        """
        value = self.extra.get("cv_tmle")
        return value if isinstance(value, CVTargeting) else None

    # ------------------------------------------------------------- contrasts

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Joint covariance matrix of the requested estimates.

        The estimands are *not* independent -- they are functionals of one targeted
        distribution and share most of their influence curve -- so a contrast built
        from two of them needs this rather than the two variances.  Computed from the
        influence curves at the right independent unit, so a clustered fit gets the
        cluster-level covariance.
        """
        chosen = self._names(names)
        # column_stack, not a list: influence_covariance takes an (n, m) matrix, and a
        # list of m curves would be read as m observations of n estimands.
        curves = np.column_stack([self[name].influence_curve for name in chosen])
        return influence_covariance(curves, cluster=self.data.cluster)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        scale: Scale = "difference",
        gradient: Callable[[FloatArray], FloatArray] | None = None,
    ) -> ParameterEstimate:
        r"""A smooth function of several estimands, with correct inference.

        Applies the delta method to the *joint* influence curve, so the correlation
        between the estimands is handled rather than ignored:
        :math:`D_\phi = \nabla\phi(\hat\psi)^\top D`.

        >>> res.contrast(lambda p: p[0] - p[1], ["ey1", "ey0"])   # doctest: +SKIP

        Pass ``gradient`` when the function's derivative is known in closed form.  The
        default is a central difference, which is accurate to about ``1e-10`` relative
        -- fine for reporting, but not for a test asserting agreement at ``1e-12``.

        The result is an ordinary :class:`~cleverly.inference.ParameterEstimate`, so it
        carries its own influence curve and can itself be fed back into a contrast.
        """
        chosen = self._names(names)
        estimates = [self[key].psi for key in chosen]
        curves = [self[key].influence_curve for key in chosen]
        value, curve = delta_method(function, estimates, curves, gradient=gradient)
        label = name or f"contrast({', '.join(chosen)})"
        return make_estimate(
            label,
            value,
            curve,
            n=self.n,
            cluster=self.data.cluster,
            scale=scale,
            alpha=self.config.alpha_sig,
        )

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        chosen = tuple(self.estimates) if names is None else tuple(names)
        missing = [key for key in chosen if key not in self.estimates]
        if missing:
            raise KeyError(
                f"estimand(s) {missing} were not requested; available: {list(self.estimates)}"
            )
        if not chosen:
            raise ValueError("no estimands selected")
        return chosen

    # ----------------------------------------------------------- diagnostics

    @cached_property
    def _legacy_sensitivity(self) -> SensitivityAnalysis:
        """The evidenced point analyses wrapped by the public capability facade."""
        from ..sensitivity.api import SensitivityAnalysis

        return SensitivityAnalysis(self)

    @cached_property
    def sensitivity(self) -> SensitivityFacade:
        """Capability-aware sensitivity analyses for this fitted method."""
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self, self._legacy_sensitivity)

    @cached_property
    def diagnostics(self) -> DiagnosticsFacade:
        """Unified support, nuisance, score, and refutation diagnostics."""
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    def validate(self) -> ValidationReport:
        """Run the inexpensive method-appropriate checks without refitting."""
        from ..assessment import validate_result

        return validate_result(self)

    @property
    def replayability(self) -> Replayability:
        """Which post-fit operations this in-memory or restored result can replay."""
        from ..assessment import replayability

        return replayability(self)

    @cached_property
    def validation(self) -> ValidationSuite:
        """Validation diagnostics for this fit -- see :mod:`cleverly.validation`."""
        from ..validation.api import ValidationSuite

        return ValidationSuite(self)

    @cached_property
    def score_verdict(self) -> ScoreCheck:
        """This fit's own answer to whether its interval is licensed.

        The same object ``result.validation.score_check()`` returns, at the default
        tolerance, held here because :meth:`summary` reports it and because a caller
        should not have to know which subsystem owns the question.  **Derived, never
        stored**: it is recomputed from the fluctuations a result carries, so a fit
        reloaded from disk answers with its own records rather than with a flag written
        at fit time that nothing could check afterwards. Whole-result persistence retains
        those records directly instead of rebuilding a partial result graph.

        Free: it reads cached arrays and refits nothing.
        """
        return self.validation.score_check()

    # ---------------------------------------------------------------- output

    def save(self, path: Any) -> Any:
        """Write this result to a trusted joblib artifact; see :func:`cleverly.load`.

        Loading joblib data can execute arbitrary Python code. Only load files from a
        trusted source in a compatible Python and dependency environment.
        """
        from .serialize import save as _save

        return _save(self, path)

    def to_frame(self) -> Any:
        """Tidy results, one row per estimand, in the caller's dataframe backend."""
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        payload: dict[str, Any] = {}
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        for key in keys:
            payload[key] = [row.get(key, None) for row in rows]
        if self.simultaneous is not None:
            bands = self.simultaneous.bands
            payload["simultaneous_lower"] = [
                bands.get(name, (None, None))[0] for name in self.estimates
            ]
            payload["simultaneous_upper"] = [
                bands.get(name, (None, None))[1] for name in self.estimates
            ]
        if self.intermediate_value is not None:
            payload["intermediate"] = [self.intermediate_value] * len(rows)
        return self.data.frame_like(payload)

    def coefficients(self, scale: str = "link") -> Any:
        """A working model's coefficients, on the link scale or exponentiated.

        ``scale="link"`` reports what the fit estimated: :math:`\\hat\\beta`, with a Wald
        interval on the scale the model is linear on.  ``scale="ratio"`` reports
        :math:`e^{\\hat\\beta}` with the interval exponentiated from that same scale, which
        is how the applied literature reports a log- or logit-link marginal structural
        model and the reason those links exist here.

        **What the exponential means depends on the link, and the two are not
        interchangeable.**  Under ``link="log"`` a coefficient exponentiates to a risk (or
        rate) ratio; under ``link="logit"`` to an *odds* ratio.  Reporting one as the other
        is a real error rather than a wording preference, so the ``scale`` column of the
        returned frame names which it is, row by row.  The **intercept** is a third thing
        again: :math:`e^{\\beta_0}` is a baseline mean (log) or a baseline odds (logit),
        not a ratio of anything, and its p-value tests :math:`\\beta_0 = 0` rather than any
        absence of effect.  It is reported rather than dropped -- it is a coefficient of
        the declared model like the others -- and labelled ``"baseline"``.

        Refused on an identity-link fit, where :math:`\\beta` is a risk difference and
        :math:`e^\\beta` is not a quantity; and on a fit with no working model at all,
        whose parameters are counterfactual means rather than coefficients.

        Nothing is re-estimated: the interval and the p-value come from the influence
        curve the fit already reported, read on the scale ``scale`` names.  This is a
        *view*, in the sense :meth:`cleverly.longitudinal.LTMLEResult.curve` is one.
        """
        if scale not in ("link", "ratio"):
            raise ValueError(f"scale must be 'link' or 'ratio'; got {scale!r}")
        msm = self.nuisance.msm
        if msm is None:
            raise ValueError(
                "this fit has no working model, so it reports counterfactual means rather "
                "than coefficients; result.to_frame() is its report. Declare msm= to "
                "project those means onto a model whose coefficients are the parameters."
            )
        if scale == "ratio" and msm.link == "identity":
            raise ValueError(
                "an identity-link working model's coefficients are risk differences, and "
                "exp() of a difference is not a quantity anybody reports. Declare "
                "link='log' for coefficients that are log risk ratios, or link='logit' "
                "for log odds ratios, and this view exponentiates those."
            )
        # Term to reported name, composed *forward* through the same rule that named the
        # parameters, never split back out of one: a term may legitimately contain a
        # bracket, and a name parsed back would then be filed under a term that does not
        # exist rather than failing.
        from ..targets import parameter_name

        ratio = "risk ratio" if msm.link == "log" else "odds ratio"
        rows: list[dict[str, Any]] = []
        for column, term in enumerate(msm.terms):
            estimate = self.estimates[parameter_name("msm", arm=term)]
            if scale == "link":
                rows.append(estimate.to_dict())
                continue
            # scale="ratio" with log_psi=beta *is* the exponentiated view: psi becomes
            # exp(beta), the interval is built on the log scale and exponentiated, and the
            # null moves from zero to one. No arithmetic of its own -- see
            # cleverly.inference.ParameterEstimate.ci.
            exponentiated = replace(
                estimate, psi=float(np.exp(estimate.psi)), log_psi=estimate.psi, scale="ratio"
            )
            row = exponentiated.to_dict()
            # An intercept is found by its column being constant one at every arm, not by
            # its name: the term names are the user's own, and "(intercept)" is a
            # convention rather than a promise.
            row["scale"] = "baseline" if _is_intercept(msm.design[:, :, column]) else ratio
            rows.append(row)
        return self.data.frame_like({key: [row[key] for row in rows] for key in rows[0]})

    def influence_frame(self) -> Any:
        """One column per estimand of per-observation influence-curve values."""
        return self.data.frame_like(
            {name: estimate.influence_curve for name, estimate in self.estimates.items()}
        )

    def summary(self) -> str:
        """A printable report of the fit."""
        data = self.data
        header = ["Targeted maximum likelihood estimation", "=" * 38]
        facts = [
            f"n = {data.n}"
            + (
                f" ({int(data.observed.sum())} with an observed outcome)"
                if data.has_missing_outcome
                else ""
            )
            + f"; covariates = {data.n_covariates}; {_arm_shares(data)}",
        ]
        if self.identified_effect is not None:
            facts.extend(self.identified_effect.summary_lines())
        if data.cluster is not None:
            facts.append(f"clusters = {data.n_clusters} (cluster-robust variance)")
        if data.is_weighted:
            report = data.weight_report()
            facts.append(
                f"observation weights ({report.name or 'weights'}, "
                + ("estimated" if report.estimated else "fixed")
                + f"): effective n = {report.effective_n:.1f}, "
                f"design effect = {report.design_effect:.2f}"
            )
            facts.append(
                "estimand: the parameter in the weight-tilted population dP_w = w dP / E[w]; "
                "see result.data.weight_report()"
            )
        if self.intermediate_value is not None:
            facts.append(describe_direct_effect(self.intermediate_value, data.intermediate_name))
        facts.extend(self.config.describe())
        if self.provenance is not None:
            facts.extend(self.provenance.describe())

        level = f"{(1 - self.config.alpha_sig) * 100:g}%"
        rows = []
        for name, estimate in self.estimates.items():
            low, high = estimate.ci
            rows.append(
                [
                    name,
                    f"{estimate.psi:.5g}",
                    f"{estimate.std_error:.4g}",
                    f"[{low:.5g}, {high:.5g}]",
                    format_pvalue(estimate.pvalue),
                ]
            )
        table = format_table(["estimand", "psi", "std_err", f"{level} CI", "p_value"], rows)

        parts = [*header, *facts, "", table]
        if self.n_repeats > 1:
            # Beside the standard error rather than in a separate report, because the
            # comparison is the whole content of the number: on its own "0.0065" says
            # nothing about whether the split mattered.
            parts.append("")
            parts.append(
                f"split noise -- sd(psi) across the {self.n_repeats} draws, a diagnostic "
                "and not a standard error:"
            )
            for name, value in self.repeat_spread().items():
                error = self[name].std_error
                share = f"{value / error:.0%} of std_err" if error > 0 else "std_err unavailable"
                parts.append(f"  {name:<5s} {value:.4g}  ({share})")
        if self.simultaneous is not None:
            parts.append("")
            parts.append(
                f"simultaneous {level} bands (multiplier bootstrap, "
                f"critical value {self.simultaneous.critical_value:.3f} vs "
                f"{self.simultaneous.pointwise_critical_value:.3f} pointwise):"
            )
            for name, (low, high) in self.simultaneous.bands.items():
                parts.append(f"  {name:<5s} [{low:.5g}, {high:.5g}]")
        if self.bootstrap is not None:
            parts.append("")
            parts.append(
                f"targeted bootstrap ({self.bootstrap.resampling} resampling, "
                f"{self.bootstrap.n_requested - self.bootstrap.n_failed} usable replicates):"
            )
            for name, estimate in self.estimates.items():
                if estimate.bootstrap is None:
                    continue
                low, high = estimate.bootstrap.ci
                parts.append(
                    f"  {name:<5s} se {estimate.bootstrap.std_error:.4g}  "
                    f"percentile CI [{low:.5g}, {high:.5g}]"
                )
        # Last, and only when it failed. An interval whose score equation is unsolved is
        # not a wider interval, it is one the theory does not license, and until now the
        # only way to find that out was to know that `validation.score_check()` existed --
        # documentation standing in for reporting. Silent on a passing fit deliberately:
        # every transcript in the README and the guide is a passing fit, and a line there
        # would be noise on the common path and easier to stop reading.
        verdict = self.score_verdict
        if not verdict.passed:
            parts.append("")
            parts.append(verdict.one_line())
        return "\n".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


@dataclass(frozen=True)
class TMLEResultSet(Mapping["float | None", TMLEResult]):
    """What :meth:`~cleverly.TMLE.fit` returns: one result per parameter it estimated.

    A controlled direct effect is defined *per* value of the intermediate: "the
    effect of ``A`` on ``Y`` holding ``Z`` fixed at ``z``" is a different quantity for
    each ``z``.  Fitting with ``intermediate=`` therefore yields one
    :class:`TMLEResult` per level, exactly as R's ``tmle()`` returns a ``tmle.list``.

    ``fit`` returns this **always**, and an ordinary fit -- no intermediate variable, one
    parameter -- is the single-entry case, keyed ``None``.  It used to return
    ``TMLEResult | TMLEResultSet``, which pushed an ``isinstance`` check onto every caller
    that could not know in advance which it would get; :class:`CoverageStudy
    <cleverly.validation.simulation.CoverageStudy>` carried one, and any user code
    branching on ``intermediate=`` carried another.  ``None`` is the key rather than an
    invented level because it is what the estimator already calls the absence of one.

    Use :meth:`single` for the ordinary case.  Attribute access is deliberately *not*
    forwarded to a lone result: ``__getitem__`` is taken by the level key and cannot also
    mean an estimand name, and implicit forwarding would be invisible to a type checker.

    The two levels of a controlled direct effect do *not* decompose the total effect into
    direct and indirect parts, and their difference is an interaction contrast rather than
    a mediated effect.  :mod:`cleverly.estimators.direct_effect` writes the parameter
    down, derives its influence function, and states the assumptions -- in particular the
    one that separates a controlled direct effect from an average treatment effect, that
    ``W`` suffices to deconfound ``Z -> Y`` as well as ``A -> Y``.
    """

    results: dict[float | None, TMLEResult]
    intermediate_name: str | None = None

    @property
    def levels(self) -> tuple[float | None, ...]:
        """The keys, ascending, with the no-intermediate key first."""
        return tuple(sorted(self.results, key=_level_order))

    def single(self) -> TMLEResult:
        """The sole result.

        Raises when the set holds more than one, because there is no defensible choice
        between two controlled direct effects: they are different parameters, and picking
        one silently is how a script ends up reporting the effect at ``Z = 0`` while its
        author believes they asked about ``Z = 1``.
        """
        if len(self.results) != 1:
            raise KeyError(
                f"this fit produced {len(self.results)} results, one per level of "
                f"{self.intermediate_name or 'the intermediate'} ({list(self.levels)}), so "
                "there is no single one to return. A controlled direct effect is a "
                "different parameter at each level -- index the level you mean, e.g. "
                f"result[{self.levels[0]!r}]."
            )
        return next(iter(self.results.values()))

    def __getitem__(self, value: float | None) -> TMLEResult:
        if isinstance(value, str):
            # The likeliest mistake, and worth catching by name: ``TMLEResult`` is indexed
            # by estimand and this is indexed by level, so a caller who has not noticed
            # which of the two they hold reaches for res["ate"] first.  Left to itself,
            # float() reports 'could not convert string to float', which says nothing
            # about what went wrong.
            raise KeyError(
                f"a result *set* is indexed by intermediate level, not by estimand: "
                f"{value!r} is an estimand name. Use .single()[{value!r}] for an ordinary "
                f"fit, or result[<level>][{value!r}] for a controlled direct effect."
            )
        key = None if value is None else float(value)
        try:
            return self.results[key]
        except KeyError:
            raise KeyError(
                f"no result for {self.intermediate_name or 'intermediate'} = {value}; "
                f"available: {list(self.levels)}"
            ) from None

    def __iter__(self) -> Iterator[float | None]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self.results)

    def to_frame(self) -> Any:
        """Stacked results with an intermediate-value column."""
        frames = [self.results[value].to_frame() for value in self.levels]
        if len(frames) == 1:
            return frames[0]
        import narwhals as nw

        stacked = nw.concat([nw.from_native(frame, eager_only=True) for frame in frames])
        return stacked.to_native()

    def summary(self) -> str:
        """The results, headed by level -- or bare, when there is only one.

        A single-entry set prints exactly what its result prints.  Heading it
        ``--- None = ... ---`` would put the internal sentinel in front of every ordinary
        fit's output.
        """
        if len(self.results) == 1 and self.levels[0] is None:
            return self.single().summary()
        blocks = []
        for value in self.levels:
            label = (
                "no intermediate" if value is None else f"{self.intermediate_name} = {value:.0f}"
            )
            blocks.append(f"--- {label} ---")
            blocks.append(self.results[value].summary())
        return "\n\n".join(blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _level_order(value: float | None) -> tuple[int, float]:
    """Sort key that tolerates the ``None`` an ordinary fit is keyed by.

    ``sorted`` on a mix of ``None`` and floats raises, and a set is only ever homogeneous
    by construction -- but relying on that would make the ordering an invariant enforced
    nowhere.
    """
    return (0, 0.0) if value is None else (1, float(value))


def _is_intercept(column: FloatArray) -> bool:
    """Whether a design column is the constant one at every arm and every row."""
    return bool(np.all(np.asarray(column, dtype=float) == 1.0))


def _arm_shares(data: CausalData) -> str:
    """How the treatment is distributed, for the summary header.

    ``P(A=1)`` for a binary treatment, unchanged.  For more arms that number is not
    just uninformative but wrong-looking -- it is the mean of the arm *codes*, so a
    three-armed fit reported ``P(A=1) = 0.98`` -- so every arm's share is listed under
    its own label instead.  A continuous treatment has no arms to take the share of, so
    it reports the dose's range: an empty ``arm shares:`` reads as a broken table rather
    than as an inapplicable question.
    """
    if data.is_continuous_treatment:
        dose = np.asarray(data.treatment, dtype=float)
        mean = float(np.average(dose, weights=data.weights))
        return f"dose: mean {mean:.4g}, range [{dose.min():.3g}, {dose.max():.3g}]"
    if data.is_binary_treatment:
        return f"P(A=1) = {data.treated_fraction:.4g}"

    def share(arm: float) -> float:
        return float(np.average(data.treatment == arm, weights=data.weights))

    shares = [f"{data.arm_label(arm)}={share(arm):.3g}" for arm in data.arm_codes]
    return f"arm shares: {', '.join(shares)}"


def attach_bootstrap(result: TMLEResult, bootstrap: BootstrapResult) -> TMLEResult:
    """Return a copy of ``result`` with bootstrap summaries attached."""
    alpha = result.config.alpha_sig
    estimates = {
        name: (
            estimate.with_bootstrap(bootstrap.summary(name, alpha))
            if name in bootstrap.draws
            else estimate
        )
        for name, estimate in result.estimates.items()
    }
    return replace(result, estimates=estimates, bootstrap=bootstrap)
