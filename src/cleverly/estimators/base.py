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
from ..exceptions import CleverlyError
from ..fluctuation.iterative import Fluctuation
from ..inference.bootstrap import BootstrapResult
from ..inference.cluster import influence_covariance
from ..inference.delta import delta_method
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.multiplier import SimultaneousBands
from ..learners.crossfit import CrossFitPlan
from ..provenance import Provenance
from ..targets import TARGETS, all_names, resolve_estimands
from ._nuisance import NuisanceEstimates, RepeatFit
from .direct_effect import describe as describe_direct_effect
from .targeting import TargetingSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..sensitivity.api import SensitivityAnalysis
    from ..validation.api import ValidationSuite

__all__ = [
    "ALL_ESTIMANDS",
    "DEFAULT_ESTIMANDS",
    "MEAN_GROUP_ESTIMANDS",
    "CVTargeting",
    "TMLEConfig",
    "TMLEResult",
    "TMLEResultSet",
    "format_table",
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

        Worth naming rather than reporting the raw settings: pooled targeting on
        cross-fitted nuisances and CV-TMLE are different estimators with different
        asymptotic arguments, and only the last of these is the canonical CV-TMLE of
        Zheng & van der Laan (2011).
        """
        if not self.cross_fit:
            return "TMLE (in-sample nuisances)"
        if self.targeting_scheme != "fold":
            return "cross-fitted TMLE"
        return "canonical CV-TMLE" if self.cv_evaluation else "fold-targeted CV-TMLE"

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
    """Fold-level detail from a fit that targeted fold by fold.

    Each fold gets its own ``epsilon``, fit against nuisance predictions from models
    trained on the other folds, and its own plug-in estimate.  Keeping the pieces is the
    point: fold estimates that disagree far more than their standard errors allow, or an
    ``epsilon`` that changes sign between folds, mean the fluctuation is being driven by
    a handful of extreme clever-covariate values rather than by the sample.

    The two reports here are genuinely different estimators, not two views of one.
    :attr:`pooled` stitches the fold-targeted predictions back together and evaluates the
    estimand once over the whole sample; :attr:`canonical` evaluates it inside each fold
    and averages, which is the construction Zheng & van der Laan (2011) analyse.  They
    coincide exactly for estimands linear in the targeted predictions at equal fold
    sizes -- ``ate``, ``ey1``, ``ey0`` -- and diverge for ``rr``, ``or``, ``att`` and
    ``atc``, where a ratio of means is not a mean of ratios and the pooled conditional
    effects weight by the whole sample's arm share rather than each fold's.  Which one
    ``result[name]`` carries is set by ``TMLE(cv_evaluation=...)``; both are always here.

    Attributes
    ----------
    variance, std_error:
        The cross-validated variance of Zheng & van der Laan (2011) -- the fold-averaged
        second moment of the fold-specific influence curves -- per estimand.  This is the
        standard error attached to :attr:`canonical`; the pooled report carries the
        ordinary influence-curve one.  The two agree when the folds are balanced and the
        score equation is solved, so a gap between them is itself informative.
    fold_estimates:
        Per-estimand tuple of fold-specific plug-in estimates.  Estimands that some fold
        could not evaluate (no units in the conditioning arm, a boundary counterfactual
        mean) are absent, and their omission was warned about at fit time.
    fold_epsilon:
        Per-targeting-group tuple of that fold's fluctuation coefficients.
    pooled, canonical:
        The two reports, per estimand.
    """

    n_folds: int
    fold_sizes: tuple[int, ...]
    variance: dict[str, float]
    fold_estimates: dict[str, tuple[float, ...]]
    fold_epsilon: dict[str, tuple[tuple[float, ...], ...]]
    pooled: dict[str, ParameterEstimate] = field(default_factory=dict)
    canonical: dict[str, ParameterEstimate] = field(default_factory=dict)

    @property
    def std_error(self) -> dict[str, float]:
        """Cross-validated standard error per estimand."""
        return {name: float(np.sqrt(value)) for name, value in self.variance.items()}

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
        if data is None:
            return payload
        return data.frame_like(payload)

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
                "canonical",
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
            "canonical: evaluated fold by fold and averaged (Zheng & van der Laan).",
            "pooled: the fold-targeted fits stitched together, evaluated once.",
            "",
        ]
        epsilon_lines = ["", "fluctuation coefficients by fold:"]
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

    def psi(self, name: str) -> float:
        """Point estimate for one estimand."""
        return self[name].psi

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
        """Fold-level detail, when the fit used ``targeting_scheme="fold"``.

        Carries both the pooled and the canonical CV-TMLE report whichever one
        ``result[name]`` was configured to show, so the two can always be compared.
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
    def sensitivity(self) -> SensitivityAnalysis:
        """Sensitivity analyses for this fit -- see :mod:`cleverly.sensitivity`."""
        from ..sensitivity.api import SensitivityAnalysis

        return SensitivityAnalysis(self)

    @cached_property
    def validation(self) -> ValidationSuite:
        """Validation diagnostics for this fit -- see :mod:`cleverly.validation`."""
        from ..validation.api import ValidationSuite

        return ValidationSuite(self)

    # ---------------------------------------------------------------- output

    def save(self, path: Any) -> Any:
        """Write this result to a single ``.npz`` file; see :func:`cleverly.load`.

        Arrays plus JSON -- no pickle, so the file does not depend on the exact
        scikit-learn version that wrote it and is not an execution vector.  After a
        round trip everything reached through :meth:`~cleverly.TMLE.retarget` works;
        the two analyses that genuinely refit need the learners to have been library
        specifications rather than fitted objects.
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
                    _format_pvalue(estimate.pvalue),
                ]
            )
        table = format_table(["estimand", "psi", "std_err", f"{level} CI", "p_value"], rows)

        parts = [*header, *facts, "", table]
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


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width table without pulling in a dataframe dependency."""
    columns: list[Sequence[str]] = (
        list(zip(*([list(headers)] + [list(row) for row in rows]), strict=True))
        if rows
        else [[header] for header in headers]
    )
    widths = [max(len(str(cell)) for cell in column) for column in columns]
    lines = [
        "  ".join(str(header).ljust(width) for header, width in zip(headers, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True))
        )
    return "\n".join(lines)


def _format_pvalue(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "nan"
    if pvalue < 1e-4:
        return "<1e-4"
    return f"{pvalue:.4f}"


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


class EstimationError(CleverlyError):
    """Raised when a fit cannot proceed for a statistical (not user-input) reason."""
