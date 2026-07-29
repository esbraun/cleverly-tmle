"""Shared estimator infrastructure: configuration, results and reporting.

Every TMLE variant in this library produces the same shape of result -- a set of
:class:`~cleverly.inference.ParameterEstimate` objects plus the nuisance fits and
targeting details that produced them -- so sensitivity analysis, validation and
reporting are written once, here, rather than per estimator.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import Estimand, FloatArray
from ..data.causal_data import CausalData
from ..exceptions import CleverlyError
from ..fluctuation.iterative import Fluctuation
from ..inference.bootstrap import BootstrapResult
from ..inference.influence import ParameterEstimate
from ..inference.multiplier import SimultaneousBands
from ._nuisance import NuisanceEstimates

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

#: Estimands available from a classic point-treatment fit, in report order.
ALL_ESTIMANDS: tuple[Estimand, ...] = ("ate", "att", "atc", "ey1", "ey0", "rr", "or")

#: Estimands produced by the two-column ``mean`` fluctuation.
MEAN_GROUP_ESTIMANDS: frozenset[str] = frozenset({"ate", "ey1", "ey0", "rr", "or"})

#: The default set: the ATE family plus the counterfactual means.  ``rr``/``or`` are
#: added automatically for a binary outcome, where they are defined.
DEFAULT_ESTIMANDS: tuple[Estimand, ...] = ("ate", "att", "atc", "ey1", "ey0")

_RATIO_ESTIMANDS: frozenset[str] = frozenset({"rr", "or"})


def resolve_estimands(
    requested: Sequence[str] | str | None,
    family: str,
) -> tuple[str, ...]:
    """Normalise and validate a requested estimand list.

    ``"all"`` expands to everything the outcome type supports.  Ratios are dropped
    with an explanation for a continuous outcome: a risk ratio of two means that can
    be negative is not a meaningful quantity.
    """
    if requested is None:
        names: tuple[str, ...] = DEFAULT_ESTIMANDS
        if family == "binomial":
            names = (*names, "rr", "or")
    elif isinstance(requested, str):
        if requested == "all":
            names = ALL_ESTIMANDS if family == "binomial" else DEFAULT_ESTIMANDS
        else:
            names = (requested,)
    else:
        names = tuple(requested)

    unknown = [name for name in names if name not in ALL_ESTIMANDS]
    if unknown:
        raise ValueError(f"unknown estimand(s) {unknown}; choose from {list(ALL_ESTIMANDS)}")

    if family != "binomial":
        ratios = [name for name in names if name in _RATIO_ESTIMANDS]
        if ratios:
            raise ValueError(
                f"estimand(s) {ratios} require a binary outcome (family='binomial'); the risk "
                "ratio and odds ratio are not defined for a continuous outcome. Drop them or "
                "dichotomise the outcome."
            )

    ordered = tuple(name for name in ALL_ESTIMANDS if name in set(names))
    if not ordered:
        raise ValueError("no estimands requested")
    return ordered


@dataclass(frozen=True)
class TMLEConfig:
    """A snapshot of the settings a fit actually used.

    Recorded on the result so a reported number can always be traced back to the
    bounds, folds and submodel that produced it -- including values the estimator
    resolved itself, such as ``g_bounds="auto"``.
    """

    family: str
    fluctuation: str
    targeting: str
    targeting_scheme: str
    cross_fit: bool
    n_folds: int
    g_bounds: tuple[float, float]
    g_bounds_conditional: tuple[float, float]
    missingness_bound: float
    q_bounds: tuple[float, float] | None
    alpha: float
    target_weights: bool
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
        else:
            lines.append(f"{self.estimator_name}: nuisances fitted in-sample (cross_fit=False)")
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
    """

    estimates: dict[str, ParameterEstimate]
    fluctuations: dict[str, Fluctuation]
    nuisance: NuisanceEstimates
    data: CausalData
    config: TMLEConfig
    estimator: Any = None
    simultaneous: SimultaneousBands | None = None
    bootstrap: BootstrapResult | None = None
    intermediate_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

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
            + f"; covariates = {data.n_covariates}; P(A=1) = {data.treated_fraction:.4g}",
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
            facts.append(
                f"controlled direct effect at {data.intermediate_name or 'Z'} = "
                f"{self.intermediate_value:.0f}"
            )
        facts.extend(self.config.describe())

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
class TMLEResultSet(Mapping[float, TMLEResult]):
    """Results for each level of an intermediate variable.

    A controlled direct effect is defined *per* value of the intermediate: "the
    effect of ``A`` on ``Y`` holding ``Z`` fixed at ``z``" is a different quantity for
    each ``z``.  Fitting with ``intermediate=`` therefore yields one
    :class:`TMLEResult` per level, exactly as R's ``tmle()`` returns a ``tmle.list``.
    """

    results: dict[float, TMLEResult]
    intermediate_name: str

    def __getitem__(self, value: float) -> TMLEResult:
        try:
            return self.results[float(value)]
        except KeyError:
            raise KeyError(
                f"no result for {self.intermediate_name} = {value}; "
                f"available: {sorted(self.results)}"
            ) from None

    def __iter__(self) -> Iterator[float]:
        return iter(sorted(self.results))

    def __len__(self) -> int:
        return len(self.results)

    def to_frame(self) -> Any:
        """Stacked results with an intermediate-value column."""
        frames = [self.results[value].to_frame() for value in sorted(self.results)]
        if len(frames) == 1:
            return frames[0]
        import narwhals as nw

        stacked = nw.concat([nw.from_native(frame, eager_only=True) for frame in frames])
        return stacked.to_native()

    def summary(self) -> str:
        blocks = []
        for value in sorted(self.results):
            blocks.append(f"--- {self.intermediate_name} = {value:.0f} ---")
            blocks.append(self.results[value].summary())
        return "\n\n".join(blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width table without pulling in a dataframe dependency."""
    columns = (
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
