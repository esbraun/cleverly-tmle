r"""CV-TMLE of the kernel-smoothed dose-response curve for a continuous treatment.

:class:`DoseResponseTMLE` fits the outcome regression and the conditional treatment density
once, out of fold, predicts the regression at every quadrature dose of the declaration's
composite rule, and then targets each declared ``(bandwidth, dose)`` pair with its own pooled
one-dimensional fluctuation (:func:`cleverly.dose_response.target_pair`).  The pairs' influence
curves are stacked, so the result carries a pointwise interval for each pair and a simultaneous
band over the whole declared set.

The band covers every declared pair at once.  It therefore also covers the smoothed curve at a
bandwidth chosen from the declared set after looking at the results; a pointwise interval read
at such a bandwidth does not.  No interval here is for the unsmoothed curve
:math:`\theta(a_0)`, which is not pathwise differentiable (roadmap F39).

See ``docs/technical-reference/dose-response.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import numpy as np

from .._inference_status import STATUS_RECORDS
from .._support import Support, combine_support
from .._typing import FloatArray
from .._warnings import warn
from ..data import CausalData
from ..dose_response import (
    DoseResponse,
    QuadratureRule,
    pair_name,
    pair_support,
    target_pair,
)
from ..exceptions import CapabilityError, DataError, DataWarning, RefusalReason
from ..inference.influence import ParameterEstimate, make_estimate
from ..inference.multiplier import DEFAULT_N_REPLICATES, SimultaneousBands, simultaneous_bands
from ..learners.density import ConditionalDensity, bin_edges, default_density_bins
from ..utils.frames import frame_from_dict
from ._nuisance import NuisanceEstimates, fit_nuisances

__all__ = [
    "DOSE_RESPONSE_BAND_MEASURED",
    "DOSE_RESPONSE_MEASURED",
    "DOSE_RESPONSE_STUDY",
    "DoseResponseResult",
    "DoseResponseTMLE",
    "dose_response_support",
]

#: The memory the node predictions may use before the fit refuses, in bytes.
NODE_MEMORY_BUDGET: Final[int] = 4 * 1024**3

#: The doubling diagnostic warns when doubling the nodes moves an estimate by more than this
#: fraction of its standard error.
QUADRATURE_TOLERANCE: Final[float] = 0.05


def _pair_cell_name(h: float, a0: float, stratum: str | None) -> str:
    """The parameter name of one pair in one stratum, as the fit declares it."""
    return pair_name(h, a0) if stratum is None else f"{pair_name(h, a0)}[{stratum}]"


@dataclass(frozen=True)
class PairSupport:
    """The support of one declared pair.

    Parameters
    ----------
    name : str
        The pair's parameter name.
    bandwidth : float
        The declared bandwidth.
    dose : float
        The declared dose.
    stratum : str or None
        The baseline stratum's label, or ``None`` on a fit without strata.
    rows : int
        Rows whose observed dose gets a nonzero weight.
    kish : float
        The Kish effective size of the clever covariate.
    min_density : float
        The smallest fitted density over the loaded observed doses and the pair's nodes.
    max_covariate : float
        The largest absolute clever covariate.
    below_floor : bool
        Whether :attr:`kish` is below the declaration's floor.
    """

    name: str
    bandwidth: float
    dose: float
    stratum: str | None
    rows: int
    kish: float
    min_density: float
    max_covariate: float
    below_floor: bool


@dataclass(frozen=True)
class DoseResponseResult:
    """A fitted kernel-smoothed dose-response curve.

    Parameters
    ----------
    curve : DoseResponse
        The declaration.
    estimates : dict
        One :class:`~cleverly.inference.ParameterEstimate` per pair, keyed by name.
    band : SimultaneousBands
        The simultaneous band over every declared pair.
    pair_support : tuple of PairSupport
        One entry per reported estimate, in report order: every pair, and within a fit with
        baseline strata every pair in every stratum.
    quadrature_shift : float
        The largest change of an estimate, relative to its standard error, when the
        quadrature nodes are doubled.
    scores : dict
        The mean weighted score of each pair's fluctuation at its solution.
    n_nodes : int
        The quadrature doses.
    alpha : float
        The significance level of the intervals and of the band.
    warnings : tuple of str
        The fit's notes: a quadrature-doubling shift above tolerance, an accepted
        composition no registered study measures (an in-sample fit, observation weights),
        and pairs below the support floor.
    backend : str or None
        The dataframe library the fit's data arrived in, which :meth:`to_frame` returns by
        default.
    """

    curve: DoseResponse
    estimates: dict[str, ParameterEstimate]
    band: SimultaneousBands
    pair_support: tuple[PairSupport, ...]
    quadrature_shift: float
    scores: dict[str, float]
    n_nodes: int
    alpha: float
    warnings: tuple[str, ...] = field(default=())
    backend: str | None = None

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    def estimate(self, h: float, a0: float, stratum: str | None = None) -> ParameterEstimate:
        """The estimate at one declared pair.

        Parameters
        ----------
        h : float
            A declared bandwidth.
        a0 : float
            A declared dose.
        stratum : str or None
            The stratum label, such as ``"V=1"``, on a fit with baseline strata.

        Returns
        -------
        ParameterEstimate
            The estimate.

        Raises
        ------
        KeyError
            When no declared pair has exactly this bandwidth and dose, and no declared pair
            has the parameter name :func:`cleverly.dose_response.pair_name` gives them.

        Notes
        -----
        A pair matches by exact float equality first, then by its declared name, which
        formats each number to ten significant digits and is unique within a declaration.
        No tolerance is applied: two declared doses can be close in relative terms and
        still be different estimands.
        """
        cells = [cell for cell in self.pair_support if cell.stratum == stratum]
        for cell in cells:
            if cell.bandwidth == h and cell.dose == a0:
                return self.estimates[cell.name]
        named = [cell for cell in cells if cell.name == _pair_cell_name(h, a0, stratum)]
        if len(named) == 1:
            return self.estimates[named[0].name]
        raise KeyError(
            f"no declared pair with bandwidth {h!r}, dose {a0!r} and stratum {stratum!r}"
        )

    def to_frame(self, backend: str | None = None) -> Any:
        """One row per pair: the estimate, its pointwise interval, the band and the support.

        Parameters
        ----------
        backend : str or None
            The dataframe library, or ``None`` for the library the fit's data arrived in.

        Returns
        -------
        DataFrame
            Columns ``bandwidth``, ``dose``, ``stratum`` (on a fit with strata),
            ``estimate``, ``std_error``, ``lower``, ``upper``, ``band_lower``,
            ``band_upper``, ``rows``, ``kish`` and ``low_support``.
        """
        columns: dict[str, list[Any]] = {
            key: []
            for key in (
                "bandwidth",
                "dose",
                "estimate",
                "std_error",
                "lower",
                "upper",
                "band_lower",
                "band_upper",
                "rows",
                "kish",
                "low_support",
            )
        }
        stratified = any(cell.stratum is not None for cell in self.pair_support)
        if stratified:
            columns = {"bandwidth": [], "dose": [], "stratum": [], **columns}
        for support in self.pair_support:
            estimate = self.estimates[support.name]
            lower, upper = estimate.ci
            band_lower, band_upper = self.band.bands[support.name]
            columns["bandwidth"].append(support.bandwidth)
            columns["dose"].append(support.dose)
            if stratified:
                columns["stratum"].append(support.stratum)
            columns["estimate"].append(estimate.psi)
            columns["std_error"].append(estimate.std_error)
            columns["lower"].append(lower)
            columns["upper"].append(upper)
            columns["band_lower"].append(band_lower)
            columns["band_upper"].append(band_upper)
            columns["rows"].append(support.rows)
            columns["kish"].append(support.kish)
            columns["low_support"].append(support.below_floor)
        return frame_from_dict(columns, backend=backend or self.backend)

    def save(self, path: Any) -> Any:
        """Write this result to a trusted joblib artifact; see :func:`cleverly.load`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path, as given.
        """
        from .serialize import save as _save

        return _save(self, path)

    def __repr__(self) -> str:
        return (
            f"DoseResponseResult(pairs={len(self.curve.pairs)}, "
            f"bandwidths={len(self.curve.bandwidths)}, doses={len(self.curve.doses)})"
        )


#: A dose with fewer distinct values than this is refused as discrete.
MIN_DISTINCT_DOSES: Final[int] = 20


def _refuse_unsupported(data: CausalData) -> None:
    if not data.is_continuous_treatment or data.treatment_kind == "vector":
        raise CapabilityError(
            "a dose-response curve needs one continuous treatment; pass "
            'treatment_kind="continuous"',
            reason=RefusalReason.INVALID_DECLARATION,
        )
    distinct = np.unique(np.asarray(data.treatment, dtype=float)).size
    if distinct < MIN_DISTINCT_DOSES:
        raise CapabilityError(
            f"the dose takes only {distinct} distinct values: a smoothed curve integrates the "
            "outcome regression over every dose in the interval, and a discrete dose has no "
            "density between its values. Fit the levels as arms with TMLE instead.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if data.cluster is not None:
        raise CapabilityError(
            "a dose-response curve does not yet support clustered rows: its intervals and "
            "band would need the package's few-cluster rule (roadmap X95).",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit without id= only if the rows are independent.",
            roadmap="X95",
        )
    if data.known_treatment is not None:
        raise CapabilityError(
            "a dose-response curve fits its own dose density; a declared treatment mechanism "
            "is not supported",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit without the declared mechanism.",
            roadmap="X95",
        )
    if data.has_missing_outcome or data.has_intermediate or data.has_missing_treatment:
        raise CapabilityError(
            "a dose-response curve does not yet support a missing outcome, a missing "
            "treatment or an intermediate variable: the second mechanism would have to be "
            "evaluated at every quadrature dose.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit the complete cases only if missingness is ignorable.",
            roadmap="X10",
        )
    if data.dose_atoms:
        raise CapabilityError(
            "a dose-response curve on a treatment with declared dose atoms is not supported",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Declare the dose without atoms.",
            roadmap="X95",
        )


#: The registered study that measures :class:`DoseResponseTMLE`.
DOSE_RESPONSE_STUDY: Final[str] = "dose-response"

#: The configurations the registered ``dose-response`` study measures, keyed by
#: ``(outcome, fluctuation, strata)``, each with the scenario that measures it.  Every scenario
#: fits cross-fitted nuisances on iid, unweighted rows with complete outcomes.  A configuration
#: outside this table, an in-sample fit or observation weights read experimental.
DOSE_RESPONSE_MEASURED: Final[Mapping[tuple[str, str, bool], str]] = {
    ("binary", "logistic", False): "binary, tail",
    ("binary", "logistic", True): "strata",
    ("continuous", "linear", False): "gaussian",
}

#: The one configuration whose simultaneous band the study measures (the band cell, on 15
#: pairs of the binary declaration).  A band of any other configuration reads experimental.
DOSE_RESPONSE_BAND_MEASURED: Final[tuple[str, str, bool]] = ("binary", "logistic", False)

#: The multiplier draws the band cell measured: the Mammen multiplier at the default draw count,
#: as band-finite-sample Wave 2 regenerated it.  The band stays on the identity scale, because
#: the curve's estimates carry no range (:func:`make_estimate` gets no ``bounds``), so the
#: default ``band_scale="logit"`` has nothing to map and its limits can leave [0, 1].  A band
#: with fewer draws reads experimental.
DOSE_RESPONSE_BAND_DRAWS: Final[int] = DEFAULT_N_REPLICATES


def dose_response_support(
    *,
    outcome: str,
    fluctuation: str,
    strata: bool,
    cross_fitted: bool,
    weighted: bool,
    n_replicates: int = DEFAULT_N_REPLICATES,
) -> tuple[Support, Support]:
    """The support of a dose-response fit's pointwise intervals and of its band.

    A fit's intervals read ``supported`` when its configuration is one of
    :data:`DOSE_RESPONSE_MEASURED` and it is cross-fitted and unweighted.  Its band reads
    ``supported`` when, in addition, the configuration is :data:`DOSE_RESPONSE_BAND_MEASURED`
    and the band uses at least :data:`DOSE_RESPONSE_BAND_DRAWS` multiplier draws.  Otherwise
    each reads ``experimental``, with every reason that applies.  A pair below the support
    floor is experimental whatever this returns (status ``low_dose_support_plugin``).

    Parameters
    ----------
    outcome : {"binary", "continuous"}
        The outcome type.
    fluctuation : {"logistic", "linear"}
        The submodel on the scaled outcome.
    strata : bool
        Whether the fit reports one curve per baseline stratum.
    cross_fitted : bool
        Whether the nuisances are fitted out of fold.
    weighted : bool
        Whether the rows carry observation weights.
    n_replicates : int, default=10_000
        The multiplier draws behind the band.

    Returns
    -------
    tuple of Support
        The support of each pair's estimate, then the support of the band.

    Examples
    --------
    >>> from cleverly.estimators.dose_response import dose_response_support
    >>> pairs, band = dose_response_support(
    ...     outcome="continuous", fluctuation="linear", strata=False,
    ...     cross_fitted=True, weighted=False,
    ... )
    >>> pairs.label, band.label
    ('supported', 'experimental')
    """
    configuration = (outcome, fluctuation, bool(strata))
    reasons: list[str] = []
    if not cross_fitted:
        reasons.append(
            "The fit is in-sample, and the registered study measures cross-fitted fits only."
        )
    if weighted:
        reasons.append("No registered study measures a dose-response curve with weighted rows.")
    if configuration not in DOSE_RESPONSE_MEASURED:
        strata_words = " with baseline strata" if strata else ""
        reasons.append(
            f"No registered study measures a {outcome} outcome with the {fluctuation} "
            f"fluctuation{strata_words}: the dose-response study measures a binary outcome with "
            "the logistic fluctuation, with and without strata, and a continuous outcome with "
            "the linear fluctuation."
        )
    limits = (
        "One law family on a declared interval, with declared sieve learners; the default "
        "learner library is not measured.",
        "Sample sizes 1,000 to 8,000.",
    )
    if reasons:
        experimental = Support("experimental", " ".join(reasons), limits=limits)
        return experimental, experimental
    scenario = DOSE_RESPONSE_MEASURED[configuration]
    pairs = Support(
        "supported",
        f"The registered {DOSE_RESPONSE_STUDY} study measures this configuration "
        f"(scenario {scenario}).",
        (DOSE_RESPONSE_STUDY,),
        limits=limits,
    )
    if configuration == DOSE_RESPONSE_BAND_MEASURED:
        if n_replicates >= DOSE_RESPONSE_BAND_DRAWS:
            return pairs, Support(
                "supported",
                f"The registered {DOSE_RESPONSE_STUDY} study measures this band (scenario "
                f"{scenario}): the Mammen multiplier at {DOSE_RESPONSE_BAND_DRAWS:,} draws on "
                "the identity scale, because the curve's estimates carry no range.",
                (DOSE_RESPONSE_STUDY,),
                limits=(
                    *limits,
                    "The band is built on the identity scale, so its limits can leave [0, 1].",
                ),
            )
        return pairs, Support(
            "experimental",
            f"The registered {DOSE_RESPONSE_STUDY} study measured this band at "
            f"{DOSE_RESPONSE_BAND_DRAWS:,} multiplier draws, and this band uses "
            f"{n_replicates:,}.",
            limits=limits,
        )
    band = Support(
        "experimental",
        "The registered dose-response study measures the simultaneous band on a binary outcome "
        "with the logistic fluctuation and no strata only.",
        limits=limits,
    )
    return pairs, band


def _pair_support(configured: Support, below_floor: bool) -> Support:
    """A pair's support: the configuration's, made experimental below the support floor."""
    if not below_floor:
        return configured
    record = STATUS_RECORDS["low_dose_support_plugin"]
    assert record.shown_reason is not None
    reasons = (configured.reason,) if configured.label == "experimental" else ()
    evidence = configured.evidence if configured.label == "experimental" else ()
    return Support(
        "experimental",
        " ".join([*reasons, record.shown_reason]),
        tuple(dict.fromkeys([*evidence, record.reopened_by])),
        limits=configured.limits,
    )


def _experimental_notes(data: CausalData, *, cross_fit: bool) -> list[str]:
    """Name the accepted compositions that no registered study has measured."""
    notes = []
    if not cross_fit:
        notes.append(
            "experimental: an in-sample fit; the registered study measures cross-fitted fits only"
        )
    if not np.allclose(data.weights, data.weights[0]):
        notes.append(
            "experimental: observation weights; no registered study measures a weighted curve"
        )
    return notes


class DoseResponseTMLE:
    """CV-TMLE of a kernel-smoothed dose-response curve at declared bandwidths and doses.

    The estimand at each pair is the smoothed curve of
    :class:`~cleverly.dose_response.DoseResponse`, not the curve at the dose itself.  The
    interval of each pair needs both nuisances consistent, with the product of their errors
    small next to that pair's standard error, which is about :math:`(nh)^{-1/2}`.

    Parameters
    ----------
    curve : DoseResponse
        The declaration.
    outcome_learner : estimator or None
        The outcome regression, fitted with the dose as a feature.  ``None`` takes the
        package default library.
    treatment_learner : estimator or None
        The classifier behind the pooled-hazard conditional density.  ``None`` takes the
        package default library.
    cross_fit : bool
        Fit the nuisances out of fold.
    n_folds : int
        The cross-fitting folds.
    learner_folds : int
        The folds of a Super Learner's inner split.
    density_bins : int or None
        The bins of the conditional density, or ``None`` for the package default.
    fluctuation : {"logistic", "linear"}
        The submodel on the scaled outcome.
    q_bounds : tuple of float or None
        The assumed outcome range of a continuous outcome, or ``None`` for the sample range.
        A cross-fitted fit of a continuous outcome needs it, as every cross-fitted fit does.
    alpha : float
        The level of the intervals and of the band.
    n_replicates : int, default=10_000
        Multiplier draws behind the band, with the Mammen multiplier.
    random_state : int or None
        Seed of the split, the learners and the band.
    n_jobs : int
        Parallel jobs for the nuisance fits.
    """

    def __init__(
        self,
        curve: DoseResponse,
        *,
        outcome_learner: Any = None,
        treatment_learner: Any = None,
        cross_fit: bool = True,
        n_folds: int = 5,
        learner_folds: int = 5,
        density_bins: int | None = None,
        fluctuation: Literal["logistic", "linear"] = "logistic",
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.05,
        n_replicates: int = DEFAULT_N_REPLICATES,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        if not isinstance(curve, DoseResponse):
            raise DataError("curve must be a cleverly.dose_response.DoseResponse")
        if fluctuation not in ("logistic", "linear"):
            raise DataError('fluctuation must be "logistic" or "linear"')
        self.curve = curve
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.cross_fit = cross_fit
        self.n_folds = n_folds
        self.learner_folds = learner_folds
        self.density_bins = density_bins
        self.fluctuation = fluctuation
        self.q_bounds = q_bounds
        self.alpha = alpha
        self.n_replicates = n_replicates
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _engine(self) -> Any:
        from .tmle import TMLE

        return TMLE(
            outcome_learner=self.outcome_learner,
            treatment_learner=self.treatment_learner,
            cross_fit=self.cross_fit,
            n_folds=self.n_folds,
            learner_folds=self.learner_folds,
            density_bins=self.density_bins,
            q_bounds=self.q_bounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        data: Any,
        *,
        outcome: str | None = None,
        treatment: str | None = None,
        covariates: Sequence[str] | None = None,
        weights: str | None = None,
        id: str | None = None,
        strata: Sequence[str] | None = None,
    ) -> DoseResponseResult:
        """Fit the curve.

        Parameters
        ----------
        data : Any
            A pandas or polars dataframe, or a prepared :class:`~cleverly.data.CausalData`
            with a continuous treatment.
        outcome : str or None
            The outcome column, when ``data`` is a dataframe.
        treatment : str or None
            The dose column, when ``data`` is a dataframe.
        covariates : Sequence[str] or None
            The covariate columns, when ``data`` is a dataframe.
        weights : str or None
            The observation-weight column.
        id : str or None
            The cluster column.
        strata : Sequence[str] or None
            Baseline strata columns, which must also be covariates.  Each stratum gets its own
            curve, :math:`E\\{\\int_D w(a)\\bar Q(a, W)\\,da \\mid V = v\\}`, from the same
            nuisance fits.

        Returns
        -------
        DoseResponseResult
            The curve, its intervals and its band.
        """
        engine = self._engine()
        prepared = engine._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            covariates=covariates,
            delta=None,
            weights=weights,
            id=id,
            intermediate=None,
            strata=strata,
            treatment_kind=None if isinstance(data, CausalData) else "continuous",
        )
        _refuse_unsupported(prepared)
        if self.cross_fit:
            engine._refuse_unbounded_cross_fitted_scale(prepared)
        return self._fit_prepared(engine, prepared)

    def _fit_prepared(self, engine: Any, data: CausalData) -> DoseResponseResult:
        curve = self.curve
        lower, upper = curve.support
        observed_low, observed_high = float(data.treatment.min()), float(data.treatment.max())
        if lower < observed_low or upper > observed_high:
            raise DataError(
                f"the dose interval [{lower:g}, {upper:g}] must lie inside the observed dose "
                f"range [{observed_low:g}, {observed_high:g}]: the fitted density is zero "
                "outside it. Declare the interval with a margin inside the expected range."
            )
        n_bins = default_density_bins(data.n) if self.density_bins is None else self.density_bins
        edges = bin_edges(data.treatment, n_bins, name=data.treatment_name)
        rule = curve.quadrature(edges)
        fine = curve.quadrature(edges, refine=2)
        nodes = np.concatenate([rule.nodes, fine.nodes])
        self._preflight_memory(data, nodes.size)
        folds = engine._folds(data, self.random_state)
        scaler = engine._scaler(data)
        outcome_task = "classification" if data.family == "binomial" else "regression"
        nuisance = fit_nuisances(
            data,
            outcome_learner=engine._resolve_learner(self.outcome_learner, task=outcome_task),
            treatment_learner=engine._resolve_learner(
                self.treatment_learner, task="classification"
            ),
            missingness_learner=None,
            intermediate_learner=None,
            folds=folds,
            scaler=scaler,
            density_bins=n_bins,
            dose_nodes=tuple(float(dose) for dose in nodes),
            n_jobs=self.n_jobs,
        )
        density = nuisance.density
        if not isinstance(density, ConditionalDensity) or not np.allclose(density.edges, edges):
            raise RuntimeError(
                "the fitted density does not use the bin edges the rule was built on"
            )
        configured, band_configured = dose_response_support(
            outcome="binary" if data.family == "binomial" else "continuous",
            fluctuation=self.fluctuation,
            strata=data.strata is not None,
            cross_fitted=self.cross_fit,
            weighted=not np.allclose(data.weights, data.weights[0]),
            n_replicates=self.n_replicates,
        )
        coarse = self._target(data, nuisance, rule, 0, configured)
        refined = self._target(data, nuisance, fine, rule.n_nodes, configured)
        estimates, supports, scores = coarse
        shift = max(
            abs(estimates[name].psi - refined[0][name].psi)
            / max(estimates[name].std_error, np.finfo(float).tiny)
            for name in estimates
        )
        band = simultaneous_bands(
            estimates,
            alpha=self.alpha,
            n_replicates=self.n_replicates,
            random_state=self.random_state,
            cluster=data.cluster,
        )
        # simultaneous_bands() leaves the band unclassified.  With every pair supported, the band
        # reads its own measured configuration; otherwise it joins the pairs' reasons.
        pair_support = combine_support(estimate.support for estimate in estimates.values())
        band = replace(
            band,
            support=(
                band_configured
                if pair_support.label == "supported"
                else combine_support([pair_support, band_configured])
            ),
        )
        notes: list[str] = []
        if shift > QUADRATURE_TOLERANCE:
            notes.append(
                f"doubling the quadrature nodes moves an estimate by {shift:.3f} standard errors; "
                "raise nodes_per_piece"
            )
            warn(notes[-1], DataWarning, stacklevel=3)
        notes.extend(_experimental_notes(data, cross_fit=self.cross_fit))
        low = [s.name for s in supports if s.below_floor]
        if low:
            notes.append(
                f"{len(low)} pair(s) have a Kish effective size below {curve.support_floor:g}; "
                "their intervals are labelled low_dose_support_plugin"
            )
            warn(notes[-1], DataWarning, stacklevel=3)
        return DoseResponseResult(
            curve=curve,
            estimates=estimates,
            band=band,
            pair_support=supports,
            quadrature_shift=float(shift),
            scores=scores,
            n_nodes=rule.n_nodes,
            alpha=self.alpha,
            warnings=tuple(notes),
            backend=data.backend,
        )

    def _preflight_memory(self, data: CausalData, n_nodes: int) -> None:
        # Every node's counterfactual design, its prediction and its density, in the parent
        # and again in each parallel worker that receives the designs.
        columns = data.covariates.shape[1] + 1
        copies = 1 if self.n_jobs == 1 else 1 + max(int(self.n_jobs), 1)
        needed = int(n_nodes) * data.n * 8 * (columns + 2) * copies
        if needed > NODE_MEMORY_BUDGET:
            raise DataError(
                f"the quadrature nodes need about {needed / 1024**3:.1f} GB "
                f"({n_nodes} nodes at {data.n} rows); declare fewer pairs or fewer "
                "nodes_per_piece, or fewer density bins"
            )

    def _target(
        self,
        data: CausalData,
        nuisance: NuisanceEstimates,
        rule: QuadratureRule,
        offset: int,
        configured: Support,
    ) -> tuple[dict[str, ParameterEstimate], tuple[PairSupport, ...], dict[str, float]]:
        curve = self.curve
        density = nuisance.density
        assert density is not None
        scaler = nuisance.scaler
        y = scaler.scale(data.outcome)
        q_obs = np.asarray(nuisance.outcome.observed, dtype=float)
        q_nodes = np.column_stack(
            [nuisance.outcome.arms[float(offset + k)] for k in range(rule.n_nodes)]
        )
        g_obs = density.density_at(data.treatment)
        g_nodes = np.column_stack(
            [density.density_at(np.full(data.n, dose)) for dose in rule.nodes]
        )
        estimates: dict[str, ParameterEstimate] = {}
        supports: list[PairSupport] = []
        scores: dict[str, float] = {}
        strata: list[tuple[str | None, FloatArray]] = [(None, np.ones(data.n))]
        if data.strata is not None:
            strata = [
                (data.stratum_label(code), (data.strata == code).astype(float))
                for code in range(data.n_strata)
            ]
        cells = [
            (p, h, a0, label, mask)
            for label, mask in strata
            for p, (h, a0) in enumerate(curve.pairs)
        ]
        for p, h, a0, label, mask in cells:
            name = _pair_cell_name(h, a0, label)
            row_weights = data.weights * mask
            weights = rule.weights[p]
            weight_obs = curve.weight(h, a0, data.treatment)
            if np.any((g_obs <= 0.0) & (weight_obs != 0.0)):
                raise DataError(
                    f"{name}: the fitted density is zero at an observed dose with positive weight"
                )
            used = weights != 0.0
            if np.any(g_nodes[:, used] <= 0.0):
                bad = rule.nodes[used][np.any(g_nodes[:, used] <= 0.0, axis=0)]
                raise DataError(
                    f"{name}: the fitted density is zero at doses {np.round(bad, 6).tolist()} "
                    "inside the kernel's support; narrow the dose interval"
                )
            h_obs = np.divide(
                weight_obs, g_obs, out=np.zeros_like(weight_obs), where=weight_obs != 0.0
            )
            node_weight = curve.weight(h, a0, rule.nodes)
            h_nodes = np.divide(
                np.broadcast_to(node_weight, g_nodes.shape),
                g_nodes,
                out=np.zeros_like(g_nodes),
                where=(node_weight != 0.0)[None, :] & (g_nodes > 0.0),
            )
            fitted = target_pair(
                y,
                q_obs,
                q_nodes,
                h_obs,
                h_nodes,
                weights,
                row_weights,
                fluctuation=self.fluctuation,
            )
            if not fitted.converged:
                # A data error, not a programming one: the Newton steps ran out on this sample,
                # and a repeated-sampling study records the draw (cleverly.validation._failures).
                raise DataError(
                    f"{name}: the fluctuation did not solve its score on this sample "
                    f"(score {fitted.score:.3g}); the pair's clever covariate is too large for "
                    "the submodel, so widen the bandwidth or narrow the dose interval"
                )
            inside = mask > 0.0
            report = pair_support(
                h_obs[inside], g_obs[inside], g_nodes[inside], weight_obs[inside], weights
            )
            below = report.kish < curve.support_floor
            supports.append(
                PairSupport(
                    name,
                    h,
                    a0,
                    label,
                    report.rows,
                    report.kish,
                    report.min_density,
                    report.max_covariate,
                    below,
                )
            )
            scores[name] = fitted.score
            estimates[name] = make_estimate(
                name,
                scaler.unscale_level(fitted.psi),
                fitted.influence_curve * scaler.range,
                n=data.n,
                cluster=data.cluster,
                scale="level",
                alpha=self.alpha,
                inference="low_dose_support_plugin" if below else "influence_curve",
                support=_pair_support(configured, below),
                # Each pair's fluctuation solves its score or the fit refuses (above).
                rule_widened=False,
            )
        return estimates, tuple(supports), scores
