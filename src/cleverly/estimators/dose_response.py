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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from .._typing import FloatArray
from ..data import CausalData
from ..dose_response import (
    DoseResponse,
    QuadratureRule,
    pair_name,
    pair_support,
    target_pair,
)
from ..exceptions import CapabilityError, DataError, RefusalReason
from ..inference.influence import ParameterEstimate, make_estimate
from ..inference.multiplier import SimultaneousBands, simultaneous_bands
from ..learners.density import ConditionalDensity, bin_edges, default_density_bins
from ..utils.frames import frame_from_dict
from ._nuisance import NuisanceEstimates, fit_nuisances

__all__ = ["DoseResponseResult", "DoseResponseTMLE"]

#: The memory the node predictions may use before the fit refuses, in bytes.
NODE_MEMORY_BUDGET: Final[int] = 4 * 1024**3

#: The doubling diagnostic warns when doubling the nodes moves an estimate by more than this
#: fraction of its standard error.
QUADRATURE_TOLERANCE: Final[float] = 0.05


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
    support : tuple of PairSupport
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
        The fit's notes: a quadrature-doubling shift above tolerance, and pairs below the
        support floor.
    """

    curve: DoseResponse
    estimates: dict[str, ParameterEstimate]
    band: SimultaneousBands
    support: tuple[PairSupport, ...]
    quadrature_shift: float
    scores: dict[str, float]
    n_nodes: int
    alpha: float
    warnings: tuple[str, ...] = field(default=())

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
        """
        for cell in self.support:
            same = np.isclose(cell.bandwidth, h) and np.isclose(cell.dose, a0)
            if same and cell.stratum == stratum:
                return self.estimates[cell.name]
        raise KeyError(
            f"no declared pair with bandwidth {h!r}, dose {a0!r} and stratum {stratum!r}"
        )

    def to_frame(self, backend: str | None = None) -> Any:
        """One row per pair: the estimate, its pointwise interval, the band and the support.

        Parameters
        ----------
        backend : str or None
            The dataframe library, or ``None`` for the default.

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
        stratified = any(cell.stratum is not None for cell in self.support)
        if stratified:
            columns = {"bandwidth": [], "dose": [], "stratum": [], **columns}
        for support in self.support:
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
        return frame_from_dict(columns, backend=backend)

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


def _refuse_unsupported(data: CausalData) -> None:
    if not data.is_continuous_treatment or data.treatment_kind == "vector":
        raise CapabilityError(
            "a dose-response curve needs one continuous treatment; pass "
            'treatment_kind="continuous"',
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if data.has_missing_outcome or data.has_intermediate or data.has_missing_treatment:
        raise CapabilityError(
            "a dose-response curve does not yet support a missing outcome, a missing "
            "treatment or an intermediate variable: the second mechanism would have to be "
            "evaluated at every quadrature dose.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap="X10",
        )
    if data.dose_atoms:
        raise CapabilityError(
            "a dose-response curve on a treatment with declared dose atoms is not supported",
            reason=RefusalReason.IMPLEMENTATION_GAP,
        )


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
    alpha : float
        The level of the intervals and of the band.
    n_replicates : int
        Multiplier draws behind the band.
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
        n_replicates: int = 1000,
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
        coarse = self._target(data, nuisance, rule, 0)
        refined = self._target(data, nuisance, fine, rule.n_nodes)
        estimates, supports, scores = coarse
        shift = max(
            abs(estimates[name].psi - refined[0][name].psi) / estimates[name].std_error
            for name in estimates
        )
        band = simultaneous_bands(
            estimates,
            alpha=self.alpha,
            n_replicates=self.n_replicates,
            random_state=self.random_state,
            cluster=data.cluster,
        )
        notes: list[str] = []
        if shift > QUADRATURE_TOLERANCE:
            notes.append(
                f"doubling the quadrature nodes moves an estimate by {shift:.3f} standard errors; "
                "raise nodes_per_piece"
            )
        low = [s.name for s in supports if s.below_floor]
        if low:
            notes.append(
                f"{len(low)} pair(s) have a Kish effective size below {curve.support_floor:g}; "
                "their intervals are labelled low_dose_support_plugin"
            )
        return DoseResponseResult(
            curve=curve,
            estimates=estimates,
            band=band,
            support=supports,
            quadrature_shift=float(shift),
            scores=scores,
            n_nodes=rule.n_nodes,
            alpha=self.alpha,
            warnings=tuple(notes),
        )

    def _preflight_memory(self, data: CausalData, n_nodes: int) -> None:
        columns = data.covariates.shape[1] + 1
        needed = int(n_nodes) * data.n * 8 * (columns + 2)
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
            name = pair_name(h, a0) if label is None else f"{pair_name(h, a0)}[{label}]"
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
                raise RuntimeError(f"{name}: the fluctuation did not solve its score")
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
                    report.max_covariate * scaler.range,
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
            )
        return estimates, tuple(supports), scores
