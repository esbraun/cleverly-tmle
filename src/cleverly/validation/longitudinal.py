r"""Did a sequential fit's targeting work, node by node?

The counterpart of :mod:`cleverly.validation.score` for a longitudinal fit.  The
point-treatment battery asks one question of one solved score equation.  A sequential fit
poses the question once per regimen and node, and a cross-fitted one poses a second
question about the *stitched* score that no single fold solved, so the verdicts arrive as
frames rather than as a scalar.

Three reports live here, one per question a stored sequential fit can answer without
refitting anything:

========================================  =====================================================
report                                    what one row says
========================================  =====================================================
:class:`LongitudinalDiagnostics`          how much data a node had, and how hard it leaned on it
:class:`LongitudinalScoreDiagnostics`     whether that node's fluctuation reached its root
:class:`LongitudinalNuisanceDiagnostics`  the loss of that node's sequential regression
========================================  =====================================================

:mod:`cleverly.assessment` routes a caller to these and re-exports every public name, so
``from cleverly.assessment import LongitudinalDiagnostics`` still works.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray
from ..data.weighting import effective_sample_size
from ..inference.cluster import cluster_sums
from ..utils.frames import emit_frame

__all__ = [
    "STITCHED_SCORE_Z_TOLERANCE",
    "LongitudinalDiagnostics",
    "LongitudinalNuisanceDiagnostics",
    "LongitudinalNuisanceRow",
    "LongitudinalScoreDiagnostics",
    "LongitudinalScoreRow",
    "LongitudinalStageRow",
]


#: How far a cross-fitted longitudinal fit's *stitched* score may sit from zero, in
#: standard errors of its own residual, before :func:`_longitudinal_scores` calls it a
#: defect rather than sampling.
#:
#: The stitched score is not a solved equation.  Each outer fold fits its ``epsilon`` on
#: the rows it does not report, so what the pooled residual has to be is a mean-zero draw,
#: and the scale to judge a mean-zero draw on is its own standard error.  Measured over 300
#: replications of ``make_longitudinal`` at ``n=500`` and five folds, the mean ``|z|`` per
#: parameter ran from 0.006 to 0.08.  Four standard errors is therefore a long way outside
#: anything the construction produces, while a fold-mapping or stitching defect -- which
#: multiplies the residual by a constant rather than perturbing it -- moves ``z`` by orders
#: of magnitude and cannot hide under it.
#:
#: Not a caller argument.  ``tolerance`` on
#: :meth:`~cleverly.assessment.DiagnosticsFacade.score_equations` is a *relative-score*
#: tolerance, and one number cannot mean both "close enough to solved" and "consistent with
#: noise"; passing it to both gates would silently apply ``1e-3`` standard errors here and
#: fail every cross-fitted fit.
STITCHED_SCORE_Z_TOLERANCE = 4.0


@dataclass(frozen=True)
class LongitudinalStageRow:
    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    n_followed: int
    assignment: float | str
    max_weight: float
    effective_n: float
    share_truncated: float
    epsilon: tuple[float, ...]
    converged: bool


@dataclass(frozen=True)
class LongitudinalDiagnostics:
    """A row per regimen and node: how much data it had, and how hard it leaned on it.

    ``n_followed`` is the number of units that followed the regimen and stayed under
    observation through the node -- the sample the regression there was fitted on.
    ``max_weight`` and ``effective_n`` describe the cumulative clever covariate, which is
    where sequential positivity shows up: they are properties of the *product* of the
    node-by-node mechanisms and can be alarming while every node looks fine.  On a weighted
    fit they describe ``w / prod g`` rather than ``1 / prod g``, because the two
    reweightings multiply -- see
    :attr:`~cleverly.longitudinal.sequential.RegimenFit.leverage`.  For the weighting's own
    cost, and the estimand statement that goes with it, see ``result.data.weight_report()``.

    ``share_assigned_1`` is the fraction of the units at risk at that node whom the regimen
    would treat.  For a static regimen it is exactly ``0`` or ``1``, so the column doubles
    as a check on the plan the fit actually ran; for a dynamic rule it is the number a
    reader needs, since what a rule assigns is a property of the data rather than of the
    declaration and appears nowhere in the settings report.

    **When any treatment node is categorical the column is** ``assigned_shares``
    **instead**, holding ``"active=0.62, none=0.38"`` in that node's label order -- the
    presentation :func:`~cleverly.estimators.base._arm_shares` uses for the same question
    about a point treatment.  A single share cannot answer it at three arms: "the fraction
    assigned arm 1" is the fraction assigned whichever label happens to sort second, which
    is not a quantity anybody asked for, and a static plan on a third arm would report ``0``
    exactly as a plan on the first arm does.  A wholly two-level panel keeps
    ``share_assigned_1`` and its values unchanged, so the switch is visible in the columns
    rather than hidden in them.

    ``share_truncated`` compares the raw and bounded cumulative probabilities on the same
    ``trained_on`` rows as the node's score.  Unlike ``max_weight``, it reveals when the
    configured cap replaced every contributing row.
    """

    rows: tuple[LongitudinalStageRow, ...]
    epsilon_names: tuple[str, ...]
    categorical: bool
    survival: bool
    competing: bool
    backend: str | None = None

    def to_frame(self, data: Any = None) -> Any:
        assignment = "assigned_shares" if self.categorical else "share_assigned_1"
        payload: dict[str, Any] = {
            "regimen": [row.regimen for row in self.rows],
            **({"cause": [row.cause for row in self.rows]} if self.competing else {}),
            **({"horizon": [row.horizon for row in self.rows]} if self.survival else {}),
            "time": [row.time for row in self.rows],
            "n_followed": [row.n_followed for row in self.rows],
            assignment: [row.assignment for row in self.rows],
            "max_weight": [row.max_weight for row in self.rows],
            "effective_n": [row.effective_n for row in self.rows],
            "share_truncated": [row.share_truncated for row in self.rows],
        }
        for position, name in enumerate(self.epsilon_names):
            payload[name] = [row.epsilon[position] for row in self.rows]
        payload["converged"] = [row.converged for row in self.rows]
        return emit_frame(payload, data, backend=self.backend)


@dataclass(frozen=True)
class LongitudinalScoreRow:
    """One verdict about one node's targeting.

    ``converged`` is the fit's own flag: whether that node's Newton step settled against
    the targeting tolerance it was configured with.  ``passed`` additionally holds the row
    to the tolerance the *caller* asked for.  They are kept apart because they can
    disagree, and because only their conjunction is safe -- a caller tolerance may tighten
    the verdict and may never license a fluctuation whose step failed.

    ``kind`` says which question the row answers, because a cross-fitted fit poses two and
    they have different right answers.

    ``component`` names an MSM score component. Such a row pools all live regimen cells,
    so ``regimen`` and ``horizon`` are ``None``. Ordinary regimen rows have no component.

    ``"solver"``
        Did the fluctuation reach the root of the equation it was *given*?  On an ordinary
        fit that equation is the node's own score and ``relative_score`` is it.  On a
        cross-fitted fit it is each outer fold's score on its training complement, and
        ``relative_score`` is the largest across the folds.  Either way the answer should
        be at solver tolerance, and a failure here is a solver failure.

    ``"stitching"``
        Is the score of the *stitched* fit where sampling alone would leave it?  Emitted
        only on a cross-fitted fit, where the answer is not zero and is not meant to be:
        every fold fits its ``epsilon`` on rows it does not report, so the pooled residual
        is noise about zero rather than a solved equation.  ``z`` is that residual over its
        own standard error and ``relative_score`` is the raw magnitude, reported so the
        reader can see what the ``z`` is a ratio of.  A stitching, indexing or fold-mapping
        defect moves ``z`` by orders of magnitude, which is what this row is for.
    """

    regimen: str | None
    cause: str | None
    horizon: int | None
    time: int
    component: str | None
    kind: str
    score: float
    relative_score: float
    #: The score over its own standard error.  ``nan`` on a ``"solver"`` row, whose claim
    #: is that the score is zero rather than that it is small relative to anything.
    z: float
    converged: bool
    passed: bool
    n_iter: int
    failure: str | None


@dataclass(frozen=True)
class LongitudinalScoreDiagnostics:
    """Stagewise targeting verdicts, gated at the tolerances they were asked for.

    ``tolerance`` bounds a ``"solver"`` row's *relative* score -- the largest score
    component as a fraction of its maximum possible magnitude, which is the quantity the
    sequential targeting loop itself gates on.  The point-treatment report answers the same
    question on a different scale, comparing the score in the outcome's own units against
    ``tolerance * se / sqrt(n)``; see :data:`~cleverly.validation.score.DEFAULT_TOLERANCE`.
    The number is carried here so a report says which gate produced its verdict.

    ``z_tolerance`` bounds a ``"stitching"`` row instead, in standard errors, because that
    row's score is not a solved equation and holding it to a relative tolerance would fail
    every cross-fitted fit for doing exactly what it is supposed to do.  A fit with no
    cross-fitting emits no such row and ``z_tolerance`` never binds.
    """

    rows: tuple[LongitudinalScoreRow, ...]
    tolerance: float
    backend: str | None = None
    z_tolerance: float = STITCHED_SCORE_Z_TOLERANCE

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "regimen": [row.regimen for row in self.rows],
                "cause": [row.cause for row in self.rows],
                "horizon": [row.horizon for row in self.rows],
                "time": [row.time for row in self.rows],
                "component": [row.component for row in self.rows],
                "kind": [row.kind for row in self.rows],
                "score": [row.score for row in self.rows],
                "relative_score": [row.relative_score for row in self.rows],
                "z": [row.z for row in self.rows],
                "converged": [row.converged for row in self.rows],
                "passed": [row.passed for row in self.rows],
                "n_iter": [row.n_iter for row in self.rows],
                "failure": [row.failure for row in self.rows],
            },
            data,
            backend=self.backend,
        )


@dataclass(frozen=True)
class LongitudinalNuisanceRow:
    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    n: int
    mse: float


@dataclass(frozen=True)
class LongitudinalNuisanceDiagnostics:
    """Stagewise loss of each sequential outcome/pseudo-outcome regression."""

    rows: tuple[LongitudinalNuisanceRow, ...]
    backend: str | None = None

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "regimen": [row.regimen for row in self.rows],
                "cause": [row.cause for row in self.rows],
                "horizon": [row.horizon for row in self.rows],
                "time": [row.time for row in self.rows],
                "n": [row.n for row in self.rows],
                "mse": [row.mse for row in self.rows],
            },
            data,
            backend=self.backend,
        )


def _assigned_shares(assigned: FloatArray, levels: Sequence[object]) -> str:
    """What a regimen assigns at one node, as ``"active=0.62, none=0.38"``.

    The categorical counterpart of ``share_assigned_1``, and written in the *labels* rather
    than the dense codes for the reason every user-facing string in this package is: a
    reader asked to translate ``2.0`` back to ``"none"`` has been handed the encoding rather
    than the answer.  Every level appears, including one the regimen never assigns, so the
    shares in a row sum to one and a zero is legible as "not this arm" rather than as a
    level the fit forgot about.

    Deliberately a string and not a column per level: the level sets are per node, so
    numeric columns would be ragged across a frame whose rows are ``(regimen, time)`` pairs,
    and most of them empty.
    """
    if not assigned.size:
        return ""
    return ", ".join(
        f"{level}={float(np.mean(assigned == float(code))):.3g}"
        for code, level in enumerate(levels)
    )


def _longitudinal_stagewise(result: Any) -> LongitudinalDiagnostics:
    """One row per node: how heavy the weights got and how much the bounds moved.

    ``max_weight`` and ``effective_n`` read ``step.clever``, and ``share_truncated`` reads
    ``fit.cumulative``.  On a cross-fitted fit those are not two views of one array: the
    covariate is stitched from each fold's own mechanism slab while ``cumulative`` is the
    out-of-fold mechanism, so ``1 / cumulative`` does not reproduce the weight.  Each column
    is read from the array that answers its own question -- what a row was weighted by, and
    how far the bounds moved the mechanism -- rather than both from whichever one is nearer.
    """
    terms = () if result.msm is None else result.msm.terms
    epsilon_names = ("epsilon",) if result.msm is None else tuple(f"epsilon[{t}]" for t in terms)
    # One column shape for the whole frame rather than one per row: the level sets are a
    # property of the data, so whether a share is answerable by a single number is settled
    # before any node is read.
    categorical = any(len(levels) > 2 for levels in result.data.treatment_levels)
    rows = []
    # Read off the fit's own fields rather than the key it is filed under: on a survival fit
    # that key is the regimen *and* the horizon, and a ``regimen`` column carrying both would
    # be the one column here nobody could group by.
    for fit in result.fits.values():
        for step in fit.steps:
            weights = (fit.obs_weights * step.clever)[step.trained_on]
            assigned = fit.assignment[step.at_risk, step.time - 1]
            assignment: float | str = (
                _assigned_shares(assigned, result.data.treatment_levels[step.time - 1])
                if categorical
                else (float(np.mean(assigned == 1.0)) if assigned.size else float("nan"))
            )
            raw = fit.cumulative_unbounded[:, step.time - 1][step.trained_on]
            bounded = fit.cumulative[:, step.time - 1][step.trained_on]
            rows.append(
                LongitudinalStageRow(
                    regimen=fit.regimen.label,
                    cause=fit.cause,
                    horizon=fit.horizon if result.data.is_survival else None,
                    time=step.time,
                    n_followed=step.n_trained,
                    assignment=assignment,
                    max_weight=float(np.max(weights)) if weights.size else float("nan"),
                    effective_n=effective_sample_size(weights, on_degenerate=0.0),
                    share_truncated=float(np.mean(raw != bounded)) if raw.size else float("nan"),
                    epsilon=tuple(float(value) for value in step.fluctuation.epsilon),
                    converged=bool(step.fluctuation.converged),
                )
            )
    return LongitudinalDiagnostics(
        tuple(rows),
        epsilon_names,
        categorical,
        result.data.is_survival,
        result.data.is_competing,
        result.data.backend,
    )


def _standardized_score(contribution: FloatArray, cluster: IntArray | None = None) -> FloatArray:
    """Standardize mean score components by their independent-unit standard errors."""
    values = np.asarray(contribution, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    if n < 2:
        return np.full(values.shape[1], np.nan)
    if cluster is None:
        standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    else:
        sums = cluster_sums(values, np.asarray(cluster))
        n_clusters = sums.shape[0]
        if n_clusters < 2:
            return np.full(values.shape[1], np.nan)
        standard_error = np.sqrt(n_clusters * np.var(sums, axis=0, ddof=1) / n**2)
    mean = np.mean(values, axis=0)
    return np.divide(
        mean,
        standard_error,
        out=np.full(values.shape[1], np.nan),
        where=standard_error > 0.0,
    )


def _stitched_score_z(step: Any, weights: FloatArray, cluster: IntArray | None = None) -> float:
    r"""The stitched score over its own standard error.

    The score is :math:`P_n[w H (Z - \bar Q^*)]`. Independent rows use the row-level
    standard error. Clustered rows first sum their contributions within cluster and use
    the same finite-sample scaling as the inference layer.

    Returns ``nan`` when the residual has no spread, which is a degenerate node rather than
    a perfect one and is not something to report a ``z`` of zero for.
    """
    contribution = weights * step.clever * (step.pseudo_outcome - step.targeted)
    return float(_standardized_score(contribution, cluster)[0])


def _msm_node_contributions(
    result: Any, msm_fit: Any, time: int
) -> tuple[FloatArray, FloatArray, Any]:
    """Per-unit pooled score and scale contributions for one longitudinal MSM node."""
    model = msm_fit.model
    weights = np.asarray(result.data.weights, dtype=float)
    contribution = np.zeros((result.data.n, model.n_terms), dtype=float)
    maximum = np.zeros_like(contribution)
    fluctuation = None
    for cell_index, (cell, cell_fit) in enumerate(zip(model.cells, msm_fit.fits, strict=True)):
        if cell.horizon < time:
            continue
        step = next(item for item in cell_fit.steps if item.time == time)
        fluctuation = step.fluctuation
        multiplier = weights * model.weights[:, cell_index] * step.clever
        design = msm_fit.fluctuation_design[:, cell_index, :]
        residual = step.pseudo_outcome - step.targeted
        contribution += multiplier[:, None] * design * residual[:, None]
        maximum += np.abs(multiplier[:, None] * design)
    if fluctuation is None:  # pragma: no cover - a model cell always reaches its own nodes
        raise RuntimeError(f"the longitudinal MSM has no live cell at time {time}")
    return contribution, maximum, fluctuation


def _longitudinal_msm_scores(result: Any, *, tolerance: float) -> LongitudinalScoreDiagnostics:
    """Pooled component-wise node diagnostics for a longitudinal working model."""
    rows = []
    for msm_fit in result.msm_fits:
        times = sorted({step.time for fit in msm_fit.fits for step in fit.steps})
        for time in times:
            contribution, maximum, fluctuation = _msm_node_contributions(result, msm_fit, time)
            pooled_score = np.mean(contribution, axis=0)
            pooled_scale = np.mean(maximum, axis=0)
            z = _standardized_score(contribution, result.data.cluster)
            for component, term in enumerate(msm_fit.model.terms):
                converged = bool(fluctuation.converged)
                if fluctuation.folds:
                    relatives = []
                    scores = []
                    for record in fluctuation.folds:
                        scale = (
                            np.asarray(record.score_scale, dtype=float)
                            if record.score_scale is not None
                            else np.asarray(fluctuation.score_scale, dtype=float)
                        )
                        score = float(np.asarray(record.score, dtype=float)[component])
                        scores.append(abs(score))
                        relatives.append(abs(score) / max(float(scale[component]), 1e-300))
                    solver_score = max(scores)
                    solver_relative = max(relatives)
                else:
                    solver_score = abs(float(pooled_score[component]))
                    solver_relative = solver_score / max(float(pooled_scale[component]), 1e-300)
                rows.append(
                    LongitudinalScoreRow(
                        None,
                        msm_fit.cause,
                        None,
                        time,
                        term,
                        "solver",
                        float(result.scaler.range * solver_score),
                        solver_relative,
                        float("nan"),
                        converged,
                        converged and solver_relative <= tolerance,
                        int(fluctuation.n_iter),
                        fluctuation.failure,
                    )
                )
                if not fluctuation.folds:
                    continue
                component_z = float(z[component])
                relative = abs(float(pooled_score[component])) / max(
                    float(pooled_scale[component]), 1e-300
                )
                rows.append(
                    LongitudinalScoreRow(
                        None,
                        msm_fit.cause,
                        None,
                        time,
                        term,
                        "stitching",
                        float(result.scaler.range * pooled_score[component]),
                        relative,
                        component_z,
                        converged,
                        bool(
                            np.isfinite(component_z)
                            and abs(component_z) <= STITCHED_SCORE_Z_TOLERANCE
                        ),
                        int(fluctuation.n_iter),
                        fluctuation.failure,
                    )
                )
    return LongitudinalScoreDiagnostics(tuple(rows), tolerance, result.data.backend)


def _longitudinal_scores(result: Any, *, tolerance: float) -> LongitudinalScoreDiagnostics:
    """Every node's targeting verdicts: one row per question the node's fit poses.

    The solver gate is a *conjunction*, and deliberately so.  Sequential targeting settles
    against its own ``tol`` -- ``1e-10``, far tighter than the default asked for here -- so
    requiring ``converged`` as well leaves the default verdict exactly what it was while
    letting a caller tighten it.  Gating on the relative score alone would do the opposite: a
    node whose Newton step failed but whose residual score happens to sit under a loose
    tolerance would be reported as passing, which is the one answer this diagnostic must
    never give.

    A cross-fitted node earns a second row, because the first one stops being able to see
    the thing that can go wrong.  Its ``K`` solves each reach their own root on their own
    training complement, so the solver row is at machine precision whatever the stitched fit
    looks like -- including when the folds were stitched back in the wrong order, or a slab
    was read for the wrong fold.  The stitching row is where that shows.
    """
    if result.msm is not None:
        return _longitudinal_msm_scores(result, tolerance=tolerance)

    rows = []
    for fit in result.fits.values():
        weights = np.asarray(fit.obs_weights, dtype=float)
        for step in fit.steps:
            fluctuation = step.fluctuation
            horizon = fit.horizon if result.data.is_survival else None
            converged = bool(fluctuation.converged)
            # On a cross-fitted node the solved equations are the folds' own, and the
            # aggregate `score` is the stitched fit's -- a different quantity, reported on
            # the row below.  The worst fold is the honest summary of `K` solves: an
            # average would let nine good folds hide one that did not move.
            solver_relative = (
                max(
                    float(
                        np.max(
                            np.abs(record.score)
                            / np.maximum(
                                record.score_scale
                                if record.score_scale is not None
                                else fluctuation.score_scale,
                                1e-300,
                            )
                        )
                    )
                    for record in fluctuation.folds
                )
                if fluctuation.folds
                else float(fluctuation.relative_score_norm)
            )
            solver_score = (
                max(float(np.max(np.abs(record.score))) for record in fluctuation.folds)
                if fluctuation.folds
                else float(fluctuation.score_norm)
            )
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    horizon,
                    step.time,
                    None,
                    "solver",
                    float(result.scaler.range * solver_score),
                    solver_relative,
                    float("nan"),
                    converged,
                    converged and solver_relative <= tolerance,
                    int(fluctuation.n_iter),
                    fluctuation.failure,
                )
            )
            if not fluctuation.folds:
                continue
            z = _stitched_score_z(step, weights, result.data.cluster)
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    horizon,
                    step.time,
                    None,
                    "stitching",
                    float(result.scaler.range * fluctuation.score_norm),
                    float(fluctuation.relative_score_norm),
                    z,
                    converged,
                    bool(np.isfinite(z) and abs(z) <= STITCHED_SCORE_Z_TOLERANCE),
                    int(fluctuation.n_iter),
                    fluctuation.failure,
                )
            )
    return LongitudinalScoreDiagnostics(tuple(rows), tolerance, result.data.backend)


def _longitudinal_nuisances(result: Any) -> LongitudinalNuisanceDiagnostics:
    rows = []
    for fit in result.fits.values():
        for step in fit.steps:
            mask = step.trained_on
            residual = np.asarray(step.pseudo_outcome[mask] - step.initial[mask], dtype=float)
            rows.append(
                LongitudinalNuisanceRow(
                    fit.regimen.label,
                    fit.cause,
                    fit.horizon if result.data.is_survival else None,
                    step.time,
                    int(mask.sum()),
                    float(np.mean(np.square(residual))) if residual.size else float("nan"),
                )
            )
    return LongitudinalNuisanceDiagnostics(tuple(rows), result.data.backend)
