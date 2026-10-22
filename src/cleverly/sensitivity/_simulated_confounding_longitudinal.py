"""The time-fixed latent of a longitudinal simulated-confounding surface (X82 part F).

One standard-normal latent value per unit is shared by every treatment node and the
terminal outcome.  Each node takes the DoWhy law of its kind: a binary node flips in the
upper latent tail, and a dose adds a signed multiple of the latent.  The outcome takes the
point-treatment outcome law.  Each cell refits the stored sequential estimator on the
perturbed panel.

The contract covers uncensored end-of-study fits whose panel has no time-varying covariate
after the first treatment node.  A later node generated under the original earlier
treatment depends on the latent given the perturbed one, so each cell reports, node by
node, the induced association of the latent with the perturbed node.  That association
includes the path through the earlier perturbed nodes by construction.

The fit-wide rules here answer for every longitudinal result, and the point-treatment rule
table hands a longitudinal result to :func:`longitudinal_fit_refusal` before any of its
point rules runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulated_confounding import ConfounderStrengthGrid, SimulatedConfoundingResult

#: The sequential estimators whose refit the surface replays, by ``config.estimator``.
SUPPORTED_ESTIMATORS = {
    "tmle": "LTMLE",
    "sdr": "SDR",
    "gcomp": "SequentialGComputation",
    "ipw": "SequentialIPW",
    "parametric_gformula": "ParametricGFormula",
}

_IPSI_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for LTMLE, SDR, "
    "SequentialGComputation, SequentialIPW and ParametricGFormula fits. An IPSI fit tilts a "
    "mechanism that reads the earlier perturbed nodes, and no contract names how the latent "
    "enters that tilt. docs/roadmap.md F13 tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="F13",
)
_OUTCOME_KIND_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for an end-of-study outcome. A survival, "
    "competing-risk or repeated outcome is an absorbing or time-indexed process, and an "
    "additive shift or a flip of it is not a law on its support. docs/roadmap.md F13 tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="F13",
)
_CENSORING_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for an uncensored panel. A flip of a "
    "treatment node breaks coarsening at random at each later censoring node that reads it. "
    "docs/roadmap.md F13 tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="F13",
)
_LATER_COVARIATE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for a panel with no time-varying "
    "covariate after the first treatment node. Such a covariate was generated under the "
    "original treatment, so it depends on the latent given the perturbed one, and the "
    "sequential regression conditions on it. docs/roadmap.md F13 tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="F13",
)
_KNOWN_LATER_NODE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding carries a declared mechanism at the first treatment node only. At a "
    "later node the declared factor reads the earlier treatment, and the factor of the "
    "perturbed node given the perturbed history is not a known function. docs/roadmap.md F13 "
    "tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="F13",
)
_NODE_LAW_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for binary and continuous treatment "
    "nodes. A node with more than two levels needs a declared arm pair at each node, and a "
    "vector node needs a law for each component. docs/roadmap.md X66 part (j) tracks them.",
    alternative=NO_ALTERNATIVE,
    roadmap="X66",
)
_CLUSTERED_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "simulated_confounding holds the time-fixed latent for an unclustered panel. A clustered "
    "panel needs the declared latent level of the point law at every node. docs/roadmap.md X66 "
    "part (k) tracks it.",
    alternative=NO_ALTERNATIVE,
    roadmap="X66",
)


def _refuse_type(result: Any) -> Refusal | None:
    from ..longitudinal.estimator import LongitudinalResult

    if type(result) is LongitudinalResult and getattr(result, "estimator", None) is not None:
        return None
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        "simulated_confounding needs a fitted LongitudinalResult that stores its estimator "
        "configuration. This is a package defect; report it.",
    )


def _refuse_estimator(result: Any) -> Refusal | None:
    name = result.config.estimator
    if name in SUPPORTED_ESTIMATORS:
        return None
    if name == "ipsi":
        return _IPSI_REFUSAL
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"simulated_confounding has no replay for the longitudinal estimator {name!r}. This is "
        "a package defect; report it.",
    )


def _refuse_outcome_kind(result: Any) -> Refusal | None:
    data = result.data
    if data.is_survival or data.is_competing or data.is_repeated:
        return _OUTCOME_KIND_REFUSAL
    return None


def _refuse_censoring(result: Any) -> Refusal | None:
    return _CENSORING_REFUSAL if result.data.has_censoring else None


def _refuse_later_covariates(result: Any) -> Refusal | None:
    blocks = result.data.time_varying_names[1:]
    return _LATER_COVARIATE_REFUSAL if any(blocks) else None


def _refuse_node_law(result: Any) -> Refusal | None:
    data = result.data
    for time in range(1, data.n_times + 1):
        kind = data.node_kind(time)
        if kind == "mixed":
            return _NODE_LAW_REFUSAL
        if kind == "categorical" and len(data.treatment_levels[time - 1]) != 2:
            return _NODE_LAW_REFUSAL
    return None


def _refuse_cluster(result: Any) -> Refusal | None:
    return _CLUSTERED_REFUSAL if result.data.cluster is not None else None


def _refuse_known_later_node(result: Any) -> Refusal | None:
    declared = result.data.known_mechanisms
    if declared is None:
        return None
    if any(values is not None for values in declared.treatment[1:]):
        return _KNOWN_LATER_NODE_REFUSAL
    return None


#: The ordered fit-wide rules of a longitudinal result.  ``result_type`` runs first, because
#: every later rule reads the result's ``config`` and ``data``.
LONGITUDINAL_RULES: tuple[tuple[str, Callable[[Any], Refusal | None]], ...] = (
    ("result_type", _refuse_type),
    ("estimator", _refuse_estimator),
    ("outcome_kind", _refuse_outcome_kind),
    ("censoring", _refuse_censoring),
    ("later_covariates", _refuse_later_covariates),
    ("node_law", _refuse_node_law),
    ("cluster", _refuse_cluster),
    ("known_later_node", _refuse_known_later_node),
)


def longitudinal_fit_refusal(result: Any) -> Refusal | None:
    """Return the first longitudinal simulated-confounding refusal for the whole fit.

    Parameters
    ----------
    result : Any
        A result that declares the longitudinal assessment family.

    Returns
    -------
    Refusal or None
        The first fit-wide refusal, or ``None`` when the time-fixed latent covers the fit.
    """
    for _name, rule in LONGITUDINAL_RULES:
        refusal = rule(result)
        if refusal is not None:
            return refusal
    return None


@dataclass(frozen=True)
class LongitudinalRequest:
    """A validated longitudinal request: what every cell of the surface reads."""

    nodes: tuple[str, ...]
    kinds: tuple[Literal["binary", "continuous"], ...]
    points: tuple[tuple[float, ...], ...]
    held: bool
    calibration_names: tuple[str, ...]
    movement_scale: Literal["estimate_difference", "log_ratio"]
    bounded_outcome_law: Literal["logit", "normal_score"] | None
    outcome_bounds: tuple[float, float] | None


def _node_kinds(data: Any) -> tuple[Literal["binary", "continuous"], ...]:
    return tuple(
        "continuous" if data.is_continuous_node(time) else "binary"
        for time in range(1, data.n_times + 1)
    )


def _grid_points(
    data: Any, grid: ConfounderStrengthGrid, held: bool
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Read the treatment axis of ``grid`` as one strength tuple per evaluated point.

    A one-axis grid applies each strength at every moved node.  A mapping keys strengths by
    node column, and a node it leaves out keeps strength zero.  A held design names one
    treatment column: it moves its first node only, and the identity nodes repeat it.
    """
    names = tuple(data.treatment_names)
    moved = names[:1] if held else names
    if not grid.components:
        values = tuple(float(value) for value in grid.treatment)  # type: ignore[arg-type]
        return moved, tuple((value,) * len(moved) for value in values)
    unknown = [column for column in grid.components if column not in names]
    if unknown:
        raise ValueError(
            f"ConfounderStrengthGrid(treatment={{...}}) names {unknown!r}, which are not "
            f"treatment nodes of this fit; its nodes are {list(names)!r}"
        )
    positions = [moved.index(column) for column in grid.components]
    points = []
    for point in grid.treatment_points:
        strengths = [0.0] * len(moved)
        for position, value in zip(positions, point, strict=True):  # type: ignore[arg-type]
            strengths[position] = float(value)
        points.append(tuple(strengths))
    return moved, tuple(points)


def _calibration_names(data: Any, benchmark_covariates: Any) -> tuple[str, ...]:
    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"benchmark_covariates holds {name!r}, which is not a column name")
        if name not in data.baseline_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; a benchmark names a baseline "
                f"covariate of this panel, one of {list(data.baseline_names)!r}"
            )
    return names


def _bounded_law(
    result: Any, bounded_outcome_law: Any
) -> tuple[Literal["logit", "normal_score"] | None, tuple[float, float] | None]:
    bounds = result.config.q_bounds
    if result.data.family != "gaussian" or bounds is None:
        if bounded_outcome_law is not None:
            raise ValueError(
                "bounded_outcome_law= reads the q_bounds of a Gaussian fit, and this fit "
                "declares none"
            )
        return None, None
    law: Literal["logit", "normal_score"]
    if bounded_outcome_law is None or bounded_outcome_law == "logit":
        law = "logit"
    elif bounded_outcome_law == "normal_score":
        law = "normal_score"
    else:
        raise ValueError(
            "bounded_outcome_law must be one of ('logit', 'normal_score'); "
            f"got {bounded_outcome_law!r}"
        )
    low, high = (float(value) for value in bounds)
    return law, (low, high)


def validate_longitudinal_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
    *,
    arm_pair: Any = None,
    latent_level: Any = None,
    latent_rho: Any = None,
    observation_reads: Any = None,
    bounded_outcome_law: Any = None,
) -> LongitudinalRequest:
    """Check a longitudinal request before any calibration, draw, or refit.

    Raises
    ------
    CapabilityError
        If a fit-wide rule refuses the fit.
    TypeError
        If ``benchmark_covariates`` holds a name that is not a string.
    ValueError
        If the estimand, the grid, or an argument does not apply to this fit.
    """
    from ..exceptions import CapabilityError

    refusal = longitudinal_fit_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    data = result.data
    if estimand not in result.estimates:
        raise ValueError(
            f"estimand {estimand!r} was not requested in this fit; it reports "
            f"{list(result.estimates)!r}"
        )
    if arm_pair is not None:
        raise ValueError(
            "arm_pair names the pairwise swap of a multi-arm point treatment; every node of "
            "this fit is binary or a dose"
        )
    if latent_level not in (None, "row") or latent_rho is not None:
        raise ValueError(
            "latent_level and latent_rho declare the latent level of a clustered fit; this "
            "panel declares no cluster, so the latent is one value per unit"
        )
    if observation_reads is not None:
        raise ValueError(
            "observation_reads declares the observation mechanisms of a fit with missing "
            "observations; this panel observes every node and the outcome"
        )
    if any(value != 0.0 for value in grid.intermediate):
        raise ValueError(
            "ConfounderStrengthGrid(intermediate=...) moves the intermediate of a "
            "controlled direct effect, and this fit declares no intermediate"
        )
    held = bool(data.treatment_held)
    nodes, points = _grid_points(data, grid, held)
    kinds = _node_kinds(data)[: len(nodes)]
    for point in points:
        for kind, value in zip(kinds, point, strict=True):
            if kind == "binary" and not 0.0 <= value <= 0.5:
                raise ValueError("binary treatment strengths must be between 0 and 0.5")
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    law, bounds = _bounded_law(result, bounded_outcome_law)
    names = _calibration_names(data, benchmark_covariates)
    scale = getattr(result[estimand], "scale", None)
    return LongitudinalRequest(
        nodes=nodes,
        kinds=kinds,
        points=points,
        held=held,
        calibration_names=names,
        movement_scale="log_ratio" if scale == "ratio" else "estimate_difference",
        bounded_outcome_law=law,
        outcome_bounds=bounds,
    )


def perturb_nodes(
    data: Any,
    latent: np.ndarray[Any, Any],
    strengths: tuple[float, ...],
    kinds: tuple[Literal["binary", "continuous"], ...],
    held: bool,
) -> np.ndarray[Any, Any]:
    """Apply the DoWhy law at each moved node, with one latent value per unit.

    A binary node flips on ``U >= Phi^{-1}(1 - s_t)``.  Those row sets are nested in
    strength order across nodes, because every node reads the same latent.  A dose adds
    ``s_t U``.  The identity nodes of a held design repeat the perturbed first node.
    """
    from .simulated_confounding import _flip_mask

    treatment = np.array(data.treatment, dtype=float, copy=True)
    for position, (strength, kind) in enumerate(zip(strengths, kinds, strict=True)):
        column = treatment[:, position]
        present = np.isfinite(column)
        if kind == "continuous":
            column[present] = column[present] + strength * latent[present]
        else:
            flip = present & _flip_mask(latent, strength)
            column[flip] = 1.0 - column[flip]
    if held:
        for position in range(1, treatment.shape[1]):
            present = np.isfinite(treatment[:, position])
            treatment[present, position] = treatment[present, 0]
    return treatment


def _flipped_declaration(data: Any, strength: float) -> Any:
    """The declared mechanisms after the first node flips with probability ``strength``.

    The latent is independent of the past, so ``g_s(a | W) = (1 - s) g(a | W) + s g(1 - a |
    W)`` at node 1.  No later node is declared, which the fit-wide rules check.
    """
    declared = data.known_mechanisms
    if declared is None or declared.treatment[0] is None or strength == 0.0:
        return "keep"
    first = np.asarray(declared.treatment[0], dtype=float)
    # Reversing the arm columns is the flip of a two-level node; the node-law rule refuses
    # any other node, and a dose node carries no declared factor.
    assert first.ndim == 2 and first.shape[1] == 2, first.shape
    flipped = (1.0 - strength) * first + strength * first[:, ::-1]
    return replace(declared, treatment=(flipped, *declared.treatment[1:]))


def _node_associations(
    latent: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    width: int,
) -> tuple[float | None, ...]:
    from .simulated_confounding import _weighted_correlation

    values = []
    for position in range(width):
        column = treatment[:, position]
        keep = np.isfinite(column)
        values.append(_weighted_correlation(latent[keep], column[keep], weights[keep]))
    return tuple(values)


def _outcome(
    data: Any,
    latent: np.ndarray[Any, Any],
    strength: float,
    request: LongitudinalRequest,
) -> np.ndarray[Any, Any]:
    from .simulated_confounding import (
        _flip_binary,
        _flip_mask,
        _gaussian_outcome,
        _logit_outcome,
        _normal_score_outcome,
    )

    values = np.asarray(data.outcome, dtype=float)
    observed = np.isfinite(values)
    if request.bounded_outcome_law == "logit":
        assert request.outcome_bounds is not None
        moved = _logit_outcome(values, observed, latent, strength, request.outcome_bounds)
    elif request.bounded_outcome_law == "normal_score":
        assert request.outcome_bounds is not None
        moved = _normal_score_outcome(values, observed, latent, strength, request.outcome_bounds)
    elif data.family == "gaussian":
        moved = _gaussian_outcome(values, latent, strength)
    else:
        moved = _flip_binary(values, _flip_mask(latent, strength))
    return np.where(observed, moved, values)


def _calibrate(data: Any, request: LongitudinalRequest) -> tuple[Any, ...]:
    """Calibrate each named baseline covariate against each moved node and the outcome.

    The design is the standardized baseline block, as the point surface uses the adjustment
    covariates.  A node row reads one node, so it describes that node's association with
    the covariate given the other baseline covariates, not given the earlier nodes.
    """
    from sklearn.preprocessing import StandardScaler

    from .simulated_confounding import (
        _NUMERIC_GAUSSIAN_METHOD,
        ObservedConfounderCalibration,
        _binary_calibration,
        _continuous_calibration,
        _weights_are_constant,
    )

    if not request.calibration_names:
        return ()
    baseline = np.asarray(data.baseline, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    if _weights_are_constant(weights):
        design = StandardScaler().fit_transform(baseline)
    else:
        design = StandardScaler().fit(baseline, sample_weight=weights).transform(baseline)

    def row(
        name: str, target: np.ndarray[Any, Any], binary: bool, role: Any, node: str | None
    ) -> Any:
        index = list(data.baseline_names).index(name)
        keep = np.isfinite(target)
        if binary:
            strength, method = _binary_calibration(
                design[keep], target[keep], [index], weights[keep]
            )
            family: Literal["binomial", "gaussian"] = "binomial"
        else:
            strength = _continuous_calibration(baseline[keep, index], target[keep], weights[keep])
            method, family = _NUMERIC_GAUSSIAN_METHOD, "gaussian"
        return ObservedConfounderCalibration(
            covariate=name,
            role=role,
            family=family,
            strength=strength,
            method=method,
            component=node,
        )

    rows = []
    for name in request.calibration_names:
        for position, (node, kind) in enumerate(zip(request.nodes, request.kinds, strict=True)):
            rows.append(row(name, data.treatment[:, position], kind == "binary", "treatment", node))
        rows.append(
            row(
                name,
                np.asarray(data.outcome, dtype=float),
                data.family == "binomial",
                "outcome",
                None,
            )
        )
    return tuple(rows)


def _refit_folds(result: Any, frozen: list[int] | None) -> Literal["original", "redrawn", "none"]:
    """Whether the refits reuse the original split, redraw it, or the fit has none."""
    if frozen is not None:
        return "original"
    repeats = getattr(result, "repeats", None) or ()
    if all(item.seed is None for item in repeats):
        return "none"
    return "redrawn"


def _frozen_seeds(result: Any, root_seed: int) -> list[int] | None:
    """The original draws' seeds, when the refit runs under the fit's own seed.

    A refit given them draws the original folds and learner seeds, so movement near the
    anchor carries no fold noise.  An explicit seed other than the fit's own redraws them.
    """
    repeats = getattr(result, "repeats", None) or ()
    seeds = [item.seed for item in repeats]
    if not seeds or any(seed is None for seed in seeds):
        return None
    own = result.estimator.random_state
    if own is not None and int(own) != root_seed:
        return None
    return [int(seed) for seed in seeds]


def longitudinal_surface(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    request: LongitudinalRequest,
    random_state: int | None,
) -> SimulatedConfoundingResult:
    """Evaluate every declared cell of a longitudinal surface.

    Parameters
    ----------
    result : LongitudinalResult
        The fitted result.
    estimand : str
        The reported parameter.
    grid : ConfounderStrengthGrid
        The declared strengths.
    request : LongitudinalRequest
        The validated request.
    random_state : int or None
        The root seed.

    Returns
    -------
    SimulatedConfoundingResult
        One cell per treatment point and outcome strength.
    """
    from ..utils.random import resolve_assessment_seed
    from ..validation.simulation import ReplicationFailure
    from .simulated_confounding import (
        SimulatedConfoundingCell,
        SimulatedConfoundingResult,
        _draw_latent,
        _latent_child_seed,
        _outcome_law_text,
    )

    data = result.data
    calibrations = _calibrate(data, request)
    root_seed = resolve_assessment_seed(result, random_state)
    refit_seed = root_seed
    frozen = _frozen_seeds(result, root_seed)
    latent_seed = _latent_child_seed(root_seed)
    latent = _draw_latent(data.n, latent_seed, None, "row", None)
    original_parameter = result[estimand]
    original = float(original_parameter.psi)
    original_inference = original_parameter.inference_value
    weights = np.asarray(data.weights, dtype=float)
    width = len(request.nodes)
    cells: list[Any] = []
    for cell_index, (point, outcome_strength) in enumerate(product(request.points, grid.outcome)):
        if outcome_strength == 0.0 and all(value == 0.0 for value in point):
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=original,
                    displacement=0.0,
                    induced_treatment_association=_node_associations(
                        latent, data.treatment, weights, width
                    ),
                    target_population_fraction=1.0,
                )
            )
            continue
        association: Any = None
        try:
            treatment = perturb_nodes(data, latent, point, request.kinds, request.held)
            association = _node_associations(latent, treatment, weights, width)
            replacement = data.with_treatment(
                treatment, known_mechanisms=_flipped_declaration(data, point[0])
            )
            replacement = replace(
                replacement, outcome=_outcome(data, latent, outcome_strength, request)
            )
            refitted = result.estimator.refit(
                replacement, random_state=refit_seed, frozen_seeds=frozen
            )
            refitted_parameter = refitted[estimand]
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=float(refitted_parameter.psi),
                    displacement=refitted_parameter.inference_value - original_inference,
                    induced_treatment_association=association,
                    target_population_fraction=1.0,
                )
            )
        except Exception as error:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=None,
                    displacement=None,
                    induced_treatment_association=association,
                    target_population_fraction=1.0,
                    failure=ReplicationFailure(
                        replicate=cell_index,
                        seed=root_seed,
                        error_type=type(error).__name__,
                        message=str(error),
                    ),
                )
            )
    declared = data.known_mechanisms
    repeats = getattr(result, "repeats", None) or ()
    return SimulatedConfoundingResult(
        estimand=estimand,
        original_estimate=original,
        movement_scale=request.movement_scale,
        grid=grid,
        cells=tuple(cells),
        calibrations=calibrations,
        root_seed=root_seed,
        latent_seed=latent_seed,
        refit_seed=refit_seed,
        n_repeats=max(len(repeats), 1),
        repeat_aggregation="coordinatewise_median",
        treatment_family="longitudinal",
        outcome_family=data.family,
        treatment_law=(
            "Each binary node is flipped in the declared upper latent-normal tail, and each "
            "dose adds signed strength times the latent value. One latent value per unit is "
            "shared by every node and the outcome."
            + (" The identity nodes of the held design repeat node 1." if request.held else "")
        ),
        outcome_law=_outcome_law_text(request.bounded_outcome_law, data.family),
        weight_report=data.weight_report(),
        known_mechanism_carried=declared is not None and declared.treatment[0] is not None,
        treatment_components=request.nodes,
        weights_estimated=bool(data.weight_spec.estimated),
        bounded_outcome_law=request.bounded_outcome_law,
        outcome_bounds=request.outcome_bounds,
        bounded_share=_bounded_share(data, request),
        backend=data.backend,
        held_decision=request.held,
        refit_folds=_refit_folds(result, frozen),
    )


def _bounded_share(data: Any, request: LongitudinalRequest) -> float | None:
    if request.bounded_outcome_law != "logit" or request.outcome_bounds is None:
        return None
    low, high = request.outcome_bounds
    values = np.asarray(data.outcome, dtype=float)
    observed = np.isfinite(values)
    at_bound = (values[observed] <= low) | (values[observed] >= high)
    return float(np.average(at_bound, weights=np.asarray(data.weights)[observed]))
