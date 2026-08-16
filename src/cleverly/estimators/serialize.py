"""Saving a result, and loading it back.

A fit is expensive and its result is the input to a lot of cheap follow-up work:
positivity diagnostics, truncation curves, the MNAR tilt, the score check, nuisance
calibration, the bootstrap.  All of those go through
:meth:`~cleverly.TMLE.retarget`, which needs arrays and settings -- not a live
estimator, and not a fitted scikit-learn object.  So a result can be written to disk
and picked up in another process, and this module does that.

**No pickle.**  Every payload here is arrays plus JSON.  Pickling a fitted
scikit-learn estimator ties the file to the exact version that wrote it, turns a
stored result into an arbitrary-code-execution vector, and is not needed:
:class:`~cleverly.estimators._nuisance.NuisanceEstimates` already keeps out-of-fold
*predictions* rather than models, which is what
:meth:`~cleverly.TMLE.retarget` consumes.

**Where the boundary is**, stated rather than implied.  After a round trip:

* everything reached through ``retarget`` works -- ``sensitivity.positivity()``,
  ``sensitivity.truncation_curve()``, ``sensitivity.missingness_tilt()``,
  ``sensitivity.omitted_variable()``, ``validation.score_check()``,
  ``validation.nuisance()``, contrasts, bands, the bootstrap;
* the two analyses that genuinely refit -- ``validation.refute()`` and
  ``sensitivity.benchmark()`` -- need the estimator rebuilt from the recipe, which
  works when the learners were library specifications and raises a specific error
  when one was a scikit-learn object (see :class:`~cleverly.estimators.recipe
  .TMLERecipe`).

The format is a ``.npz`` of every array with a ``__manifest__`` JSON entry
describing the rest.  It is versioned; a file from a future version is refused
rather than misread.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..data.causal_data import CategoricalEncoding, CausalData
from ..data.weighting import WeightSpec
from ..fluctuation.iterative import Fluctuation, FoldFluctuation, InitialFit
from ..fluctuation.mechanism import MechanismFluctuation
from ..inference.influence import ParameterEstimate
from ..interventions import IPSISet, RegimeSet, ShiftSet
from ..learners.crossfit import CrossFitPlan, Folds
from ..learners.density import ConditionalDensity
from ..msm import MSMSet
from ..provenance import Provenance
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates, Propensity, RepeatFit
from .base import TMLEConfig, TMLEResult
from .recipe import TMLERecipe
from .reduced import MissingOutcomeReducedSet, ReducedSet
from .targeting import (
    ObservationMechanismFluctuation,
    ProjectionFluctuation,
    ReductionFluctuation,
    TargetingSpec,
)

__all__ = ["FORMAT_VERSION", "load", "result_from_dict", "result_to_dict", "save"]

#: Bumped when the payload changes shape.  A reader refuses *any* other version rather
#: than guessing at fields it does not know, so a bump makes older files unreadable by
#: design -- re-run the fit, or read them with the version of cleverly that wrote them.
#:
#: ``2`` keys an initial fit's counterfactual predictions by treatment level (``arms``)
#: instead of naming two fields ``at_one`` and ``at_zero``.
#:
#: ``3`` stores the treatment mechanism as an ``(n, K)`` matrix over the arms plus the
#: arm codes it is keyed by, rather than the single ``P(A = 1 | W)`` vector, which cannot
#: describe a treatment with more than two levels.
#:
#: ``4`` records the treatment's *kind*, and the conditional density and shifts a
#: continuous fit targets.  Without the kind a dose reloaded as a discrete treatment with
#: no levels -- ``is_continuous_treatment`` silently flipped to ``False`` -- and without
#: the density the mechanism half of a shift fit was simply absent from the file.
#:
#: ``5`` records the *evaluated* working model a fit declared with ``msm=``.  Without it a
#: reloaded fit had no design to project onto, so every retargeted analysis -- the
#: truncation curve, the score check, the MNAR tilt -- would have reported the arm-indexed
#: estimands instead of the coefficients the fit was about.
#:
#: ``6`` records the fold policy the fit *declared* (``TMLEConfig.crossfit``) beside the
#: fold count it ran.  The two come apart whenever the split was capped at the rarest
#: stratum or the cluster count, and the warning that said so does not survive the fit, so
#: without it a reloaded result could not say that a 10-fold fit had run 3.
#:
#: ``7`` stores a *list* of repeats -- each draw's nuisances and fluctuations -- where the
#: previous versions stored one of each.  The shape change is why this is a version bump
#: rather than an added key: under ``repeats=R`` the reported estimates are the average
#: over all ``R`` draws, so a file holding only the first would reload as a result whose
#: own numbers none of its analyses could reproduce.  An ordinary fit writes a one-element
#: list and reads back byte-identically.
#:
#: ``8`` records the working model's **link**.  Reading an older file as an identity-link
#: fit would be right for every file that exists -- nothing else could have written one --
#: and the bump is here rather than a default because the field decides which *estimand*
#: the coefficients belong to: a log-link file read back without it would report log risk
#: ratios under a linear model's arithmetic, with intervals to match.
#:
#: ``9`` records the reduced-dimension regressions of the doubly-robust-inference variant.
#: :func:`_nuisance_from` names every field it reconstructs, so one left unwritten reloads
#: silently as its default -- and the default here is ``None``, which is not a degraded
#: version of the variant but a *different estimator*: the extra score equations would
#: have nothing to be solved against, and the reloaded fit would report a plain TMLE's
#: interval under a doubly-robust name, with nothing in the parameter's name to say so.
#: That is the same reason version 5 could not default the working model, and it is why
#: the bump lands with the arrays rather than with the estimator that will read them.
#:
#: ``10`` records the three halves a targeting step can have *beside* the outcome
#: fluctuation: ``Fluctuation.mechanism``, ``.projection`` and ``.reduction``.  Version 9
#: stored the reduced regressions on the nuisances and stopped there, and what it left
#: behind was not record-keeping but a *diagnostic*:
#: :func:`~cleverly.validation.score_check` reads these records, so a reloaded ``DRTMLE``
#: fit reported **one** fluctuation row where the live fit reported three, and answered a
#: strictly weaker question under the same name -- one that can pass where the live check
#: failed.  A diagnostic that silently narrows on a round trip is worse than one that is
#: absent.  The same omission cost every ``ipsi`` fit its mechanism tilt and every linked
#: ``msm`` fit its projection; the three are siblings on ``Fluctuation`` and are written
#: together so that a fourth is not forgotten in its turn.
#:
#: ``11`` records an optional selected outcome state from which targeting continues.
#: C-TMLE selects a pair ``(g_k, Qbar*_k)`` rather than ``g_k`` alone; omitting the
#: latter made a loaded result silently revert to an ordinary TMLE at the selected g.
#:
#: ``12`` records the finite baseline partition used by stratified targets.  Without
#: the row codes and labels a loaded result cannot reproduce the conditional targets.
#:
#: ``13`` records C-TMLE selection diagnostics, including the exact component names of
#: a jointly selected multi-arm target.  Without them a loaded result loses which vector
#: was optimized even though it retains the selected propensity and targeted outcome.
#:
#: ``14`` was the draft PR's collapsed joint-mechanism representation.
#:
#: ``15`` records the five separate missing-outcome reductions and the targeted
#: observation mechanism.  It intentionally refuses draft version-14 artifacts rather
#: than reading their joint score as the paper's separate treatment and observation scores.
FORMAT_VERSION = 15

_ARRAY_MARK = "__array__"


class _Arrays:
    """Collects arrays out of a nested payload and puts them back on the way in."""

    def __init__(self) -> None:
        self.store: dict[str, FloatArray] = {}

    def put(self, key: str, value: Any) -> Any:
        if value is None:
            return None
        self.store[key] = np.asarray(value)
        return {_ARRAY_MARK: key}

    def get(self, ref: Any) -> Any:
        if ref is None:
            return None
        return self.store[ref[_ARRAY_MARK]]


def _estimate_to(payload_arrays: _Arrays, prefix: str, est: ParameterEstimate) -> dict[str, Any]:
    return {
        "name": est.name,
        "psi": est.psi,
        "influence_curve": payload_arrays.put(f"{prefix}.ic", est.influence_curve),
        "variance": est.variance,
        "n": est.n,
        "n_clusters": est.n_clusters,
        "scale": est.scale,
        "alpha": est.alpha,
        "log_psi": est.log_psi,
    }


def _estimate_from(arrays: _Arrays, payload: dict[str, Any]) -> ParameterEstimate:
    return ParameterEstimate(
        name=payload["name"],
        psi=payload["psi"],
        influence_curve=arrays.get(payload["influence_curve"]),
        variance=payload["variance"],
        n=payload["n"],
        n_clusters=payload["n_clusters"],
        scale=payload["scale"],
        alpha=payload["alpha"],
        log_psi=payload["log_psi"],
    )


def _fit_to(arrays: _Arrays, prefix: str, fit: InitialFit) -> dict[str, Any]:
    """Store an initial fit, keying its arms the way the manifest can carry them.

    JSON object keys are strings, so an arm level is written with ``repr`` and read back
    with ``float``.  Round-tripping through ``repr`` is exact for a float, which matters
    because the key *is* the treatment level a downstream lookup asks for: a key that came
    back as ``1.0000000000000002`` would be a silently missing arm.
    """
    return {
        "observed": arrays.put(f"{prefix}.observed", fit.observed),
        "arms": {
            repr(level): arrays.put(f"{prefix}.arm@{level}", values)
            for level, values in fit.arms.items()
        },
    }


def _fit_from(arrays: _Arrays, payload: dict[str, Any]) -> InitialFit:
    return InitialFit(
        arrays.get(payload["observed"]),
        {float(level): arrays.get(key) for level, key in payload["arms"].items()},
    )


def _fluctuation_to(arrays: _Arrays, prefix: str, fl: Fluctuation) -> dict[str, Any]:
    return {
        "epsilon": arrays.put(f"{prefix}.epsilon", fl.epsilon),
        "targeted": _fit_to(arrays, f"{prefix}.targeted", fl.targeted),
        "score": arrays.put(f"{prefix}.score", fl.score),
        "converged": bool(fl.converged),
        "n_iter": int(fl.n_iter),
        "trace": list(fl.trace),
        "method": fl.method,
        "names": list(fl.names),
        "score_scale": arrays.put(f"{prefix}.score_scale", fl.score_scale),
        "score_initial": arrays.put(f"{prefix}.score_initial", fl.score_initial),
        "epsilon_std_error": arrays.put(f"{prefix}.epsilon_se", fl.epsilon_std_error),
        "n_solver_calls": int(fl.n_solver_calls),
        "failure": fl.failure,
        "hessian_condition": float(fl.hessian_condition),
        "loglik": float(fl.loglik),
        "folds": [
            {
                "index": arrays.put(f"{prefix}.fold{i}.index", f.index),
                "epsilon": arrays.put(f"{prefix}.fold{i}.epsilon", f.epsilon),
                "score": arrays.put(f"{prefix}.fold{i}.score", f.score),
                "converged": bool(f.converged),
                "n_iter": int(f.n_iter),
            }
            for i, f in enumerate(fl.folds)
        ],
        # The other halves of a targeting step that has more than one. `score_check` reads
        # all three, so a file without them answers a narrower question than the fit did
        # -- see the note on format version 10 above.
        "mechanism": _mechanism_to(arrays, f"{prefix}.mechanism", fl.mechanism),
        "projection": _projection_to(arrays, f"{prefix}.projection", fl.projection),
        "reduction": _reduction_to(arrays, f"{prefix}.reduction", fl.reduction),
    }


def _mechanism_to(arrays: _Arrays, prefix: str, mech: Any | None) -> dict[str, Any] | None:
    """Store equation (9)'s half: the tilted mechanism and how its solve went."""
    if mech is None:
        return None
    return {
        "propensity": arrays.put(f"{prefix}.propensity", mech.propensity),
        "epsilon": arrays.put(f"{prefix}.epsilon", mech.epsilon),
        "score": arrays.put(f"{prefix}.score", mech.score),
        "score_scale": arrays.put(f"{prefix}.score_scale", mech.score_scale),
        "score_initial": arrays.put(f"{prefix}.score_initial", mech.score_initial),
        "converged": bool(mech.converged),
        "n_iter": int(mech.n_iter),
        "epsilon_std_error": arrays.put(f"{prefix}.epsilon_se", mech.epsilon_std_error),
        "hessian_condition": mech.hessian_condition,
        "loglik": mech.loglik,
        "failure": mech.failure,
        "trace": [list(row) for row in mech.trace],
    }


def _mechanism_from(arrays: _Arrays, payload: dict[str, Any] | None) -> MechanismFluctuation | None:
    if payload is None:
        return None
    return MechanismFluctuation(
        propensity=arrays.get(payload["propensity"]),
        epsilon=arrays.get(payload["epsilon"]),
        score=arrays.get(payload["score"]),
        score_scale=arrays.get(payload["score_scale"]),
        score_initial=arrays.get(payload["score_initial"]),
        converged=payload["converged"],
        n_iter=payload["n_iter"],
        epsilon_std_error=arrays.get(payload["epsilon_std_error"]),
        hessian_condition=payload["hessian_condition"],
        loglik=payload["loglik"],
        failure=payload["failure"],
        trace=tuple(tuple(row) for row in payload["trace"]),
    )


def _observation_to(arrays: _Arrays, prefix: str, observation: Any | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "propensity": arrays.put(f"{prefix}.propensity", observation.propensity),
        "epsilon": arrays.put(f"{prefix}.epsilon", observation.epsilon),
        "score": arrays.put(f"{prefix}.score", observation.score),
        "score_scale": arrays.put(f"{prefix}.score_scale", observation.score_scale),
        "score_initial": arrays.put(f"{prefix}.score_initial", observation.score_initial),
        "names": list(observation.names),
        "converged": bool(observation.converged),
        "failure": observation.failure,
        "loglik": float(observation.loglik),
    }


def _observation_from(
    arrays: _Arrays, payload: dict[str, Any] | None
) -> ObservationMechanismFluctuation | None:
    if payload is None:
        return None
    return ObservationMechanismFluctuation(
        propensity=arrays.get(payload["propensity"]),
        epsilon=arrays.get(payload["epsilon"]),
        score=arrays.get(payload["score"]),
        score_scale=arrays.get(payload["score_scale"]),
        score_initial=arrays.get(payload["score_initial"]),
        names=tuple(payload["names"]),
        converged=bool(payload["converged"]),
        failure=payload["failure"],
        loglik=float(payload["loglik"]),
    )


def _projection_to(arrays: _Arrays, prefix: str, proj: Any | None) -> dict[str, Any] | None:
    """Store a linked working model's half: the coefficients the report is taken at.

    Recursive in ``folds``, which under fold-wise targeting holds one of these per fold --
    each a ``ProjectionFluctuation`` with an empty ``folds`` of its own.
    """
    if proj is None:
        return None
    return {
        "beta": arrays.put(f"{prefix}.beta", proj.beta),
        "trace": [list(row) for row in proj.trace],
        "converged": bool(proj.converged),
        "failure": proj.failure,
        "folds": [
            _projection_to(arrays, f"{prefix}.fold{i}", fold) for i, fold in enumerate(proj.folds)
        ],
    }


def _projection_from(
    arrays: _Arrays, payload: dict[str, Any] | None
) -> ProjectionFluctuation | None:
    if payload is None:
        return None
    folds = tuple(_projection_from(arrays, fold) for fold in payload["folds"])
    return ProjectionFluctuation(
        beta=arrays.get(payload["beta"]),
        trace=tuple(tuple(row) for row in payload["trace"]),
        converged=payload["converged"],
        failure=payload["failure"],
        folds=tuple(fold for fold in folds if fold is not None),
    )


def _reduction_to(arrays: _Arrays, prefix: str, red: Any | None) -> dict[str, Any] | None:
    """Store equation (10)'s half, including the reductions it was finally solved against.

    Those are the *refit*, not ``nuisance.reduced``, and the difference is the whole reason
    this record exists: the influence curve is built from these, so a file that dropped
    them could not say what the reported variance was computed at.
    """
    if red is None:
        return None
    return {
        "reduced": _reduced_to(arrays, f"{prefix}.reduced", red.reduced),
        "guard": list(red.guard),
        "bounds": [float(value) for value in red.bounds],
        "epsilon": arrays.put(f"{prefix}.epsilon", red.epsilon),
        "score": arrays.put(f"{prefix}.score", red.score),
        "score_scale": arrays.put(f"{prefix}.score_scale", red.score_scale),
        "score_initial": arrays.put(f"{prefix}.score_initial", red.score_initial),
        "names": list(red.names),
        "trace": [list(row) for row in red.trace],
        "rounds": int(red.rounds),
        "converged": bool(red.converged),
        "failure": red.failure,
        "exit_reason": red.exit_reason,
        "closing_capped": bool(red.closing_capped),
        "ill_conditioned": int(red.ill_conditioned),
        "closing": int(red.closing),
        "observation": _observation_to(arrays, f"{prefix}.observation", red.observation),
        "missingness_bound": red.missingness_bound,
    }


def _reduction_from(arrays: _Arrays, payload: dict[str, Any] | None) -> ReductionFluctuation | None:
    if payload is None:
        return None
    reduced = _reduced_from(arrays, payload["reduced"])
    assert reduced is not None  # a reduction record without its regressions is not one
    lower, upper = payload["bounds"]
    return ReductionFluctuation(
        reduced=reduced,
        guard=tuple(payload["guard"]),
        bounds=(float(lower), float(upper)),
        epsilon=arrays.get(payload["epsilon"]),
        score=arrays.get(payload["score"]),
        score_scale=arrays.get(payload["score_scale"]),
        score_initial=arrays.get(payload["score_initial"]),
        names=tuple(payload["names"]),
        trace=tuple(tuple(row) for row in payload["trace"]),
        rounds=payload["rounds"],
        converged=payload["converged"],
        failure=payload["failure"],
        exit_reason=payload["exit_reason"],
        closing_capped=payload["closing_capped"],
        ill_conditioned=payload["ill_conditioned"],
        closing=payload["closing"],
        observation=_observation_from(arrays, payload.get("observation")),
        missingness_bound=payload.get("missingness_bound"),
    )


def _fluctuation_from(arrays: _Arrays, payload: dict[str, Any]) -> Fluctuation:
    return Fluctuation(
        epsilon=arrays.get(payload["epsilon"]),
        targeted=_fit_from(arrays, payload["targeted"]),
        score=arrays.get(payload["score"]),
        converged=payload["converged"],
        n_iter=payload["n_iter"],
        trace=tuple(payload["trace"]),
        method=payload["method"],
        names=tuple(payload["names"]),
        score_scale=arrays.get(payload["score_scale"]),
        score_initial=arrays.get(payload["score_initial"]),
        epsilon_std_error=arrays.get(payload["epsilon_std_error"]),
        n_solver_calls=payload["n_solver_calls"],
        failure=payload["failure"],
        hessian_condition=payload["hessian_condition"],
        loglik=payload["loglik"],
        folds=tuple(
            FoldFluctuation(
                index=arrays.get(f["index"]).astype(np.int64),
                epsilon=arrays.get(f["epsilon"]),
                score=arrays.get(f["score"]),
                converged=f["converged"],
                n_iter=f["n_iter"],
            )
            for f in payload["folds"]
        ),
        mechanism=_mechanism_from(arrays, payload["mechanism"]),
        projection=_projection_from(arrays, payload["projection"]),
        reduction=_reduction_from(arrays, payload["reduction"]),
    )


def _data_to(arrays: _Arrays, data: CausalData) -> dict[str, Any]:
    return {
        "outcome": arrays.put("data.outcome", data.outcome),
        "treatment": arrays.put("data.treatment", data.treatment),
        "covariates": arrays.put("data.covariates", data.covariates),
        "covariate_names": list(data.covariate_names),
        "weights": arrays.put("data.weights", data.weights),
        "observed": arrays.put("data.observed", data.observed),
        "family": data.family,
        "outcome_name": data.outcome_name,
        "treatment_name": data.treatment_name,
        "treatment_levels": list(data.treatment_levels),
        # The backend is a name, so it survives a round trip. It could not while the
        # container held the input frame itself: a saved polars fit reloaded with no
        # way to know it was one, and every `to_frame()` on it came back as pandas.
        "backend": data.backend,
        # Declared, never inferred -- exactly as on the way in. A continuous treatment
        # has no levels, so a reader recovering the kind from an empty level list would
        # be guessing, and would guess "discrete" for a dose.
        "treatment_kind": data.treatment_kind,
        "delta_name": data.delta_name,
        "cluster": arrays.put("data.cluster", data.cluster),
        "cluster_name": data.cluster_name,
        "intermediate": arrays.put("data.intermediate", data.intermediate),
        "intermediate_name": data.intermediate_name,
        "weights_name": data.weights_name,
        "weight_spec": {
            "kind": data.weight_spec.kind,
            "estimated": data.weight_spec.estimated,
            "name": data.weight_spec.name,
            "scale": data.weight_spec.scale,
        },
        "dropped_covariates": list(data.dropped_covariates),
        "encodings": [
            {
                "column": e.column,
                "levels": list(e.levels),
                "dropped_level": e.dropped_level,
                "generated": list(e.generated),
            }
            for e in data.encodings
        ],
        "strata": arrays.put("data.strata", data.strata),
        "strata_names": list(data.strata_names),
        "strata_levels": [list(level) for level in data.strata_levels],
    }


def _data_from(arrays: _Arrays, payload: dict[str, Any]) -> CausalData:
    cluster = arrays.get(payload["cluster"])
    strata = arrays.get(payload["strata"])
    return CausalData(
        outcome=arrays.get(payload["outcome"]),
        treatment=arrays.get(payload["treatment"]),
        covariates=arrays.get(payload["covariates"]),
        covariate_names=tuple(payload["covariate_names"]),
        weights=arrays.get(payload["weights"]),
        observed=arrays.get(payload["observed"]).astype(bool),
        family=payload["family"],
        outcome_name=payload["outcome_name"],
        treatment_name=payload["treatment_name"],
        treatment_levels=tuple(payload["treatment_levels"]),
        treatment_kind=payload["treatment_kind"],
        # `.get` rather than `[...]`: a file written before the backend was recorded
        # still loads, and lands on the default backend as it always did.
        backend=payload.get("backend"),
        delta_name=payload["delta_name"],
        cluster=None if cluster is None else cluster.astype(np.int64),
        cluster_name=payload["cluster_name"],
        intermediate=arrays.get(payload["intermediate"]),
        intermediate_name=payload["intermediate_name"],
        weights_name=payload["weights_name"],
        weight_spec=WeightSpec(**payload["weight_spec"]),
        dropped_covariates=tuple(payload["dropped_covariates"]),
        encodings=tuple(
            CategoricalEncoding(
                column=e["column"],
                levels=tuple(e["levels"]),
                dropped_level=e["dropped_level"],
                generated=tuple(e["generated"]),
            )
            for e in payload["encodings"]
        ),
        strata=None if strata is None else strata.astype(np.int64),
        strata_names=tuple(payload["strata_names"]),
        strata_levels=tuple(tuple(level) for level in payload["strata_levels"]),
    )


def _nuisance_to(arrays: _Arrays, prefix: str, nuisance: NuisanceEstimates) -> dict[str, Any]:
    return {
        "propensity": arrays.put(f"{prefix}.propensity", nuisance.propensity.values),
        "propensity_arms": [float(arm) for arm in nuisance.propensity.arms],
        "outcome": _fit_to(arrays, f"{prefix}.outcome", nuisance.outcome),
        "targeting_outcome": (
            None
            if nuisance.targeting_outcome is None
            else _fit_to(arrays, f"{prefix}.targeting_outcome", nuisance.targeting_outcome)
        ),
        "scaler": {"lower": nuisance.scaler.lower, "upper": nuisance.scaler.upper},
        "folds": {
            "assignment": arrays.put(f"{prefix}.folds", nuisance.folds.assignment),
            "n_folds": nuisance.folds.n_folds,
        },
        "missingness": arrays.put(f"{prefix}.missingness", nuisance.missingness),
        "intermediate": arrays.put(f"{prefix}.intermediate", nuisance.intermediate),
        "treatment_covariates": list(nuisance.treatment_covariates),
        "outcome_task": nuisance.outcome_task,
        # The *evaluated* densities, not the rules that produced them. A rule is a
        # callable and cannot be written; its output can, and it is the output that
        # every reuse of the fit needs -- which is what keeps a loaded result's
        # truncation curve, MNAR tilt and score check bit-for-bit identical.
        "regimes": (
            None
            if nuisance.regimes is None
            else {
                "values": arrays.put(f"{prefix}.regimes", nuisance.regimes.values),
                "names": list(nuisance.regimes.names),
                "reference": float(nuisance.regimes.reference),
            }
        ),
        # The density is written for the same reason: it holds evaluated bin
        # probabilities and no learner, so a reloaded shift fit can be retargeted,
        # swept over truncation bounds and score-checked without refitting anything.
        "density": (
            None
            if nuisance.density is None
            else {
                "bin_probabilities": arrays.put(
                    f"{prefix}.density", nuisance.density.bin_probabilities
                ),
                "edges": arrays.put(f"{prefix}.density.edges", nuisance.density.edges),
            }
        ),
        # Every array of a ShiftSet, rather than the shifts plus a rule for rebuilding
        # them: re-evaluating on load would recompute the density ratios from the
        # reloaded density, and an evaluation that agrees to fifteen digits rather than
        # exactly is not the round trip this format promises.
        "shifts": (
            None
            if nuisance.shifts is None
            else {
                "names": list(nuisance.shifts.names),
                "deltas": [float(delta) for delta in nuisance.shifts.deltas],
                "shifted": arrays.put(f"{prefix}.shifts.shifted", nuisance.shifts.shifted),
                "ratio": arrays.put(f"{prefix}.shifts.ratio", nuisance.shifts.ratio),
                "ratio_at": arrays.put(f"{prefix}.shifts.ratio_at", nuisance.shifts.ratio_at),
                "capped": arrays.put(f"{prefix}.shifts.capped", nuisance.shifts.capped),
                "reference": float(nuisance.shifts.reference),
            }
        ),
        # The tilts, as arrays for the reason the shifts are -- and one more. An IPSISet
        # is a function of the mechanism, so rebuilding it on load would re-derive it
        # from the reloaded propensity; that agrees to fifteen digits, not exactly, and
        # this format promises exactly.
        "incremental": (
            None
            if nuisance.incremental is None
            else {
                "names": list(nuisance.incremental.names),
                "deltas": [float(delta) for delta in nuisance.incremental.deltas],
                "values": arrays.put(f"{prefix}.incremental.values", nuisance.incremental.values),
                "weights": arrays.put(
                    f"{prefix}.incremental.weights", nuisance.incremental.weights
                ),
                "derivative": arrays.put(
                    f"{prefix}.incremental.derivative", nuisance.incremental.derivative
                ),
                "propensity": arrays.put(
                    f"{prefix}.incremental.propensity", nuisance.incremental.propensity
                ),
                "reference": float(nuisance.incremental.reference),
            }
        ),
        # The working model, likewise as arrays: its design is a callable and cannot be
        # written, its output can, and the output is what every reuse needs. This is the
        # whole of why a loaded result can still be retargeted against the model the fit
        # declared, even though `recipe` records that fit as unreconstructible.
        "msm": (
            None
            if nuisance.msm is None
            else {
                "terms": list(nuisance.msm.terms),
                "design": arrays.put(f"{prefix}.msm.design", nuisance.msm.design),
                "weights": arrays.put(f"{prefix}.msm.weights", nuisance.msm.weights),
                "arms": [float(arm) for arm in nuisance.msm.arms],
                "link": str(nuisance.msm.link),
                "clever_weights": arrays.put(
                    f"{prefix}.msm.clever_weights", nuisance.msm.clever_weights
                ),
                "observed_design": arrays.put(
                    f"{prefix}.msm.observed_design", nuisance.msm.observed_design
                ),
                "observed_weights": arrays.put(
                    f"{prefix}.msm.observed_weights", nuisance.msm.observed_weights
                ),
                "dose_values": list(nuisance.msm.dose_values),
            }
        ),
        # The reduced-dimension regressions, as arrays for the reason the tilts are: they
        # are functionals of the two nuisances *and* of the split, so rebuilding them on
        # load would mean refitting three learners, which is the one thing `retarget`
        # promises never to do. `g_bounds` travels with them because it is the bound their
        # target was formed at, and a reader has to be able to find that out.
        "reduced": _reduced_to(arrays, f"{prefix}.reduced", nuisance.reduced),
    }


def _reduced_to(
    arrays: _Arrays,
    prefix: str,
    reduced: ReducedSet | MissingOutcomeReducedSet | None,
) -> dict[str, Any] | None:
    """Store a set of reduced regressions, wherever it hangs.

    Two records carry one: the nuisances' initial fit, and the refit the alternation
    finally solved against on :class:`~cleverly.estimators.targeting.ReductionFluctuation`.
    One writer so the two cannot drift into different shapes -- they are read back by one
    :func:`_reduced_from`.
    """
    if reduced is None:
        return None
    if isinstance(reduced, MissingOutcomeReducedSet):
        return {
            "kind": "missing_outcome",
            "gamma_a": arrays.put(f"{prefix}.gamma_a", reduced.gamma_a),
            "gamma_m": arrays.put(f"{prefix}.gamma_m", reduced.gamma_m),
            "r_a": arrays.put(f"{prefix}.r_a", reduced.r_a),
            "r_m": arrays.put(f"{prefix}.r_m", reduced.r_m),
            "e": arrays.put(f"{prefix}.e", reduced.e),
            "arms": [float(arm) for arm in reduced.arms],
            "g_bounds": [float(value) for value in reduced.g_bounds],
            "missingness_bound": float(reduced.missingness_bound),
        }
    return {
        "kind": "complete",
        "qr": arrays.put(f"{prefix}.qr", reduced.qr),
        "gr1": arrays.put(f"{prefix}.gr1", reduced.gr1),
        "gr2": arrays.put(f"{prefix}.gr2", reduced.gr2),
        "arms": [float(arm) for arm in reduced.arms],
        "g_bounds": [float(value) for value in reduced.g_bounds],
        "reduction": str(reduced.reduction),
    }


def _nuisance_from(arrays: _Arrays, payload: dict[str, Any]) -> NuisanceEstimates:
    return NuisanceEstimates(
        propensity=Propensity(
            arrays.get(payload["propensity"]),
            tuple(float(arm) for arm in payload["propensity_arms"]),
        ),
        outcome=_fit_from(arrays, payload["outcome"]),
        targeting_outcome=(
            None
            if payload["targeting_outcome"] is None
            else _fit_from(arrays, payload["targeting_outcome"])
        ),
        scaler=OutcomeScaler(payload["scaler"]["lower"], payload["scaler"]["upper"]),
        folds=Folds(
            arrays.get(payload["folds"]["assignment"]).astype(np.int64),
            payload["folds"]["n_folds"],
        ),
        missingness=arrays.get(payload["missingness"]),
        intermediate=arrays.get(payload["intermediate"]),
        treatment_covariates=tuple(payload["treatment_covariates"]),
        # Learner diagnostics are reporting objects rather than estimation inputs and
        # are not written; nuisance() falls back to what the arrays support.
        diagnostics={},
        outcome_task=payload["outcome_task"],
        regimes=_regimes_from(arrays, payload.get("regimes")),
        density=_density_from(arrays, payload.get("density")),
        shifts=_shifts_from(arrays, payload.get("shifts")),
        incremental=_incremental_from(arrays, payload.get("incremental")),
        msm=_msm_from(arrays, payload.get("msm")),
        reduced=_reduced_from(arrays, payload.get("reduced")),
    )


def _reduced_from(
    arrays: _Arrays, payload: dict[str, Any] | None
) -> ReducedSet | MissingOutcomeReducedSet | None:
    if payload is None:
        return None
    if payload.get("kind") == "missing_outcome":
        lower, upper = payload["g_bounds"]
        return MissingOutcomeReducedSet(
            gamma_a=arrays.get(payload["gamma_a"]),
            gamma_m=arrays.get(payload["gamma_m"]),
            r_a=arrays.get(payload["r_a"]),
            r_m=arrays.get(payload["r_m"]),
            e=arrays.get(payload["e"]),
            arms=tuple(float(arm) for arm in payload["arms"]),
            g_bounds=(float(lower), float(upper)),
            missingness_bound=float(payload["missingness_bound"]),
        )
    lower, upper = payload["g_bounds"]
    return ReducedSet(
        qr=arrays.get(payload["qr"]),
        gr1=arrays.get(payload["gr1"]),
        gr2=arrays.get(payload["gr2"]),
        arms=tuple(float(arm) for arm in payload["arms"]),
        g_bounds=(float(lower), float(upper)),
        reduction=str(payload["reduction"]),
    )


def _regimes_from(arrays: _Arrays, payload: dict[str, Any] | None) -> RegimeSet | None:
    if payload is None:
        return None
    return RegimeSet(
        tuple(payload["names"]),
        arrays.get(payload["values"]),
        float(payload["reference"]),
    )


def _msm_from(arrays: _Arrays, payload: dict[str, Any] | None) -> MSMSet | None:
    if payload is None:
        return None
    return MSMSet(
        terms=tuple(payload["terms"]),
        design=arrays.get(payload["design"]),
        weights=arrays.get(payload["weights"]),
        arms=tuple(float(arm) for arm in payload["arms"]),
        link=payload["link"],
        clever_weights=arrays.get(payload["clever_weights"]),
        observed_design=arrays.get(payload["observed_design"]),
        observed_weights=arrays.get(payload["observed_weights"]),
        dose_values=tuple(float(value) for value in payload["dose_values"]),
    )


def _density_from(arrays: _Arrays, payload: dict[str, Any] | None) -> ConditionalDensity | None:
    if payload is None:
        return None
    return ConditionalDensity(
        arrays.get(payload["bin_probabilities"]),
        arrays.get(payload["edges"]),
    )


def _shifts_from(arrays: _Arrays, payload: dict[str, Any] | None) -> ShiftSet | None:
    if payload is None:
        return None
    return ShiftSet(
        tuple(payload["names"]),
        tuple(float(delta) for delta in payload["deltas"]),
        arrays.get(payload["shifted"]),
        arrays.get(payload["ratio"]),
        arrays.get(payload["ratio_at"]),
        arrays.get(payload["capped"]).astype(bool),
        float(payload["reference"]),
    )


def _incremental_from(arrays: _Arrays, payload: dict[str, Any] | None) -> IPSISet | None:
    if payload is None:
        return None
    return IPSISet(
        tuple(payload["names"]),
        tuple(float(delta) for delta in payload["deltas"]),
        arrays.get(payload["values"]),
        arrays.get(payload["weights"]),
        arrays.get(payload["derivative"]),
        arrays.get(payload["propensity"]),
        float(payload["reference"]),
    )


def _config_to(config: TMLEConfig) -> dict[str, Any]:
    spec = config.targeting_spec
    return {
        "family": config.family,
        "targeting_spec": {
            "targeting": spec.targeting,
            "fluctuation": spec.fluctuation,
            "target_weights": spec.target_weights,
            "alpha": spec.alpha,
            "max_iter": spec.max_iter,
            "tol": spec.tol,
            "step_size": spec.step_size,
        },
        "targeting_scheme": config.targeting_scheme,
        "cross_fit": config.cross_fit,
        "n_folds": config.n_folds,
        "g_bounds": list(config.g_bounds),
        "g_bounds_conditional": list(config.g_bounds_conditional),
        "missingness_bound": config.missingness_bound,
        "q_bounds": None if config.q_bounds is None else list(config.q_bounds),
        "screen_treatment": config.screen_treatment,
        "estimands": list(config.estimands),
        "alpha_sig": config.alpha_sig,
        "random_state": config.random_state,
        "n_bootstrap": config.n_bootstrap,
        "cv_evaluation": config.cv_evaluation,
        "auto_bounds_n": config.auto_bounds_n,
        "bounded_mechanisms": list(config.bounded_mechanisms),
        "reference_arm": config.reference_arm,
        "parameter_axis": config.parameter_axis,
        # Flat, like targeting_spec above and for the same reason: a plan is numbers and
        # strings, so it needs no codec of its own.
        "crossfit": {
            "n_folds": config.crossfit.n_folds,
            "learner_folds": config.crossfit.learner_folds,
            "scheme": config.crossfit.scheme,
            "stratify_by": list(config.crossfit.stratify_by),
            "random_state": config.crossfit.random_state,
            "repeats": config.crossfit.repeats,
        },
    }


def _config_from(payload: dict[str, Any]) -> TMLEConfig:
    return TMLEConfig(
        family=payload["family"],
        targeting_spec=TargetingSpec(**payload["targeting_spec"]),
        targeting_scheme=payload["targeting_scheme"],
        cross_fit=payload["cross_fit"],
        n_folds=payload["n_folds"],
        g_bounds=tuple(payload["g_bounds"]),
        g_bounds_conditional=tuple(payload["g_bounds_conditional"]),
        missingness_bound=payload["missingness_bound"],
        q_bounds=None if payload["q_bounds"] is None else tuple(payload["q_bounds"]),
        screen_treatment=payload["screen_treatment"],
        estimands=tuple(payload["estimands"]),
        alpha_sig=payload["alpha_sig"],
        random_state=payload["random_state"],
        n_bootstrap=payload["n_bootstrap"],
        cv_evaluation=payload["cv_evaluation"],
        auto_bounds_n=payload["auto_bounds_n"],
        bounded_mechanisms=tuple(payload["bounded_mechanisms"]),
        reference_arm=float(payload["reference_arm"]),
        parameter_axis=payload["parameter_axis"],
        crossfit=CrossFitPlan(
            **{**payload["crossfit"], "stratify_by": tuple(payload["crossfit"]["stratify_by"])}
        ),
    )


def _ctmle_extra_to(arrays: _Arrays, extra: dict[str, Any]) -> dict[str, Any] | None:
    value = extra.get("ctmle")
    if value is None:
        return None
    from .ctmle import CTMLEOutcomeAdaptiveFit, CTMLESelection

    if isinstance(value, CTMLEOutcomeAdaptiveFit):
        return {
            "kind": "oat",
            "strategy": value.strategy,
            "treatment_features": list(value.treatment_features),
            "treatment_risk": value.treatment_risk,
        }
    if not isinstance(value, CTMLESelection):
        return None
    return {
        "kind": "selection",
        "strategy": value.strategy,
        "preorder": value.preorder,
        "estimand": value.estimand,
        "target_names": list(value.target_names),
        "loss": value.loss,
        "penalized": value.penalized,
        "path": [list(names) for names in value.path],
        "n_steps": list(value.n_steps),
        "train_risk": arrays.put("extra.ctmle.train_risk", value.train_risk),
        "train_loss": arrays.put("extra.ctmle.train_loss", value.train_loss),
        "penalty": arrays.put("extra.ctmle.penalty", value.penalty),
        "treatment_risk": arrays.put("extra.ctmle.treatment_risk", value.treatment_risk),
        "cv_risk": arrays.put("extra.ctmle.cv_risk", value.cv_risk),
        "selected": value.selected,
        "covariates": list(value.covariates),
    }


def _ctmle_extra_from(arrays: _Arrays, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    from .ctmle import CTMLEOutcomeAdaptiveFit, CTMLESelection

    if payload["kind"] == "oat":
        return {
            "ctmle": CTMLEOutcomeAdaptiveFit(
                strategy=payload["strategy"],
                treatment_features=tuple(payload["treatment_features"]),
                treatment_risk=float(payload["treatment_risk"]),
            )
        }
    return {
        "ctmle": CTMLESelection(
            strategy=payload["strategy"],
            preorder=payload["preorder"],
            estimand=payload["estimand"],
            target_names=tuple(payload["target_names"]),
            loss=payload["loss"],
            penalized=payload["penalized"],
            path=tuple(tuple(names) for names in payload["path"]),
            n_steps=tuple(payload["n_steps"]),
            train_risk=arrays.get(payload["train_risk"]),
            train_loss=arrays.get(payload["train_loss"]),
            penalty=arrays.get(payload["penalty"]),
            treatment_risk=arrays.get(payload["treatment_risk"]),
            cv_risk=arrays.get(payload["cv_risk"]),
            selected=int(payload["selected"]),
            covariates=tuple(payload["covariates"]),
        )
    }


def result_to_dict(result: TMLEResult) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Split a result into a JSON-safe manifest and a dictionary of arrays."""
    arrays = _Arrays()
    ctmle_extra = _ctmle_extra_to(arrays, result.extra)
    unsupported_extra = bool(set(result.extra) - {"ctmle"}) or bool(
        result.extra and ctmle_extra is None
    )
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "estimates": {
            name: _estimate_to(arrays, f"est.{name}", est) for name, est in result.estimates.items()
        },
        # Every draw, not the first one: a repeated fit's estimates are the average over
        # all R, so a file holding one of them would reload as a result whose reported
        # numbers no analysis could reproduce.
        "repeats": [
            {
                "fluctuations": {
                    group: _fluctuation_to(arrays, f"fluc.{index}.{group}", fl)
                    for group, fl in repeat.fluctuations.items()
                },
                "nuisance": _nuisance_to(arrays, f"nuisance.{index}", repeat.nuisance),
                # Plain floats in the manifest rather than an array: this is what
                # `repeat_spread` reads, and it must survive the round trip or a reloaded
                # repeated fit could not say how much the split moved it.
                "psi": {name: float(value) for name, value in repeat.psi.items()},
            }
            for index, repeat in enumerate(result.repeats)
        ],
        "data": _data_to(arrays, result.data),
        "config": _config_to(result.config),
        "intermediate_value": result.intermediate_value,
        "ctmle_extra": ctmle_extra,
        "provenance": None if result.provenance is None else result.provenance.to_dict(),
        "recipe": (
            None
            if result.estimator is None
            else TMLERecipe.from_estimator(result.estimator).to_dict()
        ),
        # Dropped deliberately, and named so the omission is visible rather than
        # discovered: the first two are reporting objects rebuilt on demand from the
        # arrays that *are* stored, and the third is whatever `extra` this format has
        # no persisted form for.  `unsupported_extra` is a bool, so it is mapped to
        # `None` rather than left to the filter below -- `False is not None` is true,
        # and letting it through named `extra` dropped in every file ever written.
        "dropped": sorted(
            key
            for key, value in (
                ("simultaneous", result.simultaneous),
                ("bootstrap", result.bootstrap),
                ("extra", unsupported_extra or None),
            )
            if value is not None
        ),
    }
    return manifest, arrays.store


def result_from_dict(manifest: dict[str, Any], store: dict[str, FloatArray]) -> TMLEResult:
    """Rebuild a result from what :func:`result_to_dict` produced."""
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"this file is format version {version}, but this cleverly reads version "
            f"{FORMAT_VERSION}. Upgrade cleverly, or re-run the fit."
        )
    arrays = _Arrays()
    arrays.store = store

    recipe_payload = manifest.get("recipe")
    recipe = None if recipe_payload is None else TMLERecipe.from_dict(recipe_payload)
    provenance = manifest.get("provenance")
    return TMLEResult(
        estimates={
            name: _estimate_from(arrays, payload) for name, payload in manifest["estimates"].items()
        },
        repeats=tuple(
            RepeatFit(
                nuisance=_nuisance_from(arrays, payload["nuisance"]),
                fluctuations={
                    group: _fluctuation_from(arrays, fluctuation)
                    for group, fluctuation in payload["fluctuations"].items()
                },
                # `.get` rather than `[...]`: a file written before the field existed is
                # readable, and reloads with an empty mapping that `repeat_spread` skips.
                psi=payload.get("psi", {}),
            )
            for payload in manifest["repeats"]
        ),
        data=_data_from(arrays, manifest["data"]),
        config=_config_from(manifest["config"]),
        estimator=_LazyEstimator(recipe) if recipe is not None else None,
        provenance=None if provenance is None else Provenance.from_dict(provenance),
        intermediate_value=manifest["intermediate_value"],
        extra=_ctmle_extra_from(arrays, manifest.get("ctmle_extra")),
    )


class _LazyEstimator:
    """Rebuilds the estimator from the recipe the first time one is asked for.

    A round-tripped result reaches this only through ``result.estimator``.  Every
    ``retarget``-based analysis goes through it and works; the two refit-based ones
    surface :class:`TMLERecipe`'s error, which says exactly why.
    """

    def __init__(self, recipe: TMLERecipe) -> None:
        self._recipe = recipe
        self._built: Any = None
        self._retargeter: Any = None

    def __getattr__(self, name: str) -> Any:
        if name == "retarget":
            if self._retargeter is None:
                self._retargeter = self._recipe.build_for_retarget()
            return self._retargeter.retarget
        if self._built is None:
            self._built = self._recipe.build()
        return getattr(self._built, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "built" if self._built is not None else "not yet built"
        return f"<estimator rebuilt from recipe ({state})>"


def save(result: TMLEResult, path: str | Path) -> Path:
    """Write a result to a single ``.npz`` file.

    >>> from cleverly.estimators.serialize import save, load    # doctest: +SKIP
    >>> save(result, "fit.npz")                                  # doctest: +SKIP
    >>> reloaded = load("fit.npz")                               # doctest: +SKIP
    """
    destination = Path(path)
    manifest, store = result_to_dict(result)
    payload = dict(store)
    payload["__manifest__"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    with destination.open("wb") as handle:
        _write_npz(handle, payload)
    return destination


def _write_npz(handle: Any, payload: dict[str, FloatArray]) -> None:
    """``savez_compressed`` with the arrays named by the manifest.

    Wrapped because numpy's stubs have twice declared the second positional parameter of
    ``savez_compressed`` as ``compress: bool``, so splatting the payload as keywords --
    which is how the function is meant to be called, and what the runtime signature
    ``(file, *args, **kwds)`` accepts -- did not type-check at the call site.

    That went round three times.  Stubs described it correctly, the suppression became an
    unused ignore and was deleted; numpy 2.4 declared ``compress: bool`` again and a
    ``# type: ignore[arg-type]`` went back; 2.4.6 fixed the stubs and ``warn_unused_ignores``
    turned the ignore itself red.  Since ``numpy`` is an unpinned dependency, *either*
    state is one release away and whichever one this line is written for is the one CI
    will eventually reject.

    So the cast, rather than a fourth turn of the same handle: it is correct under both
    versions of the stub because it does not describe ``savez_compressed`` at all, and
    there is no suppression left to go stale.  The runtime call is unchanged.
    """
    savez: Any = np.savez_compressed
    savez(handle, **payload)


def load(path: str | Path) -> TMLEResult:
    """Read back a result written by :func:`save`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(bytes(archive["__manifest__"]).decode("utf-8"))
        store = {key: archive[key] for key in archive.files if key != "__manifest__"}
    return result_from_dict(manifest, store)


def dumps(result: TMLEResult) -> bytes:
    """The same payload as :func:`save`, in memory."""
    buffer = io.BytesIO()
    manifest, store = result_to_dict(result)
    payload = dict(store)
    payload["__manifest__"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    _write_npz(buffer, payload)
    return buffer.getvalue()


def loads(blob: bytes) -> TMLEResult:
    """Read back what :func:`dumps` produced."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        manifest = json.loads(bytes(archive["__manifest__"]).decode("utf-8"))
        store = {key: archive[key] for key in archive.files if key != "__manifest__"}
    return result_from_dict(manifest, store)
