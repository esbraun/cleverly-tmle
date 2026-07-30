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
from ..inference.influence import ParameterEstimate
from ..interventions import RegimeSet, ShiftSet
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity
from ..msm import MSMSet
from ..provenance import Provenance
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates, Propensity
from .base import TMLEConfig, TMLEResult
from .recipe import TMLERecipe
from .targeting import TargetingSpec

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
FORMAT_VERSION = 5

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
    }


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
    }


def _data_from(arrays: _Arrays, payload: dict[str, Any]) -> CausalData:
    cluster = arrays.get(payload["cluster"])
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
    )


def _nuisance_to(arrays: _Arrays, nuisance: NuisanceEstimates) -> dict[str, Any]:
    return {
        "propensity": arrays.put("nuisance.propensity", nuisance.propensity.values),
        "propensity_arms": [float(arm) for arm in nuisance.propensity.arms],
        "outcome": _fit_to(arrays, "nuisance.outcome", nuisance.outcome),
        "scaler": {"lower": nuisance.scaler.lower, "upper": nuisance.scaler.upper},
        "folds": {
            "assignment": arrays.put("nuisance.folds", nuisance.folds.assignment),
            "n_folds": nuisance.folds.n_folds,
        },
        "missingness": arrays.put("nuisance.missingness", nuisance.missingness),
        "intermediate": arrays.put("nuisance.intermediate", nuisance.intermediate),
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
                "values": arrays.put("nuisance.regimes", nuisance.regimes.values),
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
                    "nuisance.density", nuisance.density.bin_probabilities
                ),
                "edges": arrays.put("nuisance.density.edges", nuisance.density.edges),
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
                "shifted": arrays.put("nuisance.shifts.shifted", nuisance.shifts.shifted),
                "ratio": arrays.put("nuisance.shifts.ratio", nuisance.shifts.ratio),
                "ratio_at": arrays.put("nuisance.shifts.ratio_at", nuisance.shifts.ratio_at),
                "capped": arrays.put("nuisance.shifts.capped", nuisance.shifts.capped),
                "reference": float(nuisance.shifts.reference),
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
                "design": arrays.put("nuisance.msm.design", nuisance.msm.design),
                "weights": arrays.put("nuisance.msm.weights", nuisance.msm.weights),
                "arms": [float(arm) for arm in nuisance.msm.arms],
            }
        ),
    }


def _nuisance_from(arrays: _Arrays, payload: dict[str, Any]) -> NuisanceEstimates:
    return NuisanceEstimates(
        propensity=Propensity(
            arrays.get(payload["propensity"]),
            tuple(float(arm) for arm in payload["propensity_arms"]),
        ),
        outcome=_fit_from(arrays, payload["outcome"]),
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
        msm=_msm_from(arrays, payload.get("msm")),
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
        tuple(payload["terms"]),
        arrays.get(payload["design"]),
        arrays.get(payload["weights"]),
        tuple(float(arm) for arm in payload["arms"]),
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
    )


def result_to_dict(result: TMLEResult) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Split a result into a JSON-safe manifest and a dictionary of arrays."""
    arrays = _Arrays()
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "estimates": {
            name: _estimate_to(arrays, f"est.{name}", est) for name, est in result.estimates.items()
        },
        "fluctuations": {
            group: _fluctuation_to(arrays, f"fluc.{group}", fl)
            for group, fl in result.fluctuations.items()
        },
        "nuisance": _nuisance_to(arrays, result.nuisance),
        "data": _data_to(arrays, result.data),
        "config": _config_to(result.config),
        "intermediate_value": result.intermediate_value,
        "provenance": None if result.provenance is None else result.provenance.to_dict(),
        "recipe": (
            None
            if result.estimator is None
            else TMLERecipe.from_estimator(result.estimator).to_dict()
        ),
        # Dropped deliberately, and named so the omission is visible rather than
        # discovered: both are reporting objects rebuilt on demand from the arrays
        # that *are* stored.
        "dropped": sorted(
            key
            for key, value in (
                ("simultaneous", result.simultaneous),
                ("bootstrap", result.bootstrap),
                ("extra", result.extra or None),
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
        fluctuations={
            group: _fluctuation_from(arrays, payload)
            for group, payload in manifest["fluctuations"].items()
        },
        nuisance=_nuisance_from(arrays, manifest["nuisance"]),
        data=_data_from(arrays, manifest["data"]),
        config=_config_from(manifest["config"]),
        estimator=_LazyEstimator(recipe) if recipe is not None else None,
        provenance=None if provenance is None else Provenance.from_dict(provenance),
        intermediate_value=manifest["intermediate_value"],
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

    def __getattr__(self, name: str) -> Any:
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

    Wrapped because numpy's stubs declare the second positional parameter of
    ``savez_compressed`` as ``compress: bool``, so splatting the payload as keywords --
    which is how the function is meant to be called, and what the runtime signature
    ``(file, *args, **kwds)`` accepts -- does not type-check at the call site.
    """
    np.savez_compressed(handle, **payload)  # type: ignore[arg-type]


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
