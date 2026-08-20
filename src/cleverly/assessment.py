"""Capability-aware post-fit diagnostics, validation, and replay metadata.

Assessment lives on a fitted result because its questions depend on the artifacts the
method actually produced.  The facades in this module do not infer support from a result
class and hope for the best: every public operation has a declaration for every public
scalar result family, including deliberate refusals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from .data.weighting import effective_sample_size
from .exceptions import CapabilityError
from .utils.frames import emit_frame
from .utils.text import format_table

__all__ = [
    "ASSESSMENT_CAPABILITIES",
    "AssessmentCapability",
    "AssessmentItem",
    "AssessmentStatus",
    "DiagnosticReport",
    "DiagnosticsFacade",
    "LongitudinalDiagnostics",
    "LongitudinalNuisanceDiagnostics",
    "LongitudinalScoreDiagnostics",
    "Replayability",
    "SensitivityFacade",
    "ValidationReport",
    "assessment_capabilities",
    "replayability",
    "validate_result",
]


class AssessmentStatus(StrEnum):
    """The five deliberately distinct outcomes of an assessment operation."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AssessmentCapability:
    """What one assessment operation needs and how it executes."""

    operation: str
    result_family: str
    methods: tuple[str, ...]
    available: bool
    status: AssessmentStatus
    required_artifacts: tuple[str, ...]
    execution: Literal["summarize", "retarget", "refit"]
    deterministic_from_saved: bool
    interpretation: str
    cost: Literal["cheap", "moderate", "expensive"]
    reason: str | None = None
    #: Arguments the caller must supply for which this operation has no default, so a
    #: combined report cannot run it. Declared here rather than special-cased by name in
    #: ``run_all``, which knows nothing about any particular operation.
    requires_arguments: tuple[str, ...] = ()


def _capability(
    operation: str,
    family: str,
    *,
    artifacts: Sequence[str],
    interpretation: str,
    execution: Literal["summarize", "retarget", "refit"] = "summarize",
    deterministic: bool = True,
    cost: Literal["cheap", "moderate", "expensive"] = "cheap",
    available: bool = True,
    status: AssessmentStatus = AssessmentStatus.PASSED,
    reason: str | None = None,
    requires_arguments: Sequence[str] = (),
) -> AssessmentCapability:
    return AssessmentCapability(
        operation=operation,
        result_family=family,
        methods=("tmle", "collaborative_tmle", "drtmle") if family == "point" else ("tmle",),
        available=available,
        status=status,
        required_artifacts=tuple(artifacts),
        execution=execution,
        deterministic_from_saved=deterministic,
        interpretation=interpretation,
        cost=cost,
        reason=reason,
        requires_arguments=tuple(requires_arguments),
    )


# This is intentionally data rather than branches hidden inside the facade.  Contract tests
# check both directions: every operation covers every family, and no declaration names a
# family or operation that the public result surface no longer exposes.
ASSESSMENT_CAPABILITIES: tuple[AssessmentCapability, ...] = (
    _capability(
        "support",
        "point",
        artifacts=("fitted analytic mechanism", "targeting weights"),
        interpretation="arm, regimen, shift, or incremental support for the identified functional",
    ),
    _capability(
        "nuisance_models",
        "point",
        artifacts=("out-of-fold nuisance predictions",),
        interpretation="held-out fit quality and calibration of each fitted nuisance",
    ),
    _capability(
        "score_equations",
        "point",
        artifacts=("targeting state", "influence curves"),
        interpretation="the score equations the selected method actually solved",
    ),
    _capability(
        "corrections",
        "point",
        artifacts=("doubly-robust correction state",),
        interpretation="the correction identities solved by guarded doubly-robust targeting",
    ),
    _capability(
        "truncation_curve",
        "point",
        artifacts=("fitted nuisance predictions", "targeting state"),
        execution="retarget",
        cost="moderate",
        interpretation="estimate stability across declared mechanism bounds",
    ),
    _capability(
        "refute",
        "point",
        artifacts=("fitted estimator configuration", "analysis data"),
        execution="refit",
        deterministic=False,
        cost="expensive",
        interpretation="behavior under placebo, noise, and subsampling perturbations",
    ),
    _capability(
        "stagewise",
        "point",
        artifacts=(),
        available=False,
        status=AssessmentStatus.NOT_APPLICABLE,
        reason="a point-treatment fit has no sequential nodes",
        interpretation="node-specific longitudinal recursion diagnostics",
    ),
    _capability(
        "support",
        "longitudinal",
        artifacts=("cumulative mechanism products", "regimen histories"),
        interpretation="history-specific support and cumulative leverage at every node",
    ),
    _capability(
        "nuisance_models",
        "longitudinal",
        artifacts=("node pseudo-outcomes", "initial node predictions"),
        interpretation="stagewise held-out loss for the sequential outcome regressions",
    ),
    _capability(
        "score_equations",
        "longitudinal",
        artifacts=("stagewise targeting state",),
        interpretation="one targeting score and convergence record per regimen and node",
    ),
    _capability(
        "corrections",
        "longitudinal",
        artifacts=(),
        available=False,
        status=AssessmentStatus.NOT_APPLICABLE,
        reason="longitudinal targeting does not use the point-treatment correction system",
        interpretation="the correction identities solved by guarded doubly-robust targeting",
    ),
    _capability(
        "truncation_curve",
        "longitudinal",
        artifacts=("sequential nuisance predictions",),
        execution="refit",
        cost="expensive",
        available=False,
        status=AssessmentStatus.UNAVAILABLE,
        reason=(
            "changing a sequential bound changes every earlier pseudo-outcome and requires "
            "a full refit"
        ),
        interpretation="estimate stability across declared mechanism bounds",
    ),
    _capability(
        "refute",
        "longitudinal",
        artifacts=("fitted sequential estimator configuration",),
        execution="refit",
        deterministic=False,
        cost="expensive",
        available=False,
        status=AssessmentStatus.UNAVAILABLE,
        reason="no evidence-backed longitudinal perturbation/refit adapter is implemented",
        interpretation="behavior under longitudinal data perturbations",
    ),
    _capability(
        "stagewise",
        "longitudinal",
        artifacts=("sequential steps", "cumulative mechanism products"),
        interpretation="risk sets, assignment, leverage, truncation, and convergence by node",
    ),
)


def _family(result: Any) -> str:
    name = type(result).__name__
    if name == "TMLEResult":
        return "point"
    if name == "LongitudinalResult":
        return "longitudinal"
    raise TypeError(f"assessment has no declared result family for {name}")


def assessment_capabilities(result: Any) -> tuple[AssessmentCapability, ...]:
    """All operation declarations for the result's family."""

    family = _family(result)
    return tuple(item for item in ASSESSMENT_CAPABILITIES if item.result_family == family)


@dataclass(frozen=True)
class AssessmentItem:
    """One immutable result in a combined diagnostic or validation report."""

    name: str
    status: AssessmentStatus
    detail: str
    next_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticReport:
    """The status of each requested diagnostic, including deliberate omissions."""

    items: tuple[AssessmentItem, ...]
    include_refits: bool = False
    backend: str | None = None

    def __getitem__(self, name: str) -> AssessmentItem:
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"no diagnostic named {name!r}; have {[item.name for item in self.items]}")

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "check": [item.name for item in self.items],
                "status": [item.status.value for item in self.items],
                "detail": [item.detail for item in self.items],
                "next_steps": ["; ".join(item.next_steps) for item in self.items],
            },
            data,
            backend=self.backend,
        )

    def summary(self) -> str:
        return format_table(
            ["diagnostic", "status", "detail"],
            [[item.name, item.status.value, item.detail] for item in self.items],
        )


@dataclass(frozen=True)
class ValidationReport:
    """The inexpensive default battery; it never refits nuisance models."""

    items: tuple[AssessmentItem, ...]
    backend: str | None = None

    @property
    def passed(self) -> bool:
        return all(
            item.status not in {AssessmentStatus.FAILED, AssessmentStatus.UNAVAILABLE}
            for item in self.items
        )

    def __bool__(self) -> bool:
        return self.passed

    def __getitem__(self, name: str) -> AssessmentItem:
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"no validation check named {name!r}")

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "check": [item.name for item in self.items],
                "status": [item.status.value for item in self.items],
                "detail": [item.detail for item in self.items],
                "next_steps": ["; ".join(item.next_steps) for item in self.items],
            },
            data,
            backend=self.backend,
        )

    def summary(self) -> str:
        heading = "Validation: PASS" if self.passed else "Validation: ATTENTION REQUIRED"
        return "\n".join(
            [
                heading,
                "-" * len(heading),
                format_table(
                    ["check", "status", "detail"],
                    [[item.name, item.status.value, item.detail] for item in self.items],
                ),
            ]
        )


@dataclass(frozen=True)
class Replayability:
    """Which post-fit actions remain possible from the current result state."""

    summarize_existing_artifacts: bool
    retarget_cached_nuisances: bool
    evaluate_stored_representer: bool
    refit_nuisances: bool
    evaluate_new_data: bool
    unreconstructible: tuple[str, ...] = ()


def replayability(result: Any) -> Replayability:
    """Derive replay capabilities from stored artifacts and the normalized method."""

    if _family(result) == "longitudinal":
        # Whole-result persistence retains the method and its unfitted learner templates.
        # Changing a bound still requires the entire recursion rather than cached retargeting.
        method = getattr(result, "method", None)
        missing = () if method is not None else ("method configuration",)
        return Replayability(True, False, False, method is not None, False, missing)

    estimator = getattr(result, "estimator", None)
    if estimator is None:
        return Replayability(True, True, False, False, False, ("estimator configuration",))
    return Replayability(True, True, False, True, False)


@dataclass(frozen=True)
class _CachedFrame:
    columns: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]
    backend: str | None

    @classmethod
    def from_frame(cls, frame: Any, backend: str | None) -> _CachedFrame:
        if type(frame).__module__.startswith("polars"):
            payload = frame.to_dict(as_series=False)
        else:
            payload = frame.to_dict(orient="list")
        columns = tuple(str(column) for column in payload)
        values = tuple(
            tuple(_python_scalar(value) for value in payload[column]) for column in columns
        )
        return cls(columns, values, backend)

    def materialize(self) -> Any:
        return emit_frame(dict(zip(self.columns, self.values, strict=True)), backend=self.backend)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _pack_cached(value: Any, backend: str | None) -> Any:
    module = type(value).__module__
    if module.startswith("pandas") or module.startswith("polars"):
        return _CachedFrame.from_frame(value, backend)
    return value


def _unpack_cached(value: Any) -> Any:
    return value.materialize() if isinstance(value, _CachedFrame) else value


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return {"array": [list(value.shape), str(value.dtype), digest]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    return {"object": repr(value), "type": type(value).__qualname__}


def _cache_key(operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    normalized = {"args": _normalize(tuple(args)), "kwargs": _normalize(kwargs)}
    return f"{operation}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"


def _cached(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    compute: Callable[[], Any],
) -> Any:
    cache = result.assessment_cache
    key = _cache_key(operation, args, kwargs)
    if key in cache:
        return _unpack_cached(cache[key])
    value = compute()
    cache[key] = _pack_cached(value, getattr(result.data, "backend", None))
    return value


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
    """The existing longitudinal diagnostic arithmetic in an immutable report."""

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
    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    score: float
    relative_score: float
    converged: bool
    n_iter: int
    failure: str | None


@dataclass(frozen=True)
class LongitudinalScoreDiagnostics:
    rows: tuple[LongitudinalScoreRow, ...]
    backend: str | None = None

    @property
    def passed(self) -> bool:
        return all(row.converged for row in self.rows)

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "regimen": [row.regimen for row in self.rows],
                "cause": [row.cause for row in self.rows],
                "horizon": [row.horizon for row in self.rows],
                "time": [row.time for row in self.rows],
                "score": [row.score for row in self.rows],
                "relative_score": [row.relative_score for row in self.rows],
                "converged": [row.converged for row in self.rows],
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


def _longitudinal_stagewise(result: Any) -> LongitudinalDiagnostics:
    terms = () if result.msm is None else result.msm.terms
    epsilon_names = ("epsilon",) if result.msm is None else tuple(f"epsilon[{t}]" for t in terms)
    categorical = any(len(levels) > 2 for levels in result.data.treatment_levels)
    rows = []
    for fit in result.fits.values():
        for step in fit.steps:
            weights = (fit.obs_weights * step.clever)[step.trained_on]
            assigned = fit.assignment[step.at_risk, step.time - 1]
            if categorical:
                levels = result.data.treatment_levels[step.time - 1]
                assignment: float | str = ", ".join(
                    f"{level}={float(np.mean(assigned == float(code))):.3g}"
                    for code, level in enumerate(levels)
                )
            else:
                assignment = float(np.mean(assigned == 1.0)) if assigned.size else float("nan")
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


def _longitudinal_scores(result: Any) -> LongitudinalScoreDiagnostics:
    rows = []
    for fit in result.fits.values():
        for step in fit.steps:
            fluctuation = step.fluctuation
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    fit.horizon if result.data.is_survival else None,
                    step.time,
                    float(result.scaler.range * fluctuation.score_norm),
                    float(fluctuation.relative_score_norm),
                    bool(fluctuation.converged),
                    int(fluctuation.n_iter),
                    fluctuation.failure,
                )
            )
    return LongitudinalScoreDiagnostics(tuple(rows), result.data.backend)


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


class DiagnosticsFacade:
    """Unified diagnostics for point and longitudinal causal results."""

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        return assessment_capabilities(self._result)

    def capability(self, operation: str) -> AssessmentCapability:
        for item in self.capabilities:
            if item.operation == operation:
                return item
        raise KeyError(f"unknown diagnostic {operation!r}")

    def _require(self, operation: str) -> AssessmentCapability:
        item = self.capability(operation)
        if not item.available:
            raise CapabilityError(f"diagnostic {operation!r} is {item.status.value}: {item.reason}")
        return item

    def stagewise(self) -> LongitudinalDiagnostics:
        self._require("stagewise")
        return _cached(
            self._result,
            "diagnostics.stagewise",
            (),
            {},
            lambda: _longitudinal_stagewise(self._result),
        )

    def support(self) -> Any:
        self._require("support")
        if _family(self._result) == "longitudinal":
            return self.stagewise()

        def compute() -> Any:
            nuisance = self._result.nuisance
            from .interventions import (
                check_incremental_support,
                check_shift_support,
                check_support,
            )
            from .sensitivity.positivity import positivity_report

            if nuisance.regimes is not None:
                return check_support(
                    nuisance.regimes,
                    self._result.data.treatment,
                    nuisance.propensity.values,
                    backend=self._result.data.backend,
                )
            if nuisance.shifts is not None and nuisance.density is not None:
                bound = self._result.config.missingness_bound
                level = self._result.intermediate_value
                mechanisms = [
                    values
                    for values in (
                        nuisance.bounded_missingness(bound),
                        None if level is None else nuisance.intermediate_density(level, bound),
                    )
                    if values is not None
                ]
                return check_shift_support(
                    nuisance.shifts,
                    nuisance.density,
                    self._result.data.treatment,
                    mechanisms=mechanisms,
                )
            if nuisance.incremental is not None:
                return check_incremental_support(nuisance.incremental, self._result.data.treatment)
            return positivity_report(self._result)

        return _cached(self._result, "diagnostics.support", (), {}, compute)

    def nuisance_models(self) -> Any:
        self._require("nuisance_models")
        if _family(self._result) == "point":
            from .validation.nuisance import nuisance_diagnostics

            return _cached(
                self._result,
                "diagnostics.nuisance_models",
                (),
                {},
                lambda: nuisance_diagnostics(self._result),
            )
        return _cached(
            self._result,
            "diagnostics.nuisance_models",
            (),
            {},
            lambda: _longitudinal_nuisances(self._result),
        )

    def score_equations(self, *, tolerance: float = 1e-3) -> Any:
        self._require("score_equations")
        from .validation.score import score_check

        if _family(self._result) == "longitudinal":
            return _cached(
                self._result,
                "diagnostics.score_equations",
                (),
                {"tolerance": tolerance},
                lambda: _longitudinal_scores(self._result),
            )
        return _cached(
            self._result,
            "diagnostics.score_equations",
            (),
            {"tolerance": tolerance},
            lambda: score_check(self._result, tolerance=tolerance),
        )

    def corrections(self, *, tolerance: float = 1e-3, identity_tolerance: float = 1e-12) -> Any:
        self._require("corrections")
        from .validation.drtmle import correction_check

        return _cached(
            self._result,
            "diagnostics.corrections",
            (),
            {"tolerance": tolerance, "identity_tolerance": identity_tolerance},
            lambda: correction_check(
                self._result,
                tolerance=tolerance,
                identity_tolerance=identity_tolerance,
            ),
        )

    def truncation_curve(
        self,
        bounds: Sequence[float] | None = None,
        *,
        estimands: Sequence[str] | None = None,
        mechanism: bool = False,
    ) -> Any:
        self._require("truncation_curve")
        from .sensitivity.positivity import truncation_curve

        if self._result.nuisance.incremental is not None and not mechanism:
            raise ValueError(
                "the propensity g is *inside* the estimand for an incremental intervention, "
                "so a propensity-bound curve would compare different parameters; use "
                "diagnostics.support(), or pass mechanism=True when a separate observation "
                "mechanism was fitted"
            )
        return _cached(
            self._result,
            "diagnostics.truncation_curve",
            (bounds,),
            {"estimands": estimands, "mechanism": mechanism},
            lambda: truncation_curve(
                self._result, bounds, estimands=estimands, mechanism=mechanism
            ),
        )

    def refute(self, **kwargs: Any) -> Any:
        self._require("refute")
        if not replayability(self._result).refit_nuisances:
            missing = replayability(self._result).unreconstructible
            raise CapabilityError(
                "refutation requires nuisance refits, but this estimator cannot be "
                f"reconstructed; unavailable slots: {list(missing)}"
            )
        from .validation.refute import refute

        return _cached(
            self._result,
            "diagnostics.refute",
            (),
            kwargs,
            lambda: refute(self._result, **kwargs),
        )

    def run_all(self, *, include_refits: bool = False) -> DiagnosticReport:
        def compute() -> DiagnosticReport:
            items = []
            for capability in self.capabilities:
                if capability.operation == "stagewise" and _family(self._result) == "point":
                    items.append(_item_from_capability(capability))
                    continue
                if capability.execution != "summarize" and not include_refits:
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            "not run by default because it retargets or refits models; "
                            "pass include_refits=True",
                        )
                    )
                    continue
                if not capability.available:
                    items.append(_item_from_capability(capability))
                    continue
                try:
                    report = getattr(self, capability.operation)()
                    items.append(_diagnostic_item(capability.operation, report))
                except (CapabilityError, ValueError) as error:
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            str(error),
                        )
                    )
            return DiagnosticReport(tuple(items), include_refits, self._result.data.backend)

        return _cached(
            self._result,
            "diagnostics.run_all",
            (),
            {"include_refits": include_refits},
            compute,
        )


def _item_from_capability(capability: AssessmentCapability) -> AssessmentItem:
    return AssessmentItem(
        capability.operation,
        capability.status,
        capability.reason or capability.interpretation,
    )


def _diagnostic_item(name: str, report: Any) -> AssessmentItem:
    passed = getattr(report, "passed", None)
    if passed is False:
        return AssessmentItem(name, AssessmentStatus.FAILED, "the diagnostic reported a failure")
    if name == "support":
        warning = _support_warning(report)
        return AssessmentItem(
            name,
            AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
            warning or "no material support warning in the stored diagnostic",
        )
    return AssessmentItem(name, AssessmentStatus.PASSED, "completed from stored fitted artifacts")


def _support_warning(report: Any) -> str | None:
    if hasattr(report, "truncated"):
        fraction = float(report.truncated.get("fraction", 0.0))
        if fraction > 0.05:
            return f"{fraction:.1%} of propensity cells were truncated"
        ratios = [
            float(item.get("ess_ratio", 1.0)) for item in getattr(report, "mechanisms", {}).values()
        ]
        if ratios and min(ratios) < 0.2:
            return "a fitted mechanism leaves less than 20% effective sample size"
    if isinstance(report, LongitudinalDiagnostics):
        if any(row.share_truncated > 0.05 for row in report.rows):
            return "cumulative mechanism truncation exceeds 5% at one or more nodes"
        if any(row.n_followed and row.effective_n / row.n_followed < 0.2 for row in report.rows):
            return "cumulative weights leave less than 20% effective sample size at a node"
    if isinstance(report, Mapping):
        # A shift or IPSI fit reports one support record per declared intervention rather
        # than a single object, so none of the attribute probes above sees it.  These
        # dataclasses carry the same quantities the other branches threshold on, at the
        # same tiers; read them by field so the two classes need no separate branches --
        # ``IncrementalSupport`` has an ``ess_ratio`` but no ``unsupported`` or
        # ``capped_fraction``, and a missing field must not read as a breach.
        for name, item in report.items():
            if int(getattr(item, "unsupported", 0)) > 0:
                return f"intervention {name!r} has units with estimated zero support"
            if float(getattr(item, "ess_ratio", 1.0)) < 0.2:
                return f"intervention {name!r} leaves less than 20% effective sample size"
            if float(getattr(item, "capped_fraction", 0.0)) > 0.05:
                return f"intervention {name!r} had more than 5% of its weights capped"
        return None
    regimes = getattr(report, "regimes", None)
    if regimes and any(item.unsupported for item in regimes.values()):
        return "one or more declared regimes has estimated zero support"
    return None


def validate_result(result: Any) -> ValidationReport:
    """Run only cheap, cache-only checks appropriate to this fitted method."""

    def compute() -> ValidationReport:
        diagnostics = result.diagnostics
        score = diagnostics.score_equations()
        score_passed = bool(getattr(score, "passed", False))
        score_rows = tuple(getattr(score, "rows", ()))
        conditioning = _reduction_conditioning_warning(result) if score_passed else None
        items = [
            AssessmentItem(
                "score_equations",
                (
                    AssessmentStatus.WARNING
                    if conditioning
                    else AssessmentStatus.PASSED
                    if score_passed
                    else AssessmentStatus.FAILED
                ),
                (
                    conditioning or f"all {len(score_rows)} stored score checks converged"
                    if score_passed
                    else "one or more stored score equations failed its convergence check"
                ),
                (("inspect result.repeats[*].fluctuations['mean'].reduction.ill_conditioned"),)
                if conditioning
                else ()
                if score_passed
                else ("inspect result.diagnostics.score_equations()",),
            )
        ]
        support = diagnostics.support()
        warning = _support_warning(support)
        items.append(
            AssessmentItem(
                "support",
                AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
                warning or "stored support diagnostics show no material warning",
                () if not warning else ("inspect result.diagnostics.support()",),
            )
        )
        nuisance = diagnostics.nuisance_models()
        finite = []
        if isinstance(nuisance, LongitudinalNuisanceDiagnostics):
            finite = [row.mse for row in nuisance.rows if np.isfinite(row.mse)]
        nuisance_status = (
            AssessmentStatus.WARNING
            if isinstance(nuisance, LongitudinalNuisanceDiagnostics) and not finite
            else AssessmentStatus.PASSED
        )
        items.append(
            AssessmentItem(
                "nuisance_models",
                nuisance_status,
                "stagewise held-out losses are available"
                if isinstance(nuisance, LongitudinalNuisanceDiagnostics)
                else "out-of-fold nuisance fit and calibration diagnostics are available",
                ("inspect result.diagnostics.nuisance_models()",),
            )
        )
        return ValidationReport(tuple(items), result.data.backend)

    return _cached(result, "validate", (), {}, compute)


def _reduction_conditioning_warning(result: Any) -> str | None:
    """Summarize successful equation-(10) solves that needed an ill-conditioned step."""
    ill_conditioned = 0
    rounds = 0
    affected = 0
    repeats = tuple(getattr(result, "repeats", ()))
    for repeat in repeats:
        fluctuation = getattr(repeat, "fluctuations", {}).get("mean")
        reduction = getattr(fluctuation, "reduction", None)
        count = int(getattr(reduction, "ill_conditioned", 0))
        rounds += int(getattr(reduction, "rounds", 0))
        if count <= 0:
            continue
        ill_conditioned += count
        affected += 1
    if ill_conditioned == 0:
        return None
    fraction = ill_conditioned / rounds if rounds else float("nan")
    draws = f" across {affected} of {len(repeats)} repeat(s)" if len(repeats) > 1 else ""
    return (
        f"all stored score checks converged, but equation (10) reported numerical difficulty in "
        f"{ill_conditioned} of {rounds} refitting round(s) ({fraction:.1%}){draws}; "
        "one or more inner solves were ill-conditioned or stopped at their numerical "
        "tolerance even though the returned score equations passed"
    )


class SensitivityFacade:
    """Capability-aware sensitivity operations with normalized persistent caching."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def _unavailable(self, operation: str, *, not_applicable: bool = False) -> CapabilityError:
        status = "not_applicable" if not_applicable else "unavailable"
        if _family(self._result) == "longitudinal":
            reason = (
                "changing a sequential mechanism bound changes every earlier pseudo-outcome "
                "and requires a full evidence-backed recursion/refit adapter"
            )
        else:
            reason = "the fitted artifacts or published sensitivity derivation are absent"
        return CapabilityError(f"sensitivity {operation!r} is {status}: {reason}")

    @property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        family = _family(self._result)
        longitudinal = family == "longitudinal"
        missing = (
            False
            if longitudinal
            else getattr(self._result.nuisance, "missingness", None) is not None
        )
        benchmarkable = not longitudinal and replayability(self._result).refit_nuisances
        available = not longitudinal
        status = AssessmentStatus.PASSED if available else AssessmentStatus.UNAVAILABLE
        reason = "no longitudinal sensitivity derivation is registered" if longitudinal else None

        def standard(
            operation: str,
            *,
            artifacts: Sequence[str],
            interpretation: str,
            cost: Literal["cheap", "moderate", "expensive"] = "cheap",
        ) -> AssessmentCapability:
            return _capability(
                operation,
                family,
                artifacts=artifacts,
                interpretation=interpretation,
                cost=cost,
                available=available,
                status=status,
                reason=reason,
            )

        return (
            standard(
                "omitted_confounding",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="omitted-confounder bias for the fitted orthogonal score",
            ),
            standard(
                "robustness_value",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="confounding strength needed to move the estimate to its null",
            ),
            standard(
                "elements",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="raw components of the omitted-confounder bias bound",
            ),
            _capability(
                "benchmark",
                artifacts=("fitted estimator configuration",),
                execution="refit",
                deterministic=False,
                cost="expensive",
                interpretation="calibration against named observed covariates",
                available=benchmarkable,
                status=AssessmentStatus.PASSED if benchmarkable else AssessmentStatus.UNAVAILABLE,
                reason=(
                    None
                    if benchmarkable
                    else "benchmarking requires a replayable point-treatment estimator"
                ),
                requires_arguments=("covariates",),
                family=family,
            ),
            standard(
                "contour",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="bias bounds over a grid of confounding strengths",
                cost="moderate",
            ),
            standard(
                "evalue",
                artifacts=("ratio-scale estimate",),
                interpretation="minimum risk-ratio association needed to explain away an effect",
            ),
            _capability(
                "missingness",
                family=family,
                artifacts=("observation mechanism", "published tilt identification"),
                execution="retarget",
                cost="moderate",
                interpretation="departure from missing-at-random identification",
                available=missing,
                status=(
                    AssessmentStatus.PASSED
                    if missing
                    else AssessmentStatus.UNAVAILABLE
                    if longitudinal
                    else AssessmentStatus.NOT_APPLICABLE
                ),
                reason=(
                    None
                    if missing
                    else "no longitudinal missingness-tilt adapter is implemented"
                    if longitudinal
                    else "the identified functional has no observation mechanism"
                ),
            ),
            _capability(
                "tipping_gamma",
                family=family,
                artifacts=("observation mechanism", "published tilt identification"),
                execution="retarget",
                cost="moderate",
                interpretation="missingness departure at which the conclusion reaches its null",
                available=missing,
                status=(
                    AssessmentStatus.PASSED
                    if missing
                    else AssessmentStatus.UNAVAILABLE
                    if longitudinal
                    else AssessmentStatus.NOT_APPLICABLE
                ),
                reason=(
                    None
                    if missing
                    else "no longitudinal missingness-tilt adapter is implemented"
                    if longitudinal
                    else "the identified functional has no observation mechanism"
                ),
            ),
        )

    def _capability(self, operation: str) -> AssessmentCapability:
        return next(item for item in self.capabilities if item.operation == operation)

    def omitted_confounding(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity(
            "omitted_confounding",
            "sensitivity.omitted_variable",
            "omitted_variable_bounds",
            args,
            kwargs,
        )

    def robustness_value(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity(
            "robustness_value", "sensitivity.omitted_variable", "robustness_value", args, kwargs
        )

    def elements(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity(
            "elements", "sensitivity.omitted_variable", "sensitivity_elements", args, kwargs
        )

    def benchmark(self, *args: Any, **kwargs: Any) -> Any:
        capability = self._capability("benchmark")
        if not capability.available:
            raise self._unavailable("benchmark")
        from .sensitivity.omitted_variable import benchmark

        return _cached(
            self._result,
            "sensitivity.benchmark",
            args,
            kwargs,
            lambda: benchmark(self._result, *args, **kwargs),
        )

    def contour(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity(
            "contour", "sensitivity.omitted_variable", "contour_data", args, kwargs
        )

    def evalue(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity("evalue", "sensitivity.evalue", "evalue", args, kwargs)

    def missingness(self, *args: Any, **kwargs: Any) -> Any:
        capability = self._capability("missingness")
        if not capability.available:
            raise self._unavailable(
                "missingness", not_applicable=capability.status == AssessmentStatus.NOT_APPLICABLE
            )
        from .sensitivity.missingness import missingness_tilt

        return _cached(
            self._result,
            "sensitivity.missingness",
            args,
            kwargs,
            lambda: missingness_tilt(self._result, *args, **kwargs),
        )

    def tipping_gamma(self, *args: Any, **kwargs: Any) -> Any:
        return self._cached_sensitivity(
            "tipping_gamma", "sensitivity.missingness", "tipping_gamma", args, kwargs
        )

    def _cached_sensitivity(
        self,
        operation: str,
        module_name: str,
        function_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        capability = self._capability(operation)
        if not capability.available:
            raise self._unavailable(
                operation, not_applicable=capability.status == AssessmentStatus.NOT_APPLICABLE
            )
        import importlib

        if operation in {"omitted_confounding", "robustness_value", "elements", "contour"}:
            args = self._with_default_parameter(args, kwargs)
        function = getattr(importlib.import_module(f"cleverly.{module_name}"), function_name)
        return _cached(
            self._result,
            f"sensitivity.{operation}",
            args,
            kwargs,
            lambda: function(self._result, *args, **kwargs),
        )

    def _with_default_parameter(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Choose the first reported linear contrast when the binary short name is absent."""
        if args or "estimand" in kwargs or "ate" in self._result.estimates:
            return args
        from .sensitivity._parameters import arm_parameters

        known = arm_parameters(self._result)
        selected = next((name for name in self._result.estimates if name in known), None)
        return args if selected is None else (selected,)

    def run_all(self, *, include_refits: bool = False) -> DiagnosticReport:
        def compute() -> DiagnosticReport:
            items = []
            for capability in self.capabilities:
                if not capability.available:
                    items.append(_item_from_capability(capability))
                    continue
                if capability.execution == "refit" and not include_refits:
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            "not run by default because it refits nuisance models",
                        )
                    )
                    continue
                if capability.requires_arguments:
                    # A combined report runs every operation argument-free, so one with a
                    # required argument and no default cannot appear in it.  Choosing a
                    # value here -- which covariates to benchmark against -- would be a
                    # scientific choice made silently on the caller's behalf.
                    needed = ", ".join(capability.requires_arguments)
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            f"needs an explicit {needed} argument, which a combined report "
                            f"has no basis to choose",
                            (
                                f"call result.sensitivity.{capability.operation}() "
                                f"directly with {needed}",
                            ),
                        )
                    )
                    continue
                try:
                    getattr(self, capability.operation)()
                except (CapabilityError, KeyError, TypeError, ValueError) as error:
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            str(error),
                        )
                    )
                else:
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.PASSED,
                            "completed from the fitted method's declared artifacts",
                        )
                    )
            return DiagnosticReport(tuple(items), include_refits, self._result.data.backend)

        return _cached(
            self._result,
            "sensitivity.run_all",
            (),
            {"include_refits": include_refits},
            compute,
        )
