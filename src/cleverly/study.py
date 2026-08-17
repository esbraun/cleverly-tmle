"""Causal-question-first public workflow for identified effects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._typing import Family
from .data import CausalData
from .estimators import TMLE, TMLEResult
from .exceptions import CapabilityError, CleverlyError, DataError
from .methods import MethodAvailability, TMLEMethod
from .targets import TARGETS
from .targets.base import Identification, parameter_name

__all__ = [
    "ATE",
    "BackdoorMeanContrast",
    "CausalStudy",
    "ExplicitAdjustmentProvider",
    "IdentifiedEffect",
    "ParameterKey",
    "PointTreatment",
]


@dataclass(frozen=True)
class PointTreatment:
    """Column roles and design declarations for a point-treatment study."""

    outcome: str
    treatment: str
    adjustment: Sequence[str] = field(default_factory=tuple)
    missingness: str | None = None
    weights: str | None = None
    # The single supported reading, which ``data.weighting.resolve_weight_kind`` already
    # refuses to widen at runtime; spelling it as a one-value ``Literal`` moves that refusal
    # to the type checker rather than restating it.
    weights_type: Literal["probability"] = "probability"
    weights_estimated: bool = False
    cluster: str | None = None
    strata: Sequence[str] = field(default_factory=tuple)
    treatment_kind: Literal["discrete", "continuous"] = "discrete"
    outcome_family: Family = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjustment", tuple(self.adjustment))
        object.__setattr__(self, "strata", tuple(self.strata))
        if not self.outcome or not self.treatment:
            raise DataError("outcome and treatment must be non-empty column names")
        if self.outcome == self.treatment:
            raise DataError("outcome and treatment must name different columns")
        if len(set(self.adjustment)) != len(self.adjustment):
            raise DataError(f"adjustment contains duplicate columns: {list(self.adjustment)}")
        if not self.adjustment:
            raise DataError(
                "logical PR 1's explicit-adjustment provider needs a non-empty adjustment set; "
                "known randomized assignment is added by the complete foundational API PR"
            )

    def prepare(self, data: Any) -> CausalData:
        """Validate and detach the analysis arrays from the caller's dataframe."""
        return CausalData.from_frame(
            data,
            outcome=self.outcome,
            treatment=self.treatment,
            covariates=self.adjustment,
            delta=self.missingness,
            weights=self.weights,
            weights_type=self.weights_type,
            weights_estimated=self.weights_estimated,
            id=self.cluster,
            strata=self.strata,
            family=self.outcome_family,
            treatment_kind=self.treatment_kind,
        )


@dataclass(frozen=True)
class ATE:
    """Average contrast of each treatment arm with a declared reference arm."""

    reference: Any = None
    name: str = field(default="ate", init=False)

    @property
    def definition(self) -> str:
        return "average treatment effect, E[Y^a] - E[Y^reference]"


@dataclass(frozen=True)
class BackdoorMeanContrast:
    """Observed-data functional identified by an explicit backdoor adjustment set."""

    outcome: str
    treatment: str
    adjustment: tuple[str, ...]
    reference: Any = None

    @property
    def expression(self) -> str:
        return "E_W[E(Y | A=a, W) - E(Y | A=reference, W)]"


@dataclass(frozen=True)
class ParameterKey:
    """Structured identity for a displayed parameter alias."""

    alias: str
    estimand: str
    treatment: Any
    reference: Any


@dataclass(frozen=True)
class ExplicitAdjustmentProvider:
    """Identify supported effects from analyst-declared adjustment variables."""

    name: str = "explicit-adjustment"

    def identify(self, study: CausalStudy, estimand: ATE) -> IdentifiedEffect:
        design = study.design
        if design.treatment_kind != "discrete":
            raise CapabilityError(
                "ATE() compares treatment arms, but this PointTreatment declares a continuous "
                "dose. Use a typed modified-treatment-policy estimand when it is available."
            )
        if design.strata:
            raise CapabilityError(
                "logical PR 1 does not expose stratified ATE parameters because their structured "
                "keys are part of the complete foundational API PR; remove strata or use the "
                "existing estimator API"
            )
        if estimand.reference is not None and estimand.reference not in study.data.treatment_levels:
            raise DataError(
                f"ATE reference {estimand.reference!r} is not a treatment level; "
                f"available: {list(study.data.treatment_levels)}"
            )
        functional = BackdoorMeanContrast(
            outcome=design.outcome,
            treatment=design.treatment,
            adjustment=tuple(design.adjustment),
            reference=estimand.reference,
        )
        return IdentifiedEffect(
            estimand=estimand,
            functional=functional,
            identification=TARGETS["ate"].identification,
            provider=self,
            _study=study,
        )


class CausalStudy:
    """Validated study data and design, before an estimand or method is chosen."""

    def __init__(self, data: Any, *, design: PointTreatment) -> None:
        self._design = design
        self._data = design.prepare(data)

    @property
    def design(self) -> PointTreatment:
        """The declared roles, read-only because the data was already prepared from them.

        ``PointTreatment`` is frozen, so its fields cannot move; what this closes is the
        *rebinding*.  A study whose ``design`` had been replaced would identify against one
        outcome column and adjustment set while holding data prepared from another, and
        :meth:`identify` is the only reader -- so nothing downstream would notice.
        """
        return self._design

    @property
    def data(self) -> CausalData:
        """Validated arrays detached from the caller's original dataframe."""
        return self._data

    def identify(
        self,
        estimand: ATE,
        *,
        provider: ExplicitAdjustmentProvider | None = None,
    ) -> IdentifiedEffect:
        """Identify an estimand without fitting nuisance models."""
        if not isinstance(estimand, ATE):
            raise CapabilityError(
                f"{type(estimand).__name__} is not in logical PR 1's evidenced catalog; "
                "the first slice supports ATE()"
            )
        return (provider or ExplicitAdjustmentProvider()).identify(self, estimand)


@dataclass(frozen=True)
class IdentifiedEffect:
    """An estimand bound to one observed-data functional and its assumptions."""

    estimand: ATE
    functional: BackdoorMeanContrast
    identification: Identification
    provider: ExplicitAdjustmentProvider
    _study: CausalStudy = field(repr=False, compare=False)

    def available_methods(self) -> tuple[MethodAvailability, ...]:
        """Return supported and known-unavailable methods with reasons."""
        return (
            MethodAvailability("tmle", True),
            MethodAvailability(
                "riesz_tmle",
                False,
                "the direct-Riesz engine and intervention-state representer are not implemented",
            ),
            MethodAvailability(
                "ep",
                False,
                "EP estimates a conditional contrast, not this scalar marginal ATE",
            ),
        )

    def summary(self) -> str:
        """Describe the causal question, identifying functional, and assumptions."""
        lines = [
            self.estimand.definition,
            f"identified by {self.provider.name}: {self.functional.expression}",
            f"adjustment set: {list(self.functional.adjustment)}",
            "assumptions:",
        ]
        lines.extend(f"  - {assumption}" for assumption in self.identification.assumptions)
        return "\n".join(lines)

    def summary_lines(self) -> tuple[str, ...]:
        """Compact identification facts for the fitted result's summary."""
        return (
            f"causal estimand: {self.estimand.definition}",
            f"identification: {self.provider.name} with adjustment set "
            f"{list(self.functional.adjustment)}",
            "identification assumptions: " + "; ".join(self.identification.assumptions),
        )

    def estimate(
        self,
        method: str | TMLEMethod = "tmle",
        **overrides: Any,
    ) -> TMLEResult:
        """Estimate this effect through the existing analytic TMLE engine."""
        if isinstance(method, str):
            availability = {item.name: item for item in self.available_methods()}
            record = availability.get(method)
            if record is None:
                raise CapabilityError(
                    f"unknown estimation method {method!r}; available declarations: "
                    f"{list(availability)}"
                )
            if not record.available:
                raise CapabilityError(f"method {method!r} cannot estimate ATE(): {record.reason}")
            normalized = TMLEMethod().with_overrides(**overrides)
        elif isinstance(method, TMLEMethod):
            normalized = method.with_overrides(**overrides)
        else:
            raise TypeError("method must be 'tmle' or a TMLEMethod")

        estimator = TMLE(
            estimands=("ate",),
            reference=self.estimand.reference,
            **normalized.estimator_kwargs(),
        )
        raw = estimator.fit(self._study.data).single()
        keys = self._parameter_keys(raw)
        return replace(
            raw,
            identified_effect=self,
            method=normalized,
            parameter_keys=keys,
        )

    def _parameter_keys(self, result: TMLEResult) -> dict[str, ParameterKey]:
        data = result.data
        reference_code = result.config.reference_arm
        reference = data.arm_label(reference_code)
        keys: dict[str, ParameterKey] = {}
        for code in data.arm_codes:
            if code == reference_code:
                continue
            treatment = data.arm_label(code)
            alias = (
                parameter_name("ate")
                if data.is_binary_treatment
                else parameter_name("ate", arm=treatment, versus=reference)
            )
            keys[alias] = ParameterKey(alias, "ate", treatment, reference)
        if set(keys) != set(result.estimates):
            raise CleverlyError(
                "structured ATE keys disagree with the estimator output: "
                f"{list(keys)} != {list(result.estimates)}"
            )
        return keys
