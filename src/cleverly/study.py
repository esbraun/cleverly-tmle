"""Causal-question-first public workflow and typed scientific contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

import narwhals as nw

from ._typing import Family
from .data import CausalData
from .estimators import CTMLE, DRTMLE, TMLE, TMLEResult
from .exceptions import CapabilityError, CleverlyError, DataError
from .inference.multiplier import SimultaneousBands, simultaneous_bands
from .interventions import Incremental, IPSISet, RegimeSet, Shift, ShiftSet
from .longitudinal import LTMLE, LongitudinalData, LongitudinalResult
from .methods import (
    CollaborativeTMLEMethod,
    DRTMLEMethod,
    EstimationMethod,
    MethodAvailability,
    Runtime,
    TMLEMethod,
)
from .msm import MSM, MSMSet
from .targets import TARGETS
from .targets.base import Identification, parameter_name
from .utils.frames import as_frame

__all__ = [
    "ATC",
    "ATE",
    "ATT",
    "BackdoorMeanContrast",
    "CausalResult",
    "CausalStudy",
    "ControlledDirectEffect",
    "CounterfactualMean",
    "Estimand",
    "ExplicitAdjustmentProvider",
    "IdentificationProvider",
    "IdentifiedEffect",
    "IncrementalEffect",
    "IncrementalMean",
    "LongitudinalTreatment",
    "MSMProjection",
    "ModifiedTreatmentPolicy",
    "ModifiedTreatmentPolicyEffect",
    "NaturalCourseMean",
    "OddsRatio",
    "ParameterKey",
    "PointTreatment",
    "PopulationAttributableFraction",
    "PopulationAttributableRisk",
    "RegimeContrast",
    "RegimeMean",
    "RiskRatio",
]


@runtime_checkable
class Estimand(Protocol):
    """Small public contract implemented by every typed causal question."""

    name: str

    @property
    def definition(self) -> str: ...


@runtime_checkable
class CausalResult(Protocol):
    """Operations shared by every fitted scalar causal-result family."""

    estimates: Mapping[str, Any]
    identified_effect: Any
    method: Any
    parameter_keys: Mapping[str, Any]
    provenance: Any

    @property
    def estimate(self) -> Any: ...

    def psi(self, name: str | None = None) -> float: ...

    @property
    def influence_curves(self) -> Mapping[str, Any]: ...

    def covariance(self, names: Sequence[str] | None = None) -> Any: ...

    def contrast(self, function: Any, names: Sequence[str], **kwargs: Any) -> Any: ...

    def summary(self) -> str: ...

    def to_frame(self) -> Any: ...

    def save(self, path: Any) -> Any: ...


@runtime_checkable
class IdentificationProvider(Protocol):
    """A provider that turns a causal estimand into an observed-data functional."""

    @property
    def name(self) -> str: ...

    def identify(self, study: CausalStudy, estimand: Any) -> IdentifiedEffect: ...


@dataclass(frozen=True)
class PointTreatment:
    """Column roles and design declarations for a point-treatment study."""

    outcome: str
    treatment: str
    adjustment: Sequence[str] = field(default_factory=tuple)
    randomized: bool = False
    missingness: str | None = None
    intermediate: str | None = None
    weights: str | None = None
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
        if not self.adjustment and not self.randomized:
            raise DataError(
                "declare a non-empty adjustment set or randomized=True; an empty set is "
                "an identification claim, not a nuisance-model default"
            )

    def prepare(self, data: Any) -> CausalData:
        if isinstance(data, CausalData):
            return data
        covariates = self.adjustment
        if self.randomized and not covariates:
            frame = as_frame(data)
            intercept = "__cleverly_randomized_intercept__"
            if intercept in frame.columns:
                raise DataError(
                    f"reserved internal column {intercept!r} is already present; rename it"
                )
            data = frame.with_columns(nw.lit(0.0).alias(intercept)).to_native()
            covariates = (intercept,)
        return CausalData.from_frame(
            data,
            outcome=self.outcome,
            treatment=self.treatment,
            covariates=covariates,
            delta=self.missingness,
            intermediate=self.intermediate,
            weights=self.weights,
            weights_type=self.weights_type,
            weights_estimated=self.weights_estimated,
            id=self.cluster,
            strata=self.strata,
            family=self.outcome_family,
            treatment_kind=self.treatment_kind,
        )


@dataclass(frozen=True)
class LongitudinalTreatment:
    """Time-ordered roles for sequential treatment, censoring, and outcomes."""

    outcome: str | Sequence[str] | Mapping[str, Sequence[str]]
    treatment: Sequence[str]
    baseline: Sequence[str]
    time_varying: Sequence[Sequence[str]] | None = None
    censoring: Sequence[str] | None = None
    cluster: str | None = None
    weights: str | None = None
    weights_type: Literal["probability"] = "probability"
    weights_estimated: bool = False
    outcome_family: Family = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "treatment", tuple(self.treatment))
        object.__setattr__(self, "baseline", tuple(self.baseline))
        if self.time_varying is not None:
            object.__setattr__(
                self, "time_varying", tuple(tuple(block) for block in self.time_varying)
            )
        if self.censoring is not None:
            object.__setattr__(self, "censoring", tuple(self.censoring))
        if not self.treatment:
            raise DataError("a longitudinal design needs at least one treatment node")

    def prepare(self, data: Any) -> LongitudinalData:
        if isinstance(data, LongitudinalData):
            return data
        return LongitudinalData.from_frame(
            data,
            outcome=self.outcome,
            treatment=self.treatment,
            baseline=self.baseline,
            time_varying=self.time_varying,
            censoring=self.censoring,
            id=self.cluster,
            weights=self.weights,
            weights_type=self.weights_type,
            weights_estimated=self.weights_estimated,
            family=self.outcome_family,
        )


@dataclass(frozen=True)
class ATE:
    reference: Any = None
    name: str = field(default="ate", init=False)

    @property
    def definition(self) -> str:
        return "average treatment effect, E[Y^a] - E[Y^reference]"


@dataclass(frozen=True)
class ATT:
    reference: Any = None
    name: str = field(default="att", init=False)

    @property
    def definition(self) -> str:
        return "average treatment effect among units receiving each comparison arm"


@dataclass(frozen=True)
class ATC:
    reference: Any = None
    name: str = field(default="atc", init=False)

    @property
    def definition(self) -> str:
        return "average treatment effect among units receiving the reference arm"


@dataclass(frozen=True)
class CounterfactualMean:
    treatment: Any = None
    name: str = field(default="ey", init=False)

    @property
    def definition(self) -> str:
        return (
            "counterfactual mean under each treatment, E[Y^a]"
            if self.treatment is None
            else f"counterfactual mean under treatment {self.treatment!r}, E[Y^a]"
        )


@dataclass(frozen=True)
class NaturalCourseMean:
    name: str = field(default="ey_obs", init=False)
    definition: str = field(default="natural-course outcome mean, E[Y]", init=False)


@dataclass(frozen=True)
class PopulationAttributableRisk:
    reference: Any = None
    name: str = field(default="par", init=False)
    definition: str = field(default="E[Y] - E[Y^reference]", init=False)


@dataclass(frozen=True)
class PopulationAttributableFraction:
    reference: Any = None
    name: str = field(default="paf", init=False)
    definition: str = field(default="1 - E[Y^reference] / E[Y]", init=False)


@dataclass(frozen=True)
class RiskRatio:
    reference: Any = None
    name: str = field(default="rr", init=False)
    definition: str = field(default="counterfactual risk ratio", init=False)


@dataclass(frozen=True)
class OddsRatio:
    reference: Any = None
    name: str = field(default="or", init=False)
    definition: str = field(default="counterfactual odds ratio", init=False)


@dataclass(frozen=True)
class RegimeMean:
    regimens: Any
    reference: str | None = None
    horizons: Sequence[int] | None = None
    name: str = field(default="ey_regime", init=False)
    definition: str = field(default="mean outcome under each declared regime", init=False)


@dataclass(frozen=True)
class RegimeContrast:
    regimens: Any
    reference: str | None = None
    horizons: Sequence[int] | None = None
    name: str = field(default="ate_regime", init=False)
    definition: str = field(default="contrast of each regime against the reference", init=False)


@dataclass(frozen=True)
class ModifiedTreatmentPolicy:
    shifts: Sequence[Shift]
    reference: str | None = None
    name: str = field(default="ey_shift", init=False)
    definition: str = field(default="mean outcome under each modified treatment policy", init=False)


@dataclass(frozen=True)
class ModifiedTreatmentPolicyEffect:
    shifts: Sequence[Shift]
    reference: str | None = None
    name: str = field(default="ate_shift", init=False)
    definition: str = field(default="contrast of modified treatment policies", init=False)


@dataclass(frozen=True)
class IncrementalMean:
    interventions: Sequence[Incremental]
    reference: str | None = None
    name: str = field(default="ey_ipsi", init=False)
    definition: str = field(
        default="mean under each incremental propensity intervention", init=False
    )


@dataclass(frozen=True)
class IncrementalEffect:
    interventions: Sequence[Incremental]
    reference: str | None = None
    name: str = field(default="ate_ipsi", init=False)
    definition: str = field(default="contrast of incremental propensity interventions", init=False)


@dataclass(frozen=True)
class MSMProjection:
    model: MSM
    regimens: Any = None
    horizons: Sequence[int] | None = None
    name: str = field(default="msm", init=False)
    definition: str = field(
        default="projection of counterfactual means onto a working model", init=False
    )


@dataclass(frozen=True)
class ControlledDirectEffect:
    intermediate: float
    contrast: ATE | ATT | ATC | RiskRatio | OddsRatio = field(default_factory=ATE)
    name: str = field(default="controlled_direct_effect", init=False)

    def __post_init__(self) -> None:
        # Identification reads ``self.contrast.name`` before anything else, so a string here
        # would surface as ``AttributeError: 'str' object has no attribute 'name'`` from deep
        # inside the provider rather than as a refusal naming what to pass instead.
        if not isinstance(self.contrast, (ATE, ATT, ATC, RiskRatio, OddsRatio)):
            raise DataError(
                "ControlledDirectEffect(contrast=...) takes a typed arm contrast -- "
                f"ATE(), ATT(), ATC(), RiskRatio(), or OddsRatio() -- not {self.contrast!r}"
            )

    @property
    def definition(self) -> str:
        return f"controlled direct effect with the intermediate fixed at {self.intermediate}"


PointEstimand = (
    ATE
    | ATT
    | ATC
    | CounterfactualMean
    | NaturalCourseMean
    | PopulationAttributableRisk
    | PopulationAttributableFraction
    | RiskRatio
    | OddsRatio
    | RegimeMean
    | RegimeContrast
    | ModifiedTreatmentPolicy
    | ModifiedTreatmentPolicyEffect
    | IncrementalMean
    | IncrementalEffect
    | MSMProjection
    | ControlledDirectEffect
)

#: The typed object to reach for when a caller passes the legacy string spelling.  Only the
#: names that used to be accepted as ``TMLE(estimands=(...,))`` strings are listed; anything
#: else gets the generic pointer to the roster in ``cleverly.__all__``.
_STRING_ESTIMANDS: dict[str, str] = {
    "ate": "ATE()",
    "att": "ATT()",
    "atc": "ATC()",
    "ey": "CounterfactualMean()",
    "ey1": "CounterfactualMean(treatment=1)",
    "ey0": "CounterfactualMean(treatment=0)",
    "ey_obs": "NaturalCourseMean()",
    "par": "PopulationAttributableRisk()",
    "paf": "PopulationAttributableFraction()",
    "rr": "RiskRatio()",
    "or": "OddsRatio()",
    "ey_regime": "RegimeMean(regimens=...)",
    "ate_regime": "RegimeContrast(regimens=...)",
    "ey_shift": "ModifiedTreatmentPolicy(shifts=...)",
    "ate_shift": "ModifiedTreatmentPolicyEffect(shifts=...)",
    "ey_ipsi": "IncrementalMean(interventions=...)",
    "ate_ipsi": "IncrementalEffect(interventions=...)",
    "msm": "MSMProjection(model=...)",
}


def _narrow_bands(
    result: Any, retained: Mapping[str, Any], method: EstimationMethod
) -> SimultaneousBands | None:
    """Joint bands over the parameters this effect reports, not the ones the engine computed.

    A typed estimand can ask for a subset of what its engine produces: ``CounterfactualMean``
    naming an arm of a multi-arm fit, or a longitudinal target whose engine reports means and
    contrasts together. Filtering ``estimates`` alone left ``simultaneous`` behind, so the
    result carried a critical value quantifying a family it no longer contained and bands
    keyed to parameters it could not index -- and ``summary()`` printed them.

    The bands are recomputed rather than dropped, from the same influence curves and the same
    ``alpha``, draw count, multiplier distribution, seed and cluster structure the engine used.
    That makes them the bands the engine itself would have produced had it been asked for this
    family alone. Below two parameters there is no joint question left, and both engines
    already return ``None`` there rather than a one-parameter band.
    """
    bands = result.simultaneous
    if bands is None or len(retained) == len(result.estimates):
        return bands
    if len(retained) < 2:
        return None
    return simultaneous_bands(
        retained,
        alpha=bands.alpha,
        n_replicates=bands.n_replicates,
        kind=bands.kind,
        random_state=getattr(method, "runtime", Runtime()).random_state,
        cluster=result.data.cluster,
    )


@dataclass(frozen=True)
class BackdoorMeanContrast:
    """A fully normalized observed-data functional for an engine adapter."""

    outcome: Any
    treatment: Any
    adjustment: tuple[str, ...]
    target: str
    axis: str = "arm"
    reference: Any = None
    interventions: Any = ()
    horizons: tuple[int, ...] | None = None
    msm: MSM | None = None
    intermediate: float | None = None
    longitudinal: bool = False

    @property
    def expression(self) -> str:
        if self.longitudinal:
            return "sequential g-formula under the declared treatment regimen"
        if self.axis == "arm":
            return "E_W[E(Y | A=a, W)] and the declared smooth contrast"
        return f"identified {self.axis}-indexed plug-in functional with influence correction"


@dataclass(frozen=True)
class ParameterKey:
    """Structured identity for a stable user-facing parameter alias."""

    alias: str
    estimand: str
    axis: str = "arm"
    value: Any = None
    reference: Any = None
    stratum: tuple[Any, ...] | None = None
    regimen: str | None = None
    horizon: int | None = None
    cause: str | None = None
    term: str | None = None

    @property
    def treatment(self) -> Any:
        return self.value


_LONGITUDINAL_IDENTIFICATION = Identification(
    assumptions=(
        "consistency and no interference",
        "sequential exchangeability given the recorded history at every node",
        "sequential positivity for treatment and remaining under observation",
    ),
    required_nuisances=("sequential_outcome_regressions", "treatment_and_censoring_mechanisms"),
    dr_condition=(
        "the longitudinal remainder is a sum of stagewise products of outcome-regression "
        "and cumulative mechanism errors"
    ),
    references=("van der Laan & Gruber (2012)",),
)


@dataclass(frozen=True)
class ExplicitAdjustmentProvider:
    """Identify effects from analyst-declared adjustment or sequential histories."""

    name: str = "explicit-adjustment"

    def identify(self, study: CausalStudy, estimand: PointEstimand) -> IdentifiedEffect:
        design = study.design
        if isinstance(design, LongitudinalTreatment):
            return self._identify_longitudinal(study, estimand)
        return self._identify_point(study, estimand)

    def _identify_point(self, study: CausalStudy, estimand: PointEstimand) -> IdentifiedEffect:
        design = study.design
        assert isinstance(design, PointTreatment)
        data = study.data
        assert isinstance(data, CausalData)
        actual = estimand.contrast if isinstance(estimand, ControlledDirectEffect) else estimand
        target = actual.name
        if isinstance(actual, CounterfactualMean) and actual.treatment is not None:
            if actual.treatment not in data.treatment_levels:
                raise DataError(
                    f"treatment {actual.treatment!r} is not available; "
                    f"choose from {list(data.treatment_levels)}"
                )
            if data.is_binary_treatment:
                code = float(data.treatment_levels.index(actual.treatment))
                target = "ey1" if code == 1.0 else "ey0"
        if target not in TARGETS:
            raise CapabilityError(f"{type(estimand).__name__} is not an evidenced point estimand")
        axis = TARGETS[target].parameter_axis
        # The rule is about the parameter *axis*, not about a list of target names.  Naming
        # targets refused `msm` -- whose axis is `msm`, indexed by working-model term, and
        # which the density-ratio path fits happily on a dose (tests/unit/test_continuous_msm.py)
        # -- while telling the caller it was "arm-indexed", which it is not.  A regime is a
        # density over arms and an incremental intervention tilts an odds, so those two stay
        # refused; both statements now come from the axis rather than from a spelling.
        if design.treatment_kind == "continuous" and axis not in {"shift", "msm"}:
            raise CapabilityError(
                f"{type(actual).__name__} is indexed by {axis}, which a continuous dose does "
                "not provide; a continuous-dose design supports modified treatment policies "
                "and MSM projections"
            )
        if getattr(actual, "horizons", None) is not None:
            raise CapabilityError(
                f"{type(actual).__name__}(horizons=...) selects follow-up times from a "
                "sequential fit, and a point-treatment design has one time point; declare "
                "LongitudinalTreatment to ask that question"
            )
        if isinstance(actual, MSMProjection) and actual.regimens is not None:
            raise CapabilityError(
                "MSMProjection(regimens=...) projects over longitudinal regimen cells; on a "
                "point-treatment design the projection is over the declared arms or doses. "
                "Drop regimens=, or declare LongitudinalTreatment"
            )
        reference = getattr(actual, "reference", None)
        if (
            reference is not None
            and design.treatment_kind == "discrete"
            and axis == "arm"
            and reference not in data.treatment_levels
        ):
            raise DataError(
                f"reference {reference!r} is not a treatment level; "
                f"available: {list(data.treatment_levels)}"
            )
        if isinstance(estimand, ControlledDirectEffect):
            if design.intermediate is None:
                raise CapabilityError(
                    "ControlledDirectEffect needs PointTreatment(intermediate=...)"
                )
            if estimand.intermediate not in (0.0, 1.0):
                raise DataError("the evidenced binary intermediate must be fixed at 0 or 1")
        elif design.intermediate is not None:
            raise CapabilityError(
                "a design with intermediate= must identify ControlledDirectEffect explicitly"
            )

        interventions: tuple[Any, ...] = ()
        msm = None
        if isinstance(actual, (RegimeMean, RegimeContrast)):
            interventions = tuple(actual.regimens)
        elif isinstance(actual, (ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect)):
            interventions = tuple(actual.shifts)
        elif isinstance(actual, (IncrementalMean, IncrementalEffect)):
            interventions = tuple(actual.interventions)
        elif isinstance(actual, MSMProjection):
            msm = actual.model
        functional = BackdoorMeanContrast(
            outcome=design.outcome,
            treatment=design.treatment,
            adjustment=tuple(design.adjustment),
            target=target,
            axis=axis,
            reference=reference,
            interventions=interventions,
            msm=msm,
            intermediate=estimand.intermediate
            if isinstance(estimand, ControlledDirectEffect)
            else None,
        )
        return IdentifiedEffect(
            estimand=estimand,
            functional=functional,
            identification=TARGETS[target].identification,
            provider=self,
            _study=study,
        )

    def _identify_longitudinal(
        self, study: CausalStudy, estimand: PointEstimand
    ) -> IdentifiedEffect:
        design = study.design
        assert isinstance(design, LongitudinalTreatment)
        if not isinstance(estimand, (RegimeMean, RegimeContrast, MSMProjection)):
            raise CapabilityError(
                f"{type(estimand).__name__} is not a longitudinal regimen functional"
            )
        regimens = estimand.regimens
        if regimens is None:
            raise DataError("a longitudinal estimand must declare regimens")
        target = "msm_regimen" if isinstance(estimand, MSMProjection) else estimand.name
        functional = BackdoorMeanContrast(
            outcome=design.outcome,
            treatment=tuple(design.treatment),
            adjustment=tuple(design.baseline),
            target=target,
            axis="regimen" if target != "msm_regimen" else "msm",
            reference=getattr(estimand, "reference", None),
            interventions=regimens,
            horizons=None if estimand.horizons is None else tuple(estimand.horizons),
            msm=estimand.model if isinstance(estimand, MSMProjection) else None,
            longitudinal=True,
        )
        return IdentifiedEffect(
            estimand=estimand,
            functional=functional,
            identification=_LONGITUDINAL_IDENTIFICATION,
            provider=self,
            _study=study,
        )


class CausalStudy:
    """Validated study data and immutable design, before method selection."""

    def __init__(self, data: Any, *, design: PointTreatment | LongitudinalTreatment) -> None:
        self._design = design
        self._data = design.prepare(data)

    @property
    def design(self) -> PointTreatment | LongitudinalTreatment:
        return self._design

    @property
    def data(self) -> CausalData | LongitudinalData:
        return self._data

    def identify(
        self,
        estimand: PointEstimand,
        *,
        provider: IdentificationProvider | None = None,
    ) -> IdentifiedEffect:
        """Bind one typed estimand to an observed-data functional, before any fitting.

        The type check is here rather than in the provider because this is the one place
        every path passes through, and because the failure it replaces was unreadable: a
        provider dereferences ``estimand.name`` as its first act, so ``identify("ate")`` --
        the spelling every legacy ``TMLE(estimands=("ate",))`` call site used -- died with
        ``AttributeError: 'str' object has no attribute 'name'`` and no mention of estimands.
        A string is refused rather than resolved: the accepted design says one public
        question normalizes to one evidenced engine request, not that strings drive a second
        convenience path.
        """
        if not isinstance(estimand, PointEstimand):
            raise CapabilityError(
                f"{estimand!r} is not a typed causal estimand; pass one of the objects "
                f"exported by cleverly, such as ATE()"
                + (
                    f". For {estimand!r}, that is {_STRING_ESTIMANDS[estimand]}"
                    if isinstance(estimand, str) and estimand in _STRING_ESTIMANDS
                    else ""
                )
            )
        return (provider or ExplicitAdjustmentProvider()).identify(self, estimand)

    def estimate(
        self,
        estimand: PointEstimand,
        method: str | EstimationMethod = "tmle",
        **overrides: Any,
    ) -> TMLEResult | LongitudinalResult:
        """Concise workflow that still constructs and stores an IdentifiedEffect."""
        return self.identify(estimand).estimate(method=method, **overrides)


@dataclass(frozen=True)
class IdentifiedEffect:
    """An estimand bound to one observed-data functional and its assumptions."""

    estimand: PointEstimand
    functional: BackdoorMeanContrast
    identification: Identification
    provider: IdentificationProvider
    _study: CausalStudy | None = field(repr=False, compare=False, default=None)

    def available_methods(self) -> tuple[MethodAvailability, ...]:
        """Structured capability records, checked before any nuisance model is built.

        A controlled direct effect is refused here rather than mid-fit.  Its functional
        target is the *contrast's* name -- ``ate``, ``rr`` -- so a check that reads only the
        target declared both variants available for it, and both engines then refused once
        fitting had already started, which is the boundary
        ``docs/architecture-invariants.md`` exists to hold.
        """
        point = not self.functional.longitudinal
        target = self.functional.target
        blocker = (
            "a controlled direct effect fixes an intermediate variable, and neither "
            "variant's score is derived for that functional"
            if self.functional.intermediate is not None
            else None
        )
        variants = (
            blocker is None
            and point
            and self.functional.axis == "arm"
            and target in {"ate", "ey", "ey1", "ey0", "rr", "or"}
        )
        return (
            MethodAvailability("tmle", True),
            MethodAvailability(
                "collaborative_tmle",
                variants,
                None
                if variants
                else blocker or "no collaborative score is evidenced for this functional",
            ),
            MethodAvailability(
                "drtmle",
                variants,
                None
                if variants
                else blocker or "no reduced-dimension correction is evidenced for this functional",
            ),
            MethodAvailability(
                "riesz_tmle",
                False,
                "the direct-Riesz engine and intervention-state representer are not implemented",
            ),
            MethodAvailability(
                "ep",
                False,
                "EP requires a conditional-contrast estimand and its distinct result family",
            ),
        )

    def summary(self) -> str:
        assumptions = "\n".join(f"  - {item}" for item in self.identification.assumptions)
        return (
            f"{self.estimand.definition}\n"
            f"identified by {self.provider.name}: {self.functional.expression}\n"
            f"adjustment/history: {list(self.functional.adjustment)}\n"
            f"assumptions:\n{assumptions}"
        )

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"causal estimand: {self.estimand.definition}",
            f"identification: {self.provider.name}; {self.functional.expression}",
            "identification assumptions: " + "; ".join(self.identification.assumptions),
        )

    def estimate(
        self,
        method: str | EstimationMethod = "tmle",
        **overrides: Any,
    ) -> TMLEResult | LongitudinalResult:
        if self._study is None:
            raise CapabilityError(
                "this effect was restored as fitted metadata and is not bound to analysis data; "
                "construct a new CausalStudy to estimate it again"
            )
        normalized = self._method(method, overrides)
        if self.functional.longitudinal:
            return self._estimate_longitudinal(normalized)
        return self._estimate_point(normalized)

    def _method(
        self, method: str | EstimationMethod, overrides: dict[str, Any]
    ) -> EstimationMethod:
        if isinstance(method, str):
            availability = {item.name: item for item in self.available_methods()}
            record = availability.get(method)
            if record is None:
                raise CapabilityError(
                    f"unknown estimation method {method!r}; declarations: {list(availability)}"
                )
            if not record.available:
                raise CapabilityError(
                    f"method {method!r} cannot estimate {type(self.estimand).__name__}: "
                    f"{record.reason}"
                )
            constructors: dict[str, type[TMLEMethod]] = {
                "tmle": TMLEMethod,
                "collaborative_tmle": CollaborativeTMLEMethod,
                "drtmle": DRTMLEMethod,
            }
            return constructors[method]().with_overrides(**overrides)
        if not isinstance(method, EstimationMethod):
            raise TypeError("method must be a named preset or an EstimationMethod")
        declared = {item.name: item for item in self.available_methods()}.get(method.name)
        if declared is None or not declared.available:
            reason = "the method does not declare support" if declared is None else declared.reason
            raise CapabilityError(
                f"method {method.name!r} cannot estimate {type(self.estimand).__name__}: {reason}"
            )
        return method.with_overrides(**overrides)

    def _estimate_point(self, method: EstimationMethod) -> TMLEResult:
        assert self._study is not None and isinstance(self._study.data, CausalData)
        functional = self.functional
        kwargs = method.estimator_kwargs()
        kwargs.update({"estimands": (functional.target,), "reference": functional.reference})
        if functional.axis == "regime":
            kwargs["interventions"] = functional.interventions
        elif functional.axis == "shift":
            kwargs["shifts"] = functional.interventions
        elif functional.axis == "ipsi":
            kwargs["incremental"] = functional.interventions
        elif functional.axis == "msm":
            kwargs["msm"] = functional.msm

        if isinstance(method, CollaborativeTMLEMethod):
            estimator: Any = CTMLE(**kwargs)
        elif isinstance(method, DRTMLEMethod):
            estimator = DRTMLE(**kwargs)
        else:
            estimator = TMLE(**kwargs)
        fit_kwargs: dict[str, Any] = {}
        if isinstance(method, DRTMLEMethod) and method.treatment_probabilities is not None:
            fit_kwargs["treatment_probabilities"] = method.treatment_probabilities
        result_set = estimator.fit(self._study.data, **fit_kwargs)
        raw = (
            result_set[functional.intermediate]
            if functional.intermediate is not None
            else result_set.single()
        )
        raw = self._select_point_parameters(raw, method)
        return replace(
            raw,
            identified_effect=self,
            method=method,
            parameter_keys=self._point_parameter_keys(raw),
        )

    def _select_point_parameters(self, result: TMLEResult, method: EstimationMethod) -> TMLEResult:
        if (
            not isinstance(self.estimand, CounterfactualMean)
            or self.estimand.treatment is None
            or self.functional.target != "ey"
        ):
            return result
        data = result.data
        if self.estimand.treatment not in data.treatment_levels:
            raise DataError(
                f"treatment {self.estimand.treatment!r} is not available; "
                f"choose from {list(data.treatment_levels)}"
            )
        # Compose the alias from the arm label the estimator itself used, not from the value
        # the caller typed.  A numeric arm arrives as a float level, so ``treatment=1`` built
        # ``ey[1]`` while the fit reported ``ey[1.0]``; nothing matched, and the narrowing
        # returned an empty result rather than raising.  Resolving through ``arm_label`` is
        # the same "compose the known names forward" rule the parameter keys follow.
        code = float(data.treatment_levels.index(self.estimand.treatment))
        alias = parameter_name("ey", arm=data.arm_label(code))
        estimates = {
            name: value
            for name, value in result.estimates.items()
            if name == alias or name.startswith(f"{alias}[")
        }
        if not estimates:
            raise CleverlyError(
                f"selecting {self.estimand.treatment!r} kept none of the estimator's "
                f"parameters {list(result.estimates)}; expected {alias!r}"
            )
        return replace(
            result,
            estimates=estimates,
            simultaneous=_narrow_bands(result, estimates, method),
        )

    def _point_parameter_keys(self, result: TMLEResult) -> dict[str, ParameterKey]:
        data = result.data
        target = self.functional.target
        reference_code = result.config.reference_arm
        reference = data.arm_label(reference_code) if self.functional.axis == "arm" else None
        base: dict[str, ParameterKey] = {}
        if self.functional.axis == "arm":
            if target == "ey":
                for code in data.arm_codes:
                    value = data.arm_label(code)
                    alias = parameter_name("ey", arm=value)
                    base[alias] = ParameterKey(alias, target, value=value)
            elif target == "ey1":
                base[target] = ParameterKey(target, target, value=data.arm_label(1.0))
            elif target == "ey0":
                base[target] = ParameterKey(target, target, value=data.arm_label(0.0))
            elif target in {"ate", "att", "atc", "rr", "or"}:
                for code in data.arm_codes:
                    if code == reference_code:
                        continue
                    value = data.arm_label(code)
                    alias = (
                        target
                        if data.is_binary_treatment
                        else parameter_name(target, arm=value, versus=reference)
                    )
                    base[alias] = ParameterKey(alias, target, value=value, reference=reference)
            elif target in {"par", "paf"}:
                alias = (
                    target if data.is_binary_treatment else parameter_name(target, arm=reference)
                )
                base[alias] = ParameterKey(alias, target, value=reference)
            elif target == "ey_obs":
                base[target] = ParameterKey(target, target)
        else:
            state = {
                "regime": result.nuisance.regimes,
                "shift": result.nuisance.shifts,
                "ipsi": result.nuisance.incremental,
                "msm": result.nuisance.msm,
            }[self.functional.axis]
            if state is None:
                raise CleverlyError(f"the {self.functional.axis} fit lost its structured state")
            if self.functional.axis == "msm":
                assert isinstance(state, MSMSet)
                names = state.terms
            else:
                assert isinstance(state, (RegimeSet, ShiftSet, IPSISet))
                names = state.names
            reference_name = names[int(state.reference)] if hasattr(state, "reference") else None
            if target.startswith("ey_"):
                for name in names:
                    alias = parameter_name(target, arm=name)
                    base[alias] = ParameterKey(alias, target, self.functional.axis, value=name)
            elif target.startswith("ate_"):
                for name in names:
                    if name == reference_name:
                        continue
                    alias = parameter_name(target, arm=name, versus=reference_name)
                    base[alias] = ParameterKey(
                        alias, target, self.functional.axis, value=name, reference=reference_name
                    )
            else:
                for term in names:
                    alias = parameter_name("msm", arm=term)
                    base[alias] = ParameterKey(alias, target, "msm", term=term)

        keys = dict(base)
        if data.has_strata:
            for code in range(data.n_strata):
                stratum = data.strata_levels[code]
                label = data.stratum_label(code)
                for alias, key in base.items():
                    conditional = f"{alias}[{label}]"
                    keys[conditional] = replace(key, alias=conditional, stratum=stratum)
        selected = {name: keys[name] for name in result.estimates if name in keys}
        if set(selected) != set(result.estimates):
            raise CleverlyError(
                "structured parameter keys disagree with the point estimator output: "
                f"{list(selected)} != {list(result.estimates)}"
            )
        return selected

    def _estimate_longitudinal(self, method: EstimationMethod) -> LongitudinalResult:
        assert self._study is not None and isinstance(self._study.data, LongitudinalData)
        functional = self.functional
        estimator = LTMLE(
            functional.interventions,
            reference=functional.reference,
            horizons=functional.horizons,
            msm=functional.msm,
            **method.estimator_kwargs(longitudinal=True),
        )
        raw = estimator.fit(self._study.data)
        prefixes = {
            "ey_regime": ("ey_regimen[", "risk_regimen[", "cif_regimen["),
            "ate_regime": ("ate_regimen[", "risk_difference["),
            "msm_regimen": ("msm_regimen[",),
        }[functional.target]
        estimates = {
            name: value for name, value in raw.estimates.items() if name.startswith(prefixes)
        }
        index = None
        if raw.parameter_index is not None:
            index = {name: raw.parameter_index[name] for name in estimates}
        raw = replace(
            raw,
            estimates=estimates,
            parameter_index=index,
            simultaneous=_narrow_bands(raw, estimates, method),
        )
        return replace(
            raw,
            identified_effect=self,
            method=method,
            parameter_keys=self._longitudinal_parameter_keys(raw),
        )

    def _longitudinal_parameter_keys(self, result: LongitudinalResult) -> dict[str, ParameterKey]:
        """Structured keys read from the index the engine composed, not rebuilt beside it.

        ``LongitudinalResult.parameter_index`` already holds ``(label, cause, horizon)`` for
        every reported parameter, recorded where the name was built. This used to rebuild that
        grid as a cartesian product and zip it against the estimates positionally, so the
        length guard could catch a missing cell but never a mis-*pairing*: swapping the cause
        and horizon loops relabels every parameter of a competing-risks fit and raises nothing.
        The reference was recovered by looking for ``"ate_"`` in the alias, which is the
        display-name parsing the result contract forbids.

        The index's label is the regimen for a mean and the engine's composed contrast label
        for a contrast, so both spellings are composed forward from the declared regimens and
        matched -- never split apart again.
        """
        keys: dict[str, ParameterKey] = {}
        if result.parameter_index:
            reference = result.config.reference
            regimens: dict[str, tuple[str, str | None]] = {}
            for item in result.config.regimens:
                regimens[item.label] = (item.label, None)
                regimens[f"{item.label} vs {reference}"] = (item.label, reference)
            for alias in result.estimates:
                if alias not in result.parameter_index:
                    raise CleverlyError(
                        f"the longitudinal estimator reported {alias!r} with no entry in its "
                        "parameter index, so its regimen, cause, and horizon are unknown"
                    )
                label, cause, horizon = result.parameter_index[alias]
                if label not in regimens:
                    raise CleverlyError(
                        f"parameter {alias!r} is indexed by {label!r}, which is neither a "
                        f"declared regimen nor a contrast against {reference!r}"
                    )
                regimen, against = regimens[label]
                keys[alias] = ParameterKey(
                    alias,
                    self.functional.target,
                    "regimen",
                    value=regimen,
                    reference=against,
                    regimen=regimen,
                    cause=cause,
                    horizon=horizon,
                )
        elif result.msm is not None:
            # A working-model fit has no parameter index -- its parameters are indexed by
            # term -- so the names are composed forward from the declared terms and causes and
            # matched. The previous rule took the term at ``position % len(terms)``, which
            # assumed terms cycle fastest and dropped the cause entirely, so every parameter
            # of a competing-risks projection claimed to be about no cause in particular.
            composed = {
                (f"msm_regimen[{term}]" if cause is None else f"msm_regimen[{term}, {cause}]"): (
                    term,
                    cause,
                )
                for cause in (result.config.causes or (None,))
                for term in result.msm.terms
            }
            for name in result.estimates:
                if name not in composed:
                    raise CleverlyError(
                        f"working-model parameter {name!r} is not composed from the declared "
                        f"terms {list(result.msm.terms)}, so its term is unknown"
                    )
                term, cause = composed[name]
                keys[name] = ParameterKey(name, "msm_regimen", "msm", term=term, cause=cause)
        if set(keys) != set(result.estimates):
            raise CleverlyError(
                "structured parameter keys disagree with the longitudinal estimator output"
            )
        return keys
