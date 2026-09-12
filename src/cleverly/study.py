"""Causal-question-first public workflow and typed scientific contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import narwhals as nw

from ._typing import Family
from .data import CausalData
from .data.validate import RANDOMIZED_INTERCEPT
from .estimators import CTMLE, DRTMLE, TMLE, TMLEResult
from .exceptions import CapabilityError, CleverlyError, DataError, MethodConfigurationError
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
from .protocol import StudyProtocol
from .targets import TARGETS
from .targets.base import (
    INTERMEDIATE_MECHANISM,
    MISSINGNESS_MECHANISM,
    OUTCOME_REGRESSION,
    Identification,
    arm_alias,
    parameter_name,
    stratum_alias,
)
from .targets.builtin import (
    COMPLETE_OUTCOME_PREFIX,
    CONDITIONING_EVENT_PREFIX,
    CONSISTENCY_PREFIX,
    DR_DESIGN_CONDITIONAL_CLAUSE,
    DR_MECHANISM_HEAD,
    INTERMEDIATE_CAVEAT_PREFIX,
    MISSINGNESS_CAVEAT_PREFIX,
    NO_CONFOUNDING_PREFIX,
    REFERENCE_ONLY_POSITIVITY_PREFIX,
    TWO_ARM_POSITIVITY_PREFIX,
)
from .targets.population_intervention import (
    POPULATION_INTERVENTION_TARGETS,
    population_intervention_refusal,
)
from .utils.frames import as_frame
from .utils.records import _DefaultingUnpickle

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .assessment import AssessmentReport

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
    "StudyProtocol",
]


@runtime_checkable
class Estimand(Protocol):
    """Define the public contract for a typed causal question.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor. A protocol is implemented, not instantiated.

    Attributes
    ----------
    name : str
        Stable name used in parameter aliases and method dispatch.
    definition : str
    """

    name: str

    @property
    def definition(self) -> str:
        """Return the causal quantity in readable notation."""
        ...


@runtime_checkable
class CausalResult(Protocol):
    """Define operations shared by fitted causal-result families.

    Concrete point-treatment and longitudinal results provide this interface. Use the
    protocol when application code accepts either result family.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor. A protocol is implemented, not instantiated.

    Attributes
    ----------
    estimates : mapping of str to ParameterEstimate
        Estimates keyed by stable parameter alias.
    identified_effect : IdentifiedEffect
        Causal question and identifying assumptions used for the fit.
    method : EstimationMethod
        Method configuration used for estimation.
    parameter_keys : mapping of str to ParameterKey
        Structured identities for the reported aliases.
    provenance : Provenance
        Runtime and dependency information for the fit.
    assessment_cache : mapping
        Saved diagnostic and sensitivity artifacts.
    assessment_family : str
        Result-owned capability family declaration.
    estimate : ParameterEstimate
        The sole parameter estimate. A concrete result raises ``ValueError`` here when it
        reports more than one parameter, so ``isinstance`` does not presence-check it.
        Use :meth:`psi` or item access to name the parameter you want.

    Notes
    -----
    :func:`typing.runtime_checkable` gives ``isinstance`` a presence check over every
    member this class body declares. On Python 3.11 that check reads each member with
    :func:`hasattr`, which *calls* a property and swallows only ``AttributeError``. A
    member whose read can raise anything else therefore turns ``isinstance`` into an
    exception. ``estimate`` is such a member, so this class declares it under
    :data:`typing.TYPE_CHECKING`: a type checker still sees it, and the runtime check
    does not read it. Add a new member here only when reading it on every supported
    result cannot raise.
    """

    estimates: Mapping[str, Any]
    identified_effect: Any
    method: Any
    parameter_keys: Mapping[str, Any]
    provenance: Any
    assessment_cache: Mapping[str, Any]
    assessment_family: str

    if TYPE_CHECKING:  # pragma: no cover - kept out of the runtime presence check

        @property
        def estimate(self) -> Any:
            """Return the primary parameter estimate."""
            ...

    def psi(self, name: str | None = None) -> float:
        """Return one point estimate by alias.

        Parameters
        ----------
        name : str or None
            Parameter alias. Use ``None`` only when the result has one estimate.

        Returns
        -------
        float
            Point estimate on the parameter's reported scale.
        """
        ...

    @property
    def influence_curves(self) -> Mapping[str, Any]:
        """Return influence curves keyed by parameter alias."""
        ...

    def covariance(self, names: Sequence[str] | None = None) -> Any:
        """Estimate covariance for selected parameters.

        Parameters
        ----------
        names : sequence of str or None
            Parameter aliases in output order. ``None`` selects every estimate.

        Returns
        -------
        ndarray
            Covariance matrix in the requested order.
        """
        ...

    def contrast(self, function: Any, names: Sequence[str], **kwargs: Any) -> Any:
        """Apply the delta method to a smooth function of estimates.

        Parameters
        ----------
        function : callable
            Function from a parameter vector to one scalar.
        names : sequence of str
            Parameter aliases passed to ``function`` in order.
        **kwargs : Any
            Options forwarded to the concrete result implementation.

        Returns
        -------
        ParameterEstimate
            Transformed estimate and influence-curve inference.
        """
        ...

    def summary(self) -> str:
        """Return a printable fit summary."""
        ...

    def to_frame(self) -> Any:
        """Return estimates in the input dataframe backend."""
        ...

    def save(self, path: Any) -> Any:
        """Serialize the fitted result.

        Parameters
        ----------
        path : path-like
            Destination for the serialized result.

        Returns
        -------
        Path
            Resolved output path.
        """
        ...

    @property
    def diagnostics(self) -> Any:
        """Return the diagnostics facade for saved fit artifacts."""
        ...

    def validate(self) -> Any:
        """Run validation checks available without new arguments."""
        ...

    def assess(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> AssessmentReport:
        """Run the applicable post-fit assessment battery.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets; cheap E-value retargets run by default.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

        Returns
        -------
        AssessmentReport
            Validation, diagnostics, and sensitivity in one report.
        """
        ...

    @property
    def sensitivity(self) -> Any:
        """Return the sensitivity-analysis facade for this result."""
        ...

    @property
    def replayability(self) -> Any:
        """Report which analyses a restored result can replay."""
        ...


@runtime_checkable
class IdentificationProvider(Protocol):
    """Convert a causal estimand into an observed-data functional.

    Implement this protocol to add an identification strategy. The provider must return an
    :class:`IdentifiedEffect` with explicit assumptions and nuisance requirements.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor. A protocol is implemented, not instantiated.

    Attributes
    ----------
    name : str
    """

    @property
    def name(self) -> str:
        """Return the provider name used in reports."""
        ...

    def identify(self, study: CausalStudy, estimand: Any) -> IdentifiedEffect:
        """Identify one estimand for a study.

        Parameters
        ----------
        study : CausalStudy
            Validated observed data and design.
        estimand : Estimand
            Typed causal quantity to identify.

        Returns
        -------
        IdentifiedEffect
            Estimand bound to an observed-data functional and assumptions.
        """
        ...


@dataclass(frozen=True)
class PointTreatment:
    """Declare column roles for a point-treatment study.

    Parameters
    ----------
    outcome : str
        Outcome column.
    treatment : str
        Discrete treatment or continuous dose column.
    adjustment : sequence of str
        Baseline adjustment columns. Supply at least one unless ``randomized=True``.
    randomized : bool
        Whether the treatment was randomized and an empty adjustment set is intentional.
    missingness : str or None
        Outcome-observation indicator. One means observed.
    intermediate : str or None
        Binary intermediate column for a controlled direct effect.
    weights : str or None
        Probability-weight column.
    weights_type : {"probability"}
        Interpretation of ``weights``.
    weights_estimated : bool
        Whether the supplied weights were estimated from these data.
    cluster : str or None
        Independent-cluster identifier used for variance estimation.
    strata : sequence of str
        Columns used to preserve strata during cross-fitting.
    treatment_kind : {"discrete", "continuous"}
        Treatment support used to select supported estimands.
    outcome_family : {"auto", "gaussian", "binomial"}
        Outcome family. ``"auto"`` infers it from observed values.

    See Also
    --------
    LongitudinalTreatment : The same declaration for time-varying treatment.
    CausalStudy : What a design is handed to along with the data.
    cleverly.datasets.make_linear_ate : A frame with the columns this example names.

    Examples
    --------
    >>> from cleverly import PointTreatment
    >>> design = PointTreatment(
    ...     outcome="Y", treatment="A", adjustment=("age", "baseline_score")
    ... )
    >>> design.treatment
    'A'

    A randomized trial declares that its empty adjustment set is deliberate:

    >>> PointTreatment(outcome="Y", treatment="A", randomized=True).adjustment
    ()
    """

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
        """Validate data and encode the declared column roles.

        Parameters
        ----------
        data : dataframe or CausalData
            Pandas, Polars, or prepared causal data.

        Returns
        -------
        CausalData
            Validated internal representation.
        """
        if isinstance(data, CausalData):
            self._check_prepared(data)
            return data
        covariates = self.adjustment
        if self.randomized and not covariates:
            frame = as_frame(data)
            if RANDOMIZED_INTERCEPT in frame.columns:
                raise DataError(
                    f"reserved internal column {RANDOMIZED_INTERCEPT!r} is already present; "
                    "rename it"
                )
            data = frame.with_columns(nw.lit(0.0).alias(RANDOMIZED_INTERCEPT)).to_native()
            covariates = (RANDOMIZED_INTERCEPT,)
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

    def _check_prepared(self, data: CausalData) -> None:
        """Reconcile an already-built container with the roles this design declares.

        A ``CausalData`` handed straight to ``CausalStudy`` used to be adopted as-is, so
        every field here was a claim nothing checked. The design is what
        ``IdentifiedEffect.functional`` records and what ``summary()`` prints, so a design
        naming ``adjustment=("Z1",)`` over a container fitted on ``W1..W3`` reported an
        adjustment set no estimate came from -- and, since this PR, persisted it.
        """
        expected: tuple[tuple[str, Any, Any], ...] = (
            ("outcome", self.outcome, data.outcome_name),
            ("treatment", self.treatment, data.treatment_name),
            ("missingness", self.missingness, data.delta_name),
            ("intermediate", self.intermediate, data.intermediate_name),
            ("cluster", self.cluster, data.cluster_name),
            ("weights", self.weights, data.weights_name),
            ("strata", tuple(self.strata), tuple(data.strata_names)),
            ("treatment_kind", self.treatment_kind, data.treatment_kind),
            # Not role names, but they change what the reported variance means, so a
            # disagreement here is as consequential as a mis-named column.
            ("weights_type", self.weights_type, data.weight_spec.kind),
            ("weights_estimated", self.weights_estimated, data.weight_spec.estimated),
        )
        for role, declared, held in expected:
            if declared != held:
                raise DataError(
                    f"the supplied CausalData was built with {role}={held!r}, but this design "
                    f"declares {role}={declared!r}; build the data from this design, or "
                    "correct the design"
                )
        # ``outcome_family`` is inferred when the design does not state one, so only a
        # stated disagreement is a disagreement.
        if self.outcome_family != "auto" and self.outcome_family != data.family:
            raise DataError(
                f"the supplied CausalData has family={data.family!r}, but this design "
                f"declares outcome_family={self.outcome_family!r}"
            )
        # ``covariate_names`` is post-encoding, so compare the columns the caller named:
        # a categorical adjustment variable arrives as several generated columns, and a
        # degenerate one may have been dropped entirely.
        generated = {name: item.column for item in data.encodings for name in item.generated}
        sources = {generated.get(name, name) for name in data.covariate_names}
        sources.update(data.dropped_covariates)
        declared_adjustment = set(self.adjustment) or {RANDOMIZED_INTERCEPT}
        if sources != declared_adjustment:
            raise DataError(
                f"the supplied CausalData adjusts for {sorted(sources)}, but this design "
                f"declares {sorted(declared_adjustment)}; an identification claim and the "
                "data it is claimed about have to be the same set"
            )


@dataclass(frozen=True)
class LongitudinalTreatment:
    """Declare time-ordered roles for a longitudinal study.

    Parameters
    ----------
    outcome : str, sequence of str, or mapping of str to sequence of str
        End-of-study outcome, survival event nodes, or competing-risk event nodes.
    treatment : sequence of str
        Treatment nodes in time order.
    baseline : sequence of str
        Covariates observed before the first treatment.
    time_varying : sequence of sequence of str or None
        Covariate blocks observed before each treatment node.
    censoring : sequence of str or None
        Observation indicators in time order. One means observed.
    cluster : str or None
        Independent-cluster identifier.
    weights : str or None
        Probability-weight column.
    weights_type : {"probability"}
        Interpretation of ``weights``.
    weights_estimated : bool
        Whether the supplied weights were estimated from these data.
    outcome_family : {"auto", "gaussian", "binomial"}
        Outcome family used by the sequential regressions.

    See Also
    --------
    PointTreatment : The same declaration for treatment given once.
    CausalStudy : What a design is handed to along with the data.
    cleverly.datasets.make_longitudinal : A frame with the nodes this example names.

    Examples
    --------
    Two treatment nodes, with the covariates measured before each one and an observation
    indicator per node:

    >>> from cleverly import LongitudinalTreatment
    >>> design = LongitudinalTreatment(
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> design.treatment
    ('A1', 'A2')

    ``time_varying`` is indexed by treatment node, so the first block is empty here: no
    covariate is measured between baseline and the first treatment.

    >>> design.time_varying
    ((), ('L2',))
    """

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
        """Validate data and construct its ordered longitudinal representation.

        Parameters
        ----------
        data : dataframe or LongitudinalData
            Pandas, Polars, or prepared longitudinal data.

        Returns
        -------
        LongitudinalData
            Validated time-ordered representation.
        """
        if isinstance(data, LongitudinalData):
            self._check_prepared(data)
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

    def _outcome_roles(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The event nodes and cause labels this outcome declaration implies.

        One field spells three designs: a name is an end-of-study outcome, a sequence is a
        survival process, and a mapping of cause to sequence is competing risks. The
        container records them as ``event_names`` plus ``cause_labels``, so composing them
        forward is what makes the three comparable without inspecting arrays.
        """
        if isinstance(self.outcome, str):
            return (), ()
        if isinstance(self.outcome, Mapping):
            causes = tuple(self.outcome)
            return tuple(node for cause in causes for node in self.outcome[cause]), causes
        return tuple(self.outcome), ()

    def _check_prepared(self, data: LongitudinalData) -> None:
        """Reconcile an already-built container with the roles this design declares.

        The point-treatment reasoning applies unchanged, and the node ordering makes it
        sharper: ``functional.adjustment`` records this design's baseline, and a sequential
        fit's history is not recoverable from the arrays afterwards.
        """
        events, causes = self._outcome_roles()
        empty: tuple[str, ...] = ()
        blocks = (
            tuple(empty for _ in self.treatment)
            if self.time_varying is None
            else tuple(tuple(block) for block in self.time_varying)
        )
        expected: tuple[tuple[str, Any, Any], ...] = (
            ("treatment", tuple(self.treatment), tuple(data.treatment_names)),
            ("baseline", tuple(self.baseline), tuple(data.baseline_names)),
            ("time_varying", blocks, tuple(data.time_varying_names)),
            ("censoring", tuple(self.censoring or ()), tuple(data.censoring_names)),
            ("cluster", self.cluster, data.cluster_name),
            ("weights", self.weights, data.weights_name),
            ("weights_type", self.weights_type, data.weight_spec.kind),
            ("weights_estimated", self.weights_estimated, data.weight_spec.estimated),
            ("outcome event nodes", events, tuple(data.event_names)),
            ("outcome causes", causes, tuple(data.cause_labels)),
        )
        for role, declared, held in expected:
            if declared != held:
                raise DataError(
                    f"the supplied LongitudinalData was built with {role}={held!r}, but this "
                    f"design declares {role}={declared!r}; build the data from this design, "
                    "or correct the design"
                )
        if not events and isinstance(self.outcome, str) and self.outcome != data.outcome_name:
            raise DataError(
                f"the supplied LongitudinalData was built with outcome={data.outcome_name!r}, "
                f"but this design declares outcome={self.outcome!r}"
            )
        if self.outcome_family != "auto" and self.outcome_family != data.family:
            raise DataError(
                f"the supplied LongitudinalData has family={data.family!r}, but this design "
                f"declares outcome_family={self.outcome_family!r}"
            )


@dataclass(frozen=True)
class ATE:
    """Compare population counterfactual means against one reference arm.

    Parameters
    ----------
    reference : Any or None
        Treatment level used as the reference. ``None`` uses the design default.

    See Also
    --------
    ATT : The same contrast restricted to the units in each comparison arm.
    ATC : The same contrast restricted to the units in the reference arm.
    RiskRatio : The same comparison on the ratio scale.
    CounterfactualMean : The arm means the contrast is taken between.

    Examples
    --------
    >>> from cleverly import ATE
    >>> estimand = ATE(reference=0)
    >>> estimand.reference
    0
    """

    reference: Any = None
    name: str = field(default="ate", init=False)

    @property
    def definition(self) -> str:
        """Return the population-average contrast definition."""
        return "average treatment effect, E[Y^a] - E[Y^reference]"


@dataclass(frozen=True)
class ATT:
    """Compare counterfactual means among units in each comparison arm.

    Parameters
    ----------
    reference : Any or None
        Treatment level used as the reference. ``None`` uses the design default.

    See Also
    --------
    ATE : The same contrast over the whole population.
    ATC : The same contrast restricted to the units in the reference arm.
    cleverly.ControlledDirectEffect : An arm contrast holding an intermediate fixed.

    Examples
    --------
    >>> from cleverly import ATT
    >>> estimand = ATT(reference=0)
    >>> estimand.reference
    0
    """

    reference: Any = None
    name: str = field(default="att", init=False)

    @property
    def definition(self) -> str:
        """Return the treated-population contrast definition."""
        return "average treatment effect among units receiving each comparison arm"


@dataclass(frozen=True)
class ATC:
    """Compare counterfactual means among units in the reference arm.

    Parameters
    ----------
    reference : Any or None
        Treatment level that defines the target population and comparison reference.

    See Also
    --------
    ATE : The same contrast over the whole population.
    ATT : The same contrast restricted to the units in each comparison arm.
    CounterfactualMean : The arm means the contrast is taken between.

    Examples
    --------
    >>> from cleverly import ATC
    >>> estimand = ATC(reference=0)
    >>> estimand.reference
    0
    """

    reference: Any = None
    name: str = field(default="atc", init=False)

    @property
    def definition(self) -> str:
        """Return the reference-population contrast definition."""
        return "average treatment effect among units receiving the reference arm"


@dataclass(frozen=True)
class CounterfactualMean:
    """Request counterfactual outcome means by treatment level.

    Parameters
    ----------
    treatment : Any or None
        One treatment level to retain. ``None`` reports every supported level.

    See Also
    --------
    ATE : The contrast between these means.
    RegimeMean : Means under declared regimens rather than fixed levels.
    ModifiedTreatmentPolicy : Means under shifts of a continuous dose.

    Examples
    --------
    >>> from cleverly import CounterfactualMean
    >>> estimand = CounterfactualMean(treatment=1)
    >>> estimand.treatment
    1
    """

    treatment: Any = None
    name: str = field(default="ey", init=False)

    @property
    def definition(self) -> str:
        """Return the counterfactual-mean definition."""
        return (
            "counterfactual mean under each treatment, E[Y^a]"
            if self.treatment is None
            else f"counterfactual mean under treatment {self.treatment!r}, E[Y^a]"
        )


@dataclass(frozen=True)
class NaturalCourseMean:
    """Request the observed-course outcome mean.

    With complete outcomes this is the empirical mean of ``Y``. It reads the observed
    law alone, so it estimates no nuisance and makes no exchangeability assumption.

    With a declared ``missingness=``, the same target is ``E[m(A, W)]``, where
    ``m(A, W) = E(Y | Delta = 1, A, W)``. The complete-case mean is a different
    parameter. This form estimates the outcome regression and the response mechanism,
    and still estimates no treatment mechanism. It needs missingness at random given
    ``(A, W)`` and response positivity. See
    :doc:`observed-data extensions </technical-reference/point-treatment-tmle>` for the
    supported compositions, which are deliberately narrow: one scalar ordinary TMLE,
    binary treatment, no cross-fitting, no repeats, and either a binary outcome or a
    bounded continuous outcome with declared ``q_bounds``. Every other composition
    refuses before any learner is fitted.
    """

    name: str = field(default="ey_obs", init=False)
    definition: str = field(default="natural-course outcome mean, E[Y]", init=False)


@dataclass(frozen=True)
class PopulationAttributableRisk:
    """Request the natural-course risk minus a reference counterfactual risk.

    Parameters
    ----------
    reference : Any or None
        Treatment level representing removal of the exposure.
    """

    reference: Any = None
    name: str = field(default="par", init=False)
    definition: str = field(default="E[Y] - E[Y^reference]", init=False)


@dataclass(frozen=True)
class PopulationAttributableFraction:
    """Request the preventable fraction relative to the natural-course risk.

    Parameters
    ----------
    reference : Any or None
        Treatment level representing removal of the exposure.
    """

    reference: Any = None
    name: str = field(default="paf", init=False)
    definition: str = field(default="1 - E[Y^reference] / E[Y]", init=False)


@dataclass(frozen=True)
class RiskRatio:
    """Compare counterfactual risks on the ratio scale.

    Parameters
    ----------
    reference : Any or None
        Denominator treatment level. ``None`` uses the design default.

    See Also
    --------
    OddsRatio : The same comparison on the odds scale.
    ATE : The same comparison on the difference scale.
    cleverly.sensitivity.evalue.evalue : How much confounding would explain a ratio away.

    Examples
    --------
    >>> from cleverly import RiskRatio
    >>> estimand = RiskRatio(reference=0)
    >>> estimand.reference
    0
    """

    reference: Any = None
    name: str = field(default="rr", init=False)
    definition: str = field(default="counterfactual risk ratio", init=False)


@dataclass(frozen=True)
class OddsRatio:
    """Compare counterfactual odds for a binary outcome.

    Parameters
    ----------
    reference : Any or None
        Denominator treatment level. ``None`` uses the design default.
    """

    reference: Any = None
    name: str = field(default="or", init=False)
    definition: str = field(default="counterfactual odds ratio", init=False)


@dataclass(frozen=True)
class RegimeMean:
    """Request mean outcomes under declared treatment regimens.

    Parameters
    ----------
    regimens : Any
        Point-treatment interventions or longitudinal regimens.
    reference : str or None
        Regimen label used as the reference for related contrasts.
    horizons : sequence of int or None
        Follow-up times to report for a longitudinal outcome.
    """

    regimens: Any
    reference: str | None = None
    horizons: Sequence[int] | None = None
    name: str = field(default="ey_regime", init=False)
    definition: str = field(default="mean outcome under each declared regime", init=False)


@dataclass(frozen=True)
class RegimeContrast:
    """Compare each declared regimen against one reference regimen.

    Parameters
    ----------
    regimens : Any
        Point-treatment interventions or longitudinal regimens.
    reference : str or None
        Reference regimen label. ``None`` uses the first regimen.
    horizons : sequence of int or None
        Follow-up times to report for a longitudinal outcome.

    See Also
    --------
    RegimeMean : The regimen means this contrast is taken between.
    ATE : The same comparison when treatment is given once.
    cleverly.LongitudinalTreatment : The design a regimen is declared against.

    Examples
    --------
    Two static longitudinal regimens, compared against never treating:

    >>> from cleverly import RegimeContrast
    >>> estimand = RegimeContrast(
    ...     regimens={"always": 1, "never": 0}, reference="never"
    ... )
    >>> estimand.reference
    'never'
    """

    regimens: Any
    reference: str | None = None
    horizons: Sequence[int] | None = None
    name: str = field(default="ate_regime", init=False)
    definition: str = field(default="contrast of each regime against the reference", init=False)


@dataclass(frozen=True)
class ModifiedTreatmentPolicy:
    """Request mean outcomes under continuous-dose shift policies.

    Parameters
    ----------
    shifts : sequence of Shift
        Named dose shifts to evaluate.
    reference : str or None
        Shift label retained as the comparison reference.

    See Also
    --------
    ModifiedTreatmentPolicyEffect : The contrast between these means.
    cleverly.interventions.Shift : The dose shift a policy is built from.
    cleverly.interventions.check_shift_support : Whether the shifted dose is supported.

    Examples
    --------
    Raise every dose by one unit without imposing a cap:

    >>> from cleverly import ModifiedTreatmentPolicy
    >>> from cleverly.interventions import Shift
    >>> estimand = ModifiedTreatmentPolicy(shifts=[Shift(delta=1.0, cap=None, name="up1")])
    >>> estimand.shifts[0].cap is None
    True
    """

    shifts: Sequence[Shift]
    reference: str | None = None
    name: str = field(default="ey_shift", init=False)
    definition: str = field(default="mean outcome under each modified treatment policy", init=False)


@dataclass(frozen=True)
class ModifiedTreatmentPolicyEffect:
    """Compare continuous-dose shift policies.

    Parameters
    ----------
    shifts : sequence of Shift
        Named dose shifts to compare.
    reference : str or None
        Reference shift label. ``None`` uses the first shift.
    """

    shifts: Sequence[Shift]
    reference: str | None = None
    name: str = field(default="ate_shift", init=False)
    definition: str = field(default="contrast of modified treatment policies", init=False)


@dataclass(frozen=True)
class IncrementalMean:
    """Request means under incremental propensity-score interventions.

    Parameters
    ----------
    interventions : sequence of Incremental
        Odds multipliers to evaluate.
    reference : str or None
        Intervention label retained as the comparison reference.
    """

    interventions: Sequence[Incremental]
    reference: str | None = None
    name: str = field(default="ey_ipsi", init=False)
    definition: str = field(
        default="mean under each incremental propensity intervention", init=False
    )


@dataclass(frozen=True)
class IncrementalEffect:
    """Compare incremental propensity-score interventions.

    Parameters
    ----------
    interventions : sequence of Incremental
        Odds multipliers to compare.
    reference : str or None
        Reference intervention label. ``None`` uses the first intervention.

    See Also
    --------
    IncrementalMean : The intervention means this contrast is taken between.
    cleverly.interventions.Incremental : The odds multiplier an intervention applies.
    ATE : The same comparison for a treatment set to a fixed level.

    Examples
    --------
    Double the odds of treatment, and compare that against halving them:

    >>> from cleverly import IncrementalEffect
    >>> from cleverly.interventions import Incremental
    >>> estimand = IncrementalEffect(
    ...     interventions=[
    ...         Incremental(delta=0.5, name="halved"),
    ...         Incremental(delta=2.0, name="doubled"),
    ...     ],
    ...     reference="halved",
    ... )
    >>> estimand.reference
    'halved'
    """

    interventions: Sequence[Incremental]
    reference: str | None = None
    name: str = field(default="ate_ipsi", init=False)
    definition: str = field(default="contrast of incremental propensity interventions", init=False)


@dataclass(frozen=True)
class MSMProjection:
    """Project counterfactual means onto a marginal structural model.

    Parameters
    ----------
    model : MSM
        Working model that defines the projection terms and weights.
    regimens : Any or None
        Longitudinal regimen cells. Leave unset for a point-treatment projection.
    horizons : sequence of int or None
        Follow-up times included in a longitudinal projection.
    """

    model: MSM
    regimens: Any = None
    horizons: Sequence[int] | None = None
    name: str = field(default="msm", init=False)
    definition: str = field(
        default="projection of counterfactual means onto a working model", init=False
    )


@dataclass(frozen=True)
class ControlledDirectEffect:
    """Compare treatments while fixing a binary intermediate variable.

    Parameters
    ----------
    intermediate : float
        Fixed intermediate level. The evidenced implementation accepts zero or one.
    contrast : ATE, ATT, ATC, RiskRatio, or OddsRatio
        Arm contrast evaluated at the fixed intermediate level.

    See Also
    --------
    ATE : The same contrast with nothing held fixed.
    cleverly.PointTreatment : Where the ``intermediate`` column is declared.
    cleverly.datasets.make_cde : A process with a known controlled direct effect.

    Examples
    --------
    >>> from cleverly import ATE, ControlledDirectEffect
    >>> estimand = ControlledDirectEffect(intermediate=0.0, contrast=ATE())
    >>> estimand.intermediate
    0.0
    """

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
        """Return the controlled-direct-effect definition."""
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


def _narrow_bootstrap(bootstrap: Any, retained: Mapping[str, Any]) -> Any:
    """Bootstrap draws for the parameters this effect reports, and no others.

    Unlike a joint band, per-parameter draws stay correct under narrowing -- each estimand's
    resampling distribution is its own. What they must not do is outlive their parameter: a
    result narrowed to one arm would still hand out ``bootstrap.draws`` for the arms it
    dropped, keyed by names it cannot itself index.
    """
    if bootstrap is None:
        return None
    return replace(
        bootstrap,
        draws={name: values for name, values in bootstrap.draws.items() if name in retained},
    )


@dataclass(frozen=True)
class BackdoorMeanContrast(_DefaultingUnpickle):
    """Store a normalized observed-data functional for an estimator adapter.

    Users normally receive this object through :meth:`CausalStudy.identify`.

    Parameters
    ----------
    outcome : Any
        Outcome column declaration.
    treatment : Any
        Treatment column declaration.
    adjustment : tuple of str
        Adjustment columns or baseline history.
    target : str
        Registered engine target.
    axis : str
        Parameter axis, such as ``"arm"``, ``"regimen"``, or ``"shift"``.
    reference : Any or None
        Reference level or label.
    interventions : Any
        Interventions, shifts, or regimens evaluated by the target.
    horizons : tuple of int or None
        Requested longitudinal follow-up times.
    msm : MSM or None
        Marginal structural model for a projection target.
    intermediate : float or None
        Fixed intermediate level for a controlled direct effect.
    longitudinal : bool
        Whether the functional uses sequential identification.
    missingness : str or None
        Outcome-observation indicator used by the observed-data regression.
    intermediate_name : str or None
        Intermediate column fixed by a controlled direct effect.
    treatment_levels : tuple
        Declared treatment levels, in the data container's stable order.
    treatment_value : Any or None
        One retained treatment level for a targeted counterfactual mean.
    schema_version : int
        Version of the design-bound identification record. Zero denotes a record
        restored from before this discriminator was persisted; retained metadata
        and identification distinguish static legacy from transitional records.
    """

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
    missingness: str | None = None
    intermediate_name: str | None = None
    treatment_levels: tuple[Any, ...] = ()
    treatment_value: Any = None
    schema_version: int = 1

    _PICKLE_BACKFILL: ClassVar[dict[str, Any]] = {"schema_version": 0}

    #: The fields the design-bound revision added, in the order they were added. A record
    #: written before all of them is a static legacy record, and one written before the
    #: trailing :attr:`_TRANSITIONAL_FIELDS` alone is a transitional record. The provenance
    #: matcher reconstructs both shapes from this tuple, and the tests that forge a
    #: functional derive their tampering matrix from it, so one edit here reaches the
    #: matcher and every refusal surface rather than five hand-written copies.
    _SCHEMA_1_FIELDS: ClassVar[tuple[str, ...]] = (
        "missingness",
        "intermediate_name",
        "treatment_levels",
        "treatment_value",
        "schema_version",
    )

    #: The trailing subset of :attr:`_SCHEMA_1_FIELDS` that a transitional record lacks.
    _TRANSITIONAL_FIELDS: ClassVar[tuple[str, ...]] = ("treatment_value", "schema_version")

    @classmethod
    def _restored_without(
        cls, record: BackdoorMeanContrast, dropped: tuple[str, ...]
    ) -> BackdoorMeanContrast:
        """Return ``record`` as a pickle written before ``dropped`` existed restores it.

        The state goes through ``__setstate__`` rather than through a written-out fill, so
        the shape compared below is the shape an old pickle actually restores to. Naming
        the fills here instead would state the restore rule a second time and let the two
        copies disagree.

        Parameters
        ----------
        record : BackdoorMeanContrast
            A reconstructed current record.
        dropped : tuple of str
            Field names the older pickle did not carry.

        Returns
        -------
        BackdoorMeanContrast
            The record with those fields filled the way :meth:`__setstate__` fills them.

        Raises
        ------
        KeyError
            When a name in ``dropped`` is not a field of this class. The tuples that name
            them are a second list of field names, so a rename leaves one of them naming a
            field that no longer exists. Dropping such a name removes nothing, and the
            comparison then accepts a record of the current shape as a legacy one.
        """
        absent = sorted(set(dropped) - set(record.__dict__))
        if absent:
            raise KeyError(
                f"{cls.__name__} carries no such field: {', '.join(absent)}; a renamed "
                "field needs its entry in _SCHEMA_1_FIELDS renamed with it, or the legacy "
                "shape below is the current shape"
            )
        restored = object.__new__(cls)
        restored.__setstate__(
            {name: value for name, value in record.__dict__.items() if name not in dropped}
        )
        return restored

    @property
    def reference_arm(self) -> Any:
        """Return the arm this functional's contrasts are taken against.

        One implementation, because the printed expression and the printed positivity
        statement both resolve it and a reader who saw them disagree could not tell which
        arm the fit used. ``"reference"`` stands in only for a restored record that
        predates :attr:`treatment_levels`, which is the one case with nothing to name.
        """
        if self.reference is not None:
            return self.reference
        if self.treatment_levels:
            return self.treatment_levels[0]
        return "reference"

    @property
    def selected_arm(self) -> Any:
        """Return the one arm an evaluated-mean functional reports, or ``None`` for all.

        ``ey1`` and ``ey0`` name an arm through the target rather than through
        ``treatment_value``, which is why the rule is written here rather than at each of
        the three places that used to repeat it.
        """
        if self.treatment_value is not None:
            return self.treatment_value
        if self.target in {"ey1", "ey0"} and self.treatment_levels:
            return self.treatment_levels[1 if self.target == "ey1" else 0]
        return None

    @property
    def expression(self) -> str:
        """Return a readable expression for the identified functional.

        Raises
        ------
        CapabilityError
            When an arm-indexed target has no entry below. The alternative is to print
            some other estimand's functional into the record that ``summary()`` shows and
            ``save()`` persists, which is a wrong answer rather than a missing one.
        """
        if self.longitudinal:
            return "sequential g-formula under the declared treatment regimen"

        def regression(arm: Any = "a") -> str:
            conditions = [f"{self.treatment}={arm}"]
            if self.intermediate is not None:
                name = self.intermediate_name or "Z"
                conditions.append(f"{name}={self.intermediate:g}")
            if self.missingness is not None:
                conditions.append(f"{self.missingness}=1")
            conditions.append("W")
            return f"E({self.outcome} | {', '.join(conditions)})"

        support = f" for a in {list(self.treatment_levels)!r}" if self.treatment_levels else ""
        if self.axis == "arm":
            reference = self.reference_arm
            comparisons = tuple(level for level in self.treatment_levels if level != reference)
            comparison_domain = f" for a in {list(comparisons)!r}" if comparisons else ""

            def marginal_mean(arm: Any) -> str:
                return f"E_W[{regression(arm)}]"

            if self.target == "ey_obs":
                if self.missingness is not None:
                    return (
                        f"E_{{{self.treatment},W}}[E({self.outcome} | "
                        f"{self.missingness}=1, {self.treatment}, W)]"
                    )
                return f"E({self.outcome}) under the observed treatment course"
            if self.target == "att":
                return (
                    f"E_{{W | {self.treatment}=a}}[{regression('a')} - "
                    f"{regression(repr(reference))}]{comparison_domain}"
                )
            if self.target == "atc":
                return (
                    f"E_{{W | {self.treatment}={reference!r}}}[{regression('a')} - "
                    f"{regression(repr(reference))}]{comparison_domain}"
                )
            if self.target in {"par", "paf"}:
                counterfactual = marginal_mean(repr(reference))
                if self.target == "par":
                    return f"E({self.outcome}) - {counterfactual}"
                return f"1 - {counterfactual} / E({self.outcome})"
            if self.target in {"ey", "ey1", "ey0"}:
                selected = self.selected_arm
                if selected is not None:
                    return marginal_mean(repr(selected))
                return f"{marginal_mean('a')}{support}"
            if self.target == "ate":
                return f"{marginal_mean('a')} - {marginal_mean(repr(reference))}{comparison_domain}"
            if self.target == "rr":
                return f"{marginal_mean('a')} / {marginal_mean(repr(reference))}{comparison_domain}"
            if self.target == "or":
                mean_a = marginal_mean("a")
                mean_r = marginal_mean(repr(reference))
                return (
                    f"({mean_a} / (1 - {mean_a})) / ({mean_r} / (1 - {mean_r})){comparison_domain}"
                )
            raise CapabilityError(
                f"the arm-indexed target {self.target!r} has no identified expression; add "
                "one beside the estimands above rather than reporting the counterfactual "
                "mean's functional under another estimand's name"
            )
        return (
            f"identified {self.axis}-indexed plug-in functional of {regression()}{support} "
            "with influence correction"
        )


@dataclass(frozen=True)
class ParameterKey:
    """Describe the structured identity behind a parameter alias.

    Parameters
    ----------
    alias : str
        Stable key used in result mappings.
    estimand : str
        Registered estimand name.
    axis : str
        Dimension indexed by ``value``.
    value : Any or None
        Arm, intervention, or cell value.
    reference : Any or None
        Reference value used by a contrast.
    stratum : tuple or None
        Target stratum values.
    regimen : str or None
        Longitudinal regimen label.
    horizon : int or None
        Follow-up time.
    cause : str or None
        Competing-risk cause label.
    term : str or None
        Marginal structural model term.
    """

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
        """Return ``value`` as the backward-compatible treatment label."""
        return self.value


def _point_target(estimand: PointEstimand, data: CausalData) -> tuple[Any, str]:
    """Return the engine estimand and target selected by one typed point question."""
    actual = estimand.contrast if isinstance(estimand, ControlledDirectEffect) else estimand
    target = actual.name
    if (
        isinstance(actual, CounterfactualMean)
        and actual.treatment is not None
        and data.is_binary_treatment
        and actual.treatment in data.treatment_levels
    ):
        code = float(data.treatment_levels.index(actual.treatment))
        target = "ey1" if code == 1.0 else "ey0"
    return actual, target


@dataclass(frozen=True)
class _ArmScope:
    """The arms one point-treatment functional evaluates, resolved once.

    ``_point_positivity`` and ``_point_mechanism_scope`` were the same ladder twice --
    same unwrap of a controlled direct effect, same reference-arm default, same
    selected-arm rule, same order of cases -- and they had already drifted apart. They
    now share this resolution and differ only in the sentence each case writes.

    Parameters
    ----------
    treatment : str
        The treatment column, as the design names it.
    reference : Any
        The arm contrasts are taken against.
    selected : Any or None
        The one arm an evaluated-mean functional reports, or ``None`` for all arms.
    levels : tuple
        The supported treatment levels, empty for a continuous dose.
    continuous : bool
        Whether the design declared a continuous dose rather than arms.
    """

    treatment: str
    reference: Any
    selected: Any
    levels: tuple[Any, ...]
    continuous: bool


def _all_arm_statements(scope: _ArmScope) -> tuple[str, str]:
    """State support at every declared arm: the shared case for ``ate``, ``rr`` and ``or``."""
    return (
        f"positivity: P({scope.treatment} = a | W) > 0 almost surely for every "
        f"supported treatment level a in {list(scope.levels)!r}",
        f"for every supported treatment level b in {list(scope.levels)!r} "
        "almost surely over the target population",
    )


def _evaluated_arm_statements(scope: _ArmScope) -> tuple[str, str]:
    """State support at the one arm a counterfactual mean evaluates."""
    if scope.selected is None:
        return _all_arm_statements(scope)
    return (
        f"positivity for the evaluated arm: P({scope.treatment} = {scope.selected!r} | W) "
        "> 0 almost surely",
        f"at b = {scope.selected!r} almost surely over the target population",
    )


def _treated_population_statements(scope: _ArmScope) -> tuple[str, str]:
    """State the reference-arm support the effect on the treated needs."""
    return (
        f"positivity for the reference arm within each comparison population: "
        f"P({scope.treatment} = {scope.reference!r} | W) > 0 wherever "
        f"P({scope.treatment} = a | W) > 0, for every comparison arm a",
        f"for b in {{a, {scope.reference!r}}} wherever P({scope.treatment} = a | W) > 0",
    )


def _reference_population_statements(scope: _ArmScope) -> tuple[str, str]:
    """State the comparison-arm support the effect on the controls needs."""
    return (
        f"positivity for each comparison arm within the reference population: "
        f"P({scope.treatment} = a | W) > 0 wherever "
        f"P({scope.treatment} = {scope.reference!r} | W) > 0, for every comparison arm a",
        f"for b in {{a, {scope.reference!r}}} wherever "
        f"P({scope.treatment} = {scope.reference!r} | W) > 0",
    )


def _reference_intervention_statements(scope: _ArmScope) -> tuple[str, str]:
    """State the support an attributable effect needs at its reference intervention."""
    return (
        f"positivity for the reference intervention: P({scope.treatment} = "
        f"{scope.reference!r} | W) > 0 almost surely",
        _all_arm_statements(scope)[1],
    )


#: One row per target that can reach :func:`_arm_scope`: the treatment support that
#: target's functional needs, and the arm and population cells a fitted intermediate or
#: response mechanism must cover.  Every estimand's prose is its own auditable entry.  A
#: target reaching :func:`_arm_scope` without a row is refused rather than given the
#: counterfactual mean's sentence under its own name.
#:
#: ``ey_obs`` is the one registered arm target with no row, and it needs none: its record
#: keeps only the complete-outcome assumption, so the two-arm positivity sentence that
#: resolves the arms is dropped before the branch that would resolve them.  Its functional
#: asks nothing of the treatment mechanism, so it has no treatment support to state.  A
#: row that returned an empty sentence instead was dropped in silence by the caller, which
#: is the shape :func:`_arm_scope` now refuses.
#: ``test_the_arm_statement_rows_cover_every_target_that_can_reach_them`` compares these
#: keys with the registry and pins that exclusion.
_ARM_STATEMENTS: dict[str, Callable[[_ArmScope], tuple[str, str]]] = {
    "ate": _all_arm_statements,
    "rr": _all_arm_statements,
    "or": _all_arm_statements,
    "msm": _all_arm_statements,
    "ey": _evaluated_arm_statements,
    "ey1": _evaluated_arm_statements,
    "ey0": _evaluated_arm_statements,
    "att": _treated_population_statements,
    "atc": _reference_population_statements,
    "par": _reference_intervention_statements,
    "paf": _reference_intervention_statements,
}

_CONTINUOUS_DOSE_STATEMENTS = (
    "positivity for the evaluated doses: the conditional treatment density is "
    "positive wherever the target functional gives a dose positive weight",
    "wherever the target functional gives a dose positive weight",
)


def _arm_scope(
    design: PointTreatment, functional: BackdoorMeanContrast
) -> tuple[_ArmScope, str, str]:
    """Resolve the arms once, and return the two statements the target's row writes.

    Returns
    -------
    tuple
        The resolved scope, the positivity sentence, and the mechanism-scope phrase.
        The positivity sentence is empty for a target that needs none.

    Raises
    ------
    CapabilityError
        When the target has no row in :data:`_ARM_STATEMENTS`, or when its row states no
        treatment support. An empty sentence is not a statement that the target needs
        none: the caller cannot tell it from a row nobody wrote, and appending it would
        put a blank assumption in the record.
    """
    scope = _ArmScope(
        treatment=design.treatment,
        reference=functional.reference_arm,
        selected=functional.selected_arm,
        levels=functional.treatment_levels,
        continuous=design.treatment_kind == "continuous",
    )
    if scope.continuous:
        return (scope, *_CONTINUOUS_DOSE_STATEMENTS)
    row = _ARM_STATEMENTS.get(functional.target)
    if row is None:
        raise CapabilityError(
            f"the target {functional.target!r} declares the two-arm positivity assumption "
            "but states no design-bound support of its own; add its row beside the "
            "estimands in _ARM_STATEMENTS rather than printing another estimand's "
            "positivity statement under its name"
        )
    positivity, cells = row(scope)
    if not positivity:
        raise CapabilityError(
            f"the row for {functional.target!r} in _ARM_STATEMENTS states no treatment "
            "support; a target that reaches here declares the two-arm positivity "
            "assumption, so write the support its functional needs rather than an empty "
            "sentence the record would drop without saying so"
        )
    return (scope, positivity, cells)


def _conditioning_event_statement(design: PointTreatment, scope: _ArmScope, target: str) -> str:
    """State which conditioning event a conditional effect needs to be defined.

    Raises
    ------
    CapabilityError
        For a target that declares the conditioning-event assumption but names no
        population of its own.
    """
    if target == "att":
        return (
            f"the conditioning event has positive probability: "
            f"P({design.treatment} = a) > 0 for every comparison arm a"
        )
    if target == "atc":
        return (
            f"the reference conditioning event has positive probability: "
            f"P({design.treatment} = {scope.reference!r}) > 0"
        )
    raise CapabilityError(
        f"the target {target!r} declares a conditioning event but names no population "
        "it conditions on; state which arm's units carry the parameter rather than "
        "keeping another estimand's sentence"
    )


def _bound_dr_condition(
    base: Identification,
    design: PointTreatment,
    functional: BackdoorMeanContrast,
    *,
    direct: bool,
) -> str:
    """State the double-robustness condition this design leaves, in one sentence.

    A design decides which mechanisms enter the remainder, and the registry cannot: a
    declared response mechanism or a fixed intermediate makes the mechanism half a
    *product*, so reporting "either Qbar or g" would name a fit where g alone is
    sufficient when it is not.

    Only the opening clause is the design's to write.  A registry that states the
    mechanism half as a condition carries
    :data:`~cleverly.targets.builtin.DR_DESIGN_CONDITIONAL_CLAUSE`, and the design-bound
    statement resolves it, so the whole sentence goes.  Anything else a registered
    sentence says is a fact about the estimand: the extra term ``att`` and ``atc`` carry
    for the randomness of the conditioning event, and the note ``msm`` carries that beta
    is a projection whatever the working model does.  Those tails follow
    :data:`~cleverly.targets.builtin.DR_MECHANISM_HEAD`, so the head is replaced and the
    tail is kept.  Replacing the whole string deleted the target-specific half of every
    conditional and projection record under a ``missingness=`` design.  A registered
    sentence carrying neither is extended rather than rewritten, because a study that
    cannot find the head it would replace still must not delete the sentence.

    Parameters
    ----------
    base : Identification
        The registered theory for the target.
    design : PointTreatment
        The declared column roles and mechanisms.
    functional : BackdoorMeanContrast
        The design-bound functional the study identified.
    direct : bool
        Whether the design fixes an intermediate level.

    Returns
    -------
    str
        The condition under which this fit's estimate stays consistent.
    """
    target = functional.target
    missingness = functional.missingness
    if target == "ey_obs" and missingness is None:
        return "the complete-data empirical mean has no nuisance-model remainder"
    if target == "ey_obs":
        return (
            "consistent if either m(A, W) = E(Y | Delta = 1, A, W) is consistent "
            f"or P({missingness} = 1 | {design.treatment}, W) is consistent; no "
            "treatment mechanism enters the remainder"
        )
    if direct:
        mechanism = "g * q_z" + (" * pi" if missingness is not None else "")
        bound = (
            "consistent if either Qbar(A, z, W) is consistent or the full mechanism "
            f"product {mechanism} is consistent"
        )
    elif (
        missingness is not None
        and functional.axis in {"arm", "msm"}
        and target not in POPULATION_INTERVENTION_TARGETS
    ):
        bound = (
            "consistent if either Qbar(A, W) is consistent or the full mechanism product "
            f"g * P({missingness} = 1 | {design.treatment}, W) is consistent; a consistent "
            "treatment mechanism alone is not sufficient once outcomes are missing"
        )
    else:
        return base.dr_condition
    if base.dr_condition.endswith(DR_DESIGN_CONDITIONAL_CLAUSE):
        return bound
    kept = base.dr_condition.removeprefix(DR_MECHANISM_HEAD)
    if kept != base.dr_condition:
        return bound + kept
    return f"{base.dr_condition}. Under this design, {bound}"


def _point_identification(
    base: Identification,
    design: PointTreatment,
    functional: BackdoorMeanContrast,
) -> Identification:
    """Bind registered theory to the treatment levels and active design mechanisms.

    The target registry owns estimand-level theory.  A study owns the treatment support
    and whether outcome observation or an intermediate variable enters that theory.  A
    fresh record combines the two so identifying one study cannot mutate another target's
    shared registry entry.

    Every fact this needs is read off ``functional``, which
    :func:`_point_functional` has already resolved.  Reading the estimand and the data a
    second time gave two implementations of the reference arm, and they printed different
    answers into the same record.

    Parameters
    ----------
    base : Identification
        The registered theory for the target.
    design : PointTreatment
        The declared column roles and mechanisms.
    functional : BackdoorMeanContrast
        The design-bound functional the study identified.

    Returns
    -------
    Identification
        The registered theory, narrowed to this design.
    """
    target = functional.target
    axis = functional.axis
    missingness = functional.missingness
    declared_level = functional.intermediate
    direct = declared_level is not None
    level = float(declared_level) if declared_level is not None else None
    resolved: tuple[_ArmScope, str, str] | None = None

    def arms() -> tuple[_ArmScope, str, str]:
        """Resolve the arm scope at most once, and only for a target that needs it."""
        nonlocal resolved
        if resolved is None:
            resolved = _arm_scope(design, functional)
        return resolved

    assumptions: list[str] = []
    for item in base.assumptions:
        if item.startswith(MISSINGNESS_CAVEAT_PREFIX):
            # The registry writes this as a condition because it does not know whether a
            # response mechanism is declared.  Drop it only where the design-bound MAR and
            # response-positivity statements below replace it; otherwise the reader keeps
            # the registry's own paragraph rather than nothing.
            if design.missingness is None:
                assumptions.append(item)
            continue
        if item.startswith(INTERMEDIATE_CAVEAT_PREFIX):
            if design.intermediate is None:
                assumptions.append(item)
            continue
        if target == "ey_obs":
            if missingness is None and item.startswith(COMPLETE_OUTCOME_PREFIX):
                assumptions.append(item)
            continue
        if target in {"att", "atc"} and item.startswith(REFERENCE_ONLY_POSITIVITY_PREFIX):
            continue
        if item.startswith(CONDITIONING_EVENT_PREFIX):
            scope, _, _ = arms()
            assumptions.append(_conditioning_event_statement(design, scope, target))
            continue
        if direct and item.startswith(CONSISTENCY_PREFIX):
            assert level is not None
            assumptions.append(
                f"consistency: {design.outcome} = {design.outcome}(a, {level:g}) when "
                f"{design.treatment} = a and {design.intermediate} = {level:g}"
            )
            continue
        if direct and item.startswith(NO_CONFOUNDING_PREFIX):
            assumptions.extend(
                (
                    "no unmeasured treatment confounding: Y(a, z) is independent of A given W",
                    "no unmeasured intermediate confounding: Y(a, z) is independent of Z "
                    "given (A, W)",
                )
            )
            continue
        if item.startswith(TWO_ARM_POSITIVITY_PREFIX):
            _, positivity, cells = arms()
            assumptions.append(positivity)
            if direct:
                assert level is not None
                assumptions.append(
                    f"intermediate positivity at {design.intermediate} = {level:g}: "
                    f"P({design.intermediate} = {level:g} | {design.treatment} = b, W) > 0 "
                    f"{cells}"
                )
            continue
        assumptions.append(item)

    nuisances = (
        [OUTCOME_REGRESSION]
        if target == "ey_obs" and missingness is not None
        else []
        if target == "ey_obs"
        else list(base.required_nuisances)
    )
    if direct:
        nuisances.append(INTERMEDIATE_MECHANISM)
    if missingness is not None and target not in POPULATION_INTERVENTION_TARGETS:
        conditioning = (
            f"({design.treatment}, {design.intermediate}, W)"
            if direct
            else f"({design.treatment}, W)"
        )
        assumptions.append(
            f"missingness at random for {missingness}: {design.outcome} is independent "
            f"of {missingness} given {conditioning}"
        )
        if target == "ey_obs":
            assumptions.append(
                f"response positivity for {missingness}: P({missingness} = 1 | "
                f"{design.treatment}, W) > 0 almost surely"
            )
        elif axis == "shift":
            assumptions.append(
                f"response positivity for {missingness}: P({missingness} = 1 | "
                f"{design.treatment} = d(a, W), W) > 0 almost surely wherever a declared "
                "shift sends an observed dose"
            )
        elif direct or target in {"att", "atc", "ey", "ey1", "ey0"}:
            _, _, cells = arms()
            assumptions.append(
                f"response positivity for {missingness}: P({missingness} = 1 | "
                f"{design.treatment} = b, W) > 0 {cells}"
            )
        else:
            assumptions.append(
                f"response positivity for {missingness}: P({missingness} = 1 | "
                f"{design.treatment} = a, W) > 0 almost surely wherever the target "
                "functional evaluates arm a"
            )
        if direct:
            assumptions.append(
                f"observation-mechanism restriction: {missingness} is independent of "
                f"{design.intermediate} given ({design.treatment}, W), so the fitted "
                f"response mechanism excludes {design.intermediate}"
            )
        nuisances.append(MISSINGNESS_MECHANISM)
    # Definedness, stated for each estimand a boundary value leaves undefined.  The three
    # are one class of condition, and `cleverly.inference.delta` raises on each of them.
    if target == "paf":
        assumptions.append(
            f"the natural-course risk E({design.outcome}) is strictly positive, so the "
            "attributable fraction is defined"
        )
    if target == "rr":
        assumptions.append(
            f"both counterfactual risks are strictly positive: "
            f"E_W[E({design.outcome} | {design.treatment} = a, W)] > 0 for every "
            f"comparison arm a and at {design.treatment} = "
            f"{functional.reference_arm!r}, so the log risk ratio is defined"
        )
    if target == "or":
        assumptions.append(
            f"both counterfactual risks lie strictly inside (0, 1): "
            f"0 < E_W[E({design.outcome} | {design.treatment} = a, W)] < 1 for every "
            f"comparison arm a and at {design.treatment} = "
            f"{functional.reference_arm!r}, so the odds are finite"
        )
    if target == "ey_obs" and missingness is None:
        assumptions.append(
            "the natural-course mean is a functional of the observed law alone: neither "
            f"the adjustment set {list(design.adjustment)} nor a treatment mechanism "
            "enters it, which is why this record requires no nuisance estimate"
        )
    elif target == "ey_obs":
        assumptions.append(
            "the natural-course mean averages m(A, W) over the empirical joint law of "
            "(A, W); the response and outcome regressions enter, but no treatment "
            "mechanism or treatment-exchangeability assumption does"
        )
    references = base.references
    if target == "ey_obs" and missingness is not None:
        references = (*references, "Díaz, Carone & van der Laan (2016)")
    return Identification(
        assumptions=tuple(assumptions),
        required_nuisances=tuple(nuisances),
        dr_condition=_bound_dr_condition(base, design, functional, direct=direct),
        references=references,
    )


def _point_functional(
    design: PointTreatment,
    data: CausalData,
    estimand: PointEstimand,
) -> BackdoorMeanContrast:
    """Construct the canonical design-bound functional for one point estimand."""
    actual, target = _point_target(estimand, data)
    registered = TARGETS.get(target)
    if registered is None:
        raise CapabilityError(f"{type(estimand).__name__} is not an evidenced point estimand")
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
    return BackdoorMeanContrast(
        outcome=design.outcome,
        treatment=design.treatment,
        adjustment=tuple(design.adjustment),
        target=target,
        axis=registered.parameter_axis,
        reference=getattr(actual, "reference", None),
        interventions=interventions,
        msm=msm,
        intermediate=(
            estimand.intermediate if isinstance(estimand, ControlledDirectEffect) else None
        ),
        missingness=design.missingness,
        intermediate_name=design.intermediate,
        treatment_levels=(
            () if design.treatment_kind == "continuous" else tuple(data.treatment_levels)
        ),
        treatment_value=(actual.treatment if isinstance(actual, CounterfactualMean) else None),
    )


def _matches_registered_point_identification(
    actual: Identification,
    registered: Identification,
    identified: Any,
    data: CausalData,
    axis: str,
) -> bool:
    """Match a complete current record or an explicitly backfilled legacy record."""
    functional = getattr(identified, "functional", None)
    study = getattr(identified, "_study", None)
    design = getattr(study, "design", None)
    if type(functional) is not BackdoorMeanContrast or type(design) is not PointTreatment:
        return False
    # Both calls sit inside the guard.  A tampered estimand reaches the narrowing as well
    # as the functional -- a controlled direct effect whose intermediate is not a number
    # raises from ``float(...)`` -- and an exception out of a provenance matcher is a
    # different failure from the refusal it is asked for.
    try:
        expected_functional = _point_functional(design, data, identified.estimand)
        expected_registered = TARGETS[expected_functional.target]
        if registered != expected_registered.identification or axis != expected_functional.axis:
            return False
        expected = _point_identification(registered, design, expected_functional)
    except (AttributeError, CapabilityError, DataError, KeyError, TypeError, ValueError):
        return False
    schema = getattr(functional, "schema_version", None)
    if schema == 0:
        # A record written before these five fields existed carries the registry's generic
        # identification, which states neither missingness at random, nor response
        # positivity, nor an intermediate.  A design declaring missingness= or
        # intermediate= cannot have produced one: both compositions post-date the schema.
        # Accepting that pairing would let a replay run against assumptions the record
        # does not state, so the back-compat allowance stops at a design that declares
        # either mechanism.
        static_legacy = BackdoorMeanContrast._restored_without(
            expected_functional, BackdoorMeanContrast._SCHEMA_1_FIELDS
        )
        transitional = BackdoorMeanContrast._restored_without(
            expected_functional, BackdoorMeanContrast._TRANSITIONAL_FIELDS
        )
        static_admissible = design.missingness is None and design.intermediate is None
        return (static_admissible and functional == static_legacy and actual == registered) or (
            functional == transitional and actual == expected
        )
    if schema != 1 or functional != expected_functional:
        return False
    return actual == expected


_LONGITUDINAL_IDENTIFICATION = Identification(
    assumptions=(
        "consistency: each observed history equals the potential history under its "
        "realized regimen",
        "no interference: one unit's potential history does not depend on other units' regimens",
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
    """Identify effects from declared adjustment sets or sequential histories.

    Parameters
    ----------
    name : str
        Provider name recorded in the identified effect.
    """

    name: str = "explicit-adjustment"

    def identify(self, study: CausalStudy, estimand: PointEstimand) -> IdentifiedEffect:
        """Bind a supported estimand to the design's observed-data functional.

        Parameters
        ----------
        study : CausalStudy
            Validated observed data and design.
        estimand : Estimand
            Typed causal quantity.

        Returns
        -------
        IdentifiedEffect
            Functional, assumptions, and method availability for the question.
        """
        design = study.design
        if isinstance(design, LongitudinalTreatment):
            return self._identify_longitudinal(study, estimand)
        return self._identify_point(study, estimand)

    def _identify_point(self, study: CausalStudy, estimand: PointEstimand) -> IdentifiedEffect:
        design = study.design
        assert isinstance(design, PointTreatment)
        data = study.data
        assert isinstance(data, CausalData)
        actual, target = _point_target(estimand, data)
        if (
            isinstance(actual, CounterfactualMean)
            and actual.treatment is not None
            and actual.treatment not in data.treatment_levels
        ):
            raise DataError(
                f"treatment {actual.treatment!r} is not available; "
                f"choose from {list(data.treatment_levels)}"
            )
        if target not in TARGETS:
            raise CapabilityError(f"{type(estimand).__name__} is not an evidenced point estimand")
        # Keyed on the outcome being missing, not on the declaration.  The engine guard
        # this fronts -- ``TargetContext.observed_mean`` -- keys on the observation mask,
        # so a design that declares a response indicator which is identically one has no
        # missing outcome, E[Y] is exactly the empirical mean, and refusing it here would
        # refuse a fit the estimator performs.  docs/roadmap.md RM8 states the stop as
        # "refused when outcomes are missing", which is this condition.
        if data.has_missing_outcome and target in POPULATION_INTERVENTION_TARGETS:
            raise population_intervention_refusal(
                (target,),
                declaration="PointTreatment(missingness=...)",
                subject=type(actual).__name__,
            )
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

        functional = _point_functional(design, data, estimand)
        return IdentifiedEffect(
            estimand=estimand,
            functional=functional,
            identification=_point_identification(
                TARGETS[target].identification, design, functional
            ),
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
    """Validate observed data against an immutable causal-study design.

    Parameters
    ----------
    data : dataframe, CausalData, or LongitudinalData
        Observed study data. Pandas and Polars dataframes are supported.
    design : PointTreatment or LongitudinalTreatment
        Column roles and treatment-time structure.
    protocol : StudyProtocol or None
        Scientific study protocol. ``None`` records that no protocol was supplied.

    Attributes
    ----------
    data : CausalData or LongitudinalData
    design : PointTreatment or LongitudinalTreatment
    protocol : StudyProtocol or None

    See Also
    --------
    PointTreatment : The design declaration for treatment given once.
    LongitudinalTreatment : The design declaration for time-varying treatment.
    IdentifiedEffect : What :meth:`identify` returns.

    Examples
    --------
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=40, seed=1)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> study.identify(ATE()).estimand.name
    'ate'
    """

    # An artifact written before StudyProtocol has no instance value. The class fallback
    # makes its public property report absence after trusted unpickling.
    _protocol: StudyProtocol | None = None

    def __init__(
        self,
        data: Any,
        *,
        design: PointTreatment | LongitudinalTreatment,
        protocol: StudyProtocol | None = None,
    ) -> None:
        if protocol is not None and not isinstance(protocol, StudyProtocol):
            raise TypeError("protocol must be a StudyProtocol or None")
        self._design = design
        self._data = design.prepare(data)
        self._protocol = protocol

    @property
    def design(self) -> PointTreatment | LongitudinalTreatment:
        """Return the immutable study design."""
        return self._design

    @property
    def data(self) -> CausalData | LongitudinalData:
        """Return the validated internal data representation."""
        return self._data

    @property
    def protocol(self) -> StudyProtocol | None:
        """Return the scientific study protocol, if one was supplied."""
        return self._protocol

    def identify(
        self,
        estimand: PointEstimand,
        *,
        provider: IdentificationProvider | None = None,
    ) -> IdentifiedEffect:
        """Bind one typed estimand to an observed-data functional.

        Parameters
        ----------
        estimand : Estimand
            Typed causal quantity, such as ``ATE()``. String aliases are not accepted.
        provider : IdentificationProvider or None
            Identification strategy. ``None`` uses explicit adjustment or history.

        Returns
        -------
        IdentifiedEffect
            Question, functional, assumptions, and available methods before fitting.

        Raises
        ------
        CapabilityError
            If the estimand type or its composition with the design is unsupported.

        See Also
        --------
        IdentifiedEffect : The identified question returned by this method.
        CausalStudy.estimate : Identify and estimate in one call.

        Notes
        -----
        Pass a typed estimand such as ``ATE()``. String aliases are refused with a
        :class:`CapabilityError` so the failure identifies the required public object.

        Examples
        --------
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> frame, _ = make_linear_ate(n=40, seed=1)
        >>> study = CausalStudy(
        ...     frame,
        ...     design=PointTreatment(
        ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ...     ),
        ... )
        >>> effect = study.identify(ATE(reference=0))
        >>> effect.estimand.reference
        0
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
        effect = (provider or ExplicitAdjustmentProvider()).identify(self, estimand)
        return replace(effect, protocol=self.protocol, _study=self)

    def estimate(
        self,
        estimand: PointEstimand,
        method: str | EstimationMethod = "tmle",
        **overrides: Any,
    ) -> TMLEResult | LongitudinalResult:
        """Identify and estimate one causal quantity.

        Parameters
        ----------
        estimand : Estimand
            Typed causal quantity.
        method : str or EstimationMethod
            Named method or configured method object.
        **overrides : Any
            Method fields overridden for this fit.

        Returns
        -------
        TMLEResult or LongitudinalResult
            Fitted result for the declared design.
        """
        return self.identify(estimand).estimate(method=method, **overrides)


@dataclass(frozen=True)
class IdentifiedEffect(_DefaultingUnpickle):  # numpydoc ignore=PR01
    """Bind an estimand to a functional and its identification assumptions.

    Parameters
    ----------
    estimand : Estimand
        Typed causal quantity.
    functional : BackdoorMeanContrast
        Normalized observed-data target.
    identification : Identification
        Assumptions, nuisance requirements, and remainder condition.
    provider : IdentificationProvider
        Provider that performed identification.
    protocol : StudyProtocol or None
        Scientific study protocol stamped by :class:`CausalStudy`.

    See Also
    --------
    CausalStudy.identify : What returns this object.
    MethodAvailability : One row of :meth:`available_methods`.
    cleverly.EstimationMethod : The configuration :meth:`estimate` accepts.

    Notes
    -----
    ``_study`` is the bound study, kept private because it is not part of the reported
    causal question.  Metadata restored from disk may omit it, and such an effect cannot
    be refitted.

    Examples
    --------
    :meth:`CausalStudy.identify` returns this object.  It carries the assumptions the
    estimate will rest on, and it is what an estimation method is passed to.

    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=40, seed=1)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> effect = study.identify(ATE())
    >>> effect.estimand.name
    'ate'
    >>> sorted(method.name for method in effect.available_methods())[:2]
    ['collaborative_tmle', 'drtmle']
    """

    estimand: PointEstimand
    functional: BackdoorMeanContrast
    identification: Identification
    provider: IdentificationProvider
    protocol: StudyProtocol | None = None
    _study: CausalStudy | None = field(repr=False, compare=False, default=None)

    def available_methods(self) -> tuple[MethodAvailability, ...]:
        """Report estimation methods available for this functional.

        Returns
        -------
        tuple of MethodAvailability
            One capability record for each named method, including refusal reasons.

        Notes
        -----
        The result includes unavailable methods and their refusal reasons. Selecting one
        raises :class:`CapabilityError` before nuisance fitting starts.
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
        """Return a readable identification summary.

        Returns
        -------
        str
            A printable block naming the estimand, the functional, and the assumptions.
        """
        assumptions = "\n".join(f"  - {item}" for item in self.identification.assumptions)
        protocol = (
            ("causal study protocol: absent",)
            if self.protocol is None
            else self.protocol.summary_lines()
        )
        return (
            f"{self.estimand.definition}\n"
            f"identified by {self.provider.name}: {self.functional.expression}\n"
            f"adjustment/history: {list(self.functional.adjustment)}\n"
            f"required nuisances: {list(self.identification.required_nuisances)}\n"
            f"assumptions:\n{assumptions}\n" + "\n".join(protocol)
        )

    def summary_lines(self) -> tuple[str, ...]:
        """Return identification facts for inclusion in result summaries.

        Returns
        -------
        tuple of str
            The same facts as lines, for a result summary to append.
        """
        protocol = (
            ("causal study protocol: absent",)
            if self.protocol is None
            else self.protocol.summary_lines()
        )
        return (
            f"causal estimand: {self.estimand.definition}",
            f"identification: {self.provider.name}; {self.functional.expression}",
            "required nuisances: " + ", ".join(self.identification.required_nuisances),
            "identification assumptions: " + "; ".join(self.identification.assumptions),
            *protocol,
        )

    def estimate(
        self,
        method: str | EstimationMethod = "tmle",
        **overrides: Any,
    ) -> TMLEResult | LongitudinalResult:
        """Estimate the identified functional.

        Parameters
        ----------
        method : str or EstimationMethod
            Named method or configured method object.
        **overrides : Any
            Method fields overridden for this fit.

        Returns
        -------
        TMLEResult or LongitudinalResult
            Fitted result matching the study design.

        Raises
        ------
        CapabilityError
            If the method is unavailable or restored metadata has no bound data.
        MethodConfigurationError
            If ``method`` does not satisfy the estimation-method contract.

        See Also
        --------
        IdentifiedEffect.available_methods : Report supported methods before fitting.
        TMLEMethod : Configure the default estimation method.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> frame, _ = make_linear_ate(n=80, seed=1)
        >>> study = CausalStudy(
        ...     frame,
        ...     design=PointTreatment(
        ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ...     ),
        ... )
        >>> effect = study.identify(ATE())
        >>> result = effect.estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     n_folds=2,
        ...     random_state=0,
        ... )
        >>> sorted(result.estimates)
        ['ate']
        """
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
            raise MethodConfigurationError("method must be a named preset or an EstimationMethod")
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
            provenance=(
                None
                if raw.provenance is None
                else replace(
                    raw.provenance,
                    protocol_fingerprint=(
                        None if self.protocol is None else self.protocol.fingerprint
                    ),
                )
            ),
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
            bootstrap=_narrow_bootstrap(result.bootstrap, estimates),
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
                    alias = arm_alias(
                        target,
                        arm=value,
                        versus=reference,
                        collapse=data.is_binary_treatment,
                    )
                    base[alias] = ParameterKey(alias, target, value=value, reference=reference)
            elif target in {"par", "paf"}:
                alias = arm_alias(target, arm=reference, collapse=data.is_binary_treatment)
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
                    conditional = stratum_alias(alias, label)
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
            provenance=replace(
                raw.provenance,
                protocol_fingerprint=(None if self.protocol is None else self.protocol.fingerprint),
            ),
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
