r"""Interventions: what "counterfactual" means for a particular estimand.

Until now an intervention was implicit.  Every counterfactual quantity in the package
was keyed by *arm* -- :attr:`~cleverly.fluctuation.iterative.InitialFit.arms` holds
:math:`\bar Q(a, W)`, :func:`~cleverly.fluctuation.submodel.mean_submodel` builds
:math:`\mathbb 1\{A = a\} / g_a(W)` -- which silently identifies "the intervention" with
"set :math:`A` to the constant :math:`a`".  That is one intervention among many, and the
identification it hides is the reason a rule :math:`d(W)` or a stochastic assignment
:math:`g^\star(\cdot \mid W)` had no way to be expressed.

A **regime** here is a conditional density over the arms,

.. math::

    g^\star(a \mid W), \qquad \sum_a g^\star(a \mid W) = 1 ,

evaluated at every row: an ``(n, K)`` matrix.  All three supported kinds are that one
object.  :class:`Static` puts all its mass on one arm, :class:`Rule` puts it on
:math:`d(W)`, and :class:`Stochastic` spreads it.  Writing them as one representation is
what lets a single clever covariate

.. math::

    h(A, W) = \frac{g^\star(A \mid W)}{g(A \mid W)}

cover the three, and collapse to :math:`\mathbb 1\{A = a\}/g_a(W)` exactly when the
regime is :class:`Static`.

**What is deliberately not here.**  Both are about the *influence function*, not about
effort -- and both are implemented, elsewhere, under keywords of their own:

- An **incremental propensity-score intervention** tilts the population mechanism,
  :math:`g^\star_\delta(1 \mid W) = \delta g_1 / (\delta g_1 + 1 - g_1)`.  Its
  :math:`g^\star` is a functional of :math:`P`, so the efficient influence function
  carries a further term for the pathwise derivative through :math:`g` (Kennedy, 2019)
  that a fixed-density regime curve lacks, and the estimator has to fluctuate the mechanism
  as well as :math:`\bar Q`.  Neither this Protocol -- whose ``density`` sees only the
  data -- nor the influence curve below can express that, which is why it is a parameter
  axis of its own: :mod:`cleverly.interventions.incremental`, the typed estimands
  ``IncrementalMean`` and ``IncrementalEffect``, and ``TMLE(incremental=)``.
  The paragraph stays here rather than being deleted, because the thing to stop a reader
  doing is writing one as a :class:`Stochastic`.  A :class:`Stochastic` must declare
  ``density_kind="known"``, and one declared ``"estimated"`` is refused (roadmap row RM25).
  A :class:`Rule` declares ``rule_kind="known"``, and a user-written :class:`Intervention`
  declares ``density_kind = "known"``, by the same three states (roadmap row RM28).
- A **modified treatment policy** reads the dose that a unit received and moves it, so
  it is not a conditional distribution over the arms.  It is a parameter axis of its own
  too: :mod:`cleverly.interventions.shift`, the typed estimands ``ModifiedTreatmentPolicy``
  and ``ModifiedTreatmentPolicyEffect``, and ``TMLE(shifts=)``.

:func:`refuse_mixed_interventions` refuses either one in a set of regimes, and names the
typed estimands that take it (roadmap row RM14).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, Protocol, cast, runtime_checkable

import numpy as np

from .._declarations import FunctionDeclaration, FunctionKind
from .._typing import FloatArray
from ..data.causal_data import CausalData
from ..exceptions import CapabilityError, DataError

__all__ = [
    "Intervention",
    "RegimeSet",
    "Rule",
    "Static",
    "Stochastic",
    "as_interventions",
    "refuse_mixed_interventions",
    "refuse_regime_densities",
]

#: How close to one a supplied stochastic density's row sums must be.  Loose enough for
#: a user's arithmetic in float32, tight enough that a genuine normalisation mistake --
#: forgetting an arm, or handing over unnormalised weights -- is caught.
_SIMPLEX_TOLERANCE = 1e-8


@runtime_checkable
class Intervention(Protocol):
    """A conditional distribution over the treatment arms.

    Implement :meth:`density`, carry a :attr:`name`, and declare :attr:`density_kind`;
    everything else -- the clever covariate, the influence curve, the positivity report,
    the parameter names -- is written against the ``(n, K)`` matrix and needs to know
    nothing about which kind of intervention produced it.

    ``density`` receives the data of the fit, so it can compute :math:`g^\\star` from the
    analysis sample, and no code can inspect what it computes.  So a user-written class
    declares ``density_kind``.  ``"known"`` says that ``density(data)`` returns a fixed
    function of the covariates, chosen independently of the analysis sample, and it is the
    one value a fit accepts.  :func:`refuse_regime_densities` refuses ``None`` and
    ``"estimated"`` before any learner (roadmap row RM28).  A plain class attribute
    ``density_kind = "known"`` satisfies the protocol.  :class:`Static`, :class:`Rule` and
    :class:`Stochastic` carry their own declarations.  A subclass of :class:`Static` can
    override ``density``, so it declares ``density_kind`` itself.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor.  A protocol is implemented, not instantiated.

    Attributes
    ----------
    name : str
    density_kind : {"known", "estimated"} or None
    """

    @property
    def density_kind(self) -> FunctionKind | None:
        """The declaration that :meth:`density` is a fixed function of the covariates.

        ``"known"`` is the one value a fit accepts.  ``None`` means undeclared, and it is
        refused with ``"estimated"``, a density computed from the analysis sample.  Any
        other value is a :class:`~cleverly.exceptions.DataError`.
        """
        ...

    @property
    def name(self) -> str:
        """What this regime is called in reported parameter names: ``ey[treat_all]``.

        A read-only property rather than a mutable attribute so that the frozen
        dataclasses below satisfy the protocol: an intervention that could be renamed
        after a fit had used it would put the reported parameter names out of step with
        the densities they came from.
        """
        ...

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` array of :math:`g^\\star(a \\mid W_i)`, columns in arm-code order.
        """
        ...


# ------------------------------------------------------------------ level lookup


def _code_for(data: CausalData, level: Any) -> float:
    """The arm code for a user-facing treatment level.

    Levels are compared as the user wrote them, so ``Static("high")`` works on a string
    treatment and ``Static(1)`` on a numeric one.  A level the data does not declare is
    an error naming the ones it does: the alternative -- an all-zero density column --
    would be read downstream as a perfectly well-formed regime that never treats anyone.
    """
    for index, declared in enumerate(data.treatment_levels):
        if declared == level or (
            isinstance(level, (int, float, np.integer, np.floating))
            and isinstance(declared, (int, float, np.integer, np.floating))
            and float(declared) == float(level)
        ):
            return float(index)
    raise DataError(
        f"{level!r} is not a level of {data.treatment_name}; its levels are "
        f"{list(data.treatment_levels)}"
    )


def _one_hot(codes: FloatArray, n_arms: int) -> FloatArray:
    """Degenerate density: all mass on the arm each row names."""
    arm_codes = np.arange(n_arms, dtype=float)
    return np.asarray(codes, dtype=float).reshape(-1, 1) == arm_codes.reshape(1, -1)


def _covariate_frame(data: CausalData) -> Any:
    """The covariates as a dataframe in the caller's backend, for a user's rule to read.

    Covariates only, deliberately.  A rule that reads the outcome is not an intervention,
    and one that reads the observed treatment is a different object again -- a regime
    depending on :math:`A` is not a function of the history a point-treatment parameter
    conditions on.  Restricting the frame is how that is enforced rather than documented.

    The columns are the *encoded* covariates, so a categorical column appears as the
    indicators :meth:`~cleverly.data.CausalData.from_frame` expanded it into
    (``region__west``), not under its original name.  ``data.covariate_names`` is the
    list a rule should be written against.
    """
    return data.frame_like(
        {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
    )


def _as_array(values: Any) -> FloatArray:
    """A numpy array from whatever a user's callable returned (Series, list, array)."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


# ------------------------------------------------------------------- the kinds


@dataclass(frozen=True)
class Static:
    """Set the treatment to one level for everybody: :math:`g^\\star = \\mathbb 1\\{a = v\\}`.

    The degenerate regime, and the one the arm-keyed path has always estimated.  It is
    here for two reasons beyond completeness: it is the reference a rule is usually
    contrasted against, and a fit whose regimes are all :class:`Static` must reproduce
    the ordinary arm fit exactly, which is what ``tests/unit/test_regimes.py`` asserts.

    Parameters
    ----------
    level : Any
        The treatment level to assign to every row, as the caller spells it.
    name : str
        Label used in reported parameter names.  Empty builds ``"always <level>"``.

    Attributes
    ----------
    density_kind : {"known", "estimated"} or None
    """

    level: Any
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"always {self.level}")

    @property
    def density_kind(self) -> FunctionKind | None:
        """``"known"`` when the exact type is :class:`Static`, and ``None`` for a subclass.

        A level is not a function of the sample, so :class:`Static` needs no declaration.  A
        subclass can override :meth:`density`, so it reads as undeclared, by the exact-type
        rule of roadmap row RM27, until it declares ``density_kind`` itself.
        """
        return "known" if type(self) is Static else None

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.
        """
        code = _code_for(data, self.level)
        return _one_hot(np.full(data.n, code), data.n_arms).astype(float)


@dataclass(frozen=True)
class Rule:
    """A dynamic regime: assign :math:`d(W)`, a deterministic function of the covariates.

    ``rule`` is handed a dataframe of the covariates (see :func:`_covariate_frame`) in
    the backend the data arrived in, and returns the *level* -- the user's own label, not
    an internal code -- to assign to each row.

    .. code-block:: python

        Rule(lambda w: np.where(w["age"] > 65, 1, 0), name="treat the elderly", rule_kind="known")

    Every returned level is checked against the declared support before it becomes a
    density, so a rule with a typo or an off-by-one fails naming the levels that exist
    rather than producing a regime nobody asked for.

    ``rule_kind="known"`` declares that ``rule`` is a fixed rowwise function of the
    covariates, chosen independently of the analysis sample.  A callable can close over any
    estimate, such as a threshold at a sample mean, and no code can inspect a closure, so
    the declaration is the check.  :func:`refuse_regime_densities` refuses ``None`` and
    ``"estimated"`` when the rule is built, and ``TMLE`` refuses them again before any
    learner (roadmap row RM28).

    Parameters
    ----------
    rule : callable
        Maps the covariate frame to the treatment level assigned to each row.
    name : str
        Label used in reported parameter names.
    rule_kind : {"known", "estimated"} or None
        The declaration that ``rule`` is a known function.  ``"known"`` is the one value a
        fit accepts.  ``None``, the default, and ``"estimated"`` raise
        :class:`~cleverly.exceptions.CapabilityError`, and any other value raises
        :class:`~cleverly.exceptions.DataError`.  It is the last field, so
        ``Rule(rule, name)`` keeps its positional order.
        :func:`dataclasses.replace` copies it, so a rule replaced with ``rule=`` keeps the
        old declaration.

    Attributes
    ----------
    density_kind : {"known", "estimated"} or None
    """

    rule: Callable[[Any], Any]
    name: str
    rule_kind: FunctionKind | None = None

    def __post_init__(self) -> None:
        refuse_regime_densities((self,))

    @property
    def density_kind(self) -> FunctionKind | None:
        """The declaration ``rule_kind``, under the name that the protocol reads."""
        return self.rule_kind

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.

        Notes
        -----
        It runs :func:`refuse_regime_densities` before it calls ``rule``, because a
        modified rule can reach it directly with a declaration this version refuses.
        """
        refuse_regime_densities((self,))
        assigned = _as_array(self.rule(_covariate_frame(data)))
        if assigned.shape[0] != data.n or assigned.ndim > 1:
            raise DataError(
                f"rule {self.name!r} returned {assigned.shape} assignments for {data.n} "
                "rows; it must return one treatment level per row"
            )
        codes = np.array([_code_for(data, level) for level in assigned.tolist()], dtype=float)
        return _one_hot(codes, data.n_arms).astype(float)


@dataclass(frozen=True)
class Stochastic:
    """A known stochastic regime: assign arm :math:`a` with probability :math:`g^\\star(a \\mid W)`.

    ``density_fn`` is handed the covariate frame and returns an ``(n, K)`` array whose
    columns are in :attr:`~cleverly.data.CausalData.arm_codes` order and whose rows sum
    to one.

    *Known* is the load-bearing word, and ``density_kind="known"`` declares it.
    :math:`g^\\star` must be a fixed function of :math:`W`, chosen independently of the
    analysis sample.  For a population-mechanism-indexed odds tilt, the regime influence
    curve omits the pathwise derivative through the mechanism.  A realized learned density
    defines a different, data-adaptive target; inference for it needs conditions this API
    does not check.  A callable can close over any estimate and no code can inspect a
    closure, so the declaration is the check.  :func:`refuse_regime_densities` refuses
    ``None`` and ``"estimated"`` when the regime is built, and ``TMLE`` refuses them again
    before any learner.  For the population odds-tilt target, use
    :class:`~cleverly.interventions.Incremental`, whose curve carries the mechanism term.

    Parameters
    ----------
    density_fn : callable
        Maps the covariate frame to an ``(n, K)`` array of arm probabilities.
    name : str
        Label used in reported parameter names.
    density_kind : {"known", "estimated"} or None
        The declaration that ``density_fn`` is a known function.  ``"known"`` is the one
        value a fit accepts.  ``None``, the default, and ``"estimated"`` raise
        :class:`~cleverly.exceptions.CapabilityError`, and any other value raises
        :class:`~cleverly.exceptions.DataError`.  It is the last field, so
        ``Stochastic(density_fn, name)`` keeps its positional order.
    """

    density_fn: Callable[[Any], Any]
    name: str
    density_kind: FunctionKind | None = None

    def __post_init__(self) -> None:
        refuse_regime_densities((self,))

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.

        Notes
        -----
        It runs :func:`refuse_regime_densities` before it calls ``density_fn``, because a
        modified regime can reach it directly with a declaration this version refuses.
        """
        refuse_regime_densities((self,))
        values = np.asarray(_as_array(self.density_fn(_covariate_frame(data))), dtype=float)
        check_regime_density(
            values, label=f"stochastic regime {self.name!r}", shape=(data.n, data.n_arms)
        )
        return values


def check_regime_density(
    values: FloatArray, *, label: str, shape: tuple[int, ...] | None = None
) -> None:
    """Check finite probability simplexes with treatment arms on the second axis."""
    if values.ndim < 2 or (shape is not None and values.shape != shape):
        raise DataError(f"{label} returned shape {values.shape}; expected {shape}")
    if not np.all(np.isfinite(values)):
        raise DataError(f"{label} contains a non-finite probability")
    if np.any(values < 0.0):
        raise DataError(f"{label} returned a negative probability")
    sums = values.sum(axis=1)
    worst = float(np.max(np.abs(sums - 1.0))) if sums.size else 0.0
    if worst > _SIMPLEX_TOLERANCE:
        raise DataError(
            f"{label} has rows summing to as far as {worst:.3g} from one; "
            "a regime is a distribution over the arms, so its rows must be normalised"
        )


# ----------------------------------------------------------------- the refusals


InterventionKind = Literal["regime", "shift", "incremental"]


class _KindText(NamedTuple):
    """The words that a refusal of one intervention kind uses."""

    accepts: str  # what a set of the kind accepts
    example: str  # one item of the kind, as a user writes it
    item: str  # what one item of the kind is
    estimands: str  # the typed estimands that hold the kind in a CausalStudy
    keyword: str  # the TMLE keyword that takes the kind on the estimator


_KIND_TEXT: dict[InterventionKind, _KindText] = {
    "regime": _KindText(
        "treatment levels and regimes",
        "Static(1)",
        "a treatment level or a regime, which assigns a distribution over the arms from the "
        "covariates",
        "RegimeMean and RegimeContrast",
        "TMLE(interventions=...)",
    ),
    "shift": _KindText(
        "Shift objects",
        "Shift(0.5, cap=None)",
        "a modified treatment policy, which moves the dose that a unit received",
        "ModifiedTreatmentPolicy and ModifiedTreatmentPolicyEffect",
        "TMLE(shifts=...)",
    ),
    "incremental": _KindText(
        "Incremental objects",
        "Incremental(2.0)",
        "an incremental propensity-score intervention, which multiplies the odds of treatment",
        "IncrementalMean and IncrementalEffect",
        "TMLE(incremental=...)",
    ),
}

#: Why an item of a kind is not a regime.  It follows a refusal in a regimen set only.
_NOT_A_REGIME: dict[InterventionKind, str] = {
    "incremental": (
        " Its g*(a | W) is a functional of P, so its influence curve carries a term for the "
        "treatment mechanism g (Kennedy 2019) that a regime curve lacks."
    ),
    "shift": (
        " A shift is a function d(A, W) of the treatment that a unit received, and a regime "
        "depends on the covariates alone (Haneuse and Rotnitzky 2013). A shift needs a "
        "continuous treatment, treatment_kind='continuous'."
    ),
}


def _intervention_kind(item: object) -> InterventionKind:
    """``"shift"`` for a Shift, ``"incremental"`` for an Incremental, else ``"regime"``.

    A bare value is a regime, because :func:`as_interventions` reads it as a Static level.
    """
    from .incremental import Incremental
    from .shift import Shift

    if isinstance(item, Shift):
        return "shift"
    if isinstance(item, Incremental):
        return "incremental"
    return "regime"


def refuse_mixed_interventions(
    items: Iterable[object], *, kind: InterventionKind, holder: str
) -> None:
    """Raise at the first item of ``items`` that is not of ``kind``.

    One fit estimates one intervention kind.  ``_KIND_TEXT`` gives, for each kind, what a
    set accepts, the typed estimands that hold it, and the ``TMLE`` keyword that takes it.
    The message names the holder, the position and the item.  For an item of another kind
    it names that kind's typed estimands and keyword, and F17 for a joint request.  In a set
    of regimes it also says why the item is not a regime.  A bare value in a shift or
    incremental set is not a regime the user meant, so its message shows the object to
    write instead.  ``CausalStudy.identify`` runs this on the set of each typed estimand,
    :func:`as_interventions` on ``interventions=``, and the ``TMLE`` constructor on
    ``shifts=`` and ``incremental=`` (roadmap row RM14).

    Parameters
    ----------
    items : iterable of object
        The set of one typed estimand or of one ``TMLE`` keyword.
    kind : {"regime", "shift", "incremental"}
        The kind that the holder accepts.
    holder : str
        The field or the keyword that the message names, such as
        ``"RegimeContrast.regimens"`` or ``"shifts="``.

    Raises
    ------
    CapabilityError
        If an item is of another kind.
    """
    text = _KIND_TEXT[kind]
    for position, item in enumerate(items, start=1):
        found = _intervention_kind(item)
        if found == kind:
            continue
        if found == "regime" and not callable(getattr(item, "density", None)):
            raise CapabilityError(
                f"{holder} accepts {text.accepts}, and item {position}, {item!r}, is a bare "
                f"value. Write it as an object, such as {text.example}."
            )
        other = _KIND_TEXT[found]
        why = _NOT_A_REGIME.get(found, "") if kind == "regime" else ""
        raise CapabilityError(
            f"{holder} accepts {text.accepts}, such as {text.example}, and item {position}, "
            f"{item!r}, is {other.item}. {other.estimands} hold that kind in a CausalStudy, "
            f"and {other.keyword} takes it on the estimator.{why} One fit estimates one "
            "intervention kind, and docs/roadmap.md F17 tracks a joint request."
        )


#: Why an estimated regime density is refused.  ``_DENSITY_DECLARATION`` reads it, and
#: :func:`refuse_regime_densities` runs that declaration at every site that checks it.
_ESTIMATED_DENSITY = (
    "a Stochastic regime with an estimated density is refused. For a population-law target "
    "whose g*(a | W) depends on P, the regime influence curve omits its pathwise derivative; "
    "the RM25 odds-tilt witness understates that target's standard error. A realized learned "
    "density instead defines a data-adaptive target whose inference needs conditions this "
    "API does not check. "
    "docs/technical-reference/scope-and-refusals.md (Wrong by construction) records the "
    "refusal, and RM25 in docs/roadmap.md records the reason. For the population odds tilt "
    "of the treatment mechanism, declare cleverly.interventions.Incremental. "
    f"{_KIND_TEXT['incremental'].estimands} hold it in a CausalStudy, and "
    f"{_KIND_TEXT['incremental'].keyword} takes it on the estimator. Its curve carries that "
    "term. Otherwise pass density_fn= as a fixed function of the covariates with "
    "density_kind='known'."
)

_UNDECLARED_DENSITY = (
    "Stochastic needs a declaration of what density_fn is. Pass density_kind='known' "
    "when g*(a | W) is a fixed function of the covariates, chosen independently of the "
    "analysis sample. A sample-derived density is refused: the regime curve omits a term "
    "for a population-law-dependent policy, while inference for a realized learned policy "
    "needs conditions this API does not check (RM25 in docs/roadmap.md)."
)

#: The regime-density declaration: the field ``density_kind``, and the texts of its
#: refusals.  :mod:`cleverly._declarations` holds the three-state check, which the MSM
#: projection-weight declaration shares.
_DENSITY_DECLARATION = FunctionDeclaration(
    "density_kind",
    meaning=(
        "It declares whether the regime density g*(a | W) is a fixed function of the "
        "covariates or one computed from the sample."
    ),
    undeclared=_UNDECLARED_DENSITY,
    estimated=_ESTIMATED_DENSITY,
)

_UNDECLARED_RULE = (
    "a treatment rule needs a declaration of what it is. Pass rule_kind='known' when every "
    "rule is a fixed rowwise function of the covariates or the node history, chosen "
    "independently of the analysis sample: Rule(rule, name, rule_kind='known'), or "
    "DynamicRegimen(label, plan, rule_kind='known') from cleverly.longitudinal. A callable "
    "written inline in regimens= carries no declaration, so write that plan as a "
    "DynamicRegimen. A rule learned from the analysis sample is refused (RM28 in "
    "docs/roadmap.md)."
)

#: Why a learned treatment rule is refused.  ``_RULE_DECLARATION`` reads it.
_ESTIMATED_RULE = (
    "a treatment rule learned from the analysis sample is refused. A realized learned rule "
    "defines a data-adaptive target, and inference for it needs conditions this API does "
    "not check. For a population-indexed rule, such as a threshold at a sample mean, the "
    "regime influence curve can omit a pathwise derivative through the learned statistic; "
    "the RM28 threshold witness measures that gap. An optimal rule can also be nonregular "
    "at ties. docs/technical-reference/scope-and-refusals.md (Wrong by construction) "
    "records the refusal, and RM28 in docs/roadmap.md records the reason. Fix the rule "
    "before the fit, or learn it on data independent of the analysis sample, and declare "
    "rule_kind='known'."
)

#: The treatment-rule declaration that :class:`Rule` and
#: :class:`~cleverly.longitudinal.DynamicRegimen` share: the field ``rule_kind``, and the
#: texts of its refusals.
_RULE_DECLARATION = FunctionDeclaration(
    "rule_kind",
    meaning=(
        "It declares whether a treatment rule is a fixed function of the covariates or the "
        "node history, or one learned from the sample."
    ),
    undeclared=_UNDECLARED_RULE,
    estimated=_ESTIMATED_RULE,
)

_UNDECLARED_INTERVENTION = (
    "a user-written Intervention needs a declaration of what its density is. Give the class "
    "a density_kind attribute of 'known' when density(data) returns a fixed function of the "
    "covariates, chosen independently of the analysis sample. density receives the data of "
    "the fit, so it can compute g*(a | W) from the analysis sample, and such a density is "
    "refused (RM28 in docs/roadmap.md). Static, Rule and Stochastic carry their own "
    "declarations. A subclass of Static can override density, so it declares density_kind "
    "itself."
)

#: Why a user-written intervention with an estimated density is refused.
#: ``_INTERVENTION_DECLARATION`` reads it.
_ESTIMATED_INTERVENTION = (
    "a user-written Intervention with an estimated density is refused. For a population-law "
    "target whose g*(a | W) depends on P, the regime influence curve omits its pathwise "
    "derivative; on the RM25 witness law, a class that computes the sample-mechanism odds "
    "tilt reports 0.62 of that target's exact standard error. A realized learned density "
    "instead defines a data-adaptive target whose inference needs conditions this API does "
    "not check. docs/technical-reference/scope-and-refusals.md (Wrong by construction) "
    "records the refusal, and RM28 in docs/roadmap.md records the reason. For the "
    "population odds tilt of the treatment mechanism, declare "
    f"cleverly.interventions.Incremental. {_KIND_TEXT['incremental'].estimands} hold it in "
    f"a CausalStudy, and {_KIND_TEXT['incremental'].keyword} takes it on the estimator. "
    "Otherwise make density(data) a fixed function of the covariates and set "
    "density_kind = 'known'."
)

#: The density declaration of a user-written :class:`Intervention`: the attribute
#: ``density_kind``, and the texts of its refusals.
_INTERVENTION_DECLARATION = FunctionDeclaration(
    "density_kind",
    meaning=(
        "It declares whether the density g*(a | W) of a user-written Intervention is a "
        "fixed function of the covariates or one computed from the sample."
    ),
    undeclared=_UNDECLARED_INTERVENTION,
    estimated=_ESTIMATED_INTERVENTION,
)


def refuse_regime_densities(interventions: Iterable[object]) -> None:
    """Raise unless every item of ``interventions`` declares a known function.

    A callable can close over any estimate, and no code can inspect a closure, so the
    status of each regime function is a declaration.  The table gives the checks for each
    item, in order.  The declaration check raises
    :class:`~cleverly.exceptions.DataError` for a value outside ``"known"``,
    ``"estimated"`` and ``None``.  It raises
    :class:`~cleverly.exceptions.CapabilityError` for ``None`` and for ``"estimated"``.

    ============================  ==========================  ===================================
    item                          callable check              declaration
    ============================  ==========================  ===================================
    a :class:`Stochastic`         ``density_fn``              ``density_kind`` (RM25)
    a :class:`Rule`               ``rule``                    ``rule_kind`` (RM28)
    any other object              a ``density`` method        ``density_kind``, ``None`` if absent
    ============================  ==========================  ===================================

    The first two rows select by ``isinstance``, so a subclass that skips
    ``__post_init__`` still refuses at the fit.  :class:`Static` meets the last row, and its
    ``density_kind`` reads ``"known"`` by its exact type.  A user-written
    :class:`Intervention` meets the last row too (roadmap row RM28).  The refusal of
    ``"estimated"`` distinguishes a population-law target from a realized learned target.

    :class:`Stochastic` and :class:`Rule` run this when they are built.  ``TMLE`` runs it
    again before any learner and at the start of every retarget.
    :meth:`Stochastic.density`, :meth:`Rule.density` and :meth:`RegimeSet.evaluate` run it
    before any regime function is evaluated, so a direct call and the simulated-confounding
    replay refuse before that function runs.  A regime changed with
    ``object.__setattr__`` can carry a declaration this version refuses (roadmap rows RM25
    and RM28).

    Parameters
    ----------
    interventions : iterable of object
        The declared regimes of a fit, or of a replay.

    Raises
    ------
    DataError
        If a regime function is not callable, or a declaration is not one of the three
        states.
    CapabilityError
        If a declaration is ``None`` or ``"estimated"``.
    """
    for item in interventions:
        # Typed ``object`` on purpose: this checks what a modified regime holds at run
        # time, which its annotations do not guarantee.
        function: object
        if isinstance(item, Stochastic):
            function = item.density_fn
            if not callable(function):
                raise DataError(
                    "Stochastic density_fn= must be callable: covariate_frame -> (n, K) arm "
                    "probabilities, one column per level of data.treatment_levels; got "
                    f"{type(function).__name__}"
                )
            _DENSITY_DECLARATION.refuse(item.density_kind)
        elif isinstance(item, Rule):
            function = item.rule
            if not callable(function):
                raise DataError(
                    "Rule rule= must be callable: covariate_frame -> one treatment level per "
                    f"row; got {type(function).__name__}"
                )
            _RULE_DECLARATION.refuse(item.rule_kind)
        else:
            function = getattr(item, "density", None)
            if not callable(function):
                raise DataError(
                    f"an Intervention needs a density(data) method; got {type(item).__name__}"
                )
            _INTERVENTION_DECLARATION.refuse(getattr(item, "density_kind", None))


# ------------------------------------------------------------------ regime sets


@dataclass(frozen=True)
class RegimeSet:
    """The regimes a fit targets, and their evaluated densities.

    Keyed by an internal float code exactly as the arms are, with the labels carried
    alongside -- the same convention, and for the same reason: every array that is per
    regime (the clever covariate's columns, the counterfactual predictions, the influence
    curves) can then be handled by code that does not count regimes or know what they
    mean, while every reported name uses what the user called them.

    Parameters
    ----------
    names : tuple of str
        Regime labels in code order.
    values : ndarray
        ``(n, K, R)`` evaluated densities.
    reference : float
        Code of the regime contrasts are taken against.

    Attributes
    ----------
    names:
        Regime labels in code order, so ``names[r]`` is the label of code ``float(r)``.
    values:
        ``(n, K, R)`` densities: ``values[i, a, r]`` is :math:`g^\\star_r(a \\mid W_i)`.
    reference:
        The regime code contrasts are taken against, defaulting to the first supplied.
    """

    names: tuple[str, ...]
    values: FloatArray
    reference: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3 or values.shape[2] != len(self.names):
            raise DataError(
                f"regime densities must be (n, K, {len(self.names)}) for regimes "
                f"{list(self.names)}; got shape {values.shape}"
            )
        if len(set(self.names)) != len(self.names):
            raise DataError(f"regime names must be distinct; got {list(self.names)}")
        if self.reference not in self.codes:
            raise DataError(
                f"reference regime code {self.reference} is not one of {list(self.codes)}"
            )
        object.__setattr__(self, "values", values)

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(
        cls,
        interventions: Sequence[Intervention],
        data: CausalData,
        *,
        reference: str | None = None,
    ) -> RegimeSet:
        """Evaluate every intervention on ``data`` and assemble the set.

        Parameters
        ----------
        interventions : sequence of Intervention
            The regimes to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data to evaluate them on.
        reference : str or None
            Label of the regime contrasts are taken against.  ``None`` uses the first one
            supplied, mirroring the arm convention where the reference is the lowest code.

        Returns
        -------
        RegimeSet
            The evaluated densities, keyed by code.

        Notes
        -----
        It runs :func:`refuse_regime_densities` on every intervention before it evaluates
        any density, so a :class:`Rule` or a user-written :class:`Intervention` meets its
        declaration check as a :class:`Stochastic` does.  A regime changed with
        ``object.__setattr__`` can reach this method directly with a declaration this
        version refuses.
        """
        if len(interventions) < 1:
            raise DataError("at least one intervention is required")
        refuse_regime_densities(interventions)
        names = tuple(str(intervention.name) for intervention in interventions)
        stacked = np.stack(
            [np.asarray(intervention.density(data), dtype=float) for intervention in interventions],
            axis=2,
        )
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the regimes {list(names)}")
            code = float(names.index(reference))
        return cls(names, stacked, code)

    # ------------------------------------------------------------------ access

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.values.shape[0])

    @property
    def n_arms(self) -> int:
        """Return the number of treatment arms."""
        return int(self.values.shape[1])

    @property
    def n_regimes(self) -> int:
        """Return the number of regimens."""
        return len(self.names)

    @property
    def codes(self) -> tuple[float, ...]:
        """Regime codes, ``(0.0, ..., R-1.0)``, in the order the regimes were supplied."""
        return tuple(float(r) for r in range(self.n_regimes))

    @property
    def labels(self) -> dict[float, str]:
        """Code to label, which is what :func:`~cleverly.targets.parameter_name` is given."""
        return {float(r): name for r, name in enumerate(self.names)}

    def label(self, code: float) -> str:
        """Return the label a regime code was supplied under.

        Parameters
        ----------
        code : float
            Regime code.

        Returns
        -------
        str
            The label, as the caller spelled it.
        """
        return self.labels[float(code)]

    def column(self, code: float) -> FloatArray:
        """Return one regime's evaluated density.

        Parameters
        ----------
        code : float
            Regime code.

        Returns
        -------
        ndarray
            ``(n, K)`` density for that regime alone.
        """
        return np.asarray(self.values[:, :, round(float(code))], dtype=float)

    def subset(self, index: Any) -> RegimeSet:
        """The same regimes on a row subset -- a bootstrap resample, a validation fold.

        The densities are sliced rather than re-evaluated.  For a regime that is a
        function of :math:`W` alone the two agree exactly, and slicing is what keeps a
        loaded result (which carries the evaluated densities but not the callables that
        made them) usable everywhere a fitted one is.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        RegimeSet
            The same regimes over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, values=self.values[idx])

    @property
    def is_static(self) -> bool:
        """Whether every regime puts all its mass on one arm, the same one for every row.

        The condition under which the regime path is estimating exactly what the arm path
        estimates, and so the one the equivalence test checks.
        """
        for r in range(self.n_regimes):
            column = self.values[:, :, r]
            if column.size and not bool(np.all(column == column[:1])):
                return False
            if not bool(np.all(np.isin(column, (0.0, 1.0)))):
                return False
        return True


def as_interventions(value: Any) -> tuple[Intervention, ...]:
    """Normalise the ``interventions=`` argument into a tuple.

    A bare level is read as :class:`Static` on it, so ``interventions=(1, 0)`` means what
    it looks like it means.  Each item meets the first row of the table that matches it.

    ==============================================  =======================================
    item                                            result
    ==============================================  =======================================
    a ``str`` or ``bytes``                          :class:`Static` on that level
    an object with a callable ``density``, named    the object as is
    an object with a callable ``density``, no name  :class:`~cleverly.exceptions.DataError`
    any other callable                              :class:`~cleverly.exceptions.DataError`
    any other value                                 :class:`Static` on that level
    ==============================================  =======================================

    The second and third rows do not use the runtime :class:`Intervention` check.  That check
    requires ``density_kind``, and an object with no declaration must reach
    :func:`refuse_regime_densities`, whose refusal says what to declare.  This function
    checks no declaration, so a fit refuses an undeclared object before its first learner
    (roadmap row RM28).  A callable is not a treatment level, and it carries no name and no
    declaration, so the fourth row refuses it and names :class:`Rule`.

    A :class:`~cleverly.interventions.Shift` or
    :class:`~cleverly.interventions.Incremental` is neither a level nor a regime, so
    :func:`refuse_mixed_interventions` refuses it before any item is read.  Both are
    implemented, under typed estimands and keywords of their own, and the ``Static``
    fallthrough would wrap the object as though it were a treatment *level* -- giving a
    regime named ``"always Shift(delta=0.5, ...)"`` and an error much further downstream,
    about something else.

    Parameters
    ----------
    value : Any
        The ``interventions=`` argument: ``None``, one item, or a list or tuple of items.

    Returns
    -------
    tuple of Intervention
        The regimes in the order given.  ``None`` gives the empty tuple.

    Raises
    ------
    DataError
        If an item is a callable with no ``density``, or an object with a ``density`` and
        no ``name``.
    CapabilityError
        If an item is a :class:`~cleverly.interventions.Shift` or an
        :class:`~cleverly.interventions.Incremental`.
    """
    if value is None:
        return ()
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    refuse_mixed_interventions(items, kind="regime", holder="interventions=")
    out: list[Intervention] = []
    for item in items:
        if isinstance(item, (str, bytes)):
            out.append(Static(item))
        elif callable(getattr(item, "density", None)):
            if not hasattr(item, "name"):
                raise DataError(
                    f"{type(item).__name__} has a density method but no name. An "
                    "Intervention carries a name, a density(data) method and a density_kind "
                    "declaration."
                )
            out.append(cast(Intervention, item))
        elif callable(item):
            label = getattr(item, "__name__", type(item).__name__)
            raise DataError(
                f"interventions= received a callable, {label}, and a callable is not a "
                "treatment level. Write a rule as Rule(rule, name, rule_kind='known'); a "
                "bare callable carries no name and no declaration."
            )
        else:
            out.append(Static(item))
    return tuple(out)
