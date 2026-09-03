"""What a target parameter is, and what it must declare.

A :class:`Target` bundles the four things that vary between estimands and nothing
else: which fluctuation solves its score equation (``group``), what scale its
inference lives on, what it needs of the outcome, and how to build the estimate
from a targeted distribution.  Everything downstream -- the variance, the
simultaneous bands, the delta method, the score diagnostic, the bootstrap -- is
written against :class:`~cleverly.inference.influence.ParameterEstimate` and so
works for a new target without further changes.

Two design points are worth stating because both are easy to get wrong.

**Name and group are separate axes.**  Most of the built-in estimands share the
``mean`` fluctuation -- one column per treatment arm: they are different functionals
of the *same* targeted distribution, not different targeting problems.  Collapsing
the two would fit one fluctuation per functional where one in total is needed and,
worse, would suggest that adding a functional requires a new score equation.

**Identification is declared, not derived.**  :class:`Identification` records the
assumptions a target rests on, which nuisances it consumes, and what its
double-robustness condition actually says.  The library does not attempt to
*derive* an identifying functional from a causal graph -- that needs a DAG the
package deliberately does not accept, and it is a research problem rather than a
feature.  What it can do is refuse to let an estimand ship without its
assumptions written down where ``summary()`` can print them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, ParameterAxis
from ..fluctuation.iterative import InitialFit
from ..fluctuation.submodel import Submodel, TargetGroup
from ..inference.influence import (
    ArmMean,
    ParameterEstimate,
    Scale,
    counterfactual_means,
    ipsi_means,
    make_estimate,
    msm_coefficients,
    regime_means,
    shift_means,
    unscale,
)
from ..utils.bounds import OutcomeScaler

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "Identification",
    "Target",
    "TargetContext",
    "parameter_name",
    "parameter_stem",
]


def parameter_name(stem: str, *, arm: Any = None, versus: Any = None) -> str:
    """The reported name of one parameter, given the arms it refers to.

    The single place the naming rule lives, so that it is one documented convention
    rather than a decision repeated at every target.

    With **two arms** the historical short names are kept -- ``"ate"``, ``"ey1"``,
    ``"ey0"``, ``"rr"``, ``"or"`` -- by the targets passing no arm at all.  They are
    unambiguous there, and every doc, test, fixture and downstream script uses them;
    renaming them would buy nothing.

    With **more than two arms** there is no unambiguous short name, so the arms appear:
    ``"ey[high]"``, ``"ate[high vs low]"``.  The labels are the user's own levels, taken
    from :attr:`~cleverly.data.CausalData.treatment_levels`, never the internal float
    codes -- a reader should not have to translate ``2.0`` back to ``"high"``.
    """
    if arm is None:
        return stem
    if versus is None:
        return f"{stem}[{arm}]"
    return f"{stem}[{arm} vs {versus}]"


def arm_alias(stem: str, *, arm: Any, versus: Any = None, collapse: bool) -> str:
    """The reported name of an arm-indexed parameter, given whether the fit collapses it.

    One rule, stated once.  Four call sites had to get from a target and a pair of arms to
    the name the fit reports it under, and all four wrote the same conditional: a two-armed
    fit keeps the bare stem, and any other fit names its arms.  Written four times it is
    four chances for a report and its sensitivity analysis to disagree about what a
    parameter is called.

    ``collapse`` is the caller's own answer to "does this fit have exactly two arms",
    because the callers read it from different objects.  A study and the sensitivity
    modules read ``data.is_binary_treatment``; :meth:`TargetContext.name_for` reads its own
    arm set and its ``always_label`` override.

    Parameters
    ----------
    stem : str
        The target the parameter came from, as :func:`parameter_name` means it.
    arm : Any
        The label of the arm the parameter is about.
    versus : Any
        The label of the reference arm of a contrast, or ``None`` for a level.
    collapse : bool
        Whether to report the bare stem instead of naming the arms.

    Returns
    -------
    str
        The reported parameter name.
    """
    if collapse:
        return parameter_name(stem)
    return parameter_name(stem, arm=arm, versus=versus)


def parameter_stem(name: str) -> str:
    """The target a reported parameter came from: everything before the ``[``.

    The inverse of :func:`parameter_name`, and the reason that function is the only
    place the convention lives.  Needed because a target now reports several parameters
    and the caller has to get back from ``"ate[medium vs low]"`` to the target ``"ate"``
    -- to order the report by target, and to re-request the target when a sensitivity
    sweep re-targets.
    """
    return name.split("[", 1)[0]


@dataclass(frozen=True)
class Identification:
    """The assumptions under which a target is the causal quantity it claims to be.

    Attributes
    ----------
    assumptions:
        What must hold for the observed-data functional to equal the causal
        parameter.  Written for a reader, not for a parser.
    required_nuisances:
        Which nuisance estimates enter the influence function.
    dr_condition:
        What double robustness actually buys for *this* estimand.  It is not the
        same sentence for every one: adding a missingness mechanism turns "``Qbar``
        right or ``g`` right" into "``Qbar`` right or the *product* ``g * pi``
        right", which is a strictly stronger requirement on the mechanism half.
    references:
        Where the influence function is derived.
    """

    assumptions: tuple[str, ...]
    required_nuisances: tuple[str, ...]
    dr_condition: str
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class Target:
    """One estimand: how to build it, and what it rests on.

    Attributes
    ----------
    group:
        Which fluctuation solves this target's score equation.  Several targets
        share one group; see the module docstring.
    scale:
        ``"level"`` for a mean, ``"difference"`` for a contrast of means,
        ``"ratio"`` for one reported on the log scale and exponentiated.
    requires_family:
        ``"binomial"`` for estimands that need the counterfactual means to be
        probabilities.  Replaces a hard-coded ratio special case.
    parameter_bounds:
        The range the parameter is defined on, where that is narrower than the real
        line.  Reported; not enforced by clipping.
    undefined_when:
        Non-empty for a target that a *subsample* can legitimately fail to support --
        an ATT where a fold contains no treated units, a risk ratio where a
        counterfactual mean sits on the boundary.  A fold that cannot evaluate such a
        target drops it and says so, using this text.  A target with an empty
        ``undefined_when`` that raises is a bug, and the error is re-raised.

        This replaces a bare ``except ValueError`` that retried without ``{"rr","or"}``
        and then returned an empty dict, which turned any exception anywhere in the
        estimate path into a silently missing fold.
    requires_binary_treatment:
        ``True`` for a target that names one of exactly two arms, or whose intervention
        does.  ``ey1`` and ``ey0`` are the plain cases -- with three arms there is no
        "the" treated mean, and ``ey`` reports one per arm instead -- and the incremental
        estimands are the substantive one: Kennedy's tilt multiplies the *odds* of
        treatment, which is an odds only with two arms.  Such a target is refused on a
        multi-arm fit rather than quietly reported for arms 0 and 1.

        The ATT and ATC used to declare this and no longer do: they are one parameter per
        non-reference arm, ``E[Y^a - Y^ref | A = a]``, which is the same derivation with
        ``1{A = a}`` and the odds ``g_a / g_ref`` in place of the binary pair.
    parameter_axis:
        What this target's parameters are indexed by: ``"arm"`` for a treatment level,
        ``"regime"`` for a regime declared with ``interventions=``, ``"shift"`` for a
        modified treatment policy declared with ``shifts=``, ``"ipsi"`` for a tilt of the
        mechanism declared with ``incremental=``, ``"msm"`` for a coefficient of a working
        model declared with ``msm=``.

        The five **partition** the registry rather than accumulating: a target is
        unavailable unless the fit's own axis matches, and declaring one axis makes the
        others unavailable in turn.  They are not alternative spellings of one report.
        A single fit reporting ``E[Y(1)]``, ``E[Y^{g*}]``, ``E[Y^{d}]`` and a working
        model's slope from one fluctuation would be reporting four different score
        equations under one heading.

        The first four also declare what "counterfactual" means for the fit.  ``"msm"``
        is the one that does not: its counterfactuals are still the arms, and the
        fluctuation still updates ``Qbar`` at every one of them.  What changes is the
        *summary* -- ``p`` score equations, one per term, in place of ``K``, one per arm.
        That is enough to make it an axis: the coefficients of a summary are not indexed
        by anything the other four name.

        ``"ipsi"`` is the one whose intervention is a functional of the observed-data law
        rather than a declaration about it, which is why it is not a kind of ``"regime"``:
        its ``q_delta`` is built out of the estimated mechanism, so its influence curve
        carries a further term and its targeting has a second score equation.

        The axis is also not the same question as ``group``.  A group is a score
        equation and several targets share one; an axis is what the resulting
        parameters are *named by*.  ``ey_shift`` and ``ate_shift`` share the ``mtp``
        group and the ``"shift"`` axis, but ``ate`` and ``att`` share the ``"arm"``
        axis across two different groups.
    build:
        Maps a :class:`TargetContext` to **one or more** estimates.  Returns a sequence
        because one target is one *functional*, not one number: with ``K`` arms ``ey``
        is a mean per arm and ``ate`` a contrast per non-reference arm.  Raises
        ``ValueError`` when the target is undefined on those rows.
    """

    name: str
    group: TargetGroup
    scale: Scale
    build: Callable[[TargetContext], Sequence[ParameterEstimate]]
    identification: Identification
    requires_family: str | None = None
    requires_binary_treatment: bool = False
    parameter_axis: ParameterAxis = "arm"
    in_default_set: bool = False
    #: Restricts which arm counts this target is *defaulted* for, without restricting
    #: which it is *defined* for.  Asking for it explicitly always works.
    #:
    #: ``"multi"`` keeps a target out of a two-armed fit's default report, for the one
    #: case where a narrower target already covers it there: ``ey`` reports a mean per
    #: arm, which on two arms is ``ey1`` and ``ey0`` under clumsier names.
    #:
    #: ``"binary"`` is the converse, and is about not changing a report rather than about
    #: redundancy: ``att`` and ``atc`` are defined for any number of arms, but on ``K``
    #: they are ``2(K - 1)`` further parameters and two further fluctuations, and a
    #: default that grew to include them would move the simultaneous bands of every
    #: multi-arm fit that already existed.  A target requiring a binary treatment is
    #: implicitly default-for-binary-only and need not say so twice.
    default_arms: Literal["any", "multi", "binary"] = "any"
    parameter_bounds: tuple[float, float] | None = None
    undefined_when: str = ""
    description: str = ""

    def supported_by(self, family: str) -> bool:
        return self.requires_family is None or self.requires_family == family

    def supports_arms(self, n_arms: int) -> bool:
        """Whether this target is defined for a treatment with ``n_arms`` levels."""
        return n_arms == 2 or not self.requires_binary_treatment

    def matches_axis(self, axis: ParameterAxis) -> bool:
        """Whether this target belongs in a fit whose parameters are indexed by ``axis``."""
        return self.parameter_axis == axis


@dataclass
class TargetContext:
    """Everything a target needs to turn a targeted fit into an estimate.

    One context is built per fluctuation and shared by every target in that group,
    which is what keeps the five mean-group estimands from recomputing the
    counterfactual means five times over.
    """

    scaled: FloatArray
    targeted: InitialFit
    submodel: Submodel
    treatment: FloatArray
    weights: FloatArray
    observed: BoolArray
    scaler: OutcomeScaler
    n: int
    cluster: IntArray | None = None
    alpha_sig: float = 0.05
    #: Arm codes, ascending, and the labels to report them under.  ``arm_labels`` maps a
    #: code to the level the user supplied, and is what :func:`parameter_name` is given.
    #:
    #: On a **regime** or **shift** fit these carry regime (or shift) codes and labels
    #: instead.  The axes are interchangeable here on purpose: a target that loops
    #: the keys of :attr:`means` and names each one is estimating a mean per arm, per
    #: regime or per shift with the same code, which is what lets the regime and shift
    #: targets reuse the arm builders.
    arms: tuple[float, ...] = (0.0, 1.0)
    arm_labels: dict[float, Any] = field(default_factory=dict)
    #: The arm contrasts are taken against.  Every non-reference arm gets one contrast.
    reference: float = 0.0
    #: ``(n, K, R)`` regime densities, for the ``regime`` fluctuation; ``None`` otherwise.
    regimes: FloatArray | None = None
    #: The doubly-robust corrections per arm, ``D*_Q + D*_g`` -- or whichever of the two a
    #: single-guard fit solved for, since
    #: :meth:`~cleverly.inference.influence.CorrectionParts.total` selects on the guard --
    #: for a fit that solved the
    #: extra score equations; ``None`` for every other fit.  Subtracted from the ``mean``
    #: group's curves and reaching nothing else -- the other groups have no reduced-
    #: dimension derivation, and a fit declaring one is refused before it gets here.
    corrections: dict[float, FloatArray] | None = None
    #: ``(n, S + 1, S)`` shift clever covariates, for the ``mtp`` fluctuation; ``None``
    #: otherwise.  Carried only to select the mean function: unlike ``regimes``, whose
    #: densities the plug-in term averages ``Qbar`` against, a shift's plug-in term is
    #: already in ``targeted.arms`` and the covariate is already in ``submodel``.
    shifts: FloatArray | None = None
    #: ``(n, K, p)`` working-model design and ``(n, K)`` weights, for the ``msm``
    #: fluctuation; ``None`` otherwise.  Two arrays rather than one product because the
    #: projection needs them apart: the Gram matrix is ``h * phi * phi'`` and the fitted
    #: values are ``phi' beta``, so neither can be recovered from ``h * phi`` alone.
    msm_design: FloatArray | None = None
    msm_weights: FloatArray | None = None
    #: The working model's link, by name.  ``"identity"`` leaves the projection linear and
    #: the clever covariate free of ``beta``; anything else puts ``dm/deta`` in both.
    msm_link: str = "identity"
    #: The :class:`~cleverly.interventions.IPSISet` **as targeted**, for the ``ipsi``
    #: fluctuation; ``None`` otherwise.  The whole object rather than an array, because
    #: the influence curve needs the tilted density, the derivative and the mechanism the
    #: two were built from, and reading them off one object is what stops the middle term
    #: being evaluated at a different ``g`` than the plug-in used.
    incremental: Any | None = None
    #: Report every parameter with its label even when there are exactly two of them.
    #: The two-arm short names (``"ate"``, ``"ey1"``) exist because they are historical
    #: and unambiguous; two *regimes* have neither property, and "the ATE" of a rule
    #: against a reference regime is not a name a reader can resolve without the labels.
    always_label: bool = False

    @cached_property
    def observed_mean(self) -> ArmMean:
        r"""The natural-course mean :math:`E[Y]` and its empirical influence curve.

        This is deliberately available only for a fully observed outcome.  Under MAR the
        efficient curve gains an outcome-regression and missingness block, and the mean
        fluctuation used for static interventions does not solve that additional score.
        Refusing that composition is what keeps PAR/PAF from silently becoming a
        complete-case parameter.
        """
        if not np.all(self.observed):
            raise ValueError(
                "ey_obs, par and paf do not yet support delta=: under missingness at "
                "random E[Y] needs its own outcome/missingness score equation, and the "
                "complete-case mean is a different parameter"
            )
        y = np.asarray(self.scaled, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        psi = float(np.average(y, weights=w))
        return ArmMean(psi, w * (y - psi))

    @cached_property
    def means(self) -> dict[float, ArmMean]:
        """Each arm's -- or regime's, or shift's -- mean and influence curve.

        On the *scaled* outcome scale, computed once and shared by every target in the
        group, which is what keeps the mean-group estimands from recomputing them one
        target at a time.

        The shift branch comes first and does *not* delegate to ``regime_means``, though
        the induced density makes the two clever covariates identical entry for entry.
        A regime's plug-in term averages ``Qbar`` over the arms; a shift's reads the dose
        the unit actually received, and the two agree only in conditional expectation
        given ``W`` -- see :func:`~cleverly.inference.influence.shift_means`, whose
        docstring states the exact variance gap, and the negative control in
        ``tests/unit/test_influence_gateaux_shift.py`` that fails if someone merges them.

        The ``msm`` branch is keyed by *coefficient* rather than by a counterfactual, and
        its values are already on the outcome's own scale -- see
        :func:`~cleverly.inference.influence.msm_coefficients` for why the projection is
        solved there.  A target reading this branch must call :meth:`finish_unscaled`.
        """
        if self.msm_design is not None:
            assert self.msm_weights is not None
            return msm_coefficients(
                self.scaled,
                self.targeted,
                self.submodel,
                self.msm_design,
                self.msm_weights,
                self.weights,
                self.scaler,
                self.observed,
                self.msm_link,
            )
        if self.incremental is not None:
            return ipsi_means(
                self.scaled,
                self.targeted,
                self.submodel,
                self.incremental,
                self.treatment,
                self.weights,
                self.observed,
            )
        if self.shifts is not None:
            return shift_means(
                self.scaled, self.targeted, self.submodel, self.weights, self.observed
            )
        if self.regimes is not None:
            return regime_means(
                self.scaled,
                self.targeted,
                self.submodel,
                self.regimes,
                self.weights,
                self.observed,
            )
        return counterfactual_means(
            self.scaled,
            self.targeted,
            self.submodel,
            self.weights,
            self.observed,
            self.corrections,
        )

    @property
    def is_binary(self) -> bool:
        return len(self.arms) == 2

    @property
    def contrast_arms(self) -> tuple[float, ...]:
        """The non-reference arms, in ascending order: one contrast each."""
        return tuple(arm for arm in self.arms if arm != self.reference)

    def label(self, arm: float) -> Any:
        """The reported label for an arm code."""
        return self.arm_labels.get(arm, arm)

    def name_for(self, stem: str, arm: float, *, versus: float | None = None) -> str:
        """The parameter name for a per-arm or per-contrast estimate of ``stem``.

        Collapses to the bare stem on a two-armed fit unless :attr:`always_label` says
        otherwise; see :func:`arm_alias`, which is the same rule the study and the
        sensitivity modules apply.
        """
        return arm_alias(
            stem,
            arm=self.label(arm),
            versus=None if versus is None else self.label(versus),
            collapse=self.is_binary and not self.always_label,
        )

    def finish(
        self,
        name: str,
        psi: float,
        ic: FloatArray,
        scale: Scale,
        *,
        log_psi: float | None = None,
    ) -> ParameterEstimate:
        """Map back to the outcome's own units and attach inference.

        ``psi`` and ``ic`` are on the *scaled* outcome scale -- which is the outcome's
        own scale for a binary outcome, and ``[0, 1]`` for a bounded continuous one
        (see :class:`~cleverly.utils.bounds.OutcomeScaler`).  The mapping back is the
        linear one the declared ``scale`` implies: a level picks up the location
        shift, a difference and every influence curve pick up the range factor only.

        That is exact for a functional **linear in the scaled counterfactual means**,
        which covers every built-in estimand.  It is *not* exact for a nonlinear one:
        the number needed to treat, say, is ``1 / ATE``, and ``1 / (range * x)`` is not
        ``range * (1 / x)``.  A target computing a nonlinear functional of the means on
        a scaled outcome must either unscale the means itself before combining them, or
        declare ``requires_family="binomial"``, where the scaler is the identity and the
        question does not arise.  The ratios do the latter.
        """
        if log_psi is None:
            value, curve = unscale(psi, ic, self.scaler, scale)
        else:
            value, curve = psi, np.asarray(ic, dtype=float)
        return make_estimate(
            name,
            value,
            curve,
            n=self.n,
            cluster=self.cluster,
            scale=scale,
            alpha=self.alpha_sig,
            log_psi=log_psi,
        )

    def finish_unscaled(
        self, name: str, psi: float, ic: FloatArray, scale: Scale
    ) -> ParameterEstimate:
        """Attach inference to an estimate that is *already* in the outcome's own units.

        The escape hatch :meth:`finish` names: a target whose functional is not linear in
        the scaled counterfactual means has to unscale the means itself, and must then not
        be unscaled again.  The ``msm`` target is the built-in case -- a coefficient vector
        has no single :class:`Scale` to map back with, so
        :func:`~cleverly.inference.influence.msm_coefficients` solves the projection where
        the coefficients are reported.
        """
        return make_estimate(
            name,
            psi,
            ic,
            n=self.n,
            cluster=self.cluster,
            scale=scale,
            alpha=self.alpha_sig,
        )
