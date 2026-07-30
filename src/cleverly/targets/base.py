"""What a target parameter is, and what it must declare.

A :class:`Target` bundles the four things that vary between estimands and nothing
else: which fluctuation solves its score equation (``group``), what scale its
inference lives on, what it needs of the outcome, and how to build the estimate
from a targeted distribution.  Everything downstream -- the variance, the
simultaneous bands, the delta method, the score diagnostic, the bootstrap -- is
written against :class:`~cleverly.inference.influence.ParameterEstimate` and so
works for a new target without further changes.

Two design points are worth stating because both are easy to get wrong.

**Name and group are separate axes.**  Five of the seven built-in estimands share
the two-column ``mean`` fluctuation: they are different functionals of the *same*
targeted distribution, not different targeting problems.  Collapsing the two
would fit five fluctuations where one is needed and, worse, would suggest that
adding a functional requires a new score equation.

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

from .._typing import BoolArray, FloatArray, IntArray
from ..fluctuation.iterative import InitialFit
from ..fluctuation.submodel import Submodel, TargetGroup
from ..inference.influence import (
    ArmMean,
    ParameterEstimate,
    Scale,
    counterfactual_means,
    make_estimate,
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
        ``True`` for a target defined only against a single contrast.  The ATT and ATC
        are the built-in cases: their clever covariate reweights one arm by the
        propensity odds, which is only an odds with two arms, and "the effect among the
        treated" does not name one parameter when there are three.  Such a target is
        refused on a multi-arm fit rather than quietly reported for arms 0 and 1.
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
    in_default_set: bool = False
    #: Restricts which arm counts this target is *defaulted* for, without restricting
    #: which it is *defined* for.  ``"multi"`` keeps a target out of a two-armed fit's
    #: default report, for the one case where a narrower target already covers it there:
    #: ``ey`` reports a mean per arm, which on two arms is ``ey1`` and ``ey0`` under
    #: clumsier names.  Asking for it explicitly still works.  A target requiring a binary
    #: treatment is implicitly default-for-binary-only and need not say so twice.
    default_arms: Literal["any", "multi"] = "any"
    parameter_bounds: tuple[float, float] | None = None
    undefined_when: str = ""
    description: str = ""

    def supported_by(self, family: str) -> bool:
        return self.requires_family is None or self.requires_family == family

    def supports_arms(self, n_arms: int) -> bool:
        """Whether this target is defined for a treatment with ``n_arms`` levels."""
        return n_arms == 2 or not self.requires_binary_treatment


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
    arms: tuple[float, ...] = (0.0, 1.0)
    arm_labels: dict[float, Any] = field(default_factory=dict)
    #: The arm contrasts are taken against.  Every non-reference arm gets one contrast.
    reference: float = 0.0

    @cached_property
    def means(self) -> dict[float, ArmMean]:
        """Each arm's counterfactual mean and influence curve, computed once.

        On the *scaled* outcome scale, and shared by every target in the group -- which is
        what keeps the mean-group estimands from recomputing them one target at a time.
        """
        return counterfactual_means(
            self.scaled, self.targeted, self.submodel, self.weights, self.observed
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

        Collapses to the bare stem on a two-armed fit; see :func:`parameter_name`.
        """
        if self.is_binary:
            return parameter_name(stem)
        return parameter_name(
            stem,
            arm=self.label(arm),
            versus=None if versus is None else self.label(versus),
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
