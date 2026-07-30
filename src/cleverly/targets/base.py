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

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..fluctuation.iterative import InitialFit
from ..fluctuation.submodel import Submodel, TargetGroup
from ..inference.influence import (
    ParameterEstimate,
    Scale,
    counterfactual_means,
    make_estimate,
    unscale,
)
from ..utils.bounds import OutcomeScaler

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = ["Identification", "Target", "TargetContext"]


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
    build:
        Maps a :class:`TargetContext` to the estimate.  Raises ``ValueError`` when
        the target is undefined on those rows.
    """

    name: str
    group: TargetGroup
    scale: Scale
    build: Callable[[TargetContext], ParameterEstimate]
    identification: Identification
    requires_family: str | None = None
    in_default_set: bool = False
    parameter_bounds: tuple[float, float] | None = None
    undefined_when: str = ""
    description: str = ""

    def supported_by(self, family: str) -> bool:
        return self.requires_family is None or self.requires_family == family


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

    @cached_property
    def means(self) -> tuple[float, FloatArray, float, FloatArray]:
        """``(psi1, IC1, psi0, IC0)`` on the scaled outcome scale, computed once."""
        return counterfactual_means(
            self.scaled, self.targeted, self.submodel, self.weights, self.observed
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
        """Map back to the outcome's own units and attach inference."""
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
