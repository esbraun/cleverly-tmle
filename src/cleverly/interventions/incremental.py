r"""Incremental propensity-score interventions: tilting the mechanism that was there.

A :mod:`regime <cleverly.interventions.base>` assigns an arm from :math:`W`, and a
:mod:`shift <cleverly.interventions.shift>` moves the dose a unit received.  An
**incremental propensity-score intervention** does neither.  It leaves the treatment
decision where it was and multiplies its *odds* by :math:`\delta` (Kennedy 2019):

.. math::

    q_\delta(1 \mid W) = \frac{\delta\, g(W)}{\delta\, g(W) + 1 - g(W)},
    \qquad D_\delta(W) = \delta\, g(W) + 1 - g(W),

so that :math:`\operatorname{odds} q_\delta = \delta \cdot \operatorname{odds} g`.
"Make everyone :math:`\delta` times as likely to be treated as they already were."
Its mean is

.. math::

    \Psi(\delta) = E\bigl[m(W)\bigr],
    \qquad
    m(W) = \frac{\delta\, g\, \bar Q(1, W) + (1 - g)\, \bar Q(0, W)}{D_\delta}.

**Why this is a fourth axis and not a** :class:`~cleverly.interventions.Stochastic`
**regime.**  :math:`q_\delta` is built out of :math:`g`, so it is a functional of the
observed-data law rather than a known design.  The efficient influence function therefore
carries a further term for the pathwise derivative through :math:`g`:

.. math::

    \varphi(O) = \underbrace{\frac{\delta A + 1 - A}{D_\delta}
                    \bigl\{Y - \bar Q(A, W)\bigr\}}_{\text{the } \bar Q \text{ score}}
               + \underbrace{\frac{\delta\,\{\bar Q(1,W) - \bar Q(0,W)\}}{D_\delta^2}
                    \bigl(A - g\bigr)}_{\partial m / \partial g}
               + m(W) - \Psi(\delta).

A `Stochastic` regime evaluated at the very same density :math:`q_\delta` has the same
mean and the same clever covariate, entry for entry, and a *different* influence curve --
it is missing the middle term.  Reporting one for the other would report a standard error
for a different functional, which is why
:func:`~cleverly.interventions.refuse_unsupported` said no to this for as long as only the
regime curve existed, and why :func:`~cleverly.inference.influence.ipsi_means` does not
delegate to :func:`~cleverly.inference.influence.regime_means`.
``tests/unit/test_influence_gateaux_ipsi.py`` keeps that as a negative control.

**What the tilt buys: no positivity assumption.**  Every other estimand in this package
divides by :math:`g` somewhere and degrades as :math:`g` approaches zero.  This one does
not.  Its clever covariate is

.. math::

    h_\delta(1, W) = \frac{\delta}{D_\delta}, \qquad h_\delta(0, W) = \frac{1}{D_\delta},

and since :math:`D_\delta` lies between :math:`\min(1, \delta)` and :math:`\max(1, \delta)`
for :math:`g \in [0, 1]`, both are bounded by :math:`\delta` and :math:`1/\delta` *whatever
the mechanism does*.  Neither is computed as a ratio here -- the :math:`g` in numerator and
denominator cancels algebraically, so the code divides only by :math:`D_\delta`, which is
bounded away from zero for any :math:`\delta > 0`.  There is no truncation to apply and
none is applied; see :meth:`IPSISet.evaluate`.

**What it costs: the mechanism must be right.**  :math:`g` appears in the estimand itself,
so the second-order remainder carries :math:`(\hat g - g_0)` as a factor in *every* term.
It vanishes identically when the mechanism is consistent, whatever the outcome regression
does -- but no accuracy in :math:`\bar Q` rescues an inconsistent :math:`\hat g`.  This
estimand is **not doubly robust**, and it is the only one here that is not; see
``tests/unit/test_remainder_ipsi.py``, which asserts the asymmetry in both directions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..data.causal_data import CausalData
from ..exceptions import DataError

__all__ = ["IPSISet", "Incremental", "IncrementalSupport", "check_incremental_support"]


@dataclass(frozen=True)
class Incremental:
    """Multiply everyone's odds of treatment by ``delta``.

    Attributes
    ----------
    delta:
        The odds multiplier, strictly positive.  ``1.0`` is the *natural course* -- the
        intervention that changes nothing, since :math:`q_1 = g` exactly -- whose mean is
        :math:`E[Y]` and which is the usual reference.  Above one shifts the population
        toward treatment, below one away from it.

        Unlike a :class:`~cleverly.interventions.Shift`'s ``cap``, nothing here has to be
        declared to keep the parameter well defined: :math:`q_\\delta` is a probability for
        any :math:`\\delta > 0` and any :math:`g`, which is the whole appeal.
    name:
        What this intervention is called in reported parameter names.  Defaults to
        ``"odds x2.5"`` style, with ``"natural course"`` for ``delta=1``.
    """

    delta: float
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta):
            raise DataError(
                f"an incremental intervention's delta must be finite; got {self.delta!r}"
            )
        if self.delta <= 0.0:
            raise DataError(
                f"an incremental intervention's delta must be strictly positive; got "
                f"{self.delta!r}. delta multiplies the odds of treatment, so delta=0 would "
                "assign nobody -- which is Static(0), a regime, not a tilt of the mechanism."
            )
        if not self.name:
            default = "natural course" if self.delta == 1.0 else f"odds x{self.delta:g}"
            object.__setattr__(self, "name", default)


@dataclass(frozen=True)
class IPSISet:
    """Every declared tilt, evaluated against the estimated mechanism.

    Holds arrays and no callables, for the reason
    :class:`~cleverly.interventions.RegimeSet` and
    :class:`~cleverly.interventions.ShiftSet` do: a fit reached through
    :meth:`~cleverly.estimators.TMLE.retarget` -- the bootstrap, a result loaded from disk
    -- targets the same declared tilts without the mechanism being refit.

    Attributes
    ----------
    names:
        One per tilt, in declaration order.  Report labels.
    deltas:
        The odds multipliers, in declaration order.  Not the keys of the per-parameter
        arrays: those are the **codes** ``0.0 .. R-1.0``, exactly as
        :class:`~cleverly.interventions.ShiftSet` keys by ordinal rather than by delta.
    values:
        ``(n, K, R)``, :math:`q_{\\delta_r}(a \\mid W_i)` -- the tilted density, columns in
        arm-code order.  Rows sum to one by construction.
    weights:
        ``(n, K, R)``, :math:`h_{\\delta_r}(a, W_i) = q_{\\delta_r}(a \\mid W_i) /
        g(a \\mid W_i)` -- the clever covariate arm by arm.  Computed in the cancelled
        form :math:`\\delta / D` and :math:`1 / D` rather than as a ratio, so it never
        divides by a small mechanism; see the module docstring.
    derivative:
        ``(n, R)``, :math:`\\delta_r / D_{\\delta_r}^2` -- the factor multiplying
        :math:`\\{\\bar Q(1, W) - \\bar Q(0, W)\\}(A - g)` in the influence curve, and the
        clever covariate of the *mechanism* fluctuation once the blip is folded in.
    propensity:
        ``(n,)``, the :math:`g(1 \\mid W)` these arrays were built from, **untruncated**.
        Carried because both the influence curve's :math:`(A - g)` and the mechanism
        fluctuation's offset need it, and because an :class:`IPSISet` recomputed at a
        fluctuated mechanism must be able to say which one it describes.
    """

    names: tuple[str, ...]
    deltas: tuple[float, ...]
    values: FloatArray
    weights: FloatArray
    derivative: FloatArray
    propensity: FloatArray
    reference: float = 0.0

    def __post_init__(self) -> None:
        r = len(self.names)
        if len(self.deltas) != r:
            raise ValueError(f"{r} tilt names but {len(self.deltas)} deltas")
        if len(set(self.names)) != r:
            raise ValueError(f"tilt names must be distinct; got {list(self.names)}")
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3 or values.shape[2] != r:
            raise ValueError(
                f"tilted densities must be (n, K, {r}) for tilts {list(self.names)}; "
                f"got shape {values.shape}"
            )
        n, k = values.shape[0], values.shape[1]
        for name, array, shape in (
            ("weights", self.weights, (n, k, r)),
            ("derivative", self.derivative, (n, r)),
            ("propensity", self.propensity, (n,)),
        ):
            if np.asarray(array).shape != shape:
                raise ValueError(
                    f"IPSISet.{name} has shape {np.asarray(array).shape}, expected {shape}"
                )
        if self.reference not in self.codes:
            raise ValueError(
                f"reference={self.reference} is not one of the tilt codes {list(self.codes)}"
            )

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(
        cls,
        incrementals: Sequence[Incremental],
        data: CausalData,
        propensity: FloatArray,
        *,
        reference: str | None = None,
    ) -> IPSISet:
        """Evaluate every tilt against an ``(n, K)`` mechanism, columns in arm-code order.

        ``propensity`` is the **untruncated** out-of-fold mechanism.  Truncating it would
        move the *parameter* rather than the estimator, since :math:`g` appears in
        :math:`\\Psi(\\delta)` itself, and no truncation is needed: nothing here divides by
        :math:`g`.  :func:`~cleverly.estimators.targeting.build_submodel` therefore hands
        the ``ipsi`` builder a bounded mechanism which that builder discards, reading these
        arrays instead.
        """
        if not incrementals:
            raise DataError("at least one incremental intervention is required")
        if data.n_arms != 2:
            raise DataError(
                f"an incremental propensity-score intervention tilts the *odds* of "
                f"treatment, which names two arms; {data.treatment_name} has "
                f"{data.n_arms} ({list(data.treatment_levels)}). Kennedy's tilt has no "
                "single-parameter generalisation to a multinomial mechanism -- one odds "
                "per contrast would be a different intervention with a different "
                "influence function."
            )
        names = tuple(str(item.name) for item in incrementals)
        if len(set(names)) != len(names):
            raise DataError(f"incremental intervention names must be distinct; got {list(names)}")
        deltas = tuple(float(item.delta) for item in incrementals)

        arms = data.arm_codes
        g = np.asarray(propensity, dtype=float)
        if g.ndim != 2 or g.shape[1] != len(arms):
            raise DataError(
                f"propensity must be (n, {len(arms)}) with columns in arm-code order; "
                f"got shape {g.shape}"
            )
        # The control column is taken as the complement rather than read off, so that the
        # tilted density sums to one exactly. This is the same choice `Propensity.bounded`
        # makes for the binary path, and for the same reason -- see CLAUDE.md on the binary
        # path being a regression surface.
        one = g[:, arms.index(1.0)]

        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(
                    f"reference={reference!r} is not one of the incremental interventions "
                    f"{list(names)}"
                )
            code = float(names.index(reference))
        return cls(*_tilt(names, deltas, one), reference=code)

    def at(self, propensity: FloatArray) -> IPSISet:
        """The same tilts recomputed at a different ``(n,)`` mechanism :math:`g(1 \\mid W)`.

        What the mechanism fluctuation calls after each update.  The declared tilts are
        unchanged -- a tilt is a statement about odds, not about a particular :math:`g` --
        so only the evaluated arrays move.
        """
        one = np.asarray(propensity, dtype=float).reshape(-1)
        if one.shape[0] != self.n:
            raise ValueError(f"expected {self.n} propensities, got {one.shape[0]}")
        return type(self)(*_tilt(self.names, self.deltas, one), reference=self.reference)

    # ----------------------------------------------------------------- access

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_tilts(self) -> int:
        return len(self.names)

    @property
    def codes(self) -> tuple[float, ...]:
        """The keys the per-parameter arrays use, ``(0.0, ..., R-1.0)``."""
        return tuple(float(index) for index in range(self.n_tilts))

    @property
    def labels(self) -> dict[float, str]:
        """Code to reported label, which is what ``parameter_name`` is given."""
        return {float(index): name for index, name in enumerate(self.names)}

    def label(self, code: float) -> str:
        return self.labels[float(code)]

    def observed(self, treatment: FloatArray) -> FloatArray:
        """``(n, R)`` -- the clever covariate at the arm each unit actually received."""
        a = np.asarray(treatment, dtype=float).reshape(-1)
        indicator = np.column_stack([(a == arm) for arm in range(self.n_arms)]).astype(float)
        return np.asarray(np.einsum("ij,ijr->ir", indicator, self.weights), dtype=float)

    def blip_weight(self, code: float) -> FloatArray:
        """``(n,)`` -- :math:`\\delta / D_\\delta^2` for one tilt, by code."""
        return np.asarray(self.derivative, dtype=float)[:, int(code)]

    def subset(self, index: Any) -> IPSISet:
        """The same tilts on a row subset -- a fold, a bootstrap resample.

        Sliced rather than re-evaluated, for the reason
        :meth:`~cleverly.interventions.RegimeSet.subset` gives: re-deriving the tilt on a
        resample would let the resample redefine the estimand.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            values=self.values[idx],
            weights=self.weights[idx],
            derivative=self.derivative[idx],
            propensity=self.propensity[idx],
        )


def _tilt(
    names: tuple[str, ...], deltas: tuple[float, ...], one: FloatArray
) -> tuple[tuple[str, ...], tuple[float, ...], FloatArray, FloatArray, FloatArray, FloatArray]:
    """The whole arithmetic of the tilt, for a binary mechanism ``g(1 | W)``.

    Factored out because :meth:`IPSISet.evaluate` and :meth:`IPSISet.at` must produce the
    same arrays from the same mechanism -- the alternating fluctuation calls the second
    once per iteration, and a drift between the two would be invisible until the score
    stopped closing.
    """
    g1 = np.asarray(one, dtype=float).reshape(-1)
    g0 = 1.0 - g1
    values, weights, derivative = [], [], []
    for delta in deltas:
        d = delta * g1 + g0
        # q_1 = delta*g1/D and q_0 = g0/D, which sum to one exactly.
        values.append(np.column_stack([g0 / d, delta * g1 / d]))
        # h_a = q_a / g_a with the mechanism cancelled: delta/D at arm 1, 1/D at arm 0.
        # Written this way there is no division by a small propensity anywhere, which is
        # the arithmetic content of "this estimand needs no positivity assumption".
        weights.append(np.column_stack([1.0 / d, np.full_like(d, delta) / d]))
        derivative.append(delta / d**2)
    return (
        names,
        deltas,
        np.stack(values, axis=2),
        np.stack(weights, axis=2),
        np.column_stack(derivative),
        g1,
    )


# ------------------------------------------------------------------ diagnostics


@dataclass(frozen=True)
class IncrementalSupport:
    """Overlap for one tilt -- which for this estimand is a statement, not a warning.

    The clever covariate is bounded by :math:`\\delta` and :math:`1/\\delta` however small
    the mechanism gets, so ``guaranteed`` is what the tilt promises before seeing any data
    and ``max_ratio`` is what it delivered.  The two agreeing is the normal case; the
    report exists so that a reader can see the effective sample size stay near :math:`n`
    where an arm-indexed fit's would have collapsed.
    """

    name: str
    delta: float
    #: ``(min, max)`` of :math:`\\delta` and :math:`1/\\delta` -- the interval the clever
    #: covariate cannot leave.  :math:`D_\\delta` lies between :math:`\\min(1, \\delta)` and
    #: :math:`\\max(1, \\delta)`, so :math:`1/D` and :math:`\\delta/D` together span exactly
    #: this.  It is a property of the declared tilt alone: no data went into it.
    guaranteed: tuple[float, float]
    min_propensity: float
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float

    def summary(self) -> str:
        low, high = self.guaranteed
        return (
            f"{self.name}: min g(1|W)={self.min_propensity:.3g}, "
            f"covariate in [{low:.3g}, {high:.3g}] by construction, "
            f"max={self.max_ratio:.3g}, "
            f"ESS={self.effective_sample_size:.0f} ({self.ess_ratio:.1%} of n)"
        )


def check_incremental_support(
    tilts: IPSISet, treatment: FloatArray
) -> dict[str, IncrementalSupport]:
    """Per-tilt overlap, in the vocabulary the other two axes' reports use.

    Reports the *bound* alongside the realised maximum because for this estimand the bound
    is the interesting number: it holds whatever the mechanism does, which is what
    distinguishes the tilt from every other intervention here.  An arm-indexed fit on the
    same data would divide by ``min_propensity``; this one never does.
    """
    observed = tilts.observed(treatment)
    n = observed.shape[0]
    out: dict[str, IncrementalSupport] = {}
    for index, name in enumerate(tilts.names):
        delta = tilts.deltas[index]
        column = observed[:, index]
        total = float(column.sum())
        ess = float(total**2 / np.sum(column**2)) if np.any(column > 0) else 0.0
        out[name] = IncrementalSupport(
            name=name,
            delta=delta,
            guaranteed=(min(delta, 1.0 / delta), max(delta, 1.0 / delta)),
            min_propensity=float(np.min(tilts.propensity)) if n else 0.0,
            max_ratio=float(column.max()) if column.size else 0.0,
            effective_sample_size=ess,
            ess_ratio=ess / n if n else 0.0,
        )
    return out
