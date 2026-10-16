r"""Incremental propensity-score interventions: tilting the mechanism that was there.

A :mod:`regime <cleverly.interventions.base>` assigns an arm from :math:`W`, and a
:mod:`shift <cleverly.interventions.policy>` moves the dose a unit received.  An
**incremental propensity-score intervention** does neither.  It leaves the treatment
decision where it was and multiplies the *odds* of each level by a declared factor
(Kennedy 2019 for two levels):

.. math::

    q_\delta(a \mid W) = \frac{\delta_a\, g(a \mid W)}{D_\delta(W)},
    \qquad D_\delta(W) = \sum_b \delta_b\, g(b \mid W),

so that :math:`q_\delta(a)/q_\delta(b) = (\delta_a/\delta_b)\, g(a)/g(b)`.  The family is
scale free.  With two levels and :math:`(\delta_0, \delta_1) = (1, \delta)` it is Kennedy's
Equation (1),

.. math::

    q_\delta(1 \mid W) = \frac{\delta\, g(W)}{\delta\, g(W) + 1 - g(W)}.

"Make everyone :math:`\delta` times as likely to be treated as they already were."
Its mean is

.. math::

    \Psi(\delta) = E\bigl[m(W)\bigr],
    \qquad
    m(W) = \sum_a q_\delta(a \mid W)\, \bar Q(a, W).

**Why this is a fourth axis and not a** :class:`~cleverly.interventions.Stochastic`
**regime.**  :math:`q_\delta` is built out of :math:`g`, so it is a functional of the
observed-data law rather than a known design.  The efficient influence function therefore
carries a further term for the pathwise derivative through :math:`g`.  The mechanism score
of level :math:`b` is :math:`1\{A=b\} - g_b` and
:math:`\partial m/\partial g_b = \delta_b\{\bar Q_b - m\}/D_\delta`, so

.. math::

    \varphi(O) = \underbrace{\frac{\delta_A}{D_\delta}
                    \bigl\{Y - \bar Q(A, W)\bigr\}}_{\text{the } \bar Q \text{ score}}
               + \underbrace{\frac{\delta_A}{D_\delta}
                    \bigl\{\bar Q(A, W) - m(W)\bigr\}}_{\text{mechanism term}}
               + m(W) - \Psi(\delta)
             = \frac{\delta_A}{D_\delta}\{Y - m(W)\} + m(W) - \Psi(\delta).

For two levels the mechanism term is
:math:`\delta\{\bar Q(1,W) - \bar Q(0,W)\}(A - g)/D_\delta^2`.  A `Stochastic` regime
evaluated at the very same density :math:`q_\delta` has the same mean and the same clever
covariate, entry for entry, and a *different* influence curve -- it is missing the
mechanism term.  Reporting one for the other would report a standard error for a different
functional.  That is why the regime path refuses an Incremental
(:func:`~cleverly.interventions.refuse_mixed_interventions`), and why
:func:`~cleverly.inference.influence.ipsi_means` does not delegate to
:func:`~cleverly.inference.influence.regime_means`.
``tests/unit/test_influence_gateaux_ipsi.py`` keeps that as a negative control.

**What the tilt buys: no positivity assumption.**  Every other estimand in this package
divides by :math:`g` somewhere and degrades as :math:`g` approaches zero.  This one does
not.  Its clever covariate is

.. math::

    h_\delta(a, W) = \frac{\delta_a}{D_\delta},

and since :math:`D_\delta` lies between :math:`\min_b \delta_b` and :math:`\max_b \delta_b`
for any mechanism on the simplex, :math:`h` lies between :math:`\min\delta/\max\delta`
and :math:`\max\delta/\min\delta` *whatever the mechanism does*.  Neither is computed
as a ratio here -- the :math:`g` in numerator and denominator cancels algebraically, so the
code divides only by :math:`D_\delta`, which is bounded away from zero for any positive
factors.  There is no truncation to apply and none is applied; see
:meth:`IPSISet.evaluate`.

**What it costs: the mechanism must be right.**  :math:`g` appears in the estimand itself,
so the second-order remainder carries :math:`(\hat g - g_0)` as a factor in *every* term:

.. math::

    R = E_0\!\left[(\hat m - m_0)\,\frac{\sum_b \delta_b(\hat g_b - g_{0b})}{\hat D}\right].

It vanishes identically when the mechanism is consistent, whatever the outcome regression
does -- but no accuracy in :math:`\bar Q` rescues an inconsistent :math:`\hat g`.  This
estimand is **not doubly robust**, and it is the only one here that is not; see
``tests/unit/test_remainder_ipsi.py``, which asserts the asymmetry in both directions.

**With ``delta=``, a consistent mechanism stops being enough.**  Missingness adds
:math:`\pi(A, W)` to the outcome half of the clever covariate and leaves the tilt alone --
:math:`q_\delta` is a functional of :math:`P(A \mid W)`, and :math:`A` and :math:`W` are
recorded however much of :math:`Y` is not.  The remainder keeps its squared
:math:`(\hat g - g_0)` term untouched and gains a cross term carrying the error in the
*product* :math:`\hat g\hat\pi` against the :math:`\bar Q` error, so it vanishes exactly
when :math:`\hat g` is consistent **and** one of :math:`\hat\pi`, :math:`\bar Q` is -- the
strictest condition on any axis here, and the one place where the guarantee above is
necessary rather than sufficient.  ``tests/unit/test_remainder_ipsi_mar.py`` states it as
an equality; note in particular that the two mechanisms cannot trade off against each
other the way they can on the arm-indexed path, because :math:`\hat g` is in the estimand.

**Two levels dispatch to the binary arithmetic.**  On two levels every array is computed
with Kennedy's form, :math:`\delta = \delta_1/\delta_0` and :math:`D = \delta g_1 + g_0`,
so the binary surface is bit-identical to the one it had before the factor form existed.
The general form runs on three or more levels.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from .._refusal import Refusal, RefusalReason
from .._typing import FloatArray
from ..data.causal_data import CausalData
from ..data.weighting import effective_sample_size, format_score_load
from ..exceptions import DataError
from .support import _intervention_loads, _InterventionLoadRow

__all__ = ["IPSISet", "Incremental", "IncrementalSupport", "check_incremental_support"]


def refuse_continuous_tilt(data: CausalData) -> None:
    """Refuse an odds tilt on a continuous treatment.

    Shared rather than exported: :class:`~cleverly.estimators.TMLE` imports it directly and
    it is deliberately absent from ``__all__``.  Asked twice, once by the estimator while it
    validates the request and once by :meth:`IPSISet.evaluate`, for the reason the old
    two-arm guard was: the estimator refuses before any nuisance is fitted, and this refuses
    anything that reaches the evaluator by another route.  A vector treatment with a dose
    component raises its own refusal when the property is read.

    Parameters
    ----------
    data : CausalData
        The validated study data.

    Raises
    ------
    DataError
        If the treatment is continuous.
    """
    if not data.is_continuous_treatment:
        return
    raise DataError(
        f"an incremental propensity-score intervention multiplies the odds of treatment "
        f"levels; {data.treatment_name} is continuous and has no levels.",
        reason=RefusalReason.DIFFERENT_ESTIMAND,
        alternative=(
            "policies=[Shift(0.5, cap=None)] is the mean outcome when every received dose "
            "rises by 0.5, and Shift(0.0, cap=None) is the natural course."
        ),
    )


#: What ``Incremental(delta=...)`` accepts.  The normalised form is the tuple of pairs.
Declared = float | Mapping[Hashable, float] | tuple[tuple[Hashable, float], ...]


@dataclass(frozen=True)
class Incremental:
    """Multiply the odds of treatment levels by declared factors.

    Parameters
    ----------
    delta : float or mapping
        On a treatment with two levels, the odds multiplier of the second sorted level,
        strictly positive.  ``1.0`` is the *natural course* -- the intervention that changes
        nothing, since :math:`q = g` exactly -- whose mean is :math:`E[Y]` and which is the
        usual reference.  Above one shifts the population toward the second level, below
        one away from it.

        On a treatment with more than two levels, and optionally on two, a mapping from a
        treatment label (as the data reports it) to its strictly positive factor
        :math:`\\delta_a`.  The odds of level :math:`a` against level :math:`b` are
        multiplied by :math:`\\delta_a/\\delta_b`.  A level the mapping does not name keeps
        the factor ``1``.  The family is scale free: multiplying every factor by one constant
        changes nothing.  ``{level1: d}`` equals ``d`` on two levels.  The labels of an
        all-categorical vector treatment are the tuples of its observed levels.

        Unlike a :class:`~cleverly.interventions.Shift`'s ``cap``, nothing here has to be
        declared to keep the parameter well defined: :math:`q_\\delta` is a probability for
        any positive factors and any :math:`g`, which is the whole appeal.
    name : str
        What this intervention is called in reported parameter names.  Defaults to
        ``"odds x2.5"`` style for a scalar, ``"odds high x2, low x0.5"`` for a mapping, and
        ``"natural course"`` when every declared factor is ``1``.

    Examples
    --------
    >>> from cleverly.interventions import Incremental
    >>> Incremental(2.0).name
    'odds x2'
    >>> Incremental({"high": 2.0, "low": 0.5}).name
    'odds high x2, low x0.5'
    >>> Incremental({"high": 2.0, "low": 0.5}).factors(("high", "low", "mid")).tolist()
    [2.0, 0.5, 1.0]
    """

    delta: Declared
    name: str = ""

    def __post_init__(self) -> None:
        delta = self.delta
        if isinstance(delta, (Mapping, tuple)):
            pairs = tuple(delta.items()) if isinstance(delta, Mapping) else delta
            if not pairs or any(len(pair) != 2 for pair in pairs):
                raise DataError(
                    "an incremental intervention given a mapping needs at least one "
                    "(label, factor) pair"
                )
            if len({key for key, _ in pairs}) != len(pairs):
                raise DataError(
                    f"an incremental intervention names a level twice: {[k for k, _ in pairs]}"
                )
            for key, value in pairs:
                self._check_factor(value, f"the factor of {key!r}")
            object.__setattr__(self, "delta", tuple((key, float(value)) for key, value in pairs))
            declared = [float(value) for _, value in pairs]
            natural = all(value == 1.0 for value in declared)
            label = ", ".join(f"{key} x{float(value):g}" for key, value in pairs)
            default = "natural course" if natural else f"odds {label}"
        else:
            self._check_factor(delta, "delta")
            default = "natural course" if delta == 1.0 else f"odds x{delta:g}"
        if not self.name:
            object.__setattr__(self, "name", default)

    @staticmethod
    def _check_factor(value: Any, what: str) -> None:
        if not np.isfinite(value):
            raise DataError(f"an incremental intervention's {what} must be finite; got {value!r}")
        if value <= 0.0:
            raise DataError(
                f"an incremental intervention's {what} must be strictly positive; got "
                f"{value!r}. A factor multiplies the odds of treatment, so a factor of 0 "
                "would assign nobody to the level, which is a regime, not a tilt of the "
                "mechanism."
            )

    def factors(self, levels: Sequence[Any], *, subject: str = "the treatment") -> FloatArray:
        """The factor of every level, in the order of ``levels``.

        Parameters
        ----------
        levels : sequence
            The treatment labels in arm-code order, as
            :attr:`~cleverly.data.CausalData.treatment_levels` reports them.
        subject : str
            What the refusals call the treatment.

        Returns
        -------
        ndarray
            ``(K,)``, strictly positive.  A scalar on two levels gives ``(1, delta)``.

        Raises
        ------
        DataError
            If a scalar names a treatment with other than two levels, or a mapping key is
            not a level.
        """
        labels = tuple(levels)
        if not isinstance(self.delta, tuple):
            if len(labels) != 2:
                example = labels[0] if labels else "level"
                raise DataError.from_refusal(
                    Refusal(
                        RefusalReason.INVALID_DECLARATION,
                        f"a scalar odds multiplier names two arms; {subject} has {len(labels)} "
                        f"levels ({list(labels)}). Declare one factor per level, for example "
                        f"Incremental({{{example!r}: 2.0}}), which multiplies the odds of "
                        f"{example!r} against every unnamed level.",
                    )
                )
            return np.array([1.0, float(cast(float, self.delta))])
        position = {label: index for index, label in reversed(list(enumerate(labels)))}
        out = np.ones(len(labels))
        for key, value in self.delta:
            if key not in position:
                raise DataError.from_refusal(
                    Refusal(
                        RefusalReason.INVALID_DECLARATION,
                        f"Incremental names {key!r}, which is not a level of {subject}; "
                        f"its levels are {list(labels)}.",
                    )
                )
            out[position[key]] = float(value)
        return out


@dataclass(frozen=True)
class IPSISet:
    """Every declared tilt, evaluated against the estimated mechanism.

    Holds arrays and no callables, for the reason
    :class:`~cleverly.interventions.RegimeSet` and
    :class:`~cleverly.interventions.PolicySet` do: a fit reached through
    :meth:`~cleverly.estimators.TMLE.retarget` -- the bootstrap, a result loaded from disk
    -- targets the same declared tilts without the mechanism being refit.

    Parameters
    ----------
    names : tuple of str
        One per tilt, in declaration order.  Report labels.
    factors : tuple of tuple of float
        One ``(K,)`` tuple per tilt, in declaration order and arm-code order: the resolved
        factor :math:`\\delta_{r,a}` of every level.  Not the keys of the per-parameter
        arrays: those are the **codes** ``0.0 .. R-1.0``, exactly as
        :class:`~cleverly.interventions.PolicySet` keys by ordinal rather than by factor.
    values : ndarray
        ``(n, K, R)``, :math:`q_{\\delta_r}(a \\mid W_i)` -- the tilted density, columns in
        arm-code order.  Rows sum to one by construction.
    weights : ndarray
        ``(n, K, R)``, :math:`h_{\\delta_r}(a, W_i) = q_{\\delta_r}(a \\mid W_i) /
        g(a \\mid W_i)` -- the clever covariate arm by arm.  Computed in the cancelled
        form :math:`\\delta_a / D` rather than as a ratio, so it never divides by a small
        mechanism; see the module docstring.
    normaliser : ndarray
        ``(n, R)``, :math:`D_{\\delta_r}(W_i) = \\sum_b \\delta_{r,b}\\, g(b \\mid W_i)`.  On
        two levels it is computed in Kennedy's form :math:`\\delta g_1 + g_0` with
        :math:`\\delta = \\delta_1/\\delta_0`, which is :math:`D` divided by :math:`\\delta_0`.
        The tilt is scale free, so :math:`q` and :math:`h` do not depend on that choice.
    propensity : ndarray
        ``(n, K)``, the :math:`g(a \\mid W)` these arrays were built from, **untruncated**.
        Carried because both the influence curve's mechanism term and the mechanism
        fluctuation's offset need it, and because an :class:`IPSISet` recomputed at a
        fluctuated mechanism must be able to say which one it describes.
    reference : float
        Code of the tilt contrasts are taken against.
    mechanism_known : bool
        Whether ``propensity`` is a declared known mechanism
        (:attr:`~cleverly.data.CausalData.known_treatment`) rather than an estimate.  Then
        :math:`q_\\delta(g_0)` is a known stochastic regime: the model has no mechanism
        scores, so the targeting step solves no mechanism equation and the influence curve
        has no mechanism term.  That curve is the curve of the
        :class:`~cleverly.interventions.Stochastic` regime at the same density (Kennedy 2019,
        the incremental curve projected off the mechanism tangent space).
    """

    names: tuple[str, ...]
    factors: tuple[tuple[float, ...], ...]
    values: FloatArray
    weights: FloatArray
    normaliser: FloatArray
    propensity: FloatArray
    reference: float = 0.0
    mechanism_known: bool = False

    def __post_init__(self) -> None:
        r = len(self.names)
        if len(self.factors) != r:
            raise ValueError(f"{r} tilt names but {len(self.factors)} factor tuples")
        if len(set(self.names)) != r:
            raise ValueError(f"tilt names must be distinct; got {list(self.names)}")
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3 or values.shape[2] != r:
            raise ValueError(
                f"tilted densities must be (n, K, {r}) for tilts {list(self.names)}; "
                f"got shape {values.shape}"
            )
        n, k = values.shape[0], values.shape[1]
        if any(len(entry) != k for entry in self.factors):
            raise ValueError(f"every tilt needs {k} factors, one per level; got {self.factors}")
        for name, array, shape in (
            ("weights", self.weights, (n, k, r)),
            ("normaliser", self.normaliser, (n, r)),
            ("propensity", self.propensity, (n, k)),
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

        Parameters
        ----------
        incrementals : sequence of Incremental
            The tilts to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data, which supplies the arm order and the labels.
        propensity : ndarray
            ``(n, K)`` estimated mechanism, columns in arm-code order, untruncated.
        reference : str or None
            Label of the tilt contrasts are taken against. ``None`` uses the first.

        Returns
        -------
        IPSISet
            Every tilt evaluated against that mechanism.
        """
        if not incrementals:
            raise DataError("at least one incremental intervention is required")
        refuse_continuous_tilt(data)
        names = tuple(str(item.name) for item in incrementals)
        if len(set(names)) != len(names):
            raise DataError(f"incremental intervention names must be distinct; got {list(names)}")
        subject = repr(data.treatment_name)
        factors = tuple(
            tuple(float(value) for value in item.factors(data.treatment_levels, subject=subject))
            for item in incrementals
        )

        arms = data.arm_codes
        g = np.asarray(propensity, dtype=float)
        if g.ndim != 2 or g.shape[1] != len(arms):
            raise DataError(
                f"propensity must be (n, {len(arms)}) with columns in arm-code order; "
                f"got shape {g.shape}"
            )
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(
                    f"reference={reference!r} is not one of the incremental interventions "
                    f"{list(names)}"
                )
            code = float(names.index(reference))
        return cls(*_tilt(names, factors, g), reference=code)

    def at(self, propensity: FloatArray) -> IPSISet:
        """The same tilts recomputed at a different ``(n, K)`` mechanism.

        What the mechanism fluctuation calls after each update.  The declared tilts are
        unchanged -- a tilt is a statement about odds, not about a particular :math:`g` --
        so only the evaluated arrays move.

        Parameters
        ----------
        propensity : ndarray
            ``(n, K)`` mechanism :math:`g(a \\mid W)` to recompute the tilts at.

        Returns
        -------
        IPSISet
            The same declared tilts, evaluated at the new mechanism.
        """
        g = np.asarray(propensity, dtype=float)
        if g.shape != (self.n, self.n_arms):
            raise ValueError(f"expected a ({self.n}, {self.n_arms}) mechanism, got {g.shape}")
        return type(self)(
            *_tilt(self.names, self.factors, g),
            reference=self.reference,
            mechanism_known=self.mechanism_known,
        )

    # ----------------------------------------------------------------- access

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.values.shape[0])

    @property
    def n_arms(self) -> int:
        """Return the number of treatment arms."""
        return int(self.values.shape[1])

    @property
    def n_tilts(self) -> int:
        """Return the number of incremental interventions."""
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
        """Return the label for one indexed policy.

        Parameters
        ----------
        code : float
            Tilt code.

        Returns
        -------
        str
            The label the tilt was declared under.
        """
        return self.labels[float(code)]

    def observed(self, treatment: FloatArray) -> FloatArray:
        """``(n, R)`` -- the clever covariate at the arm each unit actually received.

        Parameters
        ----------
        treatment : ndarray
            ``(n,)`` observed treatment, in arm codes.

        Returns
        -------
        ndarray
            ``(n, R)`` clever covariate at the arm each unit actually received.
        """
        a = np.asarray(treatment, dtype=float).reshape(-1)
        indicator = np.column_stack([(a == arm) for arm in range(self.n_arms)]).astype(float)
        return np.asarray(np.einsum("ij,ijr->ir", indicator, self.weights), dtype=float)

    @property
    def binary_derivative(self) -> FloatArray:
        """``(n, R)`` -- :math:`\\delta_r / D_{\\delta_r}^2` for a treatment with two levels.

        The factor that multiplies :math:`\\{\\bar Q(1, W) - \\bar Q(0, W)\\}(A - g)` in the
        binary influence curve, with :math:`\\delta_r = \\delta_{r,1}/\\delta_{r,0}`.  The
        three-level and larger forms read :attr:`weights` instead.

        Returns
        -------
        ndarray
            ``(n, R)`` factor for each tilt.

        Raises
        ------
        ValueError
            If the treatment does not have two levels.
        """
        if self.n_arms != 2:
            raise ValueError(f"binary_derivative needs two levels; the tilt has {self.n_arms}")
        delta = np.array([entry[1] / entry[0] for entry in self.factors])
        return np.asarray(delta[None, :] / np.asarray(self.normaliser, dtype=float) ** 2)

    def subset(self, index: Any) -> IPSISet:
        """The same tilts on a row subset -- a fold, a bootstrap resample.

        Sliced rather than re-evaluated, for the reason
        :meth:`~cleverly.interventions.RegimeSet.subset` gives: re-deriving the tilt on a
        resample would let the resample redefine the estimand.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        IPSISet
            The same tilts over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            values=self.values[idx],
            weights=self.weights[idx],
            normaliser=self.normaliser[idx],
            propensity=self.propensity[idx],
        )


def _tilt(
    names: tuple[str, ...], factors: tuple[tuple[float, ...], ...], g: FloatArray
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float, ...], ...],
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """The whole arithmetic of the tilt, for an ``(n, K)`` mechanism.

    Factored out because :meth:`IPSISet.evaluate` and :meth:`IPSISet.at` must produce the
    same arrays from the same mechanism -- the alternating fluctuation calls the second
    once per iteration, and a drift between the two would be invisible until the score
    stopped closing.

    Two levels dispatch to :func:`_tilt_binary`, which keeps Kennedy's arithmetic on
    :math:`g_1` with :math:`g_0 = 1 - g_1`, the complement, so the tilted density sums to one
    exactly and the binary arrays are bit-identical to the ones the binary path produced
    before factors existed.  Three or more levels take :func:`_tilt_general`.
    """
    build = _tilt_binary if g.shape[1] == 2 else _tilt_general
    values, weights, normaliser = build(factors, g)
    return names, factors, values, weights, normaliser, g


def _tilt_binary(
    factors: tuple[tuple[float, ...], ...], g: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(values, weights, normaliser)`` of the tilts of a two-level mechanism."""
    values, weights, normaliser = [], [], []
    g1 = g[:, 1]
    g0 = 1.0 - g1
    for entry in factors:
        delta = entry[1] / entry[0]
        d = delta * g1 + g0
        # q_1 = delta*g1/D and q_0 = g0/D, which sum to one exactly.
        values.append(np.column_stack([g0 / d, delta * g1 / d]))
        # h_a = q_a / g_a with the mechanism cancelled: delta/D at arm 1, 1/D at arm 0.
        # Written this way there is no division by a small propensity anywhere, which is
        # the arithmetic content of "this estimand needs no positivity assumption".
        weights.append(np.column_stack([1.0 / d, np.full_like(d, delta) / d]))
        normaliser.append(d)
    return np.stack(values, axis=2), np.stack(weights, axis=2), np.column_stack(normaliser)


def _tilt_general(
    factors: tuple[tuple[float, ...], ...], g: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(values, weights, normaliser)`` of the tilts of a mechanism with any ``K`` levels."""
    values, weights, normaliser = [], [], []
    for entry in factors:
        f = np.asarray(entry, dtype=float)
        d = g @ f
        values.append(g * f[None, :] / d[:, None])
        weights.append(np.broadcast_to(f[None, :], g.shape) / d[:, None])
        normaliser.append(d)
    return np.stack(values, axis=2), np.stack(weights, axis=2), np.column_stack(normaliser)


# ------------------------------------------------------------------ diagnostics


@dataclass(frozen=True)
class IncrementalSupport:
    """Overlap for one tilt -- which for this estimand is a statement, not a warning.

    The clever covariate is bounded by :math:`\\min\\delta/\\max\\delta` and
    :math:`\\max\\delta/\\min\\delta` however small the mechanism gets, so ``guaranteed`` is
    what the tilt promises before seeing any data and ``max_ratio`` is what it delivered.
    The two agreeing is the normal case; the report exists so that a reader can see the
    effective sample size stay near :math:`n` where an arm-indexed fit's would have
    collapsed.

    This row is hashable and its two siblings are not, and that difference is deliberate.
    :class:`~cleverly.interventions.RegimeSupport` and
    :class:`~cleverly.interventions.PolicySupport` carry quantile mappings that are part of
    what makes one of their rows different from another, so neither has ever had a hash.
    This row's identity is the declared tilt and what the data did with it, all of which
    was hashable until :attr:`score_load` was added.  That field holds a ``dict``, so
    adding it made every *fitted* row of this class raise ``TypeError: unhashable type:
    'dict'`` while an omitted row, whose field is ``None``, still hashed.  Declaring the
    field ``hash=False`` restores the hash rather than preserving one.  ``compare=False``
    would have gone too far: two tilts that differ only in their fitted score load are
    different rows and still compare unequal, and only the ``dict`` leaves the hash.

    Parameters
    ----------
    name : str
        Report label of the tilt.
    delta : tuple of tuple
        ``(label, factor)`` for every level, in arm-code order: the resolved factors this
        row describes.
    guaranteed : tuple of float
        ``(min factor / max factor, max factor / min factor)``: the bounds the clever
        covariate cannot leave, whatever the mechanism does.  A property of the declared
        tilt alone: no data went into it.
    min_propensity : float
        Smallest estimated :math:`g(a \\mid W)` over rows and levels, for comparison with an
        arm-indexed fit rather than because this estimand divides by it.
    max_ratio : float
        Largest clever covariate the data produced.
    effective_sample_size : float
        Kish effective sample size of those weights.
    ess_ratio : float
        That size as a share of ``n``.
    score_load : _InterventionLoadRow or None
        Concentration of the exact absolute score weights retained for this tilt's
        outcome equation, and the cross-fitting draw it describes. ``None`` means the artifact
        did not supply a usable column. The twelve keys are ``equation``, ``n_total``,
        ``n_targeted``, ``effective``, ``targeted_ratio``, ``total_ratio``, ``top_1pct``,
        ``top_5pct``, ``max_load``, ``zero_load``, ``reported_repeat`` and ``n_repeats``.
    score_load_omission : str or None
        Machine-readable reason why :attr:`score_load` is unavailable.
    """

    name: str
    delta: tuple[tuple[str, float], ...]
    guaranteed: tuple[float, float]
    min_propensity: float
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float
    #: Kept out of the generated ``__hash__`` and left in ``__eq__``, which is what gives
    #: this class its hash back.  The class docstring says what adding the field broke and
    #: why the siblings are unhashable for a different reason.
    score_load: _InterventionLoadRow | None = field(default=None, hash=False)
    score_load_omission: str | None = None

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per row of the report.
        """
        low, high = self.guaranteed
        score = format_score_load(self.score_load, style="inline")
        return (
            f"{self.name}: min g={self.min_propensity:.3g}, "
            f"covariate in [{low:.3g}, {high:.3g}] by construction, "
            f"max={self.max_ratio:.3g}, "
            f"ESS={self.effective_sample_size:.0f} ({self.ess_ratio:.1%} of n), {score}"
        )


def check_incremental_support(
    tilts: IPSISet,
    treatment: FloatArray,
    *,
    levels: Sequence[Any],
    absolute_score_weights: FloatArray | None = None,
    equations: tuple[str, ...] = (),
    n_repeats: int = 1,
) -> dict[str, IncrementalSupport]:
    """Per-tilt overlap, in the vocabulary the other two axes' reports use.

    Reports the *bound* alongside the realised maximum because for this estimand the bound
    is the interesting number: it holds whatever the mechanism does, which is what
    distinguishes the tilt from every other intervention here.  An arm-indexed fit on the
    same data would divide by ``min_propensity``; this one never does.

    Parameters
    ----------
    tilts : IPSISet
        The evaluated tilts to report on.
    treatment : ndarray
        ``(n,)`` observed treatment, in arm codes.
    levels : sequence
        The treatment labels in arm-code order, which key the reported factors.
    absolute_score_weights : ndarray or None
        Fitted ``abs(w_i * H_ij)`` columns, in tilt order. ``None`` records an omission.
    equations : tuple of str
        Fitted outcome score-equation names, in tilt order.
    n_repeats : int
        Number of stored cross-fitting draws. The retained weights describe draw 1.

    Returns
    -------
    dict of str to IncrementalSupport
        One record per tilt, keyed by its report label.
    """
    observed = tilts.observed(treatment)
    n = observed.shape[0]
    labels = tuple(tilts.names)
    score_loads, load_omission = _intervention_loads(
        labels, absolute_score_weights, equations, n, n_repeats
    )
    out: dict[str, IncrementalSupport] = {}
    for index, name in enumerate(tilts.names):
        factors = tilts.factors[index]
        column = observed[:, index]
        ess = effective_sample_size(column, on_degenerate=0.0)
        out[name] = IncrementalSupport(
            name=name,
            delta=tuple((str(level), float(f)) for level, f in zip(levels, factors, strict=True)),
            guaranteed=(min(factors) / max(factors), max(factors) / min(factors)),
            min_propensity=float(np.min(tilts.propensity)) if n else 0.0,
            max_ratio=float(column.max()) if column.size else 0.0,
            effective_sample_size=ess,
            ess_ratio=ess / n if n else 0.0,
            score_load=score_loads.get(name),
            score_load_omission=load_omission,
        )
    return out
