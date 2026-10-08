r"""Modified treatment policies: a known map of the treatment a unit received.

A :mod:`regime <cleverly.interventions.base>` sets the treatment to an arm, or spreads
mass over the arms, as a function of :math:`W` alone.  A **modified treatment policy**
reads the treatment a unit actually received and moves it.  The additive shift is the
first example:

.. math::

    d_\delta(a, w) = \begin{cases}
        a + \delta & a + \delta \le u(w) \\
        a          & \text{otherwise,}
    \end{cases}
    \qquad
    \Psi_\delta = E\bigl[\bar Q\bigl(d_\delta(A, W), W\bigr)\bigr].

"Everyone's exposure rises by :math:`\delta`, except where that would take them past what
is achievable."  Identification requires the policy to preserve the conditional treatment support.
An upper cap alone does not guarantee this: negative shifts can cross the lower boundary,
and either sign can enter a support gap (Diaz & van der Laan 2018; Haneuse & Rotnitzky 2013).

The policy classes are :class:`Shift` (additive), :class:`Scale` (multiplicative),
:class:`Piecewise` (a shift or scale per interval), :class:`ModifiedPolicy` (declared by the
analyst, with :class:`Piece` objects on a continuous treatment and a label map on a
categorical one, optionally randomized by a finite :class:`Randomizer`) and
:class:`RiskRatioTilt` (``lmtp::ipsi``).  Every one is a known function, fixed before the
fit, so the estimand is the one of Díaz, Williams, Hoffman and Schenck (2023), Theorem 2,
and at one time point that of Díaz and van der Laan (2012), Result 1.

**The clever covariate.**  For a policy :math:`d`, evaluated at an arbitrary treatment
value :math:`a`, the covariate is the ratio :math:`g^d(a \mid W) / g(a \mid W)` of the
density the policy induces to the observed one.  On a continuous treatment the induced
density is Equation (3) of Díaz et al. (2023): with pieces :math:`I_j` on which
:math:`d` is strictly monotone with inverse :math:`b_j`,

.. math::

    g^d(a \mid w) = \sum_j \mathbb 1\{b_j(a, w) \in I_j\}\,
        g\bigl(b_j(a, w) \mid w\bigr)\,\bigl|b_j'(a, w)\bigr| .

On a categorical treatment it is the discrete formula of the same section,
:math:`g^d(a \mid w) = \sum_s \mathbb 1\{d(s, w) = a\}\, g(s \mid w)`.  For the shift
:math:`r` the first formula reads

.. math::

    h_r(a, W) = \frac{g(a - \delta_r \mid W)}{g(a \mid W)}\,\mathbb 1\{a \le u_r\}
                + \mathbb 1\{a > u_r - \delta_r\}

Both indicators come from that preimage and neither is decoration: a unit lands
at :math:`b` either by *being shifted there* from :math:`b - \delta` (possible only when
:math:`b \le u`, since otherwise the shift from :math:`b - \delta` was itself held back)
or by *staying put* (when :math:`b + \delta > u`).  An identity piece contributes the
literal indicator, added after the zero-density guard, so a row whose estimated density is
zero keeps the identity term.

The first indicator is invisible whenever the cap sits at or above the largest treatment
value, which is the common case -- and that is exactly why
``tests/discrete_law_shift.py`` declares **two** caps.  Dropping it passed every check
under a loose cap and failed the Gateaux derivative under a tight one.

Three checks it must pass, and does: at :math:`\delta = 0` it is identically one and the
influence curve collapses to :math:`Y - \Psi`, which is the influence curve of
:math:`E[Y]`; at a :math:`\delta` so large that nobody can move it is identically one
again; and on doses :math:`\{0,1,2,3\}` with :math:`\delta = 1` it gives
:math:`0,\ g(0)/g(1),\ g(1)/g(2),\ g(2)/g(3) + 1` when :math:`u = 3` and
:math:`0,\ g(0)/g(1),\ (g(1)+g(2))/g(2),\ 1` when :math:`u = 2`.

**A randomized policy** :math:`d(a, w, \varepsilon)` draws a randomizer of finite known law
(Díaz et al. 2023, Section 2).  Its mean is linear in that law, so :class:`PolicySet`
expands it into one deterministic *component* per randomizer value and reports the weighted
sum of the component means and curves.  A deterministic policy is its own single component,
and its arrays are the ones it had before randomized policies existed.

**Why an MTP is not the stochastic regime that induces it.**  The two parameters agree --

.. math::

    E[\bar Q(d(A,W), W)]
      = E_W\Bigl[\sum_a g(a \mid W)\, \bar Q(d(a,W), W)\Bigr]
      = E_W\Bigl[\sum_b g^d(b \mid W)\, \bar Q(b, W)\Bigr]

-- and the clever covariates agree entry for entry.  The *influence curves do not*: a
regime's plug-in term is :math:`\sum_b g^d(b \mid W) \bar Q(b, W)`, a function of
:math:`W`, while an MTP's is :math:`\bar Q(d(A,W), W)`, which reads the treatment the unit
actually received.  They agree only in conditional expectation given :math:`W`.  See
:func:`~cleverly.inference.influence.policy_means` for the variance identity this implies
and for why the two paths must not share an implementation.

**Why this needs no second fluctuation and an incremental intervention does.**  Both have
a :math:`g^\star` involving :math:`g`, which is exactly why the resemblance is dangerous,
and the distinction got *more* useful once both were implemented rather than less.  An
:class:`~cleverly.interventions.Incremental` intervention *defines its intervention
through* :math:`g`: its :math:`q_\delta` is a functional of :math:`P`, so
:math:`\Psi(\delta)` mentions the mechanism, the efficient influence function carries a
further term for the pathwise derivative through it (Kennedy 2019), and the estimator has
to fluctuate :math:`g` as well as :math:`\bar Q`.  A policy's :math:`d(a, w)` is a known
function and :math:`\Psi_d` above mentions no mechanism at all; the induced density moving
with :math:`P` is precisely what the plug-in-at-the-observed-row term already accounts for.
So: a policy's :math:`g` is a nuisance, an incremental intervention's is half of the
estimand.  :class:`RiskRatioTilt` reads :math:`g` only through its induced density, and
its law does not depend on :math:`P` (Hoffman et al. 2024, Example 8), so it is a policy
in this sense and not a tilt of the mechanism.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Literal, TypeAlias

import numpy as np

from .._declarations import FunctionDeclaration, FunctionKind
from .._typing import BoolArray, FloatArray, IntArray
from ..data.causal_data import CausalData
from ..data.weighting import effective_sample_size, format_score_load
from ..exceptions import CapabilityError, DataError, PositivityWarning
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity, warn_if_unresolved
from .support import _intervention_loads, _InterventionLoadRow

__all__ = [
    "POLICY_TYPES",
    "ModifiedPolicy",
    "Piece",
    "Piecewise",
    "Policy",
    "PolicySet",
    "PolicySupport",
    "Randomizer",
    "RiskRatioTilt",
    "Scale",
    "Shift",
    "check_policy_support",
    "refuse_policy_declarations",
]

_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)

#: Relative tolerance of the declared-inverse check.
_INVERSE_TOLERANCE = 1e-8

#: Relative step of the central difference that checks a declared derivative, and the
#: relative gap it allows.  A central difference of a smooth map errs by O(step^2), far
#: below the tolerance, and a wrong slope (a factor, a sign of the reciprocal) is far above.
_DERIVATIVE_STEP = 1e-5
_DERIVATIVE_TOLERANCE = 1e-4

#: Grid points per piece and row on which a declared map is checked to be monotone.
_MONOTONE_GRID = 9

#: How far the share of rows sent to one dose may exceed the largest share that one
#: observed dose holds before a declared continuous policy is refused as a collapse.
_TIE_EXCESS = 0.01

#: The declaration of a user-written policy, as ``rule_kind`` declares a rule.
_POLICY_DECLARATION = FunctionDeclaration(
    field="policy_kind",
    meaning="It declares whether the modified treatment policy d(a, h) is a known function.",
    undeclared=(
        "a modified treatment policy d(a, h) can close over any estimate, and no code can "
        "inspect a closure, so its status is a declaration. Declare "
        "ModifiedPolicy(..., policy_kind='known') when d is fixed before the fit and chosen "
        "without reading the analysis sample"
    ),
    estimated=(
        "A policy fitted to the analysis sample makes the estimand data-dependent; Díaz et "
        "al. (2023) Theorem 2 assumes d does not depend on P. A policy that tilts the "
        "fitted mechanism is docs/roadmap.md X19"
    ),
)


# ------------------------------------------------------------- policy objects


@dataclass(frozen=True)
class Shift:
    """Add ``delta`` to everyone's treatment, up to ``cap``.

    Parameters
    ----------
    delta : float
        How far to move the treatment.  ``0.0`` is the *natural course* -- the policy that
        changes nothing -- whose mean is :math:`E[Y]` and which is the usual reference.
    cap : float or None
        The largest treatment value the policy will assign; a unit whose shifted dose would
        exceed it keeps its own.  **Required, with no default**, and it is a declaration
        rather than something estimated.

        Estimating :math:`u(w)` from the data would make the *parameter* data-dependent:
        the reported standard error would condition on a fitted support boundary, and every
        bootstrap replicate would target a slightly different policy.  Defaulting it to
        ``max(A)`` is worse, pinning the estimand to an extreme order statistic.  So the
        analyst says what dose is achievable, which is a question about the world.

        ``cap=None`` means no cap.  It is allowed, it is the same arithmetic with the
        indicator zeroed, and it warns with the share of rows whose shifted dose leaves the
        observed range -- because there :math:`\\bar Q` is being extrapolated and
        identification needs :math:`A + \\delta` to be supported.
    name : str
        What this policy is called in reported parameter names.  Defaults to ``"+0.5"`` /
        ``"-1"`` style, with ``"natural course"`` for ``delta=0``.
    """

    delta: float
    cap: float | None
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta):
            raise DataError(f"a shift's delta must be finite; got {self.delta!r}")
        if self.cap is not None and not np.isfinite(self.cap):
            raise DataError(f"a shift's cap must be finite or None; got {self.cap!r}")
        if not self.name:
            default = "natural course" if self.delta == 0.0 else f"{self.delta:+g}"
            object.__setattr__(self, "name", default)

    def apply(self, treatment: FloatArray) -> tuple[FloatArray, BoolArray]:
        """``(shifted, capped)`` -- the assigned dose, and which rows the cap held back.

        Parameters
        ----------
        treatment : ndarray
            ``(n,)`` observed dose.

        Returns
        -------
        shifted : ndarray
            ``(n,)`` dose the policy assigns.
        capped : ndarray
            ``(n,)`` boolean, true where the cap held the row at its own dose.
        """
        a = np.asarray(treatment, dtype=float).reshape(-1)
        moved = np.asarray(a + self.delta, dtype=float)
        if self.cap is None:
            return moved, np.zeros(a.size, dtype=bool)
        held = np.asarray(moved > float(self.cap), dtype=bool)
        return np.asarray(np.where(held, a, moved), dtype=float), held


@dataclass(frozen=True)
class Scale:
    """Multiply everyone's dose by ``factor``, up to ``cap``.

    :math:`d(a) = c\\,a` where :math:`c\\,a \\le u`, and :math:`a` elsewhere.  The map is
    strictly increasing for :math:`c > 0`, its inverse is :math:`a / c` and the derivative
    of the inverse is :math:`1 / c`, so the density ratio is

    .. math::

        h(a, W) = \\frac{g(a / c \\mid W)}{c\\, g(a \\mid W)}\\,\\mathbb 1\\{a \\le u\\}
                  + \\mathbb 1\\{c\\, a > u\\}.

    Parameters
    ----------
    factor : float
        The multiplier.  It is positive; ``1.0`` is the natural course.
    cap : float or None
        The largest dose the policy assigns.  A unit whose scaled dose would exceed it
        keeps its own, as for :class:`Shift`.
    name : str
        The report label.  Defaults to ``"x1.25"`` style, with ``"natural course"`` for
        ``factor=1``.
    """

    factor: float
    cap: float | None
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.factor) or self.factor <= 0.0:
            raise DataError(
                f"a scale's factor must be finite and positive; got {self.factor!r}. A factor "
                "of zero maps every dose to one point, which is not pathwise differentiable "
                "on a continuous treatment (Díaz et al. 2023, Section 4)"
            )
        if self.cap is not None and not np.isfinite(self.cap):
            raise DataError(f"a scale's cap must be finite or None; got {self.cap!r}")
        if not self.name:
            default = "natural course" if self.factor == 1.0 else f"x{self.factor:g}"
            object.__setattr__(self, "name", default)

    def apply(self, treatment: FloatArray) -> tuple[FloatArray, BoolArray]:
        """``(assigned, held)``: the assigned dose, and which rows the cap held back.

        Parameters
        ----------
        treatment : ndarray
            ``(n,)`` observed dose.

        Returns
        -------
        assigned : ndarray
            ``(n,)`` dose the policy assigns.
        held : ndarray
            ``(n,)`` boolean, true where the cap held the row at its own dose.
        """
        a = np.asarray(treatment, dtype=float).reshape(-1)
        moved = np.asarray(a * self.factor, dtype=float)
        if self.cap is None:
            return moved, np.zeros(a.size, dtype=bool)
        held = np.asarray(moved > float(self.cap), dtype=bool)
        return np.asarray(np.where(held, a, moved), dtype=float), held


@dataclass(frozen=True)
class Piece:
    """One interval of a declared policy and the strictly monotone map it applies there.

    Assumption 4 of Díaz, Williams, Hoffman and Schenck (2023, Section 4) asks a policy on
    a continuous treatment to be piecewise smooth invertible: the dose range splits into
    intervals :math:`I_j(h)`, and on each one the policy is a strictly monotone smooth map
    :math:`d_j` with inverse :math:`b_j`.  The density the policy induces is then their
    Equation (3), which reads :math:`b_j` and :math:`b_j'`.

    Parameters
    ----------
    lower : float or callable
        The lower end of the interval, or a function of the policy frame that returns one
        value per row.
    upper : float or callable
        The upper end, in the same form.
    map : callable or None
        ``map(a, h)``, the dose the policy assigns on this piece.  ``None`` marks an
        identity piece, which leaves the dose unchanged.
    inverse : callable or None
        ``inverse(b, h)``, the dose on this piece that the map sends to ``b``.  Required
        unless the piece is an identity piece.
    derivative : callable or None
        ``derivative(b, h)``, the derivative of ``inverse`` with respect to ``b``.
        Required unless the piece is an identity piece.
    closed : {"left", "right"}
        Which end of the interval belongs to it: ``"left"`` is ``[lower, upper)`` and
        ``"right"`` is ``(lower, upper]``.

    Attributes
    ----------
    is_identity : bool
    """

    lower: float | Callable[[Any], Any]
    upper: float | Callable[[Any], Any]
    map: Callable[[Any, Any], Any] | None = None
    inverse: Callable[[Any, Any], Any] | None = None
    derivative: Callable[[Any, Any], Any] | None = None
    closed: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        if self.closed not in ("left", "right"):
            raise DataError(f"a piece is closed 'left' or 'right'; got {self.closed!r}")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if not callable(value):
                number = float(value)
                if np.isnan(number):
                    raise DataError(f"a piece's {name} bound must not be NaN")
                object.__setattr__(self, name, number)
        if self.map is not None and (self.inverse is None or self.derivative is None):
            raise DataError(
                "a piece that moves the dose declares map, inverse and derivative. The "
                "density ratio of Díaz et al. (2023), Equation (3), reads the inverse and "
                "its derivative"
            )

    @property
    def is_identity(self) -> bool:
        """Whether the piece leaves the dose unchanged."""
        return self.map is None


@dataclass(frozen=True)
class Piecewise:
    """A policy that applies a different shift or scale on each interval of the dose.

    Hoffman, Salazar-Barreto, Williams, Rudolph and Díaz (2024, Example 5) shorten only the
    long surgeries.  Each piece holds an uncapped :class:`Shift` or :class:`Scale`, and the
    density ratio is Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) summed over
    the pieces, with the derivative :math:`1 / c` of each scaled piece's inverse.

    .. code-block:: python

        Piecewise(((-inf, 60.0, Shift(0.0, None)), (60.0, inf, Shift(-15.0, None))))

    Parameters
    ----------
    pieces : tuple of tuple
        ``(lower, upper, map)`` per piece, where ``map`` is ``Shift(delta, None)`` or
        ``Scale(factor, None)``.  The intervals do not overlap.  A dose in no interval keeps
        its value, and the density ratio counts it as an identity piece.
    closed : {"left", "right"}
        Which end of every interval belongs to it.
    name : str
        The report label.
    """

    pieces: tuple[tuple[float, float, Shift | Scale], ...]
    closed: Literal["left", "right"] = "left"
    name: str = ""

    def __post_init__(self) -> None:
        if self.closed not in ("left", "right"):
            raise DataError(f"a piecewise policy is closed 'left' or 'right'; got {self.closed!r}")
        raw = tuple(tuple(piece) for piece in self.pieces)
        if not raw:
            raise DataError("a piecewise policy needs at least one piece")
        checked: list[tuple[float, float, Shift | Scale]] = []
        for entry in raw:
            if len(entry) != 3:
                raise DataError(f"a piece is (lower, upper, map); got {entry!r}")
            lower, upper, mapping = entry
            assert not isinstance(lower, (Shift, Scale)) and not isinstance(upper, (Shift, Scale))
            if not isinstance(mapping, (Shift, Scale)) or mapping.cap is not None:
                raise DataError(
                    "each piece of a Piecewise policy maps its interval by Shift(delta, None) "
                    f"or Scale(factor, None); got {mapping!r}. The pieces themselves bound "
                    "the policy"
                )
            low, high = float(lower), float(upper)
            if not low < high:
                raise DataError(f"a piece needs lower < upper; got ({low:g}, {high:g})")
            checked.append((low, high, mapping))
        checked.sort(key=lambda piece: piece[0])
        for (_, high, _), (low, _, _) in pairwise(checked):
            if low < high:
                raise DataError("the pieces of a Piecewise policy overlap")
        object.__setattr__(self, "pieces", tuple(checked))
        if not self.name:
            label = ", ".join(mapping.name for _, _, mapping in checked)
            object.__setattr__(self, "name", f"piecewise({label})")


@dataclass(frozen=True)
class Randomizer:
    """A finite, known law for the randomizer of a randomized policy.

    Díaz, Williams, Hoffman and Schenck (2023, Section 2) admit a policy
    :math:`d(a, h, \\varepsilon)` whose randomizer :math:`\\varepsilon` is drawn
    independently across units and of everything else, from a law that does not depend on
    the observed-data distribution.  The estimator integrates it out: the policy mean is
    :math:`\\sum_e p(e)\\,E[\\bar Q(d(A, H, e), H)]`, linear in the known weights.

    Parameters
    ----------
    values : tuple
        The values the randomizer takes.  Each one names a branch of the policy.
    probabilities : tuple of float
        The probability of each value, in the order of ``values``.  They are positive and
        sum to one.
    """

    values: tuple[object, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        probabilities = tuple(float(p) for p in self.probabilities)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probabilities)
        if not values or len(values) != len(probabilities):
            raise DataError(
                f"a randomizer needs one probability per value; got {len(values)} value(s) "
                f"and {len(probabilities)} probabilities"
            )
        if len(set(values)) != len(values):
            raise DataError(f"randomizer values must be distinct; got {list(values)!r}")
        array = np.asarray(probabilities, dtype=float)
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise DataError(f"randomizer probabilities must be positive; got {array.tolist()!r}")
        if abs(float(array.sum()) - 1.0) > 1e-12:
            raise DataError(
                f"randomizer probabilities must sum to one; they sum to {float(array.sum())!r}"
            )


@dataclass(frozen=True)
class ModifiedPolicy:
    """A modified treatment policy that the analyst declares.

    On a **continuous** treatment the policy is a tuple of :class:`Piece` objects, as
    Assumption 4 of Díaz, Williams, Hoffman and Schenck (2023) requires.  The fit checks on
    the observed doses that the pieces partition the range, that each map is strictly
    monotone, and that each declared inverse inverts its map.  On a **categorical**
    treatment the policy is ``apply(a, h)``, which maps each unit's label to a label, and
    the density ratio is the discrete formula of the same paper.  A categorical node whose
    labels are numbers also takes pieces, applied to the label values.

    A **randomized** policy adds a :class:`Randomizer` with finitely many values.  Then
    ``pieces`` maps each value to its pieces, or ``apply`` takes a third argument, the
    value.

    .. code-block:: python

        ModifiedPolicy(
            "halve above 4",
            pieces=(
                Piece(-np.inf, 4.0),
                Piece(4.0, np.inf, lambda a, h: a / 2, lambda b, h: 2 * b, lambda b, h: 2.0),
            ),
            policy_kind="known",
        )

    Parameters
    ----------
    name : str
        The report label.
    pieces : tuple of Piece, mapping, or None
        The pieces, or a mapping from randomizer value to that value's pieces.
    apply : callable or None
        ``apply(a, h)``, or ``apply(a, h, e)`` with a randomizer, on a categorical
        treatment.  It returns one label per row.
    randomizer : Randomizer or None
        The known law of the randomizer.
    policy_kind : {"known", "estimated"} or None
        The declaration that the policy is a known function.  A fit accepts ``"known"``
        only.
    """

    name: str
    pieces: tuple[Piece, ...] | Mapping[object, tuple[Piece, ...]] | None = None
    apply: Callable[..., Any] | None = None
    randomizer: Randomizer | None = None
    policy_kind: FunctionKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataError("a modified policy needs a non-empty name")
        _POLICY_DECLARATION.check(self.policy_kind)
        if self.randomizer is not None and not isinstance(self.randomizer, Randomizer):
            raise CapabilityError(
                f"policy {self.name!r} draws a continuous randomizer. The estimator "
                "integrates a randomizer with finitely many values of known probability; "
                "discretize it. runif() < delta is a two-point randomizer and is accepted: "
                "Randomizer((True, False), (delta, 1 - delta))"
            )
        if (self.pieces is None) == (self.apply is None):
            raise DataError(
                f"policy {self.name!r} declares exactly one of pieces= and apply=: pieces on "
                "a continuous or numeric node, apply on a categorical one"
            )
        if self.apply is not None and not callable(self.apply):
            raise DataError(f"policy {self.name!r}: apply= must be callable")
        if self.pieces is None:
            return
        if self.randomizer is None:
            if isinstance(self.pieces, Mapping):
                raise DataError(
                    f"policy {self.name!r} maps randomizer values to pieces but declares no "
                    "randomizer"
                )
            object.__setattr__(self, "pieces", _check_pieces(self.name, self.pieces))
            return
        if not isinstance(self.pieces, Mapping) or set(self.pieces) != set(self.randomizer.values):
            raise DataError(
                f"policy {self.name!r} draws a randomizer, so pieces= maps each of its values "
                f"{list(self.randomizer.values)!r} to that value's pieces"
            )
        object.__setattr__(
            self,
            "pieces",
            {
                value: _check_pieces(self.name, self.pieces[value])
                for value in self.randomizer.values
            },
        )


def _check_pieces(name: str, pieces: Any) -> tuple[Piece, ...]:
    """The pieces as a tuple of :class:`Piece`, or a refusal naming the policy."""
    if isinstance(pieces, Piece):
        pieces = (pieces,)
    checked = tuple(pieces)
    if not checked or not all(isinstance(piece, Piece) for piece in checked):
        raise DataError(f"policy {name!r}: pieces= holds Piece objects, one per interval")
    return checked


@dataclass(frozen=True)
class RiskRatioTilt:
    """The risk-ratio incremental intervention of ``lmtp::ipsi``, a randomized policy.

    For ``delta < 1`` each treated unit keeps its treatment with probability ``delta`` and
    is otherwise untreated, so :math:`P(A^d = 1 \\mid h) = \\delta\\, g(1 \\mid h)`.  For
    ``delta > 1`` each untreated unit stays untreated with probability ``1 / delta`` and is
    otherwise treated, so :math:`P(A^d = 0 \\mid h) = g(0 \\mid h) / \\delta`.  ``delta = 1``
    is the natural course.  The policy draws a two-point randomizer of known law, as
    ``lmtp``'s ``ipsi_down`` and ``ipsi_up`` do with ``runif(n) < delta``, so it is a
    randomized modified treatment policy whose law does not depend on the data (Hoffman et
    al. 2024, Example 8), and its estimator is doubly robust.  It is not the odds-ratio
    tilt :class:`~cleverly.interventions.Incremental` (Kennedy 2019), which defines its
    intervention through the fitted mechanism.  The treatment is binary, coded 0 and 1.

    Parameters
    ----------
    delta : float
        The risk ratio, positive.
    name : str
        The report label.  Defaults to ``"rr 0.5"`` style, with ``"natural course"`` for
        ``delta=1``.
    """

    delta: float
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta) or self.delta <= 0.0:
            raise DataError(f"a risk-ratio tilt needs a finite positive delta; got {self.delta!r}")
        if not self.name:
            default = "natural course" if self.delta == 1.0 else f"rr {self.delta:g}"
            object.__setattr__(self, "name", default)


#: Every policy class a fit takes.
Policy: TypeAlias = "Shift | Scale | Piecewise | ModifiedPolicy | RiskRatioTilt"

#: The policy classes, for an ``isinstance`` check.
POLICY_TYPES: tuple[type, ...] = (Shift, Scale, Piecewise, ModifiedPolicy, RiskRatioTilt)


def refuse_policy_declarations(policies: Iterable[object]) -> None:
    """Refuse an undeclared or estimated user policy before any learner.

    :class:`Shift`, :class:`Scale`, :class:`Piecewise` and :class:`RiskRatioTilt` are known
    by their exact type.  A :class:`ModifiedPolicy` carries ``policy_kind``.

    Parameters
    ----------
    policies : iterable of object
        The declared policies.

    Raises
    ------
    DataError
        If a declaration is not one of the three states.
    CapabilityError
        If a :class:`ModifiedPolicy` declares ``policy_kind`` ``None`` or ``"estimated"``.
    """
    for policy in policies:
        if not isinstance(policy, ModifiedPolicy):
            continue
        kind: object = policy.policy_kind
        _POLICY_DECLARATION.check(kind)
        if kind is None:
            raise CapabilityError(f"policy {policy.name!r}: {_POLICY_DECLARATION.undeclared}")
        if kind == "estimated":
            raise CapabilityError(
                f"policy {policy.name!r} is declared as estimated from the sample. "
                + _POLICY_DECLARATION.estimated
            )


def describe_policy(policy: object) -> str:
    """A stable text for one policy, which a result keeps in place of the object.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    str
        The class and its numeric parameters.  A :class:`ModifiedPolicy` is described by
        its name, because its functions have no stable text.
    """
    if isinstance(policy, Shift):
        return f"Shift(delta={float(policy.delta)!r}, cap={policy.cap!r})"
    if isinstance(policy, Scale):
        return f"Scale(factor={float(policy.factor)!r}, cap={policy.cap!r})"
    if isinstance(policy, Piecewise):
        pieces = ", ".join(
            f"({lower!r}, {upper!r}, {describe_policy(mapping)})"
            for lower, upper, mapping in policy.pieces
        )
        return f"Piecewise(({pieces}), closed={policy.closed!r})"
    if isinstance(policy, RiskRatioTilt):
        return f"RiskRatioTilt(delta={float(policy.delta)!r})"
    if isinstance(policy, ModifiedPolicy):
        return f"ModifiedPolicy({policy.name!r})"
    return repr(policy)


def is_identity_policy(policy: object) -> bool:
    """Whether a policy is the natural course by its declaration.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    bool
        ``True`` for ``Shift(0.0, ...)``, ``Scale(1.0, ...)`` and ``RiskRatioTilt(1.0)``.
    """
    return (
        (isinstance(policy, Shift) and policy.delta == 0.0)
        or (isinstance(policy, Scale) and policy.factor == 1.0)
        or (isinstance(policy, RiskRatioTilt) and policy.delta == 1.0)
    )


# ------------------------------------------------------------- branches
#
# A branch is one deterministic map of the dose: a whole deterministic policy, or one
# randomizer value of a randomized one.  Every reader -- the point PolicySet, the
# longitudinal plan -- evaluates a policy through its branches, so the closed forms of
# Shift, Scale and Piecewise and the generic Equation (3) of a declared policy are written
# once.  ``frame`` is a zero-argument callable returning the policy frame; it is called
# only by a branch that reads it, so a Shift never builds one.

FrameSource: TypeAlias = Callable[[], Any]
DensityAt: TypeAlias = Callable[[FloatArray], FloatArray]


def _member(
    values: FloatArray, lower: FloatArray | float, upper: FloatArray | float, closed: str
) -> BoolArray:
    """Whether each value lies in its row's interval, closed at ``closed``."""
    with np.errstate(invalid="ignore"):
        if closed == "left":
            inside = (values >= lower) & (values < upper)
        else:
            inside = (values > lower) & (values <= upper)
    return np.asarray(inside & np.isfinite(values), dtype=bool)


def _call(function: Callable[..., Any], *arguments: Any) -> FloatArray:
    """A user's vectorised function, as a float array."""
    returned = function(*arguments)
    if hasattr(returned, "to_numpy"):
        returned = returned.to_numpy()
    return np.asarray(returned, dtype=float)


def _labels(values: Any) -> Any:
    """A one-dimensional object array of labels, keeping a tuple label whole.

    ``np.asarray`` of a list of tuples is a two-dimensional array, which would split a
    vector node's label into its components; this fills an object array element by element.
    """
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if isinstance(values, np.ndarray) and values.dtype != object:
        return np.asarray(values, dtype=object).reshape(-1)
    items = list(values)
    out = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        out[index] = item
    return out


def _per_row(values: FloatArray, n: int) -> FloatArray:
    """A user's return value as one float per row, broadcasting a scalar."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 1:
        return np.full(n, float(flat[0]))
    if flat.size != n:
        raise DataError(f"a policy function returned {flat.size} values for {n} rows")
    return flat


class _Branch:
    """One deterministic map of the dose, and its density ratio.

    ``arithmetic`` says whether the map needs numeric labels on a categorical node.
    ``continuous`` says whether the branch has a density ratio on a continuous node.
    ``identity`` says whether the branch leaves every dose unchanged by its declaration.
    """

    arithmetic = True
    continuous = True
    identity = False
    cap: float | None = None

    def assign(self, values: Any, frame: FrameSource) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def ratio(
        self, values: FloatArray, frame: FrameSource, density: DensityAt
    ) -> FloatArray:  # pragma: no cover - abstract
        raise NotImplementedError

    def held(self, values: FloatArray) -> BoolArray:
        """Rows a declared ``cap`` holds at their own dose; none for a branch with no cap."""
        return np.zeros(np.asarray(values).reshape(-1).size, dtype=bool)


class _ShiftBranch(_Branch):
    """:class:`Shift`, by its shipped closed form."""

    def __init__(self, shift: Shift) -> None:
        self.shift = shift
        self.identity = shift.delta == 0.0
        # The natural course, Shift(0.0, ...), moves no label, so it applies to any node.
        self.arithmetic = not self.identity
        self.cap = shift.cap

    def assign(self, values: Any, frame: FrameSource) -> Any:
        if self.identity and np.asarray(values).dtype == object:
            return _labels(values)
        return self.shift.apply(values)[0]

    def ratio(self, values: FloatArray, frame: FrameSource, density: DensityAt) -> FloatArray:
        return _shift_ratio(density, values, self.shift)

    def held(self, values: FloatArray) -> BoolArray:
        return np.asarray(self.shift.apply(np.asarray(values, dtype=float))[1], dtype=bool)


def _shift_ratio(density: DensityAt, values: FloatArray, shift: Shift) -> FloatArray:
    """:math:`h_r(a, W)` of a shift at the given treatment values -- the module formula."""
    a = np.asarray(values, dtype=float).reshape(-1)
    numerator = density(a - shift.delta)
    denominator = density(a)
    # A row whose *observed* dose has zero estimated density cannot be reweighted at all;
    # it is a support failure rather than a large weight, and reporting it as an infinite
    # covariate would put a NaN through the Newton solve. Zero is the honest value: the
    # ratio term contributes nothing to the score; min_density reports the denominator.
    safe = np.where(denominator > 0.0, denominator, 1.0)
    covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
    if shift.cap is not None:
        cap = float(shift.cap)
        # Reachable-from-below: a unit can only have been *shifted* to `a` if the shift
        # from `a - delta` was not itself held back, which needs `a <= cap`. Above the cap
        # the only way to be at `a` is to have stayed there, so the ratio term drops out
        # entirely and the indicator below is the whole covariate.
        covariate = covariate * (a <= cap).astype(float)
        covariate = covariate + (a > cap - shift.delta).astype(float)
    return np.asarray(covariate, dtype=float)


class _ScaleBranch(_Branch):
    """:class:`Scale`, by its closed form with the derivative ``1 / factor``."""

    def __init__(self, scale: Scale) -> None:
        self.scale = scale
        self.identity = scale.factor == 1.0
        self.cap = scale.cap

    def assign(self, values: Any, frame: FrameSource) -> Any:
        return self.scale.apply(values)[0]

    def held(self, values: FloatArray) -> BoolArray:
        return np.asarray(self.scale.apply(np.asarray(values, dtype=float))[1], dtype=bool)

    def ratio(self, values: FloatArray, frame: FrameSource, density: DensityAt) -> FloatArray:
        a = np.asarray(values, dtype=float).reshape(-1)
        factor = float(self.scale.factor)
        numerator = density(a / factor) / factor
        denominator = density(a)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        if self.scale.cap is not None:
            cap = float(self.scale.cap)
            # The held rows are exactly those ``apply`` holds, ``factor * a > cap``, so the
            # identity indicator reads the same expression.
            covariate = covariate * (a <= cap).astype(float)
            covariate = covariate + (a * factor > cap).astype(float)
        return np.asarray(covariate, dtype=float)


def _is_identity_map(mapping: Shift | Scale) -> bool:
    return (isinstance(mapping, Shift) and mapping.delta == 0.0) or (
        isinstance(mapping, Scale) and mapping.factor == 1.0
    )


class _PiecewiseBranch(_Branch):
    """:class:`Piecewise`, Equation (3) with each piece's closed-form inverse."""

    def __init__(self, policy: Piecewise) -> None:
        self.policy = policy

    def assign(self, values: Any, frame: FrameSource) -> Any:
        a = np.asarray(values, dtype=float).reshape(-1)
        out = a.copy()
        for lower, upper, mapping in self.policy.pieces:
            inside = _member(a, lower, upper, self.policy.closed)
            out = np.where(inside, mapping.apply(a)[0], out)
        return out

    def ratio(self, values: FloatArray, frame: FrameSource, density: DensityAt) -> FloatArray:
        b = np.asarray(values, dtype=float).reshape(-1)
        denominator = density(b)
        numerator = np.zeros(b.size)
        for lower, upper, mapping in self.policy.pieces:
            if _is_identity_map(mapping):
                continue
            if isinstance(mapping, Shift):
                source, jacobian = b - mapping.delta, 1.0
            else:
                source, jacobian = b / mapping.factor, 1.0 / mapping.factor
            inside = _member(source, lower, upper, self.policy.closed)
            numerator = numerator + np.where(inside, density(source) * jacobian, 0.0)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        for lower, upper, mapping in self.policy.pieces:
            if _is_identity_map(mapping):
                covariate = covariate + _member(b, lower, upper, self.policy.closed).astype(float)
        # ``assign`` leaves a dose in no declared interval where it is, so the uncovered set is
        # an identity piece of Equation (3): a value there can only have come from itself.
        covariate = covariate + self.uncovered(b).astype(float)
        return np.asarray(covariate, dtype=float)

    def uncovered(self, values: FloatArray) -> BoolArray:
        """Doses in no declared interval, which the policy leaves unchanged."""
        b = np.asarray(values, dtype=float).reshape(-1)
        covered = np.zeros(b.size, dtype=bool)
        for lower, upper, _ in self.policy.pieces:
            covered |= _member(b, lower, upper, self.policy.closed)
        return np.asarray(~covered & np.isfinite(b), dtype=bool)


class _PieceBranch(_Branch):
    """A declared :class:`ModifiedPolicy`'s pieces, Equation (3) read generically."""

    def __init__(self, name: str, pieces: tuple[Piece, ...]) -> None:
        self.name = name
        self.pieces = pieces

    def bounds(self, piece: Piece, frame: FrameSource, n: int) -> tuple[FloatArray, FloatArray]:
        """The piece's interval on every row."""
        out = []
        for value in (piece.lower, piece.upper):
            if callable(value):
                out.append(_per_row(_call(value, frame()), n))
            else:
                out.append(np.full(n, float(value)))
        return out[0], out[1]

    def membership(self, values: FloatArray, frame: FrameSource) -> list[BoolArray]:
        """Per piece, which rows' values lie in it."""
        a = np.asarray(values, dtype=float).reshape(-1)
        return [
            _member(a, *self.bounds(piece, frame, a.size), piece.closed) for piece in self.pieces
        ]

    def assign(self, values: Any, frame: FrameSource) -> Any:
        a = np.asarray(values, dtype=float).reshape(-1)
        out = a.copy()
        for piece, inside in zip(self.pieces, self.membership(a, frame), strict=True):
            if piece.map is None or not inside.any():
                continue
            out = np.where(inside, _per_row(_call(piece.map, a, frame()), a.size), out)
        return out

    def ratio(self, values: FloatArray, frame: FrameSource, density: DensityAt) -> FloatArray:
        b = np.asarray(values, dtype=float).reshape(-1)
        denominator = density(b)
        numerator = np.zeros(b.size)
        for piece in self.pieces:
            if piece.map is None:
                continue
            assert piece.inverse is not None and piece.derivative is not None
            with np.errstate(all="ignore"):
                source = _per_row(_call(piece.inverse, b, frame()), b.size)
                slope = _per_row(_call(piece.derivative, b, frame()), b.size)
            usable = np.isfinite(source) & np.isfinite(slope)
            inside = usable & _member(source, *self.bounds(piece, frame, b.size), piece.closed)
            safe_source = np.where(inside, source, b)
            jacobian = np.abs(np.where(inside, slope, 0.0))
            numerator = numerator + np.where(inside, density(safe_source) * jacobian, 0.0)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        for piece in self.pieces:
            if piece.map is None:
                inside = _member(b, *self.bounds(piece, frame, b.size), piece.closed)
                covariate = covariate + inside.astype(float)
        return np.asarray(covariate, dtype=float)


class _ApplyBranch(_Branch):
    """A declared label map on a categorical node, with no density ratio."""

    arithmetic = False
    continuous = False

    def __init__(self, function: Callable[..., Any], value: object = None, *, draws: bool) -> None:
        self.function = function
        self.value = value
        self.draws = draws

    def assign(self, values: Any, frame: FrameSource) -> Any:
        arguments = (values, frame(), self.value) if self.draws else (values, frame())
        return _labels(self.function(*arguments))


class _ConstantBranch(_Branch):
    """Set the node to one label: the second branch of a risk-ratio tilt."""

    arithmetic = False
    continuous = False

    def __init__(self, level: object) -> None:
        self.level = level

    def assign(self, values: Any, frame: FrameSource) -> Any:
        return _labels([self.level] * len(values))


class _IdentityBranch(_Branch):
    """Leave the label unchanged: the first branch of a risk-ratio tilt."""

    arithmetic = False
    continuous = False
    identity = True

    def assign(self, values: Any, frame: FrameSource) -> Any:
        return _labels(values)


def policy_branches(policy: object) -> tuple[tuple[float, _Branch], ...]:
    """Every deterministic branch of a policy and the known probability of each.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    tuple of tuple
        ``(probability, branch)`` per branch.  A deterministic policy has one branch of
        probability one.

    Raises
    ------
    DataError
        If ``policy`` is not one of the policy classes.
    """
    if isinstance(policy, Shift):
        return ((1.0, _ShiftBranch(policy)),)
    if isinstance(policy, Scale):
        return ((1.0, _ScaleBranch(policy)),)
    if isinstance(policy, Piecewise):
        return ((1.0, _PiecewiseBranch(policy)),)
    if isinstance(policy, RiskRatioTilt):
        delta = float(policy.delta)
        if delta == 1.0:
            return ((1.0, _IdentityBranch()),)
        if delta < 1.0:
            return ((delta, _IdentityBranch()), (1.0 - delta, _ConstantBranch(0)))
        keep = 1.0 / delta
        return ((keep, _IdentityBranch()), (1.0 - keep, _ConstantBranch(1)))
    if isinstance(policy, ModifiedPolicy):
        law = policy.randomizer
        if law is None:
            if policy.pieces is not None:
                assert not isinstance(policy.pieces, Mapping)
                return ((1.0, _PieceBranch(policy.name, policy.pieces)),)
            assert policy.apply is not None
            return ((1.0, _ApplyBranch(policy.apply, draws=False)),)
        if policy.pieces is not None:
            assert isinstance(policy.pieces, Mapping)
            pieces = policy.pieces
            return tuple(
                (p, _PieceBranch(policy.name, pieces[value]))
                for value, p in zip(law.values, law.probabilities, strict=True)
            )
        assert policy.apply is not None
        function = policy.apply
        return tuple(
            (p, _ApplyBranch(function, value, draws=True))
            for value, p in zip(law.values, law.probabilities, strict=True)
        )
    raise DataError(
        f"{policy!r} is not a modified treatment policy. Declare Shift, Scale, Piecewise, "
        "ModifiedPolicy or RiskRatioTilt"
    )


def lazy_frame(build: Callable[[], Any]) -> FrameSource:
    """A zero-argument callable that builds the policy frame once, on first use.

    Parameters
    ----------
    build : callable
        Builds the frame.

    Returns
    -------
    callable
        Returns the frame, building it on the first call only.
    """
    cache: list[Any] = []

    def get() -> Any:
        if not cache:
            cache.append(build())
        return cache[0]

    return get


# ------------------------------------------------------------- validation


def check_continuous_policy(
    policy: object,
    observed: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    *,
    where: str = "the treatment",
) -> None:
    """Check that a policy has a density ratio on a continuous node, before any learner.

    A policy that collapses an interval to one dose is not pathwise differentiable on a
    continuous treatment (Díaz, Williams, Hoffman and Schenck 2023, Section 4, journal page
    850), so the policy must declare pieces, whichever ratio route the fit uses.  The checks
    below read a declared :class:`ModifiedPolicy`'s pieces on the observed doses.
    :class:`Shift`, :class:`Scale` and :class:`Piecewise` are strictly monotone on each
    piece by construction.

    Parameters
    ----------
    policy : object
        A declared policy.
    observed : ndarray
        ``(n,)`` observed dose.
    frame : callable
        Returns the policy frame.
    rows : ndarray
        ``(n,)`` the rows that hold a dose at this node.
    where : str
        The node, such as ``"node 'A2' at time 2 of regimen 'up'"``.

    Raises
    ------
    CapabilityError
        If the policy has no pieces, or a piece is not strictly monotone, or the policy
        sends more rows to one dose than any observed dose holds.
    DataError
        If the pieces do not partition the observed doses, or an inverse does not invert
        its map.
    """
    name = getattr(policy, "name", repr(policy))
    if isinstance(policy, RiskRatioTilt):
        raise CapabilityError(
            f"policy {name!r} at {where} is a risk-ratio tilt, which sets a binary treatment. "
            "It has no density ratio on a continuous dose"
        )
    if not isinstance(policy, ModifiedPolicy):
        return
    if policy.pieces is None:
        raise CapabilityError(
            f"policy {name!r} at {where} declares apply= on a continuous treatment. A policy on "
            "a continuous dose declares pieces (Díaz et al. 2023, Assumption 4): an apply= "
            "that maps an interval to one dose is not pathwise differentiable (Section 4, "
            "journal page 850), and no ratio route can tell that it does not. Declare "
            "pieces=(Piece(...), ...)"
        )
    a = np.asarray(observed, dtype=float).reshape(-1)
    for _, branch in policy_branches(policy):
        assert isinstance(branch, _PieceBranch)
        _check_partition(name, branch, a, frame, rows, where)
        _check_monotone(name, branch, a, frame, rows, where)
        _check_ties(name, branch, a, frame, rows, where)


def _check_partition(
    name: str, branch: _PieceBranch, a: FloatArray, frame: FrameSource, rows: BoolArray, where: str
) -> None:
    """Every observed dose lies in exactly one piece (Assumption 4)."""
    count = np.sum(np.vstack(branch.membership(a, frame)), axis=0)
    wrong = rows & (count != 1)
    if wrong.any():
        first = float(a[np.flatnonzero(wrong)[0]])
        raise DataError(
            f"policy {name!r} at {where} does not assign {int(wrong.sum())} observed doses to "
            f"exactly one piece (first at {first:g}). Declare pieces that partition the dose "
            "range, as Assumption 4 of Díaz et al. (2023) requires"
        )


def _check_monotone(
    name: str, branch: _PieceBranch, a: FloatArray, frame: FrameSource, rows: BoolArray, where: str
) -> None:
    """Each map is strictly monotone on its piece, and its inverse inverts it."""
    if not rows.any():
        return
    low_observed, high_observed = float(a[rows].min()), float(a[rows].max())
    for index, piece in enumerate(branch.pieces, start=1):
        if piece.map is None:
            continue
        assert piece.inverse is not None and piece.derivative is not None
        lower, upper = branch.bounds(piece, frame, a.size)
        start = np.maximum(lower, low_observed)
        stop = np.minimum(upper, high_observed)
        usable = rows & np.isfinite(start) & np.isfinite(stop) & (stop > start)
        if not usable.any():
            continue
        start = np.where(usable, start, 0.0)
        stop = np.where(usable, stop, 1.0)
        grid = [start + (stop - start) * (k + 0.5) / _MONOTONE_GRID for k in range(_MONOTONE_GRID)]
        mapped = [_per_row(_call(piece.map, point, frame()), a.size) for point in grid]
        steps = np.vstack([later - earlier for earlier, later in pairwise(mapped)])
        rising = np.all(steps > 0.0, axis=0)
        falling = np.all(steps < 0.0, axis=0)
        flat = usable & ~(rising | falling)
        if flat.any():
            row = int(np.flatnonzero(flat)[0])
            values = np.array([m[row] for m in mapped])
            raise CapabilityError(
                f"policy {name!r} at {where} maps the interval [{start[row]:g}, {stop[row]:g}) to "
                f"the single dose {float(np.median(values)):g} on a continuous treatment, or "
                "is not strictly monotone there. That parameter is not pathwise "
                "differentiable (Díaz et al. 2023, Section 4, journal page 850), so no root-n "
                "interval exists. Collapse the dose into categories and declare the policy on "
                "a categorical node, or use a piecewise shift that moves every dose"
            )
        for point, image in zip(grid, mapped, strict=True):
            with np.errstate(all="ignore"):
                back = _per_row(_call(piece.inverse, image, frame()), a.size)
                slope = _per_row(_call(piece.derivative, image, frame()), a.size)
            gap = np.abs(back - point) / np.maximum(1.0, np.abs(point))
            bad = usable & ~(gap <= _INVERSE_TOLERANCE)
            if bad.any():
                row = int(np.flatnonzero(bad)[0])
                raise DataError(
                    f"policy {name!r} at {where}: inverse(apply(a)) differs from a by "
                    f"{float(gap[row]):.3g} at a = {float(point[row]):g} on piece {index}. "
                    "The density ratio reads the inverse, so it must be exact on every piece"
                )
            flat_slope = usable & ~(np.isfinite(slope) & (slope != 0.0))
            if flat_slope.any():
                row = int(np.flatnonzero(flat_slope)[0])
                raise DataError(
                    f"policy {name!r} at {where}: the derivative of the inverse is "
                    f"{float(slope[row])!r} at b = {float(image[row]):g} on piece {index}. "
                    "It must be finite and nonzero, since the density ratio multiplies by it"
                )
            # The declared derivative of the inverse is 1 / d'(a) at b = d(a).  A central
            # difference of the declared map at the same point checks its value, which the
            # density ratio multiplies by, and not only its sign and finiteness.
            step = _DERIVATIVE_STEP * np.maximum(1.0, np.abs(point))
            with np.errstate(all="ignore"):
                upper_value = _per_row(_call(piece.map, point + step, frame()), a.size)
                lower_value = _per_row(_call(piece.map, point - step, frame()), a.size)
                expected = (2.0 * step) / (upper_value - lower_value)
                gap = np.abs(slope - expected) / np.maximum(np.abs(expected), 1e-12)
            wrong = usable & ~(gap <= _DERIVATIVE_TOLERANCE)
            if wrong.any():
                row = int(np.flatnonzero(wrong)[0])
                raise DataError(
                    f"policy {name!r} at {where}: the declared derivative of the inverse is "
                    f"{float(slope[row]):.6g} at b = {float(image[row]):g} on piece {index}, "
                    f"but 1 / d'(a) from the declared map is {float(expected[row]):.6g} at "
                    f"a = {float(point[row]):g}. The density ratio multiplies by this value, "
                    "so declare the derivative of the inverse map"
                )


def _check_ties(
    name: str, branch: _PieceBranch, a: FloatArray, frame: FrameSource, rows: BoolArray, where: str
) -> None:
    """Refuse a policy that collapses many observed doses onto one assigned dose.

    Two conditions together, so a coarse dose grid with a legitimate piecewise map is not
    refused.  The busiest assigned dose holds more than ``_TIE_EXCESS`` more of the rows
    than the busiest observed dose, and it receives more distinct observed doses than the
    policy has pieces.  A strictly monotone map sends at most one dose per piece to a value
    (per covariate pattern), so only a collapse meets both.
    """
    if not rows.any():
        return
    observed = a[rows]
    assigned = np.asarray(branch.assign(a, frame), dtype=float)[rows]
    _, observed_counts = np.unique(observed, return_counts=True)
    values, assigned_counts = np.unique(assigned, return_counts=True)
    total = float(observed.size)
    share = float(assigned_counts.max()) / total
    value = float(values[int(np.argmax(assigned_counts))])
    sources = np.unique(observed[assigned == value]).size
    if share - float(observed_counts.max()) / total > _TIE_EXCESS and sources > len(branch.pieces):
        raise CapabilityError(
            f"policy {name!r} at {where} sends {share:.1%} of rows to the single dose {value:g}, "
            "more than any dose holds in the data. A policy that collapses an interval of a "
            "continuous dose to one value is not pathwise differentiable (Díaz et al. 2023, "
            "Section 4)"
        )


# ------------------------------------------------------------- discrete formula


def _numeric_levels(levels: Sequence[object]) -> bool:
    """Whether every label of a node is a number, so arithmetic policies apply to it."""
    return all(
        isinstance(level, (int, float, np.number)) and not isinstance(level, (bool, np.bool_))
        for level in levels
    )


def _equal(values: Any, level: object, numeric: bool) -> BoolArray:
    """Elementwise ``values == level``, by the label's own equality."""
    if numeric:
        try:
            as_float = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return np.zeros(len(values), dtype=bool)
        return np.asarray(as_float == float(level), dtype=bool)  # type: ignore[arg-type]
    return np.fromiter((value == level for value in values), dtype=bool, count=len(values))


def discrete_assignments(
    policy: object,
    levels: Sequence[object],
    frame: FrameSource,
    rows: BoolArray,
    *,
    where: str = "the treatment",
) -> tuple[tuple[float, IntArray], ...]:
    """Each branch of a policy as an ``(n, K)`` map of level codes, by the discrete formula.

    Entry ``[i, k]`` of a branch's matrix is the code of the level the branch assigns row
    ``i`` when its observed level is ``levels[k]``.  The induced density is then
    :math:`g^d(a \\mid h) = \\sum_s \\mathbb 1\\{d(s, h) = a\\}\\, g(s \\mid h)` (Díaz,
    Williams, Hoffman and Schenck 2023, Section 4).

    Parameters
    ----------
    policy : object
        A declared policy.
    levels : sequence of object
        The node's labels, in code order.
    frame : callable
        Returns the policy frame.
    rows : ndarray
        ``(n,)`` the rows that hold a treatment at this node.  Only these are checked.
    where : str
        The node, such as ``"treatment 'A'"``, for the refusals.

    Returns
    -------
    tuple of tuple
        ``(probability, codes)`` per branch, with ``codes`` an ``(n, K)`` integer array.

    Raises
    ------
    DataError
        If an arithmetic policy meets a node whose labels are not numbers, or a policy maps
        a label to a value that is not a level of the node.
    """
    name = getattr(policy, "name", repr(policy))
    numeric = _numeric_levels(levels)
    n = int(rows.size)
    out = []
    for probability, branch in policy_branches(policy):
        if branch.arithmetic and not numeric:
            raise DataError(
                f"policy {name!r} adds, scales or splits a dose, but {where} is "
                f"categorical with levels {list(levels)!r}. Declare ModifiedPolicy(apply=...) "
                "that maps labels to labels"
            )
        codes = np.zeros((n, len(levels)), dtype=np.int64)
        for k, level in enumerate(levels):
            source: Any = (
                np.full(n, float(level))  # type: ignore[arg-type]
                if numeric
                else _labels([level] * n)
            )
            assigned = _labels(branch.assign(source, frame))
            if assigned.size != n:
                raise DataError(
                    f"policy {name!r} at {where} returned {assigned.size} values for {n} rows; it "
                    "must return one per row"
                )
            matched = np.zeros(n, dtype=bool)
            column = np.zeros(n, dtype=np.int64)
            for code, candidate in enumerate(levels):
                here = _equal(assigned, candidate, numeric)
                column[here & ~matched] = code
                matched |= here
            unknown = rows & ~matched
            if unknown.any():
                value = assigned[np.flatnonzero(unknown)[0]]
                raise DataError(
                    f"policy {name!r} maps {int(unknown.sum())} rows of {where} to values "
                    f"that are not levels of the node (first: {value!r}). A policy on a "
                    "categorical node must map levels to levels; that dose has no support in "
                    "the data (Díaz et al. 2023, Assumption 1)"
                )
            codes[:, k] = column
        out.append((float(probability), codes))
    return tuple(out)


def induced_probabilities(codes: IntArray, probabilities: FloatArray) -> FloatArray:
    """``(n, K)`` induced :math:`g^d(\\cdot \\mid h)` of one branch, by the discrete formula.

    Parameters
    ----------
    codes : ndarray
        ``(n, K)`` the branch's level map, from :func:`discrete_assignments`.
    probabilities : ndarray
        ``(n, K)`` the fitted :math:`g(\\cdot \\mid h)`.

    Returns
    -------
    ndarray
        :math:`\\sum_s 1\\{d(s, h) = a\\}\\, g(s \\mid h)`, one column per level ``a``.
    """
    n, k = probabilities.shape
    out = np.zeros((n, k))
    rows = np.arange(n)
    for source in range(k):
        np.add.at(out, (rows, codes[:, source]), probabilities[:, source])
    return out


def discrete_ratio(induced: FloatArray, probabilities: FloatArray, at: IntArray) -> FloatArray:
    """:math:`g^d(a_i \\mid h_i) / g(a_i \\mid h_i)` at one level code per row.

    Zero where the fitted probability is zero, as for the continuous ratio.

    Parameters
    ----------
    induced : ndarray
        ``(n, K)`` induced probabilities.
    probabilities : ndarray
        ``(n, K)`` fitted probabilities.
    at : ndarray
        ``(n,)`` level codes.

    Returns
    -------
    ndarray
        One ratio per row.
    """
    rows = np.arange(at.size)
    numerator = induced[rows, at]
    denominator = probabilities[rows, at]
    safe = np.where(denominator > 0.0, denominator, 1.0)
    return np.asarray(np.where(denominator > 0.0, numerator / safe, 0.0), dtype=float)


# ------------------------------------------------------------- the evaluated set


@dataclass(frozen=True)
class PolicySet:
    """Every declared policy, evaluated on the data and on the estimated mechanism.

    Holds arrays and no callables, for the reason
    :class:`~cleverly.interventions.RegimeSet` does: a fit reached through
    :meth:`~cleverly.estimators.TMLE.retarget` -- a truncation sweep, the bootstrap, a
    result loaded from disk -- targets the same declared policies without the mechanism
    being refit or the caller's objects being reachable.

    The arrays are indexed by **component**: one deterministic branch of a declared policy.
    A deterministic policy is one component, ``mixing`` is then ``None`` and the component
    codes are the policy codes.  A randomized policy contributes one component per
    randomizer value, and row ``s`` of ``mixing`` holds the known probabilities with which
    policy ``s`` averages its components.

    Parameters
    ----------
    names : tuple of str
        One per policy, in the order they were declared.  Report labels.
    descriptions : tuple of str
        A stable text of each policy, from :func:`describe_policy`.
    shifted : ndarray
        ``(n, C)``, :math:`d_c(A_i, W_i)` per component: a dose on a continuous treatment
        and a level code on a categorical one.
    ratio : ndarray
        ``(n, C)``, :math:`h_c(A_i, W_i)` -- the clever covariate at the *observed*
        treatment.  **Untruncated** on a continuous treatment, and targeting applies no
        bound to it: the ``mtp`` covariate is this ratio divided only by the missingness and
        intermediate mechanisms a fit declares.  Those mechanisms are bounded at targeting
        time, and ``truncation_curve(mechanism=True)`` sweeps that bound.  No option bounds
        or sweeps the ratio itself.
    ratio_at : ndarray
        ``(n, C, C)``, :math:`h_c(d_s(A_i, W_i), W_i)` at ``[i, s, c]``.  The fluctuation
        updates :math:`\\bar Q` as a function of :math:`(a, W)`, so obtaining
        :math:`\\bar Q^*(d_s(A,W), W)` needs the covariate evaluated *at the assigned dose*
        -- hence a matrix per row rather than a vector.
    moved : ndarray
        ``(n, C)`` boolean, whether the component changed that row's treatment.
    reference : float
        Code of the policy contrasts are taken against.
    mixing : ndarray or None
        ``(S, C)`` known component weights of each policy, or ``None`` when every policy is
        deterministic.
    component_codes : ndarray or None
        ``(n, C, K)`` level map of each component on a categorical treatment, from
        :func:`discrete_assignments`, or ``None`` on a continuous one.
    observed_codes : ndarray or None
        ``(n,)`` observed level codes on a categorical treatment, or ``None``.
    capped : ndarray or None
        ``(n, C)`` boolean, whether a declared ``cap`` held the component at that row's own
        dose.  ``None`` on a categorical treatment, where no cap is evaluated.

    Attributes
    ----------
    n : int
    n_policies : int
    n_components : int
    codes : tuple of float
    component_keys : tuple of float
    labels : dict of float to str
    is_discrete : bool
    weights : ndarray
    policy_ratio : ndarray
    design : ndarray
    """

    names: tuple[str, ...]
    descriptions: tuple[str, ...]
    shifted: FloatArray
    ratio: FloatArray
    ratio_at: FloatArray
    moved: BoolArray
    reference: float = 0.0
    mixing: FloatArray | None = None
    component_codes: IntArray | None = None
    observed_codes: IntArray | None = None
    capped: BoolArray | None = None

    def __post_init__(self) -> None:
        s = len(self.names)
        if len(self.descriptions) != s:
            raise ValueError(f"{s} policy names but {len(self.descriptions)} descriptions")
        if len(set(self.names)) != s:
            raise ValueError(f"policy names must be distinct; got {list(self.names)}")
        n, c = np.asarray(self.shifted).shape
        for name, array, shape in (
            ("ratio", self.ratio, (n, c)),
            ("moved", self.moved, (n, c)),
            *(() if self.capped is None else (("capped", self.capped, (n, c)),)),
            ("ratio_at", self.ratio_at, (n, c, c)),
        ):
            if np.asarray(array).shape != shape:
                raise ValueError(
                    f"PolicySet.{name} has shape {np.asarray(array).shape}, expected {shape}"
                )
        if self.mixing is None and c != s:
            raise ValueError(f"{s} policies but {c} components and no mixing matrix")
        if self.mixing is not None and np.asarray(self.mixing).shape != (s, c):
            raise ValueError(f"PolicySet.mixing must be ({s}, {c})")
        if self.reference not in self.codes:
            raise ValueError(
                f"reference={self.reference} is not one of the policy codes {list(self.codes)}"
            )

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(
        cls,
        policies: Sequence[object],
        data: CausalData,
        density: ConditionalDensity | None = None,
        *,
        propensity: FloatArray | None = None,
        reference: str | None = None,
    ) -> PolicySet:
        """Evaluate every policy against the data and the estimated mechanism.

        On a continuous treatment every entry is a lookup into ``density``'s stored bin
        probabilities, so :math:`g(A \\mid W)` and :math:`g(b_j(A) \\mid W)` for one row
        necessarily come from the same out-of-fold model -- there is no second model to get
        wrong.  On a categorical treatment the ratio reads ``propensity`` by the discrete
        formula, and targeting rebuilds it from the bounded mechanism (:meth:`design_at`).

        Parameters
        ----------
        policies : sequence of policy
            The policies to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data, which supplies the observed treatment.
        density : ConditionalDensity or None
            Estimated conditional density of the dose given the covariates, on a
            continuous treatment.
        propensity : ndarray or None
            ``(n, K)`` fitted treatment probabilities, on a categorical treatment.
        reference : str or None
            Label of the policy contrasts are taken against. ``None`` uses the first.

        Returns
        -------
        PolicySet
            Every policy evaluated on the data and that mechanism.
        """
        declared = tuple(policies)
        if not declared:
            raise DataError("at least one policy is required")
        names = tuple(str(getattr(policy, "name", "")) for policy in declared)
        if len(set(names)) != len(names):
            raise DataError(f"policy names must be distinct; got {list(names)}")
        refuse_policy_declarations(declared)
        frame = lazy_frame(
            lambda: data.frame_like(
                {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
            )
        )
        everyone = np.ones(data.n, dtype=bool)
        if data.is_continuous_treatment:
            if density is None:
                raise ValueError("a continuous treatment's policies need its density")
            built = _continuous_components(declared, data, density, frame, everyone)
        else:
            if propensity is None:
                raise ValueError("a categorical treatment's policies need its mechanism")
            built = _discrete_components(declared, data, propensity, frame, everyone)
        shifted, ratio, ratio_at, moved, mixing, component_codes, observed, capped = built
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the policies {list(names)}")
            code = float(names.index(reference))
        return cls(
            names,
            tuple(describe_policy(policy) for policy in declared),
            shifted,
            ratio,
            ratio_at,
            moved,
            code,
            mixing,
            component_codes,
            observed,
            capped,
        )

    @classmethod
    def evaluate_by_classifier(
        cls,
        policies: Sequence[object],
        data: CausalData,
        learner: Any,
        folds: Folds,
        *,
        reference: str | None = None,
        n_jobs: int = 1,
    ) -> PolicySet:
        """Evaluate every policy with ratios estimated by classification.

        The ratio of each component is :func:`~cleverly.learners.density_ratio.classifier_ratio`
        (Díaz, Williams, Hoffman and Schenck 2023, Section 5.4), cross-fitted over units on
        ``folds``.  No density is fitted.  The checks of the policy are the ones the density
        route runs, so a policy that is not piecewise smooth invertible on a continuous
        treatment is refused on this route too.

        Parameters
        ----------
        policies : sequence of policy
            The policies to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data.
        learner : Learner
            The binary classifier of the stacked label.
        folds : Folds
            The outer split.
        reference : str or None
            Label of the policy contrasts are taken against.
        n_jobs : int
            Parallel workers across the outer folds.

        Returns
        -------
        PolicySet
            Every policy evaluated on the data, with classifier ratios.
        """
        from ..learners.density_ratio import classifier_ratio

        declared = tuple(policies)
        if not declared:
            raise DataError("at least one policy is required")
        names = tuple(str(getattr(policy, "name", "")) for policy in declared)
        if len(set(names)) != len(names):
            raise DataError(f"policy names must be distinct; got {list(names)}")
        refuse_policy_declarations(declared)
        frame = lazy_frame(
            lambda: data.frame_like(
                {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
            )
        )
        everyone = np.ones(data.n, dtype=bool)
        a = np.asarray(data.treatment, dtype=float).reshape(-1)
        owners: list[tuple[int, float]] = []
        columns: list[FloatArray] = []
        held: list[BoolArray] | None = None
        if data.is_continuous_treatment:
            held = []
            for index, policy in enumerate(declared):
                check_continuous_policy(policy, a, frame, everyone)
                for probability, branch in policy_branches(policy):
                    owners.append((index, probability))
                    columns.append(np.asarray(branch.assign(a, frame), dtype=float))
                    held.append(branch.held(a))
        else:
            levels = tuple(data.arm_label(code) for code in data.arm_codes)
            observed_codes = a.astype(np.int64)
            where = f"treatment {data.treatment_name!r}"
            for index, policy in enumerate(declared):
                for probability, codes in discrete_assignments(
                    policy, levels, frame, everyone, where=where
                ):
                    owners.append((index, probability))
                    columns.append(codes[np.arange(data.n), observed_codes].astype(float))
        shifted = np.column_stack(columns)
        count = shifted.shape[1]
        ratio = np.zeros((data.n, count))
        ratio_at = np.zeros((data.n, count, count))
        for c in range(count):
            evaluated = classifier_ratio(
                learner,
                data.covariates,
                a,
                ((1.0, shifted[:, c]),),
                data.weights,
                folds,
                evaluate_at=[a, *(shifted[:, s] for s in range(count))],
                groups=data.cluster,
                n_jobs=n_jobs,
            )
            ratio[:, c] = evaluated[0]
            for s in range(count):
                ratio_at[:, s, c] = evaluated[s + 1]
        moved = np.asarray(shifted != a[:, None], dtype=bool)
        mixing = None if count == len(declared) else _mixing(len(declared), owners)
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the policies {list(names)}")
            code = float(names.index(reference))
        return cls(
            names,
            tuple(describe_policy(policy) for policy in declared),
            shifted,
            ratio,
            ratio_at,
            moved,
            code,
            mixing,
            capped=None if held is None else np.column_stack(held),
        )

    # ----------------------------------------------------------------- access

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.shifted.shape[0])

    @property
    def n_policies(self) -> int:
        """Return the number of declared policies."""
        return len(self.names)

    @property
    def n_components(self) -> int:
        """Return the number of deterministic components the policies expand into."""
        return int(self.shifted.shape[1])

    @property
    def codes(self) -> tuple[float, ...]:
        """The keys the per-parameter arrays use, ``(0.0, ..., S-1.0)``."""
        return tuple(float(index) for index in range(self.n_policies))

    @property
    def component_keys(self) -> tuple[float, ...]:
        """The keys the per-component predictions use, ``(0.0, ..., C-1.0)``."""
        return tuple(float(index) for index in range(self.n_components))

    @property
    def labels(self) -> dict[float, str]:
        """Code to reported label, which is what ``parameter_name`` is given."""
        return {float(index): name for index, name in enumerate(self.names)}

    def label(self, code: float) -> str:
        """Return the label for one indexed policy.

        Parameters
        ----------
        code : float
            Policy code.

        Returns
        -------
        str
            The label the policy was declared under.
        """
        return self.labels[float(code)]

    @property
    def is_discrete(self) -> bool:
        """Whether the treatment is categorical, so the ratio reads the propensity."""
        return self.component_codes is not None

    @property
    def weights(self) -> FloatArray:
        """``(S, C)`` component weights of each policy, the identity without a randomizer."""
        if self.mixing is None:
            return np.eye(self.n_policies)
        return np.asarray(self.mixing, dtype=float)

    @property
    def policy_ratio(self) -> FloatArray:
        """``(n, S)`` each policy's clever covariate at the observed treatment.

        The known component weights average the component ratios, which is the ratio of
        the policy's induced density: :math:`g^d = \\sum_e p(e)\\, g^{d_e}`.
        """
        if self.mixing is None:
            return np.asarray(self.ratio, dtype=float)
        return np.asarray(self.ratio @ self.weights.T, dtype=float)

    @property
    def design(self) -> FloatArray:
        """``(n, C + 1, C)`` -- the covariate at the observed dose, then at each assigned one.

        One array rather than two, because :func:`~cleverly.fluctuation.submodel_for`
        dispatches on the group name alone and every builder takes the same keyword-only
        signature; ``policies=`` is that one keyword.  Row block ``0`` is the covariate at
        the observed treatment and block ``s + 1`` the covariate at :math:`d_s(A, W)`.
        """
        return np.concatenate([self.ratio[:, None, :], self.ratio_at], axis=1)

    def design_at(self, propensity: FloatArray | None) -> FloatArray:
        """:attr:`design`, rebuilt from a bounded mechanism on a categorical treatment.

        On a continuous treatment this is :attr:`design`, since the density ratio is not
        bounded.  On a categorical one the ratio reads the fitted probabilities, and
        targeting passes them bounded by ``g_bounds``, as for every other arm-indexed
        clever covariate.

        Parameters
        ----------
        propensity : ndarray or None
            ``(n, K)`` bounded treatment probabilities.

        Returns
        -------
        ndarray
            ``(n, C + 1, C)``.
        """
        if self.component_codes is None or self.observed_codes is None or propensity is None:
            return self.design
        ratio, ratio_at = _discrete_ratios(
            self.component_codes,
            np.asarray(propensity, dtype=float),
            self.observed_codes,
            self.shifted,
        )
        return np.concatenate([ratio[:, None, :], ratio_at], axis=1)

    def subset(self, index: Any) -> PolicySet:
        """The same policies on a row subset -- a fold, a bootstrap resample.

        Sliced rather than re-evaluated, for the reason
        :meth:`~cleverly.interventions.RegimeSet.subset` gives: a policy is the same
        policy on a subsample, and re-deriving it would let the resample redefine the
        estimand.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        PolicySet
            The same policies over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            shifted=self.shifted[idx],
            ratio=self.ratio[idx],
            ratio_at=self.ratio_at[idx],
            moved=self.moved[idx],
            capped=None if self.capped is None else self.capped[idx],
            component_codes=None if self.component_codes is None else self.component_codes[idx],
            observed_codes=None if self.observed_codes is None else self.observed_codes[idx],
        )


_Components: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
    FloatArray | None,
    IntArray | None,
    IntArray | None,
    BoolArray | None,
]


def _mixing(count: int, components: Sequence[tuple[int, float]]) -> FloatArray:
    """The ``(S, C)`` weights that recombine the components into the declared policies."""
    mixing = np.zeros((count, len(components)))
    for column, (row, probability) in enumerate(components):
        mixing[row, column] = probability
    return mixing


def _continuous_components(
    policies: tuple[object, ...],
    data: CausalData,
    density: ConditionalDensity,
    frame: FrameSource,
    rows: BoolArray,
) -> _Components:
    """The component arrays on a continuous treatment."""
    a = np.asarray(data.treatment, dtype=float).reshape(-1)
    for policy in policies:
        check_continuous_policy(policy, a, frame, rows)
    owners: list[tuple[int, float]] = []
    labels: list[str] = []
    branches: list[_Branch] = []
    for index, policy in enumerate(policies):
        for probability, branch in policy_branches(policy):
            owners.append((index, probability))
            labels.append(str(getattr(policy, "name", "")))
            branches.append(branch)
    shifted = np.column_stack([branch.assign(a, frame) for branch in branches])
    ratio = np.column_stack([branch.ratio(a, frame, density.density_at) for branch in branches])
    ratio_at = np.stack(
        [
            np.column_stack(
                [branch.ratio(shifted[:, s], frame, density.density_at) for branch in branches]
            )
            for s in range(len(branches))
        ],
        axis=1,
    )
    for column, (label, branch) in enumerate(zip(labels, branches, strict=True)):
        # The natural course is *meant* to move nobody -- it is the reference the
        # other policies are contrasted against, and its mean is E[Y]. Warning that a
        # zero shift crosses no bin edge, or leaves no dose outside the support,
        # would fire on the recommended way to declare a fit and teach the reader to
        # ignore the warning that matters.
        if branch.identity:
            continue
        warn_if_unresolved(density, shifted[:, column], a)
        _warn_outside_support(label, branch.cap, shifted[:, column], a)
    moved = np.asarray(shifted != a[:, None], dtype=bool)
    capped = np.column_stack([branch.held(a) for branch in branches])
    mixing = None if len(branches) == len(policies) else _mixing(len(policies), owners)
    return shifted, ratio, ratio_at, moved, mixing, None, None, capped


def _discrete_components(
    policies: tuple[object, ...],
    data: CausalData,
    propensity: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
) -> _Components:
    """The component arrays on a categorical treatment, by the discrete formula."""
    if data.has_missing_treatment:
        raise CapabilityError(
            "a modified treatment policy on a categorical treatment reads the level each unit "
            "received, and this fit declares a missing treatment. The policy mean with an "
            "unrecorded treatment is a parameter this package has not derived"
        )
    levels = tuple(data.arm_label(code) for code in data.arm_codes)
    observed = np.asarray(data.treatment, dtype=float).astype(np.int64)
    owners: list[tuple[int, float]] = []
    columns: list[IntArray] = []
    where = f"treatment {data.treatment_name!r}"
    for index, policy in enumerate(policies):
        for probability, codes in discrete_assignments(policy, levels, frame, rows, where=where):
            owners.append((index, probability))
            columns.append(codes)
    component_codes = np.stack(columns, axis=1)
    rows_index = np.arange(data.n)
    shifted = np.column_stack(
        [component_codes[rows_index, c, observed] for c in range(len(columns))]
    ).astype(float)
    ratio, ratio_at = _discrete_ratios(
        component_codes, np.asarray(propensity, dtype=float), observed, shifted
    )
    moved = np.asarray(shifted != observed[:, None], dtype=bool)
    mixing = None if len(columns) == len(policies) else _mixing(len(policies), owners)
    return shifted, ratio, ratio_at, moved, mixing, component_codes, observed, None


def _discrete_ratios(
    component_codes: IntArray, propensity: FloatArray, observed: IntArray, shifted: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """``(ratio, ratio_at)`` of every component by the discrete formula."""
    count = component_codes.shape[1]
    induced = [induced_probabilities(component_codes[:, j, :], propensity) for j in range(count)]
    targets = np.asarray(shifted, dtype=np.int64)
    ratio = np.column_stack(
        [discrete_ratio(induced[j], propensity, observed) for j in range(count)]
    )
    ratio_at = np.stack(
        [
            np.column_stack(
                [discrete_ratio(induced[j], propensity, targets[:, s]) for j in range(count)]
            )
            for s in range(count)
        ],
        axis=1,
    )
    return ratio, ratio_at


def _warn_outside_support(
    name: str, cap: float | None, shifted: FloatArray, observed: FloatArray
) -> None:
    """Warn when assigned doses leave either end of the observed range."""
    smallest, largest = float(np.min(observed)), float(np.max(observed))
    excursions = (
        ("below", smallest, float(np.mean(shifted < smallest))),
        ("above", largest, float(np.mean(shifted > largest))),
    )
    for direction, boundary, fraction in excursions:
        if not fraction:
            continue
        if direction == "below":
            description = (
                f"assigns {fraction:.1%} of rows a dose below the smallest one observed "
                f"({boundary:.3g}). An upper cap cannot prevent lower-support excursions. "
            )
        elif cap is not None:
            description = (
                f"has cap={float(cap):g}, which lies above the largest dose "
                f"observed ({boundary:.3g}), and {fraction:.1%} of rows are assigned "
                "a dose above it. "
            )
        else:
            description = (
                f"has cap=None, and {fraction:.1%} of rows are assigned a dose above "
                f"the largest one observed ({boundary:.3g}). "
            )
        warnings.warn(
            f"policy {name!r} {description}"
            "The outcome regression is extrapolating there. Identification requires the policy "
            "to preserve conditional treatment support; a cap alone does not establish this.",
            PositivityWarning,
            stacklevel=4,
        )


# ------------------------------------------------------------------ diagnostics


@dataclass(frozen=True)
class PolicySupport:
    """Overlap for one policy: how hard the density ratio is working, and where it fails.

    Parameters
    ----------
    name : str
        Report label of the policy.
    policy : str
        The policy this row describes, from
        :func:`~cleverly.interventions.policy.describe_policy`.
    min_density : float or None
        Smallest estimated density at an observed dose, the denominator of the
        density ratio evaluated at the observed dose.  On a categorical treatment it is
        the smallest fitted probability of the observed level.  ``None`` on the classifier
        route, which fits no density.
    ratio_quantiles : dict of float to float
        Quantiles of the density ratio at the observed dose.
    max_ratio : float
        The largest such ratio: how much one row can move the estimate.
    effective_sample_size : float
        Kish effective sample size of those ratios.
    ess_ratio : float
        That size as a share of ``n``.
    moved_fraction : float
        Share of rows the policy moves away from their own dose, averaged over a
        randomized policy's components with their known weights.
    capped_fraction : float or None
        Share of rows a declared ``cap`` holds at their own dose, weighted the same way.
        ``None`` on a categorical treatment, where no cap is evaluated.
    unsupported : int or None
        Rows whose assigned dose falls where the estimated density is exactly zero.
        Estimated zeros flag model support failures, not proof of nonidentification.
        ``None`` on the classifier route, which fits no density to read a zero from.
    mean_ratio : float
        Mean density ratio at the observed dose over every row.  The ratio is the
        density of the supported part of the shifted law relative to the observed law.
        Under the true density its expectation is
        ``P(d(A, W) in the conditional support of g(. | W))``, for capped and uncapped
        policies. It equals 1 when the policy preserves conditional support. An upper
        cap alone does not ensure this for negative shifts or support gaps.
    fold_mean_ratio : tuple of float
        The same mean over the rows each cross-fitting fold holds out, in fold order.
        An in-sample fit has one fold, so the tuple holds :attr:`mean_ratio` alone.  A
        fold mean far from that reference value can signal density estimation error,
        but sampling variation also affects it. The standard error reads that ratio.
        The mean is of the ratio alone, before any mechanism divides it.
    min_mechanism : float or None
        Smallest product of the further mechanisms that divide the covariate beside
        the ratio, or ``None`` when the fit declared neither. When it is not ``None``
        the quantiles and the effective sample size above are of the whole weight.
    score_load : _InterventionLoadRow or None
        Concentration of the exact absolute score weights retained for this policy's
        equation, and the cross-fitting draw it describes. ``None`` means the fitted artifact
        did not supply a usable column. The twelve keys are ``equation``, ``n_total``,
        ``n_targeted``, ``effective``, ``targeted_ratio``, ``total_ratio``, ``top_1pct``,
        ``top_5pct``, ``max_load``, ``zero_load``, ``reported_repeat`` and ``n_repeats``.
    score_load_omission : str or None
        Machine-readable reason why :attr:`score_load` is unavailable.
    """

    name: str
    policy: str
    min_density: float | None
    ratio_quantiles: dict[float, float]
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float
    moved_fraction: float
    capped_fraction: float | None
    unsupported: int | None
    mean_ratio: float
    fold_mean_ratio: tuple[float, ...]
    #: Smallest :math:`\pi(A, W)\,q_z(A, W)` among the mechanisms that divide the
    #: covariate alongside the ratio, or ``None`` when the fit declared neither.  The
    #: quantiles and ESS above are of the *whole* weight when this is not ``None``.
    min_mechanism: float | None = None
    score_load: _InterventionLoadRow | None = None
    score_load_omission: str | None = None

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per row of the report.
        """
        quantiles = ", ".join(f"{q:.0%}: {v:.3g}" for q, v in sorted(self.ratio_quantiles.items()))
        mechanism = (
            "" if self.min_mechanism is None else f", min mechanism={self.min_mechanism:.3g}"
        )
        label = "ratio" if self.min_mechanism is None else "weight"
        score = format_score_load(self.score_load, style="inline")
        folds = ", ".join(f"{value:.3g}" for value in self.fold_mean_ratio)
        capped = "" if self.capped_fraction is None else f"capped={self.capped_fraction:.1%}, "
        density = (
            "min g(A|W) not measured on the classifier route"
            if self.min_density is None
            else f"min g(A|W)={self.min_density:.3g}"
        )
        unsupported = (
            "unsupported not measured on the classifier route"
            if self.unsupported is None
            else f"unsupported={self.unsupported}"
        )
        return (
            f"{self.name}: {density}, max {label}={self.max_ratio:.3g}"
            f"{mechanism}, "
            f"ESS={self.effective_sample_size:.0f} ({self.ess_ratio:.1%} of n), "
            f"moved={self.moved_fraction:.1%}, {capped}{unsupported}, "
            f"{score}\n"
            f"    {label} quantiles -- {quantiles}\n"
            f"    mean ratio -- {self.mean_ratio:.3g} overall, per fold {folds} "
            "(under the true density: P(d(A, W) in the conditional support), "
            "equal to 1 when the policy preserves support, for capped and uncapped policies; "
            "finite-sample means also vary with sampling and density estimation). "
            "Unsupported counts refer to estimated zero density at assigned doses; "
            "these diagnostics do not establish true support or identification."
        )


def check_policy_support(
    policies: PolicySet,
    density: ConditionalDensity | None,
    treatment: FloatArray,
    *,
    propensity: FloatArray | None = None,
    mechanisms: Sequence[FloatArray] = (),
    absolute_score_weights: FloatArray | None = None,
    equations: tuple[str, ...] = (),
    n_repeats: int = 1,
    folds: Folds | None = None,
) -> dict[str, PolicySupport]:
    """Per-policy overlap, in the vocabulary :mod:`cleverly.interventions.support` uses.

    The quantity a policy's positivity rests on is not a propensity but the *ratio*
    :math:`g^d(a \\mid w) / g(a \\mid w)`: it is what multiplies each residual, so its tail
    is where one row starts to dominate the estimating equation.  A randomized policy's
    ratio is the weighted mean of its components' ratios.

    ``mechanisms`` are the further ``(n, C + 1)`` denominators a fit declared -- the
    missingness mechanism under ``delta=``, the intermediate density under
    ``intermediate=`` -- of which only column ``0``, the value at the row's own dose, is
    read here.  The weight the estimating equation forms is
    :math:`h_r(A, W) / \\{\\pi(A, W) q_z(A, W)\\}`, so the two reweightings *multiply* and
    an effective sample size taken of the ratio alone understates the strain --
    :mod:`cleverly.sensitivity.positivity` makes the same argument about ``1 / g`` and a
    population weight.  Passing nothing reports the ratio by itself, which is what a fit
    with no such mechanism means.

    Parameters
    ----------
    policies : PolicySet
        The evaluated policies to report on.
    density : ConditionalDensity or None
        The estimated density the ratios were built from, on a continuous treatment.
    treatment : ndarray
        ``(n,)`` observed dose, or level code on a categorical treatment.
    propensity : ndarray or None
        ``(n, K)`` fitted probabilities, on a categorical treatment.
    mechanisms : sequence of ndarray
        Further ``(n, C + 1)`` denominators the fit declared. Only column ``0``,
        the value at the row's own dose, is read.
    absolute_score_weights : ndarray or None
        Fitted ``abs(w_i * H_ij)`` columns, in component order. ``None`` records an
        omission.
    equations : tuple of str
        Fitted score-equation names, in component order.
    n_repeats : int
        Number of stored cross-fitting draws. The retained weights and the fold means
        describe draw 1.
    folds : Folds or None
        The partition the mechanism was cross-fitted over, which sets the rows of each
        :attr:`PolicySupport.fold_mean_ratio` entry. ``None`` reads every row as one fold.

    Returns
    -------
    dict of str to PolicySupport
        One record per policy, keyed by its report label.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    rows = np.arange(a.size)
    g: FloatArray | None = None
    if density is not None:
        observed_density = density.density_at(a)
    elif propensity is not None:
        g = np.asarray(propensity, dtype=float)
        observed_density = g[rows, a.astype(np.int64)]
    else:
        # The classifier route fits no density, so there is no estimated density to read.
        observed_density = np.full(a.size, np.nan)
    at_observed = [np.asarray(m, dtype=float)[:, 0] for m in mechanisms]
    denominator = np.ones(a.size)
    for values in at_observed:
        denominator = denominator * values
    labels = tuple(policies.names)
    score_loads: dict[str, _InterventionLoadRow]
    if policies.mixing is None:
        score_loads, load_omission = _intervention_loads(
            labels, absolute_score_weights, equations, a.size, n_repeats
        )
    else:
        score_loads, load_omission = {}, "randomized_policy_components"
    held_out = [rows] if folds is None else [test for _, test in folds]
    weights = policies.weights
    policy_ratio = policies.policy_ratio
    out: dict[str, PolicySupport] = {}
    for index, name in enumerate(policies.names):
        ratio = np.asarray(policy_ratio[:, index], dtype=float)
        weight = ratio / denominator
        finite = weight[np.isfinite(weight)]
        ess = effective_sample_size(finite, on_degenerate=0.0)
        mine = [int(c) for c in np.flatnonzero(weights[index] > 0.0)]
        moved = float(sum(weights[index, c] * np.mean(policies.moved[:, c]) for c in mine))
        capped = (
            None
            if policies.capped is None
            else float(sum(weights[index, c] * np.mean(policies.capped[:, c]) for c in mine))
        )
        measured = density is not None or g is not None
        if density is not None:
            assigned = [density.density_at(policies.shifted[:, c]) for c in mine]
        elif g is not None:
            assigned = [g[rows, policies.shifted[:, c].astype(np.int64)] for c in mine]
        else:
            # The classifier route fits no density, so no zero can be read at all.
            assigned = []
        out[name] = PolicySupport(
            name=name,
            policy=policies.descriptions[index],
            min_density=float(observed_density.min()) if measured else None,
            ratio_quantiles={q: float(np.quantile(finite, q)) for q in _QUANTILES},
            max_ratio=float(finite.max()) if finite.size else 0.0,
            effective_sample_size=ess,
            ess_ratio=ess / a.size if a.size else 0.0,
            moved_fraction=moved,
            capped_fraction=capped,
            unsupported=(
                int(sum(np.sum(values <= 0.0) for values in assigned)) if measured else None
            ),
            mean_ratio=float(np.mean(ratio)),
            fold_mean_ratio=tuple(float(np.mean(ratio[test])) for test in held_out),
            min_mechanism=float(denominator.min()) if at_observed else None,
            score_load=score_loads.get(name),
            score_load_omission=load_omission,
        )
    return out
