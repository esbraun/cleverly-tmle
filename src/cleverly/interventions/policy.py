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

import contextlib
import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np

from .._declarations import FunctionDeclaration, FunctionKind
from .._typing import BoolArray, FloatArray, IntArray
from ..data.causal_data import CausalData
from ..data.weighting import effective_sample_size, format_score_load
from ..exceptions import CapabilityError, DataError, PositivityWarning
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity, MixedDensity, warn_if_unresolved
from .support import _intervention_loads, _InterventionLoadRow
from .tuning import (
    DoseQuantile,
    FittedTuner,
    SupportBoundary,
    Target,
    check_target,
    is_tuner,
    population_refusal,
)

if TYPE_CHECKING:
    from .joint_density import JointDensityParts

__all__ = [
    "POLICY_TYPES",
    "JointPolicy",
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
    "Support",
    "check_policy_support",
    "component_source",
    "refuse_policy_declarations",
    "resolve_tuned_policy",
    "tuned_component",
    "tuned_policy",
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

#: Whether a fit refuses a continuous node with an undeclared recorded atom.  A private seam
#: for the exact-law tests, whose finite-support oracles put every dose on a few grid
#: points: :func:`_refuse_undeclared_atoms` reads it by module-global lookup at call time,
#: and it gates that refusal only.  No ratio, density or design code reads it, and no
#: public object names it.
_ATOM_CHECK: bool = True

#: Rule (A) of the undeclared-atom check: the number of distinct doses on each side whose
#: mean count is a value's expected count, the smallest count tested, and the family-wise
#: Poisson tail bound over the distinct doses.
_ATOM_WINDOW = 10
_ATOM_FLOOR = 5
_ATOM_ALPHA = 1e-6
#: Rule (D): the standardized dispersion of the counts that refuses a tied record.
_ATOM_DISPERSION = 6.0
#: Rule (C): a record of at most this many distinct doses is coarse, and a dose there with
#: more than ``_TIE_EXCESS`` of the rows and at least ``_ATOM_COARSE_ROWS`` rows is an atom.
_ATOM_COARSE_K = 100
_ATOM_COARSE_ROWS = 20
#: Rule (D) lists at most this many doses with the largest excess.
_ATOM_LISTED = 5
#: Two doses are the same atom when they agree to this relative tolerance.
_ATOM_TOLERANCE = 1e-12

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
        "al. (2023) Theorem 2 assumes d does not depend on P. For an odds tilt of the fitted "
        "mechanism, put Incremental nodes in IPSI(regimens=...) from cleverly.longitudinal"
    ),
)


def _check_declared_target(name: str, target: object, tuned: bool) -> None:
    """Refuse an unknown ``target`` and a target on a map with a declared cap.

    A tuned map with no target is refused at the fit, before any learner, as an
    undeclared ``policy_kind`` is: :func:`refuse_policy_declarations`.
    """
    if target is None:
        return
    check_target(name, target, tuned)


def _refuse_unresolved(policy: Shift | Scale) -> float | None:
    """Return the numeric cap, and refuse a cap that is a tuner no fit has resolved."""
    cap = policy.cap
    if isinstance(cap, (DoseQuantile, SupportBoundary)):
        raise CapabilityError(
            f"policy {policy.name!r} reads its cap from the treatment law "
            f"({type(cap).__name__}), so it has no map until a fit resolves the tuner "
            "on the units at risk. Pass it to a fit, which resolves it before any learner"
        )
    return None if cap is None else float(cap)


# ------------------------------------------------------------- policy objects


@dataclass(frozen=True)
class Shift:
    """Add ``delta`` to everyone's treatment, up to ``cap``.

    Parameters
    ----------
    delta : float
        How far to move the treatment.  ``0.0`` is the *natural course* -- the policy that
        changes nothing -- whose mean is :math:`E[Y]` and which is the usual reference.
    cap : float, DoseQuantile, SupportBoundary or None
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
        A tuner, :class:`~cleverly.interventions.tuning.DoseQuantile` or
        :class:`~cleverly.interventions.tuning.SupportBoundary`, reads the cap from the
        treatment law; ``target`` then declares the parameter.
    name : str
        What this policy is called in reported parameter names.  Defaults to ``"+0.5"`` /
        ``"-1"`` style, with ``"natural course"`` for ``delta=0``.
    target : {"population", "fold_average"} or None
        The parameter of a map whose cap is a tuner, and ``None`` for a declared cap.  A
        fit refuses a tuned map with no target (:mod:`cleverly.interventions.tuning`).  The
        interval of the ``"population"`` target needs a consistent outcome regression near
        the boundary at each tuned node: the variance term of the tuner reads it.
    """

    delta: float
    cap: float | DoseQuantile | SupportBoundary | None
    name: str = ""
    target: Target | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta):
            raise DataError(f"a shift's delta must be finite; got {self.delta!r}")
        if (
            self.cap is not None
            and not is_tuner(self.cap)
            and not np.isfinite(cast(float, self.cap))
        ):
            raise DataError(f"a shift's cap must be finite or None; got {self.cap!r}")
        if not self.name:
            default = "natural course" if self.delta == 0.0 else f"{self.delta:+g}"
            object.__setattr__(self, "name", default)
        _check_declared_target(self.name, self.target, is_tuner(self.cap))

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
        cap = _refuse_unresolved(self)
        a = np.asarray(treatment, dtype=float).reshape(-1)
        moved = np.asarray(a + self.delta, dtype=float)
        if cap is None:
            return moved, np.zeros(a.size, dtype=bool)
        held = np.asarray(moved > cap, dtype=bool)
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
    cap : float, DoseQuantile, SupportBoundary or None
        The largest dose the policy assigns.  A unit whose scaled dose would exceed it
        keeps its own, as for :class:`Shift`.  A tuner reads it from the treatment law.
    name : str
        The report label.  Defaults to ``"x1.25"`` style, with ``"natural course"`` for
        ``factor=1``.
    target : {"population", "fold_average"} or None
        The parameter of a map whose cap is a tuner, as for :class:`Shift`.
    """

    factor: float
    cap: float | DoseQuantile | SupportBoundary | None
    name: str = ""
    target: Target | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.factor) or self.factor <= 0.0:
            raise DataError(
                f"a scale's factor must be finite and positive; got {self.factor!r}. A factor "
                "of zero maps every dose to one point, which is not pathwise differentiable "
                "on a continuous treatment (Díaz et al. 2023, Section 4). If the dose has a "
                "recorded point mass at 0, declare it with dose_atoms and the piece with "
                "Piece(lower, upper, to_atom=0). Otherwise use a piecewise map that moves "
                "every dose, or collapse the dose into categories and declare the policy on a "
                "categorical node"
            )
        if (
            self.cap is not None
            and not is_tuner(self.cap)
            and not np.isfinite(cast(float, self.cap))
        ):
            raise DataError(f"a scale's cap must be finite or None; got {self.cap!r}")
        if not self.name:
            default = "natural course" if self.factor == 1.0 else f"x{self.factor:g}"
            object.__setattr__(self, "name", default)
        _check_declared_target(self.name, self.target, is_tuner(self.cap))

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
        cap = _refuse_unresolved(self)
        a = np.asarray(treatment, dtype=float).reshape(-1)
        moved = np.asarray(a * self.factor, dtype=float)
        if cap is None:
            return moved, np.zeros(a.size, dtype=bool)
        held = np.asarray(moved > cap, dtype=bool)
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
        Required unless the piece is an identity piece or sends its interval to an atom.
    closed : {"left", "right"}
        Which end of the interval belongs to it: ``"left"`` is ``[lower, upper)`` and
        ``"right"`` is ``(lower, upper]``.
    to_atom : float or None
        The dose the piece sends its whole interval to.  The dose must be a declared atom
        of the node (``dose_atoms=``): a dose with a recorded point mass.  The density
        ratio at the atom then adds the mass of the interval, and the fit refuses the piece
        when the dose is not a declared atom.  A piece with ``to_atom`` declares no
        ``map``, ``inverse`` or ``derivative``.

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
    to_atom: float | None = None

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
        if self.to_atom is not None:
            if self.map is not None or self.inverse is not None or self.derivative is not None:
                raise DataError(
                    "a piece declares either map= (with inverse and derivative) or to_atom=, "
                    "not both. A piece with to_atom sends its whole interval to one declared "
                    "atom, so it has no inverse"
                )
            atom = float(self.to_atom)
            if not np.isfinite(atom):
                raise DataError(f"a piece's to_atom must be finite; got {self.to_atom!r}")
            object.__setattr__(self, "to_atom", atom)
        if self.map is not None and (self.inverse is None or self.derivative is None):
            raise DataError(
                "a piece that moves the dose declares map, inverse and derivative. The "
                "density ratio of Díaz et al. (2023), Equation (3), reads the inverse and "
                "its derivative"
            )

    @property
    def is_identity(self) -> bool:
        """Whether the piece leaves the dose unchanged."""
        return self.map is None and self.to_atom is None


@dataclass(frozen=True)
class Support:
    """The declared conditional support of a continuous dose: ``[lower(h), upper(h)]``.

    Assumption 1 of Díaz, Williams, Hoffman and Schenck (2023) asks that a policy assign
    each unit a dose inside the support of :math:`A_t` given the history.  The analyst
    declares that support, because the data show only the doses that were recorded.  The
    fit then refuses, before any learner, an observed dose outside the declared support
    and a policy that assigns a dose outside it.  Each check reads the observed history,
    on which the estimator conditions, so it covers the histories the data record.

    .. code-block:: python

        Support(0.0, lambda h: 6.0 + 2.0 * h["W2"])

    Parameters
    ----------
    lower : float or callable
        The smallest dose a unit can take, or a function of the policy frame that returns
        one value per row.
    upper : float or callable
        The largest dose, in the same form.
    """

    lower: float | Callable[[Any], Any]
    upper: float | Callable[[Any], Any]

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if callable(value):
                continue
            number = float(value)
            if np.isnan(number):
                raise DataError(f"a declared support's {name} bound must not be NaN")
            object.__setattr__(self, name, number)
        if (
            not callable(self.lower)
            and not callable(self.upper)
            and not float(self.lower) < float(self.upper)
        ):
            raise DataError(
                f"a declared support needs lower < upper; got ({self.lower:g}, {self.upper:g})"
            )

    def bounds(self, frame: Callable[[], Any], n: int) -> tuple[FloatArray, FloatArray]:
        """The support on every row, ``(lower, upper)``.

        Parameters
        ----------
        frame : callable
            Returns the policy frame.  It is called only when a bound is a function.
        n : int
            The number of rows.

        Returns
        -------
        lower : ndarray
            ``(n,)`` lower bound per row.
        upper : ndarray
            ``(n,)`` upper bound per row.
        """
        out = []
        for value in (self.lower, self.upper):
            if callable(value):
                out.append(_per_row(_call(value, frame()), n))
            else:
                out.append(np.full(n, float(value)))
        return out[0], out[1]


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


@dataclass(frozen=True)
class JointPolicy:
    """A modified treatment policy on a vector node, declared one component at a time.

    Díaz, Williams, Hoffman and Schenck (2023, Section 2) let the treatment of a node be a
    vector :math:`A_t = (C_t, D_t)`, with categorical components :math:`C_t` and continuous
    components :math:`D_t`.  A joint policy maps each component.  A component left out of
    ``components`` keeps its natural value.

    .. code-block:: python

        JointPolicy(
            "flag everyone, raise the dose of the flagged",
            components={
                "flag": ModifiedPolicy("flag", apply=flag_everyone, policy_kind="known"),
                "dose": ModifiedPolicy("up if flagged", pieces=(...), policy_kind="known"),
            },
            policy_kind="known",
        )

    The rules below make the density ratio exist, and the fit checks them before any
    learner (their Assumption 4, applied on each categorical cell and each value of the
    earlier doses):

    ======================  =============================================================
    component               policy and what its frame ``h`` holds
    ======================  =============================================================
    continuous              :class:`Shift`, :class:`Scale`, :class:`Piecewise`, or a
                            :class:`ModifiedPolicy` with ``pieces``.  ``h`` holds the
                            policy frame, the **natural** values of the node's categorical
                            components, and the natural values of the doses before it in
                            ``order``, by component name
    categorical             a :class:`ModifiedPolicy` with ``apply``, an arithmetic
                            policy on numeric labels, or a :class:`RiskRatioTilt` on a
                            component coded 0 and 1.  ``h`` holds the policy frame and the
                            natural values of every other component of the node
    ======================  =============================================================

    A read of any other component of the node raises
    :class:`~cleverly.exceptions.DataError`, ``h.get(...)`` included.  With these rules
    the map of the doses is triangular for each categorical cell, so its Jacobian
    determinant is the product of the declared derivatives.  A dose map that reads a later
    dose is refused with the ``order`` that makes it triangular.  Maps that read each other
    in a cycle are refused: a joint map that is not triangular has no checkable declaration
    here, because a determinant checked on a grid does not establish that the map is
    one-to-one.  A :class:`Randomizer` may be declared here, on the joint policy, or on a
    component's :class:`ModifiedPolicy`.  A :class:`RiskRatioTilt` may set a binary
    component coded 0 and 1, as it sets a binary treatment.  The randomizers are
    independent, so the fit reads the policy as one joint randomizer over the product of
    their values, with product probabilities (:func:`joint_branches`).

    Parameters
    ----------
    name : str
        The report label.
    components : mapping
        The component column of the node mapped to that component's policy.  With a
        randomizer, a mapping from each randomizer value to such a mapping.
    randomizer : Randomizer or None
        The known law of the randomizer, or ``None`` for a deterministic policy.
    policy_kind : {"known", "estimated"} or None
        The declaration that the policy is a known function.  A fit accepts ``"known"``
        only.
    order : tuple of str or None
        The order of the doses: the map of each dose reads the natural values of the doses
        before it.  It lists every continuous component of the node.  A categorical
        component may be listed and its position has no effect, since a categorical map
        reads every natural value.  ``None`` takes the node's column order.
    """

    name: str
    components: Mapping[Any, Any]
    randomizer: Randomizer | None = None
    policy_kind: FunctionKind | None = None
    order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataError("a joint policy needs a non-empty name")
        _POLICY_DECLARATION.check(self.policy_kind)
        if self.order is not None:
            if isinstance(self.order, str):
                raise DataError(
                    f"policy {self.name!r}: order= is a tuple of component names; got the "
                    f"string {self.order!r}"
                )
            order = tuple(self.order)
            if not all(isinstance(item, str) and item for item in order):
                raise DataError(
                    f"policy {self.name!r}: order= holds component names; got {order!r}"
                )
            if len(set(order)) != len(order):
                raise DataError(f"policy {self.name!r}: order= names a component twice: {order!r}")
            object.__setattr__(self, "order", order)
        if self.randomizer is not None and not isinstance(self.randomizer, Randomizer):
            raise CapabilityError(
                f"policy {self.name!r} draws a continuous randomizer. The estimator "
                "integrates a randomizer with finitely many values of known probability; "
                "discretize it as Randomizer(values, probabilities)"
            )
        if not isinstance(self.components, Mapping):
            raise DataError(
                f"policy {self.name!r}: components= maps each component column of the node "
                "to its policy"
            )
        if self.randomizer is None:
            object.__setattr__(self, "components", _check_components(self.name, self.components))
            return
        if set(self.components) != set(self.randomizer.values):
            raise DataError(
                f"policy {self.name!r} draws a randomizer, so components= maps each of its "
                f"values {list(self.randomizer.values)!r} to that value's component policies"
            )
        object.__setattr__(
            self,
            "components",
            {
                value: _check_components(self.name, self.components[value])
                for value in self.randomizer.values
            },
        )


def _check_components(name: str, components: Any) -> dict[str, Any]:
    """The component policies of one branch of a :class:`JointPolicy`, or a refusal."""
    if not isinstance(components, Mapping):
        raise DataError(
            f"policy {name!r}: each branch maps component columns to policies; got {components!r}"
        )
    checked: dict[str, Any] = {}
    for column, policy in components.items():
        if not isinstance(column, str) or not column:
            raise DataError(f"policy {name!r}: a component is named by its column; got {column!r}")
        if isinstance(policy, JointPolicy):
            raise DataError(
                f"policy {name!r}: component {column!r} holds a joint policy. A component "
                "takes a Shift, Scale, Piecewise, ModifiedPolicy or RiskRatioTilt"
            )
        if not isinstance(policy, (Shift, Scale, Piecewise, ModifiedPolicy, RiskRatioTilt)):
            raise DataError(
                f"policy {name!r}: component {column!r} holds {policy!r}. A component takes "
                "a Shift, Scale, Piecewise, ModifiedPolicy or RiskRatioTilt"
            )
        checked[column] = policy
    return checked


#: Every policy class a fit takes.
Policy: TypeAlias = "Shift | Scale | Piecewise | ModifiedPolicy | RiskRatioTilt | JointPolicy"

#: The policy classes, for an ``isinstance`` check.
POLICY_TYPES: tuple[type, ...] = (
    Shift,
    Scale,
    Piecewise,
    ModifiedPolicy,
    RiskRatioTilt,
    JointPolicy,
)


def refuse_policy_declarations(policies: Iterable[object]) -> None:
    """Refuse an undeclared or estimated user policy before any learner.

    :class:`Shift`, :class:`Scale`, :class:`Piecewise` and :class:`RiskRatioTilt` are known
    by their exact type.  A :class:`ModifiedPolicy` carries ``policy_kind``, and so does a
    :class:`JointPolicy`, whose component policies are checked as well.

    Parameters
    ----------
    policies : iterable of object
        The declared policies.

    Raises
    ------
    DataError
        If a declaration is not one of the three states.
    CapabilityError
        If a :class:`ModifiedPolicy` or :class:`JointPolicy` declares ``policy_kind``
        ``None`` or ``"estimated"``.
    """
    for policy in policies:
        if isinstance(policy, (Shift, Scale)):
            refuse_tuned_declarations(policy)
            continue
        if isinstance(policy, JointPolicy):
            # A tuned component meets R-e1 and R-e2 as a tuned map of one dose does; the
            # fit resolves it (``cleverly.longitudinal.regimen.resolve_tuned_plans``).
            refuse_policy_declarations(
                component
                for _, components in joint_branches(policy)
                for component in components.values()
            )
        elif not isinstance(policy, ModifiedPolicy):
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


def refuse_tuned_declarations(policy: Shift | Scale) -> None:
    """Refuse a tuned map with no target (R-e1), or a population target it lacks (R-e2).

    Parameters
    ----------
    policy : Shift or Scale
        A declared policy.

    Raises
    ------
    CapabilityError
        If the cap is a tuner and ``target`` is ``None``, or ``target="population"`` with a
        :class:`~cleverly.interventions.tuning.SupportBoundary` whose learner has no
        finite-dimensional parameter.
    """
    tuned = is_tuner(policy.cap)
    check_target(policy.name, policy.target, tuned)
    cap = policy.cap
    if (
        policy.target == "population"
        and isinstance(cap, SupportBoundary)
        and not cap.finite_dimensional
    ):
        raise CapabilityError(population_refusal(policy.name, cap))


def tuned_policy(policy: object) -> bool:
    """Whether a policy is a :class:`Shift` or :class:`Scale` whose cap is a tuner.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    bool
        ``True`` for a data-tuned map.
    """
    return isinstance(policy, (Shift, Scale)) and is_tuner(policy.cap)


def tuned_cap_parts(
    policy: Shift | Scale, fitted: FittedTuner, frame: Any, n: int
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The boundary, the last moved dose, the gradient and ``|e'|`` of a fitted tuned map.

    Parameters
    ----------
    policy : Shift or Scale
        The tuned map.
    fitted : FittedTuner
        Its fitted tuner.
    frame : object
        The policy frame.
    n : int
        The rows.

    Returns
    -------
    tuple
        ``(c, e, dc/dtheta, |e'|)``: the ``(n,)`` boundary :math:`c(h)`, the ``(n,)`` last
        dose the cap lets move (:math:`c - \\delta` or :math:`c / f`), the ``(n, p)``
        gradient, and the slope of :math:`e` in :math:`c`.
    """
    boundary = fitted.boundary(frame, n)
    if isinstance(policy, Shift):
        edge, slope = boundary - float(policy.delta), 1.0
    else:
        edge, slope = boundary / float(policy.factor), 1.0 / float(policy.factor)
    return boundary, edge, fitted.design(frame, n), slope


def tuned_component(policy: object) -> tuple[str, Shift | Scale] | None:
    """The one component of a :class:`JointPolicy` whose cap is a tuner, or ``None``.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    tuple or None
        ``(component name, tuned map)``, or ``None`` when no component is tuned.

    Raises
    ------
    CapabilityError
        If the policy has a randomizer and a tuned component, or tunes two components.
    """
    if not isinstance(policy, JointPolicy):
        return None
    branches = joint_branches(policy)
    tuned = sorted(
        {
            name
            for _, components in branches
            for name, component in components.items()
            if tuned_policy(component)
        }
    )
    if not tuned:
        return None
    if len(branches) > 1:
        raise CapabilityError(
            f"policy {policy.name!r} tunes the cap of component {tuned[0]!r} inside a "
            "randomized JointPolicy. A tuned cap's curve term is derived for one "
            "deterministic map, as at a node of one dose: each branch of a randomizer would "
            "need its own derivative term, weighted by the branch probability, which this "
            "release does not build. Declare one JointPolicy without a randomizer, or the "
            "cap as a number"
        )
    if len(tuned) > 1:
        raise CapabilityError(
            f"policy {policy.name!r} tunes the caps of the components {tuned}. The curve "
            "carries one derivative term per tuned map, and this release builds one per "
            "node. Tune one component and declare the other caps as numbers"
        )
    ((_, components),) = branches
    return tuned[0], cast(Shift | Scale, components[tuned[0]])


def component_source(
    policy: JointPolicy,
    name: str,
    components: Sequence[JointComponent],
    natural: FloatArray,
    frame_with: FrameWith,
    where: str,
) -> GuardedSource:
    """The guarded frame a dose component's map reads: what its tuner may read too.

    A dose map reads the natural categorical components and the natural doses before it
    in the policy's order (:class:`JointPolicy`).  The tuner of its cap reads the same
    frame, so the tuned map stays triangular.

    Parameters
    ----------
    policy : JointPolicy
        The joint policy.
    name : str
        The dose component.
    components : sequence of JointComponent
        The node's components.
    natural : ndarray
        ``(n, d)`` natural values.
    frame_with : callable
        Builds the node's policy frame with extra columns.
    where : str
        The node, for the refusals.

    Returns
    -------
    GuardedSource
        The frame, which refuses a read of any other component.
    """
    names = [component.name for component in components]
    doses = dose_order(policy.name, policy.order, components, where)
    position = [components[index].name for index in doses].index(name)
    labels = {
        component.name: component.labels(natural[:, index])
        for index, component in enumerate(components)
    }
    readable = {other.name: labels[other.name] for other in components if not other.continuous}
    readable.update(
        {components[index].name: labels[components[index].name] for index in doses[:position]}
    )
    return guarded_source(frame_with, readable, set(names) - set(readable))


def resolve_tuned_policy(
    policy: Shift | Scale, fitted: FittedTuner
) -> Shift | Scale | ModifiedPolicy:
    """The map a fitted tuner gives: a declared cap, or pieces with the boundary ``c(h)``.

    A dose quantile gives the same :class:`Shift` or :class:`Scale` with the cap at the
    fitted quantile, so the shipped closed-form ratio reads it.  A support boundary gives a
    known :class:`ModifiedPolicy` of two pieces: the moved interval up to the last dose the
    cap lets move, :math:`e(h)`, and the held interval above it.  That is the map
    ``apply`` would give with the cap :math:`c(h)` per row.

    Parameters
    ----------
    policy : Shift or Scale
        The tuned map.
    fitted : FittedTuner
        Its tuner, fitted on the rows the fit chose.

    Returns
    -------
    object
        The resolved policy, with the same name.
    """
    if isinstance(fitted.tuner, DoseQuantile):
        return replace(policy, cap=float(fitted.theta[0]), target=None)
    boundary = fitted.boundary

    def edge(h: Any) -> FloatArray:
        values = boundary(h, len(h))
        if isinstance(policy, Shift):
            return np.asarray(values - float(policy.delta), dtype=float)
        return np.asarray(values / float(policy.factor), dtype=float)

    if isinstance(policy, Shift):
        delta = float(policy.delta)
        moved = Piece(
            -np.inf,
            edge,
            lambda a, h: np.asarray(a, dtype=float) + delta,
            lambda b, h: np.asarray(b, dtype=float) - delta,
            lambda b, h: 1.0,
            closed="right",
        )
    else:
        factor = float(policy.factor)
        moved = Piece(
            -np.inf,
            edge,
            lambda a, h: np.asarray(a, dtype=float) * factor,
            lambda b, h: np.asarray(b, dtype=float) / factor,
            lambda b, h: 1.0 / factor,
            closed="right",
        )
    held = Piece(edge, np.inf, closed="right")
    return ModifiedPolicy(policy.name, pieces=(moved, held), policy_kind="known")


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
        tail = "" if policy.target is None else f", target={policy.target!r}"
        return f"Shift(delta={float(policy.delta)!r}, cap={policy.cap!r}{tail})"
    if isinstance(policy, Scale):
        tail = "" if policy.target is None else f", target={policy.target!r}"
        return f"Scale(factor={float(policy.factor)!r}, cap={policy.cap!r}{tail})"
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
    if isinstance(policy, JointPolicy):
        return f"JointPolicy({policy.name!r})"
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
        ``True`` for ``Shift(0.0, ...)``, ``Scale(1.0, ...)`` and ``RiskRatioTilt(1.0)``, and
        for a :class:`JointPolicy` with no randomizer whose every component is one of them,
        ``JointPolicy(name, components={})`` included.
    """
    if isinstance(policy, JointPolicy):
        return policy.randomizer is None and all(
            is_identity_policy(component) for component in policy.components.values()
        )
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


@dataclass(frozen=True)
class _SmoothPiece:
    """One smooth invertible piece of a lowered map: its preimage test, inverse and slope.

    ``inside(source)`` says whether a source dose lies in the piece, ``inverse(b)`` is the
    dose on the piece that the map sends to ``b``, and ``slope(b)`` is the derivative of
    the inverse at ``b``.
    """

    inside: Callable[[FloatArray], BoolArray]
    inverse: Callable[[FloatArray], FloatArray]
    slope: Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class _Collapse:
    """An interval that a piece sends to one declared atom, with its bounds per row."""

    lower: FloatArray
    upper: FloatArray
    closed: str
    atom: float


@dataclass(frozen=True)
class LoweredMap:
    """A deterministic map of the dose in the form the mixed density ratio reads.

    Every branch kind lowers to three lists: the smooth invertible pieces of Equation (3)
    of Díaz, Williams, Hoffman and Schenck (2023), the identity pieces, and the intervals
    a piece sends to one declared atom.  The image of a declared atom is the branch's own
    ``assign`` at that dose.

    Parameters
    ----------
    smooth : tuple of _SmoothPiece
        The pieces that move the dose by a strictly monotone smooth map.
    identity : tuple of callable
        Membership tests of the pieces that leave the dose unchanged.
    collapse : tuple of _Collapse
        The intervals sent to one declared atom.
    """

    smooth: tuple[_SmoothPiece, ...]
    identity: tuple[Callable[[FloatArray], BoolArray], ...]
    collapse: tuple[_Collapse, ...] = ()


def lowered_ratio(lowered: LoweredMap, values: FloatArray, density: DensityAt) -> FloatArray:
    """Equation (3) of Díaz et al. (2023) read from a lowered map, at one value per row.

    The ratio of the continuous part: the smooth pieces' preimage densities times the
    absolute slope of each inverse, over the density at the value, plus the indicator of
    each identity piece.  A row whose density at the value is zero gets a zero ratio
    before the identity indicators are added, as in each closed form.  An interval sent to
    an atom adds nothing to the continuous part.

    Parameters
    ----------
    lowered : LoweredMap
        The map.
    values : ndarray
        ``(n,)`` doses at which the ratio is read.
    density : callable
        The continuous density at one value per row.

    Returns
    -------
    ndarray
        ``(n,)`` ratio of the continuous part.
    """
    b = np.asarray(values, dtype=float).reshape(-1)
    denominator = density(b)
    numerator = np.zeros(b.size)
    for piece in lowered.smooth:
        with np.errstate(all="ignore"):
            source = np.asarray(piece.inverse(b), dtype=float).reshape(-1)
            slope = np.asarray(piece.slope(b), dtype=float).reshape(-1)
        usable = np.isfinite(source) & np.isfinite(slope)
        safe_source = np.where(usable, source, b)
        inside = usable & np.asarray(piece.inside(safe_source), dtype=bool)
        safe_source = np.where(inside, source, b)
        jacobian = np.abs(np.where(inside, slope, 0.0))
        numerator = numerator + np.where(inside, density(safe_source) * jacobian, 0.0)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
    for member in lowered.identity:
        covariate = covariate + np.asarray(member(b), dtype=bool).astype(float)
    return np.asarray(covariate, dtype=float)


def _constant(value: float, n: int) -> Callable[[FloatArray], FloatArray]:
    return lambda b: np.full(np.asarray(b).reshape(-1).size, value)


def _shifted_back(delta: float) -> Callable[[FloatArray], FloatArray]:
    def inverse(b: FloatArray) -> FloatArray:
        return np.asarray(b, dtype=float) - delta

    return inverse


def _scaled_back(factor: float) -> Callable[[FloatArray], FloatArray]:
    def inverse(b: FloatArray) -> FloatArray:
        return np.asarray(b, dtype=float) / factor

    return inverse


def _row_call(
    function: Callable[..., Any], frame: FrameSource, n: int
) -> Callable[[FloatArray], FloatArray]:
    def evaluated(b: FloatArray) -> FloatArray:
        return _per_row(_call(function, b, frame()), n)

    return evaluated


def mixed_ratio(
    branch: _Branch, values: FloatArray, frame: FrameSource, density: MixedDensity
) -> FloatArray:
    r"""The density ratio of one branch at a node with declared atoms, at one value per row.

    The ratio is taken with respect to Lebesgue measure plus counting measure on the
    declared atoms (the technical reference, "Modified treatment policies at a node").
    At an atom :math:`a_k`,

    .. math::

        r(a_k, h) = \frac{(1 - \sum_j p_j) \sum_{C \to a_k} \int_C f(s \mid h)\,ds
                    + \sum_{j:\, d(a_j, h) = a_k} p_j(h)}{p_k(h)},

    the mass of every interval a piece sends to :math:`a_k` plus every declared atom the
    branch sends there.  Elsewhere it is Equation (3) of Díaz, Williams, Hoffman and
    Schenck (2023) on the continuous part, whose factor :math:`1 - \sum_j p_j` cancels.
    A row whose denominator is zero gets a zero ratio, as in the closed forms.

    Parameters
    ----------
    branch : _Branch
        One deterministic branch of a policy.
    values : ndarray
        ``(n,)`` doses at which the ratio is read.
    frame : callable
        Returns the policy frame.
    density : MixedDensity
        The fitted law of the dose.

    Returns
    -------
    ndarray
        ``(n,)`` ratio.
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    n = a.size
    lowered = branch.lowered(frame, n)
    atoms = density.atoms
    if density.continuous is None:
        out = np.zeros(n)
    else:
        out = lowered_ratio(lowered, a, density.continuous.density_at)
    if not atoms:
        return out
    p = np.asarray(density.atom_probabilities, dtype=float)
    induced = np.zeros((n, len(atoms)))
    for collapse in lowered.collapse:
        k = int(_atom_index(np.array([collapse.atom]), atoms)[0])
        if k < 0:
            continue
        induced[:, k] = induced[:, k] + density.mass_between(collapse.lower, collapse.upper)
    rows = np.arange(n)
    for j, atom in enumerate(atoms):
        image = np.asarray(branch.assign(np.full(n, float(atom)), frame), dtype=float)
        target = _atom_index(image, atoms)
        hit = target >= 0
        induced[rows[hit], target[hit]] = induced[rows[hit], target[hit]] + p[hit, j]
    which = _atom_index(a, atoms)
    at = which >= 0
    safe = np.maximum(which, 0)
    denominator = p[rows, safe]
    atom_ratio = np.where(
        denominator > 0.0,
        induced[rows, safe] / np.where(denominator > 0.0, denominator, 1.0),
        0.0,
    )
    return np.asarray(np.where(at, atom_ratio, out), dtype=float)


def branch_ratio(
    branch: _Branch,
    values: FloatArray,
    frame: FrameSource,
    density: ConditionalDensity | MixedDensity,
) -> FloatArray:
    """A branch's density ratio: the closed form, or the mixed ratio at a node with atoms.

    Parameters
    ----------
    branch : _Branch
        One deterministic branch of a policy.
    values : ndarray
        ``(n,)`` doses at which the ratio is read.
    frame : callable
        Returns the policy frame.
    density : ConditionalDensity or MixedDensity
        The fitted density of the dose.  A :class:`~cleverly.learners.density.MixedDensity`
        reads the mixed ratio; a binned density reads the branch's own closed form.

    Returns
    -------
    ndarray
        ``(n,)`` ratio.
    """
    if isinstance(density, MixedDensity):
        return mixed_ratio(branch, values, frame, density)
    return np.asarray(branch.ratio(values, frame, density.density_at), dtype=float)


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

    def lowered(self, frame: FrameSource, n: int) -> LoweredMap:  # pragma: no cover - abstract
        """The map as smooth pieces, identity pieces and intervals sent to an atom."""
        raise NotImplementedError


class _ShiftBranch(_Branch):
    """:class:`Shift`, by its shipped closed form."""

    def __init__(self, shift: Shift) -> None:
        self.shift = shift
        self.identity = shift.delta == 0.0
        # The natural course, Shift(0.0, ...), moves no label, so it applies to any node.
        self.arithmetic = not self.identity
        self.cap = _refuse_unresolved(shift)

    def assign(self, values: Any, frame: FrameSource) -> Any:
        if self.identity and np.asarray(values).dtype == object:
            return _labels(values)
        return self.shift.apply(values)[0]

    def ratio(self, values: FloatArray, frame: FrameSource, density: DensityAt) -> FloatArray:
        return _shift_ratio(density, values, self.shift)

    def held(self, values: FloatArray) -> BoolArray:
        return np.asarray(self.shift.apply(np.asarray(values, dtype=float))[1], dtype=bool)

    def lowered(self, frame: FrameSource, n: int) -> LoweredMap:
        delta = float(self.shift.delta)
        cap = _refuse_unresolved(self.shift)
        if cap is None:
            return LoweredMap(
                (_SmoothPiece(np.isfinite, lambda b: b - delta, _constant(1.0, n)),), ()
            )
        top = float(cap)
        # ``apply`` moves a dose s when s + delta <= cap and holds it otherwise, so the
        # moved piece is (-inf, cap - delta] and the held piece is its complement.
        return LoweredMap(
            (
                _SmoothPiece(
                    lambda s: np.isfinite(s) & (s + delta <= top),
                    lambda b: b - delta,
                    _constant(1.0, n),
                ),
            ),
            (lambda b: np.isfinite(b) & (b > top - delta),),
        )


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
    numeric = _refuse_unresolved(shift)
    if numeric is not None:
        cap = numeric
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
        self.cap = _refuse_unresolved(scale)

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
        numeric = _refuse_unresolved(self.scale)
        if numeric is not None:
            cap = numeric
            # The held rows are exactly those ``apply`` holds, ``factor * a > cap``, so the
            # identity indicator reads the same expression.
            covariate = covariate * (a <= cap).astype(float)
            covariate = covariate + (a * factor > cap).astype(float)
        return np.asarray(covariate, dtype=float)

    def lowered(self, frame: FrameSource, n: int) -> LoweredMap:
        factor = float(self.scale.factor)
        cap = _refuse_unresolved(self.scale)
        if cap is None:
            return LoweredMap(
                (_SmoothPiece(np.isfinite, lambda b: b / factor, _constant(1.0 / factor, n)),),
                (),
            )
        top = float(cap)
        return LoweredMap(
            (
                _SmoothPiece(
                    lambda s: np.isfinite(s) & (s * factor <= top),
                    lambda b: b / factor,
                    _constant(1.0 / factor, n),
                ),
            ),
            (lambda b: np.isfinite(b) & (b * factor > top),),
        )


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

    def lowered(self, frame: FrameSource, n: int) -> LoweredMap:
        closed = self.policy.closed
        smooth: list[_SmoothPiece] = []
        identity: list[Callable[[FloatArray], BoolArray]] = []
        for lower, upper, mapping in self.policy.pieces:

            def member(values: FloatArray, lower: float = lower, upper: float = upper) -> BoolArray:
                return _member(values, lower, upper, closed)

            if _is_identity_map(mapping):
                identity.append(member)
            elif isinstance(mapping, Shift):
                delta = float(mapping.delta)
                smooth.append(_SmoothPiece(member, _shifted_back(delta), _constant(1.0, n)))
            else:
                factor = float(mapping.factor)
                smooth.append(
                    _SmoothPiece(
                        member,
                        _scaled_back(factor),
                        _constant(1.0 / factor, n),
                    )
                )
        identity.append(self.uncovered)
        return LoweredMap(tuple(smooth), tuple(identity))


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
            if not inside.any():
                continue
            if piece.to_atom is not None:
                out = np.where(inside, float(piece.to_atom), out)
                continue
            if piece.map is None:
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
            if piece.is_identity:
                inside = _member(b, *self.bounds(piece, frame, b.size), piece.closed)
                covariate = covariate + inside.astype(float)
        return np.asarray(covariate, dtype=float)

    def lowered(self, frame: FrameSource, n: int) -> LoweredMap:
        smooth: list[_SmoothPiece] = []
        identity: list[Callable[[FloatArray], BoolArray]] = []
        collapse: list[_Collapse] = []
        for piece in self.pieces:
            lower, upper = self.bounds(piece, frame, n)

            def member(
                values: FloatArray,
                lower: FloatArray = lower,
                upper: FloatArray = upper,
                closed: str = piece.closed,
            ) -> BoolArray:
                return _member(values, lower, upper, closed)

            if piece.to_atom is not None:
                collapse.append(_Collapse(lower, upper, piece.closed, float(piece.to_atom)))
            elif piece.map is None:
                identity.append(member)
            else:
                inverse, derivative = piece.inverse, piece.derivative
                assert inverse is not None and derivative is not None
                smooth.append(
                    _SmoothPiece(
                        member,
                        _row_call(inverse, frame, n),
                        _row_call(derivative, frame, n),
                    )
                )
        return LoweredMap(tuple(smooth), tuple(identity), tuple(collapse))


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
    if isinstance(policy, (Shift, Scale)):
        _refuse_unresolved(policy)
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
    if isinstance(policy, JointPolicy):
        raise CapabilityError(
            f"policy {policy.name!r} is a JointPolicy, which declares one policy per component "
            "of a vector treatment. This route reads a treatment of one column. A "
            "JointPolicy applies at a vector node of a longitudinal fit, or at a point "
            "treatment=[...] of several columns"
        )
    raise DataError(
        f"{policy!r} is not a modified treatment policy. Declare Shift, Scale, Piecewise, "
        "ModifiedPolicy, RiskRatioTilt or JointPolicy"
    )


@dataclass(frozen=True)
class _DrawnApply:
    """A randomized label map read at one value of its randomizer: ``apply(a, h, value)``."""

    function: Callable[..., Any]
    value: object

    def __call__(self, a: Any, h: Any) -> Any:
        return self.function(a, h, self.value)


@dataclass(frozen=True)
class _SetLabel:
    """The label map that sets every row to one label."""

    label: float

    def __call__(self, a: Any, h: Any) -> FloatArray:
        return np.full(len(a), self.label)


def _component_draws(policy: object) -> tuple[tuple[float, object | None], ...]:
    """A component policy as deterministic draws: ``(probability, policy or None)``.

    ``None`` keeps the component at its natural value.  A :class:`RiskRatioTilt` is the
    two-point randomizer of :func:`policy_branches`: it keeps the label, or sets it to 0
    (``delta < 1``) or to 1 (``delta > 1``).  A randomized :class:`ModifiedPolicy` gives one
    deterministic policy per randomizer value.  Every other policy is one draw.
    """
    if isinstance(policy, RiskRatioTilt):
        delta = float(policy.delta)
        if delta == 1.0:
            return ((1.0, None),)
        keep, label = (delta, 0.0) if delta < 1.0 else (1.0 / delta, 1.0)
        constant = ModifiedPolicy(policy.name, apply=_SetLabel(label), policy_kind="known")
        return ((keep, None), (1.0 - keep, constant))
    if isinstance(policy, ModifiedPolicy) and policy.randomizer is not None:
        law = policy.randomizer
        draws: list[tuple[float, object | None]] = []
        for value, probability in zip(law.values, law.probabilities, strict=True):
            if policy.pieces is not None:
                assert isinstance(policy.pieces, Mapping)
                drawn = ModifiedPolicy(
                    policy.name, pieces=policy.pieces[value], policy_kind=policy.policy_kind
                )
            else:
                assert policy.apply is not None
                drawn = ModifiedPolicy(
                    policy.name,
                    apply=_DrawnApply(policy.apply, value),
                    policy_kind=policy.policy_kind,
                )
            draws.append((float(probability), drawn))
        return tuple(draws)
    return ((1.0, policy),)


def joint_branches(policy: JointPolicy) -> tuple[tuple[float, Mapping[str, Any]], ...]:
    """Every deterministic branch of a joint policy and the known probability of each.

    A randomizer on a component is a joint randomizer over the product of the components'
    draws, with product probabilities: the draws are independent of each other and of
    everything else, so the joint law is the product law (a finite partition).  A
    :class:`RiskRatioTilt` on a binary component is the two-point randomizer that keeps the
    label or sets it, and it enters the product the same way.  The joint policy's own
    randomizer multiplies each of its branches by that product.

    Parameters
    ----------
    policy : JointPolicy
        A declared joint policy.

    Returns
    -------
    tuple of tuple
        ``(probability, components)`` per branch, with ``components`` the component column
        mapped to its deterministic policy.  A deterministic joint policy has one branch of
        probability one.
    """
    law = policy.randomizer
    declared = (
        ((1.0, policy.components),)
        if law is None
        else tuple(
            (float(p), policy.components[value])
            for value, p in zip(law.values, law.probabilities, strict=True)
        )
    )
    out: list[tuple[float, Mapping[str, Any]]] = []
    for probability, components in declared:
        columns = list(components)
        for draws in itertools.product(*(_component_draws(components[c]) for c in columns)):
            mapping = {
                column: drawn
                for column, (_, drawn) in zip(columns, draws, strict=True)
                if drawn is not None
            }
            weight = probability
            for p, _ in draws:
                weight *= p
            out.append((weight, mapping))
    return tuple(out)


def _check_tilt_components(
    policy: JointPolicy, components: Sequence[JointComponent], where: str
) -> None:
    """Refuse a risk-ratio tilt on a component that is not binary and coded 0 and 1.

    The reason is the one the scalar route gives: the tilt keeps a treated unit treated
    with probability ``delta`` and otherwise sets it to 0, so it sets a binary label.
    """
    levels = {component.name: component.levels for component in components}
    branches = (
        (policy.components,)
        if policy.randomizer is None
        else tuple(policy.components[value] for value in policy.randomizer.values)
    )
    for branch in branches:
        for column, component_policy in branch.items():
            if not isinstance(component_policy, RiskRatioTilt) or column not in levels:
                continue
            labels = levels[column]
            if (
                labels
                and _numeric_levels(labels)
                and sorted(float(cast("float", label)) for label in labels) == [0.0, 1.0]
            ):
                continue
            found = "is continuous" if not labels else f"has levels {list(labels)}"
            raise DataError(
                f"policy {policy.name!r} at {where}: RiskRatioTilt sets a binary treatment "
                f"coded 0 and 1, but component {column!r} {found}. It is lmtp's ipsi(), which "
                "keeps a treated unit treated with probability delta and otherwise sets it to 0"
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
    support: Support | None = None,
    atoms: Sequence[float] = (),
    regimen: str | None = None,
    node: str | None = None,
) -> None:
    """Check that a policy has a density ratio on a continuous node, before any learner.

    A policy that collapses an interval to one dose is not pathwise differentiable on a
    continuous treatment (Díaz, Williams, Hoffman and Schenck 2023, Section 4, journal page
    850), so the policy must declare pieces, whichever ratio route the fit uses.  The checks
    below read a declared :class:`ModifiedPolicy`'s pieces on the observed doses.
    :class:`Shift`, :class:`Scale` and :class:`Piecewise` are strictly monotone on each
    piece by construction.

    With declared ``atoms`` every branch kind is checked for the existence condition of the
    mixed law: a piece's ``to_atom`` must be a declared atom, and the image of each
    declared atom must be a declared atom.  With a declared ``support`` every branch must
    assign each row at risk a dose inside its row's support.

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
    support : Support or None
        The declared conditional support of the dose, or ``None``.
    atoms : sequence of float
        The declared atoms of the dose: the doses with a recorded point mass.
    regimen : str or None
        The label the support refusal names; ``None`` names the policy.
    node : str or None
        The node the support refusal names; ``None`` reads ``where``.

    Raises
    ------
    CapabilityError
        If the policy has no pieces, or a piece is not strictly monotone, or the policy
        sends an interval to a dose that is not a declared atom, or it moves a declared
        atom to a dose that is not one.
    DataError
        If the pieces do not partition the observed doses, or an inverse does not invert
        its map, or the policy assigns a dose outside the declared support.
    """
    name = getattr(policy, "name", repr(policy))
    if isinstance(policy, RiskRatioTilt):
        raise CapabilityError(
            f"policy {name!r} at {where} is a risk-ratio tilt, which sets a binary treatment. "
            "It has no density ratio on a continuous dose"
        )
    if isinstance(policy, ModifiedPolicy) and policy.pieces is None:
        raise CapabilityError(
            f"policy {name!r} at {where} declares apply= on a continuous treatment. A policy on "
            "a continuous dose declares pieces (Díaz et al. 2023, Assumption 4): an apply= "
            "that maps an interval to one dose is not pathwise differentiable (Section 4, "
            "journal page 850), and no ratio route can tell that it does not. Declare "
            "pieces=(Piece(...), ...)"
        )
    if not isinstance(policy, (Shift, Scale, Piecewise, ModifiedPolicy)):
        return
    a = np.asarray(observed, dtype=float).reshape(-1)
    declared = tuple(float(atom) for atom in atoms)
    branches = policy_branches(policy)
    for index, (_, branch) in enumerate(branches):
        if isinstance(branch, _PieceBranch):
            _check_partition(name, branch, a, frame, rows, where)
            _check_monotone(name, branch, a, frame, rows, where)
            _check_collapses(name, branch, a.size, frame, rows, where, declared)
            _check_ties(name, branch, a, frame, rows, where, atoms=declared)
        if declared:
            label = describe_policy(policy)
            if len(branches) > 1:
                label = f"branch {index + 1} of {label}"
            _check_atom_images(name, branch, a.size, frame, rows, where, declared, label)
    if support is not None:
        _check_declared_support(
            policy,
            a,
            frame,
            rows,
            support,
            regimen=name if regimen is None else regimen,
            node=where if node is None else node,
        )


def _atom_index(values: FloatArray, atoms: Sequence[float]) -> IntArray:
    """The index of the declared atom each value equals, and ``-1`` for a value that is none.

    Two doses are the same atom when they agree to ``_ATOM_TOLERANCE`` relative to the atom.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(v.size, -1, dtype=np.int64)
    for index, atom in reversed(list(enumerate(atoms))):
        atom = float(atom)
        hit = np.abs(v - atom) <= _ATOM_TOLERANCE * max(1.0, abs(atom))
        out = np.where(hit, index, out)
    return out


def _collapse_refusal(
    name: str, where: str, lower: float, upper: float, value: float
) -> CapabilityError:
    """R-b1: a map that sends an interval of positive mass to a dose with no recorded mass."""
    return CapabilityError(
        f"policy {name!r} at {where} maps the interval [{lower:g}, {upper:g}) to the single "
        f"dose {value:g} on a continuous treatment. The natural dose has no recorded mass at "
        f"{value:g}, so the law of the assigned dose has an atom that the observed law lacks, "
        f"and no density ratio exists. The mean then contains the mean under the static dose "
        f"{value:g}, which is not pathwise differentiable (Díaz et al. 2023, Section 4, journal "
        f"page 850), so no root-n interval exists. If the dose has a recorded point mass at "
        f"{value:g}, declare it with dose_atoms and the piece with Piece({lower:g}, {upper:g}, "
        f"to_atom={value:g}). Otherwise use a piecewise map that moves every dose, or collapse "
        "the dose into categories and declare the policy on a categorical node"
    )


def _check_collapses(
    name: str,
    branch: _PieceBranch,
    n: int,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
    atoms: tuple[float, ...],
) -> None:
    """Every interval a piece sends to one dose sends it to a declared atom (R-b1)."""
    for piece in branch.pieces:
        if piece.to_atom is None:
            continue
        if _atom_index(np.array([piece.to_atom]), atoms)[0] >= 0:
            continue
        lower, upper = branch.bounds(piece, frame, n)
        row = int(np.flatnonzero(rows)[0]) if rows.any() else 0
        raise _collapse_refusal(
            name, where, float(lower[row]), float(upper[row]), float(piece.to_atom)
        )


def _check_atom_images(
    name: str,
    branch: _Branch,
    n: int,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
    atoms: tuple[float, ...],
    label: str,
) -> None:
    """Each declared atom is sent to a declared atom on every row at risk (R-b3).

    An atom of the assigned dose arises from a collapse or from the image of a declared
    atom; the existence of the density ratio needs both to be declared atoms.
    """
    if not rows.any():
        return
    for atom in atoms:
        image = np.asarray(branch.assign(np.full(n, atom), frame), dtype=float).reshape(-1)
        bad = rows & (_atom_index(image, atoms) < 0)
        if bad.any():
            moved_to = float(image[np.flatnonzero(bad)[0]])
            raise CapabilityError(atom_image_refusal(name, where, atom, moved_to, describe=label))


def atom_image_refusal(
    name: str, where: str, atom: float, moved_to: float, *, describe: str, note: str = ""
) -> str:
    """The R-b3 text: a policy moves a declared atom to a dose with no recorded mass.

    Parameters
    ----------
    name : str
        The policy's name.
    where : str
        The node or component.
    atom : float
        The declared atom.
    moved_to : float
        The dose the policy sends it to.
    describe : str
        The policy's description, or the branch it holds.
    note : str
        A sentence on the frames the check read, appended to the text.

    Returns
    -------
    str
        The text.
    """
    return (
        f"policy {name!r} at {where} moves the declared atom {atom:g} to {moved_to:g} "
        f"({describe}). The natural dose has no recorded mass at {moved_to:g}, so every row "
        f"that holds the dose {atom:g} receives a dose where the observed law has no "
        "mass. The law of the assigned dose then has an atom that the observed law "
        "lacks, and no density ratio exists: this is the failure of a collapse (Díaz et "
        "al. 2023, Section 4, journal page 850) reached from the atom. Hold the atom in "
        "place (a cap or a piece that keeps it), send it to another declared atom, or, "
        f"if the dose has recorded mass at {moved_to:g}, declare it in dose_atoms{note}"
    )


def _assigned_outside_support(
    policy: object, observed: FloatArray, frame: FrameSource, rows: BoolArray, support: Support
) -> tuple[BoolArray, FloatArray, FloatArray, FloatArray]:
    """Which rows at risk some branch assigns a dose outside the declared support.

    Returns the rows, the first offending branch's dose per row, and the support bounds.
    """
    a = np.asarray(observed, dtype=float).reshape(-1)
    lower, upper = support.bounds(frame, a.size)
    outside = np.zeros(a.size, dtype=bool)
    first = np.full(a.size, np.nan)
    for _, branch in policy_branches(policy):
        assigned = np.asarray(branch.assign(a, frame), dtype=float).reshape(-1)
        with np.errstate(invalid="ignore"):
            leaves = rows & ~((assigned >= lower) & (assigned <= upper))
        first = np.where(leaves & ~outside, assigned, first)
        outside |= leaves
    return outside, first, lower, upper


def _check_declared_support(
    policy: object,
    observed: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    support: Support,
    *,
    regimen: str,
    node: str,
) -> None:
    """R-a1: the policy assigns every row at risk a dose inside the declared support."""
    outside, first, lower, upper = _assigned_outside_support(policy, observed, frame, rows, support)
    if not outside.any():
        return
    name = getattr(policy, "name", repr(policy))
    raise _support_refusal(name, outside, first, lower, upper, regimen=regimen, node=node)


def _support_refusal(
    policy: str,
    outside: BoolArray,
    first: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    regimen: str,
    node: str,
) -> DataError:
    """The R-a1 text: ``outside`` rows are assigned ``first``, outside ``[lower, upper]``."""
    row = int(np.flatnonzero(outside)[0])
    return DataError(
        f"regimen {regimen!r} is not identified under the declared support: at node {node!r} "
        f"the policy {policy!r} assigns {int(outside.sum())} row(s) at risk a dose outside the "
        f"declared support (first: dose {float(first[row]):g} against [{float(lower[row]):g}, "
        f"{float(upper[row]):g}]). Díaz et al. (2023), Assumption 1, needs the assigned dose "
        "inside the support of A_t given the history. Cap the policy at the support boundary, "
        "for example Shift(delta, cap=...) or a Piece bound that reads the frame, or remove "
        "the regimen"
    )


def _component_frame(
    components: Sequence[JointComponent],
    index: int,
    values: FloatArray,
    frame_with: FrameWith,
) -> FrameSource:
    """The frame a component's declared support reads: its chain-rule conditioning set.

    The joint density factors as the categorical components first and then the doses in
    column order (:mod:`~cleverly.interventions.joint_density`), so the support of dose
    ``index`` is a support given the history, every categorical component and the doses
    before it.  ``values`` holds those components' values: the natural ones for the
    observed check, and a branch's assigned ones for the check of the assigned vector.
    """
    extra: dict[str, Any] = {}
    for position, other in enumerate(components):
        if position == index:
            continue
        if not other.continuous:
            extra[other.name] = other.labels(values[:, position])
        elif position < index:
            extra[other.name] = np.asarray(values[:, position], dtype=float)
    return lazy_frame(lambda: frame_with(extra))


def check_component_supports(
    policy: str,
    components: Sequence[JointComponent],
    natural: FloatArray,
    assigned: Sequence[FloatArray],
    frame_with: FrameWith,
    rows: BoolArray,
    supports: Mapping[str, Support],
    *,
    regimen: str,
) -> None:
    """R-a2 and R-a1 for the declared support of each continuous component of a vector.

    The support of a dose is read given its chain-rule conditioning set
    (:func:`_component_frame`).  An observed dose is checked on the natural values (R-a2).
    An assigned dose is checked on the values the same branch assigns, because Assumption 1
    of Díaz et al. (2023) asks that the assigned vector lie in the support of the vector
    given the history.

    The density route also reads counterfactual frames, which hold a candidate cell and the
    preimages of the earlier doses (:mod:`~cleverly.interventions.joint_density`).  There the
    value the formula evaluates is the observed vector, which R-a2 puts inside its support,
    so the assigned side needs no further check.  A preimage outside the support of its own
    frame has true density zero and adds nothing to the induced density.

    Parameters
    ----------
    policy : str
        The joint policy's name.
    components : sequence of JointComponent
        The node's components, in column order.
    natural : ndarray
        ``(n, d)`` natural values: level codes and doses.
    assigned : sequence of ndarray
        Each branch's ``(n, d)`` assigned values.
    frame_with : callable
        Returns the node's policy frame with the given columns appended.
    rows : ndarray
        ``(n,)`` the rows that hold the node.
    supports : mapping of str to Support
        The declared support of each continuous component that declares one.
    regimen : str
        The label the R-a1 refusal names.

    Raises
    ------
    DataError
        If an observed dose, or a dose some branch assigns, lies outside its row's declared
        support.
    """
    n = int(rows.size)
    for index, component in enumerate(components):
        support = supports.get(component.name)
        if support is None:
            continue
        check_observed_support(
            natural[:, index],
            _component_frame(components, index, natural, frame_with),
            rows,
            support,
            node=component.name,
        )
        outside = np.zeros(n, dtype=bool)
        first = np.full(n, np.nan)
        lower_first = np.full(n, np.nan)
        upper_first = np.full(n, np.nan)
        for values in assigned:
            lower, upper = support.bounds(
                _component_frame(components, index, values, frame_with), n
            )
            dose = np.asarray(values[:, index], dtype=float)
            with np.errstate(invalid="ignore"):
                leaves = rows & ~((dose >= lower) & (dose <= upper))
            new = leaves & ~outside
            first = np.where(new, dose, first)
            lower_first = np.where(new, lower, lower_first)
            upper_first = np.where(new, upper, upper_first)
            outside |= leaves
        if outside.any():
            raise _support_refusal(
                policy,
                outside,
                first,
                lower_first,
                upper_first,
                regimen=regimen,
                node=component.name,
            )


def check_observed_support(
    observed: FloatArray, frame: FrameSource, rows: BoolArray, support: Support, *, node: str
) -> None:
    """Refuse a declared support that excludes an observed dose (R-a2), before any learner.

    Parameters
    ----------
    observed : ndarray
        ``(n,)`` observed dose.
    frame : callable
        Returns the policy frame.
    rows : ndarray
        ``(n,)`` the rows that hold a dose at this node.
    support : Support
        The declared support.
    node : str
        The node the refusal names.

    Raises
    ------
    DataError
        If an observed dose lies outside its row's declared support.
    """
    a = np.asarray(observed, dtype=float).reshape(-1)
    lower, upper = support.bounds(frame, a.size)
    with np.errstate(invalid="ignore"):
        outside = rows & ~((a >= lower) & (a <= upper))
    if not outside.any():
        return
    row = int(np.flatnonzero(outside)[0])
    raise DataError(
        f"the declared support of node {node!r} excludes {int(outside.sum())} observed dose(s) "
        f"(first: {float(a[row]):g} against [{float(lower[row]):g}, {float(upper[row]):g}]). The "
        "declaration contradicts the data; widen it or correct the dose"
    )


def policy_moves_dose(policy: object) -> bool:
    """Whether some branch of a policy moves a dose by its declaration.

    The natural course and a policy of identity pieces move no dose, so their ratio is one
    whatever the law of the dose, and the undeclared-atom check does not run for them.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    bool
        ``False`` when every branch is the identity map by its declaration.
    """
    if isinstance(policy, (Shift, Scale, RiskRatioTilt)):
        return not is_identity_policy(policy)
    if isinstance(policy, Piecewise):
        return any(not _is_identity_map(mapping) for _, _, mapping in policy.pieces)
    if isinstance(policy, ModifiedPolicy) and policy.pieces is not None:
        return any(
            not piece.is_identity
            for _, branch in policy_branches(policy)
            for piece in cast(_PieceBranch, branch).pieces
        )
    return True


def _neighbour_counts(counts: FloatArray) -> FloatArray:
    """The mean count of up to ``_ATOM_WINDOW`` distinct doses on each side of each dose.

    At a support edge the mean is over the neighbours that exist, at least one.
    """
    k = counts.size
    width = _ATOM_WINDOW
    padded = np.concatenate([np.zeros(width), counts.astype(float), np.zeros(width)])
    present = np.concatenate([np.zeros(width), np.ones(k), np.zeros(width)])
    total = np.cumsum(np.r_[0.0, padded])
    seen = np.cumsum(np.r_[0.0, present])
    centre = np.arange(k) + width
    neighbours = total[centre + width + 1] - total[centre - width] - counts
    number = seen[centre + width + 1] - seen[centre - width] - 1.0
    return np.asarray(neighbours / np.maximum(number, 1.0), dtype=float)


def _format_doses(values: Sequence[float]) -> str:
    """Doses as the body of a Python tuple: ``"3"`` becomes ``"3,"``."""
    text = ", ".join(f"{float(value):g}" for value in values)
    return text + "," if len(values) == 1 else text


def _atom_rule_flags(
    doses: FloatArray, *, atoms: Sequence[float], bins: int | None
) -> dict[str, bool]:
    """Which rules of the undeclared-atom check fire on a record, by rule letter.

    The rules are those of :func:`_refuse_undeclared_atoms`, evaluated one by one, so a
    witness can show that only its own rule fires.  The seam ``_ATOM_CHECK`` does not
    apply here.

    Parameters
    ----------
    doses : ndarray
        The observed doses of the rows at risk.
    atoms : sequence of float
        The declared atoms, whose rows the rules do not read.
    bins : int or None
        The density route's bin count, or ``None`` on the classifier route.

    Returns
    -------
    dict of str to bool
        ``{"A": ..., "B": ..., "C": ..., "D": ...}``.
    """
    out = {}
    for rule in "ABCD":
        others = tuple(other for other in "ABCD" if other != rule)
        out[rule] = _atom_rules(doses, atoms=atoms, bins=bins, off=others) is not None
    return out


def _atom_rules(
    doses: FloatArray,
    *,
    atoms: Sequence[float],
    bins: int | None,
    off: tuple[str, ...] = (),
) -> tuple[str, list[float]] | None:
    """The finding and the flagged doses of the undeclared-atom check, or ``None``.

    ``off`` names rules to leave out, for :func:`_atom_rule_flags`.
    """
    from scipy.stats import poisson

    a = np.asarray(doses, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if atoms:
        a = a[_atom_index(a, atoms) < 0]
    total = a.size
    if total == 0:
        return None
    values, counts = np.unique(a, return_counts=True)
    k = values.size
    tied = k < total
    lam = np.maximum(_neighbour_counts(counts.astype(float)), 1e-12)
    tail = poisson.sf(counts - 1, lam)
    rule_a = (counts >= _ATOM_FLOOR) & (tail < _ATOM_ALPHA / k)
    z = float((np.sum((counts - lam) ** 2 / lam) - k) / np.sqrt(2.0 * k)) if tied else -np.inf
    rule_d = tied and z > _ATOM_DISPERSION
    rule_b = tied and bins is not None and k < bins
    heavy = (counts > _TIE_EXCESS * total) & (counts >= _ATOM_COARSE_ROWS)
    rule_c = k <= _ATOM_COARSE_K and bool(heavy.any())
    fired = {"A": bool(rule_a.any()), "D": bool(rule_d), "B": bool(rule_b), "C": bool(rule_c)}
    for rule in off:
        fired[rule] = False
    if not any(fired.values()):
        return None
    excess = counts - lam
    listed = np.argsort(-excess, kind="stable")[: min(_ATOM_LISTED, k)]
    flagged: set[int] = set()
    if fired["A"]:
        flagged |= set(np.flatnonzero(rule_a).tolist())
    if fired["D"]:
        flagged |= set(listed.tolist())
    if fired["B"]:
        flagged |= set(range(k))
    if fired["C"]:
        flagged |= set(np.flatnonzero(heavy).tolist())
    if fired["A"]:
        index = int(np.flatnonzero(rule_a)[np.argmin(tail[rule_a])])
        finding = (
            f"the dose {float(values[index]):g} on {int(counts[index])} of {total} rows at risk, "
            f"against about {float(lam[index]):.1f} rows at each neighbouring value"
        )
    elif fired["D"]:
        finding = (
            "counts that vary across neighbouring doses far more than a density allows "
            f"(dispersion z = {z:.1f}), so many doses carry small point masses; the "
            f"{listed.size} doses with the largest excess are "
            f"{', '.join(f'{float(values[i]):g}' for i in sorted(listed.tolist()))}"
        )
    elif fired["B"]:
        finding = f"only {k} distinct doses against {bins} density bins"
    else:
        index = int(np.flatnonzero(heavy)[np.argmax(counts[heavy])])
        finding = (
            f"only {k} distinct doses, and the dose {float(values[index]):g} on "
            f"{float(counts[index]) / total:.1%} of the rows"
        )
    return finding, [float(values[i]) for i in sorted(flagged)]


def _refuse_undeclared_atoms(
    doses: FloatArray,
    *,
    atoms: Sequence[float],
    moved: bool,
    bins: int | None,
    where: str,
    remedy: str | None = None,
    component: bool = False,
) -> None:
    r"""Refuse a continuous node whose recorded doses hold an undeclared atom (R-b4).

    Four rules read the at-risk observed doses that are not declared atoms, with ``k`` the
    number of distinct values and :math:`\lambda(v)` the mean count of the
    ``_ATOM_WINDOW`` distinct values on each side of ``v``.  (A) local excess: ``c(v) >= 5``
    and :math:`P(\mathrm{Poisson}(\lambda(v)) \ge c(v)) < 10^{-6} / k`.  (D) dispersion, on a
    record with ties: :math:`z = (\sum_v (c(v) - \lambda(v))^2 / \lambda(v) - k) / \sqrt{2k} > 6`.
    (B) on the density route, on a record with ties: fewer distinct values than bins.
    (C) a coarse record: ``k <= 100`` and a value with more than ``_TIE_EXCESS`` of the rows
    and at least 20 rows.  The message names the first rule in the order (A), (D), (B),
    (C) that fires, and lists every flagged value.

    The check runs only where some branch moves a dose, and only while the private seam
    ``_ATOM_CHECK`` holds, which this function reads at call time.

    Parameters
    ----------
    doses : ndarray
        The observed doses of the rows at risk.
    atoms : sequence of float
        The declared atoms, whose rows the rules do not read.
    moved : bool
        Whether some branch at the node moves a dose.
    bins : int or None
        The density route's bin count, or ``None`` on the classifier route.
    where : str
        The node the refusal names.
    remedy : str or None
        The declaration the refusal suggests, with ``{values}`` for the flagged doses.
        ``None`` gives the longitudinal mapping form.
    component : bool
        Whether ``where`` is a continuous component of a vector, whose declaration names
        the component in the mapping form.

    Raises
    ------
    DataError
        If any rule flags the node.
    """
    if not _ATOM_CHECK or not moved:
        return
    found = _atom_rules(doses, atoms=atoms, bins=bins)
    if found is None:
        return
    finding, values = found
    doses_text = _format_doses(values)
    if component:
        raise DataError(
            f"component {where!r} of a vector treatment records {finding}. A continuous dose "
            "with a recorded point mass needs that mass declared: a density has no mass at a "
            "point, so a ratio that treats the atom as density is wrong at every row a policy "
            f"moves to or from it. Declare dose_atoms={{{where!r}: ({doses_text})}}, which "
            "makes the component's factor of the joint density mixed, or code the component "
            "as categorical if every dose sits on a grid"
        )
    declare = (
        f"dose_atoms={{{where!r}: ({doses_text})}}"
        if remedy is None
        else remedy.format(values=doses_text)
    )
    raise DataError(
        f"node {where!r} records {finding}. A continuous dose with a recorded point mass needs "
        "that mass declared: a density has no mass at a point, so a ratio that treats the atom "
        "as density is wrong at every row a policy moves to or from it. Declare "
        f"{declare}, or code the node as categorical if every dose sits on a grid"
    )


def preflight_point_policies(
    policies: Sequence[object],
    data: CausalData,
    *,
    ratio: str,
    density_bins: int | None,
) -> None:
    """The declared-support and atom checks of a point policy fit, before any learner.

    In order: an observed dose outside the declared support (R-a2), an undeclared recorded
    atom (R-b4), a declared atom with no observed row (R-b2), and each policy's own checks
    with the declared support and atoms (R-b1, R-b3, R-a1).

    Parameters
    ----------
    policies : sequence of policy
        The declared policies.
    data : CausalData
        The validated data, with a continuous treatment of one column.
    ratio : {"density", "classifier"}
        The ratio route, which decides whether the bin count rule applies.
    density_bins : int or None
        The declared bin count of the density route, or ``None`` for the default.

    Raises
    ------
    DataError
        If a declaration contradicts the data, or the dose has an undeclared atom.
    CapabilityError
        If a policy has no density ratio under the declared atoms.
    """
    from ..learners.density import default_density_bins

    # Only the modified treatment policies move a dose here; every other intervention kind
    # keeps its own checks.  A JointPolicy, which this route cannot read, is refused by its
    # own text before any declaration check.
    policies = [
        policy
        for policy in policies
        if isinstance(policy, (Shift, Scale, Piecewise, ModifiedPolicy, JointPolicy))
    ]
    for policy in policies:
        if isinstance(policy, JointPolicy):
            policy_branches(policy)
    if not policies:
        return
    a = np.asarray(data.treatment, dtype=float).reshape(-1)
    everyone = np.ones(a.size, dtype=bool)
    frame = lazy_frame(
        lambda: data.frame_like(
            {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        )
    )
    node = data.treatment_name
    atoms = tuple(data.dose_atoms)
    if data.treatment_support is not None:
        check_observed_support(a, frame, everyone, data.treatment_support, node=node)
    bins: int | None = None
    if ratio == "density":
        continuous = int(np.sum(_atom_index(a, atoms) < 0)) if atoms else a.size
        bins = density_bins if density_bins is not None else default_density_bins(continuous)
    _refuse_undeclared_atoms(
        a,
        atoms=atoms,
        moved=any(policy_moves_dose(policy) for policy in policies),
        bins=bins,
        where=node,
        remedy="dose_atoms=({values})",
    )
    if atoms:
        check_declared_atoms(a, everyone, atoms, None, node=node)
    for policy in policies:
        check_continuous_policy(
            policy,
            a,
            frame,
            everyone,
            support=data.treatment_support,
            atoms=atoms,
            node=node,
        )


def check_declared_atoms(
    observed: FloatArray, rows: BoolArray, atoms: Sequence[float], folds: Folds | None, *, node: str
) -> None:
    r"""Refuse a declared atom with no observed row, overall or in an outer training fold.

    The ratio at an atom divides by :math:`P(A = a \mid H)`,
    which the data estimate only from rows that received it.

    Parameters
    ----------
    observed : ndarray
        ``(n,)`` observed dose.
    rows : ndarray
        ``(n,)`` the rows that hold a dose at this node.
    atoms : sequence of float
        The declared atoms.
    folds : Folds or None
        The outer split, or ``None`` for an in-sample fit.
    node : str
        The node the refusal names.

    Raises
    ------
    DataError
        If a declared atom has no at-risk row, or none in some training fold.
    """
    a = np.asarray(observed, dtype=float).reshape(-1)
    index = _atom_index(a, atoms)
    for k, atom in enumerate(atoms):
        holds = rows & (index == k)
        where_none = ""
        if not holds.any():
            where_none = "no row at risk at that node holds that dose"
        elif folds is not None and not folds.is_single:
            for fold, (train, _) in enumerate(folds, start=1):
                if not holds[train].any():
                    where_none = (
                        f"the training rows of outer fold {fold} hold no row at risk with that dose"
                    )
                    break
        if where_none:
            raise DataError(
                f"dose_atoms declares {float(atom):g} at node {node!r}, but {where_none}. A "
                "declared atom needs recorded mass: the ratio at the atom divides by "
                f"P(A = {float(atom):g} | H), which the data estimate only from rows that "
                "received it"
            )


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
    name: str,
    branch: _PieceBranch,
    a: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
    note: str = "",
) -> None:
    """Each map is strictly monotone on its piece, and its inverse inverts it.

    The grid spans the piece inside the observed range of ``a`` on every row's frame.
    ``note`` ends each refusal, for a caller whose frames are not the observed ones.
    """
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
            # A map flat on the observed frames sends an interval of natural mass to one
            # dose (R-b1).  On a counterfactual frame (``note``) the interval need not hold
            # natural mass, so the finding there is the weaker one: not strictly monotone.
            if not note and np.any(values[1:] == values[:-1]):
                points = np.array([g[row] for g in grid])
                if np.all(values == values[0]):
                    low, high, value = float(start[row]), float(stop[row]), float(values[0])
                else:
                    # The first run of grid points with one image: a step of the map.
                    first = int(np.flatnonzero(values[1:] == values[:-1])[0])
                    last = first + 1
                    while last + 1 < values.size and values[last + 1] == values[first]:
                        last += 1
                    low, high = float(points[first]), float(points[last])
                    value = float(values[first])
                raise _collapse_refusal(name, where, low, high, value)
            raise CapabilityError(
                f"policy {name!r} at {where} maps the interval [{start[row]:g}, "
                f"{stop[row]:g}) of a continuous treatment by a map that is not strictly "
                "monotone there. Equation (3) of Díaz et al. (2023) needs a strictly monotone "
                "smooth map on each piece (Assumption 4). Split the interval into pieces on "
                "which the map is strictly monotone" + note
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
                    "The density ratio reads the inverse, so it must be exact on every piece" + note
                )
            flat_slope = usable & ~(np.isfinite(slope) & (slope != 0.0))
            if flat_slope.any():
                row = int(np.flatnonzero(flat_slope)[0])
                raise DataError(
                    f"policy {name!r} at {where}: the derivative of the inverse is "
                    f"{float(slope[row])!r} at b = {float(image[row]):g} on piece {index}. "
                    "It must be finite and nonzero, since the density ratio multiplies by it" + note
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
                    "so declare the derivative of the inverse map" + note
                )


def _check_ties(
    name: str,
    branch: _PieceBranch,
    a: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
    *,
    atoms: tuple[float, ...] = (),
) -> None:
    """Refuse a policy that collapses many observed doses onto one assigned dose.

    Two conditions together, so a coarse dose grid with a legitimate piecewise map is not
    refused.  The busiest assigned dose holds more than ``_TIE_EXCESS`` more of the rows
    than the busiest observed dose, and it receives more distinct observed doses than the
    policy has pieces.  A strictly monotone map sends at most one dose per piece to a value
    (per covariate pattern), so only a collapse meets both.  A dose sent to a declared atom
    is a declared collapse, which the mixed ratio reads, so those rows are not counted.
    """
    if not rows.any():
        return
    observed = a[rows]
    assigned = np.asarray(branch.assign(a, frame), dtype=float)[rows]
    if atoms:
        keep = _atom_index(assigned, atoms) < 0
        if not keep.any():
            return
        observed, assigned = observed[keep], assigned[keep]
    _, observed_counts = np.unique(observed, return_counts=True)
    values, assigned_counts = np.unique(assigned, return_counts=True)
    total = float(observed.size)
    share = float(assigned_counts.max()) / total
    value = float(values[int(np.argmax(assigned_counts))])
    sources = np.unique(observed[assigned == value]).size
    if share - float(observed_counts.max()) / total > _TIE_EXCESS and sources > len(branch.pieces):
        collapsed = observed[assigned == value]
        raise _collapse_refusal(name, where, float(collapsed.min()), float(collapsed.max()), value)


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


# ------------------------------------------------------------- joint policies


@dataclass(frozen=True)
class JointComponent:
    """One component of a vector node, as a :class:`JointPolicy` reads it.

    Parameters
    ----------
    name : str
        The component's column.
    levels : tuple
        The component's labels in code order, or ``()`` for a continuous component.
    """

    name: str
    levels: tuple[object, ...] = ()

    @property
    def continuous(self) -> bool:
        """Whether the component holds a continuous dose."""
        return not self.levels

    def labels(self, values: FloatArray) -> Any:
        """The natural values of the component: labels by code, or the doses."""
        if self.continuous:
            return np.asarray(values, dtype=float)
        table = np.empty(len(self.levels), dtype=object)
        for code, level in enumerate(self.levels):
            table[code] = level
        return table[np.nan_to_num(np.asarray(values, dtype=float), nan=0.0).astype(np.int64)]


#: Builds the policy frame of a node with the given extra columns appended.
FrameWith: TypeAlias = Callable[[Mapping[str, Any]], Any]


def joint_assignments(
    policy: JointPolicy,
    components: Sequence[JointComponent],
    natural: FloatArray,
    frame_with: FrameWith,
    rows: BoolArray,
    *,
    where: str,
    supports: Mapping[str, Support] | None = None,
    regimen: str | None = None,
    atoms: Mapping[str, Sequence[float]] | None = None,
) -> tuple[tuple[float, FloatArray], ...]:
    """Each branch of a joint policy as the ``(n, d)`` component values it assigns.

    A categorical component is assigned by its label map, read at each row's natural label.
    A continuous component is checked by :func:`check_continuous_policy` on the rows that
    hold a dose, and then mapped.  Each component's frame follows the rules of
    :class:`JointPolicy`.

    Parameters
    ----------
    policy : JointPolicy
        The declared joint policy.
    components : sequence of JointComponent
        The node's components, in column order.
    natural : ndarray
        ``(n, d)`` natural values: the level code of a categorical component and the dose of
        a continuous one, filled with zero on rows that do not hold the node.
    frame_with : callable
        Returns the node's policy frame with the given columns appended.
    rows : ndarray
        ``(n,)`` the rows that hold the node.  Only these are checked, and every other row
        is set to zero.
    where : str
        The node, for the refusals.
    supports : mapping of str to Support or None
        The declared support of each continuous component that declares one, checked by
        :func:`check_component_supports`.
    regimen : str or None
        The label the support refusal names; ``None`` names the policy.
    atoms : mapping of str to sequence of float or None
        The declared atoms of each continuous component that declares them.  Each dose map
        is then checked for the existence condition of the mixed law on the observed
        frame (R-b1, R-b3).

    Returns
    -------
    tuple of tuple
        ``(probability, values)`` per branch, with ``values`` the ``(n, d)`` assigned codes
        and doses.

    Raises
    ------
    CapabilityError
        If a continuous component's policy fails :func:`check_continuous_policy`.
    DataError
        If the policy names a column that is not a component of the node, maps a label off
        the component's levels, or a dose map reads another continuous component, or a
        dose leaves its declared support.
    """
    _check_tilt_components(policy, components, where)
    values = np.asarray(natural, dtype=float)
    out = []
    for probability, declared in joint_branches(policy):
        assigned = _joint_branch(
            policy.name,
            declared,
            components,
            values,
            frame_with,
            rows,
            where,
            policy.order,
            atoms,
        )
        out.append((probability, np.where(rows[:, None], assigned, 0.0)))
    if supports:
        check_component_supports(
            policy.name,
            components,
            values,
            [assigned for _, assigned in out],
            frame_with,
            rows,
            supports,
            regimen=policy.name if regimen is None else regimen,
        )
    return tuple(out)


def joint_discrete_assignments(
    policy: JointPolicy,
    components: Sequence[JointComponent],
    levels: Sequence[tuple[object, ...]],
    frame_with: FrameWith,
    rows: BoolArray,
    *,
    where: str,
) -> tuple[tuple[float, IntArray], ...]:
    """A joint policy on an all-categorical vector node, as :func:`discrete_assignments`.

    The node's levels are the observed tuples of its components.  Each branch is read at
    every level, so the result feeds the discrete formula exactly as a label map of the
    tuples does.

    Parameters
    ----------
    policy : JointPolicy
        The declared joint policy.
    components : sequence of JointComponent
        The node's components, each with its observed labels.
    levels : sequence of tuple
        The node's levels in code order.
    frame_with : callable
        Returns the node's policy frame with the given columns appended.
    rows : ndarray
        ``(n,)`` the rows that hold the node.
    where : str
        The node, for the refusals.

    Returns
    -------
    tuple of tuple
        ``(probability, codes)`` per branch, with ``codes`` an ``(n, K)`` integer array.

    Raises
    ------
    DataError
        If a branch sends a row to a tuple that is not a level of the node.
    """
    _check_tilt_components(policy, components, where)
    n = int(rows.size)
    lookup = {tuple(level): code for code, level in enumerate(levels)}
    out = []
    for probability, declared in joint_branches(policy):
        codes = np.zeros((n, len(levels)), dtype=np.int64)
        for k, level in enumerate(levels):
            natural = np.empty((n, len(components)))
            for index, (component, value) in enumerate(zip(components, level, strict=True)):
                natural[:, index] = float(component.levels.index(value))
            assigned = _joint_branch(
                policy.name, declared, components, natural, frame_with, rows, where, policy.order
            )
            for row in np.flatnonzero(rows):
                pair = tuple(
                    component.levels[int(assigned[row, index])]
                    for index, component in enumerate(components)
                )
                if pair not in lookup:
                    raise DataError(
                        f"policy {policy.name!r} maps {level!r} to {pair!r} at {where}, which "
                        "is not a level of the node. A policy on a categorical vector node "
                        "maps observed tuples to observed tuples (Díaz et al. 2023, "
                        "Assumption 1)"
                    )
                codes[row, k] = lookup[pair]
        out.append((float(probability), codes))
    return tuple(out)


def dose_order(
    name: str, order: Sequence[str] | None, components: Sequence[JointComponent], where: str
) -> tuple[int, ...]:
    """The column indices of a node's doses in the order a joint policy declares.

    ``None`` takes the column order.  A declared order lists every dose and may list
    categorical components, whose position has no effect.

    Parameters
    ----------
    name : str
        The policy's name, for the refusals.
    order : sequence of str or None
        The declared order.
    components : sequence of JointComponent
        The node's components, in column order.
    where : str
        The node, for the refusals.

    Returns
    -------
    tuple of int
        The doses' column indices, in the declared order.

    Raises
    ------
    DataError
        If the order names a column that is not a component of the node, or leaves out a
        dose.
    """
    names = [component.name for component in components]
    doses = [index for index, component in enumerate(components) if component.continuous]
    if order is None:
        return tuple(doses)
    unknown = [item for item in order if item not in names]
    if unknown:
        raise DataError(
            f"policy {name!r} at {where} declares order={tuple(order)!r}, which names "
            f"{unknown}; the node's components are {names}"
        )
    missing = [names[index] for index in doses if names[index] not in order]
    if missing:
        raise DataError(
            f"policy {name!r} at {where} declares order={tuple(order)!r}, which leaves out the "
            f"dose(s) {missing}. The order lists every continuous component of the node"
        )
    return tuple(names.index(item) for item in order if components[names.index(item)].continuous)


class ForbiddenRead(DataError):
    """A component map read a component of its node that its frame does not hold.

    Parameters
    ----------
    column : str
        The component the map read.
    """

    def __init__(self, column: str) -> None:
        super().__init__(f"a component map read {column!r}, which its frame does not hold")
        self.column = column


class GuardedFrame:
    """A component map's frame, which refuses every read of a component it does not hold.

    A map of a vector node may read the policy frame and some components of the node; the
    rules are those of :class:`JointPolicy`.  The other components are not columns of the
    frame, so a plain frame would return a default for ``h.get("D2", 0)``, and a map that
    caught its own ``KeyError`` would read nothing.  This proxy raises
    :class:`ForbiddenRead`, a :class:`~cleverly.exceptions.DataError`, on any access by name
    to such a component: indexing, ``get``, membership and attribute access.  Every other
    access goes to the frame.

    A map can catch the exception, so the proxy also records each refused read in
    :attr:`violations`, and :class:`GuardedSource` checks the record after the map returns.

    Parameters
    ----------
    frame : object
        The frame the map reads.
    forbidden : frozenset of str
        The components of the node that the map may not read.
    """

    def __init__(self, frame: Any, forbidden: frozenset[str]) -> None:
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_forbidden", forbidden)
        object.__setattr__(self, "violations", [])

    def _refuse(self, column: str) -> ForbiddenRead:
        self.violations.append(column)
        return ForbiddenRead(column)

    def _check(self, key: Any) -> None:
        keys = list(key) if isinstance(key, (list, tuple)) else [key]
        for item in keys:
            if isinstance(item, str) and item in self._forbidden:
                raise self._refuse(item)

    def __getitem__(self, key: Any) -> Any:
        self._check(key)
        return self._frame[key]

    def get(self, key: Any, *default: Any) -> Any:
        """Return a column, or the default, after the read check.

        Parameters
        ----------
        key : object
            The column.
        *default : object
            The default, as the frame's own ``get`` takes it.

        Returns
        -------
        object
            The frame's answer.
        """
        self._check(key)
        return self._frame.get(key, *default)

    def __contains__(self, key: Any) -> bool:
        self._check(key)
        return key in self._frame

    def __getattr__(self, attribute: str) -> Any:
        if attribute in self._forbidden:
            raise self._refuse(attribute)
        return getattr(self._frame, attribute)

    def __len__(self) -> int:
        return len(self._frame)

    def __iter__(self) -> Any:
        return iter(self._frame)


class _RecordingFrame(GuardedFrame):
    """A guarded frame that also records which doses of the node a map reads."""

    def __init__(self, frame: Any, forbidden: frozenset[str], doses: frozenset[str]) -> None:
        super().__init__(frame, forbidden)
        object.__setattr__(self, "_doses", doses)
        object.__setattr__(self, "seen", set())

    def _check(self, key: Any) -> None:
        keys = list(key) if isinstance(key, (list, tuple)) else [key]
        self.seen.update(item for item in keys if isinstance(item, str) and item in self._doses)
        super()._check(key)

    def __getattr__(self, attribute: str) -> Any:
        if attribute in self._doses:
            self.seen.add(attribute)
        return super().__getattr__(attribute)


def partial_frame(frame: Any) -> FrameSource:
    """A frame source that returns ``frame``.

    Parameters
    ----------
    frame : object
        The frame.

    Returns
    -------
    callable
        Returns ``frame``.
    """
    return lambda: frame


class GuardedSource:
    """A guarded frame, built once on first use, that can be asked for refused reads later.

    Parameters
    ----------
    frame_with : callable
        Builds the node's policy frame with extra columns.
    extra : mapping
        The component columns the map may read.
    forbidden : frozenset of str
        The components it may not read.
    """

    def __init__(
        self, frame_with: FrameWith, extra: Mapping[str, Any], forbidden: frozenset[str]
    ) -> None:
        self._frame_with = frame_with
        self._extra = extra
        self._forbidden = forbidden
        self._frame: GuardedFrame | None = None

    def __call__(self) -> GuardedFrame:
        if self._frame is None:
            self._frame = GuardedFrame(self._frame_with(self._extra), self._forbidden)
        return self._frame

    def check(self) -> None:
        """Raise the first refused read, which a map that caught it did not let through.

        Raises
        ------
        ForbiddenRead
            If a map read a component that its frame does not hold.
        """
        if self._frame is not None and self._frame.violations:
            raise ForbiddenRead(self._frame.violations[0])


def guarded_source(
    frame_with: FrameWith, extra: Mapping[str, Any], forbidden: Any
) -> GuardedSource:
    """The policy frame with ``extra`` appended, guarded, and built once on first use.

    Parameters
    ----------
    frame_with : callable
        Builds the node's policy frame with extra columns.
    extra : mapping
        The component columns the map may read.
    forbidden : iterable of str
        The components it may not read.

    Returns
    -------
    GuardedSource
        Returns the guarded frame; :meth:`GuardedSource.check` raises a read a map caught.
    """
    return GuardedSource(frame_with, extra, frozenset(forbidden))


def _joint_branch(
    name: str,
    declared: Mapping[str, Any],
    components: Sequence[JointComponent],
    natural: FloatArray,
    frame_with: FrameWith,
    rows: BoolArray,
    where: str,
    order: Sequence[str] | None = None,
    atoms: Mapping[str, Sequence[float]] | None = None,
) -> FloatArray:
    """One deterministic branch of a joint policy, applied component by component.

    The categorical maps read the natural value of every other component.  Each dose map
    reads the natural categorical values and the natural doses before it in ``order``, so a
    map never reads a value this function has assigned.  ``atoms`` holds each continuous
    component's declared atoms, which its checks read.
    """
    names = [component.name for component in components]
    unknown = sorted(set(declared) - set(names))
    if unknown:
        raise DataError(
            f"policy {name!r} at {where} declares component(s) {unknown}, and the node's "
            f"components are {names}"
        )
    doses = dose_order(name, order, components, where)
    labels = {
        component.name: component.labels(natural[:, index])
        for index, component in enumerate(components)
    }
    assigned = np.array(natural, dtype=float, copy=True)
    try:
        for index, component in enumerate(components):
            if component.continuous or component.name not in declared:
                continue
            here = f"component {component.name!r} of {where}"
            readable = {
                other.name: labels[other.name] for other in components if other is not component
            }
            source = guarded_source(frame_with, readable, (component.name,))
            assigned[:, index] = _categorical_component(
                name,
                declared[component.name],
                component,
                natural[:, index],
                source,
                rows,
                here,
            )
            source.check()
        for position, index in enumerate(doses):
            component = components[index]
            if component.name not in declared:
                continue
            here = f"component {component.name!r} of {where}"
            earlier = [components[before].name for before in doses[:position]]
            # The natural categorical values, then the natural doses before this one.
            readable = {
                other.name: labels[other.name] for other in components if not other.continuous
            }
            readable.update({other: labels[other] for other in earlier})
            source = guarded_source(frame_with, readable, set(names) - set(readable))
            assigned[:, index] = _continuous_component(
                declared[component.name],
                natural[:, index],
                source,
                rows,
                here,
                (atoms or {}).get(component.name, ()),
            )
            source.check()
    except ForbiddenRead as error:
        refusal = read_order_refusal(
            name, declared, components, natural, frame_with, rows, where, doses
        )
        raise DataError(refusal) from error
    return assigned


def map_reads(
    declared: Mapping[str, Any],
    components: Sequence[JointComponent],
    natural: FloatArray,
    frame_with: FrameWith,
    rows: BoolArray,
) -> dict[str, set[str]]:
    """Which other doses each dose map of one branch reads, by running it on a full frame.

    Parameters
    ----------
    declared : mapping
        The branch's component policies.
    components : sequence of JointComponent
        The node's components.
    natural : ndarray
        ``(n, d)`` natural values.
    frame_with : callable
        Builds the node's policy frame with extra columns.
    rows : ndarray
        The rows that hold the node.

    Returns
    -------
    dict of str to set of str
        Each declared dose's name, mapped to the other doses its map reads.
    """
    labels = {
        component.name: component.labels(natural[:, index])
        for index, component in enumerate(components)
    }
    doses = frozenset(component.name for component in components if component.continuous)
    reads: dict[str, set[str]] = {}
    for index, component in enumerate(components):
        if not component.continuous or component.name not in declared:
            continue
        others = {key: value for key, value in labels.items() if key != component.name}
        recording = _RecordingFrame(frame_with(others), frozenset({component.name}), doses)
        # Only the reads are wanted here; the fit reports any error of the map itself.
        with contextlib.suppress(Exception):
            _continuous_component(
                declared[component.name], natural[:, index], partial_frame(recording), rows, ""
            )
        reads[component.name] = set(recording.seen) - {component.name}
    return reads


def read_order_refusal(
    name: str,
    declared: Mapping[str, Any],
    components: Sequence[JointComponent],
    natural: FloatArray,
    frame_with: FrameWith,
    rows: BoolArray,
    where: str,
    doses: Sequence[int],
) -> str:
    """The refusal of a map that reads a component its frame does not hold.

    A dose map that reads a later dose is a declaration error: any acyclic read graph has an
    order that makes the map triangular, and the refusal names one.  A cycle has none, and a
    joint map that is not triangular has no checkable declaration, because a determinant
    checked on a grid does not establish that the map is one-to-one.

    Parameters
    ----------
    name : str
        The policy's name.
    declared : mapping
        The branch's component policies.
    components : sequence of JointComponent
        The node's components.
    natural : ndarray
        ``(n, d)`` natural values.
    frame_with : callable
        Builds the node's policy frame with extra columns.
    rows : ndarray
        The rows that hold the node.
    where : str
        The node.
    doses : sequence of int
        The doses' column indices in the declared order.

    Returns
    -------
    str
        The refusal.
    """
    reads = map_reads(declared, components, natural, frame_with, rows)
    current = [components[index].name for index in doses]
    placed: list[str] = []
    remaining = list(current)
    while remaining:
        ready = [dose for dose in remaining if reads.get(dose, set()) <= set(placed)]
        if not ready:
            return (
                f"policy {name!r} at {where}: the maps of the doses {sorted(remaining)} read "
                "each other in a cycle, so no order makes the joint map triangular. A joint "
                "map that is not triangular has no checkable declaration: a Jacobian "
                "determinant checked on a grid does not establish that the map is one-to-one, "
                "which the density ratio of Díaz et al. (2023), Equation (3), needs. Declare "
                "dose maps that read earlier doses only"
            )
        placed.append(ready[0])
        remaining.remove(ready[0])
    offending = [
        f"{dose!r} reads {sorted(reads[dose] - set(current[: current.index(dose)]))}"
        for dose in current
        if reads.get(dose, set()) - set(current[: current.index(dose)])
    ]
    if not offending:
        return (
            f"policy {name!r} at {where}: a component map reads a component of the node that "
            "its frame does not hold. A categorical map reads every other component; a dose "
            "map reads the natural categorical components and the doses before it in order, "
            "and no map reads its own column, which it receives as its first argument"
        )
    return (
        f"policy {name!r} at {where}: in the dose order {tuple(current)!r}, "
        f"{'; '.join(offending)}. A dose map reads the policy frame, the natural categorical "
        "components and the natural doses before it in order. Declare JointPolicy(..., "
        f"order={tuple(placed)!r}), which puts every dose after the doses its map reads"
    )


def _continuous_component(
    policy: Any,
    dose: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
    atoms: Sequence[float] = (),
) -> FloatArray:
    """A continuous component's assigned dose, after the Assumption-4 and atom checks.

    With declared ``atoms`` the checks of the mixed law run on the observed frame: every
    collapse target is a declared atom (R-b1) and every atom image is one (R-b3).
    """
    observed = np.asarray(dose, dtype=float)
    check_continuous_policy(policy, observed, frame, rows, where=where, atoms=atoms)
    ((_, branch),) = policy_branches(policy)
    return _per_row(np.asarray(branch.assign(observed, frame), dtype=float), observed.size)


def _categorical_component(
    name: str,
    policy: Any,
    component: JointComponent,
    codes: FloatArray,
    frame: FrameSource,
    rows: BoolArray,
    where: str,
) -> FloatArray:
    """A categorical component's assigned level codes, read at each row's natural label."""
    levels = component.levels
    numeric = _numeric_levels(levels)
    ((_, branch),) = policy_branches(policy)
    if branch.arithmetic and not numeric:
        raise DataError(
            f"policy {name!r} adds, scales or splits a dose, but {where} is categorical with "
            f"levels {list(levels)!r}. Declare ModifiedPolicy(apply=...) that maps labels to "
            "labels"
        )
    natural = component.labels(codes)
    n = int(natural.size)
    source: Any = np.asarray(natural, dtype=float) if numeric else natural
    assigned = _labels(branch.assign(source, frame))
    if assigned.size != n:
        raise DataError(
            f"policy {name!r} at {where} returned {assigned.size} values for {n} rows; it must "
            "return one per row"
        )
    out = np.zeros(n)
    matched = np.zeros(n, dtype=bool)
    for code, level in enumerate(levels):
        here = _equal(assigned, level, numeric) & ~matched
        out[here] = float(code)
        matched |= here
    unknown = rows & ~matched
    if unknown.any():
        value = assigned[np.flatnonzero(unknown)[0]]
        raise DataError(
            f"policy {name!r} maps {int(unknown.sum())} rows of {where} to values that are not "
            f"levels of the component (first: {value!r}). A categorical component maps "
            "levels to levels (Díaz et al. 2023, Assumption 1)"
        )
    return out


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
        and a level code on a categorical one.  ``(n, C, d)`` on a vector treatment with a
        continuous component: the component codes and doses each branch assigns.
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
        dose; on a vector treatment, whether a cap held any of its doses.  ``None`` on a
        categorical treatment, where no cap is evaluated.
    joint : JointDensityParts or None
        On a vector treatment with a continuous component, fitted on the density route: the
        factors of the node's law at the observed value and at each branch's assigned
        value (:mod:`~cleverly.interventions.joint_density`).  :meth:`design_at` rebuilds
        the ratios from them with the categorical factor bounded.  ``None`` elsewhere.
    component_rows : ndarray or None
        ``(n, C)`` boolean, the rows each component's mean and curve read, or ``None`` when
        every component reads every row.  A fold-average policy (X40 part (e1)) has one
        component per outer fold, at the map fitted on that fold's training rows, and the
        component reads only the fold's held-out rows.  Its ratio is zero on the other rows,
        so its fluctuation solves the fold's own score.
    curve_terms : ndarray or None
        ``(n, S)`` the term each policy's curve adds on the scaled outcome scale, or
        ``None``.  A population-target policy with a tuned cap adds
        :math:`\\partial\\Phi_n' \\mathrm{IF}` (X40 part (e2)); every other policy adds zero.

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
    joint: JointDensityParts | None = None
    component_rows: BoolArray | None = None
    curve_terms: FloatArray | None = None

    def __post_init__(self) -> None:
        s = len(self.names)
        if len(self.descriptions) != s:
            raise ValueError(f"{s} policy names but {len(self.descriptions)} descriptions")
        if len(set(self.names)) != s:
            raise ValueError(f"policy names must be distinct; got {list(self.names)}")
        n, c = np.asarray(self.shifted).shape[:2]
        for name, array, shape in (
            ("ratio", self.ratio, (n, c)),
            ("moved", self.moved, (n, c)),
            *(() if self.capped is None else (("capped", self.capped, (n, c)),)),
            *(
                ()
                if self.component_rows is None
                else (("component_rows", self.component_rows, (n, c)),)
            ),
            *(() if self.curve_terms is None else (("curve_terms", self.curve_terms, (n, s)),)),
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
        density: ConditionalDensity | MixedDensity | None = None,
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
        density : ConditionalDensity, MixedDensity or None
            Estimated conditional density of the dose given the covariates, on a
            continuous treatment.  A dose with declared atoms reads a
            :class:`~cleverly.learners.density.MixedDensity`.
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
        by_fold: Sequence[Sequence[object]] | None = None,
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
        by_fold : sequence or None
            Per outer fold, the policies that fold's classifier stacks, in place of
            ``policies``, with the same branches: a population-target tuned cap at the
            tuner fitted on that fold's training rows
            (``test_t_e7_each_fold_trains_at_its_own_tuner``).  The ratio
            is still evaluated at the doses ``policies`` assign.  ``None`` stacks
            ``policies`` in every fold.

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
        if data.treatment_kind == "vector":
            return cls._evaluate_vector(
                declared, names, data, learner, folds, frame, reference=reference, n_jobs=n_jobs
            )
        a = np.asarray(data.treatment, dtype=float).reshape(-1)
        owners: list[tuple[int, float]] = []
        columns: list[FloatArray] = []
        held: list[BoolArray] | None = None
        if data.is_continuous_treatment:
            held = []
            for index, policy in enumerate(declared):
                check_continuous_policy(
                    policy,
                    a,
                    frame,
                    everyone,
                    support=data.treatment_support,
                    atoms=data.dose_atoms,
                    node=data.treatment_name,
                )
                for probability, branch in policy_branches(policy):
                    owners.append((index, probability))
                    columns.append(np.asarray(branch.assign(a, frame), dtype=float))
                    held.append(branch.held(a))
        else:
            levels = tuple(data.arm_label(code) for code in data.arm_codes)
            observed_codes = a.astype(np.int64)
            where = f"treatment {data.treatment_name!r}"
            for index, policy in enumerate(declared):
                for probability, codes in _point_discrete_assignments(
                    policy, data, levels, frame, everyone, where=where
                ):
                    owners.append((index, probability))
                    columns.append(codes[np.arange(data.n), observed_codes].astype(float))
        shifted = np.column_stack(columns)
        count = shifted.shape[1]
        fold_columns: list[FloatArray] | None = None
        if by_fold is not None:
            if not data.is_continuous_treatment:
                raise ValueError("by_fold applies to a continuous treatment of one column")
            fold_columns = []
            for fold_policies in by_fold:
                stacked = [
                    np.asarray(branch.assign(a, frame), dtype=float)
                    for policy in fold_policies
                    for _, branch in policy_branches(policy)
                ]
                if len(stacked) != count:
                    raise ValueError("each fold's policies need the branches of policies=")
                fold_columns.append(np.column_stack(stacked))
        ratio = np.zeros((data.n, count))
        ratio_at = np.zeros((data.n, count, count))
        # The classifier reads a vector's components as indicators, never the joint level
        # code, whose order is arbitrary.  A treatment of one column keeps its own value,
        # and ``atoms=`` adds the indicator of each declared atom.
        block = data.component_design if data.is_vector_treatment else _same
        atoms = tuple(data.dose_atoms) if data.is_continuous_treatment else ()
        for c in range(count):
            evaluated = classifier_ratio(
                learner,
                data.covariates,
                block(a),
                ((1.0, block(shifted[:, c])),),
                data.weights,
                folds,
                evaluate_at=[block(a), *(block(shifted[:, s]) for s in range(count))],
                groups=data.cluster,
                n_jobs=n_jobs,
                atoms=atoms,
                assigned_by_fold=(
                    None
                    if fold_columns is None
                    else [((1.0, block(fold[:, c])),) for fold in fold_columns]
                ),
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

    @classmethod
    def _evaluate_vector(
        cls,
        declared: tuple[object, ...],
        names: tuple[str, ...],
        data: CausalData,
        learner: Any,
        folds: Folds,
        frame: FrameSource,
        *,
        reference: str | None,
        n_jobs: int,
    ) -> PolicySet:
        """:meth:`evaluate_by_classifier` on a vector treatment with a continuous component.

        Each policy is a :class:`JointPolicy`, applied component by component by
        :func:`joint_assignments`.  The stacked classifier reads
        :meth:`~cleverly.data.CausalData.component_design` beside the covariates, so it sees
        every component.  ``shifted`` is ``(n, C, d)``.
        """
        from ..learners.density_ratio import classifier_ratio

        setup = _vector_setup(declared, data)
        natural = setup.natural
        for (_, policy, _), values in zip(setup.owners, setup.assigned, strict=True):
            _warn_outside_cell_support(policy.name, setup.components, natural, values)
        shifted = np.stack(setup.assigned, axis=1)
        count = shifted.shape[1]
        observed = data.component_design()
        assigned = [data.component_design(shifted[:, c]) for c in range(count)]
        ratio = np.zeros((data.n, count))
        ratio_at = np.zeros((data.n, count, count))
        for c in range(count):
            evaluated = classifier_ratio(
                learner,
                data.covariates,
                observed,
                ((1.0, assigned[c]),),
                data.weights,
                folds,
                evaluate_at=[observed, *assigned],
                groups=data.cluster,
                n_jobs=n_jobs,
            )
            ratio[:, c] = evaluated[0]
            for s in range(count):
                ratio_at[:, s, c] = evaluated[s + 1]
        return cls._from_vector(declared, names, setup, shifted, ratio, ratio_at, reference)

    @classmethod
    def evaluate_vector_density(
        cls,
        policies: Sequence[object],
        data: CausalData,
        learner: Any,
        folds: Folds,
        *,
        reference: str | None = None,
        n_bins: int | None = None,
        n_jobs: int = 1,
    ) -> PolicySet:
        """Evaluate joint policies at a vector treatment with a continuous component, by density.

        The treatment's law is fitted by the chain rule, and each policy's induced density is
        the sum over categorical preimages and piece paths of
        :mod:`~cleverly.interventions.joint_density`.  The ratio of component ``c`` is
        evaluated at the observed value and at every component's assigned value, from the
        same out-of-fold factors.  :meth:`design_at` rebuilds both with the categorical
        factor bounded, as for a categorical treatment.

        Parameters
        ----------
        policies : sequence of JointPolicy
            The policies to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data with a vector treatment.
        learner : Learner
            The classifier of the categorical factor and of every dose's hazards.
        folds : Folds
            The outer split.
        reference : str or None
            Label of the policy contrasts are taken against.
        n_bins : int or None
            The bin count of each dose's density.
        n_jobs : int
            Parallel workers across the outer folds.

        Returns
        -------
        PolicySet
            Every policy evaluated on the data, with the joint factors kept.
        """
        from .joint_density import (
            JointBranch,
            JointEvaluation,
            NodeLayout,
            evaluate_joint_density,
            warn_unsupported_values,
        )

        declared = tuple(policies)
        if not declared:
            raise DataError("at least one policy is required")
        names = tuple(str(getattr(policy, "name", "")) for policy in declared)
        if len(set(names)) != len(names):
            raise DataError(f"policy names must be distinct; got {list(names)}")
        refuse_policy_declarations(declared)
        setup = _vector_setup(declared, data)
        shifted = np.stack(setup.assigned, axis=1)
        count = shifted.shape[1]
        layout = NodeLayout(
            setup.components,
            setup.natural,
            np.ones(data.n, dtype=bool),
            setup.frame_with,
            np.asarray(data.covariates, dtype=float),
            setup.where,
            component_atom_layout(setup.components, data.component_atoms),
        )
        branches = [
            JointBranch(policy.name, mapping, policy.order) for _, policy, mapping in setup.owners
        ]
        everything = tuple(range(count))
        evaluations = [JointEvaluation(setup.natural, everything)] + [
            JointEvaluation(shifted[:, s], everything) for s in range(count)
        ]
        parts = evaluate_joint_density(
            layout,
            branches,
            evaluations,
            learner=learner,
            weights=data.weights,
            folds=folds,
            n_bins=n_bins,
            groups=data.cluster,
            n_jobs=n_jobs,
        )
        full = parts.ratio()
        moved = np.any(shifted != setup.natural[:, None, :], axis=2)
        for c, (_, policy, mapping) in enumerate(setup.owners):
            if not moved[:, c].any():
                continue
            for index, component in enumerate(setup.components):
                if (
                    component.continuous
                    and component.name in mapping
                    and component.name in parts.densities
                ):
                    # Only the rows whose dose this policy changes: a gated map leaves
                    # the others alone, and they say nothing about the bin width.  A dose
                    # whose every row holds a declared atom fits no density to resolve.
                    changed = shifted[:, c, index] != setup.natural[:, index]
                    if not changed.any():
                        continue
                    warn_if_unresolved(
                        parts.densities[component.name],
                        shifted[changed, c, index],
                        setup.natural[changed, index],
                        component=component.name,
                    )
            warn_unsupported_values(policy.name, parts, c + 1, moved[:, c], setup.where)
        return cls._from_vector(
            declared, names, setup, shifted, full[:, 0, :], full[:, 1:, :], reference, parts
        )

    @classmethod
    def _from_vector(
        cls,
        declared: tuple[object, ...],
        names: tuple[str, ...],
        setup: _VectorSetup,
        shifted: FloatArray,
        ratio: FloatArray,
        ratio_at: FloatArray,
        reference: str | None,
        joint: JointDensityParts | None = None,
    ) -> PolicySet:
        """The policy set of a vector treatment from its evaluated ratios."""
        moved = np.asarray(np.any(shifted != setup.natural[:, None, :], axis=2), dtype=bool)
        # A component's cap holds a row at its own dose, as a scalar branch's ``held`` reads
        # it; the component is capped at the row when any of its doses is.
        capped = np.zeros(moved.shape, dtype=bool)
        for column, (_, _, mapping) in enumerate(setup.owners):
            for index, component in enumerate(setup.components):
                mapped = mapping.get(component.name)
                if component.continuous and isinstance(mapped, (Shift, Scale)):
                    held = mapped.apply(setup.natural[:, index])[1]
                    capped[:, column] |= np.asarray(held, dtype=bool)
        owners = [
            (index, probability)
            for (index, _, _), probability in zip(setup.owners, setup.probabilities, strict=True)
        ]
        mixing = None if len(owners) == len(declared) else _mixing(len(declared), owners)
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
            capped=capped,
            joint=joint,
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
        the policy's induced density: :math:`g^d = \\sum_e p(e)\\, g^{d_e}`.  A
        fold-average policy reads, at each row, the ratio of the component that holds the
        row out, so the weights are renormalised over the components that read the row.
        """
        if self.mixing is None:
            return np.asarray(self.ratio, dtype=float)
        if self.component_rows is None:
            return np.asarray(self.ratio @ self.weights.T, dtype=float)
        rows = np.asarray(self.component_rows, dtype=float)
        mass = rows @ self.weights.T
        safe = np.where(mass > 0.0, mass, 1.0)
        return np.asarray((self.ratio * rows) @ self.weights.T / safe, dtype=float)

    @property
    def design(self) -> FloatArray:
        """``(n, C + 1, C)`` -- the covariate at the observed dose, then at each assigned one.

        One array rather than two, because :func:`~cleverly.fluctuation.submodel_for`
        dispatches on the group name alone and every builder takes the same keyword-only
        signature; ``policies=`` is that one keyword.  Row block ``0`` is the covariate at
        the observed treatment and block ``s + 1`` the covariate at :math:`d_s(A, W)`.
        """
        return np.concatenate([self.ratio[:, None, :], self.ratio_at], axis=1)

    def design_at(
        self, propensity: FloatArray | None, bounds: tuple[float, float] | None = None
    ) -> FloatArray:
        """:attr:`design`, rebuilt from a bounded mechanism on a categorical treatment.

        On a continuous treatment this is :attr:`design`, since the density ratio is not
        bounded.  On a categorical one the ratio reads the fitted probabilities, and
        targeting passes them bounded by ``g_bounds``, as for every other arm-indexed
        clever covariate.  On a vector treatment fitted on the density route the
        categorical factor is bounded into ``bounds`` by the same rule, in the numerator
        and the denominator, and the doses' factors are not bounded.

        Parameters
        ----------
        propensity : ndarray or None
            ``(n, K)`` bounded treatment probabilities.
        bounds : tuple of float or None
            The bounds of the categorical factor of a vector treatment.  ``None`` reads
            the factor unbounded.

        Returns
        -------
        ndarray
            ``(n, C + 1, C)``.
        """
        if self.joint is not None:
            if bounds is None:
                return self.design
            from .joint_density import bound_cells

            full = self.joint.ratio(bound_cells(self.joint.cell_probabilities, bounds))
            return np.asarray(full, dtype=float)
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
            joint=None if self.joint is None else self.joint.subset(idx),
            component_rows=None if self.component_rows is None else self.component_rows[idx],
            curve_terms=None if self.curve_terms is None else self.curve_terms[idx],
        )


@dataclass(frozen=True)
class _VectorSetup:
    """A point vector treatment's components and each joint policy branch's assigned values."""

    components: tuple[JointComponent, ...]
    natural: FloatArray
    frame_with: FrameWith
    where: str
    #: ``(policy index, policy, branch components)`` per branch, in component order.
    owners: tuple[tuple[int, JointPolicy, Mapping[str, Any]], ...]
    probabilities: tuple[float, ...]
    assigned: tuple[FloatArray, ...]


def preflight_vector_policies(
    policies: Sequence[object],
    data: CausalData,
    *,
    ratio: str,
    density_bins: int | None,
) -> None:
    """Apply each joint policy to a point vector before any learner, with every check.

    The undeclared-atom check of each continuous component (R-b4), a declared atom with no
    row (R-b2), the maps, Assumption 4 of each dose map, the existence condition of each
    mixed factor on the observed frame (R-b1, R-b3), and the declared support of each
    continuous component (R-a2, R-a1) read observed values only, so the fit runs them
    before its first learner.  The training-fold half of R-b2 runs in the nuisance fit.

    Parameters
    ----------
    policies : sequence of JointPolicy
        The declared policies.
    data : CausalData
        The validated data, with a vector treatment.
    ratio : {"density", "classifier"}
        The ratio route, which decides whether the bin count rule applies.
    density_bins : int or None
        The declared bin count of the density route, or ``None`` for the default.

    Raises
    ------
    CapabilityError
        If a policy is not a :class:`JointPolicy`, or a dose map fails
        :func:`check_continuous_policy`.
    DataError
        If a dose records an undeclared atom, a map is malformed, or a dose leaves its
        declared support.
    """
    assert data.treatment_components is not None
    components = tuple(
        JointComponent(name, levels)
        for name, levels in zip(data.component_names, data.component_levels, strict=True)
    )
    natural = np.asarray(data.treatment_components, dtype=float)
    everyone = np.ones(data.n, dtype=bool)
    if all(isinstance(policy, JointPolicy) for policy in policies):
        refuse_component_atoms(
            natural,
            components,
            everyone,
            moved=any(joint_policy_moves(policy) for policy in policies),
            ratio=ratio,
            density_bins=density_bins,
            atoms=data.component_atoms,
        )
    if data.component_atoms:
        check_component_atoms(
            natural, components, everyone, data.component_atoms, None, ratio=ratio
        )
    if data.component_supports or data.component_atoms:
        _vector_setup(tuple(policies), data)


def joint_policy_moves(policy: object) -> bool:
    """Whether some branch of a joint policy moves a component by its declaration.

    At a vector the ratio is one only when no component moves: a moved categorical
    component evaluates every dose factor at a counterfactual cell.

    Parameters
    ----------
    policy : object
        A declared policy.

    Returns
    -------
    bool
        ``False`` for a :class:`JointPolicy` whose every component policy is the identity.
    """
    if not isinstance(policy, JointPolicy):
        return True
    return any(
        policy_moves_dose(component)
        for _, components in joint_branches(policy)
        for component in components.values()
    )


def component_atom_layout(
    components: Sequence[JointComponent], atoms: Mapping[str, Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    """Each dose's declared atoms, in the column order of the doses, or ``()`` for none.

    Parameters
    ----------
    components : sequence of JointComponent
        The node's components, in column order.
    atoms : mapping of str to sequence of float
        The declared atoms of each continuous component that declares them.

    Returns
    -------
    tuple of tuple of float
        One entry per dose, empty when no dose declares atoms.
    """
    if not atoms:
        return ()
    return tuple(
        tuple(float(atom) for atom in atoms.get(component.name, ()))
        for component in components
        if component.continuous
    )


def check_component_atoms(
    natural: FloatArray,
    components: Sequence[JointComponent],
    rows: BoolArray,
    atoms: Mapping[str, Sequence[float]],
    folds: Folds | None,
    *,
    ratio: str,
) -> None:
    r"""R-b2 at each continuous component of a vector with declared atoms.

    Each declared atom needs rows at risk, overall and in every outer training fold, as at
    a dose of one column (:func:`check_declared_atoms`).  On the density route the factor
    at an atom divides by :math:`P(D_k = a \mid d_{<k}, c, h)`, which conditions on the
    categorical cell.  So each atom also needs rows in each training fold within every
    observed cell that records it.

    Parameters
    ----------
    natural : ndarray
        ``(n, d)`` natural values: level codes and doses.
    components : sequence of JointComponent
        The node's components, in column order.
    rows : ndarray
        ``(n,)`` the rows that hold the node.
    atoms : mapping of str to sequence of float
        The declared atoms of each continuous component that declares them.
    folds : Folds or None
        The outer split, or ``None`` for the overall check alone.
    ratio : {"density", "classifier"}
        The ratio route, which decides whether the cell rule applies.

    Raises
    ------
    DataError
        If a declared atom has no row at risk, none in some training fold, or none in some
        training fold within a cell that records it.
    """
    categorical = [index for index, component in enumerate(components) if not component.continuous]
    for index, component in enumerate(components):
        declared = tuple(atoms.get(component.name, ()))
        if not component.continuous or not declared:
            continue
        dose = np.asarray(natural[:, index], dtype=float)
        check_declared_atoms(dose, rows, declared, folds, node=component.name)
        if ratio != "density" or not categorical or folds is None or folds.is_single:
            continue
        which = _atom_index(dose, declared)
        codes = np.asarray(natural[:, categorical], dtype=np.int64)
        for k, atom in enumerate(declared):
            holds = rows & (which == k)
            for cell in np.unique(codes[holds], axis=0):
                in_cell = holds & np.all(codes == cell, axis=1)
                for fold, (train, _) in enumerate(folds, start=1):
                    if in_cell[train].any():
                        continue
                    label = ", ".join(
                        f"{components[c].name}={components[c].levels[int(code)]!r}"
                        for c, code in zip(categorical, cell, strict=True)
                    )
                    raise DataError(
                        f"dose_atoms declares {float(atom):g} at component {component.name!r}, "
                        f"but the training rows of outer fold {fold} hold no row at risk with "
                        f"that dose in the cell ({label}), which records it. A declared atom "
                        "needs recorded mass where the ratio divides by its probability: on "
                        "the density route the factor at the atom is "
                        f"P({component.name} = {float(atom):g} | earlier doses, cell, H), which "
                        "the data estimate only from rows of that cell that received it. "
                        "Collect more rows at the atom in that cell, or pass "
                        "ratio='classifier'"
                    )


def refuse_component_atoms(
    natural: FloatArray,
    components: Sequence[JointComponent],
    rows: BoolArray,
    *,
    moved: bool,
    ratio: str,
    density_bins: int | None,
    atoms: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """R-b4 at each continuous component of a vector: a recorded atom it does not declare.

    Parameters
    ----------
    natural : ndarray
        ``(n, d)`` natural values: level codes and doses.
    components : sequence of JointComponent
        The node's components, in column order.
    rows : ndarray
        ``(n,)`` the rows that hold the node.
    moved : bool
        Whether some policy at the node moves a component.
    ratio : {"density", "classifier"}
        The ratio route, which decides whether the bin count rule applies.
    density_bins : int or None
        The declared bin count of the density route, or ``None`` for the default.
    atoms : mapping of str to sequence of float or None
        The declared atoms of each continuous component, whose rows the rules do not read.

    Raises
    ------
    DataError
        If any rule flags a continuous component.
    """
    from ..learners.density import default_density_bins

    for index, component in enumerate(components):
        if not component.continuous:
            continue
        declared = tuple((atoms or {}).get(component.name, ()))
        doses = np.asarray(natural[rows, index], dtype=float)
        bins: int | None = None
        if ratio == "density":
            continuous = int(np.sum(_atom_index(doses, declared) < 0)) if declared else doses.size
            bins = density_bins if density_bins is not None else default_density_bins(continuous)
        _refuse_undeclared_atoms(
            doses, atoms=declared, moved=moved, bins=bins, where=component.name, component=True
        )


def _vector_setup(declared: tuple[object, ...], data: CausalData) -> _VectorSetup:
    """Apply each joint policy to a point vector treatment, with the observed-frame checks."""
    assert data.treatment_components is not None
    for policy in declared:
        if not isinstance(policy, JointPolicy):
            raise CapabilityError(whole_vector_refusal(policy, data.treatment_name))
    components = tuple(
        JointComponent(name, levels)
        for name, levels in zip(data.component_names, data.component_levels, strict=True)
    )
    natural = np.asarray(data.treatment_components, dtype=float)
    everyone = np.ones(data.n, dtype=bool)
    where = f"treatment {data.treatment_name!r}"

    def frame_with(extra: Mapping[str, Any]) -> Any:
        payload = {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        payload.update(extra)
        return data.frame_like(payload)

    owners: list[tuple[int, JointPolicy, Mapping[str, Any]]] = []
    probabilities: list[float] = []
    assigned: list[FloatArray] = []
    for index, policy in enumerate(declared):
        assert isinstance(policy, JointPolicy)
        evaluated = joint_assignments(
            policy,
            components,
            natural,
            frame_with,
            everyone,
            where=where,
            supports=data.component_supports,
            atoms=data.component_atoms,
        )
        for (probability, values), (_, mapping) in zip(
            evaluated, joint_branches(policy), strict=True
        ):
            owners.append((index, policy, mapping))
            probabilities.append(probability)
            assigned.append(np.asarray(values, dtype=float))
    return _VectorSetup(
        components, natural, frame_with, where, tuple(owners), tuple(probabilities), tuple(assigned)
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
    density: ConditionalDensity | MixedDensity,
    frame: FrameSource,
    rows: BoolArray,
) -> _Components:
    """The component arrays on a continuous treatment."""
    a = np.asarray(data.treatment, dtype=float).reshape(-1)
    for policy in policies:
        check_continuous_policy(
            policy,
            a,
            frame,
            rows,
            support=data.treatment_support,
            atoms=data.dose_atoms,
            node=data.treatment_name,
        )
    owners: list[tuple[int, float]] = []
    labels: list[str] = []
    branches: list[_Branch] = []
    for index, policy in enumerate(policies):
        for probability, branch in policy_branches(policy):
            owners.append((index, probability))
            labels.append(str(getattr(policy, "name", "")))
            branches.append(branch)
    shifted = np.column_stack([branch.assign(a, frame) for branch in branches])
    ratio = np.column_stack([branch_ratio(branch, a, frame, density) for branch in branches])
    ratio_at = np.stack(
        [
            np.column_stack(
                [branch_ratio(branch, shifted[:, s], frame, density) for branch in branches]
            )
            for s in range(len(branches))
        ],
        axis=1,
    )
    resolution = density.continuous if isinstance(density, MixedDensity) else density
    for column, (label, branch) in enumerate(zip(labels, branches, strict=True)):
        # The natural course is *meant* to move nobody -- it is the reference the
        # other policies are contrasted against, and its mean is E[Y]. Warning that a
        # zero shift crosses no bin edge, or leaves no dose outside the support,
        # would fire on the recommended way to declare a fit and teach the reader to
        # ignore the warning that matters.
        if branch.identity:
            continue
        if resolution is not None:
            warn_if_unresolved(resolution, shifted[:, column], a)
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
        for probability, codes in _point_discrete_assignments(
            policy, data, levels, frame, rows, where=where
        ):
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


def _same(values: FloatArray) -> FloatArray:
    """The treatment values themselves, the classifier block of a treatment of one column."""
    return values


def vector_route_refusal(treatment: str) -> str:
    """Why a vector treatment with a continuous component needs ``policies=``.

    It is the reason a continuous dose has: no arms, so no arm-indexed estimand names a
    parameter of it.
    """
    return (
        f"treatment {treatment!r} is a vector with a continuous component, so it has no arms "
        "and none of the arm-indexed estimands name a parameter it has. Say which values to "
        "compare with policies=[JointPolicy(name, components={column: policy, ...}, "
        "policy_kind='known'), ...]; JointPolicy(name, components={}, policy_kind='known') "
        "is the natural course, whose mean is E[Y]"
    )


def vector_msm_refusal(treatment: str) -> str:
    """Why ``msm=`` refuses a vector treatment with a continuous component."""
    return (
        f"treatment {treatment!r} is a vector with a continuous component, and msm= projects "
        "the dose-response curve of one dose: MSM(doses=...) declares a grid on one dose "
        "axis, and the clever covariate integrates over it by the trapezoid rule. A vector "
        "has no one dose axis, and this package defines no integration grid or cubature "
        "over a vector. Fit the policy means and contrasts with "
        "policies=[JointPolicy(...), ...]"
    )


def whole_vector_refusal(policy: object, treatment: str) -> str:
    """The refusal of a policy declared on a vector treatment with a dose as a whole."""
    return (
        f"policy {describe_policy(policy)} is declared on the vector treatment {treatment!r} "
        "as a whole. A vector with a continuous component takes one policy per component: "
        "declare JointPolicy(name, components={column: policy, ...}, policy_kind='known')"
    )


def _point_discrete_assignments(
    policy: object,
    data: CausalData,
    levels: Sequence[object],
    frame: FrameSource,
    rows: BoolArray,
    *,
    where: str,
) -> tuple[tuple[float, IntArray], ...]:
    """:func:`discrete_assignments` at a point treatment, reading a :class:`JointPolicy`.

    A joint policy applies at an all-categorical vector treatment by the discrete reduction
    of :func:`joint_discrete_assignments`.  At a treatment of one column it is refused by
    :func:`policy_branches`.
    """
    if not isinstance(policy, JointPolicy) or not data.is_vector_treatment:
        return discrete_assignments(policy, levels, frame, rows, where=where)
    components = tuple(
        JointComponent(name, component_levels)
        for name, component_levels in zip(data.component_names, data.component_levels, strict=True)
    )

    def frame_with(extra: Mapping[str, Any]) -> Any:
        payload = {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        payload.update(extra)
        return data.frame_like(payload)

    return joint_discrete_assignments(
        policy,
        components,
        [tuple(cast("Sequence[object]", level)) for level in levels],
        frame_with,
        rows,
        where=where,
    )


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


def _warn_outside_cell_support(
    name: str, components: Sequence[JointComponent], natural: FloatArray, assigned: FloatArray
) -> None:
    """Warn when a joint policy assigns a dose outside the doses of its target cell.

    Positivity at a vector treatment reads each categorical cell: when the categorical map
    moves a unit from cell ``c`` to cell ``c'``, its assigned dose must lie inside the doses
    observed in cell ``c'``.  This compares each assigned dose with the observed range of
    the same component among the units whose natural categorical values are ``c'``.
    """
    categorical = [index for index, component in enumerate(components) if not component.continuous]
    natural_cells = [tuple(row) for row in natural[:, categorical]]
    target_cells = [tuple(row) for row in assigned[:, categorical]]
    for index, component in enumerate(components):
        if not component.continuous:
            continue
        below = above = 0
        for cell in set(target_cells):
            source = np.asarray([value == cell for value in natural_cells], dtype=bool)
            target = np.asarray([value == cell for value in target_cells], dtype=bool)
            if not source.any():
                continue
            doses = natural[source, index]
            moved = assigned[target, index]
            below += int(np.count_nonzero(moved < float(np.min(doses))))
            above += int(np.count_nonzero(moved > float(np.max(doses))))
        for count, side in ((below, "below the smallest"), (above, "above the largest")):
            if not count:
                continue
            warnings.warn(
                f"policy {name!r} assigns {count / natural.shape[0]:.1%} of rows a "
                f"{component.name!r} dose {side} one observed in the categorical cell the "
                "policy sends them to. The outcome regression is extrapolating there. "
                "Identification requires the doses a policy assigns in a cell to lie inside "
                "the conditional support of the dose in that cell.",
                PositivityWarning,
                stacklevel=5,
            )


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
    outside_declared_support : int or None
        Rows a component assigns a dose outside the declared support, counted over the
        policy's components, on either route.  The fit refuses a positive count before any
        learner, so a fitted result reads 0.  ``None`` when the fit declared no support.
    atom_mass : dict of float to float or None
        The mean fitted probability :math:`P(A = a_k \\mid W)` of each declared atom, on the
        density route.  ``None`` without declared atoms, and on the classifier route, which
        fits no law of the dose.
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
    outside_declared_support: int | None = None
    atom_mass: dict[float, float] | None = None

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
        if self.outside_declared_support is not None:
            unsupported += f", outside declared support={self.outside_declared_support}"
        if self.atom_mass:
            masses = ", ".join(f"{atom:g}: {mass:.3g}" for atom, mass in self.atom_mass.items())
            unsupported += f", atom mass {masses}"

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
    density: ConditionalDensity | MixedDensity | None,
    treatment: FloatArray,
    *,
    propensity: FloatArray | None = None,
    mechanisms: Sequence[FloatArray] = (),
    absolute_score_weights: FloatArray | None = None,
    equations: tuple[str, ...] = (),
    n_repeats: int = 1,
    folds: Folds | None = None,
    declared_support: tuple[FloatArray, FloatArray] | None = None,
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
    density : ConditionalDensity, MixedDensity or None
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
    declared_support : tuple of ndarray or None
        ``(lower, upper)``, the declared support on every row, from
        :func:`declared_support_bounds`.  ``None`` when the fit declared none.

    Returns
    -------
    dict of str to PolicySupport
        One record per policy, keyed by its report label.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    rows = np.arange(a.size)
    g: FloatArray | None = None
    joint = None if policies.joint is None else policies.joint.joint()
    if joint is not None:
        # A vector treatment on the density route: the joint density of the node, at the
        # observed value (column 0) and at each component's assigned value.
        observed_density = joint[:, 0]
    elif density is not None:
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
        measured = joint is not None or density is not None or g is not None
        if joint is not None:
            assigned = [joint[:, c + 1] for c in mine]
        elif density is not None:
            assigned = [density.density_at(policies.shifted[:, c]) for c in mine]
        elif g is not None:
            assigned = [g[rows, policies.shifted[:, c].astype(np.int64)] for c in mine]
        else:
            # The classifier route fits no density, so no zero can be read at all.
            assigned = []
        outside: int | None = None
        if declared_support is not None:
            lower, upper = declared_support
            outside = 0
            for c in mine:
                dose = np.asarray(policies.shifted[:, c], dtype=float)
                with np.errstate(invalid="ignore"):
                    outside += int(np.sum(~((dose >= lower) & (dose <= upper))))
        atom_mass = (
            {
                float(atom): float(np.mean(np.asarray(density.atom_probabilities)[:, k]))
                for k, atom in enumerate(density.atoms)
            }
            if isinstance(density, MixedDensity)
            else None
        )
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
            outside_declared_support=outside,
            atom_mass=atom_mass,
        )
    return out


def declared_support_bounds(data: CausalData) -> tuple[FloatArray, FloatArray] | None:
    """The declared support of a point dose on every row, or ``None`` without one.

    Parameters
    ----------
    data : CausalData
        The validated data.

    Returns
    -------
    tuple of ndarray or None
        ``(lower, upper)`` per row.
    """
    if data.treatment_support is None:
        return None
    frame = lazy_frame(
        lambda: data.frame_like(
            {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        )
    )
    return cast(Support, data.treatment_support).bounds(frame, data.n)
