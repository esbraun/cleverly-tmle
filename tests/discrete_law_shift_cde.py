r"""The finite-support law of :mod:`tests.discrete_law_shift`, with ``Delta`` and ``Z``.

:mod:`tests.discrete_law_shift` carries the library's only non-circular proof that a
shift's reported influence curve *is* the efficient influence function.  It has no
:math:`\Delta` dimension and no intermediate variable, so it cannot reach a shift fit
declared with ``delta=`` or ``intermediate=``; :mod:`tests.discrete_law_mar` and
:mod:`tests.discrete_law_cde` have those dimensions and no doses.  This module is the
cross, and it is a **new law rather than a wider existing one** because both parents have
to go on proving their own derivations unchanged.

The observed-data support is ``(w, a, z, k)``.  ``Z`` is realised after the dose and
before the outcome, so it is recorded whether or not the outcome is:

.. math::

    (w, a, z, k), \qquad
    k \in \{\,\Delta = 1, Y = 0;\ \ \Delta = 1, Y = 1;\ \ \Delta = 0\,\},

seventy-two cells in a ``(3, 4, 2, 3)`` array.  The cell probabilities follow the time
ordering the derivation assumes,

.. math::

    p(w, a, z, k) = p_W(w)\, g(a \mid w)\, q_z(a, w)\, \pi(a, w)\,
                    p_Y(\,\cdot \mid a, z, w, \Delta = 1),

with :math:`\pi` conditioning on ``(A, W)`` and *not* on ``Z``, which is the missingness
assumption :meth:`~cleverly.data.causal_data.CausalData.missingness_design` states --
encoded here rather than assumed.

**One law rather than two.**  :func:`functional` takes ``level=None`` for the estimand a
``delta=``-only fit reports -- the shift parameter with ``Z`` left to its natural course,
obtained by summing the ``z`` axis out -- and ``level in (0, 1)`` for the shift-indexed
controlled direct effect.  Splitting them into two modules would make the two disagree by
construction rather than by arithmetic, since the first is literally a marginal of the
second's cell probabilities.

**The rule that keeps the complex step working**, inherited from
:mod:`tests.discrete_law_shift`: every operation in :func:`functional` is arithmetic, so it
stays analytic in the cell probabilities.  Do not introduce ``clip``, ``abs`` or a
comparison there.  The shift map is a *comparison* -- ``a + delta`` against the cap -- so it
is precomputed **outside**, once, as a fixed index permutation.

**Why the constants are what they are.**  They are the output of a search over laws with
denominators in ``{2, 4}``, scored by the *smallest* margin any of this module's claims
would have to clear, so that no negative control is marginal.  Three of them are
load-bearing in a way the parent laws' are not:

* :data:`PI` varies with the **dose**, by 0.5 within every covariate stratum, and differs
  between a row's own dose and its shifted one in 7 of 12 cells.  This is the whole content
  of the cross.  A mechanism depending on ``W`` alone would make
  :math:`\pi(d(a, w), w) = \pi(a, w)` identically, collapse the ``(n, S + 1)`` array to
  ``S + 1`` identical columns, and turn every mutation into a no-op -- the law would prove
  nothing while passing.  :data:`QZ` varies with the dose for the same reason.
* the controlled direct effect of the ``+1`` policy **changes sign** between the levels --
  ``-0.0859`` at ``z = 0`` against ``+0.0703`` at ``z = 1`` -- and neither equals the
  ``-0.0273`` obtained by leaving ``Z`` alone.  Confusing the levels does not perturb the
  answer, it inverts it; ignoring ``q_z`` gives a third number again.
* the two caps give genuinely different content, not just different means.  A capped row
  keeps its own dose, so :math:`\pi(d(a, w), w) = \pi(a, w)` exactly there while a moved
  row's differs -- under :data:`CAP` one dose is a stayer and under :data:`CAP_TIGHT` two
  are, so an implementation that indexed the mechanism blocks off by one, or that
  transposed ``ratio_at``'s axes, cannot pass on both.

Every property that makes the parent modules' assertions exact is preserved: each cell
probability is a multiple of ``1 / N`` (the smallest cell holds 2 rows), so :func:`frame`
*is* the law rather than a sample from it; and nothing is truncated -- ``g`` lies in
``[0.125, 0.5]`` against an ``auto`` bound near 0.0145, :math:`\pi` and :math:`q_z` in
``[0.25, 0.75]`` against a ``nuisance_bound`` of 0.01, and ``Qbar`` in ``[0.25, 0.75]``
against the ``alpha`` shrinkage window.  Half the sample has no recorded outcome.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Rows.  Every cell probability below is a multiple of ``1 / N``.  ``2048`` rather than
#: the parents' round numbers because every factor here has a denominator that is a power
#: of two: ``P_W`` in quarters, ``G`` in eighths, and ``QZ``, ``PI`` and ``QBAR`` in
#: quarters, so the finest cell is ``1 / 2048``.
N = 2048

#: ``P(W = w)``, in quarters.
P_W = np.array([0.50, 0.25, 0.25])

#: ``g(a | w)``, in eighths; rows sum to one.  Deliberately not symmetric across ``w``, so
#: a shift's clever covariate genuinely varies with the covariate.
G = np.array([[4.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 4.0], [2.0, 2.0, 2.0, 2.0]]) / 8.0

#: ``pi(a, w) = P(Delta = 1 | A = DOSES[a], W = w)``, indexed ``[w, a]``.  Varies with the
#: dose -- see the module docstring for why that is the one constant this law cannot do
#: without.
PI = np.array(
    [
        [0.50, 0.25, 0.75, 0.50],
        [0.75, 0.75, 0.25, 0.75],
        [0.50, 0.25, 0.25, 0.75],
    ]
)

#: ``q_z(a, w) = P(Z = 1 | A = DOSES[a], W = w)``, indexed ``[w, a]``.  Varies with the
#: dose for the same reason ``PI`` does.
QZ = np.array(
    [
        [0.25, 0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75, 0.50],
        [0.25, 0.75, 0.75, 0.25],
    ]
)

#: ``Qbar(a, z, w) = P(Y = 1 | A = DOSES[a], Z = z, Delta = 1, W = w)``, indexed
#: ``[w, a, z]``.  Non-monotone in the dose on purpose: a policy that shifts everyone up
#: must not be able to look right by accident.
QBAR = np.array(
    [
        [[0.75, 0.50], [0.50, 0.75], [0.75, 0.75], [0.50, 0.50]],
        [[0.75, 0.25], [0.25, 0.25], [0.50, 0.50], [0.25, 0.50]],
        [[0.50, 0.25], [0.75, 0.50], [0.50, 0.75], [0.25, 0.50]],
    ]
)

DOSES = (0.0, 1.0, 2.0, 3.0)
DELTA = 1.0

#: The declared caps.  ``CAP`` holds back only the top dose's shift; ``CAP_TIGHT`` holds
#: back the top two, so more mass lands on the "kept its own dose" branch -- and there the
#: mechanism at the assigned dose is the mechanism at the observed one.
CAP = 3.0
CAP_TIGHT = 2.0

#: The levels of the intermediate a controlled direct effect can be taken at.
LEVELS = (0, 1)

#: Index of the fourth axis: the outcome was observed and zero, observed and one, or not
#: observed.  ``Y`` is undefined -- not zero -- in the last case.
OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2


def _shift_map(cap: float) -> tuple[int, ...]:
    """``d(a) = a + DELTA if a + DELTA <= cap else a``, as indices into :data:`DOSES`.

    Computed here, once, so that :func:`functional` never compares anything.
    """
    return tuple(
        int(index + DELTA) if DOSES[index] + DELTA <= cap else index for index in range(len(DOSES))
    )


#: The policy as a permutation of dose indices, per cap.
SHIFT_MAPS: dict[float, tuple[int, ...]] = {CAP: _shift_map(CAP), CAP_TIGHT: _shift_map(CAP_TIGHT)}

#: Reported label to the shift map it means.  ``natural course`` is the identity policy,
#: whose mean is ``E[Y]`` under the same identification.
POLICIES: dict[str, tuple[int, ...]] = {
    "natural course": tuple(range(len(DOSES))),
    "+1": SHIFT_MAPS[CAP],
    "+1 (cap 2)": SHIFT_MAPS[CAP_TIGHT],
}

#: Bin edges that make the density a probability mass function: every bin one wide, each
#: holding exactly one dose.  This is what a ``ConditionalDensity`` over this law carries.
EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])

#: The support, ordered ``(w, a, z, k)``.  Row blocks of the sample follow this order.
SUPPORT: tuple[tuple[int, int, int, int], ...] = tuple(
    itertools.product(range(3), range(4), range(2), range(3))
)


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, Z = z, K = k)`` as a ``(3, 4, 2, 3)`` integer array."""
    counts = np.empty((3, 4, 2, 3))
    for w, a, z, k in SUPPORT:
        base = P_W[w] * G[w, a] * (QZ[w, a] if z == 1 else 1.0 - QZ[w, a])
        if k == UNOBSERVED:
            counts[w, a, z, k] = base * (1.0 - PI[w, a]) * N
        else:
            outcome = QBAR[w, a, z] if k == OBSERVED_ONE else 1.0 - QBAR[w, a, z]
            counts[w, a, z, k] = base * PI[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-9:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- adjust P_W, G, QZ, PI or QBAR"
        )
    return rounded.astype(int)


#: Cell counts in the realised sample.  Integral by construction -- checked above.
COUNTS = _cell_counts()

#: ``P(W, A, Z, K)``, shape ``(3, 4, 2, 3)``.  Taken from the counts rather than from the
#: constants above, so it is bit-for-bit the empirical law of :func:`frame`.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    ``Y`` is ``NaN`` wherever ``Delta`` is zero, which is how a caller would really hand
    missing outcomes to :meth:`~cleverly.TMLE.fit`; ``Z`` is recorded on every row,
    including those, because it is realised before the outcome.  Rows are laid out in
    :data:`SUPPORT` order, one contiguous block per support point.
    """
    counts = [COUNTS[w, a, z, k] for w, a, z, k in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    kind = columns[:, 3]
    return pd.DataFrame(
        {
            "W": columns[:, 0],
            "A": np.array(DOSES, dtype=float)[columns[:, 1].astype(int)],
            "Z": columns[:, 2],
            "Y": np.where(kind == UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    counts = np.array([COUNTS[w, a, z, k] for w, a, z, k in SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


def functional(probs: Any, estimand: str, level: int | None) -> Any:
    r"""The estimand, longhand, sharing no code with ``src/``.

    ``probs`` is the ``(3, 4, 2, 3)`` array of cell probabilities and may be complex -- see
    :func:`gateaux`.  ``level`` selects the parameter:

    * ``None`` -- the shift parameter with ``Z`` left alone, which is what a fit declaring
      ``shifts=`` and ``delta=`` reports.  Summing the ``z`` axis out replaces
      :math:`P(Y = 1 \mid A, Z = z, \Delta = 1, W)` by
      :math:`P(Y = 1 \mid A, \Delta = 1, W)`, which *is* the marginalisation.
    * ``0`` or ``1`` -- the controlled direct effect under the policy, the conditional mean
      taken *within* the stratum and averaged against the marginal over everyone.

    Both are

    .. math::

        \Psi_d = \sum_{w, a} P(W = w, A = a)\,
                 \bar Q\bigl(d(a),\, w\bigr),

    with the dose distribution taken over the **whole** sample -- every level, observed
    outcome or not -- and only the conditional mean taken among the recorded rows.  That
    split is the entire missing-data content, and :func:`observed_only_functional` is the
    control that says so.

    Arithmetic only: no ``clip``, no ``abs``, no comparison.
    """
    p = probs
    joint = p.sum(axis=(2, 3))  # P(w, a) -- W, A and Z are recorded for everyone
    if level is None:
        recorded = (p[:, :, :, OBSERVED_ZERO] + p[:, :, :, OBSERVED_ONE]).sum(axis=2)
        qbar = p[:, :, :, OBSERVED_ONE].sum(axis=2) / recorded
    else:
        recorded = p[:, :, level, OBSERVED_ZERO] + p[:, :, level, OBSERVED_ONE]
        qbar = p[:, :, level, OBSERVED_ONE] / recorded

    if estimand.startswith("ey_shift["):
        return _mean_under(joint, qbar, estimand[len("ey_shift[") : -1])
    if estimand.startswith("ate_shift["):
        left, right = estimand[len("ate_shift[") : -1].split(" vs ")
        return _mean_under(joint, qbar, left) - _mean_under(joint, qbar, right)
    raise ValueError(f"no oracle branch for {estimand!r}")


def _mean_under(joint: Any, qbar: Any, label: str) -> Any:
    """``sum_{w,a} P(w, a) Qbar(d(a), w)`` -- an index, never a comparison."""
    mapping = POLICIES[label]
    total = 0.0
    for a, target in enumerate(mapping):
        total = total + (joint[:, a] * qbar[:, target]).sum()
    return total


def observed_only_functional(probs: Any, estimand: str, level: int | None) -> Any:
    """**Negative control.**  :func:`functional` on the complete cases alone.

    Not the estimand -- the point of having it is that it is not.  This is what a plug-in
    that dropped the rows with no recorded outcome would converge to.  The conditional
    means do not move: they were already taken among the recorded rows.  What moves is the
    joint distribution of ``(W, A)`` the shifted predictions are averaged against, and on a
    shift law that is the weight each dose's *shifted* prediction carries -- so the error
    is in the policy's own arithmetic rather than only in a covariate marginal.  On this
    law the gap runs from 0.0234 to 0.0371 across the three policies and three levels.
    """
    p = np.asarray(probs)
    complete = p.copy()
    complete[:, :, :, UNOBSERVED] = 0.0
    return functional(complete / complete.sum(), estimand, level)


def induced(cap: float) -> np.ndarray:
    """``g^d(b | w) = sum over the preimage of b`` -- the density the policy induces."""
    mapping = SHIFT_MAPS[cap]
    out = np.zeros_like(G)
    for source, target in enumerate(mapping):
        out[:, target] += G[:, source]
    return out


def induced_regime_functional(probs: Any, label: str, level: int | None) -> Any:
    """**Negative control.**  The known stochastic regime at the induced density.

    Equal to :func:`functional` in the population -- exactly, on this law -- and a
    *different* estimator with a different influence curve, because its plug-in term
    averages over the doses instead of reading the one the unit received.  Carried across
    from :mod:`tests.discrete_law_shift` because coarsening the outcome does not touch that
    argument, and because "``pi`` and ``g`` both divide, so this is just a regime at the
    induced density" is a plausible wrong turn on precisely this path.
    """
    p = probs
    joint = p.sum(axis=(2, 3))
    if level is None:
        recorded = (p[:, :, :, OBSERVED_ZERO] + p[:, :, :, OBSERVED_ONE]).sum(axis=2)
        qbar = p[:, :, :, OBSERVED_ONE].sum(axis=2) / recorded
    else:
        recorded = p[:, :, level, OBSERVED_ZERO] + p[:, :, level, OBSERVED_ONE]
        qbar = p[:, :, level, OBSERVED_ONE] / recorded
    marginal = joint.sum(axis=1)
    star = induced(CAP) if label == "+1" else induced(CAP_TIGHT)
    return (marginal * (star * qbar).sum(axis=1)).sum()


def gateaux(estimand: str, point: int, level: int | None, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at ``level``, at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)
                 \right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function -- here of the observed-data model, contaminated at an
    observed-data support point, and derived from :func:`functional` alone.

    Twenty-four of the seventy-two points are ``Delta = 0``, where the residual term cannot
    contribute and the derivative must come out as
    :math:`\bar Q(d(A, W), W) - \Psi` exactly.  That is the sharpest single statement that
    the ``Delta`` factor is where it belongs, and no shift test in the suite has reached it
    before.

    Differentiation is by complex step rather than finite difference, for the reasons given
    in :func:`tests.discrete_law.gateaux`: full double precision, hence an exact comparison
    rather than a close one.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand, level)) / step)


def eif(estimand: str, level: int | None) -> np.ndarray:
    """The EIF of ``estimand`` at ``level``, evaluated at every support point."""
    return np.array([gateaux(estimand, point, level) for point in range(len(SUPPORT))])


#: The parameter names the shift targets report on this law, in the order the estimator
#: reports them.  This law is deliberately *not* one of the laws the registry coverage gate
#: in ``tests/unit/test_registry.py`` walks -- its estimand names are
#: :mod:`tests.discrete_law_shift`'s, and two laws claiming one name would make
#: ``truth_for`` depend on law order -- so there is no ``oracle_names`` here and these are
#: for the modules that check the coarsened shift path to read.
PER_SHIFT_NAMES: dict[str, tuple[str, ...]] = {
    "ey_shift": tuple(f"ey_shift[{label}]" for label in POLICIES),
    "ate_shift": tuple(
        f"ate_shift[{label} vs natural course]" for label in POLICIES if label != "natural course"
    ),
}

NAMES: tuple[str, ...] = (*PER_SHIFT_NAMES["ey_shift"], *PER_SHIFT_NAMES["ate_shift"])

#: Every estimand at every level, keyed ``TRUTH[level][name]`` with ``None`` for the
#: parameter that leaves ``Z`` alone.
TRUTH: dict[int | None, dict[str, float]] = {
    level: {name: float(functional(PROBS, name, level)) for name in NAMES}
    for level in (None, *LEVELS)
}


#: The nuisances as the *realised sample* has them.  Equal to :data:`G`, :data:`QZ`,
#: :data:`PI` and :data:`QBAR` mathematically; derived from the counts so that the oracle
#: nuisances are exact in the sample down to the last bit.
_RECORDED = PROBS[:, :, :, OBSERVED_ZERO] + PROBS[:, :, :, OBSERVED_ONE]
G_EXACT = PROBS.sum(axis=(2, 3)) / PROBS.sum(axis=(1, 2, 3))[:, None]
QZ_EXACT = PROBS[:, :, 1, :].sum(axis=2) / PROBS.sum(axis=(2, 3))
PI_EXACT = _RECORDED.sum(axis=2) / PROBS.sum(axis=3).sum(axis=2)
QBAR_EXACT = PROBS[:, :, :, OBSERVED_ONE] / _RECORDED

#: ``Qbar`` with ``Z`` marginalised, which is what a fit without ``intermediate=`` learns.
QBAR_MARGINAL_EXACT = PROBS[:, :, :, OBSERVED_ONE].sum(axis=2) / _RECORDED.sum(axis=2)


class DiscreteShiftCoarsenedLaw:
    """The law, duck-typed as a data-generating process for the oracle learners.

    Every accessor takes a **dose per row** rather than an arm, which is the whole
    difference from :class:`tests.discrete_law_cde.DiscreteLaw`: a modified treatment
    policy assigns a different value to every unit, so an oracle for one has to answer at a
    vector of treatments rather than at a scalar.  All of them read the cell counts, so the
    oracles and the sample cannot drift apart.
    """

    @staticmethod
    def _cells(covariates: Any, dose: Any) -> tuple[np.ndarray, np.ndarray]:
        w = np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)
        a = np.rint(np.asarray(dose, dtype=float).reshape(-1)).astype(int)
        return w, a

    def bin_probabilities(self, covariates: Any) -> np.ndarray:
        """``g(. | w)`` as the ``(n, 4)`` matrix a ``ConditionalDensity`` carries."""
        w = np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)
        return np.asarray(G_EXACT[w], dtype=float)

    def missingness(self, covariates: Any, dose: Any) -> np.ndarray:
        """``P(Delta = 1 | A = dose, W)`` -- deliberately not a function of ``Z``."""
        w, a = self._cells(covariates, dose)
        return np.asarray(PI_EXACT[w, a], dtype=float)

    def intermediate_mean(self, covariates: Any, dose: Any) -> np.ndarray:
        """``P(Z = 1 | A = dose, W)``."""
        w, a = self._cells(covariates, dose)
        return np.asarray(QZ_EXACT[w, a], dtype=float)

    def outcome_mean(self, covariates: Any, dose: Any, intermediate: float | None) -> np.ndarray:
        """``E[Y | A = dose, Z = intermediate, Delta = 1, W]``, or ``Z`` marginalised.

        ``intermediate=None`` is not a default standing in for a level -- it is the
        distinct parameter :func:`functional` reports at ``level=None``, the regression a
        fit without ``intermediate=`` actually learns.
        """
        w, a = self._cells(covariates, dose)
        if intermediate is None:
            return np.asarray(QBAR_MARGINAL_EXACT[w, a], dtype=float)
        return np.asarray(QBAR_EXACT[w, a, int(intermediate)], dtype=float)
