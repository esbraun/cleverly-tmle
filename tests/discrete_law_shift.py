"""A finite law with an ordered dose, for checking the shift influence curve exactly.

The package's standard of evidence for an influence curve is that it matches a
*numerically differentiated* one on a law a finite sample realises exactly.  That needs
finite support, which a continuous treatment does not have -- so this law puts the dose on
four ordered points.  A shift along an ordered discrete support is a perfectly good
modified treatment policy: the density is a probability mass function with unit-width
bins, and the clever covariate is the same ratio it is on a continuum.  What the finite
support buys is that ``functional`` below is a rational function of the cell probabilities
and so can be differentiated by a complex step at full double precision.

**The rule that keeps this working.**  Every operation in :func:`functional` is arithmetic,
so it stays analytic in the cell probabilities.  Do not introduce ``clip``, ``abs`` or a
comparison there.  The shift map is a *comparison* -- ``a + delta`` against the cap -- so
it is precomputed **outside** the functional, once, as a fixed index permutation
(:data:`SHIFTED`).  Inside, applying the policy is an array index.

Two caps are exported so the cap's indicator is exercised in both states: under
:data:`CAP` the top two doses are both held back, and under :data:`CAP_TIGHT` the top
*three* are, so the induced density piles up differently.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Rows.  Every cell probability below is a multiple of ``1 / N``, so ``frame()`` realises
#: the law exactly and a sample mean *is* a population mean.
N = 2000

#: ``P(W = w)``, in tenths.
P_W = np.array([0.50, 0.30, 0.20])

#: ``g(a | w)``, in twentieths; rows sum to one.  Deliberately not symmetric across ``w``,
#: so a shift's clever covariate genuinely varies with the covariate.
G = (
    np.array(
        [
            [8.0, 6.0, 4.0, 2.0],
            [2.0, 4.0, 6.0, 8.0],
            [5.0, 5.0, 5.0, 5.0],
        ]
    )
    / 20.0
)

#: ``Qbar(a, w) = P(Y = 1 | A = a, W = w)``, in tenths.  Non-monotone in the dose on
#: purpose: a policy that shifts everyone up must not be able to look right by accident.
Q = np.array(
    [
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.3, 0.5, 0.7],
        [0.1, 0.5, 0.5, 0.9],
    ]
)

DOSES = (0.0, 1.0, 2.0, 3.0)
DELTA = 1.0

#: The declared caps.  ``CAP`` holds back only the top dose's shift; ``CAP_TIGHT`` holds
#: back the top two, so more mass lands on the "kept its own dose" branch.
CAP = 3.0
CAP_TIGHT = 2.0


def _shift_map(cap: float) -> tuple[int, ...]:
    """``d(a) = a + DELTA if a + DELTA <= cap else a``, as indices into :data:`DOSES`.

    Computed here, once, so that :func:`functional` never compares anything.
    """
    return tuple(
        int(index + DELTA) if DOSES[index] + DELTA <= cap else index for index in range(len(DOSES))
    )


#: The policy as a permutation of dose indices, per cap.
SHIFT_MAPS: dict[float, tuple[int, ...]] = {CAP: _shift_map(CAP), CAP_TIGHT: _shift_map(CAP_TIGHT)}
SHIFTED = SHIFT_MAPS[CAP]

#: Bin edges that make the density a probability mass function: every bin one wide, each
#: holding exactly one dose.  This is what a ``ConditionalDensity`` over this law carries.
EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])

#: ``(w, a, y)`` in the order ``frame()`` emits rows.
SUPPORT: tuple[tuple[int, int, int], ...] = tuple(itertools.product(range(3), range(4), range(2)))


def induced(cap: float) -> np.ndarray:
    """``g^d(b | w) = sum over the preimage of b`` -- the density the policy induces.

    Exported because the *negative control* needs it: the stochastic regime at this
    density has the same mean as the shift and a different influence curve, and a test
    asserts that difference rather than trusting the derivation.
    """
    mapping = SHIFT_MAPS[cap]
    out = np.zeros_like(G)
    for source, target in enumerate(mapping):
        out[:, target] += G[:, source]
    return out


INDUCED = induced(CAP)


def clever_covariate(cap: float) -> np.ndarray:
    """``h(a, w) = g^d(a | w) / g(a | w)``, the shift's clever covariate, ``(3, 4)``."""
    return induced(cap) / G


def _cell_counts() -> np.ndarray:
    """``N * P(w, a, y)`` as integers, asserting the constants really are exact."""
    counts = np.zeros((3, 4, 2))
    for w, a, y in SUPPORT:
        probability = P_W[w] * G[w, a] * (Q[w, a] if y == 1 else 1.0 - Q[w, a])
        exact = probability * N
        assert abs(exact - round(exact)) < 1e-9, (
            f"cell ({w}, {a}, {y}) has probability {probability}, which is not a multiple "
            f"of 1/{N}; the constants at the top of this module must be chosen so a "
            "sample of N rows realises the law exactly"
        )
        counts[w, a, y] = round(exact)
    assert counts.sum() == N
    return counts.astype(int)


COUNTS = _cell_counts()
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """``N`` rows realising the law exactly, blocks in :data:`SUPPORT` order."""
    rows = [
        (float(w), float(DOSES[a]), float(y))
        for (w, a, y) in SUPPORT
        for _ in range(int(COUNTS[w, a, y]))
    ]
    return pd.DataFrame(rows, columns=["W", "A", "Y"])


def first_row_of() -> np.ndarray:
    """Index of the first row of each support point, in :data:`SUPPORT` order."""
    starts = np.cumsum([0] + [int(COUNTS[w, a, y]) for (w, a, y) in SUPPORT])[:-1]
    return np.asarray(starts, dtype=int)


def functional(probs: Any, estimand: str) -> Any:
    """The estimand, longhand, sharing no code with ``src/``.

    ``probs`` is the ``(3, 4, 2)`` array of cell probabilities, and may be complex -- see
    :func:`gateaux`.  Arithmetic only: no ``clip``, no ``abs``, no comparison.
    """
    p = probs
    joint = p[:, :, 0] + p[:, :, 1]  # P(w, a)
    qbar = p[:, :, 1] / joint  # Qbar(a, w)

    if estimand.startswith("ey_shift["):
        label = estimand[len("ey_shift[") : -1]
        return _mean_under(joint, qbar, label)
    if estimand.startswith("ate_shift["):
        left, right = estimand[len("ate_shift[") : -1].split(" vs ")
        return _mean_under(joint, qbar, left) - _mean_under(joint, qbar, right)
    raise ValueError(f"no oracle branch for {estimand!r}")


#: Reported label to the shift map it means.  ``natural course`` is the identity policy,
#: whose mean is ``E[Y]``.
POLICIES: dict[str, tuple[int, ...]] = {
    "natural course": tuple(range(len(DOSES))),
    "+1": SHIFT_MAPS[CAP],
    "+1 (cap 2)": SHIFT_MAPS[CAP_TIGHT],
}


def _mean_under(joint: Any, qbar: Any, label: str) -> Any:
    """``sum_{w,a} P(w, a) Qbar(d(a), w)`` -- an index, never a comparison."""
    mapping = POLICIES[label]
    total = 0.0
    for a, target in enumerate(mapping):
        total = total + (joint[:, a] * qbar[:, target]).sum()
    return total


def gateaux(estimand: str, point: int, *, step: float = 1e-30) -> float:
    """The Gateaux derivative at one support point, by complex step.

    ``Psi`` along the contamination path is a rational function of ``t`` and therefore
    analytic, so ``Im Psi(ih) / h`` is the derivative to full double precision with no
    subtractive cancellation.  See ``tests/discrete_law.py`` for the same argument.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif(estimand: str) -> np.ndarray:
    """The influence curve at every support point, in :data:`SUPPORT` order."""
    return np.array([gateaux(estimand, point) for point in range(len(SUPPORT))])


#: Target name to the parameter names it reports, for the registry's coverage gate.
#: The same protocol ``tests/discrete_law.py`` exposes, so ``test_registry`` can walk a
#: tuple of laws without knowing which one owns which estimand.  ``natural course`` is
#: the reference here, exactly as ``Shift(0.0, cap=None)`` declared first would be.
PER_TARGET_NAMES: dict[str, tuple[str, ...]] = {
    "ey_shift": tuple(f"ey_shift[{label}]" for label in POLICIES),
    "ate_shift": tuple(
        f"ate_shift[{label} vs natural course]" for label in POLICIES if label != "natural course"
    ),
}


def oracle_names(target: str) -> tuple[str, ...]:
    """The parameter names ``target`` reports here, or none if this law does not own it."""
    return PER_TARGET_NAMES.get(target, ())


NAMES: tuple[str, ...] = (*PER_TARGET_NAMES["ey_shift"], *PER_TARGET_NAMES["ate_shift"])

TRUTH: dict[str, float] = {name: float(functional(PROBS, name)) for name in NAMES}


def induced_regime_functional(probs: Any, label: str) -> Any:
    """**Negative control.**  The known stochastic regime at the induced density.

    Equal to :func:`functional` in the population -- that is the point -- but a *different*
    estimator with a different influence curve, because its plug-in term averages over the
    doses instead of reading the one the unit received.  A test asserts the two influence
    curves differ, so that a later "simplification" delegating the shift to
    :func:`~cleverly.inference.influence.regime_means` fails loudly.
    """
    p = probs
    joint = p[:, :, 0] + p[:, :, 1]
    qbar = p[:, :, 1] / joint
    marginal = joint.sum(axis=1)
    star = induced(CAP) if label == "+1" else induced(CAP_TIGHT)
    return (marginal * (star * qbar).sum(axis=1)).sum()


# --------------------------------------------------------------------- weighting


def cell_weights(weight_of: Any) -> np.ndarray:
    """A weight per support point, from a function of ``(w, a, y)``.

    Observation weights are a function of the observed row, so on a law with finite
    support they are twenty-four numbers.  Transcribed from
    :func:`tests.discrete_law.cell_weights` rather than imported, on the terms this module
    restates everything else: an oracle that reached into another module for the machinery
    it checks would make the two agree by construction.
    """
    return np.array([float(weight_of(w, a, y)) for w, a, y in SUPPORT], dtype=float)


def row_weights(weights: np.ndarray) -> np.ndarray:
    """Cell weights expanded to one value per row of :func:`frame`."""
    counts = [COUNTS[w, a, y] for w, a, y in SUPPORT]
    return np.repeat(np.asarray(weights, dtype=float), counts)


def tilt(probs: Any, weights: Any) -> Any:
    r"""The weighted law :math:`dP_w = w\,dP / E_P[w]`, as cell probabilities.

    Kept analytic in ``probs`` -- a ratio of linear functions -- so :func:`weighted_gateaux`
    can differentiate through it by a complex step.
    """
    p = np.asarray(probs)
    w = np.asarray(weights, dtype=float).reshape(len(SUPPORT))
    cells = np.zeros_like(p)
    for index, (a, b, c) in enumerate(SUPPORT):
        cells[a, b, c] = w[index]
    tilted = cells * p
    return tilted / tilted.sum()


def weighted_functional(probs: Any, estimand: str, weights: Any) -> Any:
    """``Psi(P_w)`` -- the shift parameter of the tilted law, longhand.

    A weight tilts the *population*, so both halves of the parameter move: the mechanism a
    weighted fit's density converges to is :math:`P_w(A \\mid W)`, hence a different clever
    covariate, and the dose distribution the plug-in averages against is the tilted one.
    Writing the parameter this way makes both statements testable rather than assumed --
    and in particular it says the weight is *not* a factor in the clever covariate's
    denominator, which is what the refusal this replaced claimed it was.
    """
    return functional(tilt(probs, weights), estimand)


def weighted_gateaux(estimand: str, point: int, weights: Any, *, step: float = 1e-30) -> float:
    r"""Gateaux derivative of :math:`P \mapsto \Psi(P_w)` at support point ``point``.

    The contamination is of :math:`P`, the law the *rows are drawn from* -- not of
    :math:`P_w`.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(weighted_functional(perturbed, estimand, weights)) / step)


def weighted_eif(estimand: str, weights: Any) -> np.ndarray:
    """The EIF of ``Psi(P_w)`` at every support point, in support order."""
    return np.array([weighted_gateaux(estimand, point, weights) for point in range(len(SUPPORT))])


def tilted_nuisances(weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(g, Qbar)`` of the tilted law -- what a weighted fit's nuisances converge to."""
    tilted = np.asarray(tilt(PROBS, weights), dtype=float)
    joint = tilted[:, :, 0] + tilted[:, :, 1]
    return joint / joint.sum(axis=1, keepdims=True), tilted[:, :, 1] / joint
