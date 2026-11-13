r"""Exact tail sizes and the change-of-measure variance of the boundaries law, by quadrature.

The tail size of an influence curve :math:`D` is
:math:`n_\kappa = \lceil (\kappa - 1) / 0.25^2 \rceil` with :math:`\kappa = E D^4 / (E D^2)^2`:
at :math:`n_\kappa` rows the plug-in variance :math:`P_n D^2` has relative SD 0.25 to first
order. A Wald clause gates only at :data:`WALD_REGIME_FACTOR` times that, rounded up to a
multiple of :data:`WALD_GATED_STEP` (:func:`gated_size`, :func:`classify`). Both constants are
imported from ``tests/studies/longitudinal_sdr_properties.py``, so the rule has one home.

:func:`law_b_moments` integrates the moments of the efficient influence function of
``tests/studies/mtp_boundaries_common.py`` (:func:`~tests.studies.mtp_boundaries_common.eif`)
over ``W1``, ``W2``, ``A1``, ``L2``, ``A2`` and ``Y``, by composite Gauss-Legendre split at
every break of the plans and the ratios. On law B' (``p5``) the node-2 law adds the atom at
``A2 = 3``. Monte Carlo cannot replace it: the cumulative ratio of ``p4`` reaches several
hundred in :math:`10^6` draws while its 99.9% quantile is about 38, and two :math:`10^6`-draw
runs read :math:`\kappa` 2,979 and 5,215.

:func:`diagonal_variance` evaluates the diagonal form of the policy-weighted variance
(:mod:`cleverly.longitudinal.policy_variance`) on the same grid with the true outcome
regressions, where the form is exact:

.. math::

    \sigma^2_{LR} = E[b^L b^R] + E\bigl[r^R_1(d^L_1(A_1))\,
        S_1(d^L_1(A_1), W)\bigr],\qquad
    S_1(a, w) = E\bigl[e^L_1 e^R_1 + r^R_2(d^L_2(A_2))\,S_2(d^L_2(A_2), H_2)
        \bigm| A_1 = a, W = w\bigr],

with :math:`S_2 = m_2 (1 - m_2)` and :math:`R` the natural plan or :math:`L` itself. It
equals :math:`E[D^L D^R]` of :func:`law_b_moments` to quadrature precision (witness 1 of the
policy-weighted variance), and the ``mutation`` arguments break the change of measure on
purpose (witness 2).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

from tests.studies import mtp_boundaries_common as law
from tests.studies.longitudinal_sdr_properties import WALD_GATED_STEP, WALD_REGIME_FACTOR

__all__ = [
    "TAIL_SD_TARGET",
    "TailSize",
    "classify",
    "diagonal_variance",
    "gated_size",
    "law_b_moments",
    "tail_size",
]

#: The relative SD of the plug-in variance at :math:`n_\kappa`.
TAIL_SD_TARGET = 0.25

Quantity = Literal["ey", "ate"]
Mutation = Literal["none", "one_branch", "first_ratio_only"]
Shift = Literal["none", "observed_dose", "no_ratio", "policy_residual"]


def tail_size(kappa: float) -> int:
    r""":math:`n_\kappa = \lceil (\kappa - 1) / 0.25^2 \rceil`."""
    return math.ceil((kappa - 1.0) / TAIL_SD_TARGET**2)


def gated_size(n_kappa: int) -> int:
    """The least multiple of :data:`WALD_GATED_STEP` at or above twice the tail size."""
    return WALD_GATED_STEP * math.ceil(WALD_REGIME_FACTOR * n_kappa / WALD_GATED_STEP)


@dataclass(frozen=True)
class TailSize:
    """The moments of one influence curve.

    Parameters
    ----------
    mass : float
        The total probability the grid integrates, 1 to quadrature precision.
    mean : float
        :math:`E D`, 0 for an influence curve.
    sd : float
        The SD of :math:`D`.
    skewness : float
        The skewness of :math:`D`.
    kappa : float
        The raw kurtosis :math:`E (D - ED)^4 / \\operatorname{Var}(D)^2`.
    """

    mass: float
    mean: float
    sd: float
    skewness: float
    kappa: float

    @property
    def n_kappa(self) -> int:
        """The tail size of :func:`tail_size`."""
        return tail_size(self.kappa)


def classify(
    sizes: Mapping[str, int], tail_sizes: Mapping[str, int | None]
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Split Wald cells by the tail-size rule.

    Parameters
    ----------
    sizes : mapping of str to int
        Each cell's n.
    tail_sizes : mapping of str to int or None
        Each cell's :math:`n_\\kappa`, or ``None`` when it was not computed.

    Returns
    -------
    tuple of frozenset of str
        The gated cells (n at least twice the tail size), the cells below the rule, and the
        cells with no computed tail size.
    """
    gated, below, missing = set(), set(), set()
    for cell, n in sizes.items():
        n_kappa = tail_sizes.get(cell)
        if n_kappa is None:
            missing.add(cell)
        elif n >= WALD_REGIME_FACTOR * n_kappa:
            gated.add(cell)
        else:
            below.add(cell)
    return frozenset(gated), frozenset(below), frozenset(missing)


# --------------------------------------------------------------------------------- grid


def _extra(u: float) -> tuple[float, ...]:
    """Breaks of the ratios beyond :func:`~tests.studies.mtp_boundaries_common._breaks`."""
    return (
        law.BRANCH_UP,
        law.BRANCH_DOWN,
        law.BRANCH_SPLIT + law.BRANCH_UP,
        u - law.BRANCH_DOWN,
        # Where the p4 policy dose crosses a jump of the p4 ratio, for the diagonal form.
        law.BRANCH_SPLIT - law.BRANCH_DOWN - law.BRANCH_UP,
        law.BRANCH_SPLIT + law.BRANCH_UP + law.BRANCH_DOWN,
        law.ATOM_SCALE * u,
        # The top of the p1 and the bottom of the p3 induced support.
        law.SCALE_FACTOR * u,
        float(law._power(0.0, u)),
        # Where the p2 policy dose crosses the jump of the p2 ratio at u - 1.
        u - 2.0 * law.STEP,
    )


def _ratio(
    label: str, node: int, a: np.ndarray, mean: np.ndarray, u: float, mutation: Mutation
) -> np.ndarray:
    """The plan's node ratio; ``one_branch`` drops the second inverse branch of ``p4``."""
    if mutation == "one_branch" and law.MAPS[label][node - 1] is law._branches:
        g = law.tn_pdf(a, mean, u)
        safe = np.where(g > 0.0, g, 1.0)
        first = law.tn_pdf(a - law.BRANCH_UP, mean, u) * (a - law.BRANCH_UP < law.BRANCH_SPLIT)
        return np.where(g > 0.0, first / safe, 0.0)
    return np.asarray(law.node_ratio(label, node, a, mean, u), dtype=float)


@dataclass(frozen=True)
class _Cell:
    """One ``(W2, W1, A1, L2, A2)`` block of the grid: weights and the curve's pieces."""

    weight: np.ndarray  # joint weight of every point, without the outcome
    m2o: np.ndarray  # outcome mean at the observed dose
    pieces: dict[str, tuple[np.ndarray, np.ndarray]]  # label -> (non-outcome part, ratio)


def _blocks(
    labels: Iterable[str], label_law: str, order: int, mutation: Mutation
) -> Iterable[_Cell]:
    """Each block of the grid with ``D = part + ratio (Y - m2o)`` for every label.

    The ``W1`` nodes go in chunks of :data:`_W1_CHUNK`, which bounds the memory of a block.
    """
    labels = tuple(labels)
    nodes, weights = law.w1_nodes(order)
    for start in range(0, nodes.size, _W1_CHUNK):
        rows = slice(start, start + _W1_CHUNK)
        yield from _chunk(labels, label_law, order, mutation, nodes[rows], weights[rows])


#: The ``W1`` nodes of one block of :func:`_blocks`.
_W1_CHUNK = 4


def _chunk(
    labels: tuple[str, ...],
    label_law: str,
    order: int,
    mutation: Mutation,
    w1: np.ndarray,
    weight_w: np.ndarray,
) -> Iterable[_Cell]:
    atom = label_law == "B_atom"
    for w2 in (0.0, 1.0):
        u = float(law.upper(w2))
        extra = _extra(u)
        a1, weight_a1 = law.dose_nodes(u, order, extra)
        w = w1[:, None]
        g1 = law.tn_pdf(a1[None, :], law.mean1(w, w2), u)
        flat_w = np.broadcast_to(w, (w1.size, a1.size)).ravel()
        flat_a = np.broadcast_to(a1[None, :], (w1.size, a1.size)).ravel()
        first: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label in labels:
            d1, _ = law.MAPS[label]
            observed = law.first_regression(
                label, label_law, flat_w, np.full(flat_w.size, w2), flat_a, order
            ).reshape(w1.size, a1.size)
            policy = law.first_regression(
                label, label_law, flat_w, np.full(flat_w.size, w2), d1(flat_a, u), order
            ).reshape(w1.size, a1.size)
            ratio = _ratio(label, 1, a1[None, :], law.mean1(w, w2), u, mutation)
            first[label] = (observed, policy, ratio)
        a2, weight_a2 = law.dose_nodes(u, order, extra)
        if atom:
            # The point mass at A2 = 3 is one more column of weight one; no Gauss node of
            # the continuous part lands on it.
            a2 = np.append(a2, law.ATOM)
            weight_a2 = np.append(weight_a2, 1.0)
        for l2 in (0.0, 1.0):
            p = law.p_l2(w, a1[None, :])
            p = p if l2 else 1.0 - p
            w3, a13, a23 = w[..., None], a1[None, :, None], a2[None, None, :]
            mean = law.mean2(w3, w2, a13, l2)
            density = law.tn_pdf(a23, mean, u)
            p_atom = law.p_atom(w3, l2) if atom else None
            if atom:
                density = (1.0 - p_atom) * density
                density[..., -1] = np.broadcast_to(p_atom, density.shape)[..., -1]
            weight = (
                0.5
                * weight_w[:, None, None]
                * weight_a1[None, :, None]
                * weight_a2[None, None, :]
                * g1[..., None]
                * p[..., None]
                * density
            )
            m2o = law.outcome_mean(w3, a13, l2, a23)
            pieces = {}
            for label in labels:
                _, d2 = law.MAPS[label]
                observed, policy, r1 = first[label]
                if atom:
                    r2 = np.asarray(law.node_ratio(label, 2, a23, mean, u, p_atom), dtype=float)
                else:
                    r2 = _ratio(label, 2, a23, mean, u, mutation)
                m2p = law.outcome_mean(w3, a13, l2, d2(a23, u))
                psi = law.truth(label, label_law)
                part = (policy - psi)[..., None] + r1[..., None] * (m2p - observed[..., None])
                cumulative = r1[..., None] if mutation == "first_ratio_only" else r1[..., None] * r2
                pieces[label] = (part, cumulative)
            yield _Cell(weight, m2o, pieces)


def _moments(pieces: Iterable[tuple[np.ndarray, np.ndarray]]) -> TailSize:
    sums = np.zeros(5)
    for value, weight in pieces:
        for k in range(5):
            sums[k] += float(np.sum(weight * value**k))
    mass, m1, m2, m3, m4 = sums
    variance = m2 - m1**2
    third = m3 - 3 * m1 * m2 + 2 * m1**3
    fourth = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    return TailSize(
        mass=float(mass),
        mean=float(m1),
        sd=float(np.sqrt(variance)),
        skewness=float(third / variance**1.5),
        kappa=float(fourth / variance**2),
    )


@cache
def law_b_moments(
    plan: str, quantity: Quantity, order: int = 40, mutation: Mutation = "none"
) -> TailSize:
    """The moments of the efficient influence function of ``ey[plan]`` or ``ate[plan]``.

    Parameters
    ----------
    plan : str
        ``"p1"`` to ``"p4"`` on law B, or ``"p5"`` on law B'.
    quantity : {"ey", "ate"}
        The level, or its difference from the natural plan on the same law.
    order : int, default=40
        The Gauss-Legendre order of each segment.
    mutation : {"none", "one_branch", "first_ratio_only"}, default="none"
        A deliberate defect: drop the second inverse branch of the ``p4`` ratio, or replace
        the cumulative ratio of the last term by the first node's ratio.

    Returns
    -------
    TailSize
        The moments, with the total mass of the grid.
    """
    label_law = law.LAW_OF[plan]
    labels = (plan, law.REFERENCE)
    psi0 = law.truth(law.REFERENCE, label_law)

    def pieces() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        # One block at a time, so the grid is never held whole.
        for cell in _blocks(labels, label_law, order, mutation):
            part, ratio = cell.pieces[plan]
            for y in (0.0, 1.0):
                chance = cell.m2o if y else 1.0 - cell.m2o
                curve = part + ratio * (y - cell.m2o)
                if quantity == "ate":
                    curve = curve - (y - psi0)
                yield curve, cell.weight * chance

    return _moments(pieces())


# ------------------------------------------------------------------ the diagonal form


def _covariance_by_moments(left: str, right: str, order: int) -> float:
    """:math:`E[D^L D^R]` on law B by the same grid as :func:`law_b_moments`."""
    total = 0.0
    for cell in _blocks((left, right), "B", order, "none"):
        part_l, ratio_l = cell.pieces[left]
        part_r, ratio_r = cell.pieces[right]
        for y in (0.0, 1.0):
            chance = cell.m2o if y else 1.0 - cell.m2o
            total += float(
                np.sum(
                    cell.weight
                    * chance
                    * (part_l + ratio_l * (y - cell.m2o))
                    * (part_r + ratio_r * (y - cell.m2o))
                )
            )
    return total


def _inner_terms(
    left: str,
    right: str,
    w: np.ndarray,
    w2: float,
    a1: np.ndarray,
    order: int,
    shift: Shift,
) -> np.ndarray:
    r""":math:`S_1(a_1, w)` at each ``(w, a1)`` point, by quadrature over ``L2`` and ``A2``."""
    u = float(law.upper(w2))
    d1_left, d2_left = law.MAPS[left]
    d1_right, d2_right = law.MAPS[right]
    flat_w, flat_a = w.ravel(), a1.ravel()
    # The policy_residual defect reads each node's regression at the plan's policy dose.
    at_left = d1_left(flat_a, u) if shift == "policy_residual" else flat_a
    at_right = d1_right(flat_a, u) if shift == "policy_residual" else flat_a
    m1_left = law.first_regression(left, "B", flat_w, np.full(flat_w.size, w2), at_left, order)
    m1_right = law.first_regression(right, "B", flat_w, np.full(flat_w.size, w2), at_right, order)
    a2, weight_a2 = law.dose_nodes(u, order, _extra(u))
    ww, aa = flat_w[:, None], flat_a[:, None]
    total = np.zeros(flat_w.size)
    for l2 in (0.0, 1.0):
        p = law.p_l2(flat_w, flat_a)
        p = p if l2 else 1.0 - p
        mean = law.mean2(ww, w2, aa, l2)
        g2 = law.tn_pdf(a2[None, :], mean, u)
        e_left = law.outcome_mean(ww, aa, l2, d2_left(a2[None, :], u)) - m1_left[:, None]
        e_right = law.outcome_mean(ww, aa, l2, d2_right(a2[None, :], u)) - m1_right[:, None]
        # The node-2 change of measure: the source plan moves the dose, the other plan's
        # ratio is read at that policy dose.
        moved = a2[None, :] if shift == "observed_dose" else d2_left(a2[None, :], u)
        ratio = (
            np.ones_like(moved) if shift == "no_ratio" else law.node_ratio(right, 2, moved, mean, u)
        )
        m2 = law.outcome_mean(ww, aa, l2, moved)
        target = m2 * (1.0 - m2)
        if shift == "policy_residual":
            # E[(Y - m_2(d^L(a)))(Y - m_2(d^R(a))) | a] at the moved dose a.
            target = target + (m2 - law.outcome_mean(ww, aa, l2, d2_left(moved, u))) * (
                m2 - law.outcome_mean(ww, aa, l2, d2_right(moved, u))
            )
        carried = ratio * target
        total += p * np.sum(weight_a2 * g2 * (e_left * e_right + carried), axis=1)
    return total.reshape(w.shape)


def diagonal_variance(
    left: str, right: str | None = None, order: int = 40, shift: Shift = "none"
) -> float:
    r"""The diagonal form :math:`\sigma^2_{LR}` on law B with the true regressions.

    Parameters
    ----------
    left : str
        The plan whose dose map pushes the measure forward.
    right : str or None, default=None
        ``left`` itself (``None``) or the natural plan.
    order : int, default=40
        The Gauss-Legendre order of each segment.
    shift : {"none", "observed_dose", "no_ratio", "policy_residual"}, default="none"
        A deliberate defect: evaluate each :math:`S_t` at :math:`A_t` instead of
        :math:`A^d_t`, drop the factor :math:`r(A^d)`, or take each node's residual against
        its regression at the policy dose :math:`A^d_t` rather than the observed dose (the
        exact-law reading of witness 3a).

    Returns
    -------
    float
        :math:`\sigma^2_{LR}`, which equals :math:`E[D^L D^R]` when ``shift="none"``.
    """
    right = left if right is None else right
    if right not in (left, law.REFERENCE):
        raise ValueError("the diagonal form here pairs a plan with itself or the natural plan")
    d1_left, _ = law.MAPS[left]
    psi_left, psi_right = law.truth(left, "B"), law.truth(right, "B")
    nodes, weights = law.w1_nodes(order)
    total = 0.0
    for start, w2 in itertools.product(range(0, nodes.size, _W1_CHUNK), (0.0, 1.0)):
        w1 = nodes[start : start + _W1_CHUNK]
        weight_w = weights[start : start + _W1_CHUNK]
        u = float(law.upper(w2))
        a1, weight_a1 = law.dose_nodes(u, order, _extra(u))
        w = np.broadcast_to(w1[:, None], (w1.size, a1.size))
        a = np.broadcast_to(a1[None, :], w.shape)
        mean = law.mean1(w, w2)
        g1 = law.tn_pdf(a, mean, u)
        joint = 0.5 * weight_w[:, None] * weight_a1[None, :] * g1
        # b = m_1(W, d_1(A_1)) - psi for each plan.
        b_left = (
            law.first_regression(
                left, "B", w.ravel(), np.full(w.size, w2), d1_left(a, u).ravel(), order
            ).reshape(w.shape)
            - psi_left
        )
        d1_right, _ = law.MAPS[right]
        b_right = (
            law.first_regression(
                right, "B", w.ravel(), np.full(w.size, w2), d1_right(a, u).ravel(), order
            ).reshape(w.shape)
            - psi_right
        )
        total += float(np.sum(joint * b_left * b_right))
        moved = a if shift == "observed_dose" else d1_left(a, u)
        ratio = (
            np.ones_like(moved) if shift == "no_ratio" else law.node_ratio(right, 1, moved, mean, u)
        )
        inner = _inner_terms(left, right, w, w2, moved, order, shift)
        total += float(np.sum(joint * ratio * inner))
    return total


def moments_covariance(left: str, right: str | None = None, order: int = 40) -> float:
    r""":math:`E[D^L D^R]` by the moment grid, the reference :func:`diagonal_variance` meets."""
    return _covariance_by_moments(left, left if right is None else right, order)
