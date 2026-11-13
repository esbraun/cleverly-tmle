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
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np

from tests.studies import mtp_boundaries_common as law
from tests.studies import mtp_boundaries_own_curve as own_b
from tests.studies import mtp_pricing_common as pricing_law
from tests.studies import mtp_pricing_own_curve as own_p
from tests.studies.longitudinal_sdr_properties import WALD_GATED_STEP, WALD_REGIME_FACTOR

__all__ = [
    "TAIL_SD_TARGET",
    "MonteCarloTail",
    "TailSize",
    "classify",
    "diagonal_variance",
    "gated_size",
    "law_b_blocks",
    "law_b_moments",
    "law_b_own_moments",
    "law_p_curves",
    "law_p_moments",
    "law_p_quadrature",
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


# ------------------------------------------------------------- law P, by Monte Carlo


#: The fits whose own curves :func:`law_p_moments` reads at each ``tau``:
#: ``(learners, estimator)``.  ``"revenue"`` is the property learners on the revenue, read at
#: ``tau = 3`` only, where the revenue cell is.
LAW_P_FITS: dict[int, tuple[tuple[str, str], ...]] = {
    tau: (
        ("primary", "ltmle"),
        ("primary", "sdr"),
        ("property", "ltmle"),
        *((("revenue", "ltmle"),) if tau == 3 else ()),
    )
    for tau in (2, 3, 5)
}
#: The rows of the limit fit, the draws the moments read, and the rows of one chunk.
LAW_P_FIT_ROWS = 10_000_000
LAW_P_DRAWS = 100_000_000
LAW_P_CHUNK = 1_000_000
#: The seeds of the limit fit (plus ``tau``) and of the draws (plus ``1000 tau + chunk``).
LAW_P_FIT_SEED = 2026111300
LAW_P_DRAW_SEED = 2026111400
#: The bootstrap resamples of the chunks and the one-sided level of the declared bound.
LAW_P_BOOTSTRAP = 10_000
LAW_P_BOOTSTRAP_SEED = 2026111500
LAW_P_LEVEL = 0.99


@dataclass(frozen=True)
class MonteCarloTail:
    """The moments of one own curve of law P from chunked draws.

    Parameters
    ----------
    draws : int
        The draws read.
    sd : float
        The SD of the curve.
    skewness : float
        Its skewness.
    kappa : float
        Its raw kurtosis.
    kappa_lower : float
        The 1% quantile of the chunk bootstrap of ``kappa``.
    kappa_upper : float
        The 99% quantile, at which the tail size is declared.
    """

    draws: int
    sd: float
    skewness: float
    kappa: float
    kappa_lower: float
    kappa_upper: float

    @property
    def n_kappa(self) -> int:
        """The tail size at the 99% upper bound of ``kappa``."""
        return tail_size(self.kappa_upper)


def _central(sums: np.ndarray) -> tuple[float, float, float]:
    """``(SD, skewness, kappa)`` from the count and the first four power sums."""
    count, s1, s2, s3, s4 = sums
    m1, m2, m3, m4 = s1 / count, s2 / count, s3 / count, s4 / count
    variance = m2 - m1**2
    third = m3 - 3 * m1 * m2 + 2 * m1**3
    fourth = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    return float(np.sqrt(variance)), float(third / variance**1.5), float(fourth / variance**2)


def _power_sums(curve: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    w = np.ones_like(curve) if weight is None else weight
    return np.array([np.sum(w * curve**k) for k in range(5)])


def law_p_fit(
    tau: int, learners: str, estimator: str, rows: int = LAW_P_FIT_ROWS
) -> own_p.OwnCurve:
    """The own curve's limit nuisances: one fit on ``rows`` draws of law P."""
    frame = pricing_law.sample(rows, LAW_P_FIT_SEED + tau, tau)
    cols = own_p.columns(frame, tau)
    del frame
    return own_p.OwnCurve(tau, estimator, learners).fit(cols)  # type: ignore[arg-type]


def law_p_curves(
    tau: int, *, rows: int = LAW_P_FIT_ROWS, checkpoint: Path | None = None
) -> dict[tuple[str, str], own_p.OwnCurve]:
    """Every fit of :data:`LAW_P_FITS` at ``tau``, kept in ``checkpoint`` when one is given."""
    curves: dict[tuple[str, str], own_p.OwnCurve] = {}
    for learners, estimator in LAW_P_FITS[tau]:
        stored = None if checkpoint is None else checkpoint / f"tau{tau}_{learners}_{estimator}.pkl"
        if stored is not None and stored.exists():
            curves[learners, estimator] = pickle.loads(stored.read_bytes())
            continue
        curves[learners, estimator] = law_p_fit(tau, learners, estimator, rows)
        if stored is not None:
            stored.write_bytes(pickle.dumps(curves[learners, estimator]))
    return curves


def law_p_moments(
    tau: int,
    *,
    draws: int = LAW_P_DRAWS,
    rows: int = LAW_P_FIT_ROWS,
    chunk: int = LAW_P_CHUNK,
    checkpoint: Path | None = None,
) -> dict[tuple[str, str, str], MonteCarloTail]:
    """The moments of each own curve of :data:`LAW_P_FITS` at ``tau``, by Monte Carlo.

    The limit nuisances come from one fit on ``rows`` draws (:func:`law_p_fit`).  The curve is
    then evaluated on ``draws`` fresh draws in chunks of ``chunk``, each chunk's power sums kept,
    and ``kappa``'s 99% upper bound read from :data:`LAW_P_BOOTSTRAP` resamples of the chunks.
    The cumulative ratio of five-node ``price_down`` puts its tail on rare rows, so a single
    :math:`10^6`-row fit read ``kappa`` 1,362 to 3,682; the bound carries that noise into the
    declared tail size.

    Parameters
    ----------
    tau : int
        Nodes, 2, 3 or 5.
    draws : int, default=LAW_P_DRAWS
        The draws the moments read, a multiple of ``chunk``.
    rows : int, default=LAW_P_FIT_ROWS
        The rows of the limit fit.
    chunk : int, default=LAW_P_CHUNK
        The rows of one chunk.
    checkpoint : pathlib.Path or None, default=None
        A directory that keeps each fit and each chunk's sums, so a stopped run resumes.

    Returns
    -------
    dict
        ``(learners, estimator, quantity) -> MonteCarloTail``, the quantities
        ``ey[<plan>]`` and ``ate[<plan>]`` of every plan.
    """
    curves = law_p_curves(tau, rows=rows, checkpoint=checkpoint)
    names: list[tuple[str, str, str]] = []
    parts: list[np.ndarray] = []
    for k in range(draws // chunk):
        stored = None if checkpoint is None else checkpoint / f"tau{tau}_chunk{k:04d}.npz"
        if stored is not None and stored.exists():
            saved = np.load(stored, allow_pickle=False)
            names = [tuple(name.split("|")) for name in saved["names"]]
            parts.append(saved["sums"])
            continue
        frame = pricing_law.sample(chunk, LAW_P_DRAW_SEED + 1000 * tau + k, tau)
        cols = own_p.columns(frame, tau)
        names, rows_of = [], []
        known = {label: own_p.ratios(cols, label, tau) for label in pricing_law.LABELS}
        for (learners, estimator), curve in curves.items():
            for quantity, values in curve.quantities(cols, known).items():
                names.append((learners, estimator, quantity))
                rows_of.append(_power_sums(values))
        sums = np.array(rows_of)
        parts.append(sums)
        if stored is not None:
            np.savez(stored, names=np.array(["|".join(n) for n in names]), sums=sums)
    stacked = np.stack(parts)  # (chunks, quantities, 5)
    rng = np.random.default_rng(LAW_P_BOOTSTRAP_SEED + tau)
    counts = rng.multinomial(
        stacked.shape[0], np.full(stacked.shape[0], 1.0 / stacked.shape[0]), size=LAW_P_BOOTSTRAP
    )
    out = {}
    for j, name in enumerate(names):
        pooled = stacked[:, j, :].sum(axis=0)
        sd, skewness, kappa = _central(pooled)
        resampled = counts @ stacked[:, j, :]
        kappas = np.array([_central(row)[2] for row in resampled])
        out[name] = MonteCarloTail(
            draws=round(pooled[0]),
            sd=sd,
            skewness=skewness,
            kappa=kappa,
            kappa_lower=float(np.quantile(kappas, 1.0 - LAW_P_LEVEL)),
            kappa_upper=float(np.quantile(kappas, LAW_P_LEVEL)),
        )
    return out


def law_p_quadrature(curve: own_p.OwnCurve, order: int) -> dict[str, TailSize]:
    """The moments of a two-node own curve by quadrature (:func:`own_p.quadrature_columns`).

    It checks the Monte Carlo of :func:`law_p_moments` at ``tau = 2``, where the law admits
    a grid: the same limit nuisances, integrated rather than drawn.
    """
    if curve.tau != 2:
        raise ValueError("the law-P quadrature is written for two nodes")
    sums: dict[str, np.ndarray] = {}
    for cols, weight in own_p.quadrature_columns(order):
        for quantity, values in curve.quantities(cols).items():
            sums[quantity] = sums.get(quantity, 0.0) + _power_sums(values, weight)
    out = {}
    for quantity, total in sums.items():
        sd, skewness, kappa = _central(total)
        out[quantity] = TailSize(
            mass=float(total[0]),
            mean=float(total[1] / total[0]),
            sd=sd,
            skewness=skewness,
            kappa=kappa,
        )
    return out


# ------------------------------------------------------- law B, the fits' own curves


#: The own curves :func:`law_b_own_moments` reads: ``(law, learners, estimator) -> plans``.
#: The primary learners read every plan of the primary scenarios; the calibration learners read
#: the plans of the calibration cells (``LTMLE`` under ``p1`` to ``p5``, ``SDR`` under ``p4``
#: and ``p5``).
LAW_B_FITS: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("B", "primary", "ltmle"): law.B_LABELS,
    ("B", "primary", "sdr"): law.B_LABELS,
    ("B", "correct", "ltmle"): law.B_LABELS,
    ("B", "correct", "sdr"): (law.REFERENCE, "p4"),
    ("B_atom", "primary", "ltmle"): law.B_ATOM_LABELS,
    ("B_atom", "primary", "sdr"): law.B_ATOM_LABELS,
    ("B_atom", "correct", "ltmle"): law.B_ATOM_LABELS,
    ("B_atom", "correct", "sdr"): law.B_ATOM_LABELS,
}


def law_b_blocks(label_law: str, order: int) -> list[own_b.Block]:
    """The grid of :func:`law_b_moments` as weighted blocks of the own-curve fit.

    Each block holds :data:`_W1_CHUNK` ``W1`` nodes at one ``W2``: the node-1 points
    ``(W1, A1)`` with their mass, and under each the ``(L2, A2)`` points with their
    conditional mass and the outcome mean.  On law B' the atom is one more ``A2`` column.
    """
    atom = label_law == "B_atom"
    nodes, weights = law.w1_nodes(order)
    out = []
    for start, w2 in itertools.product(range(0, nodes.size, _W1_CHUNK), (0.0, 1.0)):
        w1, weight_w = nodes[start : start + _W1_CHUNK], weights[start : start + _W1_CHUNK]
        u = float(law.upper(w2))
        a1, weight_a1 = law.dose_nodes(u, order, _extra(u))
        a2, weight_a2 = law.dose_nodes(u, order, _extra(u))
        if atom:
            a2, weight_a2 = np.append(a2, law.ATOM), np.append(weight_a2, 1.0)
        ww, aa = np.meshgrid(w1, a1, indexing="ij")
        mass = 0.5 * np.outer(weight_w, weight_a1) * law.tn_pdf(aa, law.mean1(ww, w2), u)
        w_flat, a_flat = ww.ravel(), aa.ravel()
        l2s, a2s, inners, ys = [], [], [], []
        for l2 in (0.0, 1.0):
            p = law.p_l2(w_flat, a_flat)
            p = p if l2 else 1.0 - p
            mean = law.mean2(w_flat[:, None], w2, a_flat[:, None], l2)
            density = law.tn_pdf(a2[None, :], mean, u)
            if atom:
                p_atom = law.p_atom(w_flat[:, None], l2)
                density = (1.0 - p_atom) * density
                density[:, -1] = p_atom[:, 0]
            shape = density.shape
            l2s.append(np.full(shape, l2))
            a2s.append(np.broadcast_to(a2[None, :], shape))
            inners.append(p[:, None] * weight_a2[None, :] * density)
            ys.append(law.outcome_mean(w_flat[:, None], a_flat[:, None], l2, a2[None, :]))
        out.append(
            own_b.Block(
                w1=w_flat,
                w2=np.full(w_flat.size, w2),
                a1=a_flat,
                mass=mass.ravel(),
                l2=np.hstack(l2s),
                a2=np.hstack(a2s),
                inner=np.hstack(inners),
                y=np.hstack(ys),
                atom=atom,
            )
        )
    return out


def law_b_own_moments(
    label_law: str, learners: str, estimator: str, order: int = 40
) -> dict[str, TailSize]:
    """The moments of a fit's own curve on law B or B', with its limit nuisances.

    The limits solve the population normal equations of the declared learners on the grid of
    :func:`law_b_blocks` (:class:`~tests.studies.mtp_boundaries_own_curve.OwnCurve`), and the
    moments of ``ey[<plan>]`` and ``ate[<plan>]`` integrate the curve on the same grid.

    Parameters
    ----------
    label_law : {"B", "B_atom"}
        The law.
    learners : {"primary", "correct"}
        The primary scenarios' learners, or the calibration cells'.
    estimator : {"ltmle", "sdr"}
        The recursion.
    order : int, default=40
        The Gauss-Legendre order of each segment.

    Returns
    -------
    dict of str to TailSize
        Each quantity's moments.
    """
    labels = LAW_B_FITS[label_law, learners, estimator]
    blocks = law_b_blocks(label_law, order)
    curve = own_b.OwnCurve(estimator, learners, labels).fit(lambda: blocks)  # type: ignore[arg-type]
    sums: dict[str, np.ndarray] = {}
    for block in blocks:
        parts = {label: curve.parts(curve.plans[label], block) for label in labels}
        reference = parts[law.REFERENCE]
        for label, (base, slope) in parts.items():
            quantities = [(f"ey[{label}]", base, slope)]
            if label != law.REFERENCE:
                quantities.append((f"ate[{label}]", base - reference[0], slope - reference[1]))
            for name, b, s in quantities:
                total = np.zeros(5)
                for y in (0.0, 1.0):
                    chance = block.y if y else 1.0 - block.y
                    value = b + s * y
                    weight = block.weight * chance
                    total += np.array([np.sum(weight * value**k) for k in range(5)])
                sums[name] = sums.get(name, 0.0) + total
    out = {}
    for name, total in sums.items():
        sd, skewness, kappa = _central(total)
        out[name] = TailSize(
            mass=float(total[0]),
            mean=float(total[1] / total[0]),
            sd=sd,
            skewness=skewness,
            kappa=kappa,
        )
    return out
