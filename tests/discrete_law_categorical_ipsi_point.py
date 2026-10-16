r"""The point-treatment laws of the study ``categorical-incremental`` (X43, family P).

A baseline :math:`W` with four values, a treatment :math:`A` with three levels labelled
``"a"``, ``"b"`` and ``"c"``, and a continuous outcome :math:`Y = \mu(A, W) + \varepsilon` with
:math:`\varepsilon \sim N(0, 1)`.  Every truth is a finite sum over the twelve
:math:`(W, A)` cells.

======  ==============================================================================
law     structure
======  ==============================================================================
``P``   the study law: :math:`g(a \mid w) \in [0.10, 0.70]` and the outcome table
        :data:`MU`
``P0``  the sharp null: :math:`\mu(a, w) = \bar\mu(w)`, the mean of :data:`MU` over the
        three levels, so every tilt has one mean and the mechanism term vanishes
``Pw``  the power law: :math:`\bar\mu(w) + s\{\mu(a, w) - \bar\mu(w)\}` with
        :math:`s` = :data:`POWER_SCALE`
======  ==============================================================================

A categorical tilt multiplies the odds of each level by a factor,
:math:`q(a \mid w) = \delta_a g(a \mid w) / D(w)` with :math:`D = \sum_b \delta_b g(b \mid w)`,
and its mean is :math:`\Psi(\delta) = E\{m(W)\}` with :math:`m = \sum_a q(a \mid W)\mu(a, W)`.
The efficient influence function is (``X43-plan.md`` 2.1)

.. math::

    \varphi = \underbrace{h(A)\{Y - \mu(A, W)\}}_{\text{outcome score}}
            + \underbrace{h(A)\{\mu(A, W) - m(W)\}}_{\text{mechanism term}} + m(W) - \Psi,
    \qquad h(a) = \delta_a / D(W).

The three parts are orthogonal: the outcome score has mean zero given :math:`(A, W)`, and the
mechanism term has mean zero given :math:`W`.  :func:`eif_parts` writes them in closed form.
:func:`gateaux` is the second, independent route: the complex-step derivative of
:func:`functional` toward one :math:`(W, A)` cell, which reads :math:`g` off the perturbed
masses and so carries the mechanism term without being told about it.  The design test
asserts that the two agree to 1e-10.

The module imports nothing from ``cleverly`` (``tests/unit/test_oracle_independence.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

#: The treatment labels, in this module's arm order.  They sort to the same order, so the
#: library's arm codes equal this module's indices.
LABELS: tuple[str, ...] = ("a", "b", "c")
K = len(LABELS)
#: The outcome noise.
SIGMA = 1.0
#: ``P(W = w)``.
P_W = np.array([0.30, 0.25, 0.25, 0.20])
#: ``g(a | w)``, indexed ``[w, a]``.
G = np.array(
    [
        [0.60, 0.25, 0.15],
        [0.30, 0.50, 0.20],
        [0.20, 0.30, 0.50],
        [0.10, 0.20, 0.70],
    ]
)
#: ``mu(a, w)``, indexed ``[w, a]``.  Not monotone or additive in ``a``.
MU = np.array(
    [
        [4.5, 0.5, 3.0],
        [1.0, 0.0, 5.0],
        [0.5, 0.5, 4.5],
        [2.5, 1.0, 4.5],
    ]
)
#: The arm-effect scale of law ``Pw``.  The design test asserts that the exact power of
#: ``ate_ipsi[T_a vs natural course]`` at the null size lies in [0.80, 0.90].
POWER_SCALE = 0.033
#: The complex step of :func:`gateaux`.
STEP = 1e-30

#: The tilts, by label.  A level a tilt does not name keeps the factor 1.
T_A: dict[str, float] = {"b": 2.0, "c": 0.5}
T_B: dict[str, float] = {"b": 0.5, "c": 2.0}
NATURAL: dict[str, float] = {"a": 1.0}
REFERENCE = "natural course"
TILTS: dict[str, dict[str, float]] = {REFERENCE: NATURAL, "T_a": T_A, "T_b": T_B}
MEANS: tuple[str, ...] = tuple(f"ey_ipsi[{name}]" for name in TILTS)
CONTRASTS: tuple[str, ...] = tuple(
    f"ate_ipsi[{name} vs {REFERENCE}]" for name in TILTS if name != REFERENCE
)
#: The gated contrast of the necessity, null and power cells.
GATED = f"ate_ipsi[T_a vs {REFERENCE}]"


def factor_vector(tilt: Mapping[str, float]) -> np.ndarray:
    """The factor of every level, in :data:`LABELS` order."""
    return np.array([float(tilt.get(label, 1.0)) for label in LABELS])


@dataclass(frozen=True)
class PointLaw:
    """One point law: the baseline masses, the mechanism and the outcome table.

    Parameters
    ----------
    name : str
        The law's label.
    p_w : ndarray
        ``P(W = w)``, shape ``(4,)``.
    g : ndarray
        ``g(a | w)``, shape ``(4, K)``.
    mu : ndarray
        ``mu(a, w)``, shape ``(4, K)``.
    """

    name: str
    p_w: np.ndarray = field(repr=False)
    g: np.ndarray = field(repr=False)
    mu: np.ndarray = field(repr=False)

    @cached_property
    def masses(self) -> np.ndarray:
        """``P(W = w, A = a)``, shape ``(4, K)``."""
        return self.p_w[:, None] * self.g


def _scaled(scale: float) -> np.ndarray:
    centre = MU.mean(axis=1, keepdims=True)
    return centre + scale * (MU - centre)


LAW_P = PointLaw("P", P_W, G, MU)
LAW_P0 = PointLaw("P0", P_W, G, _scaled(0.0))
LAW_PW = PointLaw("Pw", P_W, G, _scaled(POWER_SCALE))
LAWS: dict[str, PointLaw] = {"P": LAW_P, "P0": LAW_P0, "Pw": LAW_PW}


# ------------------------------------------------------------------------- the functional


def functional(masses: Any, mu: np.ndarray, factors: np.ndarray) -> Any:
    r""":math:`\Psi(\delta)` from the cell masses :math:`P(W, A)`; analytic in ``masses``."""
    p = np.asarray(masses)
    p_w = p.sum(axis=1)
    # A covariate value with no mass (a stratum's conditioned law) adds nothing.
    seen = np.real(p_w) > 0
    p, p_w, mu = p[seen], p_w[seen], np.asarray(mu)[seen]
    g = p / p_w[:, None]
    normaliser = (g * factors[None, :]).sum(axis=1)
    tilted = (g * factors[None, :] * mu).sum(axis=1) / normaliser
    return (p_w * tilted).sum()


def _pieces(law: PointLaw, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(D(w), h(w, a), m(w))`` of a tilt on ``law``."""
    normaliser = law.g @ factors
    weight = factors[None, :] / normaliser[:, None]
    tilted = (law.g * factors[None, :] * law.mu).sum(axis=1) / normaliser
    return normaliser, weight, tilted


def _plans(name: str) -> tuple[str, str | None]:
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")
    if stem == "ey_ipsi" and len(parts) == 1:
        return parts[0], None
    if stem == "ate_ipsi" and len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"not a point incremental parameter: {name!r}")


def truth(law: PointLaw, name: str, tilts: Mapping[str, Mapping[str, float]] = TILTS) -> float:
    """The exact value of ``ey_ipsi[...]`` or ``ate_ipsi[... vs ...]`` on ``law``."""
    left, right = _plans(name)
    value = float(functional(law.masses, law.mu, factor_vector(tilts[left])))
    if right is not None:
        value -= float(functional(law.masses, law.mu, factor_vector(tilts[right])))
    return value


@dataclass(frozen=True)
class EIFParts:
    """The three orthogonal parts of an influence function, on the ``(W, A)`` cells.

    Parameters
    ----------
    outcome : ndarray
        The coefficient :math:`c(w, a)` of the outcome residual :math:`Y - \\mu(A, W)`.
    mechanism : ndarray
        The mechanism term at each cell.
    baseline : ndarray
        :math:`m(w) - \\Psi`, shape ``(4,)``.
    """

    outcome: np.ndarray
    mechanism: np.ndarray
    baseline: np.ndarray


def eif_parts(
    law: PointLaw, name: str, tilts: Mapping[str, Mapping[str, float]] = TILTS
) -> EIFParts:
    """The closed-form parts of the influence function of ``name`` (module docstring)."""
    left, right = _plans(name)

    def one(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, weight, tilted = _pieces(law, factor_vector(tilts[label]))
        psi = float(law.p_w @ tilted)
        return weight, weight * (law.mu - tilted[:, None]), tilted - psi

    outcome, mechanism, baseline = one(left)
    if right is not None:
        other = one(right)
        outcome, mechanism, baseline = outcome - other[0], mechanism - other[1], baseline - other[2]
    return EIFParts(outcome, mechanism, baseline)


def _variance(law: PointLaw, parts: EIFParts, *, mechanism: np.ndarray | None = None) -> float:
    """The variance of the curve ``parts`` defines, with ``mechanism`` in place of its term."""
    term = parts.mechanism if mechanism is None else mechanism
    cell = term + parts.baseline[:, None]
    mean = float((law.masses * cell).sum())
    second = float((law.masses * cell**2).sum())
    noise = SIGMA**2 * float((law.masses * parts.outcome**2).sum())
    return second - mean**2 + noise


def eif_sd(law: PointLaw, name: str, tilts: Mapping[str, Mapping[str, float]] = TILTS) -> float:
    """The standard deviation of the efficient influence function of ``name``."""
    return float(np.sqrt(_variance(law, eif_parts(law, name, tilts))))


def mechanism_share(law: PointLaw, name: str) -> float:
    """The share of the influence-function variance that the mechanism term carries."""
    parts = eif_parts(law, name)
    term = float((law.masses * parts.mechanism**2).sum())
    return term / _variance(law, parts)


def dropped_sd(law: PointLaw, name: str) -> float:
    """The standard deviation of the curve without its mechanism term.

    ``mechanism_term_dropped`` reports this curve.  It equals the influence function of the
    same tilt with a known mechanism, so its ratio to :func:`eif_sd` is
    :math:`\\sqrt{1 - \\text{share}}`.
    """
    parts = eif_parts(law, name)
    return float(np.sqrt(_variance(law, parts, mechanism=np.zeros_like(parts.mechanism))))


def mechanism_coefficients(law: PointLaw, tilt: Mapping[str, float]) -> np.ndarray:
    r"""``(w, b)`` array of :math:`\partial m / \partial g_b = \delta_b(\mu_b - m)/D`."""
    factors = factor_vector(tilt)
    normaliser, _, tilted = _pieces(law, factors)
    return factors[None, :] * (law.mu - tilted[:, None]) / normaliser[:, None]


#: The levels the binary-blip control keeps: the first two arm codes, ``"a"`` and ``"b"``.
BLIP_LEVELS: tuple[int, ...] = (0, 1)


def blip_mechanism(law: PointLaw, name: str) -> np.ndarray:
    r"""The mechanism term of ``name`` built from the first two levels' scores only.

    :math:`\sum_{b \in \{a, b\}} \partial m/\partial g_b\,(1\{A = b\} - g_b)`: the term a
    binary implementation forms when it reads two columns of a three-level mechanism
    (mutation M2 of ``X43-plan.md`` 4.2).
    """
    left, right = _plans(name)
    coefficient = mechanism_coefficients(law, TILTS[left])
    if right is not None:
        coefficient = coefficient - mechanism_coefficients(law, TILTS[right])
    out = np.zeros_like(law.g)
    for a in range(K):
        for b in BLIP_LEVELS:
            out[:, a] += coefficient[:, b] * (float(a == b) - law.g[:, b])
    return out


def blip_sd(law: PointLaw, name: str) -> float:
    """The standard deviation of the binary-blip curve (:func:`blip_mechanism`)."""
    parts = eif_parts(law, name)
    return float(np.sqrt(_variance(law, parts, mechanism=blip_mechanism(law, name))))


def constant_mechanism_limit(
    law: PointLaw,
    tilts: Mapping[str, Mapping[str, float]] = TILTS,
    *,
    tolerance: float = 1e-13,
    rounds: int = 500,
) -> dict[str, float]:
    r"""The limits of a TMLE fit with a constant :math:`\hat g` and a correct :math:`\hat\mu`.

    The constant is the marginal :math:`g^*(a) = P(A = a)`, the limit of
    ``DummyClassifier(strategy="prior")``.  With :math:`\hat\mu = \mu` every outcome residual
    has mean zero given :math:`(A, W)`, so the outcome fluctuation stays at zero in the limit.
    The mechanism fluctuation is the softmax submodel
    :math:`g_\epsilon(a \mid w) \propto g(a \mid w)\exp\{\sum_r \epsilon_r H_r(a, w)\}` with
    :math:`H_r(a, w) = h_r(a)\{\mu(a, w) - m_r(w)\}` read at the current :math:`g`, one
    coefficient per tilt, refitted until every score
    :math:`E_0\{H_r(A, W)\}` vanishes (``X43-plan.md`` 3.1).  This function runs that
    iteration on the law itself: each round maximises the population log-likelihood
    :math:`\sum_{w, a} P_0(w, a) \log g_\epsilon(a \mid w)` by Newton steps.  The limit of
    each tilt's mean is :math:`E\{m_r^\dagger(W)\}` at the fixed point
    :math:`g^\dagger`, which differs from the truth because :math:`g^\dagger \neq g` (no
    double robustness, ``X43-plan.md`` 2.2).

    Returns
    -------
    dict of str to float
        The limit of ``ey_ipsi[name]`` for every tilt name.
    """
    factors = np.array([factor_vector(tilts[label]) for label in tilts])  # (R, K)
    current = np.broadcast_to(law.masses.sum(axis=0), law.g.shape).copy()

    def covariate(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normaliser = g @ factors.T  # (W, R)
        tilted = np.einsum("wa,ra,wa->wr", g, factors, law.mu) / normaliser
        weight = factors[None, :, :] / normaliser[:, :, None]  # (W, R, K)
        return weight * (law.mu[:, None, :] - tilted[:, :, None]), tilted

    for _ in range(rounds):
        h, _ = covariate(current)
        score = np.einsum("wa,wra->r", law.masses, h)
        if np.max(np.abs(score)) < tolerance:
            break
        epsilon = np.zeros(len(factors))
        for _ in range(100):
            logits = np.log(current) + np.einsum("r,wra->wa", epsilon, h)
            moved = np.exp(logits - logits.max(axis=1, keepdims=True))
            moved /= moved.sum(axis=1, keepdims=True)
            centred = h - np.einsum("wa,wra->wr", moved, h)[:, :, None]
            gradient = np.einsum("wa,wra->r", law.masses, centred)
            hessian = np.einsum("w,wa,wra,wsa->rs", law.p_w, moved, centred, centred)
            step = np.linalg.solve(hessian, gradient)
            epsilon += step
            if np.max(np.abs(step)) < 1e-15:
                break
        logits = np.log(current) + np.einsum("r,wra->wa", epsilon, h)
        current = np.exp(logits - logits.max(axis=1, keepdims=True))
        current /= current.sum(axis=1, keepdims=True)
    else:  # pragma: no cover - guards the iteration
        raise AssertionError("the population targeting iteration did not converge")
    _, tilted = covariate(current)
    return {label: float(law.p_w @ tilted[:, r]) for r, label in enumerate(tilts)}


def gateaux(law: PointLaw, name: str, w: int, a: int, *, step: float = STEP) -> float:
    """The cell part of the influence function at ``(w, a)``, by complex step.

    The derivative of :func:`functional` toward a point mass at ``(w, a)``.  With the
    outcome noise it gives :math:`E(\\varphi \\mid W = w, A = a)`.
    """
    left, right = _plans(name)
    cells = law.masses.astype(complex)
    mass = np.zeros_like(cells)
    mass[w, a] = 1.0
    perturbed = (1.0 - 1j * step) * cells + 1j * step * mass
    value = functional(perturbed, law.mu, factor_vector(TILTS[left]))
    if right is not None:
        value = value - functional(perturbed, law.mu, factor_vector(TILTS[right]))
    return float(np.imag(value) / step)


# ---------------------------------------------------------------------------- sampling

#: The stratum of each ``W`` value in the stratified cell: two strata of two values each.
STRATA_OF_W: tuple[int, ...] = (0, 0, 1, 1)


def sample(law: PointLaw, n: int, seed: int, *, strata: bool = False) -> pd.DataFrame:
    """``n`` units of ``law``: ``W`` as 0 to 3, ``A`` as its label, ``Y`` continuous.

    ``strata=True`` adds ``V``, the stratum of :data:`STRATA_OF_W`.
    """
    rng = np.random.default_rng(seed)
    flat = law.masses.reshape(-1)
    drawn = rng.choice(flat.size, size=n, p=flat / flat.sum())
    w, a = np.unravel_index(drawn, law.masses.shape)
    y = law.mu[w, a] + rng.normal(scale=SIGMA, size=n)
    frame = pd.DataFrame({"W": w.astype(float), "A": np.array(LABELS)[a], "Y": y})
    if strata:
        frame["V"] = np.asarray(STRATA_OF_W, dtype=float)[w]
    return frame


def stratum_law(law: PointLaw, stratum: int) -> PointLaw:
    """``law`` conditioned on ``V = stratum``."""
    inside = np.array([float(v == stratum) for v in STRATA_OF_W])
    p_w = inside * law.p_w
    return PointLaw(f"{law.name}|V={stratum}", p_w / p_w.sum(), law.g, law.mu)
