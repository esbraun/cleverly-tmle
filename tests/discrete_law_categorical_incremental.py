r"""The categorical incremental functional on the three-arm laws, by finite sums.

The oracle for ``Incremental({level: factor, ...})`` on a treatment with three levels.  It
sits beside :mod:`tests.discrete_law_multi` (complete outcomes) and
:mod:`tests.discrete_law_mar_multi` (outcomes missing at random), takes their cell
probabilities, and writes the tilted mean longhand:

.. math::

    q(a \mid w) = \frac{\delta_a\, g(a \mid w)}{\sum_b \delta_b\, g(b \mid w)},
    \qquad
    \Psi(\delta) = \sum_w P(w) \sum_a q(a \mid w)\, \bar Q(a, w),

with :math:`g` and :math:`\bar Q` read off the cell probabilities.  The functional is analytic
in those probabilities and the tilt reads :math:`g` *from the perturbed law*, so a complex-step
Gateaux derivative contains the mechanism term without being told about it.  The module
imports nothing from ``cleverly`` (``tests/unit/test_oracle_independence.py``).

Tilts are keyed by the analyst's labels, and :data:`tests.discrete_law_multi.LABELS` order
(low, mid, high) is this module's arm axis; ``CausalData`` sorts the labels to
(high, low, mid), so a tilt resolved by position instead of by label lands on the wrong level.

The module asserts, at import, that no term of the efficient influence function can vanish
unnoticed:

* each level's mechanism score coefficient :math:`\delta_b(\bar Q_b - m)/D` is at least
  ``0.02`` in absolute value somewhere with positive mass, for every level and both tilts;
* the mechanism term carries at least ``0.10`` of the influence-function variance;
* leaving any one level out of the mechanism term moves the influence function by more than
  ``0.02``, so a method that reads two of the three columns is visible;
* the tilt moves the mean by more than ``0.01`` from the natural course.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests import discrete_law_mar_multi as mar
from tests import discrete_law_multi as base

LABELS: tuple[str, ...] = base.LABELS
K = base.K
assert mar.LABELS == LABELS, "the two three-arm laws must share one label order"

#: The tilts, by label.  A level a tilt does not name keeps the factor ``1``.
T_A: dict[str, float] = {"mid": 2.0, "low": 0.5}
T_B: dict[str, float] = {"low": 0.25}
NATURAL: dict[str, float] = {"high": 1.0}
#: The factors of the collapsing identity (C6): ``high`` and ``mid`` against ``low``.
COLLAPSED: dict[str, float] = {"high": 4.0, "mid": 4.0}
#: Unequal non-baseline factors, for which a collapsed binary fit differs by more than 0.01.
UNEQUAL: dict[str, float] = {"high": 4.0, "mid": 0.25}

TILTS: dict[str, dict[str, float]] = {"natural course": NATURAL, "T_a": T_A, "T_b": T_B}
REFERENCE = "natural course"

#: Reported parameter names, with ``natural course`` the reference every contrast reads.
MEANS: tuple[str, ...] = tuple(f"ey_ipsi[{name}]" for name in TILTS)
CONTRASTS: tuple[str, ...] = tuple(
    f"ate_ipsi[{name} vs {REFERENCE}]" for name in TILTS if name != REFERENCE
)


def factor_vector(tilt: dict[str, float]) -> np.ndarray:
    """The factor of every level, on this module's arm axis (``LABELS`` order)."""
    return np.array([tilt.get(label, 1.0) for label in LABELS])


def _tables(probs: Any, missing: bool) -> tuple[Any, Any, Any]:
    """``P(W)``, ``g(a | w)`` and ``Qbar(a, w)`` from the cell probabilities."""
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))
    p_wa = p.sum(axis=2)
    if missing:
        observed = p[:, :, mar.OBSERVED_ZERO] + p[:, :, mar.OBSERVED_ONE]
        q = p[:, :, mar.OBSERVED_ONE] / observed
    else:
        q = p[:, :, 1] / p_wa
    return p_w, p_wa / p_w[:, None], q


def functional(probs: Any, factors: np.ndarray, *, missing: bool = False) -> Any:
    r""":math:`\Psi(\delta)` by finite sums.  Analytic in ``probs``: no ``clip``, ``abs`` or ``if``."""
    p_w, g, q = _tables(probs, missing)
    normaliser = (g * factors[None, :]).sum(axis=1)
    mixture = (g * factors[None, :] * q).sum(axis=1) / normaliser
    return (p_w * mixture).sum()


def gateaux(
    factors: np.ndarray, point: int, *, missing: bool = False, step: float = 1e-30
) -> float:
    """The Gateaux derivative of ``functional`` at support point ``point``, by complex step."""
    module = mar if missing else base
    cells = module.PROBS.astype(complex)
    mass = np.zeros_like(cells)
    mass[module.SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * cells + 1j * step * mass
    return float(np.imag(functional(perturbed, factors, missing=missing)) / step)


def eif(tilt: dict[str, float], *, missing: bool = False) -> np.ndarray:
    """The efficient influence function of ``tilt`` at every support point, in support order."""
    module = mar if missing else base
    f = factor_vector(tilt)
    return np.array([gateaux(f, point, missing=missing) for point in range(len(module.SUPPORT))])


def truth(tilt: dict[str, float], *, missing: bool = False) -> float:
    """The tilted mean under the law."""
    module = mar if missing else base
    return float(functional(module.PROBS, factor_vector(tilt), missing=missing))


def truth_of(name: str, *, missing: bool = False) -> float:
    """The truth of a reported name: ``ey_ipsi[T]`` or ``ate_ipsi[T vs natural course]``."""
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")
    values = [truth(TILTS[part], missing=missing) for part in parts]
    return values[0] if stem == "ey_ipsi" else values[0] - values[1]


def eif_of(name: str, *, missing: bool = False) -> np.ndarray:
    """The efficient influence function of a reported name, in support order."""
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")
    curves = [eif(TILTS[part], missing=missing) for part in parts]
    return curves[0] if stem == "ey_ipsi" else curves[0] - curves[1]


#: The baseline stratum ``V`` of each ``W`` value: a union of ``W`` cells, so a stratum
#: parameter is the tilted mean of the law restricted to those cells.
STRATA_OF_W: tuple[int, ...] = (0, 1, 1)


def functional_stratum(
    probs: Any, factors: np.ndarray, stratum: int, *, missing: bool = False
) -> Any:
    """The tilted mean given ``V = stratum``, analytic in the cell probabilities."""
    p_w, g, q = _tables(probs, missing)
    inside = np.array([float(v == stratum) for v in STRATA_OF_W])
    normaliser = (g * factors[None, :]).sum(axis=1)
    mixture = (g * factors[None, :] * q).sum(axis=1) / normaliser
    return (inside * p_w * mixture).sum() / (inside * p_w).sum()


def eif_stratum(tilt: dict[str, float], stratum: int) -> np.ndarray:
    """The efficient influence function of a stratum parameter, in support order."""
    f = factor_vector(tilt)
    out = []
    for point in range(len(base.SUPPORT)):
        cells = base.PROBS.astype(complex)
        mass = np.zeros_like(cells)
        mass[base.SUPPORT[point]] = 1.0
        step = 1e-30
        perturbed = (1.0 - 1j * step) * cells + 1j * step * mass
        out.append(float(np.imag(functional_stratum(perturbed, f, stratum)) / step))
    return np.array(out)


def truth_stratum(tilt: dict[str, float], stratum: int) -> float:
    """The tilted mean given ``V = stratum``."""
    return float(functional_stratum(base.PROBS, factor_vector(tilt), stratum))


def mechanism_coefficients(tilt: dict[str, float]) -> np.ndarray:
    r"""``(w, a)`` array of :math:`\partial m/\partial g_b = \delta_b(\bar Q_b - m)/D`."""
    f = factor_vector(tilt)
    _, g, q = _tables(base.PROBS, False)
    d = (g * f[None, :]).sum(axis=1)
    m = (g * f[None, :] * q).sum(axis=1) / d
    return f[None, :] * (q - m[:, None]) / d[:, None]


def mechanism_term(tilt: dict[str, float], *, skip: int | None = None) -> np.ndarray:
    """The mechanism term at every support point, optionally leaving out one level's score."""
    coefficient = mechanism_coefficients(tilt)
    _, g, _ = _tables(base.PROBS, False)
    out = []
    for w, a, _ in base.SUPPORT:
        total = 0.0
        for b in range(K):
            if b == skip:
                continue
            total += coefficient[w, b] * ((1.0 if a == b else 0.0) - g[w, b])
        out.append(total)
    return np.array(out)


def variance(values: np.ndarray) -> float:
    """The variance of ``values`` over the support, under the law."""
    p = base.PROBS.reshape(-1)
    return float((p * values**2).sum() - (p * values).sum() ** 2)


def mechanism_share(tilt: dict[str, float]) -> float:
    """The share of the influence-function variance carried by the mechanism term."""
    return variance(mechanism_term(tilt)) / variance(eif(tilt))


#: A wrong mechanism, ``(W, label)`` on this module's arm axis, large enough that the
#: remainder with a right ``Qbar`` is a visible number for both tilts (asserted below).
WRONG_G = np.array([[0.65, 0.20, 0.15], [0.35, 0.45, 0.20], [0.25, 0.20, 0.55]])


def remainder(tilt: dict[str, float], g_hat: np.ndarray, q_hat: np.ndarray) -> float:
    r""":math:`E_0[(\hat m - m_0)(\hat D - D_0)/\hat D]`, the exact remainder, longhand."""
    f = factor_vector(tilt)
    d0, dh = base.G @ f, g_hat @ f
    m0 = (base.G * f * base.Q).sum(axis=1) / d0
    mh = (g_hat * f * q_hat).sum(axis=1) / dh
    return float(np.sum(base.P_W * (mh - m0) * (dh - d0) / dh))


def _check_nonzero_witnesses() -> None:
    """The nonzero requirements of the module docstring, asserted when it is imported."""
    p_w = base.PROBS.sum(axis=(1, 2))
    natural = truth(NATURAL)
    for name, tilt in (("T_a", T_A), ("T_b", T_B)):
        coefficient = np.abs(mechanism_coefficients(tilt))
        for b in range(K):
            assert coefficient[:, b].max() >= 0.02, f"{name}: level {b} has no mechanism score"
        assert p_w.min() > 0.0
        assert mechanism_share(tilt) >= 0.10, f"{name}: mechanism share below 0.10"
        reference = eif(tilt)
        for b in range(K):
            skipped = reference - mechanism_term(tilt) + mechanism_term(tilt, skip=b)
            assert np.max(np.abs(skipped - reference)) > 0.02, f"{name}: level {b} is not visible"
        assert abs(truth(tilt) - natural) > 0.01, f"{name} does not move the mean"
        assert abs(remainder(tilt, WRONG_G, base.Q)) > 3e-3, f"{name}: WRONG_G is too mild"


_check_nonzero_witnesses()
