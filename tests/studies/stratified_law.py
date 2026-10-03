r"""A finite-support point-treatment law with three unequal baseline strata.

The law is the oracle of the baseline-strata study ``canonical-stratified-tmle``.  Every
variable takes finitely many values, so each truth, each efficient influence function and
each covariance is an exact sum over the 36 support points.

=========  =========================================================================
piece      definition
=========  =========================================================================
``V``      the stratum, 0, 1 or 2 with probabilities 0.5, 0.3 and 0.2
``W``      0, 1 or 2 given ``V``, with the rows of :data:`P_W_GIVEN_V`
``A``      binary, ``g(w, v) = expit(-0.3 + 0.5 w - 0.4 v)``
``Y``      binary, ``Q(a, w, v) = expit(-0.8 + 1.0 a + 0.6 w + 0.9 v)``
=========  =========================================================================

``V`` enters both regressions linearly through its numeric code, so an unpenalized
main-terms logistic regression on ``(A, W, V)`` is correct for the outcome and one on
``(W, V)`` is correct for the treatment.  The strata are unequal on purpose: the smallest
holds about 400 rows at ``n = 2000``, so the ``n / n_s`` embedding of a stratum curve
matters.

A stratum parameter is the marginal point-treatment parameter of the law of ``(W, A, Y)``
given ``V = s``.  Its gradient in the full model is ``I(V = s) D_s(O) / P(V = s)``, where
``D_s`` is the marginal efficient influence function computed inside the stratum.  The ATT
and ATC curves divide by the within-stratum arm share ``P(A = a | V = s)``, not by the
marginal one.

The plan declared ``Q = expit(-0.8 + 0.7 a + 0.6 w + 0.6 v)``.  It allowed a coefficient to
move before the declaration commit only to meet a stated design condition.  Two conditions
failed at the declared values, so two coefficients of ``Q`` moved:

=========================================  ===========================  ========================
condition                                  at the declared ``Q``        at this ``Q``
=========================================  ===========================  ========================
stratum ATEs at least 0.03 apart           0.1645, 0.1467 and 0.1031    0.2325, 0.1754 and
                                                                        0.0899
marginal-fluctuation control displaced at  -0.69 SD in stratum 0 and    -1.15 SD and 2.50 SD
least 1 SD at n = 2,000 in each outer      1.34 SD in stratum 2
stratum
=========================================  ===========================  ========================

The treatment coefficient moved from 0.7 to 1.0 and the ``V`` coefficient from 0.6 to 0.9.
``g`` did not change, so overlap did not change, and every arm mean stays at or below 0.94.

The control omits ``V`` from ``Q``.  Its bias inside a stratum is close to linear in ``V``, and
the marginal score cancels the average, so the bias nearly vanishes in the middle stratum:
0.22 SD here, and 0.05 SD at the declared law.  No law that keeps the arm means away from 0
and 1 moves it past 0.3 SD.  The necessity family therefore reads the two outer strata, and
the exact-law test ``tests/unit/test_stratified_influence_exact.py`` witnesses every stratum's
score block.  ``tests/unit/test_simultaneous_cell_design.py`` recomputes every number above.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

#: ``P(V = v)``.
P_V = np.array([0.5, 0.3, 0.2])

#: ``P(W = w | V = v)``, indexed ``[v, w]``.
P_W_GIVEN_V = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])

STRATA = (0, 1, 2)
LEVELS_W = (0, 1, 2)

#: The marginal estimand stems the study reports, in the package's names.
STEMS = ("ey[1]", "ey[0]", "ate", "att", "atc", "par")


def propensity(w: Any, v: Any) -> Any:
    """``g(w, v) = P(A = 1 | W = w, V = v)``."""
    return expit(-0.3 + 0.5 * np.asarray(w, dtype=float) - 0.4 * np.asarray(v, dtype=float))


def outcome(a: Any, w: Any, v: Any) -> Any:
    """``Q(a, w, v) = P(Y = 1 | A = a, W = w, V = v)``."""
    return expit(
        -0.8
        + 1.0 * np.asarray(a, dtype=float)
        + 0.6 * np.asarray(w, dtype=float)
        + 0.9 * np.asarray(v, dtype=float)
    )


#: The support, ordered ``(v, w, a, y)``.
SUPPORT: tuple[tuple[int, int, int, int], ...] = tuple(
    itertools.product(STRATA, LEVELS_W, (0, 1), (0, 1))
)
_v, _w, _a, _y = (np.array([point[k] for point in SUPPORT], dtype=float) for k in range(4))


def _mass() -> np.ndarray:
    g = propensity(_w, _v)
    q = outcome(_a, _w, _v)
    p_vw = P_V[_v.astype(int)] * P_W_GIVEN_V[_v.astype(int), _w.astype(int)]
    p_a = np.where(_a == 1.0, g, 1.0 - g)
    p_y = np.where(_y == 1.0, q, 1.0 - q)
    return np.asarray(p_vw * p_a * p_y, dtype=float)


#: ``P(O = o)`` for each support point.
PROBS = _mass()


def stratum_name(stem: str, stratum: int) -> str:
    """The package's name of one stratum parameter, as ``stratum_alias`` composes it."""
    return f"{stem}[V={stratum}]"


def names() -> tuple[str, ...]:
    """The six marginal names, then the eighteen stratum names, stratum by stratum."""
    return (*STEMS, *(stratum_name(stem, s) for s in STRATA for stem in STEMS))


@dataclass(frozen=True)
class _Pieces:
    """The nuisance values every curve reads, at each support point."""

    g: np.ndarray
    q1: np.ndarray
    q0: np.ndarray


def _pieces() -> _Pieces:
    return _Pieces(
        g=np.asarray(propensity(_w, _v)),
        q1=np.asarray(outcome(1.0, _w, _v)),
        q0=np.asarray(outcome(0.0, _w, _v)),
    )


def _within(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """The conditional point masses given ``mask``, and the mask's probability."""
    mass = float(PROBS[mask].sum())
    conditional = np.where(mask, PROBS / mass, 0.0)
    return conditional, mass


def _value_and_curve(stem: str, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """One parameter of the law given ``mask``, and its gradient on that conditional law.

    The returned curve is zero off the mask and is the conditional-law efficient influence
    function on it.  It is mean zero under the conditional law.
    """
    weights, _ = _within(mask)
    pieces = _pieces()
    g, q1, q0 = pieces.g, pieces.q1, pieces.q0
    a, y = _a, _y
    ey1 = float(np.sum(weights * q1))
    ey0 = float(np.sum(weights * q0))
    treated = float(np.sum(weights * a))
    d_ey1 = a / g * (y - q1) + q1 - ey1
    d_ey0 = (1.0 - a) / (1.0 - g) * (y - q0) + q0 - ey0
    if stem == "ey[1]":
        value, curve = ey1, d_ey1
    elif stem == "ey[0]":
        value, curve = ey0, d_ey0
    elif stem == "ate":
        value, curve = ey1 - ey0, d_ey1 - d_ey0
    elif stem == "att":
        value = float(np.sum(weights * a * (q1 - q0))) / treated
        curve = (
            a / treated * (y - q1)
            - (1.0 - a) * g / (treated * (1.0 - g)) * (y - q0)
            + a / treated * (q1 - q0 - value)
        )
    elif stem == "atc":
        untreated = 1.0 - treated
        value = float(np.sum(weights * (1.0 - a) * (q1 - q0))) / untreated
        curve = (
            a * (1.0 - g) / (untreated * g) * (y - q1)
            - (1.0 - a) / untreated * (y - q0)
            + (1.0 - a) / untreated * (q1 - q0 - value)
        )
    elif stem == "par":
        mean = float(np.sum(weights * y))
        value = mean - ey0
        curve = (y - mean) - d_ey0
    else:  # pragma: no cover - declaration guard
        raise KeyError(stem)
    return value, np.where(mask, curve, 0.0)


def _split(name: str) -> tuple[str, int | None]:
    for stratum in STRATA:
        suffix = f"[V={stratum}]"
        if name.endswith(suffix):
            return name[: -len(suffix)], stratum
    return name, None


def truth(name: str) -> float:
    """The true value of one marginal or stratum parameter."""
    stem, stratum = _split(name)
    mask = np.ones(len(SUPPORT), dtype=bool) if stratum is None else _v == stratum
    return _value_and_curve(stem, mask)[0]


def eif(name: str) -> np.ndarray:
    """The full-law efficient influence function of one parameter at each support point.

    A stratum parameter's curve is ``I(V = s) D_s / P(V = s)``.
    """
    stem, stratum = _split(name)
    if stratum is None:
        return _value_and_curve(stem, np.ones(len(SUPPORT), dtype=bool))[1]
    mask = _v == stratum
    _, mass = _within(mask)
    return _value_and_curve(stem, mask)[1] / mass


TRUTH: dict[str, float] = {name: truth(name) for name in names()}

#: The exact efficiency-bound SD of every parameter, ``sqrt(E[D^2])``.
EFFICIENCY_SD: dict[str, float] = {
    name: float(np.sqrt(np.sum(PROBS * eif(name) ** 2))) for name in names()
}


def covariance(selected: tuple[str, ...]) -> np.ndarray:
    """The exact covariance ``E[D_j D_k]`` of the selected curves."""
    curves = np.column_stack([eif(name) for name in selected])
    return np.asarray((curves * PROBS[:, None]).T @ curves, dtype=float)


def sample(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows ``(V, W, A, Y)`` as integer columns."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(SUPPORT), size=n, p=PROBS / PROBS.sum())
    support = np.asarray(SUPPORT, dtype=np.int64)[cells]
    return pd.DataFrame(support, columns=["V", "W", "A", "Y"])


def oracle_truths(selected: tuple[str, ...]) -> Mapping[str, float]:
    """The truth of each selected parameter."""
    return {name: TRUTH[name] for name in selected}
