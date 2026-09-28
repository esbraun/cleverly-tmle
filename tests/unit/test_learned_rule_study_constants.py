"""Every constant of the two RM30 law tables, recomputed by quadrature (RM30 review D3).

``docs/roadmap.md``, RM30, "The laws, and the constants computed before the declaration",
prints two tables that an untracked planning probe computed.  This test recomputes each entry
from the declared law and requires it within ``1e-5``, so no untracked probe carries a constant
that a verdict reads.  It also evaluates the study harness's trapezoid rule at the fixed rule
``d0`` of each of the four laws, to the same tolerance.

The quadrature here is independent of the harness.  The values integrate over ``W1`` with
adaptive quadrature split at each rule's threshold.  The limit of a logistic outcome learner is
the population maximum-likelihood fit, solved by Newton's method on a Gauss-Legendre grid.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, special

from tests.studies import _learned_rule_law as law
from tests.studies import learned_rule_cvtmle as gated
from tests.studies import learned_rule_cvtmle_properties as properties

TOLERANCE = 1e-5

#: The declared law, written out again rather than read from the study module.
BLIPS: dict[str, Callable[[float], float]] = {
    "non_exceptional": lambda w1: 0.1 + w1,
    "misspecified_limit": lambda w1: 0.8 * w1 + w1**2 - 0.3,
    "weak_blip": lambda w1: 0.15 * w1,
    "exceptional": lambda w1: 0.0,
}

#: The law table: the threshold of ``d0`` in ``W1`` (``None`` when ``d0`` treats no unit), the
#: learner's limit blip ``(A, A W1, A W2)`` on the logit scale, and the two values.
LAW_TABLE = {
    "non_exceptional": (-0.1, (0.1, 1.0, 0.0), 0.577115, 0.577115),
    "misspecified_limit": (0.278233, (-0.003990, 0.794092, 0.002889), 0.560251, 0.554747),
    "weak_blip": (0.0, (0.0, 0.15, 0.0), 0.521047, 0.521047),
    "exceptional": (None, (0.0, 0.0, 0.0), 0.512179, 0.512179),
}

#: The ``d1`` thresholds of ``misspecified_limit`` at ``W2 = 0`` and ``W2 = 1``.
MISSPECIFIED_D1 = {0.0: 0.005025, 1.0: 0.001387}

#: The constants table.
EFFICIENCY_BOUND = 0.664444
SHRUNKEN_SE_FACTOR = 0.70
DISPLACEMENT_THRESHOLD = 0.5
NECESSITY_LIMIT_BLIP = {0.0: 0.150594, 1.0: 0.194255}
NECESSITY_TARGET = 0.531616
NECESSITY_PLUG_IN = 0.549165
NECESSITY_BIAS = 0.017549
NECESSITY_SDS = (0.729826, 0.698330)
NECESSITY_STANDARDIZED = (1.075, 1.124)

NODES, WEIGHTS = np.polynomial.legendre.leggauss(400)


def g1(w1: np.ndarray | float, w2: float) -> np.ndarray:
    return np.asarray(special.expit(0.3 * np.asarray(w1) - 0.2 * w2))


def qbar(a: np.ndarray | float, w1: np.ndarray | float, w2: float, key: str) -> np.ndarray:
    w1 = np.asarray(w1, dtype=float)
    b = np.vectorize(BLIPS[key])(w1) if w1.ndim else BLIPS[key](float(w1))
    return np.asarray(special.expit(0.2 + 0.5 * w1 - 0.3 * w2 + np.asarray(a) * b))


def expectation(
    function: Callable[[float, float], float], breaks: dict[float, list[float]]
) -> float:
    """``E f(W1, W2)`` with ``W1 ~ U(-1, 1)`` and ``W2 ~ Bernoulli(0.5)``, split at ``breaks``."""
    total = 0.0
    for w2 in (0.0, 1.0):
        points = sorted({-1.0, 1.0, *[b for b in breaks[w2] if -1.0 < b < 1.0]})
        for left, right in itertools.pairwise(points):
            value, _ = integrate.quad(
                lambda w1, w2=w2: function(w1, w2), left, right, epsabs=1e-13, limit=200
            )
            total += 0.25 * value
    return total


def value(key: str, rule: Callable[[float, float], int], breaks: dict[float, list[float]]) -> float:
    return expectation(lambda w1, w2: float(qbar(rule(w1, w2), w1, w2, key)), breaks)


def curve_sd(
    key: str,
    rule: Callable[[float, float], int],
    breaks: dict[float, list[float]],
    working: Callable[[int, float, float], float] | None = None,
) -> float:
    """``sqrt(Var D*(d, Q, g0))`` with ``Q`` the working regression, ``Qbar0`` by default."""

    def moments(w1: float, w2: float) -> tuple[float, float]:
        d = rule(w1, w2)
        g = float(g1(w1, w2)) if d == 1 else 1.0 - float(g1(w1, w2))
        q0 = float(qbar(d, w1, w2, key))
        q = q0 if working is None else working(d, w1, w2)
        second = (q0 * (1.0 - q0) + (q0 - q) ** 2) / g + 2.0 * q * (q0 - q) + q**2
        return second, q0

    second = expectation(lambda w1, w2: moments(w1, w2)[0], breaks)
    mean = expectation(lambda w1, w2: moments(w1, w2)[1], breaks)
    return math.sqrt(second - mean**2)


def design(a: float, w1: np.ndarray, w2: float, *, drop_w1: bool) -> np.ndarray:
    one = np.ones_like(w1)
    if drop_w1:
        return np.stack([one, a * one, w2 * one, a * w2 * one], axis=-1)
    return np.stack([one, a * one, w1, w2 * one, a * w1, a * w2 * one, w1 * w2], axis=-1)


def population_fit(key: str, *, drop_w1: bool) -> np.ndarray:
    """The population logistic fit of the declared learner, by Newton's method."""
    cells = []
    for w2 in (0.0, 1.0):
        for a in (0.0, 1.0):
            pa = g1(NODES, w2) if a else 1.0 - g1(NODES, w2)
            cells.append(
                (
                    design(a, NODES, w2, drop_w1=drop_w1),
                    qbar(a, NODES, w2, key),
                    0.25 * WEIGHTS * pa,
                )
            )
    beta = np.zeros(cells[0][0].shape[-1])
    for _ in range(100):
        score = sum(((weight * (y - special.expit(x @ beta))) @ x for x, y, weight in cells))
        info = sum(
            (
                (x * (weight * special.expit(x @ beta) * (1 - special.expit(x @ beta)))[:, None]).T
                @ x
                for x, _, weight in cells
            )
        )
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return beta


def d0_rule(key: str) -> tuple[Callable[[float, float], int], dict[float, list[float]]]:
    threshold = LAW_TABLE[key][0]
    breaks = {w2: [] if threshold is None else [threshold] for w2 in (0.0, 1.0)}
    return (lambda w1, w2: int(BLIPS[key](w1) > 0.0)), breaks


def test_the_study_module_draws_the_declared_law() -> None:
    w1 = np.linspace(-1.0, 1.0, 101)
    for w2 in (0.0, 1.0):
        np.testing.assert_allclose(law.g0(w1, w2), g1(w1, w2), rtol=1e-15)
        for key in BLIPS:
            for a in (0.0, 1.0):
                np.testing.assert_allclose(
                    law.qbar0(a, w1, w2, key), qbar(a, w1, w2, key), rtol=1e-15
                )


@pytest.mark.parametrize("key", list(LAW_TABLE))
def test_the_optimal_rule_and_its_value(key: str) -> None:
    threshold, _, psi_d0, _ = LAW_TABLE[key]
    if threshold is not None:
        assert BLIPS[key](threshold - 1e-4) < 0.0 < BLIPS[key](threshold + 1e-4)
        exact = {"misspecified_limit": (-0.8 + math.sqrt(0.64 + 1.2)) / 2.0}.get(key, threshold)
        assert exact == pytest.approx(threshold, abs=TOLERANCE)
    else:
        assert all(BLIPS[key](w) <= 0.0 for w in np.linspace(-1.0, 1.0, 11))
    rule, breaks = d0_rule(key)
    assert value(key, rule, breaks) == pytest.approx(psi_d0, abs=TOLERANCE)


@pytest.mark.parametrize("key", list(LAW_TABLE))
def test_the_learners_limit_blip_and_its_rules_value(key: str) -> None:
    _, blip, _, psi_d1 = LAW_TABLE[key]
    beta = population_fit(key, drop_w1=False)
    # Columns: 1, A, W1, W2, A W1, A W2, W1 W2.
    assert (beta[1], beta[4], beta[5]) == pytest.approx(blip, abs=TOLERANCE)
    if key == "exceptional":
        return
    breaks = {w2: [-(beta[1] + beta[5] * w2) / beta[4]] for w2 in (0.0, 1.0)}
    if key == "misspecified_limit":
        assert breaks[0.0][0] == pytest.approx(MISSPECIFIED_D1[0.0], abs=TOLERANCE)
        assert breaks[1.0][0] == pytest.approx(MISSPECIFIED_D1[1.0], abs=TOLERANCE)

    def rule(w1: float, w2: float) -> int:
        return int(beta[1] + beta[4] * w1 + beta[5] * w2 > 0.0)

    assert value(key, rule, breaks) == pytest.approx(psi_d1, abs=TOLERANCE)


def test_the_exceptional_law_gives_every_rule_one_value() -> None:
    none = value("exceptional", lambda w1, w2: 0, {0.0: [], 1.0: []})
    every = value("exceptional", lambda w1, w2: 1, {0.0: [], 1.0: []})
    assert none == pytest.approx(every, abs=1e-12)
    assert none == pytest.approx(LAW_TABLE["exceptional"][3], abs=TOLERANCE)


def test_the_efficiency_bound() -> None:
    rule, breaks = d0_rule("non_exceptional")
    assert curve_sd("non_exceptional", rule, breaks) == pytest.approx(
        EFFICIENCY_BOUND, abs=TOLERANCE
    )
    assert gated.EFFICIENCY_BOUND == EFFICIENCY_BOUND
    assert dict(gated.STUDY.efficiency_bounds) == {"non_exceptional": EFFICIENCY_BOUND}


def test_the_declared_factor_and_thresholds() -> None:
    assert properties.SHRUNKEN_SE_FACTOR == SHRUNKEN_SE_FACTOR
    assert properties.TARGETING_DISPLACEMENT == DISPLACEMENT_THRESHOLD
    assert properties.FOLD_LOCALITY_DISPLACEMENT == DISPLACEMENT_THRESHOLD


def test_the_targeting_necessity_constants() -> None:
    beta = population_fit("non_exceptional", drop_w1=True)
    # Columns: 1, A, W2, A W2.
    limit = {w2: beta[1] + beta[3] * w2 for w2 in (0.0, 1.0)}
    for w2, expected in NECESSITY_LIMIT_BLIP.items():
        assert limit[w2] == pytest.approx(expected, abs=TOLERANCE)
        assert limit[w2] > 0.0
    treat_all = {0.0: [], 1.0: []}
    target = value("non_exceptional", lambda w1, w2: 1, treat_all)
    assert target == pytest.approx(NECESSITY_TARGET, abs=TOLERANCE)

    def working(a: int, w1: float, w2: float) -> float:
        return float(special.expit(beta[0] + beta[1] * a + beta[2] * w2 + beta[3] * a * w2))

    plug_in = expectation(lambda w1, w2: working(1, w1, w2), treat_all)
    assert plug_in == pytest.approx(NECESSITY_PLUG_IN, abs=TOLERANCE)
    assert plug_in - target == pytest.approx(NECESSITY_BIAS, abs=TOLERANCE)
    sds = (
        curve_sd("non_exceptional", lambda w1, w2: 1, treat_all, working),
        curve_sd("non_exceptional", lambda w1, w2: 1, treat_all),
    )
    assert sds == pytest.approx(NECESSITY_SDS, abs=TOLERANCE)
    standardized = tuple(math.sqrt(2_000) * (plug_in - target) / sd for sd in sds)
    assert standardized == pytest.approx(NECESSITY_STANDARDIZED, abs=5e-4)


@pytest.mark.parametrize("key", list(LAW_TABLE))
def test_the_harness_trapezoid_rule_at_each_optimal_rule(key: str) -> None:
    """The harness integrates a known rule to its table value, within the same tolerance."""
    threshold = LAW_TABLE[key][0]
    grid = law.fixed_rule_grid(
        lambda w1, w2: (
            np.zeros_like(w1, dtype=bool) if threshold is None else law.blip(key, w1) > 0.0
        )
    )
    psi, variance = law.rule_value(grid, key)
    assert psi == pytest.approx(LAW_TABLE[key][2], abs=TOLERANCE)
    if key == "non_exceptional":
        assert math.sqrt(variance) == pytest.approx(EFFICIENCY_BOUND, abs=TOLERANCE)
