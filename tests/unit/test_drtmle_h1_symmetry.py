r"""The canonical DR-TMLE law makes both conditioning terms vanish, and law L does not.

``docs/technical-reference/dr-tmle/theorem.md``, "The conditioning terms are linear in one
error", shows that the terms :math:`R_{3,n}` and :math:`\tilde R_{5,n}` are linear in the error
of the consistent nuisance. Each one vanishes on a law with a property this module calls H1.

* **Outcome branch** (outcome regression wrong). H1 is: :math:`\bar Q_0 - \bar Q` is a function
  of :math:`g_0`. Its measure is :math:`L_2(F)`, with
  :math:`F = \bar Q_0 - \bar Q - E[\bar Q_0 - \bar Q \mid A = a, g_0]`.
* **Mirror branch** (mechanism wrong). H1 is: :math:`1/\bar g - 1/g_0` is a function of
  :math:`\bar Q_0`. Its measure is :math:`L_2(h)`, with :math:`r = (g_0 - \bar g)/\bar g` and
  :math:`h = r - g_0 E[r \mid \bar Q_0]/E[g_0 \mid \bar Q_0]`. Arm 0 replaces :math:`g` by
  :math:`1 - g`.

**The paper law** of Benkeser et al. (2017), Section 5.1, is the law of the registered
``drtmle`` and ``drtmle_stratified`` studies: :math:`W_1 \sim U(-2, 2)`,
:math:`W_2 \sim \text{Bern}(1/2)`, :math:`L = -W_1 + 2 W_1 W_2`, :math:`g_0 = \text{expit}(L)` and
:math:`\bar Q_0 = \text{expit}(0.2 A + L)`. It is invariant under
:math:`(W_1, W_2) \to (-W_1, 1 - W_2)`. So the main-effects GLM limits have zero :math:`W`
coefficients, :math:`\bar Q_0 - \bar Q` is a function of :math:`L`, hence of :math:`g_0`, and
:math:`\bar g` is a constant. Both branches are H1. Their one-wrong cells therefore measure
second-order terms only.

**Law L** has :math:`W_1, W_2 \sim N(0, 1)`, :math:`g_0 = \text{expit}(0.6 W_1 + 0.3 W_2)` and
:math:`\bar Q_0 = \text{expit}(-0.2 + 0.5 A - W_1 + 2 W_2)`. With an outcome GLM on
:math:`(A, W_1)` and a mechanism GLM on :math:`W_2` it is not H1 in either branch.

Everything is quadrature. No float is pinned exactly; floors and ceilings have wide margins.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import expit

#: A ceiling for a measure that is zero by symmetry. The quadrature nodes are symmetric, so
#: the measured values are rounding error.
SYMMETRIC = 1e-10

#: Floors for law L. Measured by this module's quadrature: ``L_2(F)`` 0.2655 (arm 1) and 0.2658
#: (arm 0); ``L_2(h)`` 0.2500 at both arms with the ``W_2``-only GLM limit, and 0.3050 with the
#: constant ``E[g_0]``.
OUTCOME_FLOOR = 0.2
MIRROR_FLOOR = 0.05


def logistic_limit(
    design: list[np.ndarray], response: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """The population logistic MLE: the root of ``E[X (E[response | X-cell] - expit(X b))]``."""
    beta = np.zeros(len(design))
    for _ in range(100):
        fitted = expit(sum(b * x for b, x in zip(beta, design, strict=True)))
        score = np.array([(weights * x * (response - fitted)).sum() for x in design])
        info = np.array(
            [[(weights * fitted * (1 - fitted) * x * z).sum() for z in design] for x in design]
        )
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-15:
            break
    return beta


def conditional(values: np.ndarray, key: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``E[values | key]`` on a grid whose level sets of ``key`` are exact node groups."""
    rounded = np.round(key, 10)
    out = np.empty_like(values)
    for level in np.unique(rounded):
        mask = rounded == level
        out[mask] = (weights[mask] * values[mask]).sum() / weights[mask].sum()
    return out


def l2(values: np.ndarray, weights: np.ndarray) -> float:
    """``sqrt(E[values²])``."""
    return float(np.sqrt((weights * values**2).sum()))


def arm_mechanism(g: np.ndarray, arm: int) -> np.ndarray:
    """``P(A = arm | W)`` from the arm-1 mechanism."""
    return g if arm == 1 else 1.0 - g


def mirror_measure(
    g0: np.ndarray, gbar: np.ndarray, q0_arm: np.ndarray, w: np.ndarray, arm: int
) -> float:
    """``L_2(h)`` for one arm."""
    true, limit = arm_mechanism(g0, arm), arm_mechanism(gbar, arm)
    ratio = (true - limit) / limit
    h = ratio - true * conditional(ratio, q0_arm, w) / conditional(true, q0_arm, w)
    return l2(h, w)


def outcome_measure(
    q0: Callable[[float], np.ndarray],
    qbar: Callable[[float], np.ndarray],
    g0: np.ndarray,
    w: np.ndarray,
    arm: int,
) -> float:
    """``L_2(F)`` for one arm.

    Conditioning on ``A = a`` leaves the law of ``W`` unchanged on a level set of ``g_0``,
    so ``E[· | A = a, g_0]`` is ``E[· | g_0]`` under the law of ``W``.
    """
    resid = q0(arm) - qbar(arm)
    return l2(resid - conditional(resid, g0, w), w)


class PaperLaw:
    """Benkeser et al. (2017), Section 5.1, on a symmetric Gauss-Legendre grid."""

    def __init__(self, nodes: int = 200) -> None:
        x, wx = np.polynomial.legendre.leggauss(nodes)
        w1 = np.concatenate([2.0 * x, 2.0 * x])
        w2 = np.concatenate([np.zeros(nodes), np.ones(nodes)])
        self.w = np.concatenate([wx, wx]) / (2.0 * wx.sum())
        self.w1, self.w2 = w1, w2
        self.index = -w1 + 2.0 * w1 * w2
        self.g0 = expit(self.index)

    def q0(self, arm: float) -> np.ndarray:
        return np.asarray(expit(0.2 * arm + self.index))

    def outcome_glm(self) -> np.ndarray:
        """``(b_0, b_A, b_1, b_2)``: the main-effects logistic limit of ``Y`` on ``(A, W)``."""
        ones = np.ones_like(self.w1)
        stacked = [np.concatenate([ones, ones])]
        stacked.append(np.concatenate([ones, 0.0 * ones]))
        stacked += [np.concatenate([self.w1, self.w1]), np.concatenate([self.w2, self.w2])]
        response = np.concatenate([self.q0(1.0), self.q0(0.0)])
        weights = np.concatenate([self.w * self.g0, self.w * (1.0 - self.g0)])
        return logistic_limit(stacked, response, weights)

    def mechanism_glm(self) -> np.ndarray:
        """``(c_0, c_1, c_2)``: the main-effects logistic limit of ``A`` on ``W``."""
        return logistic_limit([np.ones_like(self.w1), self.w1, self.w2], self.g0, self.w)


class LawL:
    """Law L on a Gauss-Hermite grid rotated so one axis is a chosen linear index."""

    def __init__(self, direction: tuple[float, float], nodes: int = 80) -> None:
        x, wx = np.polynomial.hermite_e.hermegauss(nodes)
        wx = wx / wx.sum()
        along, across = np.meshgrid(x, x, indexing="ij")
        e = np.asarray(direction, dtype=float) / np.hypot(*direction)
        perp = np.array([-e[1], e[0]])
        self.w = np.outer(wx, wx).ravel()
        self.w1 = (along * e[0] + across * perp[0]).ravel()
        self.w2 = (along * e[1] + across * perp[1]).ravel()
        self.g0 = expit(0.6 * self.w1 + 0.3 * self.w2)

    def q0(self, arm: float) -> np.ndarray:
        return np.asarray(expit(-0.2 + 0.5 * arm - self.w1 + 2.0 * self.w2))

    def outcome_glm(self) -> np.ndarray:
        """``(b_0, b_A, b_1)``: the wrong logistic limit of ``Y`` on ``(A, W_1)``."""
        ones = np.ones_like(self.w1)
        design = [
            np.concatenate([ones, ones]),
            np.concatenate([ones, 0.0 * ones]),
            np.concatenate([self.w1, self.w1]),
        ]
        response = np.concatenate([self.q0(1.0), self.q0(0.0)])
        weights = np.concatenate([self.w * self.g0, self.w * (1.0 - self.g0)])
        return logistic_limit(design, response, weights)

    def mechanism_glm(self) -> np.ndarray:
        """``(c_0, c_2)``: the wrong logistic limit of ``A`` on ``W_2`` alone."""
        return logistic_limit([np.ones_like(self.w2), self.w2], self.g0, self.w)


class TestThePaperLawIsH1:
    """The registered canonical and stratified laws hide both terms."""

    def test_the_main_effects_limits_have_no_covariate_coefficients(self) -> None:
        law = PaperLaw()
        outcome, mechanism = law.outcome_glm(), law.mechanism_glm()
        assert np.max(np.abs(outcome[2:])) < SYMMETRIC
        assert np.max(np.abs(mechanism[1:])) < SYMMETRIC
        # The control: the limits are not degenerate. The arm coefficient is far from zero.
        assert outcome[1] > 0.5

    @pytest.mark.parametrize("arm", [0, 1])
    def test_the_outcome_branch_is_h1(self, arm: int) -> None:
        law = PaperLaw()
        b = law.outcome_glm()

        def qbar(a: float) -> np.ndarray:
            return np.asarray(expit(b[0] + b[1] * a + b[2] * law.w1 + b[3] * law.w2))

        assert outcome_measure(law.q0, qbar, law.g0, law.w, arm) < SYMMETRIC

    @pytest.mark.parametrize("arm", [0, 1])
    def test_the_mirror_branch_is_h1(self, arm: int) -> None:
        law = PaperLaw()
        c = law.mechanism_glm()
        gbar = expit(c[0] + c[1] * law.w1 + c[2] * law.w2)
        assert np.ptp(gbar) < SYMMETRIC
        assert mirror_measure(law.g0, gbar, law.q0(float(arm)), law.w, arm) < SYMMETRIC

    def test_a_covariate_coefficient_would_break_h1(self) -> None:
        """The control: an outcome limit with a ``W_1`` term is not H1 on this law.

        Without it the two tests above could pass because the measure is blind.
        """
        law = PaperLaw()

        def tilted(a: float) -> np.ndarray:
            return np.asarray(expit(-0.5 + a + 0.3 * law.w1))

        assert outcome_measure(law.q0, tilted, law.g0, law.w, 1) > 0.01


class TestLawLIsNotH1:
    """Law L exposes both terms."""

    @pytest.mark.parametrize("arm", [0, 1])
    def test_the_outcome_branch_is_not_h1(self, arm: int) -> None:
        # Rotate so that the first axis is the index of g_0, whose level sets are lines.
        law = LawL(direction=(0.6, 0.3))
        b = law.outcome_glm()

        def qbar(a: float) -> np.ndarray:
            return np.asarray(expit(b[0] + b[1] * a + b[2] * law.w1))

        assert outcome_measure(law.q0, qbar, law.g0, law.w, arm) > OUTCOME_FLOOR

    @pytest.mark.parametrize("arm", [0, 1])
    def test_the_mirror_branch_is_not_h1(self, arm: int) -> None:
        # Rotate so that the first axis is the index of Q̄_0, shared by both arms.
        law = LawL(direction=(-1.0, 2.0))
        c = law.mechanism_glm()
        gbar = expit(c[0] + c[1] * law.w2)
        assert mirror_measure(law.g0, gbar, law.q0(float(arm)), law.w, arm) > MIRROR_FLOOR

    @pytest.mark.parametrize("arm", [0, 1])
    def test_a_limit_constant_on_the_level_sets_is_not_enough(self, arm: int) -> None:
        """``ḡ = E[g_0]`` is constant, so constant on every level set of ``Q̄_0``.

        The mirror term still does not vanish. The condition is that ``1/ḡ - 1/g_0`` is a
        function of ``Q̄_0``, not that ``ḡ`` is.
        """
        law = LawL(direction=(-1.0, 2.0))
        gbar = np.full_like(law.g0, (law.w * law.g0).sum())
        assert mirror_measure(law.g0, gbar, law.q0(float(arm)), law.w, arm) > MIRROR_FLOOR
